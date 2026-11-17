"""Hook system tests (need the Chronos-2 weights). Device: $COVATTN_DEVICE (default auto)."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from einops import rearrange  # noqa: E402

from covattn.hooks import HookSpec, Patch  # noqa: E402
from covattn.model import CovBatch  # noqa: E402

pytestmark = pytest.mark.model
TOL = 1e-5


@pytest.fixture(scope="module")
def w():
    from covattn.model import Chronos2Wrapper

    return Chronos2Wrapper("amazon/chronos-2", device=os.environ.get("COVATTN_DEVICE", "auto"), batch_size=128)


def _pair(n=4, k=1, L=256, H=64, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((n, L)).cumsum(-1)
    u = rng.standard_normal((n, k, L))
    uf = rng.standard_normal((n, k, H))
    du = np.zeros_like(uf)
    du[..., :] = 1.5
    return CovBatch(y, u, uf), CovBatch(y, u, uf + du)


def _close(a, b, tol=TOL):
    a, b = a.detach().float().cpu().numpy(), b.detach().float().cpu().numpy()
    scale = max(1.0, float(np.abs(b).max()))
    assert np.abs(a - b).max() <= tol * scale, np.abs(a - b).max()


@pytest.mark.parametrize("L,k", [(256, 1), (300, 2)])
def test_hooks_off_equals_official(w, L, k):
    fac, _ = _pair(L=L, k=k)
    q_off, _ = w.forward_raw(fac)
    q_hook, rec = w.forward_with_hooks(fac, HookSpec())
    assert rec == {}
    _close(q_hook, q_off)
    np.testing.assert_allclose(q_hook.cpu().numpy(), w.predict(fac, 64), rtol=TOL, atol=TOL * 10)


def test_decomposition_is_exact(w):
    fac, _ = _pair(k=2)
    q_off, _ = w.forward_raw(fac)
    q, rec = w.forward_with_hooks(fac, HookSpec(record={"cross", "self", "group_out", "group_w"}))
    _close(q, q_off, tol=2e-5)  # eager vs SDPA attention
    for l in range(w.hooked.n_layers):
        ctx = rec["self"][l] + rec["cross"][l]  # (T, H, N, dk)
        o = w.model.encoder.block[l].layer[1].self_attention.o
        recon = rearrange(o(rearrange(ctx, "t h n k -> t n (h k)")), "t n d -> n t d")
        _close(recon, rec["group_out"][l], tol=1e-5)
        assert float(rec["group_w"][l].min()) >= 0 and float(rec["group_w"][l].max()) <= 1 + 1e-6


def test_zero_cross_removes_value_reads_only(w):
    """cross_gain = 0 removes every covariate *value* read by the target, but the covariate still
    enters through the softmax normalisation (competition with the target's self weight)."""
    fac, cf = _pair()
    g0 = torch.zeros(w.hooked.n_layers, w.hooked.n_heads)
    g1 = torch.ones(w.hooked.n_layers, w.hooked.n_heads)
    qf_rec, _ = w.forward_with_hooks(fac, HookSpec(record={"cross"}))
    qf_g1, _ = w.forward_with_hooks(fac, HookSpec(cross_gain=g1, record={"cross"}))
    assert torch.equal(qf_g1, qf_rec)  # gain 1 is an exact no-op
    qf, _ = w.forward_with_hooks(fac, HookSpec(cross_gain=g0))
    qc, _ = w.forward_with_hooks(cf, HookSpec(cross_gain=g0))
    qf1, _ = w.forward_with_hooks(fac)
    qc1, _ = w.forward_with_hooks(cf)
    full = float((qf1 - qc1).abs().max())
    assert full > 1e-3
    assert float((qf - qc).abs().max()) < full  # strongly reduced, not exactly zero (normalisation path)


def test_univariate_unaffected_by_cross_gain(w):
    rng = np.random.default_rng(1)
    uni = CovBatch(rng.standard_normal((3, 256)).cumsum(-1), np.zeros((3, 0, 256)), np.zeros((3, 0, 64)))
    q_off, _ = w.forward_raw(uni)
    q_rec, _ = w.forward_with_hooks(uni, HookSpec(record={"cross"}))
    gain = torch.full((w.hooked.n_layers, w.hooked.n_heads), 3.0)
    q_g, rec = w.forward_with_hooks(uni, HookSpec(cross_gain=gain, record={"cross"}))
    assert float(rec["cross"][0].abs().max()) == 0.0
    assert torch.equal(q_g, q_rec)  # bit-identical: the repaired path is exactly zero
    _close(q_g, q_off, tol=2e-5)


def test_full_patch_reproduces_counterfactual(w):
    fac, cf = _pair()
    q_cf, rec_cf = w.forward_with_hooks(cf, HookSpec(record={"group_out"}))
    patches = [Patch("group_out", l, "all", source=rec_cf) for l in range(w.hooked.n_layers)]
    q_p, _ = w.forward_with_hooks(fac, HookSpec(patches=patches))
    _close(q_p, q_cf, tol=2e-5)
    # same through the per-head target contexts (self + cross) of every layer
    _, rec_cf2 = w.forward_with_hooks(cf, HookSpec(record={"head"}))
    patches = [Patch("head", l, "all", source=rec_cf2) for l in range(w.hooked.n_layers)]
    q_p2, _ = w.forward_with_hooks(fac, HookSpec(patches=patches))
    _close(q_p2, q_cf, tol=2e-5)
    # patching only the cross (value) terms is generally incomplete because of the softmax normalisation
    _, rec_cf3 = w.forward_with_hooks(cf, HookSpec(record={"cross"}))
    patches = [Patch("cross", l, "all", source=rec_cf3) for l in range(w.hooked.n_layers)]
    q_p3, _ = w.forward_with_hooks(fac, HookSpec(patches=patches))
    assert float((q_p3 - q_cf).abs().max()) > 0  # documents the gap; its size is measured in Exp2


def test_restart_from_cached_layer(w):
    fac, cf = _pair()
    P = w.prepare_hooked(fac)
    _, rec_f = w.forward_with_hooks(P, HookSpec(record={"block_in"}))
    _, rec_c = w.forward_with_hooks(cf, HookSpec(record={"mlp_out"}))
    for l in (0, 5, 11):
        p = [Patch("mlp_out", l, "future", source=rec_c)]
        q_full, _ = w.forward_with_hooks(P, HookSpec(patches=p))
        q_fast, _ = w.forward_with_hooks(P, HookSpec(patches=p, start_layer=l, start_hidden=rec_f["block_in"][l]))
        assert torch.equal(q_full, q_fast)


def test_time_weights_and_reg_bias(w):
    fac, _ = _pair()
    P = w.prepare_hooked(fac)
    q0, rec = w.forward_with_hooks(P, HookSpec(record={"time_w"}))
    for l, tw in rec["time_w"].items():
        np.testing.assert_allclose(tw.sum(-1).cpu().numpy(), 1.0, atol=1e-5)
    qz, _ = w.forward_with_hooks(P, HookSpec(reg_bias=torch.zeros(w.hooked.n_layers)))
    assert torch.equal(qz, w.forward_with_hooks(P)[0])
    qb, rec_b = w.forward_with_hooks(P, HookSpec(reg_bias=torch.full((w.hooked.n_layers,), -4.0), record={"time_w"}))
    assert float(rec_b["time_w"][0][..., 0].mean()) < float(rec["time_w"][0][..., 0].mean())
