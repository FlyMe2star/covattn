"""Repair tests (need the Chronos-2 weights). Device: $COVATTN_DEVICE (default auto).

Constructive guarantee (TASK 3.6): with only the target in the group, R1 / R2 outputs equal the
original model element-wise (< 1e-5)."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from covattn.hooks import HookSpec  # noqa: E402
from covattn.model import CovBatch  # noqa: E402
from covattn.repair import gain_spec, r1_gain  # noqa: E402

pytestmark = pytest.mark.model
TOL = 1e-5


@pytest.fixture(scope="module")
def w():
    from covattn.model import Chronos2Wrapper

    return Chronos2Wrapper("amazon/chronos-2", device=os.environ.get("COVATTN_DEVICE", "auto"), batch_size=128)


def _univariate(n=5, L=300, H=64, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((n, L)).cumsum(-1) + 3 * np.sin(np.arange(L) / 7.0)
    return CovBatch(y, np.zeros((n, 0, L)), np.zeros((n, 0, H)))


def _gates(w, seed=0):
    g = torch.rand(w.hooked.n_layers, w.hooked.n_heads, generator=torch.Generator().manual_seed(seed)) * 3
    return g


@pytest.mark.parametrize("L", [256, 300])
def test_r1_r2_univariate_identical_to_original(w, L):
    b = _univariate(L=L)
    ref = w.predict(b, 64)
    hk = w.hooked
    G1 = r1_gain(hk.n_layers, hk.n_heads, [(8, 6), (9, 2), (11, 7)], 3.0)
    for G in (G1, _gates(w)):
        out = w.predict_hooked(b, gain_spec(G))
        np.testing.assert_allclose(out, ref, rtol=TOL, atol=TOL * max(1.0, float(np.abs(ref).max())))
        # and bit-identical to the hooked run without any repair
        np.testing.assert_array_equal(out, w.predict_hooked(b, HookSpec()))


def test_gates_get_gradients_and_unit_gates_are_identity(w):
    rng = np.random.default_rng(0)
    n, L, H = 3, 256, 64
    b = CovBatch(rng.standard_normal((n, L)).cumsum(-1), rng.standard_normal((n, 1, L)), rng.standard_normal((n, 1, H)))
    P = w.prepare_hooked(b)
    gates = torch.ones(w.hooked.n_layers, w.hooked.n_heads, device=w.device, requires_grad=True)
    q, _ = w.hooked.run(P, HookSpec(cross_gain=gates, grad=True))
    q_ref, _ = w.forward_raw(b)
    assert np.abs(q.detach().cpu().numpy() - q_ref.cpu().numpy()).max() <= 2e-5 * max(1.0, float(q_ref.abs().max()))
    q[:, 10].sum().backward()
    assert gates.grad is not None and float(gates.grad.abs().sum()) > 0
    assert all(p.grad is None for p in w.model.parameters())  # model weights stay frozen
