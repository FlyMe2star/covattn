"""Gate (docs/gate_protocol.md): input share, attenuation index, threshold rule, selection criterion pooling.
Light tests only: no forecasting model is loaded (a fake linear 'model' stands in for Chronos-2)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import lfilter

from covattn import gate as G
from covattn.model import CovBatch
from covattn.repair import objective_np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _arx_data(L=1024, noise=0.05, seed=0, b=(0.0, 0.5), a=(0.8,)):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(L + 200)
    y = lfilter(b, np.r_[1.0, -np.asarray(a)], u) + noise * rng.standard_normal(L + 200)
    return y[200:], u[200:]


def test_input_share_high_for_driven_low_for_unrelated():
    y, u = _arx_data()
    s, f = G.input_share(y, u)
    assert s > 0.8 and f.stable()
    rng = np.random.default_rng(1)
    s0, _ = G.input_share(y, rng.standard_normal(len(y)))
    assert 0.0 <= s0 < 0.05


def test_window_stats_picks_driving_covariate_and_probe_response():
    y, u = _arx_data()
    rng = np.random.default_rng(2)
    U = np.stack([rng.standard_normal(len(y)), u])
    st = G.window_stats(y, U, 64)
    assert st.k == 1 and st.du == pytest.approx(np.std(u))
    # true zero-state step response of 0.5 q^-1 / (1 - 0.8 q^-1) to du
    true = lfilter([0.0, 0.5], [1.0, -0.8], np.full(64, st.du))
    assert np.allclose(st.e_arx, true, rtol=0.1, atol=0.05 * st.du)


def test_attenuation_index_and_gate_rule():
    y, u = _arx_data()
    st = [G.window_stats(y, u[None], 64)]
    rho = G.attenuation_index(0.5 * st[0].e_arx[None], st)
    assert rho[0] == pytest.approx(0.5)
    s = np.array([0.9, 0.9, 0.01, 0.9])
    r = np.array([0.5, 1.2, 0.5, np.nan])
    assert G.gate_on(s, r, 0.1, 0.8).tolist() == [True, False, False, False]
    assert G.gate_on(s, r, 0.0, float("inf")).all()  # ungated corner of the grid


def test_nan_and_short_context_keep_gate_off():
    y, u = _arx_data(L=512)
    y = y.copy()
    y[:100] = np.nan  # left padding (fev)
    y[300:305] = np.nan  # interior gap (LCL)
    st = G.window_stats(y, u[None], 16)
    assert np.isfinite(st.s) and st.s > 0.5
    short = G.window_stats(y[-40:], u[None, -40:], 16)
    assert short.s == 0.0 and short.du == 0.0
    assert np.isnan(G.attenuation_index(np.ones((1, 16)), [short])[0])
    const = G.window_stats(y[200:], np.ones((1, 312)), 16)
    assert const.s == pytest.approx(0.0, abs=1e-6)


def test_select_prefers_conservative_within_tolerance():
    sc = {(0.0, float("inf")): 1.000, (0.1, 0.8): 1.0015, (0.2, 1.0): 1.0010, (0.3, 0.6): 1.0030}
    assert G.select(sc, 0.002) == (0.2, 1.0)  # within 0.002 of the best; largest tau_s (0.3 is outside)
    assert G.select({(0.1, 0.8): 1.0, (0.1, 0.6): 1.001}, 0.002) == (0.1, 0.6)  # same tau_s -> smaller tau_rho
    sc2 = {(0.0, float("inf")): 1.0, (0.5, 0.6): 1.5}
    assert G.select(sc2, 0.002) == (0.0, float("inf"))
    assert len(G.grid()) == 36 and (0.0, float("inf")) in G.grid()


class FakeModel:
    """Linear 'forecaster': forecast = c * ARX-like response of the future input + flat context mean."""

    quantiles = [0.1, 0.5, 0.9]
    median_idx = 1

    def __init__(self, c):
        self.c = c

    def predict(self, b: CovBatch, H: int) -> np.ndarray:
        resp = lfilter([0.0, 0.5], [1.0, -0.8], b.cov_future[:, 0], axis=-1)
        med = np.nanmean(b.target, -1, keepdims=True) + self.c * resp
        return np.stack([med - 1, med, med + 1], 1)


def test_gate_arrays_recovers_attenuation_of_fake_model():
    import exp14_gate as E14

    y, u = _arx_data(L=600)
    L, H = 512, 32
    fac = CovBatch(y[None, :L], u[None, None, :L], u[None, None, L:L + H])
    for c in (0.4, 1.0):
        w = FakeModel(c)
        g = E14.gate_arrays(w, fac, w.predict(fac, H))
        assert g["gate_s"][0] > 0.8
        assert g["gate_rho"][0] == pytest.approx(c, rel=0.15)


def test_objective_parts_pools_like_objective_np():
    import exp14_gate as E14

    rng = np.random.default_rng(3)
    N, Q, H, L = 6, 3, 16, 40
    q_fac = np.sort(rng.standard_normal((N, Q, H)), 1)
    q_cf = q_fac + rng.standard_normal((N, Q, H)) * 0.3
    e = rng.standard_normal((N, H))
    e[0] *= 1e-6  # masked out
    y_ctx, y_true = rng.standard_normal((N, L)), rng.standard_normal((N, H))
    qs = [0.1, 0.5, 0.9]
    ref = objective_np(q_fac, q_cf, SimpleNamespace(e=e, y_ctx=y_ctx, y_true=y_true), qs, 1, 1.0)["obj"]
    a, b, c, d = E14.objective_parts(q_fac, q_cf, e, y_ctx.std(-1), y_true, qs, 1, 0.05)
    assert b == N - 1 and d == N
    assert a / b + c / d == pytest.approx(ref)


def test_pick_switches_per_window():
    import exp14_gate as E14

    z = {"q_r2": np.ones((3, 2, 4)), "q_zeroshot": np.zeros((3, 2, 4))}
    out = E14._pick(z, "q_{m}", "r2", np.array([True, False, True]))
    assert out[:, 0, 0].tolist() == [1, 0, 1]


def test_daisy_spec_consistent():
    import daisy_prepare as D
    import exp14_gate as E14

    assert sorted(D.DATASETS) == sorted(E14.DAISY) and len(D.DATASETS) == 13
    for name, (_, ncol, tcol, ucols, ycols) in D.SPEC.items():
        used = ([tcol] if tcol is not None else []) + ucols + ycols
        assert sorted(used) == list(range(ncol)), name
