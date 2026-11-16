import numpy as np

from covattn import metrics as M
from covattn.baselines import fit_arx
from covattn.systems import SampleSpec, SystemSpec, make_counterfactual, make_rng, make_sample


def test_gain_and_shape() -> None:
    rng = np.random.default_rng(0)
    e = rng.standard_normal((10, 64))
    np.testing.assert_allclose(M.effect_gain(e, e), 1.0)
    np.testing.assert_allclose(M.effect_gain(0.4 * e, e), 0.4)
    np.testing.assert_allclose(M.shape_corr(0.4 * e + 3.0, e), 1.0)
    np.testing.assert_allclose(M.whatif_nrmse(0.4 * e, e), 0.6)
    np.testing.assert_allclose(M.gain_curve(0.5 * e, e), 0.5)


def test_memoryless_r2() -> None:
    du = np.zeros((2, 64))
    du[:, :16] = 1.0
    resp = np.stack([2.0 * du[0], np.convolve(0.8 ** np.arange(64), du[1])[:64]])
    r2 = M.memoryless_r2(resp, du)
    assert abs(r2[0] - 1.0) < 1e-12 and r2[1] < 0.9


def test_wql_mase() -> None:
    y = np.ones((3, 8))
    q = np.ones((3, 3, 8))
    assert np.allclose(M.wql(q, [0.1, 0.5, 0.9], y), 0.0)
    ctx = np.cumsum(np.ones((3, 20)), -1)
    assert np.allclose(M.mase(y + 1, y, ctx), 1.0)


def test_arx_recovers_linear_system() -> None:
    sys = SystemSpec("F2", a=0.9)
    spec = SampleSpec(context_length=1024, snr=100.0, bg_scale=0.0)
    s = make_sample(sys, spec, make_rng(0, "test", "arx"))
    cf = make_counterfactual(s, "step", 1.0)
    f = fit_arx(s.y[:1024], s.u[:1024])
    g = M.effect_gain(f.effect(cf.du)[None], cf.effect[None])[0]
    assert abs(g - 1.0) < 0.05
