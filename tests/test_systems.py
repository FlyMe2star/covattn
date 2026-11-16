import numpy as np
import pytest

from covattn.systems import (
    SampleSpec,
    SystemSpec,
    default_systems,
    effect_by_convolution,
    make_counterfactual,
    make_rng,
    make_sample,
)


@pytest.mark.parametrize("sys", [s for s in default_systems() if s.is_linear], ids=lambda s: s.key)
@pytest.mark.parametrize("shape", ["step", "pulse", "ramp"])
def test_resimulation_matches_convolution(sys: SystemSpec, shape: str) -> None:
    spec = SampleSpec(context_length=256, horizon=64, input_type="prbs", snr=5.0)
    s = make_sample(SystemSpec(**{**sys.to_dict(), "proc_noise": 0.1}), spec, make_rng(0, "test", sys.key))
    cf = make_counterfactual(s, shape, 1.0)
    conv = effect_by_convolution(s, cf.du)
    np.testing.assert_allclose(cf.effect, conv, rtol=1e-8, atol=1e-8)


def test_counterfactual_shares_context_and_noise() -> None:
    s = make_sample(SystemSpec("F2", a=0.9), SampleSpec(context_length=256), make_rng(1, "test", "x"))
    cf = make_counterfactual(s, "step", 2.0)
    assert np.allclose(cf.u_future_cf - s.u[256:], cf.du)
    assert np.abs(cf.effect).max() > 0
    # zero change -> zero effect
    cf0 = make_counterfactual(s, "step", 0.0)
    assert np.abs(cf0.effect).max() < 1e-10


def test_delay_family() -> None:
    sys = SystemSpec("F3", a=0.8, d=16)
    s = make_sample(sys, SampleSpec(context_length=256), make_rng(2, "test", "d"))
    cf = make_counterfactual(s, "step", 1.0)
    assert np.abs(cf.effect[:16]).max() < 1e-10
    assert np.abs(cf.effect[16]) > 0


def test_normalised_state_and_seeds() -> None:
    spec = SampleSpec(context_length=512, snr=float("inf"))
    s = make_sample(SystemSpec("F4", period=24, zeta=0.05), spec, make_rng(3, "test", "n"))
    assert abs(s.x[:512].std() - 1.0) < 1e-6
    assert np.allclose(s.eps, 0)
    a = make_sample(SystemSpec("F1"), spec, make_rng(3, "train", "n")).y
    b = make_sample(SystemSpec("F1"), spec, make_rng(3, "train", "n")).y
    c = make_sample(SystemSpec("F1"), spec, make_rng(3, "test", "n")).y
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_hammerstein_nonlinear_effect_runs() -> None:
    s = make_sample(SystemSpec("F5", a=0.8), SampleSpec(context_length=256), make_rng(4, "test", "h"))
    e1 = make_counterfactual(s, "step", 1.0).effect
    e2 = make_counterfactual(s, "step", 2.0).effect
    assert not np.allclose(2 * e1, e2)  # tanh makes the response nonlinear in delta
