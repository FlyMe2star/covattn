"""Model tests (need the Chronos-2 weights; run with `pytest -m model`)."""

import os

import numpy as np
import pytest
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def wrapper():
    from covattn.model import Chronos2Wrapper

    return Chronos2Wrapper("amazon/chronos-2", device=os.environ.get("COVATTN_DEVICE", "auto"), amp="off", batch_size=256)


def _batch(n=6, k=1, L=300, H=64, seed=0):
    from covattn.model import CovBatch

    rng = np.random.default_rng(seed)
    return CovBatch(rng.standard_normal((n, L)).cumsum(-1), rng.standard_normal((n, k, L)),
                    rng.standard_normal((n, k, H)))


@pytest.mark.parametrize("L,k", [(256, 1), (300, 2), (512, 1)])
def test_forward_raw_matches_pipeline(wrapper, L, k) -> None:
    b = _batch(L=L, k=k)
    q_pipe = wrapper.predict(b, 64)
    q_raw, _ = wrapper.forward_raw(b)
    np.testing.assert_allclose(q_raw.cpu().numpy(), q_pipe, rtol=1e-5, atol=1e-5)


def test_future_covariate_does_not_change_normalisation(wrapper) -> None:
    """Shifting the *future* covariate by a constant must not leave the forecast unchanged only
    because of re-normalisation; check directly that loc/scale come from the context."""
    b = _batch(n=2)
    inp = wrapper.build_model_inputs(b)
    ctx = inp["context"]
    _, (loc, scale) = wrapper.model.instance_norm(ctx)
    assert torch.allclose(loc[:, 0], torch.nanmean(ctx, -1))


def test_batch_independence(wrapper) -> None:
    b = _batch(n=8)
    full = wrapper.predict(b, 64)
    part = wrapper.predict(b.slice(3, 5), 64)
    np.testing.assert_allclose(full[3:5], part, rtol=1e-4, atol=1e-4)
