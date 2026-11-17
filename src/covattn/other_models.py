"""Other covariate-capable TSFMs behind the same `predict(CovBatch, H) -> (N, Q, H)` interface (phenomenon only).

  TimesFM 3.0  (google/timesfm-3.0-pytorch; package timesfm 3.0.2, module timesfm3): native past+future
               covariates through attention. Official benchmark defaults of `TimesFM3Evaluator.predict_batch`
               (symmetric averaging, sorted quantiles) except make_positive=False (synthetic targets can be < 0).
               Quantiles 0.1..0.9 -> median index 4.
  TimesFM 2.5  (google/timesfm-2.5-200m-pytorch): covariates via XReg, an in-context linear regression on the
               covariate values ("xreg + timesfm": the linear model is fit on the context, TimesFM forecasts its
               residuals). Quantile output columns: [mean, 0.1..0.9] -> median column 5.
Weights are downloaded from the Hugging Face Hub on first use. The timesfm 3.0 weights carry a non-commercial
licence (research use).
"""

from __future__ import annotations

import numpy as np

from covattn.model import CovBatch


class TimesFM3Adapter:
    name = "timesfm3"
    median_idx = 4

    def __init__(self, device: str = "cuda", batch_size: int = 32,
                 checkpoint: str = "google/timesfm-3.0-pytorch") -> None:
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        self.m = TimesFM3Evaluator(ModelConfig(checkpoint_path=checkpoint, per_core_batch_size=batch_size,
                                               device=device))

    def predict(self, batch: CovBatch, H: int | None = None) -> np.ndarray:
        H = H or batch.H
        ctx = [batch.target[i] for i in range(batch.n)]
        pf = [np.concatenate([batch.cov_past[i], batch.cov_future[i]], axis=-1) if batch.k else None
              for i in range(batch.n)]
        outs = self.m.predict_batch(contexts=ctx, horizon=H, past_future_covariates=pf if batch.k else None,
                                    return_quantiles=True, make_positive=False)
        qs = [np.asarray(o.quantiles, np.float32) for o in outs]
        return np.stack([q.reshape(-1, q.shape[-1])[-H:].T for q in qs])  # (N, 9, H)


class TimesFM25XRegAdapter:
    name = "timesfm25_xreg"
    median_idx = 5

    def __init__(self, max_context: int, max_horizon: int, xreg_mode: str = "xreg + timesfm",
                 checkpoint: str = "google/timesfm-2.5-200m-pytorch") -> None:
        import timesfm

        self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint)
        self.m.compile(timesfm.ForecastConfig(
            max_context=max_context, max_horizon=max_horizon, normalize_inputs=True,
            use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=False,
            fix_quantile_crossing=True, return_backcast=True))
        self.mode = xreg_mode
        self.max_horizon = max_horizon

    def predict(self, batch: CovBatch, H: int | None = None) -> np.ndarray:
        H = H or batch.H
        ins = [batch.target[i].astype(np.float64) for i in range(batch.n)]
        if batch.k == 0:
            _, q = self.m.forecast(horizon=self.max_horizon, inputs=ins)
            q = [np.asarray(x)[-self.max_horizon:] for x in q]  # return_backcast=True prepends the backcast
        else:
            cov = {f"u{j}": [np.concatenate([batch.cov_past[i, j], batch.cov_future[i, j]]).astype(np.float64)
                             for i in range(batch.n)] for j in range(batch.k)}
            _, q = self.m.forecast_with_covariates(inputs=ins, dynamic_numerical_covariates=cov, xreg_mode=self.mode)
        q = np.stack([np.asarray(x)[:H] for x in q]).astype(np.float32)  # (N, H, 10)
        return np.transpose(q, (0, 2, 1))  # (N, 10, H)
