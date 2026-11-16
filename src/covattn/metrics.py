"""Effect-faithfulness metrics and forecast-accuracy metrics.

Conventions (all arrays are numpy, samples on axis 0, horizon on the last axis):
  e_hat : predicted effect  (N, H)   e.g. q50_cf - q50_fac
  e     : true effect       (N, H)
  du    : covariate change  (N, H)
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def effect_mask(e: np.ndarray, y_scale: np.ndarray, rel_thresh: float) -> np.ndarray:
    """Keep pairs whose true effect is non-negligible: RMS(e) > rel_thresh * y_scale."""
    rms = np.sqrt((e**2).mean(-1))
    return rms > rel_thresh * np.asarray(y_scale)


def effect_gain(e_hat: np.ndarray, e: np.ndarray) -> np.ndarray:
    """g_i = <e_hat_i, e_i> / ||e_i||^2  (1 = faithful, <1 = attenuated)."""
    return (e_hat * e).sum(-1) / np.maximum((e * e).sum(-1), EPS)


def gain_curve(e_hat: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Pooled per-horizon gain g(h) = sum_i e_hat_i(h) e_i(h) / sum_i e_i(h)^2.

    Pooling over samples avoids dividing by near-zero per-sample e_i(h)."""
    num = (e_hat * e).sum(0)
    den = (e * e).sum(0)
    out = np.full(e.shape[-1], np.nan)
    ok = den > EPS
    out[ok] = num[ok] / den[ok]
    return out


def shape_corr(e_hat: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Pearson correlation per pair over the horizon (NaN if either side is constant)."""
    a = e_hat - e_hat.mean(-1, keepdims=True)
    b = e - e.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    out = np.full(e.shape[0], np.nan)
    ok = den > EPS
    out[ok] = (a * b).sum(-1)[ok] / den[ok]
    return out


def whatif_nrmse(e_hat: np.ndarray, e: np.ndarray) -> np.ndarray:
    return np.linalg.norm(e_hat - e, axis=-1) / np.maximum(np.linalg.norm(e, axis=-1), EPS)


def memoryless_r2(resp: np.ndarray, du: np.ndarray) -> np.ndarray:
    """Uncentered R^2 of resp(h) ~ beta * du(h) (same-time regression, no intercept).

    1 = response is a pure instantaneous rescaling of du (memoryless); lower values mean
    the response has dynamics (lags, accumulation, oscillation). Uncentered because du is
    constant for step changes, where a centred R^2 is undefined."""
    beta = (resp * du).sum(-1) / np.maximum((du * du).sum(-1), EPS)
    resid = resp - beta[:, None] * du
    tot = (resp * resp).sum(-1)
    out = np.full(resp.shape[0], np.nan)
    ok = tot > EPS
    out[ok] = 1.0 - (resid * resid).sum(-1)[ok] / tot[ok]
    return out


def summarize(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan"), "mean": float("nan"), "n": 0}
    q25, med, q75 = np.percentile(x, [25, 50, 75])
    return {"median": float(med), "q25": float(q25), "q75": float(q75), "mean": float(x.mean()), "n": int(x.size)}


# --------------------------------------------------------------------------------------
# forecast accuracy
# --------------------------------------------------------------------------------------
def mase(point: np.ndarray, y_true: np.ndarray, y_ctx: np.ndarray, season: int = 1) -> np.ndarray:
    """Per-series MASE with seasonal-naive in-sample scale computed on the context."""
    diffs = np.abs(y_ctx[:, season:] - y_ctx[:, :-season])
    scale = np.nanmean(diffs, axis=-1)
    err = np.nanmean(np.abs(point - y_true), axis=-1)
    return err / np.maximum(scale, EPS)


def wql(quantiles: np.ndarray, levels: np.ndarray | list[float], y_true: np.ndarray) -> np.ndarray:
    """Per-series weighted quantile loss (as in fev / GluonTS):
    WQL = sum_q sum_h 2|(y - q)(1{y<=q} - tau)| / (|Q| * sum_h |y|).

    quantiles: (N, Q, H); levels: (Q,); y_true: (N, H)."""
    lv = np.asarray(levels)[None, :, None]
    y = y_true[:, None, :]
    loss = 2.0 * np.abs((y - quantiles) * ((y <= quantiles).astype(np.float64) - lv))
    num = loss.sum(axis=(1, 2)) / len(levels)
    den = np.abs(y_true).sum(-1)
    return num / np.maximum(den, EPS)


def pooled_wql(quantiles: np.ndarray, levels: np.ndarray | list[float], y_true: np.ndarray) -> float:
    """Dataset-level WQL: total loss / total |y| (the definition used in Chronos papers)."""
    lv = np.asarray(levels)[None, :, None]
    y = y_true[:, None, :]
    loss = 2.0 * np.abs((y - quantiles) * ((y <= quantiles).astype(np.float64) - lv))
    return float(loss.sum() / len(levels) / max(np.abs(y_true).sum(), EPS))
