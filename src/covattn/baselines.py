"""Baselines. Stage 0: ARX fitted on the context only (order chosen by AIC).

ARX model:  y_t = c + sum_{i=1..na} a_i y_{t-i} + sum_{j=0..nb-1} b_j u_{t-j} + e_t
Its predicted effect of a future covariate change du is the zero-state response of
B(q)/A(q) to du, i.e. exactly what a correctly specified linear model would predict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

NA_GRID = (1, 2, 3, 4)
NB_GRID = (1, 2, 4, 8, 17, 33, 48)


@dataclass
class ARXFit:
    na: int
    nb: int
    a: np.ndarray  # (na,)
    b: np.ndarray  # (nb,)
    c: float
    aic: float

    def effect(self, du: np.ndarray) -> np.ndarray:
        A = np.concatenate([[1.0], -self.a])
        return lfilter(self.b, A, du)

    def stable(self) -> bool:
        if self.na == 0:
            return True
        roots = np.roots(np.concatenate([[1.0], -self.a]))
        return bool(np.all(np.abs(roots) < 1.0))


def _design(y: np.ndarray, u: np.ndarray, na: int, nb: int, start: int) -> tuple[np.ndarray, np.ndarray]:
    T = len(y)
    rows = T - start
    X = np.empty((rows, na + nb + 1))
    for i in range(1, na + 1):
        X[:, i - 1] = y[start - i : T - i]
    for j in range(nb):
        X[:, na + j] = u[start - j : T - j]
    X[:, -1] = 1.0
    return X, y[start:]


def fit_arx(y: np.ndarray, u: np.ndarray, na_grid=NA_GRID, nb_grid=NB_GRID) -> ARXFit:
    """Least-squares ARX on the context, (na, nb) by AIC on a common estimation window."""
    start = max(max(na_grid), max(nb_grid) - 1)
    best: ARXFit | None = None
    for na in na_grid:
        for nb in nb_grid:
            if len(y) - start < 4 * (na + nb + 1):
                continue
            X, t = _design(y, u, na, nb, start)
            theta, *_ = np.linalg.lstsq(X, t, rcond=None)
            rss = float(((t - X @ theta) ** 2).sum())
            n = len(t)
            k = na + nb + 1
            aic = n * np.log(max(rss / n, 1e-300)) + 2 * k
            if best is None or aic < best.aic:
                best = ARXFit(na, nb, theta[:na].copy(), theta[na : na + nb].copy(), float(theta[-1]), aic)
    assert best is not None, "context too short for any ARX order"
    return best


def arx_effects(y_ctx: np.ndarray, u_ctx: np.ndarray, dus: np.ndarray) -> tuple[np.ndarray, list[ARXFit]]:
    """y_ctx, u_ctx: (N, L); dus: (N, M, H) -> effects (N, M, H)."""
    out = np.empty_like(dus, dtype=np.float64)
    fits = []
    for i in range(y_ctx.shape[0]):
        f = fit_arx(y_ctx[i].astype(np.float64), u_ctx[i].astype(np.float64))
        fits.append(f)
        for m in range(dus.shape[1]):
            out[i, m] = f.effect(dus[i, m])
    return out, fits
