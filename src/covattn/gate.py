"""Training-free gate that decides, per forecast window, whether to apply the repair (docs/gate_protocol.md).

Uses only the context and the known future input:
  1. input share s_k = 1 - RSS(ARX) / RSS(AR) per covariate k (ARX orders by AIC on the `baselines.fit_arx`
     grid; AR = same na, no input, same estimation rows); k* = argmax s_k, s = s_{k*};
  2. attenuation index rho = <e_hat, e_arx> / ||e_arx||^2 for a probe du = +1 context-std of u_{k*}, a step over
     the horizon: e_arx = zero-state response of the ARX model to du, e_hat = q50(u_fut + du) - q50(u_fut) of
     zero-shot Chronos-2;
  3. repair iff s >= tau_s and rho < tau_rho (tau_rho = inf: no condition on rho).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from covattn.baselines import NA_GRID, NB_GRID, _design, fit_arx

TAU_S = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
TAU_RHO = (0.6, 0.7, 0.8, 0.9, 1.0, float("inf"))


@dataclass
class WindowStats:
    s: float  # input share of the best covariate
    k: int  # index of that covariate
    du: float  # probe size (context std of u_k)
    e_arx: np.ndarray  # (H,) ARX response to the probe


def input_share(y: np.ndarray, u: np.ndarray):
    """(s, ARXFit) for one covariate on one context."""
    f = fit_arx(y, u)
    start = max(max(NA_GRID), max(NB_GRID) - 1)  # the common estimation rows of fit_arx
    X, t = _design(y, u, f.na, f.nb, start)
    theta = np.r_[f.a, f.b, f.c]
    rss = float(((t - X @ theta) ** 2).sum())
    X0, t0 = _design(y, u, f.na, 0, start)
    th0, *_ = np.linalg.lstsq(X0, t0, rcond=None)
    rss0 = float(((t0 - X0 @ th0) ** 2).sum())
    if not np.isfinite(rss0) or rss0 <= 1e-12 * max(float((t0 ** 2).sum()), 1e-300):
        return 0.0, f
    return float(min(max(1.0 - rss / rss0, 0.0), 1.0)), f


MIN_FIT = 8  # below this nothing is fitted; otherwise fit_arx decides (it needs >= 4 rows per parameter)


def finite_context(y: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For the gate statistics only: drop leading positions until the first point where the target and every
    covariate are finite (left padding), then fill remaining NaNs by linear interpolation (edge values held)."""
    y = np.asarray(y, np.float64)
    U = np.atleast_2d(np.asarray(U, np.float64))
    ok = np.isfinite(y) & np.isfinite(U).all(0)
    if not ok.any():
        return y[:0], U[:, :0]
    a = int(np.argmax(ok))
    y, U = y[a:].copy(), U[:, a:].copy()

    def fill(x):
        m = np.isfinite(x)
        if m.all() or not m.any():
            return x if m.any() else np.zeros_like(x)
        i = np.arange(len(x))
        return np.interp(i, i[m], x[m])

    return fill(y), np.stack([fill(u) for u in U])


def window_stats(y_ctx: np.ndarray, u_ctx: np.ndarray, H: int) -> WindowStats:
    """y_ctx (L,), u_ctx (K, L). NaNs handled by `finite_context`; a context too short for any
    fit_arx order (or an unfittable one) -> s = 0 and no probe, i.e. the gate stays off."""
    K = np.atleast_2d(u_ctx).shape[0]
    y, U = finite_context(y_ctx, u_ctx)
    best = None
    if len(y) >= MIN_FIT:
        for k in range(K):
            try:
                s, f = input_share(y, U[k])
            except (AssertionError, np.linalg.LinAlgError, ValueError):
                continue
            if best is None or s > best[0]:
                best = (s, k, f, U[k])
    if best is None:
        return WindowStats(s=0.0, k=0, du=0.0, e_arx=np.zeros(H))
    s, k, f, u = best
    du = float(np.std(u))
    return WindowStats(s=s, k=k, du=du, e_arx=f.effect(np.full(H, du)))


def batch_stats(y_ctx: np.ndarray, u_ctx: np.ndarray, H: int) -> list[WindowStats]:
    return [window_stats(y_ctx[i], u_ctx[i], H) for i in range(len(y_ctx))]


def probe_future(u_fut: np.ndarray, stats: list[WindowStats]) -> np.ndarray:
    """u_fut (N, K, H) -> u_fut with +du on channel k* of every window."""
    out = np.array(u_fut, dtype=np.float32, copy=True)
    for i, st in enumerate(stats):
        out[i, st.k] += st.du
    return out


def attenuation_index(e_hat: np.ndarray, stats: list[WindowStats]) -> np.ndarray:
    """e_hat (N, H) zero-shot probe response -> rho (N,), NaN where the ARX response is ~0."""
    rho = np.full(len(stats), np.nan)
    for i, st in enumerate(stats):
        den = float((st.e_arx ** 2).sum())
        if np.isfinite(den) and den > 1e-12 * max(st.du ** 2, 1e-300) and st.du > 0:
            rho[i] = float((np.asarray(e_hat[i], np.float64) * st.e_arx).sum() / den)
    return rho


def gate_on(s: np.ndarray, rho: np.ndarray, tau_s: float, tau_rho: float) -> np.ndarray:
    s, rho = np.asarray(s, np.float64), np.asarray(rho, np.float64)
    on = s >= tau_s
    if np.isfinite(tau_rho):
        on &= np.nan_to_num(rho, nan=np.inf) < tau_rho
    return on


def grid() -> list[tuple[float, float]]:
    return [(ts, tr) for ts in TAU_S for tr in TAU_RHO]


def conservative_key(t: tuple[float, float]) -> tuple[float, float]:
    """Sort key: more conservative first (larger tau_s, then smaller tau_rho)."""
    return (-t[0], t[1])


def select(scores: dict[tuple[float, float], float], tol: float = 0.002) -> tuple[float, float]:
    """Lowest criterion; settings within tol of it -> the most conservative one."""
    best = min(scores.values())
    near = [t for t, v in scores.items() if v <= best + tol]
    return sorted(near, key=conservative_key)[0]
