"""Real data for Exp4 (public, downloaded from the Hugging Face Hub and cached there).

Covariate tasks: the five EPF electricity-price series of `autogluon/fev_datasets`, hourly, each
with two day-ahead forecasts as known future covariates (epf_de is the same German series as
the Chronos-2 README example, autogluon S3 electricity_price). Verified 2026-11-17:
  epf_de: target + "Ampirion Load Forecast", "PV+Wind Forecast" (52416 hourly points)
  epf_np: target + "Grid load forecast", "Wind power forecast"
Own data: `load_csv_task` reads a CSV with timestamp, target and covariate columns.

Univariate tasks: series from `autogluon/chronos_datasets`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EPF = ("epf_be", "epf_de", "epf_fr", "epf_np", "epf_pjm")
# name -> (horizon, season for MASE)
UNIVARIATE = {
    "exchange_rate": (30, 1),
    "monash_hospital": (12, 12),
    "monash_tourism_monthly": (24, 12),
    "nn5": (56, 7),
    "ercot": (24, 24),
}


@dataclass
class CovTask:
    name: str
    target: np.ndarray  # (T,)
    covs: np.ndarray  # (K, T) known future covariates
    cov_names: list[str]
    season: int = 24


def _hub_parquets(repo: str, prefix: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if f.startswith(prefix + "/") and f.endswith(".parquet"))
    if not files:
        raise FileNotFoundError(f"{repo}: no parquet under {prefix}/")
    return pd.concat([pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset")) for f in files], ignore_index=True)


def load_epf(name: str) -> CovTask:
    df = _hub_parquets("autogluon/fev_datasets", name)
    assert len(df) == 1, f"{name}: expected one series, got {len(df)}"
    row = df.iloc[0]
    cov_names = [c for c in df.columns if c not in ("id", "timestamp", "target")]
    covs = np.stack([np.asarray(row[c], dtype=np.float32) for c in cov_names])
    return CovTask(name, np.asarray(row["target"], dtype=np.float32), covs, cov_names, season=24)


def load_csv_task(path: str | Path, target_col: str = "target", cov_cols: list[str] | None = None,
                  season: int = 24, name: str | None = None) -> CovTask:
    """CSV with columns timestamp, <target_col>, covariates (all treated as known in the future)."""
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    cov_cols = cov_cols or [c for c in df.columns if c not in ("timestamp", target_col)]
    return CovTask(name or Path(path).stem, df[target_col].to_numpy(np.float32),
                   np.stack([df[c].to_numpy(np.float32) for c in cov_cols]), list(cov_cols), season)


def load_univariate(name: str) -> list[np.ndarray]:
    df = _hub_parquets("autogluon/chronos_datasets", name)
    return [np.asarray(v, dtype=np.float32) for v in df["target"]]


def rolling_windows(task: CovTask, L: int, H: int, n_windows: int, step: int | None = None):
    """Last n_windows forecast origins (step = H by default), most recent last.
    Returns target ctx (N, L), cov past (N, K, L), cov future (N, K, H), y_true (N, H)."""
    step = step or H
    T = len(task.target)
    origins = [T - H - i * step for i in range(n_windows)][::-1]
    origins = [o for o in origins if o - L >= 0]
    y = np.stack([task.target[o - L : o] for o in origins])
    cp = np.stack([task.covs[:, o - L : o] for o in origins])
    cf = np.stack([task.covs[:, o : o + H] for o in origins])
    yt = np.stack([task.target[o : o + H] for o in origins])
    return y, cp, cf, yt


def last_window(series: list[np.ndarray], L: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Context = up to L points before the last H (left-padded with NaN); y_true = the last H points."""
    ctx = np.full((len(series), L), np.nan, dtype=np.float32)
    yt = np.empty((len(series), H), dtype=np.float32)
    for i, s in enumerate(series):
        c = s[:-H][-L:]
        ctx[i, L - len(c):] = c
        yt[i] = s[-H:]
    return ctx, yt
