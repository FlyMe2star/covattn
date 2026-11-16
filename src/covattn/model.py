"""Chronos-2 wrapper: batched prediction through the official pipeline, plus a direct
model call with the same tensor layout (used later for hooks / interventions).

Layout facts used here are verified in docs/model_notes.md:
  * one *group* = [target, cov_1, ..., cov_K] rows sharing a group id;
  * every row is instance-normalised with loc/scale computed on its own **context only**;
    future covariates are normalised with the context loc/scale of the same row;
  * token sequence per row: [ctx patches (ceil(L/16))] [REG] [future patches (ceil(H/16))].
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

log = logging.getLogger("covattn.model")

AmpMode = Literal["off", "bf16", "fp16"]
PATCH = 16


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device, amp: AmpMode):
    if amp == "off":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def is_oom(err: BaseException) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()


@dataclass
class CovBatch:
    """N forecasting tasks, each = 1 target + K known-future covariates.

    target: (N, L); cov_past: (N, K, L); cov_future: (N, K, H)."""

    target: np.ndarray
    cov_past: np.ndarray
    cov_future: np.ndarray

    def __post_init__(self) -> None:
        self.target = np.asarray(self.target, dtype=np.float32)
        self.cov_past = np.asarray(self.cov_past, dtype=np.float32)
        self.cov_future = np.asarray(self.cov_future, dtype=np.float32)
        if self.cov_past.ndim == 2:
            self.cov_past = self.cov_past[:, None]
        if self.cov_future.ndim == 2:
            self.cov_future = self.cov_future[:, None]
        n, L = self.target.shape
        assert self.cov_past.shape[0] == n and self.cov_past.shape[2] == L, self.cov_past.shape
        assert self.cov_future.shape[:2] == self.cov_past.shape[:2], self.cov_future.shape

    @property
    def n(self) -> int:
        return self.target.shape[0]

    @property
    def k(self) -> int:
        return self.cov_past.shape[1]

    @property
    def L(self) -> int:
        return self.target.shape[1]

    @property
    def H(self) -> int:
        return self.cov_future.shape[2]

    def slice(self, a: int, b: int) -> "CovBatch":
        return CovBatch(self.target[a:b], self.cov_past[a:b], self.cov_future[a:b])

    def to_pipeline_dicts(self) -> list[dict]:
        out = []
        for i in range(self.n):
            past = {f"u{j}": self.cov_past[i, j] for j in range(self.k)}
            fut = {f"u{j}": self.cov_future[i, j] for j in range(self.k)}
            d: dict = {"target": self.target[i]}
            if self.k > 0:
                d["past_covariates"] = past
                d["future_covariates"] = fut
            out.append(d)
        return out


@dataclass(frozen=True)
class TokenLayout:
    n_ctx: int  # number of context patches
    reg: int  # index of the REG token
    n_fut: int  # number of future patches

    @property
    def seq_len(self) -> int:
        return self.n_ctx + 1 + self.n_fut

    @property
    def fut_slice(self) -> slice:
        return slice(self.reg + 1, self.reg + 1 + self.n_fut)

    @property
    def ctx_slice(self) -> slice:
        return slice(0, self.n_ctx)


def token_layout(L: int, H: int) -> TokenLayout:
    n_ctx = math.ceil(L / PATCH)
    return TokenLayout(n_ctx=n_ctx, reg=n_ctx, n_fut=math.ceil(H / PATCH))


class Chronos2Wrapper:
    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device: str = "auto",
        amp: AmpMode = "off",
        batch_size: int = 128,
        attn_implementation: str | None = None,
    ) -> None:
        from chronos import Chronos2Pipeline

        self.device = pick_device(device)
        self.amp = amp
        kwargs = {"device_map": str(self.device), "dtype": torch.float32}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        self.pipeline = Chronos2Pipeline.from_pretrained(model_id, **kwargs)
        self.model = self.pipeline.model
        self.model.eval()
        self.batch_size = batch_size  # in *series* (rows), as in the official pipeline
        self.quantiles: list[float] = list(self.pipeline.quantiles)
        self.median_idx = self.quantiles.index(0.5)
        self.oom_events: list[dict] = []

    # ---------------------------------------------------------------- official pipeline
    def predict(self, batch: CovBatch, prediction_length: int | None = None) -> np.ndarray:
        """Quantile forecasts of the target rows: (N, Q, H)."""
        H = prediction_length or batch.H
        outs: list[np.ndarray] = []
        i = 0
        rows_per_task = 1 + batch.k
        while i < batch.n:
            n_tasks = max(1, self.batch_size // rows_per_task)
            sub = batch.slice(i, min(batch.n, i + n_tasks))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with autocast_ctx(self.device, self.amp):
                        preds = self.pipeline.predict(
                            sub.to_pipeline_dicts(),
                            prediction_length=H,
                            batch_size=self.batch_size,
                            context_length=batch.L,
                        )
            except Exception as err:  # noqa: BLE001
                if is_oom(err) and self.batch_size > rows_per_task:
                    self._on_oom()
                    continue
                raise
            outs.append(np.stack([p[0].float().numpy() for p in preds]))  # target row = index 0
            i += sub.n
        return np.concatenate(outs, axis=0)

    def _on_oom(self) -> None:
        old = self.batch_size
        self.batch_size = max(1, old // 2)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.oom_events.append({"time": time.time(), "old_batch_size": old, "new_batch_size": self.batch_size})
        log.warning("OOM: batch_size %d -> %d (rows)", old, self.batch_size)

    # ---------------------------------------------------------------- direct model call
    def build_model_inputs(self, batch: CovBatch) -> dict[str, torch.Tensor | int]:
        """Same tensors the pipeline builds in TEST mode (equal-length inputs, H <= max patches)."""
        n, k, L, H = batch.n, batch.k, batch.L, batch.H
        rows = 1 + k
        ctx = np.empty((n, rows, L), dtype=np.float32)
        ctx[:, 0] = batch.target
        ctx[:, 1:] = batch.cov_past
        fut = np.full((n, rows, H), np.nan, dtype=np.float32)
        fut[:, 1:] = batch.cov_future
        n_patches = math.ceil(H / PATCH)
        if n_patches * PATCH > H:
            pad = np.full((n, rows, n_patches * PATCH - H), np.nan, dtype=np.float32)
            fut = np.concatenate([fut, pad], axis=-1)
        group_ids = np.repeat(np.arange(n), rows)
        return {
            "context": torch.from_numpy(ctx.reshape(n * rows, L)).to(self.device),
            "future_covariates": torch.from_numpy(fut.reshape(n * rows, -1)).to(self.device),
            "group_ids": torch.from_numpy(group_ids).to(self.device),
            "num_output_patches": n_patches,
        }

    @torch.no_grad()
    def forward_raw(self, batch: CovBatch, output_attentions: bool = False):
        """Call Chronos2Model.forward directly. Returns (quantiles_target (N,Q,H) tensor, model output)."""
        inp = self.build_model_inputs(batch)
        if int(inp["num_output_patches"]) > self.model.chronos_config.max_output_patches:
            raise ValueError("forward_raw does not implement the long-horizon unrolling heuristic")
        with autocast_ctx(self.device, self.amp):
            out = self.model(
                context=inp["context"],
                group_ids=inp["group_ids"],
                future_covariates=inp["future_covariates"],
                num_output_patches=int(inp["num_output_patches"]),
                output_attentions=output_attentions,
            )
        rows = 1 + batch.k
        q = out.quantile_preds.float()[::rows, :, : batch.H]
        return q, out

    # ---------------------------------------------------------------- throughput probe
    def measure_throughput(self, L: int, H: int, k: int = 1, n_tasks: int = 256, reps: int = 2) -> float:
        """Tasks/second for prediction at this (L, H, k) on the current device."""
        rng = np.random.default_rng(0)
        b = CovBatch(
            rng.standard_normal((n_tasks, L)), rng.standard_normal((n_tasks, k, L)), rng.standard_normal((n_tasks, k, H))
        )
        self.predict(b.slice(0, min(8, n_tasks)), H)  # warm-up
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            self.predict(b, H)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return reps * n_tasks / (time.perf_counter() - t0)


# Rough ratio of A100 vs RTX 4060 (laptop) throughput for this encoder at fp32. It is only used
# to print a second time estimate; the measured figure is always the one for the current device.
A100_OVER_4060 = 5.0


def time_estimates(seconds_here: float, device: torch.device) -> dict[str, float]:
    """Measured time on this device plus scaled guesses for RTX4060 / A100. Keys: 'here', 'RTX4060', 'A100'.
    When the current device is neither GPU (e.g. CPU), the GPU figures are unknown (NaN)."""
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    if "A100" in name:
        return {"here": seconds_here, "A100": seconds_here, "RTX4060": seconds_here * A100_OVER_4060}
    if "4060" in name:
        return {"here": seconds_here, "RTX4060": seconds_here, "A100": seconds_here / A100_OVER_4060}
    return {"here": seconds_here, "RTX4060": float("nan"), "A100": float("nan")}
