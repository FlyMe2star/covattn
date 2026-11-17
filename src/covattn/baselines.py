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


# ------------------------------------------------------------------------------------ context dither (reimpl.)
def dither_predict(w, batch, sigma: float, K: int, seed: int):
    """Our re-implementation of context dithering: add sigma * std(target context) Gaussian noise to the
    target context, predict K times, average the quantiles. The noise depends only on `seed`, so a
    factual / counterfactual pair called with the same seed gets identical noise."""
    from covattn.model import CovBatch

    rng = np.random.default_rng(seed)
    sd = np.nanstd(batch.target, axis=-1, keepdims=True)
    acc = None
    for _ in range(K):
        noisy = batch.target + sigma * sd * rng.standard_normal(batch.target.shape)
        q = w.predict(CovBatch(noisy, batch.cov_past, batch.cov_future), batch.H)
        acc = q if acc is None else acc + q
    return acc / K


# ------------------------------------------------------------------------------------ fine-tuning
LORA_CONFIG = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "target_modules": ["self_attention.q", "self_attention.v"]}


def finetune(w, mode: str, train_inputs: list[dict], out_dir, prediction_length: int, context_length: int,
             learning_rate: float, num_steps: int, batch_size: int, seed: int, grad_accum: int = 1) -> dict:
    """Fine-tune a copy of the model with the official `Chronos2Pipeline.fit` (mode: full | lora).
    LoRA: rank 8 on the attention q / v projections (time and group attention), merged after training.
    batch_size = effective rows per optimizer step, split into grad_accum micro-batches.
    Saves the model to out_dir/model and returns {path, trainable_params, total_params}."""
    from pathlib import Path

    import torch
    from chronos import Chronos2Pipeline

    out_dir = Path(out_dir)
    model_dir = out_dir / "model"
    torch.manual_seed(seed)
    np.random.seed(seed)  # Chronos2Dataset samples windows with the global numpy RNG
    kw = {"lora_config": dict(LORA_CONFIG)} if mode == "lora" else {}
    ft = w.pipeline.fit(
        train_inputs, prediction_length=prediction_length, finetune_mode=mode, context_length=context_length,
        learning_rate=learning_rate, num_steps=num_steps, batch_size=batch_size // grad_accum,
        gradient_accumulation_steps=grad_accum, output_dir=out_dir / "trainer",
        remove_printer_callback=True, seed=seed, data_seed=seed, logging_steps=50, **kw,
    )
    m = ft.model
    n_total = sum(p.numel() for p in m.parameters())
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    if hasattr(m, "merge_and_unload"):
        m = m.merge_and_unload()
    Chronos2Pipeline(model=m).save_pretrained(model_dir)
    import shutil

    shutil.rmtree(out_dir / "trainer" / "finetuned-ckpt", ignore_errors=True)  # duplicate of model_dir / adapter
    return {"path": str(model_dir), "trainable_params": int(n_train), "total_params": int(n_total)}
