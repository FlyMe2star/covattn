"""Repairs that act only on the *target <- covariate* read of group attention (Exp3).

R1 (training-free): multiply the cross part (sum over covariate keys of A * v, target queries, all
    positions) of the top-k heads by one gamma. Heads are ranked by the Exp2 per-head `cross`
    restoration ratio (val split); k and gamma are chosen on synthetic val.
R2 (gate learning): one scalar gate per (layer, head) on the same cross part, initialised to 1
    (= original model). All model weights stay frozen; 144 trainable parameters.
    Loss = effect matching on counterfactual pairs + lambda * quantile loss on the factual forecast
    + reg * mean (gate - 1)^2.

Constructive guarantee: with no covariate in the group the cross part is identically zero, so R1/R2
leave a univariate forecast unchanged (tests/test_repair.py).

R3 (REG logit bias) is *not* used as a repair: Exp2 did not support H2 (a negative REG bias lowered
g and raised MASE; see reports/exp2_summary_main.md). The bias intervention stays in Exp2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from covattn import metrics as M
from covattn.hooks import HookSpec
from covattn.model import CovBatch
from covattn.systems import SampleSpec, SystemSpec, make_counterfactual, make_rng, make_sample

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------ data
@dataclass
class PairBatch:
    """Counterfactual pairs: fac / cf differ only in the future covariate."""

    fac: CovBatch
    cf: CovBatch
    e: np.ndarray  # (N, H) true effect
    du: np.ndarray  # (N, H)
    y_true: np.ndarray  # (N, H) factual future
    y_ctx: np.ndarray  # (N, L)

    @property
    def n(self) -> int:
        return self.fac.n


def make_pairs(items: list[tuple[SystemSpec, str, str, float]], L: int, horizon: int, rng: np.random.Generator,
               snr: float = 10.0, bg_scale: float = 1.0) -> PairBatch:
    """items: (system, input_type, shape, delta_mult) per pair, all with context length L."""
    samples, cfs = [], []
    for sys, inp, shape, dm in items:
        spec = SampleSpec(context_length=L, horizon=horizon, input_type=inp, snr=snr, bg_scale=bg_scale)
        s = make_sample(sys, spec, rng)
        samples.append(s)
        cfs.append(make_counterfactual(s, shape, dm))
    y = np.stack([s.y[:L] for s in samples])
    u = np.stack([s.u[:L] for s in samples])
    return PairBatch(
        fac=CovBatch(y, u, np.stack([s.u[L:] for s in samples])),
        cf=CovBatch(y, u, np.stack([c.u_future_cf for c in cfs])),
        e=np.stack([c.effect for c in cfs]),
        du=np.stack([c.du for c in cfs]),
        y_true=np.stack([s.y[L:] for s in samples]),
        y_ctx=y,
    )


def random_pairs(systems: list[SystemSpec], inputs: list[str], shapes: list[str], deltas: list[float],
                 contexts: list[int], n: int, horizon: int, seed: int, split: str, tag: str) -> PairBatch:
    """A deterministic random mix (one context length per batch) for training / selection."""
    rng = make_rng(seed, split, tag)
    L = int(rng.choice(contexts))
    items = [(systems[rng.integers(len(systems))], inputs[rng.integers(len(inputs))],
              shapes[rng.integers(len(shapes))], float(deltas[rng.integers(len(deltas))])) for _ in range(n)]
    return make_pairs(items, L, horizon, rng)


# ------------------------------------------------------------------------------------ objective
def objective_np(q_fac: np.ndarray, q_cf: np.ndarray, pb: PairBatch, quantiles: list[float], mi: int,
                 lam: float, thresh: float = 0.05) -> dict:
    """Same criterion as the R2 loss: mean ||e_hat - e||^2 / ||e||^2 (kept pairs) + lam * scaled QL."""
    keep = M.effect_mask(pb.e, pb.y_ctx.std(-1), thresh)
    e_hat = q_cf[:, mi] - q_fac[:, mi]
    eff = ((e_hat - pb.e) ** 2).sum(-1) / np.maximum((pb.e ** 2).sum(-1), 1e-12)
    ql = _ql_np(q_fac, pb.y_true, quantiles) / np.maximum(pb.y_ctx.std(-1), 1e-6)
    eff_m = float(eff[keep].mean()) if keep.any() else float("nan")
    return {"obj": eff_m + lam * float(ql.mean()), "eff": eff_m, "ql": float(ql.mean()),
            "g_median": float(np.median(M.effect_gain(e_hat[keep], pb.e[keep]))) if keep.any() else float("nan")}


def _ql_np(q: np.ndarray, y: np.ndarray, quantiles: list[float]) -> np.ndarray:
    lv = np.asarray(quantiles)[None, :, None]
    yy = y[:, None, :]
    return (2 * np.abs((yy - q) * ((yy <= q).astype(np.float64) - lv))).mean(axis=(1, 2))


def _ql_torch(q: torch.Tensor, y: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    yy = y[:, None, :]
    lv = quantiles[None, :, None]
    return (2 * torch.abs((yy - q) * ((yy <= q).float() - lv))).mean(dim=(1, 2))


# ------------------------------------------------------------------------------------ R1
def head_ranking(exp2_run_dir: Path, L: int = 512, comp: str = "cross",
                 systems: list[str] | None = None) -> list[tuple[int, int]]:
    """(layer, head) sorted by pooled per-head restoration over the Exp2 cells at L (only cells of `systems` if
    given; cell files are named <system>_L<L>_<input>_<shape>.npz).

    The per-head patches exist for target future AND context positions (2 x n_layers x n_heads entries); all of
    them are ranked together, so a head's rank is set by the larger of its future- and context-position
    restoration and a head can appear twice in the returned list (callers take the first k entries; in the
    canonical run exp2/main at L=512 the top 20 are 20 distinct heads). This is the ranking used by R1 in all
    reported runs."""
    files = sorted((exp2_run_dir / "cells").glob(f"*_L{L}_*.npz"))
    if systems is not None:
        files = [f for f in files if any(f.name.startswith(f"{s}_L{L}_") for s in systems)]
    if not files:
        raise FileNotFoundError(f"no Exp2 cells for L={L} in {exp2_run_dir}")
    num, den, names = None, 0.0, None
    for f in files:
        z = np.load(f, allow_pickle=False)
        nm = z["patch_names"]
        idx = [i for i, n in enumerate(nm) if n.startswith(comp + "|") and "|h" in n]
        if not idx:
            continue
        k = z["keep"]
        nn = z["patch_num"][idx][:, k].sum(1)
        num = nn if num is None else num + nn
        den += float(z["patch_den"][k].sum())
        names = [nm[i] for i in idx]
    if names is None:
        raise ValueError(f"no per-head `{comp}` patches at L={L}")
    r = num / max(den, 1e-12)
    order = np.argsort(-r)
    out = []
    for o in order:
        _, lay, _, hh = names[o].split("|")
        out.append((int(lay[1:]), int(hh[1:])))
    return out


def r1_gain(n_layers: int, n_heads: int, heads: list[tuple[int, int]], gamma: float) -> torch.Tensor:
    g = torch.ones(n_layers, n_heads)
    for l, h in heads:
        g[l, h] = gamma
    return g


def gain_spec(gain: torch.Tensor | None) -> HookSpec:
    return HookSpec(cross_gain=gain, cross_gain_positions="all")


# ------------------------------------------------------------------------------------ R2
@dataclass
class R2Config:
    steps: int = 300
    lr: float = 0.02
    pairs_per_step: int = 16
    lam: float = 1.0
    ckpt_every: int = 50
    clamp_min: float = 0.0
    reg: float = 0.0  # weight of mean (gate - 1)^2


def train_r2(w, cfg: R2Config, batch_fn, ckpt_path: Path, metrics_cb=None) -> torch.Tensor:
    """Learn (n_layers, n_heads) gates. batch_fn(step) -> PairBatch (deterministic per step, so a
    resumed run sees the same data). Checkpoints {gates, opt, step} to ckpt_path every ckpt_every steps."""
    hk = w.hooked
    dev = w.device
    gates = torch.ones(hk.n_layers, hk.n_heads, device=dev, requires_grad=True)
    opt = torch.optim.Adam([gates], lr=cfg.lr)
    start = 0
    if ckpt_path.exists():
        st = torch.load(ckpt_path, map_location=dev)
        with torch.no_grad():
            gates.copy_(st["gates"])
        opt.load_state_dict(st["opt"])
        start = int(st["step"])
        log.info("R2: resumed from step %d", start)
    qs = torch.tensor(w.quantiles, device=dev)
    mi = w.median_idx
    for step in range(start, cfg.steps):
        pb = batch_fn(step)
        P_fac, P_cf = w.prepare_hooked(pb.fac), w.prepare_hooked(pb.cf)
        spec = HookSpec(cross_gain=gates, cross_gain_positions="all", grad=True)
        q_fac, _ = hk.run(P_fac, spec)
        q_cf, _ = hk.run(P_cf, spec)
        e = torch.as_tensor(pb.e, device=dev, dtype=torch.float32)
        keep = torch.as_tensor(M.effect_mask(pb.e, pb.y_ctx.std(-1), 0.05), device=dev)
        e_hat = q_cf[:, mi] - q_fac[:, mi]
        eff = ((e_hat - e) ** 2).sum(-1) / (e ** 2).sum(-1).clamp_min(1e-12)
        eff_l = eff[keep].mean() if bool(keep.any()) else torch.zeros((), device=dev)
        scale = torch.as_tensor(pb.y_ctx.std(-1), device=dev, dtype=torch.float32).clamp_min(1e-6)
        ql = (_ql_torch(q_fac, torch.as_tensor(pb.y_true, device=dev, dtype=torch.float32), qs) / scale).mean()
        loss = eff_l + cfg.lam * ql + cfg.reg * ((gates - 1.0) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            gates.clamp_(min=cfg.clamp_min)
        if metrics_cb is not None:
            metrics_cb({"step": step + 1, "loss": loss.item(), "eff": eff_l.item(), "ql": ql.item(),
                        "gate_mean": float(gates.mean()), "gate_max": float(gates.max())})
        if (step + 1) % cfg.ckpt_every == 0 or step + 1 == cfg.steps:
            tmp = ckpt_path.with_suffix(".tmp")
            torch.save({"gates": gates.detach().cpu(), "opt": opt.state_dict(), "step": step + 1}, tmp)
            tmp.replace(ckpt_path)
    return gates.detach().cpu()
