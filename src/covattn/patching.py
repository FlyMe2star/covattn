"""Activation / path patching and simple global interventions for Exp2.

Restoration ratio of a patch config c for pair i (factual base, counterfactual source):
    num_i = <e_patch_i, e_cf_i>,  den_i = ||e_cf_i||^2,  r_i = num_i / den_i
with e_patch = q50(fac run with c patched in from cf) - q50(fac) and e_cf = q50(cf) - q50(fac).
Pooled ratio over a set of pairs = sum(num) / sum(den). Patches at layer l restart the factual run
from the cached residual stream entering block l (exactly equal to a full run, tested).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from covattn.hooks import HookSpec, Patch, Prepared

LAYER_COMPONENTS = ("time_out", "group_out", "mlp_out", "cross", "head")
HEAD_COMPONENTS = ("cross", "head")


@dataclass(frozen=True)
class PatchConfig:
    component: str
    layer: int
    positions: str
    head: int | None = None  # None = all heads (layer level)

    @property
    def name(self) -> str:
        h = "all" if self.head is None else f"h{self.head}"
        return f"{self.component}|L{self.layer}|{self.positions}|{h}"


def build_configs(n_layers: int, n_heads: int, per_head: bool,
                  layer_positions=("future", "context"), head_positions=("future",)) -> list[PatchConfig]:
    cfgs = [PatchConfig(c, l, p) for c in LAYER_COMPONENTS for l in range(n_layers) for p in layer_positions]
    if per_head:
        cfgs += [PatchConfig(c, l, p, h) for c in HEAD_COMPONENTS for l in range(n_layers) for h in range(n_heads)
                 for p in head_positions]
    return cfgs


@torch.no_grad()
def patch_sweep(w, P_fac: Prepared, rec_fac: dict, rec_cf: dict, q_fac_med: torch.Tensor,
                configs: list[PatchConfig]) -> np.ndarray:
    """Predicted effects (n_cfg, N, H) of the factual run with each config patched in from cf."""
    out = []
    mi = w.median_idx
    for c in configs:
        p = Patch(c.component, c.layer, c.positions, None if c.head is None else [c.head], source=rec_cf)
        spec = HookSpec(patches=[p], start_layer=c.layer, start_hidden=rec_fac["block_in"][c.layer])
        q, _ = w.forward_with_hooks(P_fac, spec)
        out.append((q[:, mi] - q_fac_med).cpu().numpy())
    return np.stack(out)


def restoration(e_patch: np.ndarray, e_cf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(num (n_cfg, N), den (N,)) for the restoration ratio."""
    return (e_patch * e_cf[None]).sum(-1), (e_cf * e_cf).sum(-1)


@torch.no_grad()
def intervention_sweep(w, P_fac: Prepared, P_cf: Prepared, specs: list[HookSpec]) -> tuple[np.ndarray, np.ndarray]:
    """For each spec applied to both runs: (effects (n, N, H), factual quantiles (n, N, Q, H))."""
    eff, qf_all = [], []
    mi = w.median_idx
    for s in specs:
        qf, _ = w.forward_with_hooks(P_fac, s)
        qc, _ = w.forward_with_hooks(P_cf, s)
        eff.append((qc[:, mi] - qf[:, mi]).cpu().numpy())
        qf_all.append(qf.cpu().numpy())
    return np.stack(eff), np.stack(qf_all)


def reg_bias_specs(n_layers: int, values: list[float]) -> list[HookSpec]:
    return [HookSpec(reg_bias=torch.full((n_layers,), float(b))) for b in values]


def cross_gain_specs(n_layers: int, n_heads: int, values: list[float]) -> list[HookSpec]:
    return [HookSpec(cross_gain=torch.full((n_layers, n_heads), float(g))) for g in values]
