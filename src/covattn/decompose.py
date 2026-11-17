"""Exact decomposition statistics on recorded runs (see covattn.hooks for the record layout).

All functions take records of a factual and a counterfactual run of the *same* tasks (same row
order) and return numpy arrays. "future" / "context" refer to the target's token positions.
"""

from __future__ import annotations

import numpy as np
import torch

from covattn.hooks import HookedChronos2, Prepared

POS_GROUPS = ("future", "context")


@torch.no_grad()
def head_diff_norms(hk: HookedChronos2, P: Prepared, rec_fac: dict, rec_cf: dict, key: str) -> np.ndarray:
    """Mean over target rows and positions of ||W_O^h (x_cf - x_fac)|| per (layer, head, pos-group),
    with x = per-head context `key` in {cross, self, head}. Returns (n_layers, n_heads, 2)."""
    out = np.zeros((hk.n_layers, hk.n_heads, len(POS_GROUPS)))
    for l in range(hk.n_layers):
        W = hk.head_out_weight(l)  # (H, d, dk)
        diff = rec_cf[key][l] - rec_fac[key][l]  # (T, H, N, dk)
        for g, which in enumerate(POS_GROUPS):
            d = diff[P.pos(which)]  # (t, H, N, dk)
            proj = torch.einsum("thnk,hdk->thnd", d, W)
            out[l, :, g] = proj.norm(dim=-1).mean(dim=(0, 2)).cpu().numpy()
    return out


@torch.no_grad()
def cov_attention_mass(P: Prepared, rec: dict, n_layers: int) -> np.ndarray:
    """Group-attention mass of target queries on covariate keys, mean over rows/positions: (L, H, 2)."""
    out = []
    for l in range(n_layers):
        gw = rec["group_w"][l]  # (T, H, N)
        out.append([gw[P.pos(w)].mean(dim=(0, 2)).cpu().numpy() for w in POS_GROUPS])
    return np.transpose(np.array(out), (0, 2, 1))  # (L, H, 2)


@torch.no_grad()
def lens_effects(hk: HookedChronos2, P_fac: Prepared, P_cf: Prepared, rec_fac: dict, rec_cf: dict,
                 median_idx: int) -> np.ndarray:
    """Effect read out from the residual stream after every block through final norm + output head
    (a logit-lens style readout; early layers are not trained to be decoded). Returns (L, N, H)."""
    out = []
    for l in range(hk.n_layers):
        qf = hk.decode(rec_fac["resid_fut"][l], P_fac)[:, median_idx]
        qc = hk.decode(rec_cf["resid_fut"][l], P_cf)[:, median_idx]
        out.append((qc - qf).cpu().numpy())
    return np.stack(out)


def reg_shares(rec: dict, n_layers: int) -> np.ndarray:
    """Time-attention mass of target future queries on [REG, future, context]: (N, L, H, 3)."""
    return np.stack([rec["time_w"][l].cpu().numpy() for l in range(n_layers)], axis=1)
