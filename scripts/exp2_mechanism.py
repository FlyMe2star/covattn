"""Exp2 - where is covariate information attenuated inside Chronos-2?

Per cell (system, L, input, shape; delta = 1 std) on the **val** split (Exp2 selects heads for the
repair, so it must not touch test):
  1. exact decomposition of group attention (cross / self / head diff norms, covariate attention mass)
  2. activation patching cf -> fac: time_out, group_out, mlp_out, cross, head per layer at target
     future / context positions; per-head cross & head at future positions (only for L in
     --per-head-contexts)
  3. residual-stream readout ("lens") of the effect after every block
  4. REG sink: time-attention mass of target future queries on REG / future / context;
     REG logit bias b in {0,-1,-2,-4} on target future queries (all layers)
  5. global cross-gain sweep gamma (all layers & heads), a preview for the repair
Report: $RESULTS_DIR/reports/exp2_summary_<run>.md + figures in reports/figs/.

  python scripts/exp2_mechanism.py --quick
  python scripts/exp2_mechanism.py --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from covattn import metrics as M
from covattn.decompose import cov_attention_mass, head_diff_norms, lens_effects, reg_shares
from covattn.hooks import HookSpec
from covattn.io_utils import RunDir, fmt_seconds, results_root, setup_logging
from covattn.model import CovBatch, time_estimates
from covattn.patching import (
    build_configs,
    cross_gain_specs,
    intervention_sweep,
    patch_sweep,
    reg_bias_specs,
    restoration,
)
from covattn.systems import SampleSpec, default_systems, make_counterfactual, make_rng, make_sample, quick_systems

log = logging.getLogger("exp2")


@dataclass
class Exp2Config:
    run_name: str = "main"
    quick: bool = False
    systems: list[str] = field(default_factory=lambda: ["F1", "F2_a0.8", "F2_a0.99", "F3_a0.8_d32",
                                                         "F4_P24_z0.2", "F4_P96_z0.05"])
    contexts: list[int] = field(default_factory=lambda: [512, 2048])
    per_head_contexts: list[int] = field(default_factory=lambda: [512])
    horizon: int = 64
    input_types: list[str] = field(default_factory=lambda: ["prbs", "multisine"])
    shapes: list[str] = field(default_factory=lambda: ["step", "pulse"])
    delta_mult: float = 1.0
    n_per_cell: int = 64
    snr: float = 10.0
    bg_scale: float = 1.0
    eff_thresh: float = 0.05
    reg_bias_values: list[float] = field(default_factory=lambda: [0.0, -1.0, -2.0, -4.0])
    cross_gain_values: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    seed: int = 0
    split: str = "val"
    model_id: str = "amazon/chronos-2"
    amp: str = "off"
    device: str = "auto"
    top_k: int = 10


def parse_args() -> tuple[Exp2Config, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--contexts", type=int, nargs="+", default=None)
    p.add_argument("--n-per-cell", type=int, default=None)
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    cfg = Exp2Config(quick=a.quick, amp=a.amp, device=a.device, seed=a.seed)
    if a.quick:
        cfg.run_name = "quick"
        cfg.systems = ["F2_a0.99", "F4_P24_z0.2"]
        cfg.contexts = [256]
        cfg.per_head_contexts = [256]
        cfg.input_types = ["prbs"]
        cfg.shapes = ["step"]
        cfg.n_per_cell = 8
        cfg.reg_bias_values = [0.0, -4.0]
        cfg.cross_gain_values = [1.0, 2.0]
    if a.contexts is not None:
        cfg.contexts = a.contexts
    if a.n_per_cell is not None:
        cfg.n_per_cell = a.n_per_cell
    if a.run_name is not None:
        cfg.run_name = a.run_name
    return cfg, a


def cell_key(s: str, L: int, inp: str, shape: str) -> str:
    return f"{s}|L{L}|{inp}|{shape}"


def build_batches(cfg: Exp2Config, sys, L: int, inp: str, shape: str):
    spec = SampleSpec(context_length=L, horizon=cfg.horizon, input_type=inp, snr=cfg.snr, bg_scale=cfg.bg_scale)
    rng = make_rng(cfg.seed, cfg.split, "exp2", sys.key, L, inp)  # same factual samples for both shapes
    samples = [make_sample(sys, spec, rng) for _ in range(cfg.n_per_cell)]
    cfs = [make_counterfactual(s, shape, cfg.delta_mult) for s in samples]
    y = np.stack([s.y[:L] for s in samples])
    u = np.stack([s.u[:L] for s in samples])
    fac = CovBatch(y, u, np.stack([s.u[L:] for s in samples]))
    cf = CovBatch(y, u, np.stack([c.u_future_cf for c in cfs]))
    e = np.stack([c.effect for c in cfs])
    du = np.stack([c.du for c in cfs])
    y_true = np.stack([s.y[L:] for s in samples])
    return fac, cf, e, du, y_true, y


def run_cell(w, cfg: Exp2Config, sys, L: int, inp: str, shape: str, run: RunDir) -> dict:
    import torch

    key = cell_key(sys.key, L, inp, shape)
    hk = w.hooked
    fac, cf, e, du, y_true, y_ctx = build_batches(cfg, sys, L, inp, shape)
    P_fac, P_cf = w.prepare_hooked(fac), w.prepare_hooked(cf)
    mi = w.median_idx
    q_fac, rec_fac = w.forward_with_hooks(
        P_fac, HookSpec(record={"block_in", "cross", "self", "head", "group_w", "time_w", "resid_fut"}))
    q_cf, rec_cf = w.forward_with_hooks(
        P_cf, HookSpec(record={"time_out", "group_out", "mlp_out", "cross", "self", "head", "group_w", "time_w",
                               "resid_fut"}))
    e_hat = (q_cf[:, mi] - q_fac[:, mi]).cpu().numpy()

    # 1. decomposition
    dn = {k: head_diff_norms(hk, P_fac, rec_fac, rec_cf, k) for k in ("cross", "self", "head")}
    mass_fac = cov_attention_mass(P_fac, rec_fac, hk.n_layers)
    # 3. lens
    e_lens = lens_effects(hk, P_fac, P_cf, rec_fac, rec_cf, mi)
    # 4a. REG shares (factual run)
    rs = reg_shares(rec_fac, hk.n_layers)
    # free what patching does not need: cf keeps patch sources, fac keeps the block inputs
    for k in ("self", "group_w", "time_w", "resid_fut"):
        rec_cf.pop(k, None)
    for k in ("cross", "self", "head", "group_w", "time_w", "resid_fut"):
        rec_fac.pop(k, None)
    # 2. patching
    configs = build_configs(hk.n_layers, hk.n_heads, per_head=L in cfg.per_head_contexts)
    e_patch = patch_sweep(w, P_fac, rec_fac, rec_cf, q_fac[:, mi], configs)
    num, den = restoration(e_patch, e_hat)
    g_patch = M.effect_gain(e_patch.reshape(-1, e.shape[-1]), np.repeat(e[None], len(configs), 0).reshape(-1, e.shape[-1]))
    g_patch = g_patch.reshape(len(configs), -1)
    del rec_fac, rec_cf
    # 4b / 5. global interventions
    e_rb, qf_rb = intervention_sweep(w, P_fac, P_cf, reg_bias_specs(hk.n_layers, cfg.reg_bias_values))
    e_cg, qf_cg = intervention_sweep(w, P_fac, P_cf, cross_gain_specs(hk.n_layers, hk.n_heads, cfg.cross_gain_values))
    mase_rb = np.stack([M.mase(q[:, mi], y_true, y_ctx) for q in qf_rb])
    mase_cg = np.stack([M.mase(q[:, mi], y_true, y_ctx) for q in qf_cg])
    if w.device.type == "cuda":
        torch.cuda.empty_cache()

    keep = M.effect_mask(e, y_ctx.std(-1), cfg.eff_thresh)
    run.save_arrays(
        key, e=e.astype(np.float32), du=du.astype(np.float32), e_hat=e_hat.astype(np.float32),
        q_fac=q_fac.cpu().numpy().astype(np.float32), y_true=y_true.astype(np.float32), keep=keep,
        y_scale=y_ctx.std(-1).astype(np.float32),
        patch_names=np.array([c.name for c in configs]), patch_num=num.astype(np.float32),
        patch_den=den.astype(np.float32), patch_gain=g_patch.astype(np.float32),
        dn_cross=dn["cross"], dn_self=dn["self"], dn_head=dn["head"], cov_mass=mass_fac,
        e_lens=e_lens.astype(np.float32), reg_shares=rs.astype(np.float32),
        reg_bias_values=np.array(cfg.reg_bias_values), e_regbias=e_rb.astype(np.float32), mase_regbias=mase_rb,
        cross_gain_values=np.array(cfg.cross_gain_values), e_crossgain=e_cg.astype(np.float32),
        mase_crossgain=mase_cg,
    )
    g = M.effect_gain(e_hat[keep], e[keep])
    return {"key": key, "system": sys.key, "family": sys.family, "L": L, "input": inp, "shape": shape,
            "n": int(len(e)), "n_kept": int(keep.sum()), "g": M.summarize(g), "n_patch_configs": len(configs)}


# -------------------------------------------------------------------------------------- summary
def _pooled_restore(cells: list[dict], names: np.ndarray) -> np.ndarray:
    num = sum(c["patch_num"][:, c["keep"]].sum(1) for c in cells)
    den = sum(c["patch_den"][c["keep"]].sum() for c in cells)
    return num / max(den, 1e-12)


def write_summary(run: RunDir, cfg: Exp2Config) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    recs = run.read_metrics()
    if not recs:
        return
    cells = []
    for r in recs:
        z = dict(np.load(run.cell_path(r["key"]), allow_pickle=False))
        z.update({k: r[k] for k in ("system", "family", "L", "input", "shape")})
        cells.append(z)
    rep = results_root() / "reports"
    figdir = rep / "figs"
    figdir.mkdir(parents=True, exist_ok=True)
    nL = 12
    lines = [f"# Exp2 summary ({cfg.run_name})", "",
             f"Source: `{run.metrics_file}` ({len(cells)} cells, split={cfg.split}). Config: `{run.path / 'config.json'}`.",
             f"Pairs kept if RMS(e) > {cfg.eff_thresh} * std(y_ctx). Restoration ratio = sum<e_patch, e_cf> / sum||e_cf||^2 "
             "(pooled over kept pairs); e_* use the median quantile.", ""]

    groups = {"all": cells}
    for L in sorted({c["L"] for c in cells}):
        groups[f"L={L}"] = [c for c in cells if c["L"] == L]
    for inp in sorted({c["input"] for c in cells}):
        groups[f"input={inp}"] = [c for c in cells if c["input"] == inp]
    # input x L: the two input regimes behave very differently, so pooled medians mix them
    sub = {}
    for inp in sorted({c["input"] for c in cells}):
        for L in sorted({c["L"] for c in cells}):
            cs = [c for c in cells if c["input"] == inp and c["L"] == L]
            if cs:
                sub[f"{inp},L={L}"] = cs
    groups.update(sub)

    # ---- baseline gains
    lines += ["## Model effect gain g (median [IQR]) per cell group", "", "| group | n | g |", "|---|---|---|"]
    for gname, cs in groups.items():
        g = np.concatenate([M.effect_gain(c["e_hat"][c["keep"]], c["e"][c["keep"]]) for c in cs])
        s = M.summarize(g)
        lines.append(f"| {gname} | {s['n']} | {s['median']:.3f} [{s['q25']:.3f}, {s['q75']:.3f}] |")
    lines.append("")

    # ---- layer-level restoration heatmaps
    lines += ["## Activation patching (layer level)", ""]
    for L in sorted({c["L"] for c in cells}):
        cs = [c for c in cells if c["L"] == L]
        names = cs[0]["patch_names"]
        R = _pooled_restore(cs, names)
        idx = {n: i for i, n in enumerate(names)}
        comps = ["time_out", "group_out", "mlp_out", "cross", "head"]
        mat = np.full((len(comps) * 2, nL), np.nan)
        rows = []
        for ci, comp in enumerate(comps):
            for pi, pos in enumerate(("future", "context")):
                rows.append(f"{comp}@{pos}")
                for l in range(nL):
                    n = f"{comp}|L{l}|{pos}|all"
                    if n in idx:
                        mat[ci * 2 + pi, l] = R[idx[n]]
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=max(1.0, np.nanmax(mat)))
        ax.set_yticks(range(len(rows)), rows, fontsize=7)
        ax.set_xticks(range(nL), range(nL), fontsize=7)
        ax.set_xlabel("layer")
        ax.set_title(f"restoration ratio (cf->fac), L={L}", fontsize=9)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        f = figdir / f"exp2_{cfg.run_name}_restore_layers_L{L}"
        fig.savefig(f.with_suffix(".pdf"))
        fig.savefig(f.with_suffix(".png"), dpi=150)
        plt.close(fig)
        order = np.argsort(-np.nan_to_num(mat.ravel(), nan=-1))[: cfg.top_k]
        lines += [f"### L={L}: heatmap `{f.with_suffix('.pdf')}`", "",
                  f"Top-{cfg.top_k} layer-level components by restoration ratio:", "",
                  "| component@positions | layer | ratio |", "|---|---|---|"]
        for o in order:
            r, l = divmod(int(o), nL)
            lines.append(f"| {rows[r]} | {l} | {mat[r, l]:.3f} |")
        lines += ["", "Full table (rows: component@positions, cols: layer 0..11):", "",
                  "| | " + " | ".join(str(l) for l in range(nL)) + " |", "|---|" + "---|" * nL]
        for r, rn in enumerate(rows):
            lines.append(f"| {rn} | " + " | ".join(f"{v:.2f}" for v in mat[r]) + " |")
        lines.append("")

    # ---- per-head restoration
    for L in cfg.per_head_contexts:
        cs = [c for c in cells if c["L"] == L]
        if not cs:
            continue
        names = cs[0]["patch_names"]
        R = _pooled_restore(cs, names)
        idx = {n: i for i, n in enumerate(names)}
        for comp in ("cross", "head"):
            mat = np.full((nL, 12), np.nan)
            for l in range(nL):
                for h in range(12):
                    n = f"{comp}|L{l}|future|h{h}"
                    if n in idx:
                        mat[l, h] = R[idx[n]]
            if np.isnan(mat).all():
                continue
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
            ax.set_xlabel("head")
            ax.set_ylabel("layer")
            ax.set_title(f"per-head {comp} restoration @future, L={L}", fontsize=9)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            f = figdir / f"exp2_{cfg.run_name}_restore_heads_{comp}_L{L}"
            fig.savefig(f.with_suffix(".pdf"))
            fig.savefig(f.with_suffix(".png"), dpi=150)
            plt.close(fig)
            order = np.argsort(-np.nan_to_num(mat.ravel(), nan=-1))[: cfg.top_k]
            lines += [f"### Per-head `{comp}` restoration at future positions, L={L}: `{f.with_suffix('.pdf')}`", "",
                      "| layer | head | ratio |", "|---|---|---|"]
            lines += [f"| {int(o) // 12} | {int(o) % 12} | {mat.ravel()[o]:.3f} |" for o in order]
            srt = np.sort(np.nan_to_num(mat.ravel(), nan=0.0))[::-1]
            lines += ["", f"Sum over all heads: {np.nansum(mat):.3f}; top-10 sum {srt[:10].sum():.3f}; "
                      f"top-20 sum {srt[:20].sum():.3f}", ""]

    # ---- restoration split by input x L
    lines += ["## Restoration by input x L (top-6 layer-level; per-head totals where available)", ""]
    for gname, cs in sub.items():
        n_cfg = min(len(c["patch_names"]) for c in cs)
        names = cs[0]["patch_names"][:n_cfg]
        R = _pooled_restore([{**c, "patch_num": c["patch_num"][:n_cfg]} for c in cs], names)
        lay = [i for i, n in enumerate(names) if n.endswith("|all")]
        top = sorted(lay, key=lambda i: -R[i])[:6]
        lines.append(f"- {gname}: " + ", ".join(f"{names[i]}={R[i]:.3f}" for i in top))
        for comp in ("cross", "head"):
            ih = [i for i, n in enumerate(names) if n.startswith(comp + "|") and "|h" in n]
            if ih:
                rr = R[ih]
                o = np.argsort(-rr)
                lines.append(f"  - per-head `{comp}`@future: total {rr.sum():.3f}, top-10 {rr[o[:10]].sum():.3f}, "
                             f"top-20 {rr[o[:20]].sum():.3f}; top-5 " +
                             ", ".join(names[ih[j]].split("|")[1] + names[ih[j]].split("|")[3] for j in o[:5]))
    lines.append("")

    # ---- decomposition
    lines += ["## Exact decomposition: ||W_O^h (x_cf - x_fac)|| at target positions (mean over cells)", ""]
    for L in sorted({c["L"] for c in cells}):
        cs = [c for c in cells if c["L"] == L]
        for k in ("cross", "self"):
            D = np.mean([c[f"dn_{k}"] for c in cs], axis=0)  # (L, H, 2)
            fut = D[:, :, 0]
            order = np.argsort(-fut.ravel())[: cfg.top_k]
            lines += [f"L={L}, `{k}` @future - top-{cfg.top_k} (layer, head): "
                      + ", ".join(f"({int(o) // 12},{int(o) % 12}) {fut.ravel()[o]:.3f}" for o in order),
                      f"  layer sums @future: " + ", ".join(f"{v:.2f}" for v in fut.sum(1)),
                      f"  layer sums @context: " + ", ".join(f"{v:.2f}" for v in D[:, :, 1].sum(1)), ""]
        mass = np.mean([c["cov_mass"] for c in cs], axis=0)
        lines += [f"L={L}, covariate attention mass of target queries per layer (mean over heads) "
                  f"@future: " + ", ".join(f"{v:.2f}" for v in mass[:, :, 0].mean(1)),
                  f"  @context: " + ", ".join(f"{v:.2f}" for v in mass[:, :, 1].mean(1)), ""]

    # ---- lens
    lines += ["## Residual-stream readout of the effect (gain per layer, median over kept pairs)", "",
              "Readout = final norm + output head applied after block l. Early layers are not trained for this, "
              "so only the late-layer trend is interpretable.", "", "| group | " +
              " | ".join(f"l{l}" for l in range(nL)) + " |", "|---|" + "---|" * nL]
    for gname, cs in groups.items():
        gl = []
        for l in range(nL):
            gl.append(np.median(np.concatenate([M.effect_gain(c["e_lens"][l][c["keep"]], c["e"][c["keep"]]) for c in cs])))
        lines.append(f"| {gname} | " + " | ".join(f"{v:.2f}" for v in gl) + " |")
    lines.append("")

    # ---- REG sink
    lines += ["## REG sink", ""]
    share_all, g_all, cell_share, cell_g, cell_lbl = [], [], [], [], []
    for c in cells:
        k = c["keep"]
        s = c["reg_shares"][k][..., 0].mean(axis=(1, 2))  # per-sample REG mass, mean over layers & heads
        g = M.effect_gain(c["e_hat"][k], c["e"][k])
        share_all.append(s)
        g_all.append(g)
        cell_share.append(float(np.mean(s)))
        cell_g.append(float(np.median(g)))
        cell_lbl.append(f"{c['system']}|L{c['L']}|{c['input']}|{c['shape']}")
    share_all, g_all = np.concatenate(share_all), np.concatenate(g_all)
    rho, pv = spearmanr(share_all, g_all)
    rho_c, pv_c = spearmanr(cell_share, cell_g)
    per_layer = np.mean([c["reg_shares"][c["keep"]][..., 0].mean(axis=(0, 2)) for c in cells], axis=0)
    fut_l = np.mean([c["reg_shares"][c["keep"]][..., 1].mean(axis=(0, 2)) for c in cells], axis=0)
    ctx_l = np.mean([c["reg_shares"][c["keep"]][..., 2].mean(axis=(0, 2)) for c in cells], axis=0)
    lines += [f"Time-attention mass of target future queries, per layer (mean over heads/cells):",
              "  REG: " + ", ".join(f"{v:.2f}" for v in per_layer),
              "  other future: " + ", ".join(f"{v:.2f}" for v in fut_l),
              "  context: " + ", ".join(f"{v:.2f}" for v in ctx_l), "",
              f"Spearman(REG mass, g): sample level rho={rho:.3f} (p={pv:.2g}, n={len(g_all)}); "
              f"cell level rho={rho_c:.3f} (p={pv_c:.2g}, n={len(cell_g)}).", "",
              "Within input x L (sample level):"]
    for gname, cs in sub.items():
        s = np.concatenate([c["reg_shares"][c["keep"]][..., 0].mean(axis=(1, 2)) for c in cs])
        g = np.concatenate([M.effect_gain(c["e_hat"][c["keep"]], c["e"][c["keep"]]) for c in cs])
        r_, p_ = spearmanr(s, g)
        lines.append(f"  {gname}: rho={r_:.3f} (p={p_:.2g}, n={len(g)}), mean REG mass {s.mean():.3f}")
    lines.append("")
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.0))
    axs[0].scatter(cell_share, cell_g, s=10)
    axs[0].set_xlabel("mean REG mass (target future queries)")
    axs[0].set_ylabel("median g")
    rb = cells[0]["reg_bias_values"]
    lines += ["REG logit bias on target future queries (all layers): median g / median MASE ratio vs b=0", "",
              "| group | " + " | ".join(f"b={b:g}" for b in rb) + " |", "|---|" + "---|" * len(rb)]
    for gname, cs in groups.items():
        gs, ms = [], []
        for bi in range(len(rb)):
            gs.append(np.median(np.concatenate([M.effect_gain(c["e_regbias"][bi][c["keep"]], c["e"][c["keep"]]) for c in cs])))
            ms.append(np.median(np.concatenate([c["mase_regbias"][bi] / c["mase_regbias"][0] for c in cs])))
        lines.append(f"| {gname} | " + " | ".join(f"{a:.3f} / {m:.3f}" for a, m in zip(gs, ms)) + " |")
        if gname == "all":
            axs[1].plot(rb, gs, marker="o")
    axs[1].set_xlabel("REG logit bias b")
    axs[1].set_ylabel("median g")
    fig.tight_layout()
    f = figdir / f"exp2_{cfg.run_name}_reg"
    fig.savefig(f.with_suffix(".pdf"))
    fig.savefig(f.with_suffix(".png"), dpi=150)
    plt.close(fig)
    lines += ["", f"Figure: `{f.with_suffix('.pdf')}`", ""]

    # ---- global cross gain
    cgv = cells[0]["cross_gain_values"]
    lines += ["## Global cross-gain sweep (all layers & heads, target queries): median g / median shape r / "
              "median MASE ratio vs gamma=1", "",
              "| group | " + " | ".join(f"γ={v:g}" for v in cgv) + " |", "|---|" + "---|" * len(cgv)]
    i1 = int(np.argmin(np.abs(cgv - 1.0)))
    for gname, cs in groups.items():
        cellsv = []
        for gi in range(len(cgv)):
            gg = np.median(np.concatenate([M.effect_gain(c["e_crossgain"][gi][c["keep"]], c["e"][c["keep"]]) for c in cs]))
            rr = np.nanmedian(np.concatenate([M.shape_corr(c["e_crossgain"][gi][c["keep"]], c["e"][c["keep"]]) for c in cs]))
            mm = np.median(np.concatenate([c["mase_crossgain"][gi] / c["mase_crossgain"][i1] for c in cs]))
            cellsv.append(f"{gg:.3f} / {rr:.3f} / {mm:.3f}")
        lines.append(f"| {gname} | " + " | ".join(cellsv) + " |")
    lines.append("")
    lines += ["Per cell: gamma maximising median g, and whether that max reaches 0.9:", ""]
    for inp in sorted({c["input"] for c in cells}):
        best = []
        for c in (c for c in cells if c["input"] == inp):
            gs = [np.median(M.effect_gain(c["e_crossgain"][gi][c["keep"]], c["e"][c["keep"]])) for gi in range(len(cgv))]
            best.append((max(gs), float(cgv[int(np.argmax(gs))])))
        cnt = {f"{v:g}": int(sum(b[1] == v for b in best)) for v in cgv}
        lines.append(f"- input={inp}: {len(best)} cells, max g >= 0.9 in {sum(b[0] >= 0.9 for b in best)}; "
                     f"argmax gamma counts {cnt}")
    lines.append("")
    out = rep / f"exp2_summary_{cfg.run_name}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("summary written: %s", out)


# -------------------------------------------------------------------------------------- main
def main() -> None:
    cfg, args = parse_args()
    run = RunDir("exp2", cfg.run_name)
    setup_logging(run.path / "log.txt")
    log.info("config: %s", json.dumps(asdict(cfg)))
    if args.summary_only:
        write_summary(run, cfg)
        return
    run.write_config(cfg, resume=args.resume, ignore_keys=("contexts", "device"))

    from covattn.model import Chronos2Wrapper

    w = Chronos2Wrapper(cfg.model_id, device=cfg.device, amp=cfg.amp, batch_size=2 * cfg.n_per_cell)
    lookup = {s.key: s for s in default_systems() + quick_systems()}
    cells = [(lookup[s], L, inp, sh) for L in cfg.contexts for s in cfg.systems for inp in cfg.input_types
             for sh in cfg.shapes]
    done = run.done_keys() if args.resume else set()
    todo = [c for c in cells if cell_key(c[0].key, c[1], c[2], c[3]) not in done]
    log.info("cells: %d total, %d done, %d to run", len(cells), len(cells) - len(todo), len(todo))

    # time estimate: forwards per cell x measured forward time at each L
    est = 0.0
    for L in sorted({c[1] for c in todo}):
        tps = w.measure_throughput(L, cfg.horizon, k=1, n_tasks=cfg.n_per_cell, reps=2)
        n_cfg = len(build_configs(12, 12, per_head=L in cfg.per_head_contexts))
        fwd = 2 + 0.55 * n_cfg + 2 * (len(cfg.reg_bias_values) + len(cfg.cross_gain_values))  # restart saves ~45%
        n_cells = sum(1 for c in todo if c[1] == L)
        est += n_cells * fwd * cfg.n_per_cell / tps * 1.3  # +30% for eager attention / recording
        log.info("L=%d: %.0f tasks/s, %d patch configs, %d cells", L, tps, n_cfg, n_cells)
    te = time_estimates(est, w.device)
    log.info("ESTIMATED TIME - measured on %s: %s | RTX4060: %s | L4: %s | A100(guess): %s", w.device,
             fmt_seconds(te["here"]), fmt_seconds(te["RTX4060"]), fmt_seconds(te["L4"]), fmt_seconds(te["A100"]))

    t0 = time.time()
    for i, (sys, L, inp, sh) in enumerate(todo):
        tc = time.time()
        rec = run_cell(w, cfg, sys, L, inp, sh, run)
        rec["seconds"] = time.time() - tc
        rec["device"] = str(w.device)
        run.append_metrics(rec)
        eta = (time.time() - t0) / (i + 1) * (len(todo) - i - 1)
        log.info("[%d/%d] %s g=%.3f %.1fs ETA %s", i + 1, len(todo), rec["key"], rec["g"]["median"], rec["seconds"],
                 fmt_seconds(eta))
    write_summary(run, cfg)
    canon = {cell_key(s, L, inp, sh) for L in cfg.contexts for s in cfg.systems for inp in cfg.input_types
             for sh in cfg.shapes}
    if not canon - run.done_keys():
        run.mark_done({"n_cells": len(canon), "time": time.ctime()})
        log.info("all %d cells done -> done.marker", len(canon))


if __name__ == "__main__":
    main()
