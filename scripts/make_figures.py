"""Figures 1-7 (PDF + PNG, LNCS text width 12.2 cm) from the saved result files only.

  fig1_gain_by_family : Exp1 - effect gain g by system family and context length (box plots), ARX as reference;
                        non-periodic inputs and multisine in separate panels (two regimes)
  fig2_examples       : Exp3 - true vs predicted effect (zero-shot, R1, R2) for pre-declared cells; in each cell
                        the pair shown is the one with the median zero-shot g (rule-based, not hand-picked)
  fig3_restoration    : Exp2 - per-head group-attention `cross` restoration (future positions; the R1 heads are
                        outlined) and layer-level restoration of time-attn / group-attn / MLP outputs, L=512
  fig4_reg_sink       : Exp2 - REG attention mass vs g per cell, and the REG logit-bias intervention
  fig5_ablations      : Exp3 R1 selection grid (val) + Exp5 head-set ablation (if Exp5 exists)
  fig6_other_models   : Exp8 - g of Chronos-2, TimesFM 3.0, TimesFM 2.5 + XReg and ARX by input type and family
  fig7_real_world     : Exp6 - fev-bench per-task WQL ratios with the geometric mean [95% CI] of Table 5, and the
                        semi-synthetic g per method

Style rules (LNCS): width 4.8 in = text width, every text element >= 6 pt, vector PDF with embedded TrueType
fonts, colour-blind-safe categorical colours plus marker shape where colour alone would carry meaning.

  python scripts/make_figures.py --exp3-run v2 --exp5-run main
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from covattn import metrics as M  # noqa: E402
from covattn.io_utils import results_root, setup_logging  # noqa: E402

log = logging.getLogger("figures")
LNCS_W = 4.8  # inches; LNCS text width 12.2 cm
MIN_PT = 6.0  # LNCS: lettering not smaller than 6 pt
# reference palette (categorical slots 1-3), neutral reference ink, single-hue sequential ramp
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "ink": "#0b0b0b", "ink2": "#52514e",
     "ref": "#9a9893", "ref2": "#c9c7c1", "grid": "#e4e3df"}
SEQ = ["#a9c7ef", "#6fa3e3", "#2a78d6", "#1b4f8f"]  # light -> dark blue (ordered L)
METHOD_COLOR = {"zeroshot": C["orange"], "r1": C["blue"], "r2": C["aqua"], "dither": C["ink2"],
                "ft_full": C["ref"], "ft_lora": C["ref2"], "arx": C["ref"], "nocov": C["ref2"],
                "r1_val": C["blue"], "r2_val": C["aqua"]}
METHOD_LABEL = {"zeroshot": "zero-shot", "r1": "R1", "r2": "R2", "dither": "dither", "ft_full": "full FT",
                "ft_lora": "LoRA", "arx": "ARX", "nocov": "no covariates", "r1_val": "R1-val", "r2_val": "R2-val"}
FAM_LABEL = {"F1": "F1\nstatic", "F2": "F2\nlag", "F3": "F3\ndelay", "F4": "F4\nresonant", "F5": "F5\ntanh"}
MEDIAN_IDX = 10


def style() -> None:
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": C["ink2"],
        "axes.linewidth": 0.6, "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.labelcolor": C["ink"],
        "text.color": C["ink"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.linewidth": 1.2, "legend.frameon": False, "pdf.fonttype": 42,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def _check_fonts(fig) -> None:
    small = sorted({round(t.get_fontsize(), 2) for t in fig.findobj(matplotlib.text.Text)
                    if t.get_text().strip() and t.get_visible() and t.get_fontsize() < MIN_PT})
    if small:
        log.warning("text below %.0f pt: %s", MIN_PT, small)


def save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    _check_fonts(fig)
    p = out / f"{name}.pdf"
    fig.savefig(p)
    fig.savefig(out / f"{name}.png", dpi=220)
    plt.close(fig)
    log.info("wrote %s", p)
    return p


def _npz(d: Path, key: str) -> dict:
    safe = "".join(c if c.isalnum() or c in "-_.=" else "_" for c in key)
    return dict(np.load(d / "cells" / f"{safe}.npz", allow_pickle=False))


def _metrics(d: Path) -> list[dict]:
    return [json.loads(x) for x in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def _clip_marks(ax, bp, col) -> None:
    """Mark whiskers that leave the y-range with a small triangle at the axis edge."""
    lo, hi = ax.get_ylim()
    for wl in bp["whiskers"]:
        x, y = wl.get_xdata()[0], wl.get_ydata()
        if np.max(y) > hi:  # drawn just outside the axes so it never hides behind a box of the same colour
            ax.plot([x], [hi + 0.03 * (hi - lo)], marker="^", markersize=2.6, color=col, clip_on=False, zorder=4)
        if np.min(y) < lo:
            ax.plot([x], [lo - 0.03 * (hi - lo)], marker="v", markersize=2.6, color=col, clip_on=False, zorder=4)


# ------------------------------------------------------------------------------------ figure 1
def fig1(exp1: Path, out: Path) -> None:
    cfg = json.loads((exp1 / "config.json").read_text())["config"]
    Ls, thr = cfg["contexts"], cfg["eff_thresh"]
    fams = ["F1", "F2", "F3", "F4", "F5"]
    data = {}  # (panel, fam, L or 'arx') -> list of g
    for r in _metrics(exp1):
        z = _npz(exp1, r["key"])
        sel = np.isclose(z["combo_delta"], 1.0)  # delta = 1 (g is flat in delta, see Exp1 summary)
        panel = "multisine" if r["input"] == "multisine" else "other"
        for ci in np.where(sel)[0]:
            e = z["e"][:, ci]
            keep = M.effect_mask(e, z["y_scale"], thr)
            data.setdefault((panel, r["family"], r["L"]), []).append(M.effect_gain(z["e_hat"][keep, ci], e[keep]))
            ok = keep & z["arx_stable"].astype(bool)
            data.setdefault((panel, r["family"], "arx"), []).append(M.effect_gain(z["e_arx"][ok, ci], e[ok]))
    fig, axs = plt.subplots(1, 2, figsize=(LNCS_W, 2.25), sharey=True)
    nb = len(Ls) + 1
    wdt = 0.82 / nb
    ylim = (-0.6, 1.6)
    for ax, panel, title in zip(axs, ("other", "multisine"), ("(a) PRBS / AR(1) / random-steps inputs",
                                                             "(b) multisine input")):
        ax.set_ylim(*ylim)
        for fi, fam in enumerate(fams):
            for j, L in enumerate(list(Ls) + ["arx"]):
                g = data.get((panel, fam, L))
                if not g:
                    continue
                g = np.concatenate(g)
                x = fi - 0.41 + wdt * (j + 0.5)
                col = C["ref"] if L == "arx" else SEQ[j % len(SEQ)]
                med = "white" if (L == "arx" or j >= 2) else C["ink"]
                bp = ax.boxplot([g], positions=[x], widths=wdt * 0.82, showfliers=False, patch_artist=True,
                                medianprops={"color": med, "linewidth": 1.0},
                                boxprops={"facecolor": col, "edgecolor": col, "linewidth": 0.6},
                                whiskerprops={"color": col, "linewidth": 0.6}, capprops={"color": col, "linewidth": 0.6})
                _clip_marks(ax, bp, col)
        ax.axhline(1.0, color=C["ink2"], linewidth=0.6, linestyle="--", zorder=0)
        ax.set_xticks(range(len(fams)), [FAM_LABEL[f] for f in fams])
        ax.set_xlim(-0.55, len(fams) - 0.45)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axs[0].set_yticks([-0.5, 0, 0.5, 1.0, 1.5])
    axs[0].set_ylabel("effect gain $g$  (1 = exact)")
    handles = [plt.Line2D([], [], linestyle="none")]
    handles += [plt.Rectangle((0, 0), 1, 1, color=SEQ[j % len(SEQ)]) for j in range(len(Ls))]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=C["ref"]))
    fig.legend(handles, ["Chronos-2:"] + [f"$L$={L}" for L in Ls] + ["ARX (context fit)"], ncol=len(Ls) + 2,
               loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig1_gain_by_family")


# ------------------------------------------------------------------------------------ figure 2
FIG2_CELLS = [("F2_a0.8", "F2 lag, $a$=0.8 (ID)"), ("F2_a0.99", "F2 lag, $a$=0.99 (OOD-param)"),
              ("F4_P24_z0.2", "F4, $P$=24, $\\zeta$=0.2 (OOD-family)")]


def fig2(exp3: Path, out: Path, L: int = 512, inp: str = "prbs") -> None:
    shapes = ("step", "pulse")
    fig, axs = plt.subplots(len(shapes), len(FIG2_CELLS), figsize=(LNCS_W, 2.75), sharex=True)
    for ci, (sys_key, title) in enumerate(FIG2_CELLS):
        z = _npz(exp3, f"{sys_key}|L{L}|{inp}")
        for si, sh in enumerate(shapes):
            ax = axs[si, ci]
            k = np.where(z[f"keep_{sh}"])[0]
            g0 = M.effect_gain(z[f"ehat_zeroshot_{sh}"][k], z[f"e_{sh}"][k])
            i = k[np.argsort(g0)[len(g0) // 2]]  # pair with the median zero-shot g
            e = z[f"e_{sh}"][i]
            h = np.arange(1, len(e) + 1)
            ax.plot(h, e, color=C["ink"], linewidth=1.7, zorder=5)
            hs = []
            for m in ("zeroshot", "r1", "r2"):
                eh = z[f"ehat_{m}_{sh}"][i]
                gi = float(M.effect_gain(eh[None], e[None])[0])
                (ln,) = ax.plot(h, eh, color=METHOD_COLOR[m], linewidth=0.95, label=f"$g$={gi:.2f}")
                hs.append(ln)
            ax.legend(handles=hs, loc="best", fontsize=MIN_PT, handlelength=0.8, borderaxespad=0.15,
                      labelspacing=0.1, handletextpad=0.3, frameon=True, framealpha=0.9, edgecolor="none",
                      borderpad=0.2)
            if si == 0:
                ax.set_title(title, fontsize=6.5)
            if ci == 0:
                ax.set_ylabel(f"{sh} effect")
            if si == len(shapes) - 1:
                ax.set_xlabel("horizon step")
    h_ = [plt.Line2D([], [], color=C["ink"], linewidth=1.7)] + [plt.Line2D([], [], color=METHOD_COLOR[m])
                                                                   for m in ("zeroshot", "r1", "r2")]
    fig.legend(h_, ["true effect", "Chronos-2 zero-shot", "R1", "R2"], ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(h_pad=0.4, w_pad=0.4)
    save(fig, out, "fig2_examples")


# ------------------------------------------------------------------------------------ figure 3
def _restore_by_name(cells: list[dict]) -> dict[str, float]:
    num, den = {}, {}
    for c in cells:
        k = c["keep"]
        d = float(c["patch_den"][k].sum())
        for i, n in enumerate(c["patch_names"]):
            n = str(n)
            num[n] = num.get(n, 0.0) + float(c["patch_num"][i][k].sum())
            den[n] = den.get(n, 0.0) + d
    return {n: num[n] / max(den[n], 1e-12) for n in num}


def fig3(exp2: Path, exp3: Path, out: Path, L: int = 512) -> None:
    cells = []
    for r in _metrics(exp2):
        if r["L"] == L:
            z = _npz(exp2, r["key"])
            z["input"] = r["input"]
            cells.append(z)
    R = _restore_by_name(cells)
    nL, nH = 12, 12
    head = np.full((nL, nH), np.nan)
    for l in range(nL):
        for h in range(nH):
            head[l, h] = R.get(f"cross|L{l}|future|h{h}", np.nan)
    rows = [("time_out", "future"), ("time_out", "context"), ("group_out", "future"), ("group_out", "context"),
            ("mlp_out", "future"), ("mlp_out", "context")]
    rname = {"time_out": "time attn", "group_out": "group attn", "mlp_out": "MLP"}
    lay = np.array([[R.get(f"{c}|L{l}|{p}|all", np.nan) for l in range(nL)] for c, p in rows])
    fig, axs = plt.subplots(1, 2, figsize=(LNCS_W, 2.35), gridspec_kw={"width_ratios": [1, 1.2]})
    im = axs[0].imshow(head, cmap="Blues", vmin=0, aspect="auto", origin="lower")
    r1f = exp3 / "r1_select.json"
    n_out = 0
    if r1f.exists():  # outline the heads R1 amplifies (top-k of its ranking)
        r1 = json.loads(r1f.read_text())
        for l, h in r1["ranking"][: r1["best"]["k"]]:
            axs[0].add_patch(plt.Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False, edgecolor=C["orange"], linewidth=0.8))
            n_out += 1
    axs[0].set_xlabel("head")
    axs[0].set_ylabel("layer")
    axs[0].set_title("(a) group-attn cross term per head" + (f"\n(outlined: the {n_out} R1 heads)" if n_out else ""),
                     loc="left")
    axs[0].set_xticks(range(0, nH, 2))
    axs[0].set_yticks(range(0, nL, 2))
    axs[0].grid(False)
    cb = fig.colorbar(im, ax=axs[0], fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=MIN_PT)
    cb.outline.set_linewidth(0.4)
    im = axs[1].imshow(lay, cmap="Blues", vmin=0, aspect="auto")
    axs[1].set_yticks(range(len(rows)), [f"{rname[c]}, {p}" for c, p in rows])
    axs[1].set_xticks(range(0, nL, 2))
    axs[1].set_xlabel("layer")
    axs[1].set_title("(b) whole sub-layer outputs\n(at target future / context positions)", loc="left")
    axs[1].grid(False)
    cb = fig.colorbar(im, ax=axs[1], fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=MIN_PT)
    cb.outline.set_linewidth(0.4)
    cb.set_label("restored share of the effect", fontsize=MIN_PT + 0.5)
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig3_restoration")
    (out / "fig3_restoration_note.txt").write_text(
        f"Restoration ratio = sum<e_patch, e_cf> / sum||e_cf||^2 pooled over the {len(cells)} Exp2 cells at L={L} "
        f"({exp2}). Panel (a) shows target future positions; outlined = the {n_out} heads R1 amplifies ({r1f}). The "
        "R1 ranking pools future AND context per-head restoration (rank = larger of the two), so a few outlined "
        "heads are faint in this future-only map. Time attention was patched per layer only (no per-head time-attn "
        "patches exist), so panel (b) is layer-level.\n", encoding="utf-8")


# ------------------------------------------------------------------------------------ figure 4
def fig4(exp2: Path, out: Path) -> None:
    from scipy.stats import spearmanr

    recs = _metrics(exp2)
    pts = []
    for r in recs:
        z = _npz(exp2, r["key"])
        k = z["keep"]
        g = M.effect_gain(z["e_hat"][k], z["e"][k])
        pts.append((r["input"], r["L"], float(z["reg_shares"][k][..., 0].mean()), float(np.median(g)), z))
    fig, axs = plt.subplots(1, 3, figsize=(LNCS_W, 1.8))
    inputs = sorted({p[0] for p in pts}, key=lambda s: (s != "prbs", s))
    col = {inp: [C["blue"], C["orange"], C["aqua"]][i] for i, inp in enumerate(inputs)}
    Ls = sorted({p[1] for p in pts})
    mk = {L: m for L, m in zip(Ls, ("o", "s", "^", "D"))}
    for inp in inputs:
        for L in Ls:
            q = [p for p in pts if p[0] == inp and p[1] == L]
            axs[0].scatter([p[2] for p in q], [p[3] for p in q], s=12, marker=mk[L], color=col[inp],
                           edgecolor="white", linewidth=0.5, label=f"{inp}, $L$={L}")
    rho, pv = spearmanr([p[2] for p in pts], [p[3] for p in pts])
    axs[0].set_title(f"(a) cells: $\\rho_s$={rho:.2f}", loc="left")
    axs[0].set_xlabel("REG attention mass")
    axs[0].set_ylabel("median $g$")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.2)
    rb = pts[0][4]["reg_bias_values"]
    for inp in inputs:
        cs = [p[4] for p in pts if p[0] == inp]
        gs = [np.median(np.concatenate([M.effect_gain(c["e_regbias"][bi][c["keep"]], c["e"][c["keep"]]) for c in cs]))
              for bi in range(len(rb))]
        ms = [np.median(np.concatenate([c["mase_regbias"][bi] / c["mase_regbias"][0] for c in cs]))
              for bi in range(len(rb))]
        axs[1].plot(rb, gs, marker="o", markersize=3, color=col[inp], label=inp)
        axs[2].plot(rb, ms, marker="o", markersize=3, color=col[inp], label=inp)
    axs[1].set_title("(b) REG bias: $g$", loc="left")
    axs[1].set_xlabel("REG logit bias $b$")
    axs[1].set_ylabel("median $g$")
    axs[2].axhline(1.0, color=C["ink2"], linewidth=0.6, linestyle="--", zorder=0)
    axs[2].set_title("(c) REG bias: MASE", loc="left")
    axs[2].set_xlabel("REG logit bias $b$")
    axs[2].set_ylabel("MASE / MASE($b$=0)")
    fig.tight_layout(w_pad=0.5)
    save(fig, out, "fig4_reg_sink")
    log.info("fig4: cell-level Spearman rho=%.3f p=%.3g (n=%d)", rho, pv, len(pts))


# ------------------------------------------------------------------------------------ figure 5
def fig5(exp3: Path, exp5: Path | None, out: Path) -> None:
    r1 = json.loads((exp3 / "r1_select.json").read_text())
    grid = [d for d in r1["grid"] if d["k"] > 0]
    ks, gms = sorted({d["k"] for d in grid}), sorted({d["gamma"] for d in grid})
    look = {(d["k"], d["gamma"]): d for d in grid}
    have5 = exp5 is not None and (exp5 / "metrics.jsonl").exists()
    fig, axs = plt.subplots(1, 2 if have5 else 1, figsize=(LNCS_W, 2.05) if have5 else (LNCS_W * 0.55, 2.05),
                            squeeze=False, gridspec_kw={"width_ratios": [1, 1.15]} if have5 else None)
    ax = axs[0, 0]
    zs = r1["zero_shot"]["obj"]
    for i, k in enumerate(ks):
        ax.plot(gms, [look[(k, g)]["obj"] if (k, g) in look else np.nan for g in gms], marker="o", markersize=3,
                color=(SEQ + [C["ink"]])[i % (len(SEQ) + 1)], label=f"$k$={k}")
    ax.axhline(zs, color=C["ref"], linestyle="--", linewidth=0.8, label="zero-shot")
    b = r1["best"]
    ax.scatter([b["gamma"]], [b["obj"]], s=42, facecolor="none", edgecolor=C["orange"], linewidth=1.0, zorder=5,
               label="chosen")
    ax.set_xlabel("gain $\\gamma$")
    ax.set_ylabel("validation objective (lower = better)")
    ax.set_title("(a) R1: top-$k$ heads $\\times$ gain $\\gamma$", loc="left")
    ax.legend(fontsize=MIN_PT, ncol=2, handlelength=1.2, columnspacing=0.8, labelspacing=0.25)
    if have5:
        recs = _metrics(exp5)
        mi = json.loads((exp5 / "methods.json").read_text()).get("heads", {})
        methods = mi.get("methods", [])
        cells = [_npz(exp5, r["key"]) for r in recs if r["key"].startswith("heads|")]
        inp_of = [r["input"] for r in recs if r["key"].startswith("heads|")]
        ax = axs[0, 1]
        names = list(methods)
        y = np.arange(len(names))
        for j, inp in enumerate(("prbs", "multisine")):
            gs = []
            for m in names:
                g = [M.effect_gain(c[f"ehat_{m}_{sh}"][c[f"keep_{sh}"]], c[f"e_{sh}"][c[f"keep_{sh}"]])
                     for c, i in zip(cells, inp_of) if i == inp for sh in ("step", "pulse")]
                gs.append(float(np.median(np.concatenate(g))) if g else np.nan)
            ax.barh(y + (j - 0.5) * 0.38, gs, height=0.36, color=[C["blue"], C["orange"]][j], label=f"{inp} input")
        lab = {"zeroshot": "zero-shot", "r1": "R1", "r2": "R2", "r1_trainrank": "R1, train-only rank"}
        ax.set_yticks(y, [lab.get(m, m.replace("r1_layers", "top-$k$ in L").replace("random", "random #"))
                          for m in names])
        ax.invert_yaxis()
        ax.axvline(1.0, color=C["ink2"], linestyle="--", linewidth=0.6)
        ax.set_xlabel("median $g$ (test, $L$=512)")
        ax.set_title("(b) which heads get the gain", loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.3), ncol=2, handlelength=1.0, borderaxespad=0)
        ax.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig5_ablations")


# ------------------------------------------------------------------------------------ figure 6
OTHER = [("chronos2", "Chronos-2", C["blue"], "o"), ("timesfm3", "TimesFM 3.0", C["aqua"], "s"),
         ("timesfm25_xreg", "TimesFM 2.5 + XReg", C["orange"], "D"), ("arx", "ARX (context fit)", C["ref"], "^")]


def fig6(exp8: Path, exp3: Path, out: Path) -> None:
    shapes = ("step", "pulse")
    g = {}  # (model, kind, value) -> list of g arrays
    recs = _metrics(exp8)
    seen = set()
    for r in recs:
        z = _npz(exp8, r["key"])
        sets = [(r["model"], z)]
        ck = (r["system"], r["L"], r["input"])
        if ck not in seen:  # ARX on the same samples, from the Exp3 cell
            seen.add(ck)
            z3 = _npz(exp3, f"{r['system']}|L{r['L']}|{r['input']}")
            sets.append(("arx", {**{f"ehat_{s}": z3[f"ehat_arx_{s}"] for s in shapes},
                                 **{f"e_{s}": z3[f"e_{s}"] for s in shapes}, **{f"keep_{s}": z3[f"keep_{s}"] for s in shapes}}))
        for m, zz in sets:
            gg = np.concatenate([M.effect_gain(zz[f"ehat_{s}"][zz[f"keep_{s}"]], zz[f"e_{s}"][zz[f"keep_{s}"]])
                                 for s in shapes])
            for kind, val in (("input", r["input"]), ("family", r["family"]), ("all", "all")):
                g.setdefault((m, kind, val), []).append(gg)
    panels = [("input", ["all", "prbs", "multisine", "ar1", "steps"],
               {"all": "all pairs", "prbs": "PRBS", "multisine": "multisine", "ar1": "AR(1)", "steps": "random steps"},
               "(a) by covariate input"),
              ("family", ["F1", "F2", "F3", "F4", "F5"], {f: FAM_LABEL[f].replace("\n", " ") for f in FAM_LABEL},
               "(b) by system family")]
    fig, axs = plt.subplots(1, 2, figsize=(LNCS_W, 2.2), sharex=True)
    dy = 0.18
    for ax, (kind, vals, lab, title) in zip(axs, panels):
        for vi, v in enumerate(vals):
            for mi, (m, _, col, mk) in enumerate(OTHER):
                key = (m, "all", "all") if v == "all" else (m, kind, v)
                if key not in g:
                    continue
                x = np.concatenate(g[key])
                q1, med, q3 = np.percentile(x, [25, 50, 75])
                yy = vi + (mi - 1.5) * dy
                ax.plot([q1, q3], [yy, yy], color=col, linewidth=1.0, solid_capstyle="butt")
                ax.plot([med], [yy], marker=mk, markersize=3.2, color=col, markeredgecolor="white",
                        markeredgewidth=0.4, linestyle="none")
        ax.axvline(1.0, color=C["ink2"], linewidth=0.6, linestyle="--", zorder=0)
        if kind == "input":
            ax.axhline(0.5, color=C["grid"], linewidth=0.8)
        ax.set_yticks(range(len(vals)), [lab[v] for v in vals])
        ax.set_ylim(len(vals) - 0.5, -0.5)
        ax.set_title(title, loc="left")
        ax.set_xlabel("effect gain $g$ (median, IQR)")
        ax.grid(axis="y", visible=False)
    axs[0].set_xlim(-0.05, 1.55)
    hs = [plt.Line2D([], [], color=col, marker=mk, markersize=3.5, linewidth=1.0) for _, _, col, mk in OTHER]
    fig.legend(hs, [lab for _, lab, _, _ in OTHER], ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               handlelength=1.4)
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig6_other_models")


# ------------------------------------------------------------------------------------ figure 7
def fig7(exp6: Path, rep: Path, out: Path, run: str) -> None:
    aggf = rep / f"exp6_fev_aggregate_{run}.json"
    agg = json.loads(aggf.read_text())
    fig, axs = plt.subplots(1, 2, figsize=(LNCS_W, 2.25), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    order = [m for m in ("nocov", "r1", "r2", "r1_val", "r2_val", "ft_full", "ft_lora") if m in agg["methods"]]
    rng = np.random.default_rng(0)  # jitter only
    for i, m in enumerate(order):
        a = agg["methods"][m]
        v = np.asarray(a["wql_ratio"])
        ax.scatter(v, i + rng.uniform(-0.17, 0.17, len(v)), s=4, color=METHOD_COLOR[m], alpha=0.45, linewidth=0,
                   zorder=2)
        lo, hi = a["wql_ci"]
        ax.plot([lo, hi], [i, i], color=C["ink"], linewidth=1.1, zorder=3)
        ax.plot([a["wql_gmean"]], [i], marker="D", markersize=3.6, color=C["ink"], markeredgecolor="white",
                markeredgewidth=0.4, zorder=4)
    ax.axvline(1.0, color=C["ink2"], linewidth=0.6, linestyle="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2], ["0.5", "0.75", "1", "1.5", "2"])
    ax.minorticks_off()
    lims = np.concatenate([agg["methods"][m]["wql_ratio"] for m in order])
    ax.set_xlim(max(0.4, lims.min() * 0.9), min(3.0, lims.max() * 1.1))
    ax.set_yticks(range(len(order)), [METHOD_LABEL[m] for m in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("test WQL / zero-shot WQL (log)")
    ax.set_title(f"(a) fev-bench, {len(agg['tasks'])} covariate tasks", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend([plt.Line2D([], [], marker="o", color=C["ref"], linestyle="none", markersize=2.5),
               plt.Line2D([], [], marker="D", color=C["ink"], markersize=3.2, linewidth=1.0)],
              ["one task", "geo. mean, 95% CI"], loc="center right", fontsize=MIN_PT, handlelength=1.2,
              frameon=True, framealpha=0.9, edgecolor="none", borderpad=0.25)
    # (b) semi-synthetic g per method
    ax = axs[1]
    recs = [r for r in _metrics(exp6) if r["part"] == "semi"]
    ms = ["zeroshot", "r1", "r2", "dither", "ft_full", "ft_lora", "arx"]
    gs = {m: [] for m in ms}
    for r in recs:
        z = _npz(exp6, r["key"])
        for m in ms:
            if f"ehat_{m}_step" not in z:
                continue
            for sh in ("step", "pulse"):
                k = z[f"keep_{sh}"]
                gs[m].append(M.effect_gain(z[f"ehat_{m}_{sh}"][k], z[f"e_{sh}"][k]))
    ms = [m for m in ms if gs[m]]
    for i, m in enumerate(ms):
        x = np.concatenate(gs[m])
        col = METHOD_COLOR[m]
        horiz = {"orientation": "horizontal"} if tuple(int(v) for v in matplotlib.__version__.split(".")[:2]) >= (3, 10) \
            else {"vert": False}
        bp = ax.boxplot([x], positions=[i], **horiz, widths=0.55, showfliers=False, patch_artist=True,
                        medianprops={"color": "white" if m in ("r1", "r2", "dither", "zeroshot") else C["ink"],
                                     "linewidth": 1.0},
                        boxprops={"facecolor": "white" if m == "arx" else col,  # ARX: reference, outline only
                                  "edgecolor": C["ink2"] if m == "arx" else col, "linewidth": 0.7},
                        whiskerprops={"color": col, "linewidth": 0.6}, capprops={"color": col, "linewidth": 0.6})
        del bp
    ax.axvline(1.0, color=C["ink2"], linewidth=0.6, linestyle="--", zorder=0)
    ax.set_yticks(range(len(ms)), [METHOD_LABEL[m] for m in ms])
    ax.set_ylim(len(ms) - 0.5, -0.5)
    ax.set_xlabel("effect gain $g$")
    ax.set_title("(b) semi-synthetic EPF what-if", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig7_real_world")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp1-run", default="main")
    p.add_argument("--exp2-run", default="main")
    p.add_argument("--exp3-run", default="v2")
    p.add_argument("--exp5-run", default="main")
    p.add_argument("--exp6-run", default="main")
    p.add_argument("--exp8-run", default="main")
    p.add_argument("--only", nargs="+", default=None, choices=["1", "2", "3", "4", "5", "6", "7"])
    a = p.parse_args()
    setup_logging()
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    style()
    root = results_root()
    out = root / "reports" / "figs"
    only = set(a.only or "1234567")
    e3 = root / "exp3" / a.exp3_run
    jobs = [("1", lambda: fig1(root / "exp1" / a.exp1_run, out)),
            ("2", lambda: fig2(e3, out)),
            ("3", lambda: fig3(root / "exp2" / a.exp2_run, e3, out)),
            ("4", lambda: fig4(root / "exp2" / a.exp2_run, out)),
            ("5", lambda: fig5(e3, root / "exp5" / a.exp5_run, out)),
            ("6", lambda: fig6(root / "exp8" / a.exp8_run, e3, out)),
            ("7", lambda: fig7(root / "exp6" / a.exp6_run, root / "reports", out, a.exp6_run))]
    for k, fn in jobs:
        if k in only:
            try:
                fn()
            except FileNotFoundError as e:
                log.warning("figure %s skipped - missing results: %s", k, e)


if __name__ == "__main__":
    main()
