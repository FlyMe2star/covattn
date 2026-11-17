"""Exp6 - real-world evidence (pre-declared, every task reported).

Part `fev`: all fev-bench tasks with known dynamic covariates and one target (src/covattn/fev_tasks.py, copied
  from the fev repo; 28 tasks). Per task: the first `n_series` series by id, the official horizon and number of
  rolling test windows, context up to 2048. Methods: zero-shot, zero-shot without covariates, R1 (Exp3 choice),
  R2, full FT, LoRA, and two validation-gated variants that pick, per task, using ONLY the `n_val` windows just
  before the first test window:
    R1-val: gamma in {1 (= zero-shot), 1.5, 2, 2.5} on the R1 heads, lowest validation WQL;
    R2-val: zero-shot or R2, lowest validation WQL.
  Metrics: MASE (fev seasonality) and WQL on the 9 fev quantile levels; relative to zero-shot per task, geometric
  mean over tasks, win / loss counts.
Part `semi`: semi-synthetic what-if test on real backgrounds. Background = an EPF price series; covariate = the
  first known covariate of the NEXT EPF market, circularly shifted by 26 weeks (no causal link to the background),
  standardised; injected effect b * B(q)/A(q) u with b set so that std(effect) = kappa * std(background) on the
  context. Counterfactual = step / pulse of 1 std in the future covariate; the true effect is exact (linear filter).
  Methods: zero-shot, R1, R2, dither, full FT, LoRA, ARX.

  python scripts/exp6_real_world.py --quick
  python scripts/exp6_real_world.py --resume --exp3-run v2      # Colab
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from scipy.signal import lfilter

from covattn import metrics as M
from covattn.baselines import arx_effects, dither_predict
from covattn.fev_tasks import FEV_REVISION, TASKS
from covattn.io_utils import RunDir, fmt_seconds, results_root, setup_logging
from covattn.model import CovBatch
from covattn.real_data import EPF, fev_windows, load_epf, load_fev_task
from covattn.repair import gain_spec, r1_gain
from covattn.systems import SystemSpec, delta_u, transfer_coeffs

log = logging.getLogger("exp6")
FEV_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MEDIAN_IDX = 10


@dataclass
class Exp6Config:
    run_name: str = "main"
    quick: bool = False
    exp3_run: str = "v2"
    parts: list[str] = field(default_factory=lambda: ["fev", "semi"])
    # fev
    tasks: list[str] = field(default_factory=lambda: [t["config"] for t in TASKS])
    n_series: int = 100
    n_val: int = 3
    L_max: int = 2048
    r1_gammas: list[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    use_ft: bool = True
    # semi-synthetic
    semi_backgrounds: list[str] = field(default_factory=lambda: list(EPF))
    semi_systems: list[str] = field(default_factory=lambda: ["F2_a0.8", "F3_a0.8_d4", "F4_P24_z0.2"])
    semi_kappas: list[float] = field(default_factory=lambda: [0.5, 1.0])
    semi_contexts: list[int] = field(default_factory=lambda: [512, 2048])
    semi_windows: int = 40
    semi_stride: int = 168
    semi_shift: int = 26 * 168
    semi_burn: int = 500
    horizon: int = 64
    shapes: list[str] = field(default_factory=lambda: ["step", "pulse"])
    eff_thresh: float = 0.05
    dither_K: int = 8
    seed: int = 0
    fev_revision: str = FEV_REVISION
    model_id: str = "amazon/chronos-2"
    amp: str = "off"
    device: str = "auto"
    batch_size: int = 128


def parse_args() -> tuple[Exp6Config, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--exp3-run", default=None)
    p.add_argument("--parts", nargs="+", default=None, choices=["fev", "semi"])
    p.add_argument("--no-ft", action="store_true", help="skip the fine-tuned models")
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    cfg = Exp6Config(quick=a.quick, amp=a.amp, device=a.device, batch_size=a.batch_size)
    if a.quick:
        cfg.run_name, cfg.exp3_run = "quick", "quick"
        cfg.tasks = ["proenfo_gfc12", "rohlik_orders_1W"]
        cfg.n_series, cfg.L_max = 3, 256
        cfg.semi_backgrounds, cfg.semi_systems, cfg.semi_kappas = ["epf_be", "epf_de"], ["F2_a0.8"], [0.5]
        cfg.semi_contexts, cfg.semi_windows, cfg.dither_K = [256], 4, 2
    if a.run_name is not None:
        cfg.run_name = a.run_name
    if a.exp3_run is not None:
        cfg.exp3_run = a.exp3_run
    if a.parts is not None:
        cfg.parts = a.parts
    if a.no_ft:
        cfg.use_ft = False
    return cfg, a


# ------------------------------------------------------------------------------------ methods
def build_methods(w, cfg: Exp6Config) -> tuple[dict, dict]:
    d3 = results_root() / "exp3" / cfg.exp3_run
    r1 = json.loads((d3 / "r1_select.json").read_text())
    k, gamma = r1["best"]["k"], r1["best"]["gamma"]
    heads = [tuple(x) for x in r1["ranking"][:k]]
    hk = w.hooked
    gates = torch.tensor(json.loads((d3 / "r2_gates.json").read_text())["gates"])
    sg = json.loads((d3 / "dither_select.json").read_text())["best"]["sigma"]
    methods = {"zeroshot": lambda bt, s: w.predict(bt, bt.H),
               "r1": lambda bt, s: w.predict_hooked(bt, gain_spec(r1_gain(hk.n_layers, hk.n_heads, heads, gamma))),
               "r2": lambda bt, s: w.predict_hooked(bt, gain_spec(gates)),
               "dither": lambda bt, s: dither_predict(w, bt, sg, cfg.dither_K, seed=s)}
    for gm in cfg.r1_gammas:
        if gm != gamma:
            G = r1_gain(hk.n_layers, hk.n_heads, heads, gm)
            methods[f"r1_g{gm:g}"] = lambda bt, s, G=G: w.predict_hooked(bt, gain_spec(G))
    info = {"k": k, "gamma": gamma, "dither_sigma": sg, "exp3_dir": str(d3)}
    if cfg.use_ft:
        from covattn.model import Chronos2Wrapper

        for mode in ("full", "lora"):
            f = d3 / f"ft_{mode}" / "info.json"
            if f.exists():
                wf = Chronos2Wrapper(json.loads(f.read_text())["path"], device=str(w.device), amp=cfg.amp,
                                     batch_size=cfg.batch_size)
                methods[f"ft_{mode}"] = lambda bt, s, wf=wf: wf.predict(bt, bt.H)
    return methods, info


# ------------------------------------------------------------------------------------ fev part
def _season_scale(ctx: np.ndarray, s: int) -> np.ndarray:
    s = max(1, min(s, ctx.shape[1] - 1))
    d = np.abs(ctx[:, s:] - ctx[:, :-s])
    with np.errstate(all="ignore"):
        return np.nanmean(d, axis=-1)


def _scores(q: np.ndarray, yt: np.ndarray, ctx: np.ndarray, season: int, lv_idx: list[int]) -> dict:
    """MASE: mean over windows with a finite, positive scale; WQL: pooled over valid targets, fev levels."""
    med = q[:, MEDIAN_IDX]
    scale = _season_scale(ctx, season)
    with np.errstate(all="ignore"):
        mae = np.nanmean(np.abs(med - yt), axis=-1)
    ok = np.isfinite(scale) & (scale > 0) & np.isfinite(mae)
    mase = float(np.mean(mae[ok] / scale[ok])) if ok.any() else float("nan")
    lv = np.asarray(FEV_LEVELS)[None, :, None]
    qq = q[:, lv_idx]
    y = yt[:, None, :]
    valid = np.isfinite(y) & np.isfinite(qq)
    loss = np.where(valid, 2.0 * np.abs((y - qq) * ((y <= qq).astype(np.float64) - lv)), 0.0)
    den = np.nansum(np.abs(yt))
    wql = float(loss.sum() / len(FEV_LEVELS) / den) if den > 0 else float("nan")
    return {"mase": mase, "wql": wql}


def run_fev_task(task: dict, w, methods: dict, cfg: Exp6Config, run: RunDir) -> dict:
    t0 = time.time()
    series = load_fev_task(task, cfg.n_series)
    H, S = task["horizon"], task["seasonality"]
    win = fev_windows(series, H, task["num_windows"], cfg.n_val, cfg.L_max, min_ctx=max(S + 1, 8))
    lv_idx = [int(np.argmin(np.abs(np.asarray(w.quantiles) - l))) for l in FEV_LEVELS]
    rec = {"key": f"fev|{task['config']}", "part": "fev", "task": task["config"], "horizon": H, "season": S,
           "n_series": len(series), "known": task["known"], "load_seconds": time.time() - t0, "methods": {}}
    arrays = {}
    for split in ("test", "val"):
        if split not in win:
            continue
        ctx, cp, cf, yt, sid = win[split]
        rec[f"n_{split}"] = len(ctx)
        arrays[f"y_{split}"] = yt
        batches = {"cov": CovBatch(ctx, cp, cf), "nocov": CovBatch(ctx, cp[:, :0], cf[:, :0])}
        for m, fn in methods.items():
            variants = ("cov", "nocov") if m == "zeroshot" else ("cov",)
            for v in variants:
                q = fn(batches[v], cfg.seed + 11)
                name = m if v == "cov" else "nocov"
                arrays[f"q_{name}_{split}"] = q[:, lv_idx + [MEDIAN_IDX]].astype(np.float32)
                rec["methods"].setdefault(name, {})[split] = _scores(q, yt, ctx, S, lv_idx)
    # validation gating (only validation windows are used for the choice)
    g = rec["methods"]
    def choose(cands: list[str]) -> str:
        vals = {c: g[c]["val"]["wql"] for c in cands if c in g and "val" in g[c] and np.isfinite(g[c]["val"]["wql"])}
        return min(vals, key=vals.get) if vals else "zeroshot"
    r1c = ["zeroshot", "r1"] + [m for m in g if m.startswith("r1_g")]
    rec["gate"] = {"r1_val": choose(r1c), "r2_val": choose(["zeroshot", "r2"])}
    for gname, pick in rec["gate"].items():
        g[gname] = {"test": g[pick]["test"], "picked": pick}
    run.save_arrays(rec["key"], **arrays)
    return rec


# ------------------------------------------------------------------------------------ semi-synthetic part
def _sys(key: str) -> SystemSpec:
    from covattn.systems import default_systems

    return {s.key: s for s in default_systems()}[key]


def run_semi_cell(bg: str, cov_src: str, sys_key: str, kappa: float, L: int, w, methods: dict, cfg: Exp6Config,
                  run: RunDir, cache: dict) -> dict:
    if bg not in cache:
        cache[bg] = load_epf(bg)
    if cov_src not in cache:
        cache[cov_src] = load_epf(cov_src)
    y_bg_all = cache[bg].target.astype(np.float64)
    u_all = np.roll(cache[cov_src].covs[0].astype(np.float64), cfg.semi_shift)
    u_all = (u_all - u_all.mean()) / u_all.std()
    sysp = _sys(sys_key)
    B, A = transfer_coeffs(sysp)
    H, T = cfg.horizon, min(len(y_bg_all), len(u_all))
    origins = [T - H - i * cfg.semi_stride for i in range(cfg.semi_windows)][::-1]
    origins = [o for o in origins if o - L - cfg.semi_burn >= 0]
    ys, us, ufs, yts, bs, es, dus = [], [], [], [], [], [], []
    for o in origins:
        a = o - L - cfg.semi_burn
        u = u_all[a : o + H]
        x = lfilter(B, A, u)[cfg.semi_burn:]  # (L+H,)
        bg_ = y_bg_all[o - L : o + H]
        b = kappa * np.std(bg_[:L]) / max(np.std(x[:L]), 1e-12)
        y = bg_ + b * x
        ys.append(y[:L]); yts.append(y[L:]); us.append(u[cfg.semi_burn: cfg.semi_burn + L]); ufs.append(u[-H:])
        bs.append(b)
        es.append([b * lfilter(B, A, delta_u(sh, H, 1.0)) for sh in cfg.shapes])
        dus.append([delta_u(sh, H, 1.0) for sh in cfg.shapes])
    y, uc, uf, yt = (np.asarray(v, np.float32) for v in (ys, us, ufs, yts))
    e, du = np.asarray(es, np.float32), np.asarray(dus, np.float32)  # (N, S, H)
    arrays = {"y_true": yt, "y_scale": y.std(-1)}
    for si, sh in enumerate(cfg.shapes):
        arrays[f"e_{sh}"] = e[:, si]
        arrays[f"keep_{sh}"] = M.effect_mask(e[:, si], y.std(-1), cfg.eff_thresh)
    key = f"semi|{bg}|{sys_key}|k{kappa:g}|L{L}"
    rec = {"key": key, "part": "semi", "background": bg, "cov_source": cov_src, "system": sys_key, "kappa": kappa,
           "L": L, "n": len(y), "methods": {}}
    fac = CovBatch(y, uc, uf)
    for m, fn in methods.items():
        if m.startswith("r1_g"):
            continue
        q = fn(fac, cfg.seed + 13)
        arrays[f"med_{m}"] = q[:, MEDIAN_IDX].astype(np.float32)
        mrec = {}
        for si, sh in enumerate(cfg.shapes):
            qc = fn(CovBatch(y, uc, uf + du[:, si]), cfg.seed + 13)
            eh = (qc[:, MEDIAN_IDX] - q[:, MEDIAN_IDX]).astype(np.float32)
            arrays[f"ehat_{m}_{sh}"] = eh
            k = arrays[f"keep_{sh}"]
            gg = M.effect_gain(eh[k], e[k, si])
            mrec[f"g_{sh}"] = float(np.median(gg)) if len(gg) else float("nan")
        rec["methods"][m] = mrec
    eff, _ = arx_effects(y, uc, du)
    for si, sh in enumerate(cfg.shapes):
        arrays[f"ehat_arx_{sh}"] = eff[:, si].astype(np.float32)
    run.save_arrays(key, **arrays)
    return rec


# ------------------------------------------------------------------------------------ summary
LABEL = {"zeroshot": "Zero-shot", "nocov": "Zero-shot, no covariates", "r1": "R1", "r2": "R2",
         "r1_val": "R1-val (gated)", "r2_val": "R2-val (gated)", "dither": "Dither (reimpl.)",
         "ft_full": "Full fine-tune", "ft_lora": "LoRA", "arx": "ARX (context fit)"}


def _gmean(x: list[float]) -> float:
    x = np.asarray([v for v in x if np.isfinite(v) and v > 0])
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


def _load(run: RunDir, key: str) -> dict:
    return dict(np.load(run.cell_path(key), allow_pickle=False))


def write_summary(run: RunDir, cfg: Exp6Config) -> None:
    recs = run.read_metrics()
    rep = results_root() / "reports"
    (rep / "tables").mkdir(parents=True, exist_ok=True)
    lines = [f"# Exp6 summary ({cfg.run_name})", "", f"Source: `{run.metrics_file}`. Exp3 artefacts: `{cfg.exp3_run}`. "
             f"fev-bench task list at fev revision {cfg.fev_revision[:10]}.", ""]
    fev = [r for r in recs if r["part"] == "fev"]
    if fev:
        order = ["nocov", "r1", "r2", "r1_val", "r2_val", "ft_full", "ft_lora"]
        ms = [m for m in order if any(m in r["methods"] for r in fev)]
        lines += [f"## A. fev-bench covariate tasks ({len(fev)} of {len(cfg.tasks)} tasks; first {cfg.n_series} "
                  f"series per task; test windows as in fev-bench)", "",
                  "Per task: test WQL of zero-shot, and each method's WQL relative to zero-shot (<1 = better). "
                  "`picked` = what the validation gate chose.", "",
                  "| task | series | test win. | zero-shot WQL | " + " | ".join(LABEL[m] for m in ms) + " | R1-val picked | R2-val picked |",
                  "|---|---|---|---|" + "---|" * len(ms) + "---|---|"]
        rel = {m: {"wql": [], "mase": []} for m in ms}
        for r in sorted(fev, key=lambda r: r["task"]):
            z = r["methods"]["zeroshot"]["test"]
            row = []
            for m in ms:
                if m in r["methods"]:
                    t = r["methods"][m]["test"]
                    rw, rm = t["wql"] / z["wql"], t["mase"] / z["mase"]
                    rel[m]["wql"].append(rw)
                    rel[m]["mase"].append(rm)
                    row.append(f"{rw:.3f}")
                else:
                    row.append("-")
            lines.append(f"| {r['task']} | {r['n_series']} | {r.get('n_test', 0)} | {z['wql']:.4f} | " + " | ".join(row)
                         + f" | {r['gate']['r1_val']} | {r['gate']['r2_val']} |")
        lines += ["", "Aggregate over tasks (geometric mean of the ratio to zero-shot, 95% bootstrap CI over tasks with "
                  "2000 resamples; wins / losses = tasks with ratio < 0.99 / > 1.01):", "",
                  "| method | WQL gmean [95% CI] | MASE gmean [95% CI] | WQL wins / losses | tasks |", "|---|---|---|---|---|"]
        tex = ["\\begin{tabular}{l cccc}", "\\toprule", "Method & WQL & MASE & wins/losses & tasks \\\\", "\\midrule"]
        brng = np.random.default_rng(0)

        def ci(v: list[float]) -> tuple[float, float]:
            v = np.asarray(v)
            bs = [_gmean(list(v[brng.integers(0, len(v), len(v))])) for _ in range(2000)]
            return tuple(np.nanpercentile(bs, [2.5, 97.5]))

        for m in ms:
            v = rel[m]["wql"]
            wins = sum(x < 0.99 for x in v)
            loss = sum(x > 1.01 for x in v)
            cw, cm = ci(v), ci(rel[m]["mase"])
            lines.append(f"| {LABEL[m]} | {_gmean(v):.3f} [{cw[0]:.3f}, {cw[1]:.3f}] | {_gmean(rel[m]['mase']):.3f} "
                         f"[{cm[0]:.3f}, {cm[1]:.3f}] | {wins} / {loss} | {len(v)} |")
            tex.append(f"{LABEL[m]} & {_gmean(v):.3f} [{cw[0]:.2f}, {cw[1]:.2f}] & {_gmean(rel[m]['mase']):.3f} & "
                       f"{wins}/{loss} & {len(v)} \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]
        (rep / "tables" / f"table5_fev_{cfg.run_name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        # does R1 help more where covariates matter? (covariate value = WQL(no cov) / WQL(cov), zero-shot)
        cv, r1r = [], []
        for r in fev:
            z = r["methods"]["zeroshot"]["test"]["wql"]
            if "nocov" in r["methods"] and "r1" in r["methods"]:
                cv.append(r["methods"]["nocov"]["test"]["wql"] / z)
                r1r.append(r["methods"]["r1"]["test"]["wql"] / z)
        if len(cv) >= 3:
            from scipy.stats import spearmanr

            rho, p = spearmanr(cv, r1r)
            lines += ["", f"Covariate value (zero-shot WQL without / with covariates) vs R1 WQL ratio over tasks: "
                      f"Spearman rho = {rho:.3f} (p = {p:.2g}, n = {len(cv)}).", ""]
    semi = [r for r in recs if r["part"] == "semi"]
    if semi:
        cells = []
        for r in semi:
            z = _load(run, r["key"])
            z.update({k: r[k] for k in ("background", "system", "kappa", "L")})
            cells.append(z)
        ms = [m for m in LABEL if f"ehat_{m}_{cfg.shapes[0]}" in cells[0]]

        def stats(cs, m):
            gs, rs, ns, er = [], [], [], []
            for c in cs:
                for sh in cfg.shapes:
                    k = c[f"keep_{sh}"]
                    eh, e = c[f"ehat_{m}_{sh}"][k], c[f"e_{sh}"][k]
                    gs.append(M.effect_gain(eh, e)); rs.append(M.shape_corr(eh, e)); ns.append(M.whatif_nrmse(eh, e))
                if f"med_{m}" in c:
                    er.append(np.abs(c[f"med_{m}"] - c["y_true"]).mean(-1)
                              / np.maximum(np.abs(c["med_zeroshot"] - c["y_true"]).mean(-1), 1e-12))
            g, rr, nn = (np.concatenate(x) for x in (gs, rs, ns))
            return (float(np.median(g)), float(np.nanmedian(rr)), float(np.median(nn)),
                    float(np.median(np.concatenate(er))) if er else np.nan, len(g))

        groups = [("all", cells)] + [(f"system={s}", [c for c in cells if c["system"] == s]) for s in cfg.semi_systems]
        groups += [(f"kappa={k:g}", [c for c in cells if c["kappa"] == k]) for k in cfg.semi_kappas]
        groups += [(f"L={L}", [c for c in cells if c["L"] == L]) for L in cfg.semi_contexts]
        groups += [(f"background={b}", [c for c in cells if c["background"] == b]) for b in cfg.semi_backgrounds]
        lines += ["## B. Semi-synthetic what-if on real EPF backgrounds", "",
                  f"Covariate = first known covariate of the next EPF market, shifted {cfg.semi_shift} steps; injected "
                  f"effect std = kappa x background std; counterfactual = 1 std step / pulse; {cfg.semi_windows} "
                  "windows per background. Per method: median g / shape r / what-if NRMSE / factual error ratio vs "
                  "zero-shot / n pairs.", ""]
        tex = ["\\begin{tabular}{l cccc}", "\\toprule", "Method & $g$ & $r$ & NRMSE & err.\\ ratio \\\\", "\\midrule"]
        for gname, cs in groups:
            cs = [c for c in cs]
            if not cs:
                continue
            lines += [f"### {gname} ({len(cs)} cells)", "", "| method | g | r | NRMSE | err ratio | n |",
                      "|---|---|---|---|---|---|"]
            for m in ms:
                s = stats(cs, m)
                lines.append(f"| {LABEL[m]} | {s[0]:.3f} | {s[1]:.3f} | {s[2]:.3f} | {s[3]:.3f} | {s[4]} |")
                if gname == "all":
                    tex.append(f"{LABEL[m]} & {s[0]:.2f} & {s[1]:.2f} & {s[2]:.2f} & "
                               f"{'--' if not np.isfinite(s[3]) else f'{s[3]:.2f}'} \\\\")
            lines.append("")
        tex += ["\\bottomrule", "\\end{tabular}"]
        (rep / "tables" / f"table6_semisynth_{cfg.run_name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    out = rep / f"exp6_summary_{cfg.run_name}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("summary written: %s", out)


# ------------------------------------------------------------------------------------ main
def main() -> None:
    cfg, args = parse_args()
    run = RunDir("exp6", cfg.run_name)
    setup_logging(run.path / "log.txt")
    log.info("config: %s", json.dumps(asdict(cfg)))
    if args.summary_only:
        write_summary(run, cfg)
        return
    run.write_config(cfg, resume=args.resume, ignore_keys=("device", "batch_size", "parts"))
    from covattn.model import Chronos2Wrapper

    w = Chronos2Wrapper(cfg.model_id, device=cfg.device, amp=cfg.amp, batch_size=cfg.batch_size)
    torch.manual_seed(cfg.seed)
    methods, info = build_methods(w, cfg)
    (run.path / "methods.json").write_text(json.dumps({"methods": list(methods), **info}, indent=1))
    log.info("methods: %s %s", list(methods), info)
    log.info("ESTIMATED TIME (rough): fev part ~1-2 h on L4 (28 tasks, dominated by long-context hooked passes and "
             "data download), semi part ~20-30 min")
    done = run.done_keys() if args.resume else set()
    jobs = []
    if "fev" in cfg.parts:
        tk = {t["config"]: t for t in TASKS}
        jobs += [("fev", tk[n]) for n in cfg.tasks if f"fev|{n}" not in done]
    if "semi" in cfg.parts:
        bgs = cfg.semi_backgrounds
        for i, bg in enumerate(bgs):
            src = EPF[(EPF.index(bg) + 1) % len(EPF)]
            for s in cfg.semi_systems:
                for kp in cfg.semi_kappas:
                    for L in cfg.semi_contexts:
                        if f"semi|{bg}|{s}|k{kp:g}|L{L}" not in done:
                            jobs.append(("semi", (bg, src, s, kp, L)))
    cache: dict = {}
    t_start = time.time()
    for i, (part, job) in enumerate(jobs):
        t0 = time.time()
        try:
            if part == "fev":
                rec = run_fev_task(job, w, {m: f for m, f in methods.items() if m != "dither"}, cfg, run)
                msg = {m: round(v["test"]["wql"], 4) for m, v in rec["methods"].items() if "test" in v}
            else:
                rec = run_semi_cell(*job, w, methods, cfg, run, cache)
                msg = {m: round(v[f"g_{cfg.shapes[0]}"], 3) for m, v in rec["methods"].items()}
        except Exception:  # noqa: BLE001 - report and continue; the task is then missing from the tables
            log.exception("%s %s FAILED - continuing", part, job if part == "semi" else job["config"])
            if w.device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        rec["seconds"] = time.time() - t0
        rec["device"] = str(w.device)
        run.append_metrics(rec)
        el = time.time() - t_start
        log.info("[%d/%d] %s %.0fs ETA %s | %s", i + 1, len(jobs), rec["key"], rec["seconds"],
                 fmt_seconds(el / (i + 1) * (len(jobs) - i - 1)), msg)
    write_summary(run, cfg)
    n_all = (len(cfg.tasks) if "fev" in cfg.parts else 0) + (len(cfg.semi_backgrounds) * len(cfg.semi_systems) * len(
        cfg.semi_kappas) * len(cfg.semi_contexts) if "semi" in cfg.parts else 0)
    if len(run.done_keys()) >= n_all:
        run.mark_done({"n": n_all})
        log.info("all %d jobs done -> done.marker", n_all)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        log.exception("Exp6 crashed")
        raise
