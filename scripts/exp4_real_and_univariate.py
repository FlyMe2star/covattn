"""Exp4 - real covariate tasks and preservation of univariate skill.

Uses the repairs / baselines produced by Exp3 (`$RESULTS_DIR/exp3/<exp3-run>`): R1 choice, R2 gates,
dither sigma, fine-tuned models (if present).

Part A (covariate tasks, EPF x5, hourly, H=24, rolling origins over the last n_windows days):
  MASE / WQL with the true future covariates, with future covariates shuffled in time
  (per window, seeded), and without covariates. Faithfulness proxies: error increase when the
  covariates are shuffled / removed.
Part B (univariate datasets): last-window MASE / WQL. R1/R2 must equal zero-shot exactly; the
  max abs difference is recorded.

  python scripts/exp4_real_and_univariate.py --quick
  python scripts/exp4_real_and_univariate.py --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from covattn import metrics as M
from covattn.baselines import dither_predict
from covattn.hooks import HookSpec
from covattn.io_utils import RunDir, fmt_seconds, results_root, setup_logging
from covattn.latex import DOWN, esc, mark_best, rows_to_tex
from covattn.model import CovBatch
from covattn.real_data import EPF, UNIVARIATE, last_window, load_epf, load_univariate, rolling_windows
from covattn.repair import gain_spec, r1_gain

log = logging.getLogger("exp4")

METHOD_LABEL = {"zeroshot": "Zero-shot", "r1": "R1 (top-k gain)", "r2": "R2 (gates)", "dither": "Dither (reimpl.)",
                "ft_full": "Full fine-tune", "ft_lora": "LoRA"}
TRAINING_FREE = ("zeroshot", "r1", "r2", "dither")
COL_SHORT = {"r1": "R1", "r2": "R2", "dither": "Dither", "ft_full": "Full FT", "ft_lora": "LoRA"}
DATASET_SHORT = {"epf_be": "BE", "epf_de": "DE", "epf_fr": "FR", "epf_np": "NP", "epf_pjm": "PJM",
                 "exchange_rate": "Exchange", "monash_hospital": "Hospital", "monash_tourism_monthly": "Tourism",
                 "nn5": "NN5", "ercot": "ERCOT"}


@dataclass
class Exp4Config:
    run_name: str = "main"
    quick: bool = False
    exp3_run: str = "main"
    cov_datasets: list[str] = field(default_factory=lambda: list(EPF))
    cov_context: int = 2048
    cov_horizon: int = 24
    n_windows: int = 60
    n_shuffles: int = 3
    uni_datasets: list[str] = field(default_factory=lambda: list(UNIVARIATE))
    uni_context: int = 2048
    uni_max_series: int = 0  # 0 = all
    dither_K: int = 8
    seed: int = 0
    model_id: str = "amazon/chronos-2"
    amp: str = "off"
    device: str = "auto"
    batch_size: int = 128


def parse_args() -> tuple[Exp4Config, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--exp3-run", default=None)
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    cfg = Exp4Config(quick=a.quick, amp=a.amp, device=a.device, batch_size=a.batch_size)
    if a.quick:
        cfg.run_name, cfg.exp3_run = "quick", "quick"
        cfg.cov_datasets, cfg.cov_context, cfg.n_windows, cfg.n_shuffles = ["epf_de"], 512, 4, 1
        cfg.uni_datasets, cfg.uni_context, cfg.uni_max_series, cfg.dither_K = ["monash_hospital"], 512, 8, 2
    if a.run_name is not None:
        cfg.run_name = a.run_name
    if a.exp3_run is not None:
        cfg.exp3_run = a.exp3_run
    return cfg, a


def build_methods(w, cfg: Exp4Config) -> tuple[dict, dict]:
    from covattn.model import Chronos2Wrapper

    d3 = results_root() / "exp3" / cfg.exp3_run
    hk = w.hooked
    r1 = json.loads((d3 / "r1_select.json").read_text())
    b = r1["best"]
    G1 = r1_gain(hk.n_layers, hk.n_heads, [tuple(x) for x in r1["ranking"][: b["k"]]], b["gamma"])
    G2 = torch.tensor(json.loads((d3 / "r2_gates.json").read_text())["gates"])
    sg = json.loads((d3 / "dither_select.json").read_text())["best"]["sigma"]
    methods = {
        "zeroshot": lambda bt, s: w.predict(bt, bt.H),
        "r1": lambda bt, s: w.predict_hooked(bt, gain_spec(G1)),
        "r2": lambda bt, s: w.predict_hooked(bt, gain_spec(G2)),
        "dither": lambda bt, s: dither_predict(w, bt, sg, cfg.dither_K, seed=s),
    }
    info = {"r1": b, "dither_sigma": sg, "exp3_dir": str(d3)}
    for mode in ("full", "lora"):
        f = d3 / f"ft_{mode}" / "info.json"
        if f.exists():
            wf = Chronos2Wrapper(json.loads(f.read_text())["path"], device=str(w.device), amp=cfg.amp,
                                 batch_size=cfg.batch_size)
            methods[f"ft_{mode}"] = lambda bt, s, wf=wf: wf.predict(bt, bt.H)
        else:
            log.warning("no fine-tuned %s model in %s - method skipped", mode, d3)
    return methods, info


def _scores(q: np.ndarray, yt: np.ndarray, ctx: np.ndarray, season: int, qs: list[float], mi: int) -> dict:
    return {"mase": float(np.nanmean(M.mase(q[:, mi], yt, ctx, season))), "wql": M.pooled_wql(q, qs, yt)}


def run_cov_task(name: str, w, methods: dict, cfg: Exp4Config, run: RunDir) -> dict:
    task = load_epf(name)
    y, cp, cf, yt = rolling_windows(task, cfg.cov_context, cfg.cov_horizon, cfg.n_windows)
    rng = np.random.default_rng(cfg.seed)
    shuffles = []
    for _ in range(cfg.n_shuffles):
        perm = np.stack([np.stack([rng.permutation(cfg.cov_horizon) for _ in range(cp.shape[1])]) for _ in range(len(y))])
        shuffles.append(np.take_along_axis(cf, perm, axis=-1))
    variants = {"cov": CovBatch(y, cp, cf), "nocov": CovBatch(y, cp[:, :0], cf[:, :0])}
    for i, s in enumerate(shuffles):
        variants[f"shuf{i}"] = CovBatch(y, cp, s)
    rec = {"key": f"cov|{name}", "part": "cov", "dataset": name, "n_windows": len(y), "covariates": task.cov_names,
           "methods": {}}
    arrays = {"y_true": yt}
    for m, fn in methods.items():
        r = {}
        for vname, bt in variants.items():
            q = fn(bt, cfg.seed + 11)
            if vname in ("cov", "nocov"):
                arrays[f"q_{m}_{vname}"] = q.astype(np.float32)
            r[vname] = _scores(q, yt, y, task.season, w.quantiles, w.median_idx)
        sh = [r[k] for k in r if k.startswith("shuf")]
        r["shuf"] = {k: float(np.mean([s[k] for s in sh])) for k in ("mase", "wql")}
        rec["methods"][m] = {v: r[v] for v in ("cov", "shuf", "nocov")}
        log.info("%s %s: %s", name, m, {v: round(r[v]["mase"], 4) for v in ("cov", "shuf", "nocov")})
    run.save_arrays(rec["key"], **arrays)
    return rec


def run_uni_task(name: str, w, methods: dict, cfg: Exp4Config, run: RunDir) -> dict:
    H, season = UNIVARIATE[name]
    series = [s[~np.isnan(s)] if np.isnan(s).any() else s for s in load_univariate(name)]
    series = [s for s in series if len(s) > H + max(season, 1) + 1]
    if cfg.uni_max_series:
        series = series[: cfg.uni_max_series]
    L = min(cfg.uni_context, max(len(s) - H for s in series))
    ctx, yt = last_window(series, L, H)
    bt = CovBatch(ctx, np.zeros((len(ctx), 0, L)), np.zeros((len(ctx), 0, H)))
    rec = {"key": f"uni|{name}", "part": "uni", "dataset": name, "n_series": len(series), "H": H, "L": L,
           "season": season, "methods": {}}
    arrays = {"y_true": yt}
    q_hook_off = w.predict_hooked(bt, HookSpec())
    for m, fn in methods.items():
        q = fn(bt, cfg.seed + 13)
        arrays[f"q_{m}"] = q.astype(np.float32)
        r = _scores(q, yt, ctx, season, w.quantiles, w.median_idx)
        if m in ("r1", "r2"):
            r["max_abs_diff_vs_hooks_off"] = float(np.abs(q - q_hook_off).max())
            r["max_abs_diff_vs_zeroshot"] = float(np.abs(q - arrays["q_zeroshot"]).max())
            r["max_rel_diff_vs_zeroshot"] = float(np.abs(q - arrays["q_zeroshot"]).max() / max(np.abs(arrays["q_zeroshot"]).max(), 1e-12))
        rec["methods"][m] = r
        log.info("%s %s: %s", name, m, {k: round(v, 6) for k, v in r.items()})
    run.save_arrays(rec["key"], **arrays)
    return rec


def write_summary(run: RunDir, cfg: Exp4Config) -> None:
    recs = run.read_metrics()
    if not recs:
        return
    rep = results_root() / "reports"
    (rep / "tables").mkdir(parents=True, exist_ok=True)
    lines = [f"# Exp4 summary ({cfg.run_name})", "", f"Source: `{run.metrics_file}`. Exp3 run: `{cfg.exp3_run}`.", ""]
    cov = [r for r in recs if r["part"] == "cov"]
    uni = [r for r in recs if r["part"] == "uni"]
    if cov:
        meths = [m for m in METHOD_LABEL if m in cov[0]["methods"]]
        lines += ["## A. Covariate tasks (EPF): MASE with covariates; relative MASE change when future "
                  "covariates are shuffled / removed (positive = model relies on them)", "",
                  "| dataset | method | MASE | WQL | shuffled vs cov | no-cov vs cov |", "|---|---|---|---|---|---|"]
        for r in cov:
            for m in meths:
                x = r["methods"][m]
                c = x["cov"]["mase"]
                lines.append(f"| {r['dataset']} ({r['n_windows']} win.) | {METHOD_LABEL[m]} | {c:.3f} | "
                             f"{x['cov']['wql']:.4f} | {100 * (x['shuf']['mase'] / c - 1):+.1f}% | "
                             f"{100 * (x['nocov']['mase'] / c - 1):+.1f}% |")
        lines.append("")
        gm = {m: float(np.exp(np.mean([np.log(r["methods"][m]["cov"]["mase"] / r["methods"]["zeroshot"]["cov"]["mase"])
                                       for r in cov]))) for m in meths}
        lines += ["Geometric mean of MASE relative to zero-shot (with covariates): " +
                  ", ".join(f"{METHOD_LABEL[m]} {v:.3f}" for m, v in gm.items()), ""]
        # Bold = lowest MASE among the training-free methods; fine-tuned rows are references. The two change
        # columns describe covariate reliance (not better/worse), so nothing is marked there.
        cand = [m in TRAINING_FREE for m in meths]
        mcols = [mark_best([r["methods"][m]["cov"]["mase"] for m in meths], ".3f", "min", cand) for r in cov]
        n = len(cov)
        hdr = [f"& \\multicolumn{{{n}}}{{c}}{{MASE{DOWN}, with covariates}} & "
               "\\multicolumn{2}{c}{$\\Delta$MASE, covariates} \\\\",
               f"\\cmidrule(lr){{2-{n + 1}}} \\cmidrule(lr){{{n + 2}-{n + 3}}}",
               "Method & " + " & ".join(DATASET_SHORT.get(r["dataset"], esc(r["dataset"])) for r in cov) +
               " & shuffled & removed \\\\"]
        rows = []
        for i, m in enumerate(meths):
            if m == "ft_full" and i > 0:
                rows.append("\\midrule")
            ds = [r["methods"][m] for r in cov]
            dshuf = np.mean([100 * (x["shuf"]["mase"] / x["cov"]["mase"] - 1) for x in ds])
            dno = np.mean([100 * (x["nocov"]["mase"] / x["cov"]["mase"] - 1) for x in ds])
            rows.append([METHOD_LABEL[m]] + [c[i] for c in mcols] + [f"{dshuf:+.1f}\\%", f"{dno:+.1f}\\%"])
        tex = "\\setlength{\\tabcolsep}{4pt}\n" + rows_to_tex("l " + "c" * n + " cc", hdr, rows)
        (rep / "tables" / f"table2_real_cov_{cfg.run_name}.tex").write_text(tex, encoding="utf-8")
    if uni:
        meths = [m for m in METHOD_LABEL if m in uni[0]["methods"]]
        lines += ["## B. Univariate datasets (last window): MASE / WQL", "",
                  "| dataset | " + " | ".join(METHOD_LABEL[m] for m in meths) + " |", "|---|" + "---|" * len(meths)]
        for r in uni:
            lines.append(f"| {r['dataset']} (n={r['n_series']}, H={r['H']}) | " +
                         " | ".join(f"{r['methods'][m]['mase']:.3f} / {r['methods'][m]['wql']:.4f}" for m in meths) + " |")
        lines.append("")
        for m in ("r1", "r2"):
            if m in meths:
                d = max(r["methods"][m]["max_abs_diff_vs_hooks_off"] for r in uni)
                d0 = max(r["methods"][m]["max_rel_diff_vs_zeroshot"] for r in uni)
                lines.append(f"- {METHOD_LABEL[m]}: max |q - q_hooks_off| = {d:.3g} (bit-identical if 0); "
                             f"max |q - q_zeroshot pipeline| / max|q| = {d0:.3g} (eager vs SDPA float error)")
        lines.append("")
        # Rows whose printed values equal zero-shot in every column are merged into the zero-shot row (R1/R2 do not
        # act on univariate inputs; the exact max differences are in the markdown summary).
        def printed(m: str) -> list[str]:
            return [f"{r['methods'][m][k]:{f}}" for r in uni for k, f in (("mase", ".3f"), ("wql", ".4f"))]

        same = [m for m in meths if m != "zeroshot" and printed(m) == printed("zeroshot")]
        show = [m for m in meths if m not in same]
        label = {m: METHOD_LABEL[m] for m in show}
        if same:
            label["zeroshot"] = ("\\begin{tabular}[b]{@{}c@{}}Zero-shot\\\\= " +
                                 " = ".join(METHOD_LABEL[m].split(" (")[0] for m in same) + "\\end{tabular}")
        label.update({m: COL_SHORT[m] for m in show if m in COL_SHORT})
        # transposed (datasets x metric as rows, methods as columns) so that it fits the LNCS text width;
        # bold = lowest value among the training-free methods in each row
        cand = [m in TRAINING_FREE for m in show]
        n_free = sum(cand)
        hdr = [f"& & \\multicolumn{{{n_free}}}{{c}}{{training-free}} & \\multicolumn{{{len(show) - n_free}}}{{c}}"
               "{fine-tuned (ref.)} \\\\",
               f"\\cmidrule(lr){{3-{2 + n_free}}} \\cmidrule(lr){{{3 + n_free}-{2 + len(show)}}}",
               f"Dataset & Metric{DOWN} & " + " & ".join(label[m] for m in show) + " \\\\"]
        rows = []
        for j, r in enumerate(uni):
            if j > 0:
                rows.append("\\addlinespace[2pt]")
            name = DATASET_SHORT.get(r["dataset"], esc(r["dataset"]))
            for k, f in (("mase", ".3f"), ("wql", ".4f")):
                rows.append([name if k == "mase" else "", k.upper()] +
                            mark_best([r["methods"][m][k] for m in show], f, "min", cand))
        tex = rows_to_tex("l l " + "c" * len(show), hdr, rows)
        (rep / "tables" / f"table3_univariate_{cfg.run_name}.tex").write_text(tex, encoding="utf-8")
    out = rep / f"exp4_summary_{cfg.run_name}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("summary written: %s", out)


def main() -> None:
    cfg, args = parse_args()
    run = RunDir("exp4", cfg.run_name)
    setup_logging(run.path / "log.txt")
    log.info("config: %s", json.dumps(asdict(cfg)))
    if args.summary_only:
        write_summary(run, cfg)
        return
    run.write_config(cfg, resume=args.resume, ignore_keys=("device", "batch_size"))
    from covattn.model import Chronos2Wrapper

    w = Chronos2Wrapper(cfg.model_id, device=cfg.device, amp=cfg.amp, batch_size=cfg.batch_size)
    methods, info = build_methods(w, cfg)
    (run.path / "methods.json").write_text(json.dumps({"methods": list(methods), **info}, indent=1, default=str))
    log.info("methods: %s (%s)", list(methods), info)
    log.info("ESTIMATED TIME: small (%d covariate tasks x %d windows, %d univariate datasets); roughly 10-20 min "
             "on L4, most of it dither and data download", len(cfg.cov_datasets), cfg.n_windows, len(cfg.uni_datasets))
    tasks = [("cov", n) for n in cfg.cov_datasets] + [("uni", n) for n in cfg.uni_datasets]
    done = run.done_keys() if args.resume else set()
    for part, name in tasks:
        key = f"{part}|{name}"
        if key in done:
            continue
        t0 = time.time()
        try:
            rec = (run_cov_task if part == "cov" else run_uni_task)(name, w, methods, cfg, run)
        except Exception:  # noqa: BLE001  - report and continue with the other datasets
            log.exception("task %s failed", key)
            continue
        rec["seconds"] = time.time() - t0
        rec["device"] = str(w.device)
        run.append_metrics(rec)
        log.info("done %s in %s", key, fmt_seconds(rec["seconds"]))
    write_summary(run, cfg)
    if len(run.done_keys()) == len(tasks):
        run.mark_done({"n_tasks": len(tasks)})


if __name__ == "__main__":
    main()
