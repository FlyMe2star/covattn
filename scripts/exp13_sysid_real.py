"""Exp13 - forecasts of real system-identification rigs with known future inputs (protocol docs/sysid_protocol.md).

Records from scripts/sysid_prepare.py (data/sysid/processed or --data). Per benchmark and context length: 64
windows spread evenly over the eligible records; horizon 64. Methods: zero-shot with / without covariates, R1, R2,
dither (from exp3/<run>, nothing tuned here) and ARX fitted on the context and simulated over the horizon with the
known future input (reference). Primary metric: MAE ratio to zero-shot (geometric means; bootstrap CIs).

  python scripts/exp13_sysid_real.py --quick --inputs-root <Drive results>   # local check (CPU is fine)
  python scripts/exp13_sysid_real.py --resume --data <dir with records.npz>   # Colab
  python scripts/exp13_sysid_real.py --summary-only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from covattn.baselines import dither_predict, fit_arx
from covattn.io_utils import RunDir, fmt_seconds, results_root, setup_logging
from covattn.model import CovBatch
from covattn.repair import gain_spec, r1_gain

log = logging.getLogger("exp13")
LEVELS9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BENCHES = ["cascaded_tanks", "wh", "silverbox", "parwh", "f16", "fsm"]
LABEL = {"zeroshot": "Zero-shot", "nocov": "Zero-shot, no covariate", "r1": "R1", "r2": "R2",
         "dither": "Dither (reimpl.)", "arx": "ARX (context fit, reference)"}
INPUTS: dict = {"root": None}


@dataclass
class Exp13Config:
    run_name: str = "main"
    quick: bool = False
    exp3_run: str = "v2"
    benches: list[str] = field(default_factory=lambda: list(BENCHES))
    contexts: list[int] = field(default_factory=lambda: [512, 2048])
    horizon: int = 64
    windows_per_bench: int = 64
    periodic_thresh: float = 0.95
    dither_K: int = 8
    n_boot: int = 2000
    seed: int = 0
    model_id: str = "amazon/chronos-2"
    amp: str = "off"
    device: str = "auto"
    batch_size: int = 128


def parse_args() -> tuple[Exp13Config, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--exp3-run", default=None)
    p.add_argument("--data", default=None, help="directory with records.npz / records.json")
    p.add_argument("--inputs-root", default=None, help="where exp3/ artefacts are read (default RESULTS_DIR)")
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    INPUTS["root"] = Path(a.inputs_root) if a.inputs_root else results_root()
    INPUTS["data"] = Path(a.data) if a.data else Path(__file__).resolve().parents[1] / "data" / "sysid" / "processed"
    cfg = Exp13Config(amp=a.amp, device=a.device, batch_size=a.batch_size, quick=a.quick)
    if a.quick:
        cfg.run_name, cfg.contexts, cfg.windows_per_bench, cfg.dither_K, cfg.n_boot = "quick", [512], 4, 2, 100
    if a.run_name is not None:
        cfg.run_name = a.run_name
    if a.exp3_run is not None:
        cfg.exp3_run = a.exp3_run
    return cfg, a


# ------------------------------------------------------------------------------------ data
def load_records() -> tuple[list[dict], dict]:
    meta = json.loads((INPUTS["data"] / "records.json").read_text(encoding="utf-8"))
    z = np.load(INPUTS["data"] / "records.npz", allow_pickle=False)
    for m in meta:
        m["u"], m["y"] = z[f"u{m['i']}"].astype(np.float64), z[f"y{m['i']}"].astype(np.float64)
    return meta, {}


def windows(recs: list[dict], L: int, H: int, n_total: int) -> list[tuple[int, int]]:
    """(record index into recs, origin) for one benchmark and L: n_total windows spread over eligible records."""
    elig = [i for i, r in enumerate(recs) if len(r["y"]) >= L + H]
    if not elig:
        return []
    per = math.ceil(n_total / len(elig))
    out = []
    for i in elig:
        N = len(recs[i]["y"])
        os_ = np.unique(np.linspace(L, N - H, per).round().astype(int))
        out += [(i, int(o)) for o in os_]
    return out


def periodicity(u: np.ndarray) -> float:
    """max over channels and lags P in [16, n/2] of corr(u_t, u_{t+P}) over the window's input."""
    best = -1.0
    for x in np.atleast_2d(u):
        x = x - x.mean()
        n = len(x)
        for P in range(16, n // 2 + 1):
            a, b = x[:-P], x[P:]
            den = np.sqrt((a * a).sum() * (b * b).sum())
            if den > 0:
                best = max(best, float((a * b).sum() / den))
    return best


def arx_forecast(y_ctx: np.ndarray, u_ctx: np.ndarray, u_fut: np.ndarray) -> np.ndarray:
    """ARX (AIC orders, first input channel only) fitted on the context, simulated over the horizon."""
    f = fit_arx(y_ctx, u_ctx[0])
    y = np.r_[y_ctx, np.zeros(u_fut.shape[1])]
    u = np.r_[u_ctx[0], u_fut[0]]
    L = len(y_ctx)
    for t in range(L, len(y)):
        y[t] = f.c + sum(f.a[i] * y[t - 1 - i] for i in range(f.na)) + sum(f.b[j] * u[t - j] for j in range(f.nb))
    return y[L:]


# ------------------------------------------------------------------------------------ methods
def build_methods(w, cfg: Exp13Config) -> tuple[dict, dict]:
    d3 = INPUTS["root"] / "exp3" / cfg.exp3_run
    r1 = json.loads((d3 / "r1_select.json").read_text())
    k, gamma = r1["best"]["k"], r1["best"]["gamma"]
    hk = w.hooked
    G1 = r1_gain(hk.n_layers, hk.n_heads, [tuple(x) for x in r1["ranking"][:k]], gamma)
    gates = torch.tensor(json.loads((d3 / "r2_gates.json").read_text())["gates"])
    sg = json.loads((d3 / "dither_select.json").read_text())["best"]["sigma"]
    methods = {"zeroshot": lambda bt, s: w.predict(bt, bt.H),
               "r1": lambda bt, s: w.predict_hooked(bt, gain_spec(G1)),
               "r2": lambda bt, s: w.predict_hooked(bt, gain_spec(gates)),
               "dither": lambda bt, s: dither_predict(w, bt, sg, cfg.dither_K, seed=s)}
    return methods, {"k": k, "gamma": gamma, "dither_sigma": sg, "exp3_dir": str(d3)}


# ------------------------------------------------------------------------------------ one cell
def run_cell(bench: str, L: int, recs_all: list[dict], w, methods: dict, cfg: Exp13Config, run: RunDir) -> dict:
    recs = [r for r in recs_all if r["bench"] == bench]
    H = cfg.horizon
    win = windows(recs, L, H, cfg.windows_per_bench)
    if not win:
        raise ValueError(f"{bench}: no record with N >= {L + H}")
    K = recs[0]["u"].shape[0]
    y = np.stack([recs[i]["y"][o - L:o] for i, o in win])
    uc = np.stack([recs[i]["u"][:, o - L:o] for i, o in win])
    uf = np.stack([recs[i]["u"][:, o:o + H] for i, o in win])
    yt = np.stack([recs[i]["y"][o:o + H] for i, o in win])
    arrays = {"y_true": yt.astype(np.float32), "y_ctx_std": y.std(1).astype(np.float32),
              "rec": np.asarray([recs[i]["i"] for i, _ in win], np.int32), "origin": np.asarray([o for _, o in win]),
              "periodicity": np.asarray([periodicity(recs[i]["u"][:, o - L:o + H]) for i, o in win], np.float32),
              "arx_memory": np.asarray([recs[i]["arx_memory"] for i, _ in win], np.float32)}
    lv = [int(np.argmin(np.abs(np.asarray(w.quantiles) - l))) for l in LEVELS9]
    mi = w.median_idx
    fac = CovBatch(y, uc, uf)
    rec = {"key": f"sysid|{bench}|L{L}", "bench": bench, "L": L, "n": len(win), "k": K,
           "records": sorted({recs[i]["name"] for i, _ in win}), "methods": {}}
    for m, fn in methods.items():
        q = fn(fac, cfg.seed + 19)
        arrays[f"q9_{m}"], arrays[f"med_{m}"] = q[:, lv].astype(np.float32), q[:, mi].astype(np.float32)
    q = w.predict(CovBatch(y, uc[:, :0], uf[:, :0]), H)
    arrays["q9_nocov"], arrays["med_nocov"] = q[:, lv].astype(np.float32), q[:, mi].astype(np.float32)
    arx = []
    for j in range(len(win)):
        try:
            arx.append(arx_forecast(y[j], uc[j], uf[j]))
        except Exception:  # noqa: BLE001 - e.g. context too short for any order
            arx.append(np.full(H, np.nan))
    arrays["med_arx"] = np.asarray(arx, np.float32)
    mae0 = np.abs(arrays["med_zeroshot"] - yt).mean(1)
    for m in [*methods, "nocov", "arx"]:
        r = np.abs(arrays[f"med_{m}"] - yt).mean(1) / mae0
        rec["methods"][m] = {"mae_ratio_gmean": float(np.exp(np.nanmean(np.log(r))))}
    run.save_arrays(rec["key"], **arrays)
    return rec


# ------------------------------------------------------------------------------------ summary
def _wql_rows(q9: np.ndarray, y: np.ndarray) -> np.ndarray:
    lv = np.asarray(LEVELS9)[None, :, None]
    yy = y[:, None, :]
    return (2.0 * np.abs((yy - q9) * ((yy <= q9).astype(np.float64) - lv))).mean((1, 2))


def write_summary(run: RunDir, cfg: Exp13Config) -> None:
    from scipy.stats import spearmanr

    recs = run.read_metrics()
    rep = results_root() / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    cand = ["r1", "r2", "dither", "nocov", "arx"]
    lines = [f"# Exp13 summary ({cfg.run_name}) - real system-identification rigs", "",
             f"Source: `{run.metrics_file}`. Protocol: docs/sysid_protocol.md. Exp3 artefacts: `{cfg.exp3_run}`.",
             "Ratios to zero-shot with covariates (< 1 = better): geometric mean over windows per benchmark, then over "
             "benchmarks; 95% CI from a bootstrap over windows within every benchmark. Reliance = gmean MAE(no "
             "covariate) / MAE(with covariate) of zero-shot (> 1: the inputs help).", ""]
    agg: dict = {}
    for L in cfg.contexts:
        cells = {r["bench"]: (r, dict(np.load(run.cell_path(r["key"]), allow_pickle=False)))
                 for r in recs if r["L"] == L}
        if not cells:
            continue
        logs = {}
        for b, (r, z) in cells.items():
            yt = z["y_true"].astype(np.float64)
            mae0 = np.abs(z["med_zeroshot"] - yt).mean(1)
            w0 = _wql_rows(z["q9_zeroshot"], yt)
            logs[b] = {m: np.log(np.abs(z[f"med_{m}"] - yt).mean(1) / mae0) for m in cand if f"med_{m}" in z}
            logs[b].update({f"{m}_wql": np.log(_wql_rows(z[f"q9_{m}"], yt) / w0) for m in cand if f"q9_{m}" in z})
        boots = {b: [rng.integers(0, len(cells[b][1]["y_true"]), len(cells[b][1]["y_true"])) for _ in range(cfg.n_boot)]
                 for b in cells}

        def gm(x):
            return float(np.exp(np.nanmean(x)))

        def ci(b, key):
            v = logs[b][key]
            s = [gm(v[ix]) for ix in boots[b]]
            return float(np.nanpercentile(s, 2.5)), float(np.nanpercentile(s, 97.5))

        def overall(key):
            bs = [b for b in cells if key in logs[b]]
            point = gm([np.nanmean(logs[b][key]) for b in bs])
            s = [gm([np.nanmean(logs[b][key][boots[b][t]]) for b in bs]) for t in range(cfg.n_boot)]
            return point, float(np.nanpercentile(s, 2.5)), float(np.nanpercentile(s, 97.5))

        lines += [f"## L = {L}", "", "### MAE ratio to zero-shot, per benchmark", "",
                  "| benchmark | windows | records | " + " | ".join(LABEL[m] for m in cand) + " |",
                  "|---|---|---|" + "---|" * len(cand)]
        agg[L] = {"bench": {}, "overall": {}}
        for b in [x for x in cfg.benches if x in cells]:
            r, z = cells[b]
            row, d = [], {}
            for m in cand:
                if m in logs[b]:
                    lo, hi = ci(b, m)
                    d[m] = {"mae": gm(logs[b][m]), "ci": [lo, hi], "wql": gm(logs[b][f"{m}_wql"])
                            if f"{m}_wql" in logs[b] else None}
                    row.append(f"{d[m]['mae']:.3f} [{lo:.3f}, {hi:.3f}]")
                else:
                    row.append("-")
            agg[L]["bench"][b] = d
            lines.append(f"| {b} | {r['n']} | {len(r['records'])} | " + " | ".join(row) + " |")
        lines += ["", "### Overall (geometric mean over benchmarks)", "",
                  "| method | MAE ratio [95% CI] | WQL ratio [95% CI] | benchmarks < 0.99 / > 1.01 (MAE) |",
                  "|---|---|---|---|"]
        for m in cand:
            if not any(m in logs[b] for b in cells):
                continue
            pm, lo, hi = overall(m)
            pw, lw, hw = overall(f"{m}_wql") if any(f"{m}_wql" in logs[b] for b in cells) else (np.nan,) * 3
            vals = [agg[L]["bench"][b][m]["mae"] for b in agg[L]["bench"] if m in agg[L]["bench"][b]]
            agg[L]["overall"][m] = {"mae": pm, "mae_ci": [lo, hi], "wql": pw, "wql_ci": [lw, hw]}
            lines.append(f"| {LABEL[m]} | {pm:.3f} [{lo:.3f}, {hi:.3f}] | {pw:.3f} [{lw:.3f}, {hw:.3f}] | "
                         f"{sum(v < 0.99 for v in vals)} / {sum(v > 1.01 for v in vals)} |")
        # P3: conditions
        rg, mem, per_g, nonper_g = [], [], [], []
        for b, (r, z) in cells.items():
            lr = logs[b]["r2"]
            for rid in np.unique(z["rec"]):
                sel = z["rec"] == rid
                rg.append(1 - gm(lr[sel]))
                mem.append(float(z["arx_memory"][sel][0]))
            per = z["periodicity"] >= cfg.periodic_thresh
            per_g += list(lr[per])
            nonper_g += list(lr[~per])
        ok = np.isfinite(mem)
        rho, p = spearmanr(np.asarray(mem)[ok], np.asarray(rg)[ok]) if ok.sum() >= 3 else (np.nan, np.nan)
        rel = {b: float(np.exp(-np.nanmean(logs[b]["nocov"]))) for b in cells}
        rel_txt = ", ".join(f"{b} {1 / v:.2f}" for b, v in rel.items())
        lines += ["", "### Conditions and sanity", "",
                  f"- P3 R2 gain (1 - record gmean MAE ratio) vs ARX memory over records: Spearman rho = {rho:.3f} "
                  f"(p = {p:.2g}, n = {int(ok.sum())})",
                  f"- P3 R2 MAE ratio in periodic windows (p >= {cfg.periodic_thresh}): "
                  f"{gm(per_g) if per_g else float('nan'):.3f} (n = {len(per_g)}); non-periodic: "
                  f"{gm(nonper_g) if nonper_g else float('nan'):.3f} (n = {len(nonper_g)})",
                  f"- P4 covariate reliance (MAE no covariate / with covariate): {rel_txt}", ""]
        agg[L]["p3"] = {"spearman": rho, "p": p, "periodic": gm(per_g) if per_g else None,
                        "nonperiodic": gm(nonper_g) if nonper_g else None, "n_periodic": len(per_g)}
        agg[L]["reliance"] = {b: 1 / v for b, v in rel.items()}
    # Sensitivity (not in the protocol; added after the run): some F-16 / WH / ParWH windows are silent (no
    # excitation, output ~ 0), where every MAE is ~1e-9 and ratios are meaningless (ARX ratio exactly 0 there).
    # Drop windows whose context or horizon std is < 1% of the record's std and recompute the overall ratios.
    try:
        meta = json.loads((INPUTS["data"] / "records.json").read_text(encoding="utf-8"))
        zr = np.load(INPUTS["data"] / "records.npz", allow_pickle=False)
        rstd = {m["i"]: float(np.std(zr[f"y{m['i']}"])) for m in meta}
    except FileNotFoundError:
        rstd = None
    if rstd is not None:
        lines += ["## Sensitivity (added after the run, not pre-declared): silent windows removed", "",
                  "Windows whose context or horizon std is below 1% of the record's std are dropped.", "",
                  "| L | method | MAE ratio [95% CI] | benchmarks < 0.99 / > 1.01 | windows dropped |", "|---|---|---|---|---|"]
        agg["sensitivity_silent_removed"] = {}
        for L in cfg.contexts:
            cells = {r["bench"]: dict(np.load(run.cell_path(r["key"]), allow_pickle=False)) for r in recs if r["L"] == L}
            if not cells:
                continue
            keep, logs, dropped = {}, {}, 0
            for b, z in cells.items():
                rs = np.asarray([rstd[int(i)] for i in z["rec"]])
                k = (z["y_ctx_std"] >= 0.01 * rs) & (z["y_true"].std(1) >= 0.01 * rs)
                dropped += int((~k).sum())
                yt = z["y_true"][k].astype(np.float64)
                mae0 = np.abs(z["med_zeroshot"][k] - yt).mean(1)
                logs[b] = {m: np.log(np.abs(z[f"med_{m}"][k] - yt).mean(1) / mae0) for m in cand}
                keep[b] = int(k.sum())
            for m in cand:
                bs = list(cells)
                pt = float(np.exp(np.mean([np.nanmean(logs[b][m]) for b in bs])))
                s = []
                for _ in range(cfg.n_boot):
                    s.append(float(np.exp(np.mean([np.nanmean(logs[b][m][rng.integers(0, keep[b], keep[b])])
                                                   for b in bs]))))
                per_b = [float(np.exp(np.nanmean(logs[b][m]))) for b in bs]
                lo, hi = np.nanpercentile(s, [2.5, 97.5])
                agg["sensitivity_silent_removed"][f"L{L}_{m}"] = {"mae": pt, "ci": [float(lo), float(hi)],
                                                                   "per_bench": dict(zip(bs, per_b))}
                lines.append(f"| {L} | {LABEL[m]} | {pt:.3f} [{lo:.3f}, {hi:.3f}] | {sum(v < 0.99 for v in per_b)} / "
                             f"{sum(v > 1.01 for v in per_b)} | {dropped} |")
        lines.append("")
    (rep / f"exp13_sysid_{cfg.run_name}.json").write_text(json.dumps(agg, indent=1, default=float), encoding="utf-8")
    out = rep / f"exp13_summary_{cfg.run_name}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("summary written: %s", out)


# ------------------------------------------------------------------------------------ main
def main() -> None:
    cfg, args = parse_args()
    run = RunDir("exp13", cfg.run_name)
    setup_logging(run.path / "log.txt")
    log.info("config: %s", json.dumps(asdict(cfg)))
    if args.summary_only:
        write_summary(run, cfg)
        return
    run.write_config(cfg, resume=args.resume, ignore_keys=("device", "batch_size"))
    from covattn.model import Chronos2Wrapper

    recs, _ = load_records()
    w = Chronos2Wrapper(cfg.model_id, device=cfg.device, amp=cfg.amp, batch_size=cfg.batch_size)
    torch.manual_seed(cfg.seed)
    methods, info = build_methods(w, cfg)
    (run.path / "methods.json").write_text(json.dumps({"methods": list(methods), **info}, indent=1))
    done = run.done_keys() if args.resume else set()
    jobs = [(b, L) for L in cfg.contexts for b in cfg.benches if f"sysid|{b}|L{L}" not in done]
    t_start = time.time()
    for i, (b, L) in enumerate(jobs):
        t0 = time.time()
        try:
            rec = run_cell(b, L, recs, w, methods, cfg, run)
        except ValueError as e:
            log.info("skip %s L=%d: %s", b, L, e)
            continue
        except Exception:  # noqa: BLE001
            log.exception("cell %s L=%d FAILED - continuing", b, L)
            if w.device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        rec["seconds"], rec["device"] = time.time() - t0, str(w.device)
        run.append_metrics(rec)
        el = time.time() - t_start
        log.info("[%d/%d] %s n=%d %.0fs ETA %s | %s", i + 1, len(jobs), rec["key"], rec["n"], rec["seconds"],
                 fmt_seconds(el / (i + 1) * (len(jobs) - i - 1)),
                 {m: round(v["mae_ratio_gmean"], 3) for m, v in rec["methods"].items()})
    write_summary(run, cfg)
    expected = {f"sysid|{b}|L{L}" for L in cfg.contexts for b in cfg.benches
                if windows([r for r in recs if r["bench"] == b], L, cfg.horizon, cfg.windows_per_bench)}
    if expected <= run.done_keys():
        run.mark_done({"cells": sorted(expected)})
        log.info("all %d cells done -> done.marker", len(expected))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        log.exception("Exp13 crashed")
        raise
