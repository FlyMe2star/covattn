"""Exp12 - what-if on a real pricing trial (Low Carbon London dToU 2013), protocol docs/lcl_protocol.md.

The price signal was set by the experimenters and a flat-tariff group serves as control, so the true effect of
the price on the ToU group's load is estimated without any forecasting model (scripts/lcl_feasibility.py:
multiplicative difference-in-differences, placebo noise level). Here the forecasting models predict the same
effect: origin 00:00 of every 2013 event day, horizon 48 half-hours (prices were announced a day ahead), known
covariate = price in p/kWh. Factual run = actual prices, counterfactual run = all Normal;
predicted effect ê = q50(factual) - q50(counterfactual).

Truth runs (from lcl_feasibility.py, in $RESULTS_DIR/lcl/):
  primary     feasibility_shift+1_bst-2  (protocol amendment 1: clock alignment)
  unshifted   feasibility                (sensitivity)
Methods: zero-shot, zero-shot without covariate (accuracy only), R1, R2, dither (all from exp3/<run>, nothing tuned
on LCL); --ft adds full fine-tune / LoRA from exp3/<run> as references.

Metrics (fixed in the protocol): calibration slope beta = sum<ê,e>/sum||ê||^2 (1 = right size, > 1 = attenuated),
noise-debiased gain g = sum<ê,e>/(sum||e||^2 - N * placebo mean ||e||^2), both with 95% block-bootstrap CIs over
weeks; per level (High / Low) mean relative effect over the event slots vs DiD, sign agreement, Spearman across
event segments; factual MAE / WQL on event days and their ratio to zero-shot.

  python scripts/exp12_lcl_whatif.py --quick
  python scripts/exp12_lcl_whatif.py --resume            # full run (minutes on one GPU)
  python scripts/exp12_lcl_whatif.py --summary-only
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from covattn.baselines import dither_predict
from covattn.io_utils import RunDir, fmt_seconds, results_root, setup_logging
from covattn.model import CovBatch
from covattn.repair import gain_spec, r1_gain

log = logging.getLogger("exp12")
SPD = 48
NORMAL_PRICE = 11.76
LEVELS9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TRUTHS = {"primary": "feasibility_shift+1_bst-2", "unshifted": "feasibility"}
LABEL = {"zeroshot": "Zero-shot", "nocov": "Zero-shot, no covariate", "r1": "R1", "r2": "R2",
         "dither": "Dither (reimpl.)", "ft_full": "Full fine-tune", "ft_lora": "LoRA"}
TARIFF_CODE = {"Normal": 0, "High": 1, "Low": 2}
INPUTS: dict = {"root": None}  # where truth runs and exp3 artefacts are read from (--inputs-root)


@dataclass
class Exp12Config:
    run_name: str = "main"
    quick: bool = False
    exp3_run: str = "v2"
    truths: list[str] = field(default_factory=lambda: list(TRUTHS))
    contexts: list[int] = field(default_factory=lambda: [512, 2048])
    horizon: int = SPD
    max_origins: int = 0  # 0 = all event days
    dither_K: int = 8
    use_ft: bool = False
    n_boot: int = 2000
    seed: int = 0
    model_id: str = "amazon/chronos-2"
    amp: str = "off"
    device: str = "auto"
    batch_size: int = 128


def parse_args() -> tuple[Exp12Config, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--exp3-run", default=None)
    p.add_argument("--ft", action="store_true", help="add full fine-tune / LoRA from exp3 as references")
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--inputs-root", default=None,
                   help="read lcl/ truth runs and exp3/ artefacts from here (default: RESULTS_DIR); lets a local "
                        "--quick write to ./results while reading the Drive inputs")
    a = p.parse_args()
    INPUTS["root"] = Path(a.inputs_root) if a.inputs_root else results_root()
    cfg = Exp12Config(quick=a.quick, amp=a.amp, device=a.device, batch_size=a.batch_size, use_ft=a.ft)
    if a.quick:
        cfg.run_name, cfg.contexts, cfg.max_origins, cfg.dither_K, cfg.n_boot = "quick", [512], 12, 2, 200
    if a.run_name is not None:
        cfg.run_name = a.run_name
    if a.exp3_run is not None:
        cfg.exp3_run = a.exp3_run
    return cfg, a


# ------------------------------------------------------------------------------------ data
def load_truth(name: str) -> dict:
    d = INPUTS["root"] / "lcl" / TRUTHS[name]
    full = pd.read_parquet(d / "group_series_full.parquet")
    s13 = pd.read_parquet(d / "series_2013.parquet")
    rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
    return {"dir": d, "full": full, "s13": s13, "noise_sq": float(rep["day_profile"]["mean_sq_placebo"]),
            "shift": rep.get("tariff_shift_applied", {"all": 0, "bst_extra": 0})}


def origins(tr: dict, L: int, H: int, max_n: int) -> list[int]:
    """Grid indices of 00:00 of every 2013 event day with a complete truth and a full context window."""
    full, s13 = tr["full"], tr["s13"]
    t_full = pd.DatetimeIndex(full["t"])
    e = s13["effect_kwh"].to_numpy().reshape(-1, SPD)
    ev = s13["event_day"].to_numpy().reshape(-1, SPD)[:, 0]
    days = pd.DatetimeIndex(s13["t"]).normalize()[::SPD]
    out = []
    for d, ok_ev, ed in zip(days, ev, e):
        if not ok_ev or not np.isfinite(ed).all():
            continue
        o = int(np.searchsorted(t_full, d))
        if o - L < 0 or o + H > len(t_full):
            continue
        out.append(o)
    return out[:max_n] if max_n else out


# ------------------------------------------------------------------------------------ methods
def _remap(path: str) -> str:
    """exp3 stores Colab paths; locally the same Drive folder is under results_root()/../.."""
    if Path(path).exists():
        return path
    pre = "/content/drive/MyDrive/"
    if path.startswith(pre):
        alt = INPUTS["root"].parents[1] / path[len(pre):]
        if alt.exists():
            return str(alt)
    return path


def build_methods(w, cfg: Exp12Config) -> tuple[dict, dict]:
    d3 = INPUTS["root"] / "exp3" / cfg.exp3_run
    r1 = json.loads((d3 / "r1_select.json").read_text())
    k, gamma = r1["best"]["k"], r1["best"]["gamma"]
    heads = [tuple(x) for x in r1["ranking"][:k]]
    hk = w.hooked
    G1 = r1_gain(hk.n_layers, hk.n_heads, heads, gamma)
    gates = torch.tensor(json.loads((d3 / "r2_gates.json").read_text())["gates"])
    sg = json.loads((d3 / "dither_select.json").read_text())["best"]["sigma"]
    methods = {"zeroshot": lambda bt, s: w.predict(bt, bt.H),
               "r1": lambda bt, s: w.predict_hooked(bt, gain_spec(G1)),
               "r2": lambda bt, s: w.predict_hooked(bt, gain_spec(gates)),
               "dither": lambda bt, s: dither_predict(w, bt, sg, cfg.dither_K, seed=s)}
    info = {"k": k, "gamma": gamma, "dither_sigma": sg, "exp3_dir": str(d3)}
    if cfg.use_ft:
        from covattn.model import Chronos2Wrapper

        for mode in ("full", "lora"):
            f = d3 / f"ft_{mode}" / "info.json"
            if f.exists():
                path = _remap(json.loads(f.read_text())["path"])
                wf = Chronos2Wrapper(path, device=str(w.device), amp=cfg.amp, batch_size=cfg.batch_size)
                methods[f"ft_{mode}"] = lambda bt, s, wf=wf: wf.predict(bt, bt.H)
                info[f"ft_{mode}_path"] = path
    return methods, info


# ------------------------------------------------------------------------------------ one cell
def run_cell(truth: str, L: int, w, methods: dict, cfg: Exp12Config, run: RunDir, cache: dict) -> dict:
    if truth not in cache:
        cache[truth] = load_truth(truth)
    tr = cache[truth]
    H = cfg.horizon
    full, s13 = tr["full"], tr["s13"]
    y_all = full["tou_kwh"].to_numpy(np.float64)
    p_all = full["price"].to_numpy(np.float64)
    tar_all = full["tariff"].to_numpy()
    t_full = pd.DatetimeIndex(full["t"])
    t13 = pd.DatetimeIndex(s13["t"])
    e13, cf13 = s13["effect_kwh"].to_numpy(), s13["cf_kwh"].to_numpy()
    os_ = origins(tr, L, H, cfg.max_origins)
    j13 = [int(np.searchsorted(t13, t_full[o])) for o in os_]
    y = np.stack([y_all[o - L:o] for o in os_])
    pc = np.stack([p_all[o - L:o] for o in os_])
    pf = np.stack([p_all[o:o + H] for o in os_])
    arrays = {
        "origin": np.asarray([t_full[o].value for o in os_], np.int64),
        "y_true": np.stack([y_all[o:o + H] for o in os_]).astype(np.float32),
        "e_true": np.stack([e13[j:j + H] for j in j13]).astype(np.float32),
        "cf_true": np.stack([cf13[j:j + H] for j in j13]).astype(np.float32),
        "tariff_fut": np.stack([[TARIFF_CODE[v] for v in tar_all[o:o + H]] for o in os_]).astype(np.int8),
        "ctx_nan_share": np.isnan(y).mean(1).astype(np.float32),
    }
    fac = CovBatch(y, pc, pf)
    cfb = CovBatch(y, pc, np.full_like(pf, NORMAL_PRICE))
    lv_idx = [int(np.argmin(np.abs(np.asarray(w.quantiles) - l))) for l in LEVELS9]
    mi = w.median_idx
    rec = {"key": f"lcl|{truth}|L{L}", "truth": truth, "truth_dir": str(tr["dir"]), "L": L, "n": len(os_),
           "noise_sq": tr["noise_sq"], "shift": tr["shift"], "methods": {}}
    for m, fn in methods.items():
        q_f = fn(fac, cfg.seed + 17)
        q_c = fn(cfb, cfg.seed + 17)  # same seed: common random numbers for dither
        arrays[f"q9_{m}"] = q_f[:, lv_idx].astype(np.float32)
        arrays[f"med_fac_{m}"] = q_f[:, mi].astype(np.float32)
        arrays[f"med_cf_{m}"] = q_c[:, mi].astype(np.float32)
        eh = q_f[:, mi] - q_c[:, mi]
        rec["methods"][m] = cell_stats(eh, arrays["e_true"], tr["noise_sq"])
    q_n = w.predict(CovBatch(y, pc[:, None, :][:, :0], pf[:, None, :][:, :0]), H)
    arrays["q9_nocov"] = q_n[:, lv_idx].astype(np.float32)
    arrays["med_fac_nocov"] = q_n[:, mi].astype(np.float32)
    run.save_arrays(rec["key"], **arrays)
    return rec


def cell_stats(eh: np.ndarray, e: np.ndarray, noise_sq: float) -> dict:
    num = float((eh * e).sum())
    return {"beta": num / max(float((eh ** 2).sum()), 1e-12),
            "g_debiased": num / max(float((e ** 2).sum()) - len(e) * noise_sq, 1e-12),
            "g_naive": num / max(float((e ** 2).sum()), 1e-12)}


# ------------------------------------------------------------------------------------ summary
def _segments(tariff_fut: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """(row, level code, slot mask) for every day x level with at least one slot of that level."""
    out = []
    for i, row in enumerate(tariff_fut):
        for code in (1, 2):
            m = row == code
            if m.any():
                out.append((i, code, m))
    return out


def _wql(q9: np.ndarray, y: np.ndarray) -> float:
    lv = np.asarray(LEVELS9)[None, :, None]
    yy = y[:, None, :]
    loss = 2.0 * np.abs((yy - q9) * ((yy <= q9).astype(np.float64) - lv))
    return float(loss.sum() / len(LEVELS9) / np.abs(y).sum())


def cell_summary(z: dict, noise_sq: float, n_boot: int, seed: int) -> dict:
    e, cf_t, yt = z["e_true"].astype(np.float64), z["cf_true"].astype(np.float64), z["y_true"].astype(np.float64)
    weeks = pd.DatetimeIndex(z["origin"]).isocalendar().week.to_numpy()
    uw = np.unique(weeks)
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([np.flatnonzero(weeks == w_) for w_ in rng.choice(uw, len(uw))]) for _ in range(n_boot)]
    segs = _segments(z["tariff_fut"])
    true_rel = np.asarray([e[i][m].mean() / cf_t[i][m].mean() for i, _, m in segs])
    lv = np.asarray([c for _, c, _ in segs])
    methods = [k[len("med_fac_"):] for k in z if k.startswith("med_fac_")]
    out = {"n": len(e), "n_weeks": len(uw), "n_segments": len(segs),
           "truth_rel": {"High": float(true_rel[lv == 1].mean()), "Low": float(true_rel[lv == 2].mean())},
           "methods": {}}
    mae0 = np.abs(z["med_fac_zeroshot"] - yt).mean()
    wql0 = _wql(z["q9_zeroshot"], yt)
    for m in methods:
        r = {"mae": float(np.abs(z[f"med_fac_{m}"] - yt).mean()), "wql": _wql(z[f"q9_{m}"], yt)}
        r["mae_ratio"], r["wql_ratio"] = r["mae"] / mae0, r["wql"] / wql0
        if f"med_cf_{m}" in z:
            eh = z[f"med_fac_{m}"].astype(np.float64) - z[f"med_cf_{m}"].astype(np.float64)
            st = cell_stats(eh, e, noise_sq)
            bs = np.asarray([[v for v in cell_stats(eh[b], e[b], noise_sq).values()] for b in boots])
            for j, kname in enumerate(st):
                r[kname] = st[kname]
                r[f"{kname}_ci"] = [float(np.percentile(bs[:, j], 2.5)), float(np.percentile(bs[:, j], 97.5))]
            cf_m = z[f"med_cf_{m}"].astype(np.float64)
            pred_rel = np.asarray([eh[i][mk].mean() / cf_m[i][mk].mean() for i, _, mk in segs])
            r["pred_rel"] = {"High": float(pred_rel[lv == 1].mean()), "Low": float(pred_rel[lv == 2].mean())}
            r["sign_agree"] = float((np.sign(pred_rel) == np.sign(true_rel)).mean())
            from scipy.stats import spearmanr

            rho, p = spearmanr(pred_rel, true_rel)
            r["spearman_segments"], r["spearman_p"] = float(rho), float(p)
        out["methods"][m] = r
    return out


def write_summary(run: RunDir, cfg: Exp12Config) -> None:
    recs = run.read_metrics()
    rep = results_root() / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    lines = [f"# Exp12 summary ({cfg.run_name}) - LCL dToU pricing trial, what-if vs model-free DiD truth", "",
             f"Source: `{run.metrics_file}`. Protocol: docs/lcl_protocol.md. Exp3 artefacts: `{cfg.exp3_run}`.",
             "beta = sum<ê,e>/sum||ê||^2 (1 = right size, >1 = attenuated); g = sum<ê,e>/(sum||e||^2 - N*noise) "
             "(1 = right size, <1 = attenuated); g naive = without noise correction (biased low). 95% CIs: block "
             "bootstrap over ISO weeks. rel. effect = mean over High / Low segments of mean effect / mean "
             "counterfactual load over the segment's slots. MAE / WQL: factual forecast on event days.", ""]
    agg = {}
    for r in sorted(recs, key=lambda r: (list(TRUTHS).index(r["truth"]), r["L"])):
        z = dict(np.load(run.cell_path(r["key"]), allow_pickle=False))
        s = cell_summary(z, r["noise_sq"], cfg.n_boot, cfg.seed)
        agg[r["key"]] = s
        lines += [f"## truth = {r['truth']} ({TRUTHS[r['truth']]}), L = {r['L']} ({s['n']} event days, "
                  f"{s['n_weeks']} weeks, {s['n_segments']} event segments)", "",
                  f"DiD relative effect: High {s['truth_rel']['High']:+.3f}, Low {s['truth_rel']['Low']:+.3f}", "",
                  "| method | beta [95% CI] | g [95% CI] | g naive | rel. High | rel. Low | sign agree | Spearman | "
                  "MAE ratio | WQL ratio |", "|---|---|---|---|---|---|---|---|---|---|"]
        for m in [k for k in LABEL if k in s["methods"]]:
            v = s["methods"][m]
            if "beta" in v:
                lines.append(f"| {LABEL[m]} | {v['beta']:.2f} [{v['beta_ci'][0]:.2f}, {v['beta_ci'][1]:.2f}] | "
                             f"{v['g_debiased']:.2f} [{v['g_debiased_ci'][0]:.2f}, {v['g_debiased_ci'][1]:.2f}] | "
                             f"{v['g_naive']:.2f} | {v['pred_rel']['High']:+.3f} | {v['pred_rel']['Low']:+.3f} | "
                             f"{v['sign_agree']:.2f} | {v['spearman_segments']:.2f} | {v['mae_ratio']:.3f} | "
                             f"{v['wql_ratio']:.3f} |")
            else:
                lines.append(f"| {LABEL[m]} | - | - | - | - | - | - | - | {v['mae_ratio']:.3f} | {v['wql_ratio']:.3f} |")
        lines.append("")
    (rep / f"exp12_lcl_{cfg.run_name}.json").write_text(json.dumps(agg, indent=1), encoding="utf-8")
    out = rep / f"exp12_summary_{cfg.run_name}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("summary written: %s", out)


# ------------------------------------------------------------------------------------ main
def main() -> None:
    cfg, args = parse_args()
    run = RunDir("exp12", cfg.run_name)
    setup_logging(run.path / "log.txt")
    log.info("config: %s", json.dumps(asdict(cfg)))
    if args.summary_only:
        write_summary(run, cfg)
        return
    run.write_config(cfg, resume=args.resume, ignore_keys=("device", "batch_size"))
    from covattn.model import Chronos2Wrapper

    w = Chronos2Wrapper(cfg.model_id, device=cfg.device, amp=cfg.amp, batch_size=cfg.batch_size)
    torch.manual_seed(cfg.seed)
    methods, info = build_methods(w, cfg)
    (run.path / "methods.json").write_text(json.dumps({"methods": list(methods), **info}, indent=1))
    log.info("methods: %s %s", list(methods), info)
    done = run.done_keys() if args.resume else set()
    jobs = [(t, L) for t in cfg.truths for L in cfg.contexts if f"lcl|{t}|L{L}" not in done]
    cache: dict = {}
    t_start = time.time()
    for i, (truth, L) in enumerate(jobs):
        t0 = time.time()
        try:
            rec = run_cell(truth, L, w, methods, cfg, run, cache)
        except Exception:  # noqa: BLE001 - report and continue
            log.exception("cell %s L=%d FAILED - continuing", truth, L)
            if w.device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        rec["seconds"] = time.time() - t0
        rec["device"] = str(w.device)
        run.append_metrics(rec)
        el = time.time() - t_start
        log.info("[%d/%d] %s n=%d %.0fs ETA %s | %s", i + 1, len(jobs), rec["key"], rec["n"], rec["seconds"],
                 fmt_seconds(el / (i + 1) * (len(jobs) - i - 1)),
                 {m: round(v["beta"], 2) for m, v in rec["methods"].items()})
    write_summary(run, cfg)
    n_all = len(cfg.truths) * len(cfg.contexts)
    if len(run.done_keys()) >= n_all:
        run.mark_done({"n": n_all})
        log.info("all %d cells done -> done.marker", n_all)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        log.exception("Exp12 crashed")
        raise
