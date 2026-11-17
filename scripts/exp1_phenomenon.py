"""Exp1 - quantify covariate-effect attenuation of Chronos-2 on synthetic forced systems.

Grid: system config x context length L x input type; per cell N factual samples, each with
counterfactuals for every (shape, delta). One cell = one resumable sub-task.

  python scripts/exp1_phenomenon.py --quick
  python scripts/exp1_phenomenon.py --resume                      # full grid
  python scripts/exp1_phenomenon.py --resume --contexts 256 512 1024  # skip L=2048 locally
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from covattn import metrics as M
from covattn.baselines import arx_effects
from covattn.io_utils import RunDir, fmt_seconds, results_root, setup_logging
from covattn.model import CovBatch, time_estimates
from covattn.systems import (
    SampleSpec,
    SystemSpec,
    default_systems,
    make_counterfactual,
    make_rng,
    make_sample,
    quick_systems,
)

log = logging.getLogger("exp1")


@dataclass
class Exp1Config:
    run_name: str = "main"
    quick: bool = False
    systems: list[str] = field(default_factory=list)  # filled from the grid
    contexts: list[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    horizon: int = 64
    input_types: list[str] = field(default_factory=lambda: ["prbs", "multisine", "ar1", "steps"])
    n_per_cell: int = 32
    shapes: list[str] = field(default_factory=lambda: ["step", "pulse"])
    delta_mults: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    snr: float = 10.0
    bg_scale: float = 1.0
    eff_thresh: float = 0.05  # keep pairs with RMS(e) > eff_thresh * std(y_ctx)
    seed: int = 0
    split: str = "test"
    model_id: str = "amazon/chronos-2"
    amp: str = "off"
    batch_size: int = 128
    device: str = "auto"
    arx: bool = True
    n_examples_saved: int = 4


def parse_args() -> tuple[Exp1Config, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--contexts", type=int, nargs="+", default=None)
    p.add_argument("--n-per-cell", type=int, default=None)
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off")
    p.add_argument("--batch-size", type=int, default=128, help="rows (series) per forward pass; halves on OOM")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-arx", action="store_true")
    p.add_argument("--model-id", default="amazon/chronos-2")
    p.add_argument("--estimate-only", action="store_true", help="print time estimate and exit")
    p.add_argument("--summary-only", action="store_true", help="rebuild the summary from saved results (no model)")
    a = p.parse_args()

    cfg = Exp1Config(quick=a.quick, amp=a.amp, batch_size=a.batch_size, device=a.device, seed=a.seed,
                     arx=not a.no_arx, model_id=a.model_id)
    if a.quick:
        cfg.run_name = "quick"
        cfg.systems = [s.key for s in quick_systems()]
        cfg.contexts = [256, 512]
        cfg.input_types = ["prbs", "ar1"]
        cfg.n_per_cell = 8
        cfg.delta_mults = [1.0]
    else:
        cfg.systems = [s.key for s in default_systems()]
    if a.contexts is not None:
        cfg.contexts = a.contexts
    if a.n_per_cell is not None:
        cfg.n_per_cell = a.n_per_cell
    if a.run_name is not None:
        cfg.run_name = a.run_name
    return cfg, a


def system_lookup() -> dict[str, SystemSpec]:
    return {s.key: s for s in default_systems() + quick_systems()}


def cell_key(sys_key: str, L: int, inp: str) -> str:
    return f"{sys_key}|L{L}|{inp}"


def run_cell(wrapper, cfg: Exp1Config, sys: SystemSpec, L: int, inp: str, run: RunDir) -> dict:
    key = cell_key(sys.key, L, inp)
    spec = SampleSpec(context_length=L, horizon=cfg.horizon, input_type=inp, snr=cfg.snr, bg_scale=cfg.bg_scale)
    rng = make_rng(cfg.seed, cfg.split, "exp1", sys.key, L, inp)
    samples = [make_sample(sys, spec, rng) for _ in range(cfg.n_per_cell)]
    combos = [(sh, dm) for sh in cfg.shapes for dm in cfg.delta_mults]
    N, Mn, H = len(samples), len(combos), cfg.horizon

    e = np.empty((N, Mn, H))
    du = np.empty((N, Mn, H))
    tgt, cp, cf_list = [], [], []
    for s in samples:
        tgt.append(s.y[:L])
        cp.append(s.u[:L])
    fut_fac = [s.u[L:] for s in samples]
    for i, s in enumerate(samples):
        for m, (sh, dm) in enumerate(combos):
            c = make_counterfactual(s, sh, dm)
            e[i, m], du[i, m] = c.effect, c.du
            cf_list.append(c.u_future_cf)
    tgt_a, cp_a = np.stack(tgt), np.stack(cp)
    # one big batch: N factual tasks followed by N*M counterfactual tasks
    batch = CovBatch(
        target=np.concatenate([tgt_a, np.repeat(tgt_a, Mn, axis=0)]),
        cov_past=np.concatenate([cp_a, np.repeat(cp_a, Mn, axis=0)]),
        cov_future=np.concatenate([np.stack(fut_fac), np.stack(cf_list)]),
    )
    q = wrapper.predict(batch, H)  # (N + N*M, Q, H)
    q_fac = q[:N]
    q_cf = q[N:].reshape(N, Mn, q.shape[1], H)
    mi = wrapper.median_idx
    e_hat = q_cf[:, :, mi] - q_fac[:, None, mi]
    e_hat_mean = q_cf.mean(2) - q_fac.mean(1)[:, None]

    y_true = np.stack([s.y[L:] for s in samples])
    y_scale = tgt_a.std(-1)
    if cfg.arx:
        e_arx, fits = arx_effects(tgt_a, cp_a, du)
        arx_orders = np.array([[f.na, f.nb] for f in fits])
        arx_stable = np.array([f.stable() for f in fits])
    else:
        e_arx = np.full_like(e, np.nan)
        arx_orders = np.zeros((N, 2), dtype=int)
        arx_stable = np.zeros(N, dtype=bool)

    ne = min(cfg.n_examples_saved, N)
    run.save_arrays(
        key,
        e=e.astype(np.float32), du=du.astype(np.float32), e_hat=e_hat.astype(np.float32),
        e_hat_mean=e_hat_mean.astype(np.float32), e_arx=e_arx.astype(np.float32),
        q_fac=q_fac.astype(np.float32), y_true=y_true.astype(np.float32), y_scale=y_scale.astype(np.float32),
        arx_orders=arx_orders, arx_stable=arx_stable,
        combo_shape=np.array([c[0] for c in combos]), combo_delta=np.array([c[1] for c in combos]),
        ex_y_ctx=tgt_a[:ne].astype(np.float32), ex_u=np.stack([s.u for s in samples[:ne]]).astype(np.float32),
        b=np.array([s.b for s in samples]),
    )

    rec: dict = {"key": key, "system": sys.key, "family": sys.family, "sys_params": sys.to_dict(), "L": L,
                 "input": inp, "n_samples": N, "per_combo": []}
    point = q_fac[:, mi]
    rec["mase_fac"] = M.summarize(M.mase(point, y_true, tgt_a))
    rec["wql_fac_pooled"] = M.pooled_wql(q_fac, wrapper.quantiles, y_true)
    for m, (sh, dm) in enumerate(combos):
        keep = M.effect_mask(e[:, m], y_scale, cfg.eff_thresh)
        ek, eh, dk, ea = e[keep, m], e_hat[keep, m], du[keep, m], e_arx[keep, m]
        stab = arx_stable[keep]
        rec["per_combo"].append({
            "shape": sh, "delta_mult": dm, "n_kept": int(keep.sum()),
            "gain": M.summarize(M.effect_gain(eh, ek)),
            "gain_meanq": M.summarize(M.effect_gain(e_hat_mean[keep, m], ek)),
            "shape_r": M.summarize(M.shape_corr(eh, ek)),
            "nrmse": M.summarize(M.whatif_nrmse(eh, ek)),
            "memless_r2_model": M.summarize(M.memoryless_r2(eh, dk)),
            "memless_r2_true": M.summarize(M.memoryless_r2(ek, dk)),
            "gain_curve": M.gain_curve(eh, ek) if keep.any() else [],
            "arx_gain": M.summarize(M.effect_gain(ea, ek)),
            "arx_gain_stable_only": M.summarize(M.effect_gain(ea[stab], ek[stab])),
            "arx_nrmse": M.summarize(M.whatif_nrmse(ea, ek)),
        })
    return rec


# -------------------------------------------------------------------------------------- summary
def write_summary(run: RunDir, cfg: Exp1Config) -> None:
    import pandas as pd

    rows = []
    for rec in run.read_metrics():
        for pc in rec["per_combo"]:
            rows.append({"system": rec["system"], "family": rec["family"], "L": rec["L"], "input": rec["input"],
                         "shape": pc["shape"], "delta_mult": pc["delta_mult"], "key": rec["key"]})
    if not rows:
        return
    # pool per-sample gains from the saved arrays (exact medians, not medians of medians)
    pooled: dict[tuple, dict[str, list]] = {}
    for rec in run.read_metrics():
        z = np.load(run.cell_path(rec["key"]))
        for m, (sh, dm) in enumerate(zip(z["combo_shape"], z["combo_delta"])):
            keep = M.effect_mask(z["e"][:, m], z["y_scale"], cfg.eff_thresh)
            ek, eh, ea, dk = z["e"][keep, m], z["e_hat"][keep, m], z["e_arx"][keep, m], z["du"][keep, m]
            ms = "multisine" if rec["input"] == "multisine" else "non-multisine"
            for gk in [(rec["system"], rec["L"], str(sh)), ("FAMILY:" + rec["family"], rec["L"], str(sh)),
                       ("DELTA", float(dm), str(sh)), ("INPUT:" + rec["input"], rec["L"], str(sh)),
                       (f"FAMILY[{ms}]:" + rec["family"], rec["L"], str(sh))]:
                d = pooled.setdefault(gk, {"g": [], "r": [], "nrmse": [], "arx": [], "mlm": [], "mlt": []})
                d["g"] += M.effect_gain(eh, ek).tolist()
                d["r"] += M.shape_corr(eh, ek).tolist()
                d["nrmse"] += M.whatif_nrmse(eh, ek).tolist()
                d["arx"] += M.effect_gain(ea, ek).tolist()
                d["mlm"] += M.memoryless_r2(eh, dk).tolist()
                d["mlt"] += M.memoryless_r2(ek, dk).tolist()

    table = []
    for (grp, L, sh), d in pooled.items():
        g, r, nr, ax = (M.summarize(np.array(d[k])) for k in ("g", "r", "nrmse", "arx"))
        mlm, mlt = M.summarize(np.array(d["mlm"])), M.summarize(np.array(d["mlt"]))
        table.append({"group": grp, "L_or_delta": L, "shape": sh, "n": g["n"], "g_median": g["median"],
                      "g_q25": g["q25"], "g_q75": g["q75"], "shape_r_median": r["median"],
                      "nrmse_median": nr["median"], "arx_g_median": ax["median"],
                      "memless_r2_model": mlm["median"], "memless_r2_true": mlt["median"]})
    df = pd.DataFrame(table)
    rep_dir = results_root() / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    csv = run.path / "summary_table.csv"
    df.to_csv(csv, index=False)

    def md(sub: pd.DataFrame) -> str:
        cols = ["group", "L_or_delta", "shape", "n", "g_median", "g_q25", "g_q75", "shape_r_median",
                "nrmse_median", "arx_g_median", "memless_r2_model", "memless_r2_true"]
        sub = sub[cols].sort_values(["group", "shape", "L_or_delta"])
        head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        body = ""
        for _, rr in sub.iterrows():
            body += "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in rr.values) + " |\n"
        return head + body

    fam = df[df.group.str.startswith("FAMILY:")]
    fam_split = df[df.group.str.startswith("FAMILY[")]
    inp = df[df.group.str.startswith("INPUT:")]
    dl = df[df.group == "DELTA"]
    syst = df[~df.group.str.contains(":") & (df.group != "DELTA")]
    txt = [f"# Exp1 summary ({cfg.run_name})", "",
           f"Generated from `{run.metrics_file}` and per-cell arrays. Config: `{run.path / 'config.json'}`.",
           f"Pairs kept if RMS(e) > {cfg.eff_thresh} * std(y_ctx). g = <e_hat, e>/||e||^2 with e_hat = q50_cf - q50_fac.",
           "Pooled over input types and delta multipliers unless stated.", "",
           "## By family x L", "", md(fam), "", "## By input type x L", "", md(inp), "",
           "## By family x L, split multisine / non-multisine inputs", "", md(fam_split), "",
           "## Linearity check (pooled over systems and L)", "", md(dl), "",
           "## By system x L", "", md(syst)]
    out = rep_dir / f"exp1_summary_{cfg.run_name}.md"
    out.write_text("\n".join(txt), encoding="utf-8")
    log.info("summary written: %s and %s", out, csv)


# -------------------------------------------------------------------------------------- main
def main() -> None:
    cfg, args = parse_args()
    run = RunDir("exp1", cfg.run_name)
    setup_logging(run.path / "log.txt")
    log.info("config: %s", json.dumps(asdict(cfg)))
    if args.summary_only:
        write_summary(run, cfg)
        return
    if not args.estimate_only:
        run.write_config(cfg, resume=args.resume, ignore_keys=("contexts", "batch_size", "device"))

    from covattn.model import Chronos2Wrapper

    wrapper = Chronos2Wrapper(cfg.model_id, device=cfg.device, amp=cfg.amp, batch_size=cfg.batch_size)
    log.info("device=%s amp=%s quantiles=%s", wrapper.device, cfg.amp, wrapper.quantiles)

    lookup = system_lookup()
    cells = [(lookup[s], L, inp) for L in cfg.contexts for s in cfg.systems for inp in cfg.input_types]
    done = run.done_keys() if args.resume else set()
    todo = [c for c in cells if cell_key(c[0].key, c[1], c[2]) not in done]
    log.info("cells: %d total, %d done, %d to run", len(cells), len(cells) - len(todo), len(todo))

    # time estimate from a throughput probe per context length
    tasks_per_cell = cfg.n_per_cell * (1 + len(cfg.shapes) * len(cfg.delta_mults))
    est = 0.0
    for L in sorted({c[1] for c in todo}):
        tps = wrapper.measure_throughput(L, cfg.horizon, k=1, n_tasks=256)
        n_cells = sum(1 for c in todo if c[1] == L)
        est += n_cells * tasks_per_cell / tps
        log.info("throughput L=%d: %.0f tasks/s on %s", L, tps, wrapper.device)
    est_arx = len(todo) * cfg.n_per_cell * 0.03 if cfg.arx else 0.0  # ~30 ms/sample measured order
    te = time_estimates(est, wrapper.device)
    log.info("ESTIMATED TIME (model) - measured on %s: %s | RTX4060: %s | L4: %s | A100(guess): %s ; + ARX on CPU ~%s", wrapper.device, fmt_seconds(te["here"]),
             fmt_seconds(te["RTX4060"]), fmt_seconds(te["L4"]), fmt_seconds(te["A100"]), fmt_seconds(est_arx))
    if args.estimate_only:
        return

    t0 = time.time()
    for i, (sys, L, inp) in enumerate(todo):
        tc = time.time()
        rec = run_cell(wrapper, cfg, sys, L, inp, run)
        rec["seconds"] = time.time() - tc
        rec["batch_size_rows"] = wrapper.batch_size
        rec["device"] = str(wrapper.device)
        run.append_metrics(rec)
        g_step = next((pc["gain"]["median"] for pc in rec["per_combo"] if pc["shape"] == "step"), float("nan"))
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(todo) - i - 1)
        log.info("[%d/%d] %s  g_step(median)=%.3f  %.1fs  ETA %s", i + 1, len(todo), rec["key"], g_step,
                 rec["seconds"], fmt_seconds(eta))
    if wrapper.oom_events:
        (run.path / "oom_events.json").write_text(json.dumps(wrapper.oom_events, indent=1), encoding="utf-8")
    write_summary(run, cfg)
    # done.marker only when the canonical grid (all four L for the full run) is complete
    canon_L = cfg.contexts if cfg.quick else Exp1Config().contexts
    canon = {cell_key(s, L, inp) for L in canon_L for s in cfg.systems for inp in cfg.input_types}
    missing = canon - run.done_keys()
    if not missing:
        run.mark_done({"n_cells": len(canon), "time": time.ctime()})
        log.info("all %d cells done -> done.marker", len(canon))
    else:
        log.info("partial run: %d/%d canonical cells missing (e.g. %s)", len(missing), len(canon), sorted(missing)[0])


if __name__ == "__main__":
    main()
