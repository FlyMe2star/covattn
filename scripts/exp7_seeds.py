"""Exp7 - seed robustness of the Exp3 results.

  * R2 gates trained with training seeds {0 (= Exp3 run), 1, 2} (same config as Exp3, only the seed of the
    training-pair stream changes);
  * the synthetic test set re-drawn with sample seeds {0, 1, 2} (seed 0 = the Exp3 samples);
  * methods per sample seed: zero-shot, R1 (Exp3 choice; deterministic, no training seed), R2 x 3 training seeds,
    dither (its noise seed follows the sample seed).
Fine-tuning baselines keep one seed (Exp3). Summary: per method mean +- sd over seeds, and a cluster bootstrap
(resampling test cells, 1000 reps) 95% CI for the pooled median g and error ratio.

  python scripts/exp7_seeds.py --quick
  python scripts/exp7_seeds.py --resume --exp3-run v2      # Colab
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch

from covattn import metrics as M
from covattn.baselines import dither_predict
from covattn.io_utils import RunDir, fmt_seconds, results_root, setup_logging
from covattn.latex import rows_to_tex
from covattn.repair import R2Config, gain_spec, r1_gain, random_pairs, train_r2

import exp3_repair as E3  # noqa: E402  (scripts/ is on sys.path when this file is run)

log = logging.getLogger("exp7")


@dataclass
class Exp7Config:
    run_name: str = "main"
    quick: bool = False
    exp3_run: str = "v2"
    train_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    sample_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    contexts: list[int] = field(default_factory=lambda: [512, 2048])
    n_boot: int = 1000
    amp: str = "off"
    device: str = "auto"
    batch_size: int = 128


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--exp3-run", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    cfg = Exp7Config(quick=a.quick, device=a.device, batch_size=a.batch_size)
    if a.quick:
        cfg.run_name, cfg.exp3_run, cfg.contexts, cfg.n_boot = "quick", "quick", [256], 50
    if a.run_name is not None:
        cfg.run_name = a.run_name
    if a.exp3_run is not None:
        cfg.exp3_run = a.exp3_run
    return cfg, a


def exp3_config(cfg: Exp7Config) -> "E3.Exp3Config":
    d3 = results_root() / "exp3" / cfg.exp3_run
    c = json.loads((d3 / "config.json").read_text())["config"]
    c3 = E3.Exp3Config(**{k: v for k, v in c.items() if k in E3.Exp3Config.__dataclass_fields__})
    return replace(c3, contexts=list(cfg.contexts), device=cfg.device, batch_size=cfg.batch_size)


def r2_gates(w, cfg: Exp7Config, c3, seed: int, run: RunDir) -> torch.Tensor:
    d3 = results_root() / "exp3" / cfg.exp3_run
    if seed == c3.seed:
        return torch.tensor(json.loads((d3 / "r2_gates.json").read_text())["gates"])
    f = run.path / f"r2_gates_seed{seed}.json"
    if f.exists():
        return torch.tensor(json.loads(f.read_text())["gates"])
    lk = E3.systems_by_key()
    sys_ = [lk[k] for k in c3.train_systems]
    mf = run.path / f"r2_train_seed{seed}.jsonl"

    def batch_fn(step: int):
        return random_pairs(sys_, c3.input_types, c3.shapes, c3.train_deltas, c3.train_contexts, c3.r2_pairs,
                            c3.horizon, seed, "train", f"exp3-r2-{step}")

    def cb(m: dict) -> None:
        with mf.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(m) + "\n")
        if m["step"] % 100 == 0:
            log.info("R2 seed %d step %d: %s", seed, m["step"], {k: round(v, 4) for k, v in m.items() if k != "step"})

    t0 = time.time()
    E3.LAM[0] = c3.lam
    g = train_r2(w, R2Config(steps=c3.r2_steps, lr=c3.r2_lr, pairs_per_step=c3.r2_pairs, lam=c3.lam, reg=c3.r2_reg),
                 batch_fn, run.path / f"r2_ckpt_seed{seed}.pt", cb)
    f.write_text(json.dumps({"gates": g.tolist(), "train_seconds": time.time() - t0, "seed": seed}))
    log.info("R2 seed %d trained in %s: mean %.3f min %.3f max %.3f", seed, fmt_seconds(time.time() - t0),
             float(g.mean()), float(g.min()), float(g.max()))
    return g


# ------------------------------------------------------------------------------------ summary
TAB_LABEL = {"zeroshot": "Zero-shot", "r1": "R1", "dither": "Dither (reimpl.)",
             **{f"r2_s{k}": f"R2, training seed {k}" for k in range(10)}}


def _pooled(cs: list[dict], m: str, shapes: list[str]) -> tuple[float, float]:
    gs, er = [], []
    for c in cs:
        for sh in shapes:
            k = c[f"keep_{sh}"]
            gs.append(M.effect_gain(c[f"ehat_{m}_{sh}"][k], c[f"e_{sh}"][k]))
        if f"qfac_{m}" in c:
            yt = c["y_true"]
            er.append(np.abs(c[f"qfac_{m}"][:, E3.MEDIAN_IDX] - yt).mean(-1)
                      / np.maximum(np.abs(c["qfac_zeroshot"][:, E3.MEDIAN_IDX] - yt).mean(-1), 1e-12))
    return float(np.median(np.concatenate(gs))), float(np.median(np.concatenate(er))) if er else np.nan


def write_summary(run: RunDir, cfg: Exp7Config) -> None:
    recs = run.read_metrics()
    if not recs:
        return
    c3 = exp3_config(cfg)
    cells = []
    for r in recs:
        z = dict(np.load(run.cell_path(r["key"]), allow_pickle=False))
        z.update({k: r[k] for k in ("sample_seed", "input", "L", "setting")})
        cells.append(z)
    methods = ["zeroshot", "r1"] + [f"r2_s{s}" for s in cfg.train_seeds] + ["dither"]
    methods = [m for m in methods if f"ehat_{m}_{c3.shapes[0]}" in cells[0]]
    rep = results_root() / "reports"
    (rep / "tables").mkdir(parents=True, exist_ok=True)
    lines = [f"# Exp7 summary ({cfg.run_name}) - seed robustness", "",
             f"Source: `{run.metrics_file}` ({len(cells)} cells). Sample seeds {cfg.sample_seeds}, R2 training seeds "
             f"{cfg.train_seeds}, L {cfg.contexts}. Median g and median factual-error ratio vs zero-shot, pooled over "
             "cells and both shapes.", ""]
    groups = [("all", lambda c: True), ("input=prbs", lambda c: c["input"] == "prbs"),
              ("input=multisine", lambda c: c["input"] == "multisine")]
    rng = np.random.default_rng(0)
    trows = []  # rows of the LaTeX table ("all" group)

    def sd(v: np.ndarray) -> float:
        return float(v.std(ddof=1)) if len(v) > 1 else 0.0
    for gname, sel in groups:
        if not any(sel(c) for c in cells):
            continue
        lines += [f"## {gname}", "", "| method | " + " | ".join(f"seed {s}: g / err" for s in cfg.sample_seeds)
                  + " | g mean +- sd | err mean +- sd | g 95% CI (seed 0) | err 95% CI (seed 0) |",
                  "|---|" + "---|" * (len(cfg.sample_seeds) + 4)]
        agg = {}
        for m in methods:
            per = [_pooled([c for c in cells if c["sample_seed"] == s and sel(c)], m, c3.shapes) for s in cfg.sample_seeds]
            agg[m] = per
            c0 = [c for c in cells if c["sample_seed"] == cfg.sample_seeds[0] and sel(c)]
            bs = []
            for _ in range(cfg.n_boot):
                idx = rng.integers(0, len(c0), len(c0))
                bs.append(_pooled([c0[i] for i in idx], m, c3.shapes))
            bs = np.asarray(bs)
            ci_g = np.nanpercentile(bs[:, 0], [2.5, 97.5])
            ci_e = np.nanpercentile(bs[:, 1], [2.5, 97.5]) if np.isfinite(bs[:, 1]).any() else (np.nan, np.nan)
            gv, ev = np.array([p[0] for p in per]), np.array([p[1] for p in per])
            lines.append(f"| {m} | " + " | ".join(f"{a:.3f} / {b:.3f}" for a, b in per)
                         + f" | {gv.mean():.3f} +- {gv.std(ddof=1) if len(gv) > 1 else 0:.3f} | "
                         f"{ev.mean():.3f} +- {ev.std(ddof=1) if len(ev) > 1 else 0:.3f} | "
                         f"[{ci_g[0]:.3f}, {ci_g[1]:.3f}] | [{ci_e[0]:.3f}, {ci_e[1]:.3f}] |")
            if gname == "all":
                trows.append((m, [TAB_LABEL.get(m, m), f"{gv.mean():.3f} $\\pm$ {sd(gv):.3f}",
                                  f"[{ci_g[0]:.3f}, {ci_g[1]:.3f}]", f"{ev.mean():.3f} $\\pm$ {sd(ev):.3f}",
                                  f"[{ci_e[0]:.3f}, {ci_e[1]:.3f}]"]))
        r2 = [m for m in methods if m.startswith("r2_s")]
        if r2:
            allg = np.array([agg[m][i][0] for m in r2 for i in range(len(cfg.sample_seeds))])
            alle = np.array([agg[m][i][1] for m in r2 for i in range(len(cfg.sample_seeds))])
            lines += ["", f"R2 over {len(r2)} training x {len(cfg.sample_seeds)} sample seeds: g {allg.mean():.3f} +- "
                      f"{allg.std(ddof=1):.3f}, err ratio {alle.mean():.3f} +- {alle.std(ddof=1):.3f}"]
            if gname == "all":
                trows.append(("r2_all", [f"R2, all {len(r2)}$\\times${len(cfg.sample_seeds)} seed pairs",
                                         f"{allg.mean():.3f} $\\pm$ {sd(allg):.3f}", "--",
                                         f"{alle.mean():.3f} $\\pm$ {sd(alle):.3f}", "--"]))
        lines.append("")
    # mean +- sd over the sample seeds (R2: per training seed, then pooled over training x sample seeds);
    # 95% CI: cluster bootstrap over the cells of sample seed 0
    order = ["zeroshot", "r1", "dither"] + [m for m, _ in trows if m.startswith("r2_s")] + ["r2_all"]
    row_of = dict(trows)
    hdr = [" & \\multicolumn{2}{c}{$g$} & \\multicolumn{2}{c}{err.\\ ratio vs zero-shot} \\\\",
           "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
           "Method & mean $\\pm$ sd & 95\\% CI & mean $\\pm$ sd & 95\\% CI \\\\"]
    rows = []
    for m in [m for m in order if m in row_of]:
        if m in ("r2_s0", "r2_all"):
            rows.append("\\midrule")
        rows.append(row_of[m])
    (rep / "tables" / f"table7_seeds_{cfg.run_name}.tex").write_text(
        "\\setlength{\\tabcolsep}{4pt}\n" + rows_to_tex("l cc cc", hdr, rows), encoding="utf-8")
    (rep / f"exp7_summary_{cfg.run_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("summary written")


def main() -> None:
    cfg, args = parse_args()
    run = RunDir("exp7", cfg.run_name)
    setup_logging(run.path / "log.txt")
    log.info("config: %s", json.dumps(asdict(cfg)))
    if args.summary_only:
        write_summary(run, cfg)
        return
    run.write_config(cfg, resume=args.resume, ignore_keys=("device", "batch_size"))
    from covattn.model import Chronos2Wrapper

    c3 = exp3_config(cfg)
    w = Chronos2Wrapper(c3.model_id, device=cfg.device, amp=cfg.amp, batch_size=cfg.batch_size)
    torch.manual_seed(c3.seed)
    d3 = results_root() / "exp3" / cfg.exp3_run
    r1 = json.loads((d3 / "r1_select.json").read_text())
    sg = json.loads((d3 / "dither_select.json").read_text())["best"]["sigma"]
    hk = w.hooked
    G1 = r1_gain(hk.n_layers, hk.n_heads, [tuple(x) for x in r1["ranking"][: r1["best"]["k"]]], r1["best"]["gamma"])
    gates = {s: r2_gates(w, cfg, c3, s, run) for s in cfg.train_seeds}
    methods = {"zeroshot": lambda bt, s: w.predict(bt, bt.H), "r1": lambda bt, s: w.predict_hooked(bt, gain_spec(G1))}
    for s, G in gates.items():
        methods[f"r2_s{s}"] = lambda bt, sd, G=G: w.predict_hooked(bt, gain_spec(G))
    methods["dither"] = lambda bt, s: dither_predict(w, bt, sg, c3.dither_K, seed=s)
    lk = E3.systems_by_key()
    done = run.done_keys() if args.resume else set()
    jobs = [(ss, lk[s], L, inp) for ss in cfg.sample_seeds for L in cfg.contexts for s in c3.test_systems
            for inp in c3.input_types]
    jobs = [j for j in jobs if f"s{j[0]}|{E3.cell_key(j[1].key, j[2], j[3])}" not in done]
    log.info("%d cells to run (L4 rough: ~5 s per cell)", len(jobs))
    t_start = time.time()
    for i, (ss, sys, L, inp) in enumerate(jobs):
        c3s = replace(c3, seed=ss)
        key = f"s{ss}|{E3.cell_key(sys.key, L, inp)}"
        rec = E3.eval_cell(methods, w, c3s, sys, L, inp, run, save_key=key)
        rec["key"], rec["sample_seed"] = key, ss
        run.append_metrics(rec)
        el = time.time() - t_start
        log.info("[%d/%d] %s ETA %s | g_step %s", i + 1, len(jobs), key, fmt_seconds(el / (i + 1) * (len(jobs) - i - 1)),
                 {m: round(v.get(f"g_{c3.shapes[0]}", float("nan")), 3) for m, v in rec["methods"].items()})
    write_summary(run, cfg)
    n_all = len(cfg.sample_seeds) * len(cfg.contexts) * len(c3.test_systems) * len(c3.input_types)
    if len(run.done_keys()) == n_all:
        run.mark_done({"n_cells": n_all})


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        log.exception("Exp7 crashed")
        raise
