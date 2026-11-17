"""Exp5 - ablations of the training-free repairs on the synthetic test split.

Reuses the Exp3 artefacts (R1 ranking + chosen k/gamma, R2 gates). Parts (each cell resumable):
  heads : which heads get the gain, at Exp3's k and gamma, L=512, delta=1, snr=10 -
          R1 (Exp2 ranking over all Exp2 systems), R1 with a ranking from the train systems only,
          3 fixed random head sets, top-k inside layers 0-3 / 4-7 / 8-11; plus zero-shot and R2.
          The samples are the Exp3 test samples, so zero-shot / R1 / R2 here must equal Exp3 (checked).
  delta : counterfactual size delta in {0.25, 0.5, 2, 4} (zero-shot, R1, R2) - linearity of the effect
  L     : context length 256 and 1024 (512 / 2048 are in Exp3)
  snr   : measurement-noise level snr in {3, 30, inf} (10 is the default used everywhere else)
Not re-run here (already in other results; collected by the summary):
  k x gamma grid -> Exp3 r1_select.json (val split); cross-family generalisation -> Exp3 OOD-family
  (R2 trained on F1/F2/F3, tested on F4); L=512/2048 -> Exp3.

  python scripts/exp5_ablations.py --quick
  python scripts/exp5_ablations.py --resume --exp3-run v2         # Colab
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
from covattn.io_utils import RunDir, fmt_seconds, results_root, setup_logging
from covattn.model import CovBatch, time_estimates
from covattn.repair import gain_spec, head_ranking, r1_gain
from covattn.systems import SampleSpec, default_systems, make_counterfactual, make_rng, make_sample, quick_systems

log = logging.getLogger("exp5")
MEDIAN_IDX = 10  # 21 Chronos-2 quantiles; index 10 = 0.5
TRAIN_SYSTEMS = ["F1", "F2_a0.5", "F2_a0.8", "F2_a0.95", "F3_a0.8_d1", "F3_a0.8_d4", "F3_a0.8_d16", "F3_a0.8_d32"]


@dataclass
class Exp5Config:
    run_name: str = "main"
    quick: bool = False
    exp3_run: str = "v2"
    exp2_run: str = "main"
    rank_context: int = 512
    test_systems: list[str] = field(default_factory=lambda: [s.key for s in default_systems()])
    train_systems: list[str] = field(default_factory=lambda: list(TRAIN_SYSTEMS))
    input_types: list[str] = field(default_factory=lambda: ["prbs", "multisine", "ar1", "steps"])
    shapes: list[str] = field(default_factory=lambda: ["step", "pulse"])
    base_L: int = 512
    base_delta: float = 1.0
    base_snr: float = 10.0
    horizon: int = 64
    n_per_cell: int = 32
    eff_thresh: float = 0.05
    n_random_sets: int = 3
    layer_bands: list[list[int]] = field(default_factory=lambda: [[0, 3], [4, 7], [8, 11]])
    deltas: list[float] = field(default_factory=lambda: [0.25, 0.5, 2.0, 4.0])
    contexts: list[int] = field(default_factory=lambda: [256, 1024])
    snrs: list[float] = field(default_factory=lambda: [3.0, 30.0, float("inf")])
    parts: list[str] = field(default_factory=lambda: ["heads", "delta", "L", "snr"])
    seed: int = 0
    model_id: str = "amazon/chronos-2"
    amp: str = "off"
    device: str = "auto"
    batch_size: int = 128


def parse_args() -> tuple[Exp5Config, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--exp3-run", default=None)
    p.add_argument("--parts", nargs="+", default=None, choices=["heads", "delta", "L", "snr"])
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    cfg = Exp5Config(quick=a.quick, amp=a.amp, device=a.device, batch_size=a.batch_size)
    if a.quick:
        cfg.run_name, cfg.exp3_run, cfg.exp2_run, cfg.rank_context = "quick", "quick", "quick", 256
        cfg.test_systems = ["F2_a0.8", "F4_P24_z0.2"]
        cfg.train_systems = ["F2_a0.99"]  # quick Exp2 has no train systems; this only exercises the code path
        cfg.input_types = ["prbs"]
        cfg.base_L, cfg.n_per_cell, cfg.n_random_sets = 256, 4, 2
        cfg.deltas, cfg.contexts, cfg.snrs = [2.0], [128], [float("inf")]
    if a.run_name is not None:
        cfg.run_name = a.run_name
    if a.exp3_run is not None:
        cfg.exp3_run = a.exp3_run
    if a.parts is not None:
        cfg.parts = a.parts
    return cfg, a


def systems_by_key() -> dict:
    return {s.key: s for s in default_systems() + quick_systems()}


# ------------------------------------------------------------------------------------ methods
def head_sets(cfg: Exp5Config, ranking: list, k: int, n_layers: int, n_heads: int) -> tuple[dict[str, list], list]:
    """Named head sets of size k for the `heads` part (fixed seeds, all sets are reported)."""
    exp2_dir = results_root() / "exp2" / cfg.exp2_run
    exp2_systems = json.loads((exp2_dir / "config.json").read_text())["config"]["systems"]
    tr = [s for s in exp2_systems if s in cfg.train_systems]
    sets = {"r1": ranking[:k]}
    if tr:
        sets["r1_trainrank"] = head_ranking(exp2_dir, cfg.rank_context, "cross", systems=tr)[:k]
    else:
        log.warning("no Exp2 cells of train systems in %s - skipping the train-only ranking", exp2_dir)
    allh = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    for i in range(cfg.n_random_sets):
        rng = make_rng(cfg.seed, "test", "exp5-random-heads", i)
        sets[f"random{i + 1}"] = [allh[j] for j in sorted(rng.choice(len(allh), size=k, replace=False))]
    for lo, hi in cfg.layer_bands:
        sets[f"r1_layers{lo}-{hi}"] = [x for x in ranking if lo <= x[0] <= hi][:k]
    return sets, tr


def build_methods(w, cfg: Exp5Config, part: str) -> tuple[dict, dict]:
    d3 = results_root() / "exp3" / cfg.exp3_run
    r1 = json.loads((d3 / "r1_select.json").read_text())
    k, gamma = r1["best"]["k"], r1["best"]["gamma"]
    ranking = [tuple(x) for x in r1["ranking"]]
    hk = w.hooked
    gates = torch.tensor(json.loads((d3 / "r2_gates.json").read_text())["gates"])
    G1 = r1_gain(hk.n_layers, hk.n_heads, ranking[:k], gamma)
    methods = {"zeroshot": lambda bt: w.predict(bt, bt.H),
               "r1": lambda bt: w.predict_hooked(bt, gain_spec(G1)),
               "r2": lambda bt: w.predict_hooked(bt, gain_spec(gates))}
    info = {"k": k, "gamma": gamma, "exp3_dir": str(d3)}
    if part == "heads":
        sets, tr = head_sets(cfg, ranking, k, hk.n_layers, hk.n_heads)
        info["trainrank_systems"] = tr
        info["head_sets"] = {n: [list(x) for x in v] for n, v in sets.items()}
        for name, hs in sets.items():
            if name == "r1":
                continue
            G = r1_gain(hk.n_layers, hk.n_heads, hs, gamma)
            methods[name] = lambda bt, G=G: w.predict_hooked(bt, gain_spec(G))
    return methods, info


def part_variants(cfg: Exp5Config, part: str) -> list[tuple[str, int, float, float]]:
    """(variant name, L, delta, snr)."""
    b = (cfg.base_L, cfg.base_delta, cfg.base_snr)
    if part == "heads":
        return [("base", *b)]
    if part == "delta":
        return [(f"d{d:g}", b[0], d, b[2]) for d in cfg.deltas]
    if part == "L":
        return [(f"L{L}", L, b[1], b[2]) for L in cfg.contexts]
    if part == "snr":
        return [(f"snr{s:g}", b[0], b[1], s) for s in cfg.snrs]
    raise ValueError(part)


# ------------------------------------------------------------------------------------ evaluation
def cell_key(part: str, variant: str, s: str, L: int, inp: str) -> str:
    return f"{part}|{variant}|{s}|L{L}|{inp}"


def eval_cell(methods: dict, cfg: Exp5Config, sys, L: int, inp: str, delta: float, snr: float,
              key: str, run: RunDir) -> dict:
    spec = SampleSpec(context_length=L, horizon=cfg.horizon, input_type=inp, snr=snr)
    rng = make_rng(cfg.seed, "test", "exp3", sys.key, L, inp)  # same stream as the Exp3 test cells
    samples = [make_sample(sys, spec, rng) for _ in range(cfg.n_per_cell)]
    y = np.stack([s.y[:L] for s in samples])
    u = np.stack([s.u[:L] for s in samples])
    y_true = np.stack([s.y[L:] for s in samples])
    fac = CovBatch(y, u, np.stack([s.u[L:] for s in samples]))
    cfs = {sh: [make_counterfactual(s, sh, delta) for s in samples] for sh in cfg.shapes}
    arrays: dict = {"y_true": y_true.astype(np.float32), "y_scale": y.std(-1).astype(np.float32)}
    rec: dict = {"key": key, "system": sys.key, "family": sys.family, "L": L, "input": inp, "delta": delta,
                 "snr": snr, "methods": {}}
    for sh in cfg.shapes:
        arrays[f"e_{sh}"] = np.stack([c.effect for c in cfs[sh]]).astype(np.float32)
        arrays[f"keep_{sh}"] = M.effect_mask(arrays[f"e_{sh}"], y.std(-1), cfg.eff_thresh)
    for name, fn in methods.items():
        q_fac = fn(fac)
        med = q_fac[:, MEDIAN_IDX]
        arrays[f"med_{name}"] = med.astype(np.float32)
        mrec = {"mase": float(np.median(M.mase(med, y_true, y)))}
        for sh in cfg.shapes:
            cfb = CovBatch(y, u, np.stack([c.u_future_cf for c in cfs[sh]]))
            eh = (fn(cfb)[:, MEDIAN_IDX] - med).astype(np.float32)
            arrays[f"ehat_{name}_{sh}"] = eh
            kk = arrays[f"keep_{sh}"]
            g = M.effect_gain(eh[kk], arrays[f"e_{sh}"][kk])
            mrec[f"g_{sh}"] = float(np.median(g)) if len(g) else float("nan")
        rec["methods"][name] = mrec
    run.save_arrays(key, **arrays)
    return rec


# ------------------------------------------------------------------------------------ summary
LABEL = {"zeroshot": "Zero-shot", "r1": "R1 (Exp2 ranking)", "r2": "R2 (gates)",
         "r1_trainrank": "R1, ranking from train systems only"}


def _label(m: str) -> str:
    if m in LABEL:
        return LABEL[m]
    if m.startswith("random"):
        return f"random heads, draw {m[6:]}"
    if m.startswith("r1_layers"):
        return f"top-k inside layers {m[9:]}"
    return m


def _stats(cs: list[dict], m: str, shapes: list[str]) -> tuple[float, float, float, int]:
    gs, rs, er = [], [], []
    for c in cs:
        for sh in shapes:
            k = c[f"keep_{sh}"]
            eh, e = c[f"ehat_{m}_{sh}"][k], c[f"e_{sh}"][k]
            gs.append(M.effect_gain(eh, e))
            rs.append(M.shape_corr(eh, e))
        if f"med_{m}" in c and "med_zeroshot" in c:
            er.append(np.abs(c[f"med_{m}"] - c["y_true"]).mean(-1)
                      / np.maximum(np.abs(c["med_zeroshot"] - c["y_true"]).mean(-1), 1e-12))
    g, r = np.concatenate(gs), np.concatenate(rs)
    return (float(np.median(g)) if len(g) else np.nan, float(np.nanmedian(r)) if len(r) else np.nan,
            float(np.median(np.concatenate(er))) if er else np.nan, len(g))


def _load(run_dir, key: str) -> dict:
    safe = "".join(c if c.isalnum() or c in "-_.=" else "_" for c in key)
    return dict(np.load(run_dir / "cells" / f"{safe}.npz", allow_pickle=False))


def write_summary(run: RunDir, cfg: Exp5Config) -> None:
    recs = run.read_metrics()
    if not recs:
        return
    cells = []
    for r in recs:
        z = _load(run.path, r["key"])
        z.update({k: r[k] for k in ("system", "L", "input", "delta", "snr")})
        z["part"], z["variant"] = r["key"].split("|")[:2]
        cells.append(z)
    rep = results_root() / "reports"
    (rep / "tables").mkdir(parents=True, exist_ok=True)
    lines = [f"# Exp5 summary ({cfg.run_name})", "", f"Source: `{run.metrics_file}` ({len(cells)} cells, test split). "
             f"Exp3 artefacts: `{cfg.exp3_run}`. Per method: median g / median shape r / median ratio of factual "
             "abs. error (median quantile) vs zero-shot / n kept pairs. Pooled over systems and both shapes.", ""]
    mfile = run.path / "methods.json"
    info = json.loads(mfile.read_text()) if mfile.exists() else {}
    groups = lambda cs: [("all inputs", cs)] + [(f"input={i}", [c for c in cs if c["input"] == i])  # noqa: E731
                                                for i in cfg.input_types]
    tex_rows = []

    # ---- heads
    hc = [c for c in cells if c["part"] == "heads"]
    if hc:
        methods = [m for m in info.get("heads", {}).get("methods", []) if f"ehat_{m}_{cfg.shapes[0]}" in hc[0]]
        hi = info.get("heads", {})
        lines += ["## Which heads get the gain (L=%d, delta=%g, snr=%g; k=%s, gamma=%s)" % (
            cfg.base_L, cfg.base_delta, cfg.base_snr, hi.get("k"), hi.get("gamma")), "",
            f"Train-only ranking uses the Exp2 cells of {hi.get('trainrank_systems')}; random sets are drawn with "
            f"fixed seeds 1..{cfg.n_random_sets} (all draws are shown).", ""]
        for gname, cs in groups(hc):
            if not cs:
                continue
            lines += [f"### {gname} ({len(cs)} cells)", "", "| head set | g | r | err ratio | n |", "|---|---|---|---|---|"]
            for m in methods:
                s = _stats(cs, m, cfg.shapes)
                lines.append(f"| {_label(m)} | {s[0]:.3f} | {s[1]:.3f} | {s[2]:.3f} | {s[3]} |")
                if gname == "all inputs":
                    tex_rows.append(("heads", _label(m), s))
            lines.append("")
        # overlap of the head sets with the R1 set
        hs = hi.get("head_sets", {})
        if "r1" in hs:
            base = {tuple(x) for x in hs["r1"]}
            lines += ["Overlap with the R1 head set: " + ", ".join(
                f"{_label(n)} {len(base & {tuple(x) for x in v})}/{len(base)}" for n, v in hs.items() if n != "r1"), ""]
        # consistency with Exp3 (same samples, same methods)
        d3 = results_root() / "exp3" / cfg.exp3_run
        diffs = []
        for c, r in zip(cells, recs):
            if c["part"] != "heads":
                continue
            key3 = f"{r['system']}|L{r['L']}|{r['input']}"
            try:
                z3 = _load(d3, key3)
            except FileNotFoundError:
                continue
            for m in ("zeroshot", "r1", "r2"):
                for sh in cfg.shapes:
                    if f"ehat_{m}_{sh}" in z3 and f"ehat_{m}_{sh}" in c:
                        diffs.append(float(np.abs(z3[f"ehat_{m}_{sh}"] - c[f"ehat_{m}_{sh}"]).max()))
        if diffs:
            lines += [f"Check vs Exp3 `{cfg.exp3_run}` (same samples): max |e_hat(Exp5) - e_hat(Exp3)| over "
                      f"{len(diffs)} zero-shot/R1/R2 arrays = {max(diffs):.2e}.", ""]

    # ---- sweeps
    for part, title, var_of in (("delta", "Counterfactual size delta (linearity)", "delta"),
                                ("L", "Context length L", "L"), ("snr", "Measurement noise (snr)", "snr")):
        pc = [c for c in cells if c["part"] == part]
        if not pc:
            continue
        base = [c for c in cells if c["part"] == "heads"]  # base setting (delta=1, L=512, snr=10)
        allc = pc + base
        vals = sorted({float(c[var_of]) for c in allc})
        methods = ["zeroshot", "r1", "r2"]
        lines += [f"## {title}", "", f"Base setting from the `heads` part ({var_of}="
                  f"{ {'delta': cfg.base_delta, 'L': cfg.base_L, 'snr': cfg.base_snr}[var_of]:g}).", ""]
        for gname, _ in groups(allc):
            sel = (lambda c: True) if gname == "all inputs" else (lambda c, i=gname[6:]: c["input"] == i)
            lines += [f"### {gname}", "", f"| method | " + " | ".join(f"{var_of}={v:g}" for v in vals) + " |",
                      "|---|" + "---|" * len(vals)]
            for m in methods:
                row = []
                for v in vals:
                    cs = [c for c in allc if float(c[var_of]) == v and sel(c)]
                    s = _stats(cs, m, cfg.shapes) if cs else (np.nan,) * 4
                    row.append(f"{s[0]:.3f} / {s[2]:.3f}")
                    if gname == "all inputs":
                        tex_rows.append((part, f"{_label(m)} @ {var_of}={v:g}", s))
                lines.append(f"| {_label(m)} | " + " | ".join(row) + " |")
            lines.append("")
        lines += ["(cells: median g / err ratio vs zero-shot)", ""]

    # ---- results collected from Exp3 (not re-run)
    d3 = results_root() / "exp3" / cfg.exp3_run
    f = d3 / "r1_select.json"
    if f.exists():
        r1 = json.loads(f.read_text())
        ks = sorted({d["k"] for d in r1["grid"] if d["k"] > 0})
        gms = sorted({d["gamma"] for d in r1["grid"] if d["k"] > 0})
        look = {(d["k"], d["gamma"]): d for d in r1["grid"]}
        zs = r1["zero_shot"]
        lines += [f"## Top-k x gamma (validation split, from Exp3 `{cfg.exp3_run}` R1 selection)", "",
                  f"Zero-shot: objective {zs['obj']:.4f}, median g {zs['g_median']:.3f}. "
                  "Cells: objective / median g (lower objective is better).", "",
                  "| k | " + " | ".join(f"gamma={g:g}" for g in gms) + " |", "|---|" + "---|" * len(gms)]
        for k in ks:
            lines.append(f"| {k} | " + " | ".join(
                f"{look[(k, g)]['obj']:.3f} / {look[(k, g)]['g_median']:.3f}" if (k, g) in look else "-"
                for g in gms) + " |")
        lines += ["", "Cross-family generalisation: see Exp3 OOD-family (F4; R2 gates trained on F1/F2/F3) in "
                  f"`exp3_summary_{cfg.exp3_run}.md`; L=512/2048 are in the same file.", ""]
    out = rep / f"exp5_summary_{cfg.run_name}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    if tex_rows:
        tex = ["\\begin{tabular}{l l ccc}", "\\toprule", "Ablation & Variant & $g$ & $r$ & err.\\ ratio \\\\", "\\midrule"]
        last = None
        for part, lbl, s in tex_rows:
            if last is not None and part != last:
                tex.append("\\midrule")
            tex.append(f"{part if part != last else ''} & {lbl.replace('_', ' ')} & {s[0]:.2f} & {s[1]:.2f} & "
                       f"{s[2]:.2f} \\\\")
            last = part
        tex += ["\\bottomrule", "\\end{tabular}"]
        (rep / "tables" / f"table4_ablations_{cfg.run_name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    log.info("summary written: %s", out)


# ------------------------------------------------------------------------------------ main
def main() -> None:
    cfg, args = parse_args()
    run = RunDir("exp5", cfg.run_name)
    setup_logging(run.path / "log.txt")
    log.info("config: %s", json.dumps(asdict(cfg)))
    if args.summary_only:
        write_summary(run, cfg)
        return
    run.write_config(cfg, resume=args.resume, ignore_keys=("device", "batch_size", "parts"))
    from covattn.model import Chronos2Wrapper

    w = Chronos2Wrapper(cfg.model_id, device=cfg.device, amp=cfg.amp, batch_size=cfg.batch_size)
    torch.manual_seed(cfg.seed)
    lk = systems_by_key()
    done = run.done_keys() if args.resume else set()
    mfile = run.path / "methods.json"
    minfo = json.loads(mfile.read_text()) if mfile.exists() else {}
    jobs = []
    for part in cfg.parts:
        methods, info = build_methods(w, cfg, part)
        minfo[part] = {"methods": list(methods), **info}
        for variant, L, delta, snr in part_variants(cfg, part):
            for s in cfg.test_systems:
                for inp in cfg.input_types:
                    key = cell_key(part, variant, s, L, inp)
                    if key not in done:
                        jobs.append((methods, lk[s], L, inp, delta, snr, key))
    mfile.write_text(json.dumps(minfo, indent=1, default=str))
    n_pass = sum(len(j[0]) * (1 + len(cfg.shapes)) for j in jobs) * cfg.n_per_cell
    tps = w.measure_throughput(cfg.base_L, cfg.horizon, n_tasks=64)
    est = time_estimates(3.0 * n_pass / tps, w.device)  # hooked (eager) passes ~3x slower than SDPA
    log.info("%d cells to run (%d forward rows). ESTIMATED TIME (rough) - here: %s | L4: %s | A100(guess): %s",
             len(jobs), n_pass, *(fmt_seconds(est[k]) for k in ("here", "L4", "A100")))
    t_start = time.time()
    for i, (methods, sys, L, inp, delta, snr, key) in enumerate(jobs):
        t0 = time.time()
        rec = eval_cell(methods, cfg, sys, L, inp, delta, snr, key, run)
        rec["device"] = str(w.device)
        run.append_metrics(rec)
        el = time.time() - t_start
        log.info("[%d/%d] %s %.1fs ETA %s | g_step: %s", i + 1, len(jobs), key, time.time() - t0,
                 fmt_seconds(el / (i + 1) * (len(jobs) - i - 1)),
                 {m: round(v[f"g_{cfg.shapes[0]}"], 3) for m, v in rec["methods"].items()})
    write_summary(run, cfg)
    n_all = sum(len(part_variants(cfg, p)) for p in ("heads", "delta", "L", "snr")) * len(cfg.test_systems) * len(
        cfg.input_types)
    if len(run.done_keys()) == n_all:
        run.mark_done({"n_cells": n_all})
        log.info("all %d cells done -> done.marker", n_all)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - make sure the traceback reaches log.txt on Drive
        log.exception("Exp5 crashed")
        raise
