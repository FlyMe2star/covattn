"""Exp10 - semi-synthetic what-if on the real series of all 28 fev-bench covariate tasks (Exp6 used 5 EPF series).

Background = a real target series of the task (the first `n_bg` series by id that are long enough; about
`target_windows` forecast origins per task, the last ones of each series, stride H). Covariate = a real covariate from an EPF
market (rotating over the 5 markets, shifted by 26 weeks and standardised), so it is independent of the
background. Injected effect: x = B(q)/A(q) u for 3 systems, scaled to kappa * std(background context); the true
what-if effect of a 1-std step / pulse in the future covariate is then known exactly.

Pre-declared rules (no task is selected by result): L = 512 if the task has a series of length >= 512 + H,
else 256 if >= 256 + H, else the task is skipped and listed. Windows with missing values are dropped.

  python scripts/exp10_semisynth_fev.py --quick
  python scripts/exp10_semisynth_fev.py --resume --exp3-run v2        # Colab
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
from covattn.baselines import arx_effects
from covattn.fev_tasks import TASKS
from covattn.io_utils import RunDir, fmt_seconds, results_root, setup_logging
from covattn.latex import DOWN, TO1, UP, mark_best, rows_to_tex
from covattn.model import CovBatch
from covattn.real_data import EPF, load_epf, load_fev_task
from covattn.systems import default_systems, delta_u, make_rng, transfer_coeffs

import exp6_real_world as E6  # noqa: E402  (scripts/ is on sys.path when this file is run)

log = logging.getLogger("exp10")
MEDIAN_IDX = 10
LABEL = {"zeroshot": "Zero-shot", "r1": "R1", "r2": "R2", "dither": "Dither (reimpl.)", "ft_full": "Full fine-tune",
         "ft_lora": "LoRA", "arx": "ARX (context fit)"}
ENERGY = ("entsoe", "epf", "proenfo", "solar")


def domain(task: str) -> str:
    return "energy" if task.startswith(ENERGY) else "retail"


@dataclass
class Exp10Config:
    run_name: str = "main"
    quick: bool = False
    exp3_run: str = "v2"
    tasks: list[str] = field(default_factory=lambda: [t["config"] for t in TASKS])
    systems: list[str] = field(default_factory=lambda: ["F2_a0.8", "F3_a0.8_d4", "F4_P24_z0.2"])
    kappas: list[float] = field(default_factory=lambda: [0.5, 1.0])
    contexts: list[int] = field(default_factory=lambda: [512, 256, 128])  # first that fits, per task
    horizon: int = 64
    n_load: int = 30  # series loaded per task (first by id)
    n_bg: int = 8  # background series used per task (first eligible by id)
    target_windows: int = 40  # windows per task, split evenly over the eligible series (stride H)
    shift: int = 26 * 168
    burn: int = 500
    eff_thresh: float = 0.05
    shapes: list[str] = field(default_factory=lambda: ["step", "pulse"])
    use_ft: bool = True
    seed: int = 0
    device: str = "auto"
    batch_size: int = 128


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--exp3-run", default=None)
    p.add_argument("--no-ft", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    cfg = Exp10Config(quick=a.quick, device=a.device, batch_size=a.batch_size, use_ft=not a.no_ft)
    if a.quick:
        cfg.run_name, cfg.exp3_run = "quick", "quick"
        cfg.tasks, cfg.systems, cfg.kappas = ["epf_be", "m5_1W"], ["F2_a0.8"], [1.0]
        cfg.contexts, cfg.n_load, cfg.n_bg, cfg.target_windows = [256, 128], 3, 2, 4
    if a.run_name is not None:
        cfg.run_name = a.run_name
    if a.exp3_run is not None:
        cfg.exp3_run = a.exp3_run
    return cfg, a


def _sys(key: str):
    return {s.key: s for s in default_systems()}[key]


def build_windows(task: dict, cfg: Exp10Config, donors: dict, t_idx: int) -> dict | None:
    """Background windows of one task and their donor covariate segments (shared by all systems / kappas)."""
    series = load_fev_task(task, cfg.n_load)
    H = cfg.horizon
    lens = [int(np.isfinite(y).sum()) for _, y, _ in series]
    L = next((c for c in cfg.contexts if any(n >= c + H for n in lens)), None)
    if L is None:
        return None
    di = t_idx % len(EPF)
    if task["config"].startswith(EPF[di]):  # never take the covariate from the background's own market
        di = (di + 1) % len(EPF)
    donor_name = EPF[di]
    u_all = donors[donor_name]
    eligible = [(sid, y) for sid, y, _ in series if len(y) >= L + H][: cfg.n_bg]
    per = -(-cfg.target_windows // max(len(eligible), 1))  # windows per series (ceil)
    bgs, us, sids = [], [], []
    for sid, y in eligible:
        for j in range(per):
            o = len(y) - H - j * H
            if o - L < 0:
                break
            bg = y[o - L: o + H].astype(np.float64)
            if not np.isfinite(bg).all() or np.std(bg[:L]) <= 0:
                continue
            rng = make_rng(cfg.seed, "test", "exp10", task["config"], sid, j)
            a = int(rng.integers(0, len(u_all) - (L + H + cfg.burn)))
            bgs.append(bg)
            us.append(u_all[a: a + L + H + cfg.burn])
            sids.append(f"{sid}#{j}")
    if not bgs:
        return None
    return {"L": L, "bg": np.stack(bgs), "u": np.stack(us), "ids": sids, "donor": donor_name}


def run_cell(task: str, win: dict, sys_key: str, kappa: float, methods: dict, cfg: Exp10Config, run: RunDir) -> dict:
    L, H = win["L"], cfg.horizon
    B, A = transfer_coeffs(_sys(sys_key))
    ys, yts, ucs, ufs, es, dus = [], [], [], [], [], []
    for bg, u in zip(win["bg"], win["u"]):
        x = lfilter(B, A, u)[cfg.burn:]
        b = kappa * np.std(bg[:L]) / max(np.std(x[:L]), 1e-12)
        y = bg + b * x
        ys.append(y[:L]); yts.append(y[L:]); ucs.append(u[cfg.burn: cfg.burn + L]); ufs.append(u[-H:])
        es.append([b * lfilter(B, A, delta_u(sh, H, 1.0)) for sh in cfg.shapes])
        dus.append([delta_u(sh, H, 1.0) for sh in cfg.shapes])
    y, uc, uf, yt = (np.asarray(v, np.float32) for v in (ys, ucs, ufs, yts))
    e, du = np.asarray(es, np.float32), np.asarray(dus, np.float32)
    arrays = {"y_true": yt, "y_scale": y.std(-1)}
    for si, sh in enumerate(cfg.shapes):
        arrays[f"e_{sh}"] = e[:, si]
        arrays[f"keep_{sh}"] = M.effect_mask(e[:, si], y.std(-1), cfg.eff_thresh)
    key = f"{task}|{sys_key}|k{kappa:g}"
    rec = {"key": key, "task": task, "domain": domain(task), "system": sys_key, "kappa": kappa, "L": L, "n": len(y),
           "donor": win["donor"], "methods": {}}
    fac = CovBatch(y, uc, uf)
    for m, fn in methods.items():
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
def write_summary(run: RunDir, cfg: Exp10Config) -> None:
    recs = run.read_metrics()
    recs = [r for r in recs if "task" in r]
    if not recs:
        return
    cells = []
    for r in recs:
        z = dict(np.load(run.cell_path(r["key"]), allow_pickle=False))
        z.update({k: r[k] for k in ("task", "domain", "system", "kappa", "L")})
        cells.append(z)
    ms = [m for m in LABEL if f"ehat_{m}_{cfg.shapes[0]}" in cells[0]]

    def per_pair(cs, m):
        g, r, n, er = [], [], [], []
        for c in cs:
            for sh in cfg.shapes:
                k = c[f"keep_{sh}"]
                eh, e = c[f"ehat_{m}_{sh}"][k], c[f"e_{sh}"][k]
                g.append(M.effect_gain(eh, e)); r.append(M.shape_corr(eh, e)); n.append(M.whatif_nrmse(eh, e))
            if f"med_{m}" in c:
                er.append(np.abs(c[f"med_{m}"] - c["y_true"]).mean(-1)
                          / np.maximum(np.abs(c["med_zeroshot"] - c["y_true"]).mean(-1), 1e-12))
        g, r, n = (np.concatenate(x) for x in (g, r, n))
        return (float(np.median(g)), float(np.nanmedian(r)), float(np.median(n)),
                float(np.median(np.concatenate(er))) if er else float("nan"), len(g))

    tasks = sorted({c["task"] for c in cells})
    skipped = [t for t in cfg.tasks if t not in tasks]
    lines = [f"# Exp10 summary ({cfg.run_name}) - semi-synthetic what-if on fev-bench series", "",
             f"Source: `{run.metrics_file}` ({len(cells)} cells, {len(tasks)} tasks). Per method: median g / shape r / "
             "what-if NRMSE / factual error ratio vs zero-shot / n pairs.",
             f"Tasks skipped by the length rule or with no usable window: {skipped if skipped else 'none'}.",
             f"Context length per task: " + ", ".join(f"{t}: {next(c['L'] for c in cells if c['task'] == t)}" for t in tasks), ""]
    groups = [("all", cells)] + [(f"domain={d}", [c for c in cells if c["domain"] == d]) for d in ("energy", "retail")]
    groups += [(f"system={s}", [c for c in cells if c["system"] == s]) for s in cfg.systems]
    groups += [(f"kappa={k:g}", [c for c in cells if c["kappa"] == k]) for k in cfg.kappas]
    groups += [(f"L={L}", [c for c in cells if c["L"] == L]) for L in sorted({c["L"] for c in cells})]
    allrows = {}
    for gname, cs in groups:
        if not cs:
            continue
        lines += [f"## {gname} ({len(cs)} cells, {len({c['task'] for c in cs})} tasks)", "",
                  "| method | g | r | NRMSE | err ratio | n |", "|---|---|---|---|---|---|"]
        for m in ms:
            s = per_pair(cs, m)
            if gname == "all":
                allrows[m] = s
            lines.append(f"| {LABEL[m]} | {s[0]:.3f} | {s[1]:.3f} | {s[2]:.3f} | {s[3]:.3f} | {s[4]} |")
        lines.append("")
    # per task: median g of zero-shot / R1 / R2 and whether the repair moves g closer to 1 (pre-declared count)
    lines += ["## Per task (all systems and kappas pooled): median g", "",
              "| task | domain | L | " + " | ".join(LABEL[m] for m in ms if m != "arx") + " |",
              "|---|---|---|" + "---|" * len([m for m in ms if m != "arx"])]
    closer = {m: 0 for m in ("r1", "r2") if m in ms}
    overshoot = {m: 0 for m in closer}
    pertask = {}
    for t in tasks:
        cs = [c for c in cells if c["task"] == t]
        gs = {m: per_pair(cs, m)[0] for m in ms if m != "arx"}
        pertask[t] = gs
        for m in closer:
            closer[m] += abs(1 - gs[m]) < abs(1 - gs["zeroshot"])
            overshoot[m] += gs[m] > 1.2
        lines.append(f"| {t} | {domain(t)} | {cs[0]['L']} | " + " | ".join(f"{gs[m]:.3f}" for m in gs) + " |")
    lines += ["", "Tasks where the repair's median g is closer to 1 than zero-shot's: "
              + ", ".join(f"{LABEL[m]} {closer[m]}/{len(tasks)}" for m in closer)
              + "; tasks with median g > 1.2 (overshoot): " + ", ".join(f"{LABEL[m]} {overshoot[m]}" for m in overshoot)]
    rep = results_root() / "reports"
    (rep / "tables").mkdir(parents=True, exist_ok=True)
    (rep / f"exp10_summary_{cfg.run_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (rep / f"exp10_pertask_{cfg.run_name}.json").write_text(json.dumps(
        {"pertask": pertask, "closer": closer, "overshoot": overshoot, "skipped": skipped,
         "all": {m: list(v) for m, v in allrows.items()}}, indent=1), encoding="utf-8")
    order = list(allrows)
    cand = [m in ("zeroshot", "r1", "r2", "dither") for m in order]
    cand_err = [c and m != "zeroshot" for c, m in zip(cand, order)]
    cols = [mark_best([allrows[m][j] for m in order], ".2f", mode, cand if j < 3 else cand_err)
            for j, mode in ((0, "one"), (1, "max"), (2, "min"), (3, "min"))]
    hdr = [f"Method & $g${TO1} & $r${UP} & NRMSE{DOWN} & err.\\ ratio{DOWN} \\\\"]
    rows = []
    for i, m in enumerate(order):
        if m in ("ft_full", "arx") and i > 0:
            rows.append("\\midrule")
        rows.append([LABEL[m]] + [c[i] for c in cols])
    (rep / "tables" / f"table10_semifev_{cfg.run_name}.tex").write_text(rows_to_tex("l cccc", hdr, rows),
                                                                        encoding="utf-8")
    log.info("summary written")


def main() -> None:
    cfg, args = parse_args()
    run = RunDir("exp10", cfg.run_name)
    setup_logging(run.path / "log.txt")
    log.info("config: %s", json.dumps(asdict(cfg)))
    if args.summary_only:
        write_summary(run, cfg)
        return
    run.write_config(cfg, resume=args.resume, ignore_keys=("device", "batch_size"))
    from covattn.model import Chronos2Wrapper

    w = Chronos2Wrapper("amazon/chronos-2", device=cfg.device, batch_size=cfg.batch_size)
    torch.manual_seed(cfg.seed)
    e6 = E6.Exp6Config(exp3_run=cfg.exp3_run, r1_gammas=[], use_ft=cfg.use_ft, device=cfg.device,
                       batch_size=cfg.batch_size)
    methods, info = E6.build_methods(w, e6)
    methods = {m: f for m, f in methods.items() if not m.startswith("r1_g")}
    (run.path / "methods.json").write_text(json.dumps({"methods": list(methods), **info}, indent=1))
    log.info("methods: %s", list(methods))
    donors = {}
    for name in EPF:
        u = np.roll(load_epf(name).covs[0].astype(np.float64), cfg.shift)
        donors[name] = (u - u.mean()) / u.std()
    done = run.done_keys() if args.resume else set()
    tk = {t["config"]: t for t in TASKS}
    t_start = time.time()
    n_jobs = len(cfg.tasks) * len(cfg.systems) * len(cfg.kappas)
    n_done = 0
    for ti, tname in enumerate(cfg.tasks):
        keys = [f"{tname}|{s}|k{k:g}" for s in cfg.systems for k in cfg.kappas]
        if all(k in done for k in keys):
            n_done += len(keys)
            continue
        try:
            win = build_windows(tk[tname], cfg, donors, ti)
        except Exception:  # noqa: BLE001
            log.exception("%s: loading FAILED - skipped", tname)
            continue
        if win is None:
            log.info("%s: no series long enough / no usable window - skipped (pre-declared rule)", tname)
            continue
        log.info("%s: L=%d, %d windows, donor %s", tname, win["L"], len(win["bg"]), win["donor"])
        for s in cfg.systems:
            for k in cfg.kappas:
                key = f"{tname}|{s}|k{k:g}"
                n_done += 1
                if key in done:
                    continue
                try:
                    rec = run_cell(tname, win, s, k, methods, cfg, run)
                except Exception:  # noqa: BLE001
                    log.exception("%s FAILED", key)
                    continue
                run.append_metrics(rec)
                el = time.time() - t_start
                log.info("[%d/%d] %s ETA %s | g_step %s", n_done, n_jobs, key,
                         fmt_seconds(el / max(n_done, 1) * (n_jobs - n_done)),
                         {m: round(v["g_step"], 3) for m, v in rec["methods"].items()})
    write_summary(run, cfg)
    if len(run.done_keys()) >= n_jobs:
        run.mark_done({"n": n_jobs})


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        log.exception("Exp10 crashed")
        raise
