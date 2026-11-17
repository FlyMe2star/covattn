"""Exp3 - repairs vs baselines on synthetic forced systems with exact counterfactuals.

Stages (each resumable; finished stages are skipped with --resume):
  1. R1 selection on val: top-k heads (Exp2 per-head `cross` ranking) x gamma grid
  2. context-dither sigma selection on val (same objective)
  3. R2 gate learning on train (144 scalars, checkpointed)
  4. full fine-tuning and LoRA with the official Chronos2Pipeline.fit (same data / steps)
  5. test evaluation: zero-shot, R1, R2, dither (reimpl.), full FT, LoRA, ARX
Train systems = F1, F2 (a <= 0.95), F3. Test settings: ID (same systems, test split), OOD-param
(F2 a=0.99), OOD-family (F4), OOD-nonlinear (F5). Selection objective (R1, dither) = R2 loss:
mean ||e_hat - e||^2/||e||^2 + lam * scaled quantile loss of the factual forecast.

  python scripts/exp3_repair.py --quick
  python scripts/exp3_repair.py --resume                 # everything (Colab)
  python scripts/exp3_repair.py --resume --no-finetune   # skip stage 4 and the FT methods
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from covattn import metrics as M
from covattn.baselines import arx_effects, dither_predict, finetune
from covattn.io_utils import RunDir, fmt_seconds, results_root, setup_logging
from covattn.latex import mark_best, rows_to_tex
from covattn.model import CovBatch, time_estimates
from covattn.repair import (
    R2Config,
    gain_spec,
    head_ranking,
    make_pairs,
    objective_np,
    r1_gain,
    random_pairs,
    train_r2,
)
from covattn.systems import SampleSpec, default_systems, make_counterfactual, make_rng, make_sample

log = logging.getLogger("exp3")

TRAIN_SYSTEMS = ["F1", "F2_a0.5", "F2_a0.8", "F2_a0.95", "F3_a0.8_d1", "F3_a0.8_d4", "F3_a0.8_d16", "F3_a0.8_d32"]
SETTINGS = ("ID", "OOD-param", "OOD-family", "OOD-nonlinear")


def setting_of(key: str, train: list[str]) -> str:
    if key in train:
        return "ID"
    if key.startswith("F2") or key.startswith("F3") or key == "F1":
        return "OOD-param"
    if key.startswith("F4"):
        return "OOD-family"
    return "OOD-nonlinear"


@dataclass
class Exp3Config:
    run_name: str = "main"
    quick: bool = False
    train_systems: list[str] = field(default_factory=lambda: list(TRAIN_SYSTEMS))
    test_systems: list[str] = field(default_factory=lambda: [s.key for s in default_systems()])
    input_types: list[str] = field(default_factory=lambda: ["prbs", "multisine", "ar1", "steps"])
    shapes: list[str] = field(default_factory=lambda: ["step", "pulse"])
    train_deltas: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    test_delta: float = 1.0
    contexts: list[int] = field(default_factory=lambda: [512, 2048])
    train_contexts: list[int] = field(default_factory=lambda: [512, 2048])
    horizon: int = 64
    n_per_cell: int = 32
    eff_thresh: float = 0.05
    lam: float = 1.0
    # selection on val
    exp2_run: str = "main"
    rank_context: int = 512
    val_batches: int = 8
    val_pairs: int = 64
    r1_k: list[int] = field(default_factory=lambda: [5, 10, 20, 40, 144])
    r1_gamma: list[float] = field(default_factory=lambda: [1.25, 1.5, 2.0, 2.5, 3.0, 4.0])
    dither_sigma: list[float] = field(default_factory=lambda: [0.1, 0.3, 0.5])
    dither_K: int = 8
    # R2
    r2_steps: int = 400
    r2_lr: float = 0.01
    r2_pairs: int = 32
    r2_reg: float = 0.05
    # fine-tuning baselines (official fit; same data and steps for both)
    ft_series_per_combo: int = 64
    ft_series_length: int = 2048
    ft_steps: int = 1000
    ft_batch_rows: int = 256  # effective rows per step
    ft_grad_accum: int = 4  # micro-batch = 64 rows (the first L4 run died at the start of full FT, suspected OOM)
    ft_lr_full: float = 1e-5
    ft_lr_lora: float = 1e-4
    seed: int = 0
    model_id: str = "amazon/chronos-2"
    amp: str = "off"
    device: str = "auto"
    batch_size: int = 128


def parse_args() -> tuple[Exp3Config, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-finetune", action="store_true", help="skip stage 4 and the fine-tuned methods")
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--add-methods", nargs="+", default=None, metavar="M",
                   help="with --resume: evaluate only these methods (e.g. ft_lora) on the finished cells and merge "
                        "them into the existing cell files (same samples, other methods untouched)")
    a = p.parse_args()
    cfg = Exp3Config(quick=a.quick, amp=a.amp, device=a.device, batch_size=a.batch_size)
    if a.quick:
        cfg.run_name = "quick"
        cfg.train_systems = ["F2_a0.8", "F3_a0.8_d4"]
        cfg.test_systems = ["F2_a0.8", "F2_a0.99", "F4_P24_z0.2"]
        cfg.input_types = ["prbs"]
        cfg.contexts = [256]
        cfg.train_contexts = [256]
        cfg.n_per_cell = 4
        cfg.val_batches, cfg.val_pairs = 1, 8
        cfg.r1_k, cfg.r1_gamma = [5, 144], [1.5, 2.0]
        cfg.dither_sigma, cfg.dither_K = [0.3], 2
        cfg.r2_steps, cfg.r2_pairs = 3, 4
        cfg.ft_series_per_combo, cfg.ft_series_length, cfg.ft_steps, cfg.ft_batch_rows = 4, 512, 2, 16
        cfg.ft_grad_accum = 2
        cfg.exp2_run = "quick"
        cfg.rank_context = 256
    if a.run_name is not None:
        cfg.run_name = a.run_name
    return cfg, a


def systems_by_key() -> dict:
    from covattn.systems import quick_systems

    return {s.key: s for s in default_systems() + quick_systems()}


# ------------------------------------------------------------------------------------ stages 1-2: selection
def val_sets(cfg: Exp3Config) -> list:
    lk = systems_by_key()
    sys_ = [lk[k] for k in cfg.train_systems]
    return [random_pairs(sys_, cfg.input_types, cfg.shapes, cfg.train_deltas, cfg.train_contexts, cfg.val_pairs,
                         cfg.horizon, cfg.seed, "val", f"exp3-val-{i}") for i in range(cfg.val_batches)]


def eval_objective(w, vs, predict_fn) -> dict:
    accs = []
    for pb in vs:
        q_fac, q_cf = predict_fn(pb.fac, 0), predict_fn(pb.cf, 0)
        accs.append(objective_np(q_fac, q_cf, pb, w.quantiles, w.median_idx, LAM[0]))
    return {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}


LAM = [1.0]


def stage_select(w, cfg: Exp3Config, run: RunDir) -> tuple[dict, dict]:
    f_r1, f_di = run.path / "r1_select.json", run.path / "dither_select.json"
    if f_r1.exists() and f_di.exists():
        return json.loads(f_r1.read_text()), json.loads(f_di.read_text())
    vs = val_sets(cfg)
    hk = w.hooked
    ranking = head_ranking(results_root() / "exp2" / cfg.exp2_run, cfg.rank_context, "cross")
    t0 = time.time()
    zs = eval_objective(w, vs, lambda b, s: w.predict(b, b.H))
    grid = [{"k": 0, "gamma": 1.0, **zs}]
    log.info("val zero-shot: %s (%.1fs)", zs, time.time() - t0)
    for k in cfg.r1_k:
        for gm in cfg.r1_gamma:
            G = r1_gain(hk.n_layers, hk.n_heads, ranking[:k], gm)
            r = eval_objective(w, vs, lambda b, s, G=G: w.predict_hooked(b, gain_spec(G)))
            grid.append({"k": k, "gamma": gm, **r})
            log.info("R1 k=%d gamma=%.2f: obj=%.4f eff=%.4f ql=%.4f g=%.3f", k, gm, r["obj"], r["eff"], r["ql"],
                     r["g_median"])
    best = min(grid, key=lambda d: d["obj"])
    r1 = {"best": best, "grid": grid, "ranking": ranking, "zero_shot": zs}
    f_r1.write_text(json.dumps(r1, indent=1))
    dgrid = []
    for sg in cfg.dither_sigma:
        r = eval_objective(w, vs, lambda b, s, sg=sg: dither_predict(w, b, sg, cfg.dither_K, seed=cfg.seed + 7))
        dgrid.append({"sigma": sg, **r})
        log.info("dither sigma=%.2f: obj=%.4f g=%.3f", sg, r["obj"], r["g_median"])
    di = {"best": min(dgrid, key=lambda d: d["obj"]), "grid": dgrid}
    f_di.write_text(json.dumps(di, indent=1))
    return r1, di


# ------------------------------------------------------------------------------------ stage 3: R2
def stage_r2(w, cfg: Exp3Config, run: RunDir) -> torch.Tensor:
    done = run.path / "r2_gates.json"
    if done.exists():
        return torch.tensor(json.loads(done.read_text())["gates"])
    lk = systems_by_key()
    sys_ = [lk[k] for k in cfg.train_systems]
    mf = run.path / "r2_train.jsonl"

    def batch_fn(step: int):
        return random_pairs(sys_, cfg.input_types, cfg.shapes, cfg.train_deltas, cfg.train_contexts, cfg.r2_pairs,
                            cfg.horizon, cfg.seed, "train", f"exp3-r2-{step}")

    def cb(m: dict) -> None:
        with mf.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(m) + "\n")
        if m["step"] % 25 == 0 or m["step"] == 1:
            log.info("R2 step %d: %s", m["step"], {k: round(v, 4) for k, v in m.items() if k != "step"})

    t0 = time.time()
    gates = train_r2(w, R2Config(steps=cfg.r2_steps, lr=cfg.r2_lr, pairs_per_step=cfg.r2_pairs, lam=cfg.lam,
                                 reg=cfg.r2_reg),
                     batch_fn, run.path / "r2_ckpt.pt", cb)
    done.write_text(json.dumps({"gates": gates.tolist(), "train_seconds": time.time() - t0}))
    return gates


# ------------------------------------------------------------------------------------ stage 4: fine-tuning
def ft_inputs(cfg: Exp3Config) -> list[dict]:
    lk = systems_by_key()
    out = []
    for key in cfg.train_systems:
        for inp in cfg.input_types:
            rng = make_rng(cfg.seed, "train", "exp3-ft", key, inp)
            spec = SampleSpec(context_length=cfg.ft_series_length, horizon=cfg.horizon, input_type=inp)
            for _ in range(cfg.ft_series_per_combo):
                s = make_sample(lk[key], spec, rng)
                out.append({"target": s.y.astype(np.float32), "past_covariates": {"u0": s.u.astype(np.float32)},
                            "future_covariates": {"u0": None}})
    return out


def stage_ft(w, cfg: Exp3Config, run: RunDir) -> dict:
    infos = {}
    data = None
    for mode, lr in (("full", cfg.ft_lr_full), ("lora", cfg.ft_lr_lora)):
        d = run.path / f"ft_{mode}"
        f = d / "info.json"
        if f.exists():
            infos[mode] = json.loads(f.read_text())
            continue
        data = data or ft_inputs(cfg)
        t0 = time.time()
        log.info("fine-tuning (%s): %d series, %d steps", mode, len(data), cfg.ft_steps)
        try:
            info = finetune(w, mode, data, d, cfg.horizon, max(cfg.train_contexts), lr, cfg.ft_steps,
                            cfg.ft_batch_rows, cfg.seed, grad_accum=cfg.ft_grad_accum)
        except Exception:  # noqa: BLE001 - keep the rest of Exp3 alive; the method is then missing from the tables
            log.exception("fine-tuning (%s) FAILED - continuing without it", mode)
            if w.device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        info["train_seconds"] = time.time() - t0
        info["n_series"] = len(data)
        f.write_text(json.dumps(info, indent=1))
        log.info("fine-tuned (%s) in %s: %s", mode, fmt_seconds(info["train_seconds"]), info)
        infos[mode] = info
    return infos


# ------------------------------------------------------------------------------------ stage 5: evaluation
def cell_key(s: str, L: int, inp: str) -> str:
    return f"{s}|L{L}|{inp}"


def eval_cell(methods: dict, w, cfg: Exp3Config, sys, L: int, inp: str, run: RunDir,
              save_key: str | None = None) -> dict:
    spec = SampleSpec(context_length=L, horizon=cfg.horizon, input_type=inp)
    rng = make_rng(cfg.seed, "test", "exp3", sys.key, L, inp)
    samples = [make_sample(sys, spec, rng) for _ in range(cfg.n_per_cell)]
    y = np.stack([s.y[:L] for s in samples])
    u = np.stack([s.u[:L] for s in samples])
    y_true = np.stack([s.y[L:] for s in samples])
    fac = CovBatch(y, u, np.stack([s.u[L:] for s in samples]))
    cfs = {sh: [make_counterfactual(s, sh, cfg.test_delta) for s in samples] for sh in cfg.shapes}
    arrays: dict = {"y_true": y_true.astype(np.float32), "y_scale": y.std(-1).astype(np.float32)}
    rec: dict = {"key": cell_key(sys.key, L, inp), "system": sys.key, "family": sys.family, "L": L, "input": inp,
                 "setting": setting_of(sys.key, cfg.train_systems), "methods": {}}
    for sh in cfg.shapes:
        arrays[f"e_{sh}"] = np.stack([c.effect for c in cfs[sh]]).astype(np.float32)
        arrays[f"keep_{sh}"] = M.effect_mask(arrays[f"e_{sh}"], y.std(-1), cfg.eff_thresh)
    seed_cell = int(make_rng(cfg.seed, "test", "exp3-dither", sys.key, L, inp).integers(2**31))
    for name, fn in methods.items():
        mrec: dict = {}
        if name == "arx":
            dus = np.stack([np.stack([c.du for c in cfs[sh]]) for sh in cfg.shapes], 1)
            eff, _ = arx_effects(y, u, dus)
            for si, sh in enumerate(cfg.shapes):
                arrays[f"ehat_{name}_{sh}"] = eff[:, si].astype(np.float32)
        else:
            q_fac = fn(fac, seed_cell)
            arrays[f"qfac_{name}"] = q_fac.astype(np.float32)
            mi = w.median_idx
            mrec["mase"] = float(np.median(M.mase(q_fac[:, mi], y_true, y)))
            mrec["wql"] = M.pooled_wql(q_fac, w.quantiles, y_true)
            for sh in cfg.shapes:
                cfb = CovBatch(y, u, np.stack([c.u_future_cf for c in cfs[sh]]))
                arrays[f"ehat_{name}_{sh}"] = (fn(cfb, seed_cell)[:, mi] - q_fac[:, mi]).astype(np.float32)
        for sh in cfg.shapes:
            k = arrays[f"keep_{sh}"]
            g = M.effect_gain(arrays[f"ehat_{name}_{sh}"][k], arrays[f"e_{sh}"][k])
            mrec[f"g_{sh}"] = float(np.median(g)) if len(g) else float("nan")
        rec["methods"][name] = mrec
    run.save_arrays(save_key or rec["key"], **arrays)
    return rec


def build_methods(w, cfg: Exp3Config, r1: dict, di: dict, gates: torch.Tensor, ft: dict) -> tuple[dict, dict]:
    hk = w.hooked
    b = r1["best"]
    G1 = r1_gain(hk.n_layers, hk.n_heads, [tuple(x) for x in r1["ranking"][: b["k"]]], b["gamma"])
    G2 = gates.clone()
    sg = di["best"]["sigma"]
    methods = {
        "zeroshot": lambda bt, s: w.predict(bt, bt.H),
        "r1": lambda bt, s: w.predict_hooked(bt, gain_spec(G1)),
        "r2": lambda bt, s: w.predict_hooked(bt, gain_spec(G2)),
        "dither": lambda bt, s: dither_predict(w, bt, sg, cfg.dither_K, seed=s),
    }
    params = {"zeroshot": 0, "r1": 0, "r2": int(G2.numel()), "dither": 0, "arx": None}
    from covattn.model import Chronos2Wrapper

    for mode in ("full", "lora"):
        if mode in ft:
            wf = Chronos2Wrapper(ft[mode]["path"], device=str(w.device), amp=cfg.amp, batch_size=cfg.batch_size)
            methods[f"ft_{mode}"] = lambda bt, s, wf=wf: wf.predict(bt, bt.H)
            params[f"ft_{mode}"] = ft[mode]["trainable_params"]
    methods["arx"] = None
    return methods, params


def add_methods(names: list[str], methods: dict, w, cfg: Exp3Config, run: RunDir, cells: list) -> None:
    """Evaluate `names` on already finished cells and merge them into the cell npz and metrics.jsonl.
    The samples are regenerated from the same seeds; the stored targets/effects must match exactly."""
    missing = [n for n in names if n not in methods]
    if missing:
        raise RuntimeError(f"methods {missing} are not available (fine-tuning failed or not run?)")
    sub = {n: methods[n] for n in names}
    recs = {r["key"]: r for r in run.read_metrics()}
    t_start = time.time()
    for i, (sys, L, inp) in enumerate(cells):
        key = cell_key(sys.key, L, inp)
        if key not in recs:
            log.warning("cell %s not finished - run without --add-methods first", key)
            continue
        old = dict(np.load(run.cell_path(key), allow_pickle=False))
        if all(f"ehat_{n}_{cfg.shapes[0]}" in old for n in names) and all(n in recs[key]["methods"] for n in names):
            continue
        tmp_run_key = key + "#add"
        rec = eval_cell(sub, w, cfg, sys, L, inp, run, save_key=tmp_run_key)
        new = dict(np.load(run.cell_path(tmp_run_key), allow_pickle=False))
        run.cell_path(tmp_run_key).unlink()
        for k in ("y_true",) + tuple(f"e_{sh}" for sh in cfg.shapes):
            if not np.array_equal(old[k], new[k]):
                raise RuntimeError(f"{key}: regenerated {k} differs from the stored one - not merging")
        own = {f"qfac_{n}" for n in names} | {f"ehat_{n}_{sh}" for n in names for sh in cfg.shapes}
        old.update({k: v for k, v in new.items() if k in own})
        run.save_arrays(key, **old)
        recs[key]["methods"].update(rec["methods"])
        log.info("[%d/%d] %s +%s %s", i + 1, len(cells), key, names,
                 {m: round(v.get(f"g_{cfg.shapes[0]}", float("nan")), 3) for m, v in rec["methods"].items()})
    tmp = run.metrics_file.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in recs.values()), encoding="utf-8")
    os.replace(tmp, run.metrics_file)
    log.info("added %s to %d cells in %s", names, len(recs), fmt_seconds(time.time() - t_start))


# ------------------------------------------------------------------------------------ summary
METHOD_LABEL = {"zeroshot": "Zero-shot", "r1": "R1 (top-k gain)", "r2": "R2 (gates)", "dither": "Dither (reimpl.)",
                "ft_full": "Full fine-tune", "ft_lora": "LoRA", "arx": "ARX (context fit)"}


TRAINING_FREE = ("zeroshot", "r1", "r2", "dither")


def write_summary(run: RunDir, cfg: Exp3Config) -> None:
    recs = run.read_metrics()
    if not recs:
        return
    cells = []
    for r in recs:
        z = dict(np.load(run.cell_path(r["key"]), allow_pickle=False))
        z.update({k: r[k] for k in ("system", "L", "input", "setting")})
        cells.append(z)
    methods = [m for m in METHOD_LABEL if f"ehat_{m}_{cfg.shapes[0]}" in cells[0]]
    params = json.loads((run.path / "method_params.json").read_text()) if (run.path / "method_params.json").exists() else {}
    rep = results_root() / "reports"
    (rep / "tables").mkdir(parents=True, exist_ok=True)

    def stats(cs, m):
        gs, rs, ns = [], [], []
        for c in cs:
            for sh in cfg.shapes:
                k = c[f"keep_{sh}"]
                eh, e = c[f"ehat_{m}_{sh}"][k], c[f"e_{sh}"][k]
                gs.append(M.effect_gain(eh, e))
                rs.append(M.shape_corr(eh, e))
                ns.append(M.whatif_nrmse(eh, e))
        g, r, n = (np.concatenate(x) if x else np.array([]) for x in (gs, rs, ns))
        mase = np.nan
        if f"qfac_{m}" in cs[0] and "qfac_zeroshot" in cs[0]:
            ratios = []
            for c in cs:
                yt, mi = c["y_true"], MEDIAN_IDX
                ratios.append(np.abs(c[f"qfac_{m}"][:, mi] - yt).mean(-1) / np.maximum(
                    np.abs(c["qfac_zeroshot"][:, mi] - yt).mean(-1), 1e-12))
            mase = float(np.median(np.concatenate(ratios)))
        return (float(np.median(g)) if len(g) else np.nan, float(np.nanmedian(r)) if len(r) else np.nan,
                float(np.median(n)) if len(n) else np.nan, mase, len(g))

    lines = [f"# Exp3 summary ({cfg.run_name})", "",
             f"Source: `{run.metrics_file}` ({len(cells)} cells, test split, delta={cfg.test_delta}). "
             f"Cells: {len(cfg.test_systems)} systems x L{cfg.contexts} x {cfg.input_types}; shapes {cfg.shapes}.",
             "Per method: median g / median shape r / median what-if NRMSE / median ratio of factual "
             "abs. error (median quantile) vs zero-shot; n = kept pairs.", ""]
    for f in ("r1_select.json", "dither_select.json"):
        if (run.path / f).exists():
            b = json.loads((run.path / f).read_text())["best"]
            lines.append(f"- {f}: chosen {b}")
    lines.append("")
    groups = [("all", cells)] + [(s, [c for c in cells if c["setting"] == s]) for s in SETTINGS]
    groups += [(f"input={i}", [c for c in cells if c["input"] == i]) for i in cfg.input_types]
    groups += [(f"L={L}", [c for c in cells if c["L"] == L]) for L in cfg.contexts]
    groups = [(n, cs) for n, cs in groups if cs]
    table: dict = {}
    for gname, cs in groups:
        lines += [f"## {gname} ({len(cs)} cells)", "", "| method | params | g | r | NRMSE | err ratio | n |",
                  "|---|---|---|---|---|---|---|"]
        for m in methods:
            s = stats(cs, m)
            table[(gname, m)] = s
            lines.append(f"| {METHOD_LABEL[m]} | {params.get(m, '-')} | {s[0]:.3f} | {s[1]:.3f} | {s[2]:.3f} | "
                         f"{s[3]:.3f} | {s[4]} |")
        lines.append("")
    out = rep / f"exp3_summary_{cfg.run_name}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    # LaTeX table 1: settings as column groups (g / what-if NRMSE), plus params; 14 numeric columns (with shape r)
    # do not fit the LNCS text width, so r is left to the markdown summary (Table 1r variant below has it).
    # Bold = best training-free method per column (g: closest to 1; NRMSE: lowest); fine-tuned models and ARX are
    # reference rows.
    sets = [s for s in SETTINGS if (s, methods[0]) in table]
    cand = [m in TRAINING_FREE for m in methods]
    short = {"r1": "R1", "r2": "R2", "dither": "Dither", "ft_full": "Full FT", "ft_lora": "LoRA", "arx": "ARX"}

    def params_str(m: str) -> str:
        p = params.get(m)
        return "--" if p is None else (f"{p/1e6:.1f}M" if p >= 1e6 else (f"{p/1e3:.0f}k" if p >= 1e4 else str(p)))

    def build(metrics: list[tuple[int, str, str]]) -> str:
        nm = len(metrics)
        cols = {(s, j): mark_best([table[(s, m)][j] for m in methods], ".2f", mode, cand)
                for s in sets for j, mode, _ in metrics}
        hdr = ["Method & Params & " + " & ".join(f"\\multicolumn{{{nm}}}{{c}}{{{s}}}" for s in sets) + " \\\\",
               " ".join(f"\\cmidrule(lr){{{3 + nm * i}-{2 + nm * (i + 1)}}}" for i in range(len(sets))),
               " & & " + " & ".join([" & ".join(h for _, _, h in metrics)] * len(sets)) + " \\\\"]
        rows = []
        for i, m in enumerate(methods):
            if m in ("ft_full", "arx") and i > 0:
                rows.append("\\midrule")
            rows.append([short.get(m, METHOD_LABEL[m]), params_str(m)] +
                        [cols[(s, j)][i] for s in sets for j, _, _ in metrics])
        sep = "3pt" if nm < 3 else "2pt"  # wide variant only fits in a smaller font (appendix: \footnotesize)
        return f"\\setlength{{\\tabcolsep}}{{{sep}}}\n" + rows_to_tex("l r " + " ".join(["c" * nm] * len(sets)),
                                                                      hdr, rows)

    (rep / "tables" / f"table1_synthetic_{cfg.run_name}.tex").write_text(
        build([(0, "one", "$g$"), (2, "min", "NRMSE")]), encoding="utf-8")
    (rep / "tables" / f"table1r_synthetic_with_r_{cfg.run_name}.tex").write_text(  # wide variant (appendix)
        build([(0, "one", "$g$"), (1, "max", "$r$"), (2, "min", "NRMSE")]), encoding="utf-8")
    log.info("summary written: %s", out)


MEDIAN_IDX = 10  # 21 Chronos-2 quantiles 0.01..0.99; index 10 = 0.5 (docs/model_notes.md)


# ------------------------------------------------------------------------------------ main
def main() -> None:
    cfg, args = parse_args()
    LAM[0] = cfg.lam
    run = RunDir("exp3", cfg.run_name)
    setup_logging(run.path / "log.txt")
    log.info("config: %s", json.dumps(asdict(cfg)))
    if args.summary_only:
        write_summary(run, cfg)
        return
    run.write_config(cfg, resume=args.resume, ignore_keys=("device", "batch_size"))
    from covattn.model import Chronos2Wrapper

    w = Chronos2Wrapper(cfg.model_id, device=cfg.device, amp=cfg.amp, batch_size=cfg.batch_size)
    torch.manual_seed(cfg.seed)

    # rough time estimate from measured throughput at the longest context
    tps = w.measure_throughput(max(cfg.contexts), cfg.horizon, n_tasks=64)
    n_meth = 4 + cfg.dither_K + (0 if args.no_finetune else 2)
    n_cells = len(cfg.test_systems) * len(cfg.contexts) * len(cfg.input_types)
    fw = n_cells * cfg.n_per_cell * (1 + len(cfg.shapes)) * n_meth
    fw += cfg.val_batches * cfg.val_pairs * 2 * (1 + len(cfg.r1_k) * len(cfg.r1_gamma) * 2 + len(cfg.dither_sigma) * cfg.dither_K)
    fw += cfg.r2_steps * cfg.r2_pairs * 2 * 6
    sec = fw / tps
    if not args.no_finetune:
        sec += 2 * cfg.ft_steps * 0.5
    est = time_estimates(sec, w.device)
    log.info("ESTIMATED TIME (very rough; hooked passes and FT counted coarsely) - here: %s | RTX4060: %s | "
             "L4: %s | A100(guess): %s", *(fmt_seconds(est[k]) for k in ("here", "RTX4060", "L4", "A100")))

    r1, di = stage_select(w, cfg, run)
    log.info("R1 chosen: %s | dither chosen: %s", r1["best"], di["best"])
    gates = stage_r2(w, cfg, run)
    log.info("R2 gates: mean %.3f, min %.3f, max %.3f", float(gates.mean()), float(gates.min()), float(gates.max()))
    ft = {} if args.no_finetune else stage_ft(w, cfg, run)
    methods, params = build_methods(w, cfg, r1, di, gates, ft)
    (run.path / "method_params.json").write_text(json.dumps(params, indent=1))

    lk = systems_by_key()
    cells = [(lk[s], L, inp) for L in cfg.contexts for s in cfg.test_systems for inp in cfg.input_types]
    if args.add_methods:
        add_methods(args.add_methods, methods, w, cfg, run, cells)
        write_summary(run, cfg)
        return
    done = run.done_keys() if args.resume else set()
    todo = [c for c in cells if cell_key(c[0].key, c[1], c[2]) not in done]
    log.info("eval cells: %d total, %d to run; methods %s", len(cells), len(todo), list(methods))
    t_start = time.time()
    for i, (sys, L, inp) in enumerate(todo):
        t0 = time.time()
        rec = eval_cell(methods, w, cfg, sys, L, inp, run)
        rec["device"] = str(w.device)
        run.append_metrics(rec)
        el = time.time() - t_start
        log.info("[%d/%d] %s %.1fs ETA %s | g_step: %s", i + 1, len(todo), rec["key"], time.time() - t0,
                 fmt_seconds(el / (i + 1) * (len(todo) - i - 1)),
                 {m: round(v.get(f"g_{cfg.shapes[0]}", float("nan")), 3) for m, v in rec["methods"].items()})
    write_summary(run, cfg)
    if len(run.done_keys()) == len(cells):
        run.mark_done({"n_cells": len(cells)})
        log.info("all %d cells done -> done.marker", len(cells))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - make sure the traceback reaches log.txt on Drive
        log.exception("Exp3 crashed")
        raise
