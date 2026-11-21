"""Exp14 - training-free gate deciding per window whether to apply the repair (protocol docs/gate_protocol.md).

Gate (src/covattn/gate.py): input share s of the best covariate (ARX vs AR on the context) and attenuation index
rho of zero-shot Chronos-2 against the context ARX for a +1 std step probe; repair iff s >= tau_s and rho < tau_rho.
One decision per window (from the factual context and future input), used for both members of a what-if pair.

Parts (run in this order; each is resumable):
  select    (a) Exp3 selection sets (exp3/<run> config, split "val") and (b) semi-synthetic EPF pairs (kappa down
            to 0.1, Exp6 windows shifted by 13 weeks). When all cells exist, the thresholds of R2 and R1 are chosen
            (criterion repair.objective_np, lambda = 1, mean of (a) and (b); ties within 0.002 -> more conservative)
            and frozen in exp14/<run>/gate_select.json (never overwritten).
  confirm   DaISy (13 datasets, scripts/daisy_prepare.py), windows as Exp13; needs gate_select.json.
  describe  data already seen (not confirmatory): Exp13 rigs, Exp3 synthetic test, LCL (Exp12 cells), fev-bench
            (Exp6 cells). Only zero-shot and the probe are run; the repaired forecasts are read from those runs.

  python scripts/exp14_gate.py --quick --part select confirm describe              # Colab, first
  python scripts/exp14_gate.py --resume --part select --sysid-data D --daisy-data E  # Colab
  python scripts/exp14_gate.py --summary-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp3_repair as E3  # noqa: E402
import exp6_real_world as E6  # noqa: E402
import exp12_lcl_whatif as E12  # noqa: E402
import exp13_sysid_real as E13  # noqa: E402

from covattn import gate as G  # noqa: E402
from covattn import metrics as M  # noqa: E402
from covattn.io_utils import RunDir, fmt_seconds, results_root, setup_logging  # noqa: E402
from covattn.model import CovBatch  # noqa: E402
from covattn.real_data import EPF, fev_windows, load_epf, load_fev_task  # noqa: E402
from covattn.repair import _ql_np, gain_spec, objective_np, r1_gain  # noqa: E402
from covattn.systems import delta_u, make_counterfactual, make_rng, make_sample, transfer_coeffs  # noqa: E402
from covattn.systems import SampleSpec  # noqa: E402

log = logging.getLogger("exp14")
REPAIRS = ["r2", "r1"]  # R2 primary
LEVELS9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
LABEL = {"zeroshot": "Zero-shot", "r2": "R2 (ungated)", "r2_gated": "R2 gated", "r1": "R1 (ungated)",
         "r1_gated": "R1 gated"}
PARTS = ["select", "confirm", "describe"]
DESCRIBE = ["exp13", "exp3", "lcl", "fev"]
INPUTS: dict = {}
DAISY = ["CD_player_arm", "ballbeam", "dryer", "dryer2", "evaporator", "exchanger", "flexible_structure", "flutter",
         "glassfurnace", "heating_system", "robot_arm", "thermic_res_wall", "winding"]


@dataclass
class Exp14Config:
    run_name: str = "main"
    quick: bool = False
    exp3_run: str = "v2"
    exp6_run: str = "main"
    exp12_run: str = "main"
    exp13_run: str = "main"
    lam: float = 1.0
    tie_tol: float = 0.002
    eff_thresh: float = 0.05
    # selection (a): Exp3 val sets; (quick: fewer)
    val_batches: int | None = None  # None = as in the exp3 config
    val_pairs: int | None = None
    # selection (b): semi-synthetic EPF
    semi_backgrounds: list[str] = field(default_factory=lambda: list(EPF))
    semi_systems: list[str] = field(default_factory=lambda: ["F1", "F2_a0.8", "F3_a0.8_d4", "F4_P24_z0.2"])
    semi_kappas: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 1.0])
    semi_contexts: list[int] = field(default_factory=lambda: [512, 2048])
    semi_windows: int = 40
    semi_stride: int = 168
    semi_cov_shift: int = 26 * 168  # as Exp6
    semi_window_shift: int = 13 * 168  # validation windows = Exp6 windows shifted back by 13 weeks
    semi_burn: int = 500
    shapes: list[str] = field(default_factory=lambda: ["step", "pulse"])
    horizon: int = 64
    # confirmation (DaISy) - as Exp13
    daisy: list[str] = field(default_factory=lambda: list(DAISY))
    contexts: list[int] = field(default_factory=lambda: [512, 2048])
    windows_per_bench: int = 64
    silent: float = 0.01
    n_boot: int = 2000
    # describe
    describe: list[str] = field(default_factory=lambda: list(DESCRIBE))
    describe_max_cells: int = 0  # 0 = all (quick: a few)
    fev_n_series: int | None = None  # None = as in the exp6 config
    seed: int = 0
    model_id: str = "amazon/chronos-2"
    amp: str = "off"
    device: str = "auto"
    batch_size: int = 128


def parse_args() -> tuple[Exp14Config, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--part", nargs="+", choices=PARTS, default=PARTS)
    p.add_argument("--describe", nargs="+", choices=DESCRIBE, default=None)
    p.add_argument("--inputs-root", default=None, help="where exp3/6/12/13 and lcl/ are read (default RESULTS_DIR)")
    p.add_argument("--sysid-data", default=None, help="dir with the Exp13 records.npz / records.json")
    p.add_argument("--daisy-data", default=None, help="dir with the DaISy records.npz / records.json")
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    INPUTS["root"] = Path(a.inputs_root) if a.inputs_root else results_root()
    INPUTS["sysid"] = Path(a.sysid_data) if a.sysid_data else repo / "data" / "sysid" / "processed"
    INPUTS["daisy"] = Path(a.daisy_data) if a.daisy_data else repo / "data" / "daisy" / "processed"
    E12.INPUTS["root"] = INPUTS["root"]
    E13.INPUTS["root"] = INPUTS["root"]
    cfg = Exp14Config(amp=a.amp, device=a.device, batch_size=a.batch_size, quick=a.quick)
    if a.quick:
        cfg.run_name, cfg.val_batches, cfg.val_pairs = "quick", 1, 8
        cfg.semi_backgrounds, cfg.semi_systems, cfg.semi_kappas = ["epf_be"], ["F2_a0.8"], [0.1]
        cfg.semi_contexts, cfg.semi_windows = [512], 4
        cfg.daisy, cfg.contexts, cfg.windows_per_bench, cfg.n_boot = ["dryer", "winding"], [512], 4, 100
        cfg.describe_max_cells, cfg.fev_n_series = 1, 2
    if a.describe is not None:
        cfg.describe = a.describe
    if a.run_name is not None:
        cfg.run_name = a.run_name
    return cfg, a


def _cfg(run: str) -> dict:
    return json.loads((INPUTS["root"] / run / "config.json").read_text(encoding="utf-8"))["config"]


def exp3_config(cfg: Exp14Config) -> E3.Exp3Config:
    d = _cfg(f"exp3/{cfg.exp3_run}")
    c = E3.Exp3Config(**{k: v for k, v in d.items() if k in E3.Exp3Config.__dataclass_fields__})
    if cfg.val_batches is not None:
        c.val_batches = cfg.val_batches
    if cfg.val_pairs is not None:
        c.val_pairs = cfg.val_pairs
    return c


# ------------------------------------------------------------------------------------ model and gate
def build_methods(w, cfg: Exp14Config) -> tuple[dict, dict]:
    d3 = INPUTS["root"] / "exp3" / cfg.exp3_run
    r1 = json.loads((d3 / "r1_select.json").read_text())
    k, gamma = r1["best"]["k"], r1["best"]["gamma"]
    hk = w.hooked
    G1 = r1_gain(hk.n_layers, hk.n_heads, [tuple(x) for x in r1["ranking"][:k]], gamma)
    gates = torch.tensor(json.loads((d3 / "r2_gates.json").read_text())["gates"])
    methods = {"zeroshot": lambda bt: w.predict(bt, bt.H),
               "r1": lambda bt: w.predict_hooked(bt, gain_spec(G1)),
               "r2": lambda bt: w.predict_hooked(bt, gain_spec(gates))}
    return methods, {"k": k, "gamma": gamma, "exp3_dir": str(d3), "quantiles": list(map(float, w.quantiles)),
                     "median_idx": int(w.median_idx)}


def gate_arrays(w, fac: CovBatch, q_zs: np.ndarray) -> dict:
    """s, rho, k, du per window. q_zs: zero-shot forecast of `fac` (N, Q, H)."""
    st = G.batch_stats(fac.target, fac.cov_past, fac.H)
    probe = CovBatch(fac.target, fac.cov_past, G.probe_future(fac.cov_future, st))
    q_p = w.predict(probe, fac.H)
    mi = w.median_idx
    rho = G.attenuation_index(q_p[:, mi] - q_zs[:, mi], st)
    return {"gate_s": np.asarray([x.s for x in st], np.float32), "gate_rho": rho.astype(np.float32),
            "gate_k": np.asarray([x.k for x in st], np.int16), "gate_du": np.asarray([x.du for x in st], np.float32)}


# ------------------------------------------------------------------------------------ part select
def select_jobs(cfg: Exp14Config) -> list[tuple]:
    jobs = [("val", i) for i in range(exp3_config(cfg).val_batches)]
    for bg in cfg.semi_backgrounds:
        src = EPF[(EPF.index(bg) + 1) % len(EPF)]
        for s in cfg.semi_systems:
            for kp in cfg.semi_kappas:
                for L in cfg.semi_contexts:
                    jobs.append(("semi", (bg, src, s, kp, L)))
    return jobs


def job_key(kind: str, a) -> str:
    if kind == "val":
        return f"val|{a}"
    bg, _, s, kp, L = a
    return f"semi|{bg}|{s}|k{kp:g}|L{L}"


def run_val(i: int, vs: list, w, methods: dict) -> tuple[dict, dict]:
    pb = vs[i]
    arrays = {"y_ctx": pb.y_ctx.astype(np.float32), "y_true": pb.y_true.astype(np.float32),
              "e": pb.e.astype(np.float32)}
    for m, fn in methods.items():
        arrays[f"qfac_{m}"] = fn(pb.fac).astype(np.float32)
        arrays[f"qcf_{m}"] = fn(pb.cf).astype(np.float32)
    arrays.update(gate_arrays(w, pb.fac, arrays["qfac_zeroshot"]))
    return arrays, {"L": pb.fac.L, "n": pb.n}


def semi_inputs(bg: str, cov_src: str, sys_key: str, kappa: float, L: int, cfg: Exp14Config, cache: dict):
    """Exp6 part `semi` construction (run_semi_cell) with the origins shifted back by cfg.semi_window_shift."""
    for name in (bg, cov_src):
        if name not in cache:
            cache[name] = load_epf(name)
    y_bg_all = cache[bg].target.astype(np.float64)
    u_all = np.roll(cache[cov_src].covs[0].astype(np.float64), cfg.semi_cov_shift)
    u_all = (u_all - u_all.mean()) / u_all.std()
    B, A = transfer_coeffs(E6._sys(sys_key))
    H, T = cfg.horizon, min(len(y_bg_all), len(u_all))
    origins = [T - H - cfg.semi_window_shift - i * cfg.semi_stride for i in range(cfg.semi_windows)][::-1]
    origins = [o for o in origins if o - L - cfg.semi_burn >= 0]
    ys, us, ufs, yts, es = [], [], [], [], []
    for o in origins:
        a = o - L - cfg.semi_burn
        u = u_all[a:o + H]
        x = lfilter(B, A, u)[cfg.semi_burn:]
        bg_ = y_bg_all[o - L:o + H]
        b = kappa * np.std(bg_[:L]) / max(np.std(x[:L]), 1e-12)
        y = bg_ + b * x
        ys.append(y[:L]); yts.append(y[L:]); us.append(u[cfg.semi_burn:cfg.semi_burn + L]); ufs.append(u[-H:])
        es.append([b * lfilter(B, A, delta_u(sh, H, 1.0)) for sh in cfg.shapes])
    du = np.stack([delta_u(sh, H, 1.0) for sh in cfg.shapes]).astype(np.float32)  # (S, H)
    return (np.asarray(ys, np.float32), np.asarray(us, np.float32), np.asarray(ufs, np.float32),
            np.asarray(yts, np.float32), np.asarray(es, np.float32), du, origins)


def run_semi(args: tuple, w, methods: dict, cfg: Exp14Config, cache: dict) -> tuple[dict, dict]:
    y, uc, uf, yt, e, du, origins = semi_inputs(*args, cfg, cache)
    arrays = {"y_ctx": y, "y_true": yt}
    for si, sh in enumerate(cfg.shapes):
        arrays[f"e_{sh}"] = e[:, si]
    fac = CovBatch(y, uc, uf)
    for m, fn in methods.items():
        arrays[f"qfac_{m}"] = fn(fac).astype(np.float32)
        for si, sh in enumerate(cfg.shapes):
            arrays[f"qcf_{m}_{sh}"] = fn(CovBatch(y, uc, uf + du[si])).astype(np.float32)
    arrays.update(gate_arrays(w, fac, arrays["qfac_zeroshot"]))
    return arrays, {"L": args[4], "n": len(y), "origins": origins}


def _pick(z: dict, name: str, m: str, on: np.ndarray) -> np.ndarray:
    a, b = z[name.format(m=m)], z[name.format(m="zeroshot")]
    return np.where(on.reshape((-1,) + (1,) * (a.ndim - 1)), a, b)


def crit_val(cells: list[dict], m: str, on_fn, quantiles, mi, lam) -> float:
    """Exp3 R1 selection criterion: mean over the val batches of repair.objective_np."""
    vals = []
    for z in cells:
        on = on_fn(z)
        pb = SimpleNamespace(e=z["e"].astype(np.float64), y_ctx=z["y_ctx"].astype(np.float64),
                             y_true=z["y_true"].astype(np.float64))
        vals.append(objective_np(_pick(z, "qfac_{m}", m, on).astype(np.float64),
                                 _pick(z, "qcf_{m}", m, on).astype(np.float64), pb, quantiles, mi, lam)["obj"])
    return float(np.mean(vals))


def objective_parts(q_fac, q_cf, e, y_ctx_std, y_true, quantiles, mi, thresh) -> tuple[float, int, float, int]:
    """Sums behind repair.objective_np, so that pairs can be pooled across cells of different L:
    (sum of eff over kept pairs, n kept, sum of scaled QL, n pairs)."""
    keep = M.effect_mask(e, y_ctx_std, thresh)
    e_hat = q_cf[:, mi] - q_fac[:, mi]
    eff = ((e_hat - e) ** 2).sum(-1) / np.maximum((e ** 2).sum(-1), 1e-12)
    ql = _ql_np(q_fac, y_true, quantiles) / np.maximum(y_ctx_std, 1e-6)
    return float(eff[keep].sum()), int(keep.sum()), float(ql.sum()), len(ql)


def crit_semi(cells: list[dict], m: str, on_fn, quantiles, mi, lam, shapes, thresh) -> float:
    """Same criterion pooled over all (window, shape) pairs of the semi-synthetic cells."""
    se = ne = sq = nq = 0
    for z in cells:
        on = on_fn(z)
        ys = z["y_ctx"].astype(np.float64).std(-1)
        qf = _pick(z, "qfac_{m}", m, on).astype(np.float64)
        for sh in shapes:
            qc = _pick(z, f"qcf_{{m}}_{sh}", m, on).astype(np.float64)
            a, b, c, d = objective_parts(qf, qc, z[f"e_{sh}"].astype(np.float64), ys, z["y_true"].astype(np.float64),
                                         quantiles, mi, thresh)
            se, ne, sq, nq = se + a, ne + b, sq + c, nq + d
    return (se / ne if ne else float("nan")) + lam * sq / nq


def choose_thresholds(run: RunDir, cfg: Exp14Config, info: dict) -> dict:
    out = run.path / "gate_select.json"
    if out.exists():
        log.info("thresholds already frozen: %s", out)
        return json.loads(out.read_text())
    recs = {r["key"]: r for r in run.read_metrics() if r.get("part") == "select"}
    val = [dict(np.load(run.cell_path(k), allow_pickle=False)) for k in sorted(recs) if k.startswith("val|")]
    semi = [dict(np.load(run.cell_path(k), allow_pickle=False)) for k in sorted(recs) if k.startswith("semi|")]
    qs, mi = info["quantiles"], info["median_idx"]
    res: dict = {"time": time.ctime(), "n_val_batches": len(val), "n_semi_cells": len(semi),
                 "criterion": "mean of (a) Exp3 val objective_np and (b) pooled semi objective, lambda = %g" % cfg.lam}
    never = lambda z: np.zeros(len(z["gate_s"]), bool)  # noqa: E731
    res["zeroshot"] = {"a": crit_val(val, "r2", never, qs, mi, cfg.lam),
                       "b": crit_semi(semi, "r2", never, qs, mi, cfg.lam, cfg.shapes, cfg.eff_thresh)}
    res["zeroshot"]["crit"] = 0.5 * (res["zeroshot"]["a"] + res["zeroshot"]["b"])
    for m in REPAIRS:
        rows = []
        for ts, tr in G.grid():
            fn = lambda z, ts=ts, tr=tr: G.gate_on(z["gate_s"], z["gate_rho"], ts, tr)  # noqa: E731
            a = crit_val(val, m, fn, qs, mi, cfg.lam)
            b = crit_semi(semi, m, fn, qs, mi, cfg.lam, cfg.shapes, cfg.eff_thresh)
            fire = float(np.mean(np.concatenate([fn(z) for z in val + semi])))
            rows.append({"tau_s": ts, "tau_rho": tr, "a": a, "b": b, "crit": 0.5 * (a + b), "fire": fire})
        scores = {(r["tau_s"], r["tau_rho"]): r["crit"] for r in rows}
        ts, tr = G.select(scores, cfg.tie_tol)
        best = next(r for r in rows if r["tau_s"] == ts and r["tau_rho"] == tr)
        ungated = next(r for r in rows if r["tau_s"] == 0.0 and not np.isfinite(r["tau_rho"]))
        res[m] = {"tau_s": ts, "tau_rho": tr, "chosen": best, "ungated": ungated, "grid": rows}
        log.info("%s: chosen tau_s=%g tau_rho=%g crit=%.4f (ungated %.4f, zero-shot %.4f), fires %.1f%%", m, ts, tr,
                 best["crit"], ungated["crit"], res["zeroshot"]["crit"], 100 * best["fire"])
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def selection_complete(run: RunDir, cfg: Exp14Config) -> bool:
    return {job_key(*j) for j in select_jobs(cfg)} <= run.done_keys()


def load_thresholds(run: RunDir) -> dict:
    f = run.path / "gate_select.json"
    if not f.exists():
        raise FileNotFoundError(f"{f}: run --part select first (thresholds must be frozen before confirm / describe)")
    d = json.loads(f.read_text())
    return {m: (float(d[m]["tau_s"]), float(d[m]["tau_rho"])) for m in REPAIRS}


# ------------------------------------------------------------------------------------ part confirm (DaISy)
def load_records(path: Path) -> list[dict]:
    meta = json.loads((path / "records.json").read_text(encoding="utf-8"))
    z = np.load(path / "records.npz", allow_pickle=False)
    for m in meta:
        m["u"], m["y"] = z[f"u{m['i']}"].astype(np.float64), z[f"y{m['i']}"].astype(np.float64)
    return meta


def window_batch(recs: list[dict], win: list[tuple[int, int]], L: int, H: int):
    y = np.stack([recs[i]["y"][o - L:o] for i, o in win])
    uc = np.stack([recs[i]["u"][:, o - L:o] for i, o in win])
    uf = np.stack([recs[i]["u"][:, o:o + H] for i, o in win])
    yt = np.stack([recs[i]["y"][o:o + H] for i, o in win])
    return y, uc, uf, yt


def run_daisy(bench: str, L: int, recs_all: list[dict], w, methods: dict, cfg: Exp14Config) -> tuple[dict, dict]:
    recs = [r for r in recs_all if r["bench"] == bench]
    H = cfg.horizon
    win = E13.windows(recs, L, H, cfg.windows_per_bench)
    if not win:
        raise ValueError(f"{bench}: no record with N >= {L + H}")
    y, uc, uf, yt = window_batch(recs, win, L, H)
    lv = [int(np.argmin(np.abs(np.asarray(w.quantiles) - l))) for l in LEVELS9]
    arrays = {"y_true": yt.astype(np.float32), "y_ctx_std": y.std(1).astype(np.float32),
              "rec": np.asarray([recs[i]["i"] for i, _ in win], np.int32), "origin": np.asarray([o for _, o in win]),
              "rec_std": np.asarray([np.std(recs[i]["y"]) for i, _ in win], np.float32)}
    fac = CovBatch(y, uc, uf)
    q0 = None
    for m, fn in methods.items():
        q = fn(fac)
        q0 = q if m == "zeroshot" else q0
        arrays[f"q9_{m}"], arrays[f"med_{m}"] = q[:, lv].astype(np.float32), q[:, w.median_idx].astype(np.float32)
    arrays.update(gate_arrays(w, fac, q0))
    return arrays, {"bench": bench, "L": L, "n": len(win), "k": int(uc.shape[1]),
                    "records": sorted({recs[i]["name"] for i, _ in win})}


# ------------------------------------------------------------------------------------ part describe
def _rd(path: Path) -> RunDir:
    """Read-only view of an existing run directory (no mkdir)."""
    rd = RunDir.__new__(RunDir)
    rd.path, rd.metrics_file = path, path / "metrics.jsonl"
    return rd


def _cells(rel: str) -> list[tuple[dict, str]]:
    rd = _rd(INPUTS["root"] / rel)
    return [(r, f"{rel}/cells/{rd.cell_path(r['key']).name}") for r in rd.read_metrics()]


SRC_FIELDS = ("key", "system", "L", "input", "setting", "truth", "noise_sq", "task", "bench")


def zs_and_gate(w, fac: CovBatch) -> dict:
    q = w.predict(fac, fac.H)
    out = gate_arrays(w, fac, q)
    out["med_zeroshot_rerun"] = q[:, w.median_idx].astype(np.float32)
    return out


def describe_jobs(cfg: Exp14Config) -> list[tuple[str, dict, Path]]:
    jobs = []
    srcs = {"exp13": f"exp13/{cfg.exp13_run}", "exp3": f"exp3/{cfg.exp3_run}", "lcl": f"exp12/{cfg.exp12_run}",
            "fev": f"exp6/{cfg.exp6_run}"}
    for part in cfg.describe:
        cs = _cells(srcs[part])
        if part == "fev":
            cs = [c for c in cs if c[0].get("part") == "fev"]
        if part == "exp3":
            cs = [c for c in cs if "system" in c[0]]
        if cfg.describe_max_cells:
            cs = cs[:cfg.describe_max_cells]
        jobs += [(part, r, p) for r, p in cs]
    return jobs


def describe_key(part: str, r: dict) -> str:
    return f"{part}|{r['key']}"


def _check(name: str, a: np.ndarray, b: np.ndarray) -> None:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or not np.allclose(a, b, rtol=1e-4, atol=1e-5, equal_nan=True):
        raise ValueError(f"{name}: regenerated windows do not match the stored ones")


def run_describe(part: str, r: dict, rel: str, w, cfg: Exp14Config, cache: dict) -> tuple[dict, dict]:
    z = np.load(INPUTS["root"] / rel, allow_pickle=False)
    if part == "exp13":
        if "sysid" not in cache:
            cache["sysid"] = load_records(INPUTS["sysid"])
        by_i = {m["i"]: m for m in cache["sysid"]}
        L, H = r["L"], z["y_true"].shape[1]
        recs = [by_i[int(i)] for i in z["rec"]]
        win = list(zip(range(len(recs)), (int(o) for o in z["origin"])))
        y, uc, uf, yt = window_batch(recs, win, L, H)
        _check(r["key"], yt, z["y_true"])
        fac = CovBatch(y, uc, uf)
    elif part == "exp3":
        c3 = exp3_config(cfg)
        sys_ = E3.systems_by_key()[r["system"]]
        spec = SampleSpec(context_length=r["L"], horizon=c3.horizon, input_type=r["input"])
        rng = make_rng(c3.seed, "test", "exp3", sys_.key, r["L"], r["input"])
        ss = [make_sample(sys_, spec, rng) for _ in range(c3.n_per_cell)]
        L = r["L"]
        _check(r["key"], np.stack([s.y[L:] for s in ss]), z["y_true"])
        fac = CovBatch(np.stack([s.y[:L] for s in ss]), np.stack([s.u[:L] for s in ss]),
                       np.stack([s.u[L:] for s in ss]))
    elif part == "lcl":
        c12 = _cfg(f"exp12/{cfg.exp12_run}")
        if ("lcl", r["truth"]) not in cache:
            cache[("lcl", r["truth"])] = E12.load_truth(r["truth"])
        tr = cache[("lcl", r["truth"])]
        L, H = r["L"], c12["horizon"]
        full = tr["full"]
        y_all, p_all = full["tou_kwh"].to_numpy(np.float64), full["price"].to_numpy(np.float64)
        os_ = E12.origins(tr, L, H, c12["max_origins"])
        _check(r["key"], np.stack([y_all[o:o + H] for o in os_]), z["y_true"])
        fac = CovBatch(np.stack([y_all[o - L:o] for o in os_]), np.stack([p_all[o - L:o] for o in os_]),
                       np.stack([p_all[o:o + H] for o in os_]))
    elif part == "fev":
        c6 = _cfg(f"exp6/{cfg.exp6_run}")
        task = {t["config"]: t for t in E6.TASKS}[r["task"]]
        n_series = cfg.fev_n_series or c6["n_series"]
        series = load_fev_task(task, n_series)
        win = fev_windows(series, task["horizon"], task["num_windows"], c6["n_val"], c6["L_max"],
                          min_ctx=max(task["seasonality"] + 1, 8))
        ctx, cp, cf, yt, _ = win["test"]
        if cfg.fev_n_series is None:
            _check(r["key"], yt, z["y_test"])
        fac = CovBatch(ctx, cp, cf)
    else:
        raise ValueError(part)
    arrays = zs_and_gate(w, fac)
    return arrays, {"source": rel, "src": {k: r[k] for k in SRC_FIELDS if k in r}, "n": fac.n, "L": fac.L,
                    "H": fac.H, "k": fac.k}


# ------------------------------------------------------------------------------------ summary
def _gm(x) -> float:
    return float(np.exp(np.nanmean(x)))


def _wql9(q9: np.ndarray, y: np.ndarray) -> np.ndarray:
    lv = np.asarray(LEVELS9)[None, :, None]
    yy = y[:, None, :]
    return (2.0 * np.abs((yy - q9) * ((yy <= q9).astype(np.float64) - lv))).mean((1, 2))


def _fmt_t(t: tuple[float, float]) -> str:
    return f"tau_s = {t[0]:g}, tau_rho = {t[1]:g}"


def summary_select(run: RunDir) -> list[str]:
    f = run.path / "gate_select.json"
    if not f.exists():
        return ["## Selection", "", "not finished (gate_select.json missing)", ""]
    d = json.loads(f.read_text())
    L = ["## Selection (simulated / semi-synthetic only)", "",
         f"{d['n_val_batches']} Exp3 val batches (a), {d['n_semi_cells']} semi-synthetic cells (b); {d['criterion']}; "
         "lower = better. Frozen " + d["time"] + ".", "",
         "| | tau_s | tau_rho | criterion | (a) | (b) | windows gated on |", "|---|---|---|---|---|---|---|",
         f"| Zero-shot | - | - | {d['zeroshot']['crit']:.4f} | {d['zeroshot']['a']:.4f} | {d['zeroshot']['b']:.4f} | 0% |"]
    for m in REPAIRS:
        for lab, r in ((f"{m.upper()} ungated", d[m]["ungated"]), (f"{m.upper()} gated (chosen)", d[m]["chosen"])):
            L.append(f"| {lab} | {r['tau_s']:g} | {r['tau_rho']:g} | {r['crit']:.4f} | {r['a']:.4f} | {r['b']:.4f} | "
                     f"{100 * r['fire']:.0f}% |")
    L += ["", "Full grids are in gate_select.json.", ""]
    return L


def _gated_med(z: dict, m: str, on: np.ndarray, key: str = "med_{m}") -> np.ndarray:
    return _pick(z, key, m, on)


def summary_confirm(run: RunDir, cfg: Exp14Config, th: dict, rng, agg: dict) -> list[str]:
    recs = [r for r in run.read_metrics() if r.get("part") == "confirm"]
    if not recs:
        return ["## Confirmation (DaISy)", "", "not run yet", ""]
    cand = ["r2", "r2_gated", "r1", "r1_gated"]
    out = ["## Confirmation on DaISy (pre-declared)", "",
           f"Thresholds from selection: R2 {_fmt_t(th['r2'])}; R1 {_fmt_t(th['r1'])}. MAE ratio to zero-shot "
           "(geometric mean over windows, then datasets; 95% bootstrap CI over windows within datasets); silent "
           f"windows (context or horizon std < {cfg.silent:g} x record std) removed.", ""]
    agg["confirm"] = {}
    for L in cfg.contexts:
        cells = {r["bench"]: (r, dict(np.load(run.cell_path(r["key"]), allow_pickle=False)))
                 for r in recs if r["L"] == L}
        if not cells:
            continue
        logs, fire, nwin, drop = {}, {}, {}, 0
        for b, (r, z) in cells.items():
            k = (z["y_ctx_std"] >= cfg.silent * z["rec_std"]) & (z["y_true"].std(1) >= cfg.silent * z["rec_std"])
            drop += int((~k).sum())
            zz = {kk: v[k] for kk, v in z.items() if v.shape[:1] == k.shape}
            yt = zz["y_true"].astype(np.float64)
            mae0 = np.abs(zz["med_zeroshot"] - yt).mean(1)
            logs[b], fire[b] = {}, {}
            for m in REPAIRS:
                on = G.gate_on(zz["gate_s"], zz["gate_rho"], *th[m])
                fire[b][m] = float(on.mean()) if len(on) else float("nan")
                logs[b][m] = np.log(np.abs(zz[f"med_{m}"] - yt).mean(1) / mae0)
                logs[b][f"{m}_gated"] = np.log(np.abs(_gated_med(zz, m, on) - yt).mean(1) / mae0)
            nwin[b] = int(k.sum())
        bs = [b for b in cfg.daisy if b in cells and nwin[b] > 0]
        boots = {b: [rng.integers(0, nwin[b], nwin[b]) for _ in range(cfg.n_boot)] for b in bs}
        per = {b: {m: _gm(logs[b][m]) for m in cand} for b in bs}
        out += [f"### L = {L}", "", "| dataset | windows | R2 gate on | " + " | ".join(LABEL[m] for m in cand) + " |",
                "|---|---|---|" + "---|" * len(cand)]
        for b in bs:
            out.append(f"| {b} | {nwin[b]} | {100 * fire[b]['r2']:.0f}% | " +
                       " | ".join(f"{per[b][m]:.3f}" for m in cand) + " |")
        out += ["", "| method | overall MAE ratio [95% CI] | datasets < 0.99 / > 1.01 |", "|---|---|---|"]
        ov = {}
        for m in cand:
            pt = _gm([np.nanmean(logs[b][m]) for b in bs])
            s = [_gm([np.nanmean(logs[b][m][boots[b][t]]) for b in bs]) for t in range(cfg.n_boot)]
            lo, hi = np.nanpercentile(s, [2.5, 97.5])
            v = [per[b][m] for b in bs]
            ov[m] = {"mae": pt, "ci": [float(lo), float(hi)], "n_better": sum(x < 0.99 for x in v),
                     "n_worse": sum(x > 1.01 for x in v)}
            out.append(f"| {LABEL[m]} | {pt:.3f} [{lo:.3f}, {hi:.3f}] | {ov[m]['n_better']} / {ov[m]['n_worse']} |")
        fire_all = {m: float(np.mean([fire[b][m] for b in bs])) for m in REPAIRS}
        out += ["", f"Windows dropped as silent: {drop}. Mean share of windows gated on: R2 {100 * fire_all['r2']:.0f}%, "
                f"R1 {100 * fire_all['r1']:.0f}%.", ""]
        agg["confirm"][f"L{L}"] = {"overall": ov, "per_dataset": per, "fire": fire, "windows": nwin, "dropped": drop}
        if L == 512:
            g, u = ov["r2_gated"], ov["r2"]
            c1 = g["mae"] < 1 and g["ci"][1] < 1
            c2 = g["n_worse"] <= u["n_worse"] and g["mae"] <= u["mae"] + 0.02
            c3 = u["mae"] < 1 and u["ci"][1] < 1
            agg["confirm"]["criteria"] = {"C1": c1, "C2": c2, "C3": c3}
            yes = lambda x: "met" if x else "NOT met"  # noqa: E731
            out += ["### Pre-declared outcomes (L = 512)", "",
                    f"- C1 (gated R2 overall < 1, CI below 1): {yes(c1)} ({g['mae']:.3f} [{g['ci'][0]:.3f}, "
                    f"{g['ci'][1]:.3f}])",
                    f"- C2 (safety: gated R2 datasets > 1.01 = {g['n_worse']} <= ungated {u['n_worse']}, and overall "
                    f"{g['mae']:.3f} <= ungated {u['mae']:.3f} + 0.02): {yes(c2)}",
                    f"- C3 (ungated R2 overall < 1, CI below 1; replication of Exp13 P1): {yes(c3)} ({u['mae']:.3f} "
                    f"[{u['ci'][0]:.3f}, {u['ci'][1]:.3f}])",
                    "- Stop rule: " + ("C3 failed -> the real-world line of work stops." if not c3 else
                                       ("C3 holds, C1 and C2 hold -> the gate is supported." if c1 and c2 else
                                        "C3 holds but C1 or C2 failed -> the gate is not claimed; ungated results "
                                        "reported as they are.")), ""]
    return out


def summary_describe(run: RunDir, cfg: Exp14Config, th: dict, agg: dict) -> list[str]:
    recs = {r["key"]: r for r in run.read_metrics() if r.get("part") == "describe"}
    out = ["## Descriptive checks on data seen before the protocol (not confirmatory)", ""]
    agg["describe"] = {}

    def gate_of(z, m):
        return G.gate_on(z["gate_s"], z["gate_rho"], *th[m])

    def load(key):
        r = recs[key]
        return (r, dict(np.load(INPUTS["root"] / r["source"], allow_pickle=False)),
                dict(np.load(run.cell_path(key), allow_pickle=False)))

    # Exp13
    ks = [k for k in recs if k.startswith("exp13|")]
    if ks:
        out += ["### Exp13 lab rigs (MAE ratio to zero-shot, geometric means; all windows as in Exp13)", "",
                "| cell | windows | R2 gate on | R1 gate on | " + " | ".join(LABEL[m] for m in
                                                                          ["r2", "r2_gated", "r1", "r1_gated"]) + " |",
                "|---|---|---|---|---|---|---|---|"]
        byL: dict = {}
        for k in sorted(ks):
            r, src, gz = load(k)
            yt = src["y_true"].astype(np.float64)
            mae0 = np.abs(src["med_zeroshot"] - yt).mean(1)
            row, d = [], {}
            for m in REPAIRS:
                on = gate_of(gz, m)
                d[m] = np.log(np.abs(src[f"med_{m}"] - yt).mean(1) / mae0)
                d[f"{m}_gated"] = np.log(np.abs(np.where(on[:, None], src[f"med_{m}"], src["med_zeroshot"]) - yt)
                                         .mean(1) / mae0)
                d[f"fire_{m}"] = on.mean()
            byL.setdefault(r["L"], []).append(d)
            out.append(f"| {k[6:]} | {len(yt)} | {100 * d['fire_r2']:.0f}% | {100 * d['fire_r1']:.0f}% | " +
                       " | ".join(f"{_gm(d[m]):.3f}" for m in ["r2", "r2_gated", "r1", "r1_gated"]) + " |")
        out.append("")
        for L, ds in sorted(byL.items()):
            v = {m: _gm([np.nanmean(d[m]) for d in ds]) for m in ["r2", "r2_gated", "r1", "r1_gated"]}
            agg["describe"][f"exp13_L{L}"] = v
            out.append(f"- L = {L}, overall over benchmarks: " + ", ".join(f"{LABEL[m]} {x:.3f}" for m, x in v.items()))
        out.append("")
    # Exp3 synthetic test
    ks = [k for k in recs if k.startswith("exp3|")]
    if ks:
        c3 = exp3_config(cfg)
        mi = E3.MEDIAN_IDX
        groups: dict = {"all": []}
        for k in ks:
            r, src, gz = load(k)
            setting = E3.setting_of(r["src"]["system"], c3.train_systems)
            groups["all"].append((src, gz))
            groups.setdefault(setting, []).append((src, gz))
        out += ["### Exp3 synthetic test (medians over kept pairs, as in the Exp3 summary)", "",
                "| group | method | g | NRMSE | err ratio | gate on |", "|---|---|---|---|---|---|"]
        agg["describe"]["exp3"] = {}
        for gname in ["all", *E3.SETTINGS]:
            if gname not in groups:
                continue
            for m in ["zeroshot", "r2", "r2_gated", "r1", "r1_gated"]:
                gs, ns, rs, fires = [], [], [], []
                for src, gz in groups[gname]:
                    base = m.replace("_gated", "")
                    on = gate_of(gz, base) if m.endswith("_gated") else np.full(len(gz["gate_s"]), base != "zeroshot")
                    fires.append(on)
                    for sh in c3.shapes:
                        kk = src[f"keep_{sh}"]
                        eh = np.where(on[:, None], src[f"ehat_{base}_{sh}"], src[f"ehat_zeroshot_{sh}"])
                        gs.append(M.effect_gain(eh[kk], src[f"e_{sh}"][kk]))
                        ns.append(M.whatif_nrmse(eh[kk], src[f"e_{sh}"][kk]))
                    qf = np.where(on[:, None], src[f"qfac_{base}"][:, mi], src["qfac_zeroshot"][:, mi])
                    yt = src["y_true"]
                    rs.append(np.abs(qf - yt).mean(-1) / np.maximum(np.abs(src["qfac_zeroshot"][:, mi] - yt).mean(-1),
                                                                     1e-12))
                g, n, er = (float(np.median(np.concatenate(x))) for x in (gs, ns, rs))
                fr = float(np.mean(np.concatenate(fires)))
                agg["describe"]["exp3"][f"{gname}|{m}"] = {"g": g, "nrmse": n, "err_ratio": er, "fire": fr}
                out.append(f"| {gname} | {LABEL[m]} | {g:.3f} | {n:.3f} | {er:.3f} | {100 * fr:.0f}% |")
        out.append("")
    # LCL
    ks = [k for k in recs if k.startswith("lcl|")]
    if ks:
        out += ["### LCL pricing trial (Exp12 cells; expected: the gate mostly stays off)", "",
                "| cell | method | gate on | g (debiased) | beta | rel. High (truth) | rel. Low (truth) | sign agree | "
                "MAE ratio |", "|---|---|---|---|---|---|---|---|---|"]
        agg["describe"]["lcl"] = {}
        for k in sorted(ks):
            r, src, gz = load(k)
            z = dict(src)
            fires = {}
            for m in REPAIRS:
                on = gate_of(gz, m)
                fires[m] = float(on.mean())
                for key in ("med_fac", "med_cf", "q9"):
                    z[f"{key}_{m}_gated"] = np.where(on.reshape((-1,) + (1,) * (src[f"{key}_{m}"].ndim - 1)),
                                                     src[f"{key}_{m}"], src[f"{key}_zeroshot"])
            for drop in [x for x in z if x.startswith("med_fac_") and x[8:] not in
                         ("zeroshot", "r2", "r1", "r2_gated", "r1_gated")]:
                z.pop(drop)
            s = E12.cell_summary(z, r["src"]["noise_sq"], cfg.n_boot, cfg.seed)
            agg["describe"]["lcl"][k] = {"fire": fires, **{m: s["methods"][m] for m in s["methods"]}}
            for m in ["zeroshot", "r2", "r2_gated", "r1", "r1_gated"]:
                q = s["methods"][m]
                fr = fires.get(m.replace("_gated", ""), 0.0) if m.endswith("_gated") else (0.0 if m == "zeroshot"
                                                                                            else 1.0)
                out.append(f"| {k[4:]} | {LABEL[m]} | {100 * fr:.0f}% | {q['g_debiased']:.2f} | {q['beta']:.2f} | "
                           f"{q['pred_rel']['High']:.2f} ({s['truth_rel']['High']:.2f}) | {q['pred_rel']['Low']:.2f} "
                           f"({s['truth_rel']['Low']:.2f}) | {q['sign_agree']:.2f} | {q['mae_ratio']:.3f} |")
        out.append("")
    # fev
    ks = [k for k in recs if k.startswith("fev|")]
    if ks:
        rows, ratios = [], {m: [] for m in ["r2", "r2_gated", "r1", "r1_gated"]}
        for k in sorted(ks):
            r, src, gz = load(k)
            yt = src["y_test"].astype(np.float64)
            if len(gz["gate_s"]) != len(yt):
                continue
            w0 = _fev_wql(src["q_zeroshot_test"], yt)
            vals, fires = {}, {}
            for m in REPAIRS:
                on = gate_of(gz, m)
                fires[m] = float(on.mean())
                vals[m] = _fev_wql(src[f"q_{m}_test"], yt) / w0
                vals[f"{m}_gated"] = _fev_wql(np.where(on[:, None, None], src[f"q_{m}_test"], src["q_zeroshot_test"]),
                                              yt) / w0
            for m in ratios:
                ratios[m].append(vals[m])
            rows.append(f"| {k[8:]} | {len(yt)} | {100 * fires['r2']:.0f}% | " +
                        " | ".join(f"{vals[m]:.3f}" for m in ratios) + " |")
        out += ["### fev-bench (WQL ratio to zero-shot, test windows of Exp6)", "",
                "| task | windows | R2 gate on | " + " | ".join(LABEL[m] for m in ratios) + " |",
                "|---|---|---|" + "---|" * len(ratios), *rows, ""]
        gm = {m: _gm(np.log(v)) for m, v in ratios.items() if v}
        agg["describe"]["fev"] = {"gmean": gm, "n_tasks": len(rows),
                                  "wins_losses": {m: [int(sum(x < 0.99 for x in v)), int(sum(x > 1.01 for x in v))]
                                                  for m, v in ratios.items()}}
        out += [f"Geometric mean over {len(rows)} tasks: " + ", ".join(f"{LABEL[m]} {x:.3f}" for m, x in gm.items()), ""]
    return out


def _fev_wql(q10: np.ndarray, yt: np.ndarray) -> float:
    """Pooled fev WQL on the 9 stored levels (q10[:, :9]), as exp6._scores."""
    lv = np.asarray(LEVELS9)[None, :, None]
    qq, y = q10[:, :9].astype(np.float64), yt[:, None, :]
    valid = np.isfinite(y) & np.isfinite(qq)
    loss = np.where(valid, 2.0 * np.abs((y - qq) * ((y <= qq).astype(np.float64) - lv)), 0.0)
    den = np.nansum(np.abs(yt))
    return float(loss.sum() / 9 / den) if den > 0 else float("nan")


def write_summary(run: RunDir, cfg: Exp14Config) -> None:
    rep = results_root() / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    lines = [f"# Exp14 summary ({cfg.run_name}) - training-free gate", "",
             f"Protocol: docs/gate_protocol.md. Run: `{run.path}`. Repairs from exp3/{cfg.exp3_run}.", ""]
    lines += summary_select(run)
    agg: dict = {}
    if (run.path / "gate_select.json").exists():
        th = load_thresholds(run)
        lines += summary_confirm(run, cfg, th, np.random.default_rng(cfg.seed), agg)
        lines += summary_describe(run, cfg, th, agg)
    (rep / f"exp14_gate_{cfg.run_name}.json").write_text(json.dumps(agg, indent=1, default=float), encoding="utf-8")
    out = rep / f"exp14_summary_{cfg.run_name}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("summary written: %s", out)


# ------------------------------------------------------------------------------------ main
def main() -> None:
    cfg, args = parse_args()
    run = RunDir("exp14", cfg.run_name)
    setup_logging(run.path / "log.txt")
    log.info("config: %s", json.dumps(asdict(cfg)))
    if args.summary_only:
        write_summary(run, cfg)
        return
    run.write_config(cfg, resume=args.resume, ignore_keys=("device", "batch_size", "describe"))
    from covattn.model import Chronos2Wrapper

    w = Chronos2Wrapper(cfg.model_id, device=cfg.device, amp=cfg.amp, batch_size=cfg.batch_size)
    torch.manual_seed(cfg.seed)
    methods, info = build_methods(w, cfg)
    (run.path / "methods.json").write_text(json.dumps({"methods": list(methods), **info}, indent=1))
    done = run.done_keys()
    jobs: list = []
    if "select" in args.part:
        jobs += [("select", j) for j in select_jobs(cfg)]
    if "confirm" in args.part:
        if "select" not in args.part:
            load_thresholds(run)  # refuse to run before the thresholds are frozen
        jobs += [("confirm", (b, L)) for L in cfg.contexts for b in cfg.daisy]
    if "describe" in args.part:
        if "select" not in args.part:
            load_thresholds(run)
        jobs += [("describe", j) for j in describe_jobs(cfg)]
    cache: dict = {}
    vs = None
    daisy = None
    t_start, n_run = time.time(), 0
    for i, (part, job) in enumerate(jobs):
        if part in ("confirm", "describe") and not (run.path / "gate_select.json").exists():
            if not selection_complete(run, cfg):
                log.error("selection incomplete -> thresholds not frozen; %s parts not run", part)
                break
            choose_thresholds(run, cfg, info)
        if part == "select":
            key = job_key(*job)
        elif part == "confirm":
            key = f"daisy|{job[0]}|L{job[1]}"
        else:
            key = describe_key(job[0], job[1])
        if key in done:
            continue
        t0 = time.time()
        try:
            if part == "select" and job[0] == "val":
                vs = vs or E3.val_sets(exp3_config(cfg))
                arrays, rec = run_val(job[1], vs, w, methods)
            elif part == "select":
                arrays, rec = run_semi(job[1], w, methods, cfg, cache)
            elif part == "confirm":
                daisy = daisy or load_records(INPUTS["daisy"])
                arrays, rec = run_daisy(job[0], job[1], daisy, w, methods, cfg)
            else:
                arrays, rec = run_describe(job[0], job[1], job[2], w, cfg, cache)
        except ValueError as e:
            log.info("skip %s: %s", key, e)
            continue
        except Exception:  # noqa: BLE001
            log.exception("%s FAILED - continuing", key)
            if w.device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        run.save_arrays(key, **arrays)
        rec.update({"key": key, "part": part, "seconds": time.time() - t0, "device": str(w.device),
                    "s_median": float(np.median(arrays["gate_s"]))})
        run.append_metrics(rec)
        n_run += 1
        el = time.time() - t_start
        left = len(jobs) - i - 1
        log.info("[%d/%d] %s n=%d %.0fs ETA <= %s", i + 1, len(jobs), key, rec["n"], rec["seconds"],
                 fmt_seconds(el / n_run * left))
    if "select" in args.part:
        if selection_complete(run, cfg):
            choose_thresholds(run, cfg, info)
        else:
            log.warning("selection incomplete: thresholds not chosen yet (rerun with --resume)")
    write_summary(run, cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        log.exception("Exp14 crashed")
        raise
