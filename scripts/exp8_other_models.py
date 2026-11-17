"""Exp8 - is the attenuation specific to Chronos-2? Effect gain g of other covariate-capable TSFMs on the same
synthetic test samples as Exp3 (test split, sample seed 0, delta = 1, step and pulse).

Models: Chronos-2 (zero-shot, reference), TimesFM 3.0 (native covariates), TimesFM 2.5 with XReg
(in-context linear regression on the covariate values + TimesFM on the residuals). See covattn/other_models.py.
ARX (context fit) is read from the Exp3 cells of the same samples. Phenomenon only: no hooks or repairs.

  python scripts/exp8_other_models.py --quick
  python scripts/exp8_other_models.py --resume                 # Colab
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
from covattn.latex import TO1, mark_best, rows_to_tex
from covattn.model import CovBatch
from covattn.systems import (SampleSpec, default_systems, delta_u, make_counterfactual, make_rng, make_sample,
                             quick_systems)

log = logging.getLogger("exp8")
LABEL = {"chronos2": "Chronos-2", "timesfm3": "TimesFM 3.0", "timesfm25_xreg": "TimesFM 2.5 + XReg",
         "arx": "ARX (context fit)"}
SUBSET_LABEL = {"all": "All test pairs", "family=F1": "F1 static", "family=F2": "F2 lag",
                "family=F3": "F3 delay", "family=F4": "F4 resonant", "family=F5": "F5 tanh",
                "input=prbs": "PRBS input", "input=multisine": "multisine input", "input=ar1": "AR(1) input",
                "input=steps": "random-steps input", "L=512": "$L=512$", "L=2048": "$L=2048$"}


@dataclass
class Exp8Config:
    run_name: str = "main"
    quick: bool = False
    exp3_run: str = "v2"
    models: list[str] = field(default_factory=lambda: ["chronos2", "timesfm3", "timesfm25_xreg"])
    systems: list[str] = field(default_factory=lambda: [s.key for s in default_systems()])
    contexts: list[int] = field(default_factory=lambda: [512, 2048])
    input_types: list[str] = field(default_factory=lambda: ["prbs", "multisine", "ar1", "steps"])
    shapes: list[str] = field(default_factory=lambda: ["step", "pulse"])
    horizon: int = 64
    n_per_cell: int = 32
    delta: float = 1.0
    eff_thresh: float = 0.05
    seed: int = 0
    device: str = "auto"
    batch_size: int = 128


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--exp3-run", default=None, help="Exp3 run the ARX column is read from (default v2)")
    p.add_argument("--device", default="auto")
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    cfg = Exp8Config(quick=a.quick, device=a.device)
    if a.quick:
        cfg.run_name, cfg.exp3_run = "quick", "quick"
        cfg.systems, cfg.contexts, cfg.input_types, cfg.n_per_cell = ["F2_a0.8", "F4_P24_z0.2"], [256], ["prbs"], 4
    if a.run_name is not None:
        cfg.run_name = a.run_name
    if a.models is not None:
        cfg.models = a.models
    if a.exp3_run is not None:
        cfg.exp3_run = a.exp3_run
    return cfg, a


def load_model(name: str, cfg: Exp8Config, device: str):
    if name == "chronos2":
        from covattn.model import Chronos2Wrapper

        w = Chronos2Wrapper("amazon/chronos-2", device=device, batch_size=cfg.batch_size)
        w.median_idx_ = w.median_idx
        return w, w.median_idx
    if name == "timesfm3":
        from covattn.other_models import TimesFM3Adapter

        m = TimesFM3Adapter(device=device, batch_size=32)
        return m, m.median_idx
    if name == "timesfm25_xreg":
        from covattn.other_models import TimesFM25XRegAdapter

        m = TimesFM25XRegAdapter(max_context=max(cfg.contexts), max_horizon=cfg.horizon)
        return m, m.median_idx
    raise ValueError(name)


def cell_key(m: str, s: str, L: int, inp: str) -> str:
    return f"{m}|{s}|L{L}|{inp}"


def eval_cell(model, mi: int, name: str, cfg: Exp8Config, sys, L: int, inp: str, run: RunDir) -> dict:
    spec = SampleSpec(context_length=L, horizon=cfg.horizon, input_type=inp)
    rng = make_rng(cfg.seed, "test", "exp3", sys.key, L, inp)  # the Exp3 test samples
    samples = [make_sample(sys, spec, rng) for _ in range(cfg.n_per_cell)]
    y = np.stack([s.y[:L] for s in samples])
    u = np.stack([s.u[:L] for s in samples])
    yt = np.stack([s.y[L:] for s in samples])
    q = model.predict(CovBatch(y, u, np.stack([s.u[L:] for s in samples])), cfg.horizon)
    arrays = {"y_true": yt.astype(np.float32), "med": q[:, mi].astype(np.float32)}
    rec = {"key": cell_key(name, sys.key, L, inp), "model": name, "system": sys.key, "family": sys.family, "L": L,
           "input": inp, "mase": float(np.median(M.mase(q[:, mi], yt, y))), "g": {}}
    for sh in cfg.shapes:
        cfs = [make_counterfactual(s, sh, cfg.delta) for s in samples]
        e = np.stack([c.effect for c in cfs]).astype(np.float32)
        qc = model.predict(CovBatch(y, u, np.stack([c.u_future_cf for c in cfs])), cfg.horizon)
        eh = (qc[:, mi] - q[:, mi]).astype(np.float32)
        keep = M.effect_mask(e, y.std(-1), cfg.eff_thresh)
        arrays.update({f"e_{sh}": e, f"ehat_{sh}": eh, f"keep_{sh}": keep})
        g = M.effect_gain(eh[keep], e[keep])
        rec["g"][sh] = float(np.median(g)) if len(g) else float("nan")
    run.save_arrays(rec["key"], **arrays)
    return rec


def write_summary(run: RunDir, cfg: Exp8Config) -> None:
    recs = run.read_metrics()
    if not recs:
        return
    cells = []
    for r in recs:
        z = dict(np.load(run.cell_path(r["key"]), allow_pickle=False))
        z.update({k: r[k] for k in ("model", "system", "family", "L", "input")})
        cells.append(z)
    d3 = results_root() / "exp3" / cfg.exp3_run
    arx = []  # ARX on the same samples, from Exp3 cells
    for c in cells:
        if c["model"] != cfg.models[0]:
            continue
        safe = "".join(ch if ch.isalnum() or ch in "-_.=" else "_" for ch in f"{c['system']}|L{c['L']}|{c['input']}")
        f = d3 / "cells" / f"{safe}.npz"
        if f.exists():
            z = dict(np.load(f, allow_pickle=False))
            arx.append({"model": "arx", **{k: c[k] for k in ("system", "family", "L", "input")},
                        **{f"e_{sh}": z[f"e_{sh}"] for sh in cfg.shapes}, **{f"keep_{sh}": z[f"keep_{sh}"] for sh in cfg.shapes},
                        **{f"ehat_{sh}": z[f"ehat_arx_{sh}"] for sh in cfg.shapes}})
    cells += arx
    models = [m for m in cfg.models + ["arx"] if any(c["model"] == m for c in cells)]

    def stats(cs):
        g, r = [], []
        for c in cs:
            for sh in cfg.shapes:
                k = c[f"keep_{sh}"]
                g.append(M.effect_gain(c[f"ehat_{sh}"][k], c[f"e_{sh}"][k]))
                r.append(M.shape_corr(c[f"ehat_{sh}"][k], c[f"e_{sh}"][k]))
        if not g:
            return (np.nan,) * 4 + (0,)
        g, r = np.concatenate(g), np.concatenate(r)
        return (float(np.median(g)), float(np.percentile(g, 25)), float(np.percentile(g, 75)), float(np.nanmedian(r)),
                len(g)) if len(g) else (np.nan,) * 4 + (0,)

    groups = [("all", lambda c: True)] + [(f"family={f}", lambda c, f=f: c["family"] == f) for f in
                                          ("F1", "F2", "F3", "F4", "F5")]
    groups += [(f"input={i}", lambda c, i=i: c["input"] == i) for i in cfg.input_types]
    groups += [(f"L={L}", lambda c, L=L: c["L"] == L) for L in cfg.contexts]
    lines = [f"# Exp8 summary ({cfg.run_name}) - other TSFMs", "", f"Source: `{run.metrics_file}`. Same test samples "
             f"as Exp3 `{cfg.exp3_run}` (ARX taken from there). Per model: median g [IQR] / median shape r / n pairs.", "",
             "| group | " + " | ".join(LABEL.get(m, m) for m in models) + " |", "|---|" + "---|" * len(models)]
    # LaTeX: median g per group; models closest to 1 in bold among the TSFMs (ARX is a reference column).
    trows = []
    for gname, sel in groups:
        if not any(sel(c) for c in cells):
            continue
        row, gs = [], []
        for m in models:
            s = stats([c for c in cells if c["model"] == m and sel(c)])
            row.append(f"{s[0]:.3f} [{s[1]:.2f}, {s[2]:.2f}] / {s[3]:.2f} / {s[4]}")
            gs.append(s[0])
        lines.append(f"| {gname} | " + " | ".join(row) + " |")
        trows.append((gname, mark_best(gs, ".2f", "one", [m != "arx" for m in models])))
    head = {"chronos2": "Chronos-2", "timesfm3": "TimesFM 3.0", "timesfm25_xreg": "TimesFM 2.5+XReg$^\\dagger$",
            "arx": "ARX (ref.)"}
    hdr = [f"Subset (median $g${TO1}) & " + " & ".join(head.get(m, LABEL.get(m, m)) for m in models) + " \\\\"]
    rows = []
    for gname, vals in trows:
        kind = gname.split("=")[0]
        if kind != "all" and not any(r == "\\midrule" and k == kind for r, k in rows):
            rows.append(("\\midrule", kind))
        rows.append(([SUBSET_LABEL.get(gname, gname)] + vals, kind))
    tex = ("\\setlength{\\tabcolsep}{3pt}\n" + rows_to_tex("l " + "c" * len(models), hdr, [r for r, _ in rows])).splitlines()
    # Memoryless check: a response that is a scaled copy of the covariate change cannot carry dynamics.
    # pulse: corr over the horizon between e_hat and du (du is non-constant); step: du is constant, so a
    # memoryless response is constant too and its shape r vs the effect is undefined (float noise only).
    lines += ["", "## Memoryless check", "",
              "Per model: median corr(e_hat, du) over the horizon for the pulse; share of step pairs whose e_hat is "
              "constant over the horizon (std < 1e-3 x max|e_hat|; shape r is then undefined and only reflects "
              "float noise); median shape r on pulse pairs only.", "",
              "| model | corr(e_hat, du), pulse | step e_hat constant | shape r, pulse only |", "|---|---|---|---|"]
    du_p = delta_u("pulse", cfg.horizon, 1.0)
    for m in models:
        cs = [c for c in cells if c["model"] == m]
        cp, rp, const = [], [], []
        for c in cs:
            k = c["keep_pulse"]
            eh = c["ehat_pulse"][k]
            cp.append(M.shape_corr(eh, np.broadcast_to(du_p, eh.shape)))
            rp.append(M.shape_corr(eh, c["e_pulse"][k]))
            if "step" in cfg.shapes:
                es = c["ehat_step"][c["keep_step"]]
                const.append(es.std(-1) < 1e-3 * np.maximum(np.abs(es).max(-1), 1e-12))
        cp, rp = np.concatenate(cp), np.concatenate(rp)
        cs_frac = f"{np.concatenate(const).mean():.3f}" if const else "-"
        lines.append(f"| {LABEL.get(m, m)} | {np.nanmedian(cp):.3f} | {cs_frac} | {np.nanmedian(rp):.2f} |")
    rep = results_root() / "reports"
    (rep / "tables").mkdir(parents=True, exist_ok=True)
    (rep / "tables" / f"table8_other_models_{cfg.run_name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (rep / f"exp8_summary_{cfg.run_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("summary written")


def main() -> None:
    cfg, args = parse_args()
    run = RunDir("exp8", cfg.run_name)
    setup_logging(run.path / "log.txt")
    log.info("config: %s", json.dumps(asdict(cfg)))
    if args.summary_only:
        write_summary(run, cfg)
        return
    run.write_config(cfg, resume=args.resume, ignore_keys=("device", "batch_size", "models"))
    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    lk = {s.key: s for s in default_systems() + quick_systems()}
    done = run.done_keys() if args.resume else set()
    for name in cfg.models:
        jobs = [(lk[s], L, inp) for L in cfg.contexts for s in cfg.systems for inp in cfg.input_types
                if cell_key(name, s, L, inp) not in done]
        if not jobs:
            continue
        try:
            model, mi = load_model(name, cfg, device)
        except Exception:  # noqa: BLE001 - a model that does not load is reported and skipped
            log.exception("model %s could not be loaded - skipped", name)
            continue
        t_start = time.time()
        for i, (sys, L, inp) in enumerate(jobs):
            try:
                rec = eval_cell(model, mi, name, cfg, sys, L, inp, run)
            except Exception:  # noqa: BLE001
                log.exception("%s %s L%d %s FAILED", name, sys.key, L, inp)
                continue
            run.append_metrics(rec)
            el = time.time() - t_start
            log.info("[%s %d/%d] %s ETA %s | g %s", name, i + 1, len(jobs), rec["key"],
                     fmt_seconds(el / (i + 1) * (len(jobs) - i - 1)), {k: round(v, 3) for k, v in rec["g"].items()})
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    write_summary(run, cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        log.exception("Exp8 crashed")
        raise
