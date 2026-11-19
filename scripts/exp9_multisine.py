"""Exp9 - why do multisine covariates collapse the effect gain? A controlled interpolation between regimes.

Hypothesis (univariate periodic shortcut): with a multisine covariate the target is itself periodic and can be
extrapolated from its own past, so the model barely uses the covariate, and a change of the future covariate is
not propagated. Test: covariate u = a * multisine + (1 - a) * PRBS (each standardised; systems.gen_input "mix<a>"),
a in {0, .25, .5, .75, 1}, on the Exp3 test systems and L in {512, 2048}. Per cell we measure
  - the effect gain g of zero-shot, R1 and R2 (same artefacts as Exp3 run `exp3_run`), and
  - covariate reliance: the per-series ratio of zero-shot MAE without vs with covariates (> 1 = covariates help).
The hypothesis predicts that g falls and covariate reliance falls together as a grows.

  python scripts/exp9_multisine.py --quick
  python scripts/exp9_multisine.py --resume --exp3-run v2        # Colab
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
from covattn.io_utils import RunDir, fmt_seconds, results_root, setup_logging
from covattn.latex import rows_to_tex
from covattn.model import CovBatch
from covattn.repair import gain_spec, r1_gain

import exp3_repair as E3  # noqa: E402  (scripts/ is on sys.path when this file is run)

log = logging.getLogger("exp9")
LABEL = {"zeroshot": "Zero-shot", "r1": "R1", "r2": "R2"}


@dataclass
class Exp9Config:
    run_name: str = "main"
    quick: bool = False
    exp3_run: str = "v2"
    alphas: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    contexts: list[int] = field(default_factory=lambda: [512, 2048])
    systems: list[str] | None = None  # None = the Exp3 test systems
    device: str = "auto"
    batch_size: int = 128
    n_boot: int = 1000


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
    cfg = Exp9Config(quick=a.quick, device=a.device, batch_size=a.batch_size)
    if a.quick:
        cfg.run_name, cfg.exp3_run, cfg.contexts, cfg.alphas, cfg.n_boot = "quick", "quick", [256], [0.0, 1.0], 50
    if a.run_name is not None:
        cfg.run_name = a.run_name
    if a.exp3_run is not None:
        cfg.exp3_run = a.exp3_run
    return cfg, a


def exp3_config(cfg: Exp9Config) -> "E3.Exp3Config":
    d3 = results_root() / "exp3" / cfg.exp3_run
    c = json.loads((d3 / "config.json").read_text())["config"]
    c3 = E3.Exp3Config(**{k: v for k, v in c.items() if k in E3.Exp3Config.__dataclass_fields__})
    return replace(c3, contexts=list(cfg.contexts), device=cfg.device, batch_size=cfg.batch_size)


def inp_name(a: float) -> str:
    return f"mix{a:g}"


# ------------------------------------------------------------------------------------ summary
def _cell_stats(c: dict, m: str, shapes: list[str]) -> np.ndarray:
    return np.concatenate([M.effect_gain(c[f"ehat_{m}_{sh}"][c[f"keep_{sh}"]], c[f"e_{sh}"][c[f"keep_{sh}"]])
                           for sh in shapes])


def _reliance(c: dict) -> np.ndarray:
    """Per series: zero-shot MAE without covariates / with covariates (median quantile)."""
    yt = c["y_true"]
    mae_cov = np.abs(c["qfac_zeroshot"][:, E3.MEDIAN_IDX] - yt).mean(-1)
    mae_no = np.abs(c["qfac_nocov"][:, E3.MEDIAN_IDX] - yt).mean(-1)
    return mae_no / np.maximum(mae_cov, 1e-12)


def write_summary(run: RunDir, cfg: Exp9Config) -> None:
    from scipy.stats import spearmanr

    recs = run.read_metrics()
    if not recs:
        return
    c3 = exp3_config(cfg)
    shapes = c3.shapes
    cells = []
    for r in recs:
        z = dict(np.load(run.cell_path(r["key"]), allow_pickle=False))
        z.update({"alpha": r["alpha"], "L": r["L"], "system": r["system"], "family": r["family"]})
        cells.append(z)
    methods = [m for m in LABEL if f"ehat_{m}_{shapes[0]}" in cells[0]]
    alphas = sorted({c["alpha"] for c in cells})
    Ls = sorted({c["L"] for c in cells})
    rng = np.random.default_rng(0)

    def pooled(cs: list[dict]) -> dict:
        out = {m: float(np.median(np.concatenate([_cell_stats(c, m, shapes) for c in cs]))) for m in methods}
        out["reliance"] = float(np.median(np.concatenate([_reliance(c) for c in cs])))
        return out

    def boot_ci(cs: list[dict], m: str) -> tuple[float, float]:
        per = [_cell_stats(c, m, shapes) for c in cs]
        bs = [np.median(np.concatenate([per[i] for i in rng.integers(0, len(per), len(per))])) for _ in range(cfg.n_boot)]
        return tuple(np.percentile(bs, [2.5, 97.5]))

    lines = [f"# Exp9 summary ({cfg.run_name}) - multisine regime", "",
             f"Source: `{run.metrics_file}` ({len(cells)} cells). Covariate u = a * multisine + (1 - a) * PRBS. "
             "Median effect gain g (step and pulse pooled) and covariate reliance = median per-series zero-shot MAE "
             "ratio without / with covariates (> 1: covariates help). 95% CI: cluster bootstrap over cells.", "",
             "## By mixing weight a (all L)", "",
             "| a | " + " | ".join(f"g {LABEL[m]} [95% CI]" for m in methods) + " | reliance | cells |",
             "|---|" + "---|" * (len(methods) + 2)]
    table = {}
    for a in alphas:
        cs = [c for c in cells if c["alpha"] == a]
        s = pooled(cs)
        cis = {m: boot_ci(cs, m) for m in methods}
        table[a] = (s, cis)
        lines.append(f"| {a:g} | " + " | ".join(f"{s[m]:.3f} [{cis[m][0]:.3f}, {cis[m][1]:.3f}]" for m in methods)
                     + f" | {s['reliance']:.3f} | {len(cs)} |")
    lines += ["", "## By mixing weight a and context length L", "",
              "| a | L | " + " | ".join(f"g {LABEL[m]}" for m in methods) + " | reliance |",
              "|---|---|" + "---|" * (len(methods) + 1)]
    for a in alphas:
        for L in Ls:
            cs = [c for c in cells if c["alpha"] == a and c["L"] == L]
            if cs:
                s = pooled(cs)
                lines.append(f"| {a:g} | {L} | " + " | ".join(f"{s[m]:.3f}" for m in methods) + f" | {s['reliance']:.3f} |")
    # cell-level association between covariate reliance and zero-shot gain
    cg = np.array([np.median(_cell_stats(c, "zeroshot", shapes)) for c in cells])
    cr = np.array([np.median(_reliance(c)) for c in cells])
    rho, p = spearmanr(cr, cg)
    lines += ["", "## Covariate reliance vs zero-shot gain (cell level)", "",
              f"- all cells: Spearman rho = {rho:.3f} (p = {p:.2g}, n = {len(cells)})"]
    for a in alphas:
        idx = [i for i, c in enumerate(cells) if c["alpha"] == a]
        if len(idx) >= 5:
            r_, p_ = spearmanr(cr[idx], cg[idx])
            lines.append(f"- a = {a:g}: rho = {r_:.3f} (p = {p_:.2g}, n = {len(idx)})")
    for L in Ls:
        idx = [i for i, c in enumerate(cells) if c["L"] == L]
        if len(idx) >= 5:
            r_, p_ = spearmanr(cr[idx], cg[idx])
            lines.append(f"- L = {L}: rho = {r_:.3f} (p = {p_:.2g}, n = {len(idx)})")
    rep = results_root() / "reports"
    (rep / "tables").mkdir(parents=True, exist_ok=True)
    (rep / f"exp9_summary_{cfg.run_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (rep / f"exp9_cells_{cfg.run_name}.json").write_text(json.dumps(
        {"cells": [{"alpha": c["alpha"], "L": c["L"], "system": c["system"], "g_zeroshot": float(g), "reliance": float(r)}
                   for c, g, r in zip(cells, cg, cr)],
         "by_alpha": {f"{a:g}": {"median": table[a][0], "ci": {m: list(v) for m, v in table[a][1].items()}}
                      for a in alphas},
         "by_alpha_L": {f"{a:g}|{L}": pooled([c for c in cells if c["alpha"] == a and c["L"] == L])
                        for a in alphas for L in Ls if any(c["alpha"] == a and c["L"] == L for c in cells)},
         "spearman_all": [float(rho), float(p)]}, indent=1), encoding="utf-8")
    hdr = ["$a$ (multisine share) & " + " & ".join(f"$g$ {LABEL[m]}" for m in methods) + " & reliance \\\\"]
    rows = [[f"{a:g}"] + [f"{table[a][0][m]:.2f}" for m in methods] + [f"{table[a][0]['reliance']:.2f}"] for a in alphas]
    (rep / "tables" / f"table9_multisine_{cfg.run_name}.tex").write_text(
        rows_to_tex("c " + "c" * (len(methods) + 1), hdr, rows), encoding="utf-8")
    log.info("summary written")


def main() -> None:
    cfg, args = parse_args()
    run = RunDir("exp9", cfg.run_name)
    setup_logging(run.path / "log.txt")
    log.info("config: %s", json.dumps(asdict(cfg)))
    if args.summary_only:
        write_summary(run, cfg)
        return
    run.write_config(cfg, resume=args.resume, ignore_keys=("device", "batch_size"))
    from covattn.model import Chronos2Wrapper

    c3 = exp3_config(cfg)
    w = Chronos2Wrapper(c3.model_id, device=cfg.device, amp=c3.amp, batch_size=cfg.batch_size)
    torch.manual_seed(c3.seed)
    d3 = results_root() / "exp3" / cfg.exp3_run
    r1 = json.loads((d3 / "r1_select.json").read_text())
    hk = w.hooked
    G1 = r1_gain(hk.n_layers, hk.n_heads, [tuple(x) for x in r1["ranking"][: r1["best"]["k"]]], r1["best"]["gamma"])
    G2 = torch.tensor(json.loads((d3 / "r2_gates.json").read_text())["gates"])

    def nocov(bt: CovBatch, s: int) -> np.ndarray:
        return w.predict(CovBatch(bt.target, bt.cov_past[:, :0], bt.cov_future[:, :0]), bt.H)

    methods = {"zeroshot": lambda bt, s: w.predict(bt, bt.H), "r1": lambda bt, s: w.predict_hooked(bt, gain_spec(G1)),
               "r2": lambda bt, s: w.predict_hooked(bt, gain_spec(G2)), "nocov": nocov}
    lk = E3.systems_by_key()
    systems = cfg.systems or (c3.test_systems if not cfg.quick else c3.test_systems[:2])
    done = run.done_keys() if args.resume else set()
    jobs = [(lk[s], L, a) for L in cfg.contexts for a in cfg.alphas for s in systems]
    jobs = [j for j in jobs if E3.cell_key(j[0].key, j[1], inp_name(j[2])) not in done]
    log.info("%d cells to run (A100 rough: ~3-5 s per cell)", len(jobs))
    t_start = time.time()
    for i, (sys, L, a) in enumerate(jobs):
        try:
            rec = E3.eval_cell(methods, w, c3, sys, L, inp_name(a), run)
        except Exception:  # noqa: BLE001 - report and continue
            log.exception("%s L%d a=%g FAILED", sys.key, L, a)
            continue
        rec["alpha"] = a
        run.append_metrics(rec)
        el = time.time() - t_start
        log.info("[%d/%d] %s ETA %s | g_%s %s", i + 1, len(jobs), rec["key"], fmt_seconds(el / (i + 1) * (len(jobs) - i - 1)),
                 c3.shapes[0], {m: round(v.get(f"g_{c3.shapes[0]}", float("nan")), 3) for m, v in rec["methods"].items()
                                if m != "nocov"})
    write_summary(run, cfg)
    n_all = len(cfg.contexts) * len(cfg.alphas) * len(systems)
    if len(run.done_keys()) >= n_all:
        run.mark_done({"n_cells": n_all})


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        log.exception("Exp9 crashed")
        raise
