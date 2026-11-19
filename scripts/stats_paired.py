"""Paired significance of the main synthetic comparisons (Exp3 test cells, no model needed).

For each method m and setting, the statistic is the difference in median |1 - g| between zero-shot and m
(positive = m is closer to exact), and the difference in median err. ratio. The 95% CI comes from a cluster
bootstrap over cells (the same resampled cells for both methods, i.e. paired), and the p-value is the share of
bootstrap draws on the wrong side of 0 (two-sided). Also: Wilcoxon signed-rank test on per-pair |1 - g|.

  RESULTS_DIR=... python scripts/stats_paired.py --exp3-run v2
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import wilcoxon

from covattn import metrics as M
from covattn.io_utils import RunDir, results_root

MEDIAN_IDX = 10
SHAPES = ("step", "pulse")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp3-run", default="v2")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    run = RunDir("exp3", a.exp3_run)
    recs = run.read_metrics()
    cells = []
    for r in recs:
        z = dict(np.load(run.cell_path(r["key"]), allow_pickle=False))
        z.update({"setting": r["setting"], "input": r["input"]})
        cells.append(z)
    methods = ["r1", "r2", "dither"]
    rng = np.random.default_rng(0)

    def dev(c: dict, m: str) -> np.ndarray:  # per pair |1 - g|
        return np.concatenate([np.abs(1 - M.effect_gain(c[f"ehat_{m}_{sh}"][c[f"keep_{sh}"]], c[f"e_{sh}"][c[f"keep_{sh}"]]))
                               for sh in SHAPES])

    def err(c: dict, m: str) -> np.ndarray:
        yt = c["y_true"]
        return (np.abs(c[f"qfac_{m}"][:, MEDIAN_IDX] - yt).mean(-1)
                / np.maximum(np.abs(c["qfac_zeroshot"][:, MEDIAN_IDX] - yt).mean(-1), 1e-12))

    groups = [("all", cells), ("multisine", [c for c in cells if c["input"] == "multisine"]),
              ("non-multisine", [c for c in cells if c["input"] != "multisine"])]
    groups += [(s, [c for c in cells if c["setting"] == s]) for s in ("ID", "OOD-param", "OOD-family", "OOD-nonlinear")]
    out, lines = {}, [f"# Paired significance vs zero-shot (Exp3 `{a.exp3_run}` test cells)", "",
                      "Delta|1-g| = median |1-g| (zero-shot) - median |1-g| (method); > 0 means closer to exact. "
                      "Delta err = median err. ratio - 1 (< 0 means more accurate than zero-shot). 95% CI / p: paired "
                      f"cluster bootstrap over cells ({a.n_boot} draws); Wilcoxon: signed-rank on per-pair |1-g|.", "",
                      "| group | method | cells | Delta|1-g| [95% CI] | p (boot) | Wilcoxon p | Delta err [95% CI] |",
                      "|---|---|---|---|---|---|---|"]
    for gname, cs in groups:
        if not cs:
            continue
        d0 = [dev(c, "zeroshot") for c in cs]
        for m in methods:
            dm = [dev(c, m) for c in cs]
            em = [err(c, m) for c in cs]
            stat = float(np.median(np.concatenate(d0)) - np.median(np.concatenate(dm)))
            estat = float(np.median(np.concatenate(em)) - 1.0)
            bs, be = [], []
            for _ in range(a.n_boot):
                idx = rng.integers(0, len(cs), len(cs))
                bs.append(np.median(np.concatenate([d0[i] for i in idx])) - np.median(np.concatenate([dm[i] for i in idx])))
                be.append(np.median(np.concatenate([em[i] for i in idx])) - 1.0)
            bs, be = np.asarray(bs), np.asarray(be)
            pb = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
            w = wilcoxon(np.concatenate(d0), np.concatenate(dm))
            ci, cie = np.percentile(bs, [2.5, 97.5]), np.percentile(be, [2.5, 97.5])
            out[f"{gname}|{m}"] = {"delta_dev": stat, "ci": list(ci), "p_boot": pb, "p_wilcoxon": float(w.pvalue),
                                   "delta_err": estat, "ci_err": list(cie), "cells": len(cs)}
            lines.append(f"| {gname} | {m} | {len(cs)} | {stat:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {pb:.3g} | "
                         f"{w.pvalue:.2g} | {estat:+.3f} [{cie[0]:+.3f}, {cie[1]:+.3f}] |")
    rep = results_root() / "reports"
    (rep / f"stats_paired_{a.exp3_run}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (rep / f"stats_paired_{a.exp3_run}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
