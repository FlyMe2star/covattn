"""Fit one output multiplier on Exp2 validation and compare it with R1 on Exp3 test."""

import json

import numpy as np
import pandas as pd

from covattn import metrics
from covattn.io_utils import RunDir


def main():
    val = RunDir("exp2", "main")
    test = RunDir("exp3", "v2")
    out = RunDir("exp15", "explore-output-scale")
    cross, power = [], []
    for rec in val.read_metrics():
        with np.load(val.cell_path(rec["key"])) as data:
            keep = data["keep"]
            e = data["e"][keep].astype(float)
            pred = data["e_hat"][keep].astype(float)
            cross.append(metrics.effect_gain(pred, e))
            power.append((pred**2).sum(1) / (e**2).sum(1))
    cross, power = np.concatenate(cross), np.concatenate(power)
    assert len(cross) == 3072
    # Squared NRMSE of a scaled effect is a quadratic for each pair.
    grid = np.arange(401) / 100
    objective = [np.median(np.sqrt(np.maximum(0, 1 - 2*a*cross + a*a*power))) for a in grid]
    alpha = float(grid[np.argmin(objective)])
    print(f"Frozen validation multiplier: {alpha:.2f}", flush=True)
    pairs = []
    for rec in test.read_metrics():
        with np.load(test.cell_path(rec["key"])) as data:
            for shape in ("step", "pulse"):
                keep = data[f"keep_{shape}"]
                e = data[f"e_{shape}"][keep].astype(float)
                zero = data[f"ehat_zeroshot_{shape}"][keep].astype(float)
                rows = pd.DataFrame(index=range(len(e)))
                rows["system"], rows["input"] = rec["system"], rec["input"]
                for name, pred in (("zero", zero), ("scaled", alpha*zero),
                                   ("r1", data[f"ehat_r1_{shape}"][keep])):
                    rows[f"{name}_nrmse"] = metrics.whatif_nrmse(pred, e)
                    rows[f"{name}_gain_error"] = np.abs(1 - metrics.effect_gain(pred, e))
                pairs.append(rows)
    pairs = pd.concat(pairs, ignore_index=True)
    assert len(pairs) == 9724 and pairs.system.nunique() == 19
    scores = pairs.drop(columns=["input", "system"])
    table = pd.concat({"all": scores.median().to_frame().T,
                       "covered": scores[pairs.input.isin(["prbs", "multisine"])].median().to_frame().T,
                       "input": scores.groupby(pairs.input).median(),
                       "system": scores.groupby(pairs.system).median()}, names=["group", "subset"])
    table.to_csv(out.path / "scores.csv")
    groups = [g[["r1_nrmse", "scaled_nrmse"]].to_numpy() for _, g in pairs.groupby("system")]
    draws = np.random.default_rng(0).integers(len(groups), size=(2000, len(groups)))
    diff = []
    for draw in draws:
        r1, scaled = np.median(np.concatenate([groups[i] for i in draw]), axis=0)
        diff.append(r1 - scaled)
    result = {"alpha": alpha, "validation_pairs": len(cross), "test_pairs": len(pairs),
              "r1_minus_scaled": float(scores.r1_nrmse.median() - scores.scaled_nrmse.median()),
              "ci95": np.quantile(diff, [0.025, 0.975]).tolist()}
    (out.path / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(table.loc[["all", "covered", "input"]].to_string(), result)


if __name__ == "__main__":
    main()
