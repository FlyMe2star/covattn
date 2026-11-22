"""Layer-matched head sets and position-specific R1 on the Exp5 test windows."""

import argparse
from collections import Counter
from dataclasses import asdict
from functools import partial
import json
import logging
import time

import numpy as np
import pandas as pd

from covattn import metrics
from covattn.hooks import HookSpec
from covattn.io_utils import RunDir, fmt_seconds, results_root, setup_logging
from covattn.model import Chronos2Wrapper
from covattn.repair import r1_gain
from exp5_ablations import Exp5Config, eval_cell, systems_by_key


def matched_sets(heads, draws, n_heads):
    counts = sorted(Counter(layer for layer, _ in heads).items())
    rng = np.random.default_rng(0)
    return {
        f"matched_{i:02}": [(layer, int(h)) for layer, count in counts
                            for h in rng.choice(n_heads, count, replace=False)]
        for i in range(1, draws + 1)
    }


def summarize(run, cfg):
    records = run.read_metrics()
    expected = {(s, i) for s in cfg.test_systems for i in cfg.input_types}
    assert {(r["system"], r["input"]) for r in records} == expected and len(records) == len(expected)
    cases, factual = [], []
    for rec in records:
        with np.load(run.cell_path(rec["key"])) as data:
            base = np.abs(data["med_zeroshot"] - data["y_true"]).mean(axis=1)
            for method in rec["methods"]:
                mae = np.abs(data[f"med_{method}"] - data["y_true"]).mean(axis=1)
                factual.append(pd.DataFrame({"system": rec["system"], "method": method,
                                             "mae_ratio": mae / np.maximum(base, 1e-12)}))
                for shape in cfg.shapes:
                    keep = data[f"keep_{shape}"]
                    true = data[f"e_{shape}"][keep]
                    pred = data[f"ehat_{method}_{shape}"][keep]
                    gain = metrics.effect_gain(pred, true)
                    cases.append(pd.DataFrame({"system": rec["system"], "method": method,
                                               "gain": gain, "gain_error": np.abs(1 - gain),
                                               "nrmse": metrics.whatif_nrmse(pred, true)}))
    cases = pd.concat(cases, ignore_index=True)
    factual = pd.concat(factual, ignore_index=True)
    for name, keys in (("all", ["method"]), ("system", ["system", "method"])):
        effect = cases.groupby(keys)[["gain", "gain_error", "nrmse"]].median()
        scores = effect.join(factual.groupby(keys)["mae_ratio"].median())
        scores.to_csv(run.path / f"{name}.csv")
        if name == "all":
            random = scores.loc[scores.index.str.startswith("matched_"), "gain_error"]
            rank = int(1 + (random <= scores.loc["r1", "gain_error"]).sum())
            outcome = {"rank": rank, "sets_including_r1": 1 + len(random),
                       "one_sided_fraction": rank / (1 + len(random))}
            (run.path / "rank.json").write_text(json.dumps(outcome, indent=2))
            print(scores.loc[["zeroshot", "r1", "context", "future"]].to_string(), outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    cfg = Exp5Config(run_name="explore-r1-controls-quick" if args.quick else "explore-r1-controls",
                     quick=args.quick)
    if args.quick:
        cfg.test_systems, cfg.input_types, cfg.n_per_cell = ["F2_a0.8"], ["prbs"], 4
    run = RunDir("exp15", cfg.run_name)
    setup_logging(run.path / "log.txt")
    if args.summary_only:
        summarize(run, cfg)
        return
    selected = json.loads((results_root() / "exp3" / "v2" / "r1_select.json").read_text())
    k, gamma = selected["best"]["k"], selected["best"]["gamma"]
    assert (k, gamma) == (20, 2.5)
    heads = [tuple(pair) for pair in selected["ranking"][:k]]
    sets = {"r1": heads, **matched_sets(heads, 2 if args.quick else 39, 12)}
    run.write_config({"eval": asdict(cfg), "head_sets": sets, "gamma": gamma}, resume=args.resume)
    done = run.done_keys() if args.resume else set()
    jobs = [(s, i) for s in cfg.test_systems for i in cfg.input_types if f"{s}|{i}" not in done]
    if jobs:
        w = Chronos2Wrapper(cfg.model_id, device=cfg.device, amp=cfg.amp, batch_size=cfg.batch_size)
        methods = {"zeroshot": lambda bt: w.predict(bt, bt.H)}
        for name, hs in sets.items():
            gain = r1_gain(w.hooked.n_layers, w.hooked.n_heads, hs, gamma)
            methods[name] = partial(w.predict_hooked, spec=HookSpec(cross_gain=gain))
            if name == "r1":
                for pos in ("context", "future"):
                    methods[pos] = partial(w.predict_hooked, spec=HookSpec(cross_gain=gain, cross_gain_positions=pos))
        systems = systems_by_key()
        start = time.monotonic()
        for i, (system, inp) in enumerate(jobs, 1):
            rec = eval_cell(methods, cfg, systems[system], cfg.base_L, inp, cfg.base_delta, cfg.base_snr,
                            f"{system}|{inp}", run)
            run.append_metrics(rec)
            logging.info("[%d/%d] %s ETA %s", i, len(jobs), rec["key"],
                         fmt_seconds((time.monotonic() - start) * (len(jobs) - i) / i))
    summarize(run, cfg)
    run.mark_done()


if __name__ == "__main__":
    main()
