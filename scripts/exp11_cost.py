"""Exp11 - inference cost of zero-shot, R1, R2 and dither (wall-clock per batch on the current GPU).

Zero-shot uses the official pipeline (SDPA attention); R1/R2 run the hooked model (eager group attention with the
cross-term decomposition), dither runs the pipeline K times. Training cost of R2 / full FT / LoRA is read from the
Exp3 run (r2_gates.json, ft_*/info.json) and reported alongside, with the GPU each was trained on.

  python scripts/exp11_cost.py --quick
  python scripts/exp11_cost.py --exp3-run v2        # Colab (A100)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from covattn.baselines import dither_predict
from covattn.io_utils import RunDir, results_root, setup_logging
from covattn.model import CovBatch
from covattn.repair import gain_spec, r1_gain
from covattn.systems import SampleSpec, default_systems, make_rng, make_sample

log = logging.getLogger("exp11")


@dataclass
class Exp11Config:
    run_name: str = "main"
    quick: bool = False
    exp3_run: str = "v2"
    contexts: list[int] = field(default_factory=lambda: [512, 2048])
    n_series: int = 64
    horizon: int = 64
    repeats: int = 5
    device: str = "auto"
    batch_size: int = 128


def _sync(dev) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--exp3-run", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    cfg = Exp11Config(quick=a.quick, device=a.device)
    if a.quick:
        cfg.run_name, cfg.exp3_run, cfg.contexts, cfg.n_series, cfg.repeats = "quick", "quick", [256], 8, 2
    cfg.run_name = a.run_name or cfg.run_name
    cfg.exp3_run = a.exp3_run or cfg.exp3_run
    run = RunDir("exp11", cfg.run_name)
    setup_logging(run.path / "log.txt")
    run.write_config(cfg, resume=True)
    from covattn.model import Chronos2Wrapper

    w = Chronos2Wrapper("amazon/chronos-2", device=cfg.device, batch_size=cfg.batch_size)
    d3 = results_root() / "exp3" / cfg.exp3_run
    r1 = json.loads((d3 / "r1_select.json").read_text())
    hk = w.hooked
    G1 = r1_gain(hk.n_layers, hk.n_heads, [tuple(x) for x in r1["ranking"][: r1["best"]["k"]]], r1["best"]["gamma"])
    G2 = torch.tensor(json.loads((d3 / "r2_gates.json").read_text())["gates"])
    sg = json.loads((d3 / "dither_select.json").read_text())["best"]
    methods = {"zeroshot": lambda bt: w.predict(bt, bt.H), "r1": lambda bt: w.predict_hooked(bt, gain_spec(G1)),
               "r2": lambda bt: w.predict_hooked(bt, gain_spec(G2)),
               "dither": lambda bt: dither_predict(w, bt, sg["sigma"], 8, seed=0)}
    sys = {s.key: s for s in default_systems()}["F2_a0.8"]
    rows = []
    for L in cfg.contexts:
        rng = make_rng(0, "test", "exp11", L)
        smp = [make_sample(sys, SampleSpec(context_length=L, horizon=cfg.horizon), rng) for _ in range(cfg.n_series)]
        bt = CovBatch(np.stack([s.y[:L] for s in smp]), np.stack([s.u[:L] for s in smp]),
                      np.stack([s.u[L:] for s in smp]))
        for m, fn in methods.items():
            fn(bt)  # warm-up
            ts = []
            for _ in range(cfg.repeats):
                _sync(w.device)
                t0 = time.perf_counter()
                fn(bt)
                _sync(w.device)
                ts.append(time.perf_counter() - t0)
            ms_per_series = 1000 * float(np.median(ts)) / cfg.n_series
            rows.append({"L": L, "method": m, "ms_per_series": ms_per_series})
            log.info("L=%d %s: %.2f ms per series (median of %d)", L, m, ms_per_series, cfg.repeats)
    train = {}
    f = d3 / "r2_gates.json"
    if f.exists():
        train["r2"] = {"seconds": json.loads(f.read_text()).get("train_seconds"), "params": int(G2.numel())}
    for mode in ("full", "lora"):
        f = d3 / f"ft_{mode}" / "info.json"
        if f.exists():
            info = json.loads(f.read_text())
            train[f"ft_{mode}"] = {"seconds": info.get("train_seconds"), "params": info.get("trainable_params")}
    gpu_train = json.loads((d3 / "config.json").read_text()).get("env", {}).get("gpu")
    gpu_now = torch.cuda.get_device_name(0) if w.device.type == "cuda" else "cpu"
    out = {"inference": rows, "training": train, "gpu_inference": gpu_now, "gpu_training_exp3": gpu_train}
    rep = results_root() / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / f"exp11_cost_{cfg.run_name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines = [f"# Exp11 cost ({cfg.run_name})", "", f"Inference GPU: {gpu_now}; batch of {cfg.n_series} series, H = "
             f"{cfg.horizon}, median of {cfg.repeats} runs after warm-up.", "", "| L | method | ms per series |",
             "|---|---|---|"] + [f"| {r['L']} | {r['method']} | {r['ms_per_series']:.2f} |" for r in rows]
    lines += ["", f"Training (Exp3 `{cfg.exp3_run}`, GPU {gpu_train}): " +
              ", ".join(f"{k}: {v['seconds'] / 60:.1f} min, {v['params']} trainable params" for k, v in train.items()
                        if v.get("seconds"))]
    (rep / f"exp11_summary_{cfg.run_name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run.mark_done()
    log.info("written %s", rep / f"exp11_summary_{cfg.run_name}.md")


if __name__ == "__main__":
    main()
