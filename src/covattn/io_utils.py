"""Result directories, reproducibility metadata, resumable sub-tasks, logging, device helpers."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

log = logging.getLogger("covattn")

REPO_ROOT = Path(__file__).resolve().parents[2]


def results_root() -> Path:
    """RESULTS_DIR env var (Drive on Colab) or <repo>/results."""
    return Path(os.environ.get("RESULTS_DIR", REPO_ROOT / "results")).expanduser()


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    # the pipeline warns on every call when prediction_length > default; keep logs readable
    logging.getLogger("transformers").setLevel(logging.ERROR)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            dirty = subprocess.run(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
            ).stdout.strip()
            return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "not-a-git-repo"


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    except (AssertionError, RuntimeError) as err:  # broken/hidden CUDA device must not stop a CPU run
        info["gpu"] = f"unavailable ({type(err).__name__})"
    try:
        import chronos

        info["chronos"] = getattr(chronos, "__version__", "unknown")
    except ImportError:
        pass
    return info


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


class RunDir:
    """<results_root>/<exp_name>/<run_name>/ with config.json, metrics.jsonl, done.marker and
    per-subtask artefacts. Sub-tasks are identified by string keys; `--resume` skips keys
    already present in metrics.jsonl (written only after the artefact is saved)."""

    def __init__(self, exp_name: str, run_name: str, root: Path | None = None) -> None:
        self.path = (root or results_root()) / exp_name / run_name
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "cells").mkdir(exist_ok=True)
        self.metrics_file = self.path / "metrics.jsonl"

    def write_config(self, config: Any, resume: bool, ignore_keys: tuple[str, ...] = ()) -> None:
        """Save config.json. On --resume the stored config must match except for `ignore_keys`
        (keys that only select *which* sub-tasks run or how fast, e.g. context subsets, batch size)."""
        cfg_file = self.path / "config.json"
        payload = {"config": _jsonable(config), "git_commit": git_commit(), "env": env_info(), "time": time.ctime()}
        if resume and cfg_file.exists():
            old = json.loads(cfg_file.read_text(encoding="utf-8"))
            strip = lambda c: {k: v for k, v in c.items() if k not in ignore_keys}  # noqa: E731
            if strip(old.get("config", {})) != strip(payload["config"]):
                raise RuntimeError(
                    f"--resume with a different config than {cfg_file}. Use a new --run-name or delete the run."
                )
            hist = self.path / "resume_history.jsonl"
            with hist.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
            return
        if not resume and self.metrics_file.exists() and self.metrics_file.stat().st_size > 0:
            raise RuntimeError(f"{self.path} already has results. Pass --resume or choose another --run-name.")
        cfg_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def done_keys(self) -> set[str]:
        if not self.metrics_file.exists():
            return set()
        keys = set()
        for line in self.metrics_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    keys.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    log.warning("skipping malformed metrics line")
        return keys

    def append_metrics(self, record: dict[str, Any]) -> None:
        with self.metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(record)) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read_metrics(self) -> list[dict[str, Any]]:
        if not self.metrics_file.exists():
            return []
        return [json.loads(l) for l in self.metrics_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    def cell_path(self, key: str, suffix: str = ".npz") -> Path:
        safe = "".join(c if c.isalnum() or c in "-_.=" else "_" for c in key)
        return self.path / "cells" / f"{safe}{suffix}"

    def save_arrays(self, key: str, **arrays: np.ndarray) -> Path:
        p = self.cell_path(key)
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, p)
        return p

    def mark_done(self, extra: dict[str, Any] | None = None) -> None:
        (self.path / "done.marker").write_text(json.dumps(_jsonable(extra or {"time": time.ctime()})), encoding="utf-8")

    @property
    def is_done(self) -> bool:
        return (self.path / "done.marker").exists()


def default_num_workers() -> int:
    """DataLoader workers: 0 on Windows (spawn + notebook issues), else min(4, cpus)."""
    if os.name == "nt":
        return 0
    return min(4, os.cpu_count() or 1)


def fmt_seconds(s: float) -> str:
    if not np.isfinite(s):
        return "?"
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.1f}min"
    return f"{s / 3600:.2f}h"


def iter_chunks(n: int, size: int) -> Iterable[tuple[int, int]]:
    for i in range(0, n, size):
        yield i, min(n, i + size)
