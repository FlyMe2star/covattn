"""DaISy datasets (confirmatory set of docs/gate_protocol.md) -> uniform records in the Exp13 format.

Inclusion was fixed in docs/gate_protocol.md before download: real measured data, at least one input, not
described as closed loop, N >= 576. Columns follow each dataset's description (time columns dropped); every
output is a target (one record per output) with all inputs as covariates.

heating_system: the description lists u and y but no column order; the file has 3 columns and the first is
equally spaced with step 2.0 (= the stated 2.0 s sampling interval), so column 1 is time, 2 is u, 3 is y.

  python scripts/daisy_prepare.py --download      # raw files -> data/daisy/raw (DaISy, KU Leuven)
  python scripts/daisy_prepare.py                 # -> data/daisy/processed/records.npz / records.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://ftp.esat.kuleuven.be/pub/SISTA/data"
# name -> (DaISy folder, number of columns, time column or None, input columns, output columns), 0-based
SPEC = {
    "CD_player_arm": ("mechanical", 4, None, [0, 1], [2, 3]),
    "ballbeam": ("mechanical", 2, None, [0], [1]),
    "dryer": ("mechanical", 2, None, [0], [1]),
    "dryer2": ("process_industry", 7, 0, [1, 2, 3], [4, 5, 6]),
    "evaporator": ("process_industry", 6, None, [0, 1, 2], [3, 4, 5]),
    "exchanger": ("process_industry", 3, 0, [1], [2]),
    "flexible_structure": ("mechanical", 30, None, [0, 1], list(range(2, 30))),
    "flutter": ("mechanical", 2, None, [0], [1]),
    "glassfurnace": ("process_industry", 10, 0, [1, 2, 3], [4, 5, 6, 7, 8, 9]),
    "heating_system": ("thermic", 3, 0, [1], [2]),
    "robot_arm": ("mechanical", 2, None, [0], [1]),
    "thermic_res_wall": ("thermic", 3, None, [0, 1], [2]),
    "winding": ("process_industry", 7, None, [0, 1, 2, 3, 4], [5, 6]),
}
DATASETS = list(SPEC)


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for name, (folder, *_) in SPEC.items():
        dest = raw / f"{name}.dat.gz"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"{BASE}/{folder}/{name}.dat.gz"
        tmp = dest.with_suffix(".tmp")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.replace(dest)
        print(f"downloaded {name}: {dest.stat().st_size} bytes")


def load(raw: Path, name: str) -> np.ndarray:
    with gzip.open(raw / f"{name}.dat.gz", "rt") as f:
        a = np.loadtxt(f, dtype=np.float64)
    ncol, tcol = SPEC[name][1], SPEC[name][2]
    if a.ndim != 2 or a.shape[1] != ncol:
        raise ValueError(f"{name}: expected {ncol} columns, got {a.shape}")
    if tcol is not None:
        d = np.diff(a[:, tcol])
        if not ((d > 0).all() and np.allclose(d, d[0])):
            raise ValueError(f"{name}: column {tcol + 1} is not an equally spaced time column")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: non-finite values")
    return a


def records(raw: Path) -> list[dict]:
    R = []
    for name in DATASETS:
        a = load(raw, name)
        _, _, _, ucols, ycols = SPEC[name]
        u = a[:, ucols].T.copy()
        for j, c in enumerate(ycols):
            R.append(dict(bench=name, name=f"y{j + 1}", u=u, y=a[:, c].copy()))
    return R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/daisy/raw")
    ap.add_argument("--out", default="data/daisy/processed")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    if args.download:
        download(raw)
        return
    out.mkdir(parents=True, exist_ok=True)
    meta, arrays = [], {}
    for i, r in enumerate(records(raw)):
        meta.append(dict(i=i, bench=r["bench"], name=r["name"], n=len(r["y"]), k=r["u"].shape[0]))
        arrays[f"u{i}"], arrays[f"y{i}"] = r["u"].astype(np.float32), r["y"].astype(np.float32)
        print(f"{r['bench']:20s} {r['name']:5s} N={len(r['y']):6d} K={r['u'].shape[0]}")
    np.savez_compressed(out / "records.npz", **arrays)
    (out / "records.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{len(meta)} records -> {out}")


if __name__ == "__main__":
    main()
