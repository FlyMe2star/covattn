"""Low Carbon London (LCL) smart-meter trial: raw zip -> household-level half-hourly matrix + tariff.

Source (CC BY 4.0, UK Power Networks), London Datastore dataset vqm0d
"SmartMeter Energy Consumption Data in London Households":
  Partitioned LCL Data.zip  (168 CSVs; LCLid, stdorToU, DateTime, KWH/hh, Acorn, Acorn_grouped), or
  LCL-FullData.zip          (the same rows in one ~10 GB CSV); fetched and verified by lcl_download.py
  Tariffs.xlsx              (2013 half-hourly dToU price signal: Normal / High / Low)
The dToU group (stdorToU == "ToU") received day-ahead High / Low / Normal price signals during 2013;
the Std group was on a flat tariff throughout and serves as the control group.

No model is run here. CSVs are read in chunks of 1M rows and the matrix is a disk memmap, so peak
memory stays well below 1 GB. Outputs (default `data/lcl/processed/`):
  households.parquet  one row per household (row order of the matrix): id, group, acorn, acorn_grouped,
                      first / last reading, number of readings
  hh_kwh.npy          (n_households, n_slots) float32 kWh on a 30-min grid from GRID_START, NaN = missing
  tariff.parquet      tariff signal and price on the same grid (Normal outside 2013, flagged by in_2013)
  prepare_log.json    row counts and data-quality counters

Usage:
  python scripts/lcl_prepare.py --raw data/lcl/raw --out data/lcl/processed
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

GRID_START = pd.Timestamp("2011-11-01 00:00")
GRID_END = pd.Timestamp("2014-03-01 00:00")  # exclusive
SLOT = pd.Timedelta("30min")
MAX_HH = 6000  # dataset description: 5,567 households
PRICE_P_PER_KWH = {"Low": 3.99, "Normal": 11.76, "High": 67.20}  # dataset description, London Datastore
ZIP_NAMES = ("Partitioned_LCL_Data.zip", "LCL-FullData.zip")  # as saved by scripts/lcl_download.py
CHUNK_ROWS = int(os.environ.get("LCL_CHUNK_ROWS", 1_000_000))
KAGGLE_DIR = "kaggle"  # mirror jeanmidev/smart-meters-in-london: informations_households.csv + halfhourly/block_*.zip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/lcl/raw")
    ap.add_argument("--out", default="data/lcl/processed")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_slots = int((GRID_END - GRID_START) / SLOT)
    tmp = out / "hh_kwh.tmp.npy"
    try:
        _build(raw, out, tmp, n_slots)
    finally:
        if tmp.exists():  # never leave the 1 GB scratch matrix behind (it would be copied to Drive)
            tmp.unlink()


def _csv_members(path: Path):
    """Yield (name, bytes-reader factory) for every CSV in the zip, also inside nested zips."""
    z = zipfile.ZipFile(path)
    members = sorted(z.namelist())
    print(f"zip members: {len(members)}; first: {members[:3]}")
    for n in members:
        if n.lower().endswith(".csv"):
            yield n, (lambda n=n: z.open(n))
        elif n.lower().endswith(".zip"):
            inner = zipfile.ZipFile(io.BytesIO(z.read(n)))
            for m in sorted(inner.namelist()):
                if m.lower().endswith(".csv"):
                    yield f"{n}/{m}", (lambda m=m, inner=inner: inner.open(m))


def _chunks(files):
    """Yield (member name, DataFrame of <= CHUNK_ROWS rows); also handles the single 10 GB CSV."""
    for name, opener in files:
        with opener() as f:
            for df in pd.read_csv(io.TextIOWrapper(f, encoding="utf-8-sig"), dtype=str, chunksize=CHUNK_ROWS):
                yield name, df


def _group(v: str) -> str:
    s = v.strip().lower()
    if "tou" in s:  # "ToU", "dToU", "DTOU"
        return "ToU"
    if s.startswith("std") or s.startswith("standard"):
        return "Std"
    return v.strip()


def _parse_time(s: pd.Series) -> pd.Series:
    s = s.str.strip()
    sample = s.dropna().iloc[0] if s.notna().any() else ""
    if "/" in sample:  # e.g. 12/10/2012 00:30 -> UK day-first
        return pd.to_datetime(s, dayfirst=True, errors="coerce")
    return pd.to_datetime(s.str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce")


def _build(raw: Path, out: Path, tmp: Path, n_slots: int) -> None:
    mat = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(MAX_HH, n_slots))
    mat[:] = np.nan
    meta: dict[str, dict] = {}
    log = dict(rows=0, bad_time=0, off_grid=0, out_of_grid=0, duplicate=0, missing_kwh=0, group_conflict=0,
               raw_group_values={})

    src = next((raw / n for n in ZIP_NAMES if (raw / n).exists()), None)
    info = None
    if src is not None:
        print(f"source: London Datastore {src.name}")
        files = list(_csv_members(src))
    elif (raw / KAGGLE_DIR / "informations_households.csv").exists():
        kd = raw / KAGGLE_DIR
        print(f"source: Kaggle mirror {kd}")
        info = pd.read_csv(kd / "informations_households.csv", dtype=str)
        info.columns = [c.strip().lower() for c in info.columns]
        info = info.set_index(info["lclid"].str.strip())
        print(f"households in info file: {len(info)}, groups {info['stdortou'].value_counts().to_dict()}")
        files = [f for p in sorted((kd / "halfhourly").glob("block_*.zip"), key=lambda p: int(p.stem.split('_')[1]))
                 for f in _csv_members(p)]
    else:
        raise SystemExit(f"none of {ZIP_NAMES} or {KAGGLE_DIR}/ in {raw}; run scripts/lcl_download.py first")
    if not files:
        raise SystemExit("no CSV files found in the zip (see member list above)")
    print(f"{len(files)} csv files; reading in chunks of {CHUNK_ROWS} rows")
    for i, (name, df) in enumerate(_chunks(files)):
        if i == 0:
            print(f"first file {name}: columns {list(df.columns)}")
            print(df.head(3).to_string())
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"tstp": "datetime"})
        if info is not None:  # Kaggle half-hourly blocks carry no group / Acorn columns
            key = df["lclid"].str.strip()
            missing = ~key.isin(info.index)
            if missing.any():
                raise SystemExit(f"{name}: {key[missing].nunique()} households not in informations_households.csv")
            for c in ("stdortou", "acorn", "acorn_grouped"):
                df[c] = info.loc[key, c].to_numpy()
        need = {"lclid", "stdortou", "datetime"}
        kwh_cols = [c for c in df.columns if c.startswith("kwh") or "energy" in c]
        if not need <= set(df.columns) or len(kwh_cols) != 1:
            raise SystemExit(f"{name}: unexpected columns {list(df.columns)} (need {sorted(need)} + one kWh column)")
        kwh_col = kwh_cols[0]
        for c in ("acorn", "acorn_grouped"):
            if c not in df.columns:
                df[c] = ""
        for v, c in df["stdortou"].value_counts().items():
            log["raw_group_values"][v] = log["raw_group_values"].get(v, 0) + int(c)
        df["stdortou"] = df["stdortou"].map(_group)
        t = _parse_time(df["datetime"])
        if i == 0:
            print(f"time parse check: {df['datetime'].iloc[0]!r} -> {t.iloc[0]}")
        kwh = pd.to_numeric(df[kwh_col].str.strip(), errors="coerce").to_numpy(np.float32)
        ids = df["lclid"].str.strip().to_numpy()
        log["rows"] += len(df)
        ok = t.notna().to_numpy().copy()
        log["bad_time"] += int((~ok).sum())
        tf = t.dt.floor("30min")
        # off-grid stamps (e.g. 12:37:27) are meter events with "Null" kWh that collide with the regular
        # reading of the same slot; they are dropped, not floored, so they can never replace a real value
        off = ((tf != t) & t.notna()).to_numpy()
        log["off_grid"] += int(off.sum())
        ok &= ~off
        col = ((tf - GRID_START) // SLOT).to_numpy(dtype=np.float64, na_value=-1)
        on_grid = ok.copy()
        ok &= (col >= 0) & (col < n_slots)
        log["out_of_grid"] += int((on_grid & ~ok).sum())
        log["missing_kwh"] += int(np.isnan(kwh[ok]).sum())

        for hid, sub in df.loc[ok, ["lclid", "stdortou", "acorn", "acorn_grouped"]].groupby("lclid", sort=False):
            hid = hid.strip()
            grp = sub["stdortou"].iloc[0].strip()
            if hid not in meta:
                meta[hid] = dict(row=len(meta), group=grp, acorn=sub["acorn"].iloc[0].strip(),
                                 acorn_grouped=sub["acorn_grouped"].iloc[0].strip())
            if sub["stdortou"].str.strip().nunique() > 1 or meta[hid]["group"] != grp:
                log["group_conflict"] += 1
        assert len(meta) <= MAX_HH
        rows = np.fromiter((meta[h.strip()]["row"] for h in ids[ok]), dtype=np.int64, count=int(ok.sum()))
        cols = col[ok].astype(np.int64)
        vals = kwh[ok]
        # keep the first reading of a duplicated (household, slot), also across files
        order = np.lexsort((np.arange(len(rows)), cols, rows))
        rows, cols, vals = rows[order], cols[order], vals[order]
        first = np.ones(len(rows), bool)
        first[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
        filled = ~np.isnan(mat[rows, cols]) | ~first
        log["duplicate"] += int(filled.sum())
        # a slot already filled holds the earlier value; write only fresh slots
        fresh = first.copy()
        fresh[first] = np.isnan(mat[rows[first], cols[first]])
        # missing kWh stays NaN but still occupies the slot as "seen"; use a sentinel then reset
        v = vals[fresh]
        v = np.where(np.isnan(v), np.float32(-1.0), v)
        mat[rows[fresh], cols[fresh]] = v
        if (i + 1) % 10 == 0:
            print(f"  {i + 1} chunks, {len(meta)} households, {log['rows']} rows", flush=True)
    print(f"  done: {len(meta)} households, {log['rows']} rows")
    mat.flush()

    n_hh = len(meta)
    if n_hh == 0:
        raise SystemExit("no household rows parsed")
    final = np.lib.format.open_memmap(out / "hh_kwh.npy", mode="w+", dtype=np.float32, shape=(n_hh, n_slots))
    for a in range(0, n_hh, 500):
        blk = np.array(mat[a:min(a + 500, n_hh)])
        blk[blk == -1.0] = np.nan  # sentinel: reading present but kWh unparseable ("Null")
        final[a:a + 500] = blk
    final.flush()
    del mat, final
    tmp.unlink()

    hh = pd.DataFrame([dict(id=k, **v) for k, v in meta.items()]).sort_values("row").reset_index(drop=True)
    arr = np.load(out / "hh_kwh.npy", mmap_mode="r")
    first_s, last_s, n_read = [], [], []
    for a in range(0, n_hh, 500):
        fin = np.isfinite(np.asarray(arr[a:a + 500]))
        any_ = fin.any(1)
        first_s += list(np.where(any_, fin.argmax(1), -1))
        last_s += list(np.where(any_, n_slots - 1 - fin[:, ::-1].argmax(1), -1))
        n_read += list(fin.sum(1))
    hh["first"] = [GRID_START + s * SLOT if s >= 0 else pd.NaT for s in first_s]
    hh["last"] = [GRID_START + s * SLOT if s >= 0 else pd.NaT for s in last_s]
    hh["n_readings"] = n_read
    hh.to_parquet(out / "households.parquet", index=False)

    tar = pd.read_excel(raw / "Tariffs.xlsx", sheet_name="Sheet1")
    tar.columns = ["t", "tariff"]
    tar["t"] = pd.to_datetime(tar["t"])
    grid = pd.DataFrame({"t": pd.date_range(GRID_START, GRID_END - SLOT, freq="30min")})
    g = grid.merge(tar, on="t", how="left")
    g["in_2013"] = g["tariff"].notna()
    g["tariff"] = g["tariff"].fillna("Normal")
    g["price"] = g["tariff"].map(PRICE_P_PER_KWH).astype("float32")
    g.to_parquet(out / "tariff.parquet", index=False)

    log.update(n_households=n_hh, groups=hh["group"].value_counts().to_dict(), n_slots=n_slots,
               grid_start=str(GRID_START), tariff_2013=g.loc[g["in_2013"], "tariff"].value_counts().to_dict())
    (out / "prepare_log.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
