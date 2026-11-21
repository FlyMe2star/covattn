"""LCL scripts: zip parsing (dedup, Null, off-grid times) and recovery of a known price effect."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
HEADER = "LCLid,stdorToU,DateTime,KWH/hh (per half hour) ,Acorn,Acorn_grouped"


def _tariff(path: Path) -> None:
    t = pd.date_range("2013-01-01", periods=365 * 48, freq="30min")
    tariff = np.array(["Normal"] * len(t), dtype=object)
    rng = np.random.default_rng(1)
    for d in rng.choice(365, 120, replace=False):
        s = d * 48 + 34 if rng.random() < 0.5 else d * 48 + 10
        tariff[s:s + 8] = "High" if rng.random() < 0.5 else "Low"
    pd.DataFrame({"TariffDateTime": t, "Tariff": tariff}).to_excel(path, index=False)


def _run(script: str, *args: str, env: dict | None = None) -> None:
    subprocess.run([sys.executable, str(REPO / "scripts" / script), *args], check=True, cwd=REPO,
                   env={**os.environ, **(env or {})}, capture_output=True)


def test_prepare_parsing(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _tariff(raw / "Tariffs.xlsx")
    f0 = [HEADER, "A,Std,2012-10-12 00:30:00.0000000,0 ,X,Affluent", "A,Std,2012-10-12 01:00:00.0000000,0.25 ,X,Affluent",
          "A,Std,2012-10-12 01:00:00.0000000,9 ,X,Affluent", "B,ToU,2013-01-01 00:00:07.0000000,Null,Y,Comfortable"]
    f1 = [HEADER, "B,ToU,2013-01-01 00:30:00.0000000,0.5 ,Y,Comfortable", "A,Std,2012-10-12 00:30:00.0000000,7 ,X,Affluent"]
    with zipfile.ZipFile(raw / "Partitioned_LCL_Data.zip", "w") as z:
        z.writestr("d/LCL_0.csv", "\n".join(f0) + "\n")
        z.writestr("d/LCL_1.csv", "\n".join(f1) + "\n")
    _run("lcl_prepare.py", "--raw", str(raw), "--out", str(tmp_path / "out"))
    a = np.load(tmp_path / "out" / "hh_kwh.npy")
    hh = pd.read_parquet(tmp_path / "out" / "households.parquet")
    t0 = pd.Timestamp("2011-11-01")
    s = lambda ts: int((pd.Timestamp(ts) - t0) / pd.Timedelta("30min"))  # noqa: E731
    assert list(hh["id"]) == ["A", "B"] and list(hh["group"]) == ["Std", "ToU"]
    assert a[0, s("2012-10-12 00:30")] == 0.0  # first reading kept, also across files
    assert a[0, s("2012-10-12 01:00")] == 0.25  # within-file duplicate dropped
    assert np.isnan(a[1, s("2013-01-01 00:00")])  # off-grid "Null" reading dropped, slot stays missing
    assert a[1, s("2013-01-01 00:30")] == 0.5
    assert np.isfinite(a).sum() == 3
    log = json.loads((tmp_path / "out" / "prepare_log.json").read_text())
    assert log["duplicate"] == 2 and log["off_grid"] == 1 and log["missing_kwh"] == 0


def test_prepare_format_variants(tmp_path: Path) -> None:
    """Nested zip, no Acorn columns, Standard/DTOU labels, UK day-first dates."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _tariff(raw / "Tariffs.xlsx")
    csv = "\n".join(["LCLid,stdorToU,DateTime,KWH/hh (per half hour)",
                     "A,Standard,12/10/2012 00:30,0.1", "B,DTOU,01/02/2013 13:00,0.2"]) + "\n"
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("LCL_0.csv", csv)
    with zipfile.ZipFile(raw / "Partitioned_LCL_Data.zip", "w") as z:
        z.writestr("Partitioned LCL Data/part0.zip", inner.getvalue())
    _run("lcl_prepare.py", "--raw", str(raw), "--out", str(tmp_path / "out"))
    a = np.load(tmp_path / "out" / "hh_kwh.npy")
    hh = pd.read_parquet(tmp_path / "out" / "households.parquet")
    t0 = pd.Timestamp("2011-11-01")
    s = lambda ts: int((pd.Timestamp(ts) - t0) / pd.Timedelta("30min"))  # noqa: E731
    assert list(hh["group"]) == ["Std", "ToU"]
    assert np.isclose(a[0, s("2012-10-12 00:30")], 0.1) and np.isclose(a[1, s("2013-02-01 13:00")], 0.2)
    assert not (tmp_path / "out" / "hh_kwh.tmp.npy").exists()


def test_prepare_fulldata_chunked(tmp_path: Path) -> None:
    """Single-CSV LCL-FullData.zip read in tiny chunks gives the same dedup / Null handling."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _tariff(raw / "Tariffs.xlsx")
    rows = [HEADER, "A,Std,2012-10-12 00:30:00.0000000,0 ,X,Affluent", "A,Std,2012-10-12 01:00:00.0000000,0.25 ,X,Affluent",
            "A,Std,2012-10-12 01:00:00.0000000,9 ,X,Affluent", "B,ToU,2013-01-01 00:00:07.0000000,Null,Y,Comfortable",
            "B,ToU,2013-01-01 00:30:00.0000000,0.5 ,Y,Comfortable", "A,Std,2012-10-12 00:30:00.0000000,7 ,X,Affluent"]
    with zipfile.ZipFile(raw / "LCL-FullData.zip", "w") as z:
        z.writestr("CC_LCL-FullData.csv", "\n".join(rows) + "\n")
    _run("lcl_prepare.py", "--raw", str(raw), "--out", str(tmp_path / "out"), env={"LCL_CHUNK_ROWS": "2"})
    a = np.load(tmp_path / "out" / "hh_kwh.npy")
    t0 = pd.Timestamp("2011-11-01")
    s = lambda ts: int((pd.Timestamp(ts) - t0) / pd.Timedelta("30min"))  # noqa: E731
    assert a[0, s("2012-10-12 00:30")] == 0.0 and a[0, s("2012-10-12 01:00")] == 0.25
    assert np.isnan(a[1, s("2013-01-01 00:00")]) and a[1, s("2013-01-01 00:30")] == 0.5
    log = json.loads((tmp_path / "out" / "prepare_log.json").read_text())
    assert log["duplicate"] == 2 and log["rows"] == 6


def test_prepare_kaggle_layout(tmp_path: Path) -> None:
    """Kaggle mirror: groups from informations_households.csv, blocks in numeric order, off-grid Null dropped."""
    raw = tmp_path / "raw"
    kd = raw / "kaggle" / "halfhourly"
    kd.mkdir(parents=True)
    _tariff(raw / "Tariffs.xlsx")
    (raw / "kaggle" / "informations_households.csv").write_text(
        "LCLid,stdorToU,Acorn,Acorn_grouped,file\nA,Std,ACORN-A,Affluent,block_2\nB,ToU,ACORN-,ACORN-,block_10\n")
    blocks = {2: ["A,2012-10-12 00:30:00.0000000, 0.1 ", "A,2012-10-12 00:37:27.0000000,Null"],
              10: ["B,2013-01-01 00:30:00.0000000, 0.5 "]}
    for k, rows in blocks.items():
        with zipfile.ZipFile(kd / f"block_{k}.zip", "w") as z:
            z.writestr(f"block_{k}.csv", "\n".join(["LCLid,tstp,energy(kWh/hh)", *rows]) + "\n")
    _run("lcl_prepare.py", "--raw", str(raw), "--out", str(tmp_path / "out"))
    a = np.load(tmp_path / "out" / "hh_kwh.npy")
    hh = pd.read_parquet(tmp_path / "out" / "households.parquet")
    t0 = pd.Timestamp("2011-11-01")
    s = lambda ts: int((pd.Timestamp(ts) - t0) / pd.Timedelta("30min"))  # noqa: E731
    assert list(hh["id"]) == ["A", "B"] and list(hh["group"]) == ["Std", "ToU"]
    assert list(hh["acorn_grouped"]) == ["Affluent", "ACORN-"]
    assert np.isclose(a[0, s("2012-10-12 00:30")], 0.1) and np.isclose(a[1, s("2013-01-01 00:30")], 0.5)
    log = json.loads((tmp_path / "out" / "prepare_log.json").read_text())
    assert log["off_grid"] == 1 and log["duplicate"] == 0 and np.isfinite(a).sum() == 2


def test_download_check_rejects_bad_files(tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import lcl_download as D
    finally:
        sys.path.pop(0)
    html = tmp_path / "x.zip"
    html.write_bytes(b"<!DOCTYPE html><html>error</html>")
    assert "size" in D.check(html, {"size": 999})
    assert D.check(html, {"size": html.stat().st_size}) == "not a zip file"
    good = tmp_path / "g.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("a.csv", "x\n1\n")
    assert D.check(good, {"size": good.stat().st_size}) is None
    assert D.check(tmp_path / "missing.zip", {"size": 1}) == "missing"


@pytest.mark.parametrize("lag", [0, 1])
def test_feasibility_recovers_known_effect(tmp_path: Path, lag: int) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _tariff(tmp_path / "Tariffs.xlsx")
    tar13 = pd.read_excel(tmp_path / "Tariffs.xlsx")
    t = pd.date_range("2011-11-01", "2014-02-28 23:30", freq="30min")
    tariff = pd.Series("Normal", index=t, dtype=object)
    tariff.loc[tar13["TariffDateTime"]] = tar13["Tariff"].to_numpy()
    tar = pd.DataFrame({"t": t, "tariff": tariff.to_numpy(), "in_2013": t.year == 2013})
    tar["price"] = tar["tariff"].map({"Low": 3.99, "Normal": 11.76, "High": 67.2}).astype("float32")
    tar.to_parquet(data / "tariff.parquet", index=False)
    rng = np.random.default_rng(0)
    n = len(t)
    hod = (t.hour * 2 + t.minute // 30).to_numpy()
    base = 0.15 + 0.1 * np.sin(2 * np.pi * (hod - 30) / 48)
    common = 0.02 * rng.standard_normal(n)
    mult = np.where(tar["tariff"] == "High", 0.9, np.where(tar["tariff"] == "Low", 1.1, 1.0))
    if lag:  # load responds `lag` slots after the stamped signal (readings stamped at interval end)
        mult = np.concatenate([np.ones(lag), mult[:-lag]])
    grp = np.array(["ToU"] * 40 + ["Std"] * 80)
    x = np.empty((len(grp), n), np.float32)
    for i, g in enumerate(grp):
        x[i] = base * (1 + common) * rng.uniform(0.7, 1.3) * np.exp(0.3 * rng.standard_normal(n)) * (mult if g == "ToU" else 1)
    np.save(data / "hh_kwh.npy", x)
    pd.DataFrame({"id": [f"h{i}" for i in range(len(grp))], "group": grp, "acorn": "A",
                  "acorn_grouped": "Affluent"}).to_parquet(data / "households.parquet", index=False)
    _run("lcl_feasibility.py", "--data", str(data), env={"RESULTS_DIR": str(tmp_path / "res")})
    rep = json.loads((tmp_path / "res" / "lcl" / "feasibility" / "report.json").read_text())
    if lag:
        for period in ("GMT", "BST"):
            assert rep["alignment_high_runs"][period]["strongest_shift"] == lag
        _run("lcl_feasibility.py", "--data", str(data), "--tariff-shift", str(lag),
             env={"RESULTS_DIR": str(tmp_path / "res")})
        rep = json.loads((tmp_path / "res" / "lcl" / f"feasibility_shift+{lag}_bst+0" / "report.json").read_text())
    assert abs(rep["event_runs"]["High"]["rel_effect_median"] + 0.10) < 0.02
    assert abs(rep["event_runs"]["Low"]["rel_effect_median"] - 0.10) < 0.02
    for lv in ("High", "Low"):
        assert abs(rep["placebo_2012_schedule"][lv]["rel_effect_mean"]) < 0.02
    for period in ("GMT", "BST"):
        assert rep["alignment_high_runs"][period]["strongest_shift"] == 0
