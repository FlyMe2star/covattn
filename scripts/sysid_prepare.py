"""Real system-identification benchmarks -> uniform records + model-free characteristics (no forecasting model).

Measured lab rigs with an input designed by the experimenters (exogenous, open loop), from the
nonlinear-benchmark collection (loader URLs in github.com/MaartenSchoukens/nonlinear_benchmarks, BSD-3):
  cascaded_tanks  pump voltage -> lower tank level (Ts 4 s), estimation + validation record
  ced             coupled electric drives, input -> belt speed (Ts 0.02 s), PRBS x3 + uniform x2 records
  wh              Wiener-Hammerstein electronic circuit, full benchmark record (fs 51.2 kHz)
  silverbox       electronic Duffing oscillator: SNLS80mV (noise "arrow" + multisines), Schroeder80mV
  parwh           parallel Wiener-Hammerstein circuit: validation arrow, validation per amplitude, estimation
                  phases 0-1 per amplitude (periods concatenated)
  f16             F-16 ground vibration test: force -> acceleration 1, full multisine + sine sweep levels 1-7
  fsm             fine steering mirror, 2 inputs -> each of 2 outputs (both inputs as covariates)
Excluded in advance: EMPS (closed-loop position control; the input is computed from the output, so a change of
the future input is not an exogenous intervention).

Outputs (data/sysid/processed/): records.npz (u: list of (K, N) inputs, y: (N,), meta) and records.json with
per-record metadata and the model-free characteristics:
  arx_memory   samples until 95% of the cumulative |impulse response| of an ARX fitted to the record's first
               min(N, 8192) samples (orders by AIC, extended grid), capped at 4096; NaN if the fit is unstable
Window-level periodicity is computed by exp13 for each forecast window (rule in docs/sysid_protocol.md).

  python scripts/sysid_prepare.py --raw data/sysid/raw --out data/sysid/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter

from covattn.baselines import fit_arx

NA = (1, 2, 3, 4, 6, 8)
NB = (1, 2, 4, 8, 17, 33, 48)


def _v(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def records(raw: Path) -> list[dict]:
    R: list[dict] = []

    def add(bench, name, u, y, ts, note=""):
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))
        R.append(dict(bench=bench, name=name, u=u, y=_v(y), ts=float(ts), note=note))

    m = loadmat(next((raw / "cascaded_tanks").rglob("dataBenchmark.mat")))
    ts = float(m["Ts"][0, 0])
    add("cascaded_tanks", "est", _v(m["uEst"]), m["yEst"], ts)
    add("cascaded_tanks", "val", _v(m["uVal"]), m["yVal"], ts)

    m = loadmat(next((raw / "ced").rglob("DATAPRBS.MAT")))
    for i in (1, 2, 3):
        add("ced", f"prbs{i}", _v(m[f"u{i}"]), m[f"z{i}"], 0.02)
    m = loadmat(next((raw / "ced").rglob("DATAUNIF.MAT")))
    for i in (11, 12):
        add("ced", f"unif{i}", _v(m[f"u{i}"]), m[f"z{i}"], 0.02)

    m = loadmat(next((raw / "wh").rglob("WienerHammerBenchMark.mat")))
    add("wh", "full", _v(m["uBenchMark"]), m["yBenchMark"], 1.0 / float(m["fs"][0, 0]))

    for f in ("SNLS80mV", "Schroeder80mV"):
        m = loadmat(next((raw / "silverbox").rglob(f"{f}.mat")))
        add("silverbox", f, _v(m["V1"]), m["V2"], 1.0 / 610.35)

    m = loadmat(next((raw / "parwh").rglob("ParWHData.mat")))
    fs = float(m["fs"][0, 0])
    add("parwh", "val_arrow", _v(m["uValArr"][:, 0]), m["yValArr"][:, 0], 1 / fs, "first column of the arrow arrays")
    for a in range(m["uVal"].shape[3]):
        add("parwh", f"val_amp{a}", m["uVal"][:, :, 0, a].T.reshape(-1), m["yVal"][:, :, 0, a].T.reshape(-1), 1 / fs)
        for ph in (0, 1):
            add("parwh", f"est_amp{a}_ph{ph}", m["uEst"][:, :, ph, a].T.reshape(-1),
                m["yEst"][:, :, ph, a].T.reshape(-1), 1 / fs)

    for kind in ("FullMSine", "SineSw"):
        for f in sorted((raw / "f16").rglob(f"F16Data_{kind}_Level*.mat")):
            m = loadmat(f)
            add("f16", f.stem.replace("F16Data_", ""), _v(m["Force"]), _v(m["Acceleration"][0]),
                1.0 / float(m["Fs"][0, 0]), "output = acceleration 1")

    z = np.load(next((raw / "fsm").rglob("combined_data.npz")), allow_pickle=False)
    for key in sorted(k for k in z.files if k.startswith("u_") and k.endswith("_test")):
        lvl = key[2:-5]
        u, y = z[key], z[f"y_{lvl}_test"]  # (N, periods, realisations, channels)
        for r in range(u.shape[2]):
            uu = np.stack([u[:, :, r, c].T.reshape(-1) for c in range(u.shape[3])])
            for c in range(y.shape[3]):
                add("fsm", f"{lvl}_test_r{r}_out{c}", uu, y[:, :, r, c].T.reshape(-1), np.nan,
                    "both inputs as covariates")
    return R


def arx_memory(u: np.ndarray, y: np.ndarray, n_fit: int = 8192, cap: int = 4096) -> tuple[float, dict]:
    u0, y0 = u[0, :n_fit], y[:n_fit]
    f = fit_arx(y0, u0, na_grid=NA, nb_grid=NB)
    info = dict(na=f.na, nb=f.nb, stable=f.stable())
    if not f.stable():
        return float("nan"), info
    h = np.abs(lfilter(f.b, np.concatenate([[1.0], -f.a]), np.r_[1.0, np.zeros(cap - 1)]))
    c = np.cumsum(h)
    return float(np.searchsorted(c, 0.95 * c[-1]) + 1), info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/sysid/raw")
    ap.add_argument("--out", default="data/sysid/processed")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = records(Path(args.raw))
    meta = []
    arrays = {}
    for i, r in enumerate(R):
        assert r["u"].shape[1] == len(r["y"]), (r["bench"], r["name"])
        mem, info = arx_memory(r["u"], r["y"])
        meta.append(dict(i=i, bench=r["bench"], name=r["name"], n=len(r["y"]), k=r["u"].shape[0], ts=r["ts"],
                         note=r["note"], arx_memory=mem, **{f"arx_{k}": v for k, v in info.items()}))
        arrays[f"u{i}"], arrays[f"y{i}"] = r["u"].astype(np.float32), r["y"].astype(np.float32)
        print(f"{r['bench']:15s} {r['name']:28s} N={len(r['y']):7d} K={r['u'].shape[0]} memory={mem:7.0f} {info}")
    np.savez_compressed(out / "records.npz", **arrays)
    (out / "records.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
