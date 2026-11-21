"""LCL dToU trial: model-free ground-truth price effects and their signal-to-noise (no forecasting model).

Question: is the real effect of the day-ahead price signal on the dToU group's load large and precise
enough to serve as ground truth for Chronos-2's what-if response? Everything here is fixed before any
model is run on these data.

Design (pre-declared):
  panel     households with >= 90% finite half-hourly readings in 2013, per group (ToU, Std);
            group load = mean kWh over finite panel households in each slot.
  contrast  D_t = log mean_ToU_t - log mean_Std_t (multiplicative difference-in-differences).
  event day any slot of the day has a High or Low signal. reference days: the K=10 nearest days (2013)
            whose 48 slots are all Normal and that have the same day type (Mon-Fri vs Sat/Sun).
  baseline  b_s = median of D over the reference days, per slot s.
  effect    counterfactual ToU load under all-Normal prices  cf_{d,s} = Std_{d,s} * exp(b_s);
            e_{d,s} = ToU_{d,s} - cf_{d,s} (kWh, all 48 slots of the day, so shifting is included).
  placebo   the same estimator on every all-Normal day (excluded from its own reference set) gives
            the null distribution: what e looks like when no price signal was sent.
  2012 test the 2013 schedule moved back 364 days (same weekdays) onto 2012, where nobody received
            price signals; estimated "effects" there should be centred on zero.

Reported: per event run (contiguous High or Low slots) the mean relative effect over its slots with a
placebo z-score; per event day the share of ||e||^2 that is signal (1 - placebo / event); coverage and
balance of the two groups.

Usage:
  python scripts/lcl_feasibility.py --data data/lcl/processed      # writes $RESULTS_DIR/lcl/feasibility
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from covattn.io_utils import results_root

K_REF = 10
MIN_COVERAGE = 0.90
SPD = 48  # slots per day


def group_means(arr: np.ndarray, rows: np.ndarray, block: int = 400) -> np.ndarray:
    s = np.zeros(arr.shape[1])
    n = np.zeros(arr.shape[1])
    for a in range(0, len(rows), block):
        x = np.asarray(arr[rows[a:a + block]], dtype=np.float64)
        f = np.isfinite(x)
        s += np.where(f, x, 0).sum(0)
        n += f.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / n, n


def daily(x: np.ndarray, start_slot: int, n_days: int) -> np.ndarray:
    return x[start_slot:start_slot + n_days * SPD].reshape(n_days, SPD)


def effects(tou: np.ndarray, std: np.ndarray, days: pd.DatetimeIndex, is_event: np.ndarray,
            masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """tou/std: (n_days, 48). Returns cf (n_days, 48), e (n_days, 48), n_ref per day."""
    D = np.log(tou) - np.log(std)
    weekend = days.day_of_week >= 5
    normal = ~is_event & np.isfinite(D).all(1)
    cf = np.full_like(tou, np.nan)
    nref = np.zeros(len(days), int)
    idx = np.arange(len(days))
    for d in range(len(days)):
        cand = idx[normal & (weekend == weekend[d]) & (idx != d)]
        if len(cand) == 0:
            continue
        ref = cand[np.argsort(np.abs(cand - d), kind="stable")[:K_REF]]
        nref[d] = len(ref)
        b = np.median(D[ref], axis=0)
        cf[d] = std[d] * np.exp(b)
    return cf, tou - cf, nref


def runs_of(tariff: np.ndarray) -> list[tuple[int, int, str]]:
    out, s = [], None
    for i, v in enumerate(list(tariff) + ["Normal"]):
        if s is not None and v != tariff[s]:
            out.append((s, i, tariff[s]))
            s = None
        if s is None and i < len(tariff) and v != "Normal":
            s = i
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lcl/processed")
    args = ap.parse_args()
    data = Path(args.data)
    out = results_root() / "lcl" / "feasibility"
    out.mkdir(parents=True, exist_ok=True)

    hh = pd.read_parquet(data / "households.parquet")
    arr = np.load(data / "hh_kwh.npy", mmap_mode="r")
    tar = pd.read_parquet(data / "tariff.parquet")
    t = pd.DatetimeIndex(tar["t"])
    s2013 = int(np.searchsorted(t, pd.Timestamp("2013-01-01")))
    s2012 = int(np.searchsorted(t, pd.Timestamp("2012-01-02")))  # 2013-01-01 minus 364 days

    # panel by 2013 coverage
    cov = np.zeros(len(hh))
    cov12 = np.zeros(len(hh))
    for a in range(0, len(hh), 400):
        x = np.asarray(arr[a:a + 400])
        cov[a:a + 400] = np.isfinite(x[:, s2013:s2013 + 365 * SPD]).mean(1)
        cov12[a:a + 400] = np.isfinite(x[:, s2012:s2012 + 365 * SPD]).mean(1)
    hh["cov2013"], hh["cov2012"] = cov, cov12
    report: dict = {"design": dict(K_REF=K_REF, MIN_COVERAGE=MIN_COVERAGE, contrast="log ratio ToU/Std",
                                   baseline="median over nearest all-Normal same-daytype days")}
    report["households"] = hh.groupby("group").agg(n=("id", "size"), cov2013_median=("cov2013", "median"),
                                                   n_panel=("cov2013", lambda c: int((c >= MIN_COVERAGE).sum())),
                                                   cov2012_median=("cov2012", "median")).to_dict("index")
    report["acorn_grouped_share_panel"] = (hh[hh.cov2013 >= MIN_COVERAGE].groupby("group")["acorn_grouped"]
                                           .value_counts(normalize=True).round(3).unstack(0).fillna(0).to_dict())

    means = {}
    for g in ("ToU", "Std"):
        rows = np.flatnonzero((hh["group"] == g).to_numpy() & (cov >= MIN_COVERAGE))
        if len(rows) == 0:
            raise SystemExit(f"no {g} households with >= {MIN_COVERAGE:.0%} coverage in 2013; "
                             f"groups present: {hh['group'].value_counts().to_dict()}")
        means[g], cnt = group_means(arr, rows)
        report.setdefault("panel_active_min_2013", {})[g] = int(cnt[s2013:s2013 + 365 * SPD].min())

    tariff13 = tar["tariff"].to_numpy()[s2013:s2013 + 365 * SPD]
    days13 = pd.date_range("2013-01-01", periods=365, freq="D")
    tday = tariff13.reshape(365, SPD)
    is_event = (tday != "Normal").any(1)
    masks = {lv: (tday == lv) for lv in ("High", "Low")}

    res = {}
    for label, start in (("2013", s2013), ("2012_placebo_schedule", s2012)):
        tou, std = daily(means["ToU"], start, 365), daily(means["Std"], start, 365)
        days = days13 if label == "2013" else days13 - pd.Timedelta(days=364)
        cf, e, nref = effects(tou, std, days, is_event, masks)
        res[label] = dict(tou=tou, std=std, cf=cf, e=e, nref=nref)

    e13, cf13, tou13 = res["2013"]["e"], res["2013"]["cf"], res["2013"]["tou"]
    normal_days = np.flatnonzero(~is_event & np.isfinite(e13).all(1))
    weekend = days13.day_of_week >= 5
    run_rows = []
    for a, b, lv in runs_of(tariff13):
        d0, s0 = divmod(a, SPD)
        sl = np.arange(a, b) - d0 * SPD  # slots relative to the start day (may exceed 47 if run crosses midnight)
        flat_e, flat_cf = e13.reshape(-1), cf13.reshape(-1)
        rel = np.nanmean(flat_e[a:b]) / np.nanmean(flat_cf[a:b])
        # placebo: same clock slots on all-Normal days of the same day type
        null = []
        for d in normal_days[weekend[normal_days] == weekend[d0]]:
            lo, hi = d * SPD + sl[0], d * SPD + sl[-1] + 1
            if hi <= len(flat_e):
                null.append(np.nanmean(flat_e[lo:hi]) / np.nanmean(flat_cf[lo:hi]))
        null = np.asarray(null)
        null = null[np.isfinite(null)]
        run_rows.append(dict(level=lv, start=str(days13[0] + pd.Timedelta(minutes=30 * a)), n_slots=b - a,
                             rel_effect=rel, null_sd=null.std(ddof=1), z=(rel - null.mean()) / null.std(ddof=1),
                             n_null=len(null)))
    runs = pd.DataFrame(run_rows)
    runs.to_csv(out / "event_runs.csv", index=False)

    def summ(df: pd.DataFrame) -> dict:
        return dict(n=len(df), rel_effect_median=float(df.rel_effect.median()), rel_effect_mean=float(df.rel_effect.mean()),
                    null_sd_median=float(df.null_sd.median()), z_median=float(df.z.median()),
                    share_abs_z_gt2=float((df.z.abs() > 2).mean()),
                    share_expected_sign=float(((df.rel_effect < 0) if df.level.iloc[0] == "High" else (df.rel_effect > 0)).mean()),
                    pooled_mean_se=float(df.null_sd.median() / np.sqrt(len(df))))
    report["event_runs"] = {lv: summ(runs[runs.level == lv]) for lv in ("High", "Low")}

    # day-profile signal share (what the effect gain g will be computed on)
    ev_days = np.flatnonzero(is_event & np.isfinite(e13).all(1))
    sq_ev = (e13[ev_days] ** 2).sum(1)
    sq_null = (e13[normal_days] ** 2).sum(1)
    report["day_profile"] = dict(n_event_days=len(ev_days), n_normal_days=len(normal_days),
                                 mean_sq_event=float(sq_ev.mean()), mean_sq_placebo=float(sq_null.mean()),
                                 signal_share=float(1 - sq_null.mean() / sq_ev.mean()),
                                 tou_mean_kwh_per_slot=float(np.nanmean(tou13)))
    for lv in ("High", "Low"):
        m = masks[lv][ev_days]
        report["day_profile"][f"mean_abs_e_in_{lv}_slots_kwh"] = float(np.nanmean(np.abs(e13[ev_days][m])))

    # 2012: nobody had price signals; the 2013 schedule mapped onto 2012 should show no effect
    e12, cf12 = res["2012_placebo_schedule"]["e"], res["2012_placebo_schedule"]["cf"]
    fake = []
    for a, b, lv in runs_of(tariff13):
        fe, fc = e12.reshape(-1)[a:b], cf12.reshape(-1)[a:b]
        if np.isfinite(fe).all():
            fake.append(dict(level=lv, rel=np.mean(fe) / np.mean(fc)))
    fake = pd.DataFrame(fake)
    report["placebo_2012_schedule"] = ({lv: dict(n=int((fake.level == lv).sum()),
                                                 rel_effect_mean=float(fake[fake.level == lv].rel.mean()),
                                                 rel_effect_median=float(fake[fake.level == lv].rel.median()))
                                        for lv in ("High", "Low")} if len(fake) else "no 2012 data")

    series = pd.DataFrame({"t": t[s2013:s2013 + 365 * SPD], "tou_kwh": tou13.reshape(-1),
                           "std_kwh": res["2013"]["std"].reshape(-1), "cf_kwh": cf13.reshape(-1),
                           "effect_kwh": e13.reshape(-1), "tariff": tariff13,
                           "event_day": np.repeat(is_event, SPD)})
    series.to_parquet(out / "series_2013.parquet", index=False)
    ctx = pd.DataFrame({"t": t, "tou_kwh": means["ToU"], "std_kwh": means["Std"], "tariff": tar["tariff"],
                        "price": tar["price"]})
    ctx.to_parquet(out / "group_series_full.parquet", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
