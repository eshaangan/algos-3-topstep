"""FOMC-cycle even-week effect -- Cieslak, Morse & Vissing-Jorgensen (2019, JF),
and the Uppal (2025) out-of-sample challenge to it.

CMVJ: since 1994 the entire US equity premium is earned in weeks 0, 2, 4 and 6 of
FOMC cycle time. Uppal updates their data to 2023 and finds the result gone,
weakening from 2004, with the pre-2004 version driven by a few outlier days.

The distinction that matters here is that CMVJ's week 0 spans days -1..+3, so it
CONTAINS the Lucca-Moench pre-FOMC drift -- which this project already trades as
fomc_drift_v1. CMVJ say so themselves. So the question is not "do even weeks pay"
but "do weeks 2, 4 and 6 pay ON TOP of the week-0 drift we already own". Only the
second is new velocity; the first would just be re-counting an existing edge.

Cycle time (CMVJ, reproduced in Uppal Table 1), counted in trading days from the
scheduled announcement (day 0):

    week -1 : days -6..-2      week 3 : days 14..18
    week  0 : days -1..3       week 4 : days 19..23
    week  1 : days 4..8        week 5 : days 24..28
    week  2 : days 9..13       week 6 : days 29..33

Usage:
    python3 rule_based_v1/validation/research_fomc_cycle_weeks.py \
        --instrument es --out runs/fomc_cycle_es.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ET = "America/New_York"
CALENDAR = "data/processed/fomc_announcements.csv"

# (week, first_day, last_day) in FOMC cycle time
WEEK_BOUNDS = [(-1, -6, -2), (0, -1, 3), (1, 4, 8), (2, 9, 13),
               (3, 14, 18), (4, 19, 23), (5, 24, 28), (6, 29, 33)]
EVEN = (0, 2, 4, 6)
ODD = (-1, 1, 3, 5)
EVEN_EX0 = (2, 4, 6)          # the only part that would be NEW velocity for us

INSTRUMENTS = {
    "es": {"path": "data/processed/es_bars_2010_2025.h5", "kind": "hdf",
           "hdf_key": "bars_5min", "timestamp_col": "timestamp", "tz": None,
           "point_value": 5.0, "tick_size": 0.25, "dev_end": "2017-12-31"},
    "mnq": {"path": "data/processed/mnq_1m_all.parquet", "kind": "parquet",
            "hdf_key": None, "timestamp_col": "ts", "tz": "America/Chicago",
            "point_value": 2.0, "tick_size": 0.25, "dev_end": "2023-12-31"},
}

COMMISSION_PER_SIDE = 0.62
SLIPPAGE_TICKS = 1.0


def load_daily(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["path"])
    raw = (pd.read_hdf(path, key=cfg["hdf_key"]) if cfg["kind"] == "hdf"
           else pd.read_parquet(path))
    idx = pd.to_datetime(raw[cfg["timestamp_col"]])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(cfg["tz"], ambiguous="NaT", nonexistent="NaT")
    bars = raw.drop(columns=[cfg["timestamp_col"]]).set_index(idx.dt.tz_convert(ET))
    bars = bars[~bars.index.isna()].sort_index()
    mins = bars.index.hour * 60 + bars.index.minute
    rth = bars[(mins >= 9 * 60 + 30) & (mins <= 15 * 60 + 59)]
    day = rth.groupby(rth.index.normalize()).agg(
        open_=("open", "first"), close=("close", "last"), n=("close", "size"))
    day = day[day["n"] >= 60]
    day["intraday"] = day["close"] - day["open_"]            # RTH open -> close
    day["c2c"] = day["close"].diff()                          # prior close -> close
    return day


def assign_cycle(day: pd.DataFrame, announcements: pd.DatetimeIndex) -> pd.DataFrame:
    """Trading days relative to the scheduled announcement, CMVJ convention."""
    sess = day.index
    ann_pos = np.array(sorted(sess.searchsorted(a) for a in announcements
                              if 0 <= sess.searchsorted(a) < len(sess)))
    # keep only announcements that land on (or map into) a real session
    ann_pos = np.unique(ann_pos)
    pos = np.arange(len(sess))
    prev_i = np.searchsorted(ann_pos, pos, side="right") - 1
    next_i = np.searchsorted(ann_pos, pos, side="left")
    since = np.where(prev_i >= 0, pos - ann_pos[np.clip(prev_i, 0, None)], 10_000)
    to = np.where(next_i < len(ann_pos), ann_pos[np.clip(next_i, None, len(ann_pos) - 1)] - pos, 10_000)
    # Days within 6 sessions of the NEXT meeting are counted backward from it
    # (that is what makes week 0 start on day -1); everything else counts forward.
    cycle_day = np.where(to <= 6, -to, np.where(since <= 33, since, 9_999))
    out = day.copy()
    out["cycle_day"] = cycle_day
    week = np.full(len(out), np.nan)
    for w, lo, hi in WEEK_BOUNDS:
        week[(cycle_day >= lo) & (cycle_day <= hi)] = w
    out["cycle_week"] = week
    return out


def describe(x: pd.Series, point_value: float, contracts: int, label: str) -> dict:
    x = x.dropna()
    if len(x) < 5:
        return {"label": label, "n": int(len(x))}
    t = stats.ttest_1samp(x, 0)
    return {"label": label, "n": int(len(x)),
            "mean_points": float(x.mean()),
            "mean_usd": float(x.mean() * point_value * contracts),
            "total_usd": float(x.sum() * point_value * contracts),
            "t_stat": float(t.statistic), "p_value": float(t.pvalue),
            "win_rate": float((x > 0).mean())}


def compare(a: pd.Series, b: pd.Series, label: str) -> dict:
    a, b = a.dropna(), b.dropna()
    t = stats.ttest_ind(a, b, equal_var=False)
    return {"label": label, "n_a": int(len(a)), "n_b": int(len(b)),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": float(a.mean() - b.mean()),
            "t_stat": float(t.statistic), "p_value": float(t.pvalue)}


def run_era(frame: pd.DataFrame, col: str, point_value: float, contracts: int,
            era: str) -> dict:
    print(f"\n--- {era}  (return measure: {col}) ---")
    print("  week      n   mean pts    $/day/2ct     t      WR")
    per_week = []
    for w, lo, hi in WEEK_BOUNDS:
        s = describe(frame.loc[frame["cycle_week"] == w, col], point_value, contracts, f"week{w}")
        s["week"] = w
        per_week.append(s)
        if s["n"] >= 5:
            print(f"  {w:>4}  {s['n']:5d}   {s['mean_points']:+8.2f}   {s['mean_usd']:+9.2f}  "
                  f"{s['t_stat']:+6.2f}  {s['win_rate']:5.1%}")
    even = frame.loc[frame["cycle_week"].isin(EVEN), col]
    odd = frame.loc[frame["cycle_week"].isin(ODD), col]
    ex0 = frame.loc[frame["cycle_week"].isin(EVEN_EX0), col]
    res = {
        "era": era, "measure": col, "per_week": per_week,
        "even": describe(even, point_value, contracts, "even(0,2,4,6)"),
        "odd": describe(odd, point_value, contracts, "odd(-1,1,3,5)"),
        "even_ex_week0": describe(ex0, point_value, contracts, "even(2,4,6)"),
        "even_vs_odd": compare(even, odd, "even vs odd"),
        "even_ex0_vs_odd": compare(ex0, odd, "even(2,4,6) vs odd"),
    }
    e, o, x = res["even"], res["odd"], res["even_ex_week0"]
    print(f"  EVEN (0,2,4,6)  n={e['n']:5d}  ${e['mean_usd']:+7.2f}/day  t={e['t_stat']:+5.2f}")
    print(f"  ODD (-1,1,3,5)  n={o['n']:5d}  ${o['mean_usd']:+7.2f}/day  t={o['t_stat']:+5.2f}")
    print(f"  EVEN minus week 0 (2,4,6)   n={x['n']:5d}  ${x['mean_usd']:+7.2f}/day  t={x['t_stat']:+5.2f}"
          f"    <-- the only NEW velocity")
    print(f"  even vs odd     diff={res['even_vs_odd']['diff']:+.3f} pts  t={res['even_vs_odd']['t_stat']:+5.2f}  p={res['even_vs_odd']['p_value']:.3f}")
    print(f"  (2,4,6) vs odd  diff={res['even_ex0_vs_odd']['diff']:+.3f} pts  t={res['even_ex0_vs_odd']['t_stat']:+5.2f}  p={res['even_ex0_vs_odd']['p_value']:.3f}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="CMVJ FOMC-cycle even-week test")
    ap.add_argument("--instrument", choices=sorted(INSTRUMENTS), required=True)
    ap.add_argument("--calendar", type=Path, default=Path(CALENDAR))
    ap.add_argument("--contracts", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = INSTRUMENTS[args.instrument]
    day = load_daily(cfg)
    ann = pd.DatetimeIndex(pd.read_csv(args.calendar)["announcement_date"]).tz_localize(ET)
    frame = assign_cycle(day, ann)

    covered = frame["cycle_week"].notna().mean()
    print(f"{args.instrument.upper()}  sessions={len(frame)}  "
          f"{frame.index.min().date()} .. {frame.index.max().date()}")
    print(f"announcements in range: {int(((ann >= frame.index.min()) & (ann <= frame.index.max())).sum())}"
          f"   sessions mapped to a cycle week: {covered:.1%}")

    report = {"instrument": args.instrument, "sessions": int(len(frame)),
              "coverage": float(covered), "results": []}
    dev_end = pd.Timestamp(cfg["dev_end"], tz=ET)
    eras = [("FULL", frame),
            (f"DEV .. {cfg['dev_end']}", frame[frame.index <= dev_end]),
            (f"VAL {cfg['dev_end']} ..", frame[frame.index > dev_end])]
    for col in ("c2c", "intraday"):
        print(f"\n{'=' * 78}\n{col.upper()}  "
              f"({'prior close -> close, CMVJ-equivalent' if col == 'c2c' else 'RTH open -> close, no overnight risk'})\n{'=' * 78}")
        for era, sub in eras:
            report["results"].append(run_era(sub, col, cfg["point_value"], args.contracts, era))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
