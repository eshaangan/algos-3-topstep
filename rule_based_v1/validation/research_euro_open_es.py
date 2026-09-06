"""euro_open_v1 on the 15.5-year ES ETH tape.

The watchlist entry (2026-07-11) is: long MNQ 02:00->05:00 ET when the prior RTH
day closed down AND its range exceeded the 20-day median. On MNQ 2020-2026 that
was +$26/night per 2 micros at t=1.84 -- a clean mechanism gradient but below the
GO bar, held pending more evidence.

This is that evidence. ES ETH now reaches back to 2010-06, which is 2.4x the MNQ
sample and, crucially, includes the decade before the effect was ever looked at.

Reported as a gradient, because that is how the effect was characterised and a
real conditional edge should strengthen monotonically along it:
    unconditional -> prior-day-down -> prior-day-down AND high-range
with prior-day-UP as the control that must stay flat.

Usage:
    python3 rule_based_v1/validation/research_euro_open_es.py --out runs/euro_open_es.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ET = "America/New_York"
RTH_START, RTH_END = 9 * 60 + 30, 16 * 60


def load_px(path: Path, ts_col: str, tz: str | None) -> pd.Series:
    raw = pd.read_parquet(path)
    idx = pd.to_datetime(raw[ts_col])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    s = pd.Series(raw["close"].to_numpy(float), index=idx.dt.tz_convert(ET))
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


def rth_day_stats(px: pd.Series) -> pd.DataFrame:
    mins = px.index.hour * 60 + px.index.minute
    rth = px[(mins >= RTH_START) & (mins < RTH_END)]
    g = rth.groupby(rth.index.normalize())
    day = pd.DataFrame({"first": g.first(), "last": g.last(),
                        "hi": g.max(), "lo": g.min(), "n": g.size()})
    day = day[day["n"] >= 60]
    day["ret"] = day["last"] - day["first"]
    day["range"] = day["hi"] - day["lo"]
    # 20-day median of prior ranges, strictly backward-looking
    day["range_med20"] = day["range"].shift(1).rolling(20, min_periods=20).median()
    day["down"] = day["ret"] < 0
    day["hirange"] = day["range"] > day["range_med20"]
    return day


def session_holds(px: pd.Series, days: pd.DatetimeIndex, start_h: float, end_h: float,
                  point_value: float, contracts: int, cost: float) -> pd.DataFrame:
    rows = []
    for d in days:
        t0 = d + pd.Timedelta(hours=start_h)
        t1 = d + pd.Timedelta(hours=end_h)
        seg_in = px[t0:t0 + pd.Timedelta(minutes=10)]
        seg_out = px[t1 - pd.Timedelta(minutes=30):t1]
        if seg_in.empty or seg_out.empty:
            continue
        entry, exit_px = float(seg_in.iloc[0]), float(seg_out.iloc[-1])
        path = px[seg_in.index[0]:seg_out.index[-1]]
        if len(path) < 10:
            continue
        rows.append({"day": d, "entry": entry, "exit": exit_px,
                     "points": exit_px - entry,
                     "pnl": (exit_px - entry) * point_value * contracts - cost,
                     "mae": float(path.min() - entry) * point_value * contracts})
    return pd.DataFrame(rows)


def stat(tr: pd.DataFrame, label: str) -> dict:
    if len(tr) < 10:
        return {"label": label, "n": int(len(tr))}
    p = tr["pnl"]
    t = stats.ttest_1samp(p, 0)
    yrs = tr.assign(y=tr["day"].dt.year).groupby("y")["pnl"].mean()
    return {"label": label, "n": int(len(p)), "mean": float(p.mean()),
            "total": float(p.sum()), "t_stat": float(t.statistic),
            "p_value": float(t.pvalue), "win_rate": float((p > 0).mean()),
            "pos_years": float((yrs > 0).mean()), "n_years": int(len(yrs)),
            "mae_p50": float(tr["mae"].median()), "mae_p05": float(tr["mae"].quantile(0.05)),
            "mae_worst": float(tr["mae"].min())}


def show(s: dict) -> None:
    if s.get("n", 0) < 10:
        print(f"  {s['label']:<34} n={s.get('n', 0):<4} (too few)")
        return
    print(f"  {s['label']:<34} n={s['n']:<5} ${s['mean']:+7.2f}  t={s['t_stat']:+5.2f}  "
          f"WR={s['win_rate']:5.1%}  posYr={s['pos_years']:4.0%}  "
          f"MAE p05 ${s['mae_p05']:.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="euro_open_v1 on ES 15.5y ETH")
    ap.add_argument("--bars", type=Path, default=Path("data/processed/es_1min_eth_frontmonth.parquet"))
    ap.add_argument("--ts-col", default="et")
    ap.add_argument("--tz", default=None)
    ap.add_argument("--point-value", type=float, default=5.0)
    ap.add_argument("--contracts", type=int, default=2)
    ap.add_argument("--tick-size", type=float, default=0.25)
    ap.add_argument("--start-hour", type=float, default=2.0)
    ap.add_argument("--end-hour", type=float, default=5.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cost = 2 * args.contracts * (0.62 + 1.0 * args.tick_size * args.point_value)
    px = load_px(args.bars, args.ts_col, args.tz)
    day = rth_day_stats(px)
    print(f"bars {len(px):,}  {px.index.min().date()} .. {px.index.max().date()}   "
          f"RTH days {len(day)}   cost ${cost:.2f}/round-turn  {args.contracts} contracts")
    print(f"window {args.start_hour:g}:00 -> {args.end_hour:g}:00 ET, weekdays\n")

    # the hold happens on day D, conditioned on the RTH session of D-1
    dd = day.dropna(subset=["range_med20"]).copy()
    nxt = pd.Series(day.index, index=day.index).shift(-1)
    dd["hold_day"] = nxt.reindex(dd.index)
    dd = dd.dropna(subset=["hold_day"])
    dd = dd[pd.DatetimeIndex(dd["hold_day"]).dayofweek < 5]

    all_holds = session_holds(px, pd.DatetimeIndex(dd["hold_day"]), args.start_hour,
                              args.end_hour, args.point_value, args.contracts, cost)
    cond = dd.set_index(pd.DatetimeIndex(dd["hold_day"]))[["down", "hirange"]]
    tr = all_holds.set_index("day").join(cond).reset_index().rename(columns={"index": "day"})

    groups = {
        "unconditional": tr,
        "prior day DOWN": tr[tr["down"]],
        "prior DOWN + hi-range (SPEC)": tr[tr["down"] & tr["hirange"]],
        "prior day UP (control)": tr[~tr["down"]],
        "prior UP + hi-range (control)": tr[(~tr["down"]) & tr["hirange"]],
    }
    report = {"window": [args.start_hour, args.end_hour], "cost": cost, "eras": {}}

    for era, sub in (("FULL 2010-2025", tr),
                     ("pre-2020", tr[tr["day"] < pd.Timestamp("2020-01-01", tz=ET)]),
                     ("2020+", tr[tr["day"] >= pd.Timestamp("2020-01-01", tz=ET)])):
        print(f"--- {era} ---")
        rows = []
        for name, g in groups.items():
            gg = g[g["day"].isin(sub["day"])]
            s = stat(gg, name)
            rows.append(s)
            show(s)
        report["eras"][era] = rows
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
