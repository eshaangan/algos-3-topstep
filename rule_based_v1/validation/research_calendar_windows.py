"""Calendar-window holds on the long ETH tape: expiration week and month turn.

The only things that have ever worked in this project are long holds over
concentrated calendar windows (weekend, Monday RTH, pre-FOMC). This tests two
more windows of that shape which have a mechanism behind them and have not been
tested here:

  expiry_week  -- Monday through the third Friday. Stivers & Sun (2011) find that
                  expiration-week returns on stocks with large option open
                  interest show up in the index itself. Distinct from the
                  expiration-DAY pinning already killed: that was a last-hours
                  strike magnet, this is a week-long directional premium from
                  hedging demand unwinding.
  month_turn   -- last N business days plus first M of the next month. The plain
                  turn-of-month is already dead here on MES daily data; this is
                  kept only as a same-shape reference so the expiry-week number
                  has a calendar-window peer to be judged against, NOT as a new
                  claim.

Holds run from the first session's 09:30 ET to the last session's 16:00 ET, so a
week-long window costs one round turn, not five.

Usage:
    python3 rule_based_v1/validation/research_calendar_windows.py --out runs/calwin_es.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ET = "America/New_York"


def load_px(path: Path, ts_col: str, tz: str | None) -> pd.Series:
    raw = pd.read_parquet(path)
    idx = pd.to_datetime(raw[ts_col])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    s = pd.Series(raw["close"].to_numpy(float), index=idx.dt.tz_convert(ET))
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


def rth_sessions(px: pd.Series) -> pd.DatetimeIndex:
    mins = px.index.hour * 60 + px.index.minute
    rth = px[(mins >= 9 * 60 + 30) & (mins <= 16 * 60)]
    g = rth.groupby(rth.index.normalize()).size()
    return pd.DatetimeIndex(g[g >= 60].index)


def hold(px: pd.Series, d0: pd.Timestamp, d1: pd.Timestamp) -> tuple[float, float, float] | None:
    t0 = d0 + pd.Timedelta(hours=9, minutes=30)
    t1 = d1 + pd.Timedelta(hours=16)
    a = px[t0:t0 + pd.Timedelta(minutes=10)]
    b = px[t1 - pd.Timedelta(minutes=30):t1]
    if a.empty or b.empty:
        return None
    entry, exit_px = float(a.iloc[0]), float(b.iloc[-1])
    path = px[a.index[0]:b.index[-1]]
    return entry, exit_px, float(path.min() - entry)


def expiry_weeks(sessions: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Monday..third Friday of each month, clipped to real sessions."""
    out = []
    for (y, m), grp in pd.Series(sessions, index=sessions).groupby([sessions.year, sessions.month]):
        fri = [d for d in pd.date_range(f"{y}-{m:02d}-01", periods=31, freq="D")
               if d.month == m and d.dayofweek == 4]
        if len(fri) < 3:
            continue
        third = fri[2].tz_localize(ET)
        week = grp[(grp >= third - pd.Timedelta(days=4)) & (grp <= third)]
        if len(week) >= 3:
            out.append((week.iloc[0], week.iloc[-1]))
    return out


def month_turns(sessions: pd.DatetimeIndex, last_n: int, first_m: int) -> list:
    out = []
    by_month = pd.Series(sessions, index=sessions).groupby([sessions.year, sessions.month])
    keys = sorted(by_month.groups)
    for i in range(len(keys) - 1):
        a = by_month.get_group(keys[i])
        b = by_month.get_group(keys[i + 1])
        if len(a) < last_n or len(b) < first_m:
            continue
        out.append((a.iloc[-last_n], b.iloc[first_m - 1]))
    return out


def run(px: pd.Series, windows: list, point_value: float, contracts: int,
        cost: float, label: str, split: pd.Timestamp) -> dict:
    rows = []
    for d0, d1 in windows:
        r = hold(px, d0, d1)
        if r is None:
            continue
        entry, exit_px, mae = r
        rows.append({"start": d0, "end": d1, "points": exit_px - entry,
                     "pnl": (exit_px - entry) * point_value * contracts - cost,
                     "mae": mae * point_value * contracts})
    tr = pd.DataFrame(rows)
    if len(tr) < 20:
        print(f"  {label:<26} n={len(tr)} (too few)")
        return {"label": label, "n": int(len(tr))}

    def bit(x: pd.DataFrame) -> tuple[float, float]:
        return float(x["pnl"].mean()), float(stats.ttest_1samp(x["pnl"], 0).statistic)

    m, t = bit(tr)
    a, b = tr[tr["start"] < split], tr[tr["start"] >= split]
    ma, ta = bit(a) if len(a) >= 20 else (float("nan"), float("nan"))
    mb, tb = bit(b) if len(b) >= 20 else (float("nan"), float("nan"))
    yrs = tr.assign(y=tr["start"].dt.year).groupby("y")["pnl"].mean()
    print(f"  {label:<26} n={len(tr):<5} ${m:+8.2f}  t={t:+5.2f}   "
          f"half1 ${ma:+7.2f}/t={ta:+5.2f}  half2 ${mb:+7.2f}/t={tb:+5.2f}  "
          f"posYr={100 * (yrs > 0).mean():3.0f}%  MAEp05 ${tr['mae'].quantile(0.05):.0f}")
    return {"label": label, "n": int(len(tr)), "mean": m, "t_stat": t,
            "mean_a": ma, "t_a": ta, "mean_b": mb, "t_b": tb,
            "pos_years": float((yrs > 0).mean()),
            "mae_p05": float(tr["mae"].quantile(0.05)),
            "mae_worst": float(tr["mae"].min())}


def main() -> None:
    ap = argparse.ArgumentParser(description="calendar-window holds")
    ap.add_argument("--bars", type=Path, default=Path("data/processed/es_1min_eth_frontmonth.parquet"))
    ap.add_argument("--ts-col", default="et")
    ap.add_argument("--tz", default=None)
    ap.add_argument("--point-value", type=float, default=5.0)
    ap.add_argument("--contracts", type=int, default=2)
    ap.add_argument("--tick-size", type=float, default=0.25)
    ap.add_argument("--split", default="2018-01-01")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cost = 2 * args.contracts * (0.62 + 1.0 * args.tick_size * args.point_value)
    px = load_px(args.bars, args.ts_col, args.tz)
    sess = rth_sessions(px)
    split = pd.Timestamp(args.split, tz=ET)
    print(f"sessions {len(sess)}  {sess.min().date()} .. {sess.max().date()}  "
          f"cost ${cost:.2f}  {args.contracts} contracts\n")

    out = []
    print("EXPIRATION WEEK (Mon -> third Friday, one round turn)")
    out.append(run(px, expiry_weeks(sess), args.point_value, args.contracts, cost,
                   "expiry_week", split))
    # same-length non-expiry weeks as the control this needs
    all_weeks = []
    wk = pd.Series(sess, index=sess).groupby([sess.isocalendar().year, sess.isocalendar().week])
    exp_starts = {d0 for d0, _ in expiry_weeks(sess)}
    for _, grp in wk:
        if len(grp) >= 3 and grp.iloc[0] not in exp_starts:
            all_weeks.append((grp.iloc[0], grp.iloc[-1]))
    out.append(run(px, all_weeks, args.point_value, args.contracts, cost,
                   "non-expiry week (control)", split))

    print("\nMONTH TURN (reference shape only, plain ToM already dead here)")
    for ln, fm in ((4, 3), (3, 3), (1, 3)):
        out.append(run(px, month_turns(sess, ln, fm), args.point_value, args.contracts,
                       cost, f"month_turn last{ln}+first{fm}", split))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
