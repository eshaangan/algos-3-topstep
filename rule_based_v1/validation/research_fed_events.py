"""Fed-calendar events in the fomc_drift_v1 window: does the family extend?

fomc_drift_v1 (long from 14:00 ET the day before a decision to 13:55 ET on the
decision day) is the only strategy here that passed dev AND a sealed holdout. It
is limited to 8 events/yr, and that limit -- not its size -- is what sets the
portfolio's ~15-week median time-to-pass.

So the question worth asking is whether the SAME window works on other scheduled
Fed releases. This runs it on:

  decision  -- the incumbent, as a positive control that must reproduce
  minutes   -- released exactly three weeks after each decision at 14:00 ET

The ledger's refined mechanism claim is that the drift attaches to POLICY
DECISIONS, not to data releases. Minutes are neither: they are scheduled policy
COMMUNICATION with no new decision attached. That makes this a genuine test of
where the boundary sits, with a real prior in both directions, rather than a
fishing expedition -- and it would double Fed velocity if it holds.

Usage:
    python3 rule_based_v1/validation/research_fed_events.py --event minutes --out runs/fed_minutes.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ET = "America/New_York"

INSTRUMENTS = {
    "mnq": {"path": "data/processed/mnq_1m_all.parquet", "ts": "ts",
            "tz": "America/Chicago", "pv": 2.0, "tick": 0.25},
    "es": {"path": "data/processed/es_1min_eth_frontmonth.parquet", "ts": "et",
           "tz": None, "pv": 5.0, "tick": 0.25},
}


def load_px(cfg: dict) -> pd.Series:
    raw = pd.read_parquet(cfg["path"])
    idx = pd.to_datetime(raw[cfg["ts"]])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(cfg["tz"], ambiguous="NaT", nonexistent="NaT")
    s = pd.Series(raw["close"].to_numpy(float), index=idx.dt.tz_convert(ET))
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


def rth_sessions(px: pd.Series) -> pd.DatetimeIndex:
    mins = px.index.hour * 60 + px.index.minute
    rth = px[(mins >= 9 * 60 + 30) & (mins <= 16 * 60)]
    g = rth.groupby(rth.index.normalize()).size()
    return pd.DatetimeIndex(g[g >= 60].index)


def minutes_dates(decisions: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Three weeks after each decision, rolled forward to the next session if needed.

    Verified against federalreserve.gov: the 2024-01-31 decision's minutes were
    released 2024-02-21, and 2024-03-20 -> 2024-04-10, both exactly 21 days.
    """
    out = []
    for d in decisions:
        cand = d + pd.Timedelta(days=21)
        pos = sessions.searchsorted(cand, "left")
        if pos < len(sessions) and (sessions[pos] - cand).days <= 4:
            out.append(sessions[pos])
    return pd.DatetimeIndex(sorted(set(out)))


def drift_trades(px: pd.Series, sessions: pd.DatetimeIndex, days: pd.DatetimeIndex,
                 release_h: float, pv: float, contracts: int, cost: float) -> pd.DataFrame:
    """Long from release_time on the prior session to 5 minutes before release."""
    rows = []
    for d in days:
        pos = sessions.searchsorted(d, "left")
        if pos == 0 or pos >= len(sessions) or sessions[pos] != d:
            continue
        t_in = sessions[pos - 1] + pd.Timedelta(hours=release_h)
        t_out = d + pd.Timedelta(hours=release_h) - pd.Timedelta(minutes=5)
        a = px[t_in - pd.Timedelta(minutes=20):t_in + pd.Timedelta(minutes=10)]
        b = px[t_out - pd.Timedelta(minutes=30):t_out]
        if a.empty or b.empty or b.index[-1] <= a.index[-1]:
            continue
        entry, exit_px = float(a.iloc[-1]), float(b.iloc[-1])
        path = px[a.index[-1]:b.index[-1]]
        rows.append({"day": d, "entry": entry, "points": exit_px - entry,
                     "pnl": (exit_px - entry) * pv * contracts - cost,
                     "mae": float(path.min() - entry) * pv * contracts})
    return pd.DataFrame(rows)


def stat(tr: pd.DataFrame, label: str) -> dict:
    if len(tr) < 8:
        return {"label": label, "n": int(len(tr))}
    p = tr["pnl"]
    t = stats.ttest_1samp(p, 0)
    q05 = tr["mae"].quantile(0.05)
    return {"label": label, "n": int(len(p)), "mean": float(p.mean()),
            "bps": float((tr["points"] / tr["entry"] * 1e4).mean()),
            "t_stat": float(t.statistic), "p_value": float(t.pvalue),
            "win_rate": float((p > 0).mean()),
            "mae_p50": float(tr["mae"].median()), "mae_p05": float(q05),
            "mae_worst": float(tr["mae"].min()),
            "eff": float(p.mean() / abs(q05)) if q05 < 0 else float("nan")}


def show(s: dict) -> None:
    if s.get("n", 0) < 8:
        print(f"  {s['label']:<26} n={s.get('n', 0)} (too few)")
        return
    print(f"  {s['label']:<26} n={s['n']:<5} ${s['mean']:+8.2f}  {s['bps']:+6.1f}bps  "
          f"t={s['t_stat']:+5.2f}  WR={s['win_rate']:5.1%}  "
          f"MAE p50/p05 ${s['mae_p50']:>6,.0f}/${s['mae_p05']:>7,.0f}  eff={s['eff']:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fed events in the fomc_drift window")
    ap.add_argument("--instrument", choices=sorted(INSTRUMENTS), default="mnq")
    ap.add_argument("--calendar", type=Path, default=Path("data/processed/fomc_announcements.csv"))
    ap.add_argument("--event", choices=("decision", "minutes", "both"), default="both")
    ap.add_argument("--release-hour", type=float, default=14.0)
    ap.add_argument("--contracts", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = INSTRUMENTS[args.instrument]
    cost = 2 * args.contracts * (0.62 + 1.0 * cfg["tick"] * cfg["pv"])
    px = load_px(cfg)
    sess = rth_sessions(px)
    dec_all = pd.DatetimeIndex(pd.read_csv(args.calendar)["announcement_date"]).tz_localize(ET)
    dec = dec_all.intersection(sess)
    mins = minutes_dates(dec_all, sess)
    mins = pd.DatetimeIndex([d for d in mins if d not in set(dec)])

    print(f"{args.instrument.upper()}  window: prior session {args.release_hour:g}:00 ET -> "
          f"{args.release_hour:g}:00 minus 5min   {args.contracts} contracts  cost ${cost:.2f}")
    print(f"  sessions {len(sess)}  {sess.min().date()} .. {sess.max().date()}")
    print(f"  decisions in range {len(dec)}   minutes in range {len(mins)}\n")

    ctrl = sess.difference(dec).difference(mins)
    sets = {"decision (incumbent)": dec, "minutes": mins, "control": ctrl}
    report = {"instrument": args.instrument, "rows": []}
    books = {}
    print("EVENT SETS")
    for name, days in sets.items():
        tr = drift_trades(px, sess, days, args.release_hour, cfg["pv"], args.contracts, cost)
        books[name] = tr
        s = stat(tr, name)
        report["rows"].append(s)
        show(s)

    print()
    for name in ("decision (incumbent)", "minutes"):
        a, b = books[name], books["control"]
        if len(a) >= 8 and len(b) >= 8:
            w = stats.ttest_ind(a["pnl"], b["pnl"], equal_var=False)
            print(f"  {name} minus control: ${a['pnl'].mean() - b['pnl'].mean():+8.2f}  "
                  f"t={w.statistic:+5.2f}  p={w.pvalue:.3f}")

    print("\nERA SPLIT (dev 2020-2022 / val 2023-2026)")
    cut = pd.Timestamp("2022-12-31", tz=ET)
    for name in ("decision (incumbent)", "minutes"):
        tr = books[name]
        for era, sub in (("dev", tr[tr["day"] <= cut]), ("val", tr[tr["day"] > cut])):
            s = stat(sub, f"{name} {era}")
            report["rows"].append(s)
            show(s)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
