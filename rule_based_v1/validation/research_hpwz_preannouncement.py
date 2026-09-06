"""HPWZ pre-announcement drift on ISM and GDP releases.

Hu, Pan, Wang & Zhu (NBER w25817): large overnight returns, with no abnormal
variance, ahead of NFP, ISM and GDP -- the same shape as the pre-FOMC drift this
project already trades. Their window is the previous trading day's 16:00 close to
five minutes before the release.

The control is the point of this script. The ledger already establishes that
overnight drift in equity index futures is real but not harvestable, so a positive
ISM overnight return proves nothing by itself. Every event window is therefore
reported against the SAME CLOCK WINDOW on non-event days.

Pre-registration: rule_based_v1/validation/vps_prereg/vps_hpwz_preannouncement.yaml

Usage:
    python3 rule_based_v1/validation/research_hpwz_preannouncement.py \
        --instrument es --event ism --out runs/hpwz_ism_es.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy import stats

ET = "America/New_York"

EVENTS = {
    "ism": {"release_et": "10:00", "label": "ISM Manufacturing PMI"},
    "gdp": {"release_et": "08:30", "label": "BEA GDP advance estimate"},
}

INSTRUMENTS = {
    "es": {"path": "data/processed/es_1min_eth_frontmonth.parquet", "ts": "et",
           "point_value": 5.0, "tick_size": 0.25, "dev_end": "2017-12-31"},
    "mnq": {"path": "data/processed/mnq_1m_all.parquet", "ts": "ts",
            "tz": "America/Chicago",
            "point_value": 2.0, "tick_size": 0.25, "dev_end": "2023-12-31"},
}

COMMISSION_PER_SIDE = 0.62
SLIPPAGE_TICKS = 1.0


def round_turn_cost(point_value: float, tick_size: float, contracts: int) -> float:
    return 2.0 * contracts * (COMMISSION_PER_SIDE + SLIPPAGE_TICKS * tick_size * point_value)


def load_bars(cfg: dict) -> pd.Series:
    raw = pd.read_parquet(cfg["path"])
    idx = pd.to_datetime(raw[cfg["ts"]])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(cfg["tz"], ambiguous="NaT", nonexistent="NaT")
    s = pd.Series(raw["close"].to_numpy(), index=idx.dt.tz_convert(ET))
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


def ism_release_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """First US business day of each month. ISM releases at 10:00 ET on that day."""
    lo = start.tz_localize(None) - pd.Timedelta(days=40)
    hi = end.tz_localize(None) + pd.Timedelta(days=40)
    hol = USFederalHolidayCalendar().holidays(start=lo, end=hi)
    days = pd.bdate_range(lo, hi).difference(pd.DatetimeIndex(hol))
    first = pd.Series(days).groupby([days.year, days.month]).min()
    out = pd.DatetimeIndex(sorted(first.values))
    return out[(out >= start.tz_localize(None)) & (out <= end.tz_localize(None))]


def session_closes_1600(px: pd.Series) -> pd.Series:
    """Last print at or before 16:00 ET on each calendar date that actually has one.

    Weekends and holidays simply do not appear, so "the previous session's 16:00
    close" is a lookup rather than a calendar-arithmetic guess. Getting this wrong
    silently drops Mondays, which matters here because ISM lands on the first
    business day of the month and is therefore often a Monday.
    """
    at_or_before = px[(px.index.hour < 16) | ((px.index.hour == 16) & (px.index.minute == 0))]
    by_day = at_or_before.groupby(at_or_before.index.normalize())
    last_ts = by_day.apply(lambda s: s.index[-1])
    last_px = by_day.last()
    keep = (last_ts.index + pd.Timedelta(hours=16) - last_ts) <= pd.Timedelta(minutes=15)
    out = pd.Series(last_px[keep].to_numpy(), index=pd.DatetimeIndex(last_ts[keep].to_numpy()))
    return out.sort_index()


def window_returns(px: pd.Series, closes: pd.Series, target_days: pd.DatetimeIndex,
                   release_et: str) -> pd.DataFrame:
    """Prior session's 16:00 ET close -> 5 minutes before the release."""
    hh, mm = (int(x) for x in release_et.split(":"))
    close_days = pd.DatetimeIndex([t.normalize() for t in closes.index])
    rows = []
    for day in target_days:
        j = close_days.searchsorted(day, side="left") - 1   # strictly previous session
        if j < 0:
            continue
        entry_ts = closes.index[j]
        exit_t = day + pd.Timedelta(hours=hh, minutes=mm) - pd.Timedelta(minutes=5)
        seg_out = px[:exit_t]
        if seg_out.empty or seg_out.index[-1] <= entry_ts:
            continue
        if (exit_t - seg_out.index[-1]) > pd.Timedelta(minutes=15):
            continue
        # the entry must be the session immediately before, not across a long gap
        if (day - entry_ts) > pd.Timedelta(days=5):
            continue
        entry, exit_px = float(closes.iloc[j]), float(seg_out.iloc[-1])
        path = px[entry_ts:seg_out.index[-1]]
        rows.append({"day": day, "entry_time": entry_ts, "exit_time": seg_out.index[-1],
                     "entry": entry, "exit": exit_px, "points": exit_px - entry,
                     "mae_points": float(path.min() - entry),
                     "mfe_points": float(path.max() - entry)})
    return pd.DataFrame(rows)


def summarise(tr: pd.DataFrame, point_value: float, contracts: int, cost: float, label: str) -> dict:
    if tr.empty or len(tr) < 5:
        return {"label": label, "n": int(len(tr))}
    pts = tr["points"]
    net = pts * point_value * contracts - cost
    t = stats.ttest_1samp(net, 0)
    bps = (tr["points"] / tr["entry"] * 10_000)
    return {"label": label, "n": int(len(tr)),
            "mean_points": float(pts.mean()), "mean_bps": float(bps.mean()),
            "mean_gross": float((pts * point_value * contracts).mean()),
            "mean_net": float(net.mean()), "total_net": float(net.sum()),
            "t_stat": float(t.statistic), "p_value": float(t.pvalue),
            "win_rate": float((net > 0).mean()),
            "mae_p50": float(tr["mae_points"].median() * point_value * contracts),
            "mae_p05": float(tr["mae_points"].quantile(0.05) * point_value * contracts),
            "mae_worst": float(tr["mae_points"].min() * point_value * contracts)}


def welch(a: pd.Series, b: pd.Series) -> dict:
    t = stats.ttest_ind(a, b, equal_var=False)
    return {"diff_points": float(a.mean() - b.mean()),
            "t_stat": float(t.statistic), "p_value": float(t.pvalue)}


def main() -> None:
    ap = argparse.ArgumentParser(description="HPWZ pre-announcement drift study")
    ap.add_argument("--instrument", choices=sorted(INSTRUMENTS), required=True)
    ap.add_argument("--event", choices=sorted(EVENTS), required=True)
    ap.add_argument("--calendar", type=Path, default=None,
                    help="CSV with a release_date column; required for gdp")
    ap.add_argument("--contracts", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg, ev = INSTRUMENTS[args.instrument], EVENTS[args.event]
    px = load_bars(cfg)
    closes = session_closes_1600(px)
    sessions = pd.DatetimeIndex([t.normalize() for t in closes.index])
    lo, hi = sessions.min(), sessions.max()

    if args.event == "ism":
        rel = ism_release_dates(lo, hi).tz_localize(ET)
    else:
        if args.calendar is None:
            raise SystemExit("--calendar with real BEA release dates is required for gdp")
        rel = pd.DatetimeIndex(pd.read_csv(args.calendar)["release_date"]).tz_localize(ET)
    rel = pd.DatetimeIndex([d for d in rel if d in set(sessions)])
    non_event = sessions.difference(rel)

    cost = round_turn_cost(cfg["point_value"], cfg["tick_size"], args.contracts)
    print(f"{args.instrument.upper()}  {ev['label']}  release {ev['release_et']} ET")
    print(f"  sessions {len(sessions)}  {lo.date()} .. {hi.date()}   "
          f"event days {len(rel)}   control days {len(non_event)}")
    print(f"  window: prior session 16:00 ET -> {ev['release_et']} minus 5 min, "
          f"{args.contracts} contracts, cost ${cost:.2f}\n")

    ev_tr = window_returns(px, closes, rel, ev["release_et"])
    pl_tr = window_returns(px, closes, non_event, ev["release_et"])

    report = {"instrument": args.instrument, "event": args.event,
              "sessions": int(len(sessions)), "n_event_days": int(len(rel)), "rows": []}
    dev_end = pd.Timestamp(cfg["dev_end"], tz=ET)
    eras = [("FULL", lo, hi), (f"DEV..{cfg['dev_end']}", lo, dev_end),
            (f"VAL {cfg['dev_end']}..", dev_end, hi)]

    print(f"{'era':<22}{'n':>5}{'bps':>8}{'gross$':>9}{'net$':>9}{'t':>7}{'WR':>7}   MAE p50/p05/worst")
    for era, a, b in eras:
        for name, tr in (("EVENT", ev_tr), ("control", pl_tr)):
            sub = tr[(tr["day"] > a) & (tr["day"] <= b)] if era != "FULL" else tr
            s = summarise(sub, cfg["point_value"], args.contracts, cost, f"{era}/{name}")
            s["era"] = era
            s["kind"] = name
            report["rows"].append(s)
            if s.get("n", 0) >= 5:
                print(f"{era + ' ' + name:<22}{s['n']:>5}{s['mean_bps']:>8.2f}"
                      f"{s['mean_gross']:>9.2f}{s['mean_net']:>9.2f}{s['t_stat']:>7.2f}"
                      f"{s['win_rate']:>7.1%}   ${s['mae_p50']:.0f}/${s['mae_p05']:.0f}/${s['mae_worst']:.0f}")
        ea = ev_tr[(ev_tr["day"] > a) & (ev_tr["day"] <= b)] if era != "FULL" else ev_tr
        pa = pl_tr[(pl_tr["day"] > a) & (pl_tr["day"] <= b)] if era != "FULL" else pl_tr
        if len(ea) >= 5 and len(pa) >= 5:
            w = welch(ea["points"], pa["points"])
            w["era"] = era
            report.setdefault("event_vs_control", []).append(w)
            print(f"{'  -> event - control':<22}{'':>5}{'':>8}"
                  f"{w['diff_points'] * cfg['point_value'] * args.contracts:>9.2f}{'':>9}"
                  f"{w['t_stat']:>7.2f}          p={w['p_value']:.3f}")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
