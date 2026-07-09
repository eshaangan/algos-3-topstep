"""MGC (micro gold) edge batteries on deep Rithmic 1-min history — DEV era only.

Rationale: index futures are measured out (cost-walled fast, sample-starved slow).
Gold has different participants (central banks, physical, Asia flow), real intraday
ranges vs costs (~$400-700/day per micro vs ~$2.5 round trip), and 6y of free data.

Era protocol (FROZEN): DEV 2020-01..2023-12 (selection) | VAL 2024-01..2025-06
(one look per DEV survivor) | FINAL 2025-07.. sealed single shot.

Families this battery (counted):
  S1  session-hold 18:00->16:45 long (gold overnight/carry premium)
  S2  session-hold | prior day down (reversal-conditioned)
  I1  range_ignite on 5-min (gold-tuned window 08:00-13:30 ET)
  I2  vwap_pullback resume
  I3  london_open breakout (03:00 ET +/- first 30min range, hold <=2h)

    python rule_based_v1/validation/research_mgc.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rule_based_v1.validation.harness import SimParams, simulate, aggregate_stats  # noqa: E402

PV, TICK, COMM, NC = 10.0, 0.10, 0.62, 2
DEV_END, VAL_END = "2024-01-01", "2025-07-01"


def load_1m(dirpath: str) -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(f"{dirpath}/m_*.parquet"))]
    b = pd.concat(frames).drop_duplicates("ts").sort_values("ts")
    ts = pd.to_datetime(b["ts"]).dt.tz_localize("America/Chicago",
                                                ambiguous="NaT", nonexistent="NaT")
    return (b.assign(et=ts.dt.tz_convert("America/New_York"))
             .dropna(subset=["et"]).set_index("et").sort_index())


def five_min(b: pd.DataFrame, lo_min: int, hi_min: int) -> pd.DataFrame:
    mins = b.index.hour * 60 + b.index.minute
    w = b[(mins >= lo_min) & (mins < hi_min)]
    g = w.resample("5min")
    bars = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                         "low": g["low"].min(), "close": g["close"].last()}).dropna()
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    bars["atr"] = tr.ewm(span=14, min_periods=14).mean()
    day = bars.index.normalize()
    bars["cum_mean"] = bars["close"].groupby(day).expanding().mean().droplevel(0)  # TWAP proxy
    return bars


def rep_daily(name, u):
    u = u.dropna()
    if len(u) < 60:
        print(f"  {name:38s} n={len(u)} (thin)")
        return
    t = u.mean() / (u.std(ddof=1) / np.sqrt(len(u)))
    yr = u.groupby(u.index.year).sum()
    print(f"  {name:38s} n={len(u):>4} ${u.mean():+7.2f}/d t={t:+5.2f} "
          f"posYrs={(yr > 0).mean():.0%} ({', '.join(f'{y}:{v:+,.0f}' for y, v in yr.items())})")


def rep_sim(name, bars, d, pt=1.5, sl=1.0, h=4):
    t = simulate(bars[["open", "high", "low", "close"]],
                 pd.DataFrame({"direction": d, "atr": bars["atr"]}),
                 SimParams(pt_atr=pt, sl_atr=sl, horizon_bars=h, point_value=PV,
                           tick_size=TICK, commission_per_side=COMM, n_contracts=NC,
                           max_trades_per_day=6, cooldown_bars=2, decision_lag_secs=3.0))
    a = aggregate_stats(t)
    if not a["n_trades"]:
        print(f"  {name:38s} 0tr")
        return
    pnl = t["pnl"].to_numpy()
    tt = pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl)))
    yr = t.assign(y=pd.to_datetime(t["entry_time"]).dt.year).groupby("y")["pnl"].sum()
    print(f"  {name:38s} {a['n_trades']:>5}tr WR={a['win_rate']:.0%} ${a['total_pnl']:>9,.0f} "
          f"t={tt:+5.2f} posYrs={(yr > 0).mean():.0%} ({', '.join(f'{y}:{v:+,.0f}' for y, v in yr.items())})")


def main():
    b = load_1m(str(ROOT / "data" / "hist_1m24_mgc"))
    print(f"MGC 1-min: {len(b):,} bars {b.index[0].date()}..{b.index[-1].date()}")
    dev1m = b[b.index < pd.Timestamp(DEV_END, tz="America/New_York")]

    # S: session-hold series (18:00 -> next 16:45)
    rows = []
    for d in sorted({x for x in dev1m.index.normalize().unique()}):
        ent = d + pd.Timedelta(hours=18)
        if ent.weekday() in (4, 5):
            continue
        w = dev1m.loc[ent: ent + pd.Timedelta(hours=22, minutes=45)]
        if len(w) < 400:
            continue
        entry = w["open"].iloc[0] + TICK
        pnl = (w["close"].iloc[-1] - TICK - entry) * PV * NC - 2 * COMM * NC
        rows.append({"day": w.index[-1].normalize(), "pnl": pnl})
    sh = pd.DataFrame(rows).set_index("day")["pnl"]
    print("\n[S] session-hold (long, 2 micros) — DEV")
    rep_daily("S1 always", sh)
    # prior-day down conditioning
    daily_close = dev1m["close"].resample("1D").last().dropna()
    prev_ret = daily_close.diff().shift(1).reindex(sh.index, method="ffill")
    rep_daily("S2 | prior day down", sh[prev_ret < 0])

    # I: intraday families on 5-min, NY-active window 08:00-13:30 ET
    bars = five_min(dev1m, 8 * 60, 13 * 60 + 30)
    mins = bars.index.hour * 60 + bars.index.minute
    ok = (mins >= 8 * 60 + 30) & (mins <= 13 * 60)
    rng = bars["high"] - bars["low"]
    pos = (bars["close"] - bars["low"]) / rng.replace(0, np.nan)
    print("\n[I] intraday 5-min families — DEV")
    d1 = pd.Series(0, index=bars.index, dtype=int)
    d1[ok & (rng >= 1.8 * bars["atr"]) & (pos >= 0.75)] = 1
    d1[ok & (rng >= 1.8 * bars["atr"]) & (pos <= 0.25)] = -1
    rep_sim("I1 range_ignite", bars, d1)
    stretch = (bars["close"] - bars["cum_mean"]) / bars["atr"]
    near = stretch.abs() <= 0.25
    up = bars["close"].rolling(24, min_periods=12).mean() > bars["cum_mean"]
    d2 = pd.Series(0, index=bars.index, dtype=int)
    d2[ok & near & up & (stretch.shift(2) > 0.5)] = 1
    d2[ok & near & ~up & (stretch.shift(2) < -0.5)] = -1
    rep_sim("I2 twap_pullback", bars, d2)
    # I3: london-open breakout — 03:00-03:30 range, break by 05:30, hold <=2h
    lbars = five_min(b[b.index < pd.Timestamp(DEV_END, tz="America/New_York")], 3 * 60, 8 * 60)
    lm = lbars.index.hour * 60 + lbars.index.minute
    day = lbars.index.normalize()
    orh = lbars["high"].where(lm < 210).groupby(day).transform("max")
    orl = lbars["low"].where(lm < 210).groupby(day).transform("min")
    d3 = pd.Series(0, index=lbars.index, dtype=int)
    entry_w = (lm >= 210) & (lm <= 330)
    d3[entry_w & (lbars["close"] > orh)] = 1
    d3[entry_w & (lbars["close"] < orl)] = -1
    first = d3[d3 != 0].groupby(d3[d3 != 0].index.normalize()).head(1)
    d3 = pd.Series(0, index=lbars.index, dtype=int)
    d3.loc[first.index] = first.values
    rep_sim("I3 london_breakout", lbars, d3, h=24)


if __name__ == "__main__":
    main()
