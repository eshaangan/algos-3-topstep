"""5-30min zone research on 1s-bar history — DEV ONLY (< 2026-04-01; holdout sealed).

The one horizon never explored: fast enough for capital velocity, slow enough that
measured costs (~1pt) are small vs 15-40pt targets. Data: full-span Rithmic 1s bars
with volume (hist_1sv), aggregated to 5-min bars with session VWAP.

Families (mechanism-first, 8 configs this run — counted):
  orb_fast     15-min opening range breakout, entries to 11:00 ET
  vwap_fade    fade stretched (close-vwap)/ATR back toward vwap
  range_ignite 5-min range expansion closing near its extreme -> continuation

Costs: harness latency model (default), decision_lag=3s (stream pipeline).

    python rule_based_v1/validation/research_5to30.py --bars-dir data/hist_1sv
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rule_based_v1.validation.harness import (  # noqa: E402
    SimParams, simulate, aggregate_stats, monthly_breakdown, deflated_sharpe_ratio,
)

DEV_END = "2026-04-01"
N_TRIALS = 8


def day_5min(f: str) -> pd.DataFrame | None:
    b = pd.read_parquet(f)
    # bar_end_datetime is NAIVE America/Chicago (async_rithmic renders local time);
    # parsing it as UTC shifts sessions 5-6h — the bug that garbaged the first run
    ts = pd.to_datetime(b["ts"]).dt.tz_localize("America/Chicago",
                                                ambiguous="NaT", nonexistent="NaT")
    b = b.assign(ts=ts).dropna(subset=["ts"]).set_index("ts").sort_index()
    et = b.index.tz_convert("America/New_York")
    b = b[(et.hour * 60 + et.minute >= 570) & (et.hour * 60 + et.minute < 960)]
    if len(b) < 3000:
        return None
    g = b.resample("5min")
    bars = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                         "low": g["low"].min(), "close": g["close"].last(),
                         "vol": g["vol"].sum()}).dropna(subset=["close"])
    pv = (b["close"] * b["vol"]).resample("5min").sum()
    bars["vwap"] = (pv.cumsum() / bars["vol"].cumsum().replace(0, np.nan)).ffill()
    return bars


def load_bars(bars_dir: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(f"{bars_dir}/bars_*.parquet")):
        d = day_5min(f)
        if d is not None:
            frames.append(d)
    bars = pd.concat(frames).sort_index()
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    bars["atr"] = tr.ewm(span=14, min_periods=14).mean()
    return bars


def et_minutes(bars: pd.DataFrame) -> np.ndarray:
    et = bars.index.tz_convert("America/New_York")
    return (et.hour * 60 + et.minute).to_numpy()


def sig_orb_fast(bars: pd.DataFrame) -> pd.Series:
    """Break of the 9:30-9:45 range, entries until 11:00 ET."""
    mins = et_minutes(bars)
    day = bars.index.tz_convert("America/New_York").date
    d = pd.Series(0, index=bars.index, dtype=int)
    df = pd.DataFrame({"day": day, "min": mins, "high": bars["high"], "low": bars["low"],
                       "close": bars["close"]})
    for _, g in df.groupby("day"):
        orw = g[g["min"] < 585]                       # 9:30,9:35,9:40 bars
        if len(orw) < 3:
            continue
        hi, lo = orw["high"].max(), orw["low"].min()
        w = g[(g["min"] >= 585) & (g["min"] <= 660)]  # 9:45-11:00
        up = w[w["close"] > hi]
        dn = w[w["close"] < lo]
        if len(up) and (not len(dn) or up.index[0] < dn.index[0]):
            d.loc[up.index[0]] = 1
        elif len(dn):
            d.loc[dn.index[0]] = -1
    return d


def sig_vwap_fade(bars: pd.DataFrame, z: float) -> pd.Series:
    stretch = (bars["close"] - bars["vwap"]) / bars["atr"]
    mins = et_minutes(bars)
    d = pd.Series(0, index=bars.index, dtype=int)
    ok = (mins >= 600) & (mins <= 930)                # 10:00-15:30
    d[(stretch >= z) & ok] = -1
    d[(stretch <= -z) & ok] = 1
    return d


def sig_range_ignite(bars: pd.DataFrame, k: float) -> pd.Series:
    rng = bars["high"] - bars["low"]
    pos = (bars["close"] - bars["low"]) / rng.replace(0, np.nan)
    mins = et_minutes(bars)
    ok = (mins >= 585) & (mins <= 930)
    d = pd.Series(0, index=bars.index, dtype=int)
    d[ok & (rng >= k * bars["atr"]) & (pos >= 0.8)] = 1
    d[ok & (rng >= k * bars["atr"]) & (pos <= 0.2)] = -1
    return d


def run_family(name, sig, dev, p):
    t = simulate(dev[["open", "high", "low", "close"]],
                 pd.DataFrame({"direction": sig, "atr": dev["atr"]}), p)
    a = aggregate_stats(t)
    if not a["n_trades"]:
        print(f"  {name:24s} no trades")
        return
    mb = monthly_breakdown(t)
    dsr = deflated_sharpe_ratio(t["pnl"].to_numpy(), n_trials=N_TRIALS).get("dsr")
    posm = float((mb["pnl"] > 0).mean()) if len(mb) else 0.0
    print(f"  {name:24s} {a['n_trades']:>4}tr WR={a['win_rate']:.0%} "
          f"${a['total_pnl']:>8,.0f} DD=${a['max_dd']:>7,.0f} posMo={posm:.0%} "
          f"DSR={dsr if dsr is None else round(dsr,3)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars-dir", default=str(ROOT / "data" / "hist_1sv"))
    args = ap.parse_args()
    bars = load_bars(args.bars_dir)
    dev = bars[bars.index < pd.Timestamp(DEV_END, tz="UTC")]
    days = len({d.date() for d in dev.index})
    print(f"DEV: {len(dev)} 5-min bars over {days} days (< {DEV_END}); holdout SEALED")

    base = dict(point_value=2.0, tick_size=0.25, commission_per_side=0.62,
                n_contracts=2, max_trades_per_day=6, cooldown_bars=2,
                decision_lag_secs=3.0)
    print("\n[orb_fast] hold<=6 bars (30min)")
    for pt, sl in ((1.5, 1.0), (2.0, 1.0)):
        run_family(f"orb_fast PT{pt}/SL{sl}", sig_orb_fast(dev), dev,
                   SimParams(pt_atr=pt, sl_atr=sl, horizon_bars=6, **base))
    print("[vwap_fade] hold<=4 bars (20min)")
    for z in (1.5, 2.0):
        for pt, sl in ((1.5, 1.0),):
            run_family(f"vwap_fade z{z} PT{pt}/SL{sl}", sig_vwap_fade(dev, z), dev,
                       SimParams(pt_atr=pt, sl_atr=sl, horizon_bars=4, **base))
    print("[range_ignite] hold<=4 bars (20min)")
    for k in (2.0, 2.5):
        for pt, sl in ((1.5, 1.0),):
            run_family(f"range_ignite k{k} PT{pt}/SL{sl}", sig_range_ignite(dev, k), dev,
                       SimParams(pt_atr=pt, sl_atr=sl, horizon_bars=4, **base))
    print(f"\ntrials this run: {N_TRIALS} (cumulative ledger in preregister.yaml)")


if __name__ == "__main__":
    main()
