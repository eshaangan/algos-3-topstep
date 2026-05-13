"""VWAP MR parameter sweep — find best config to improve 59.5% baseline WR.

Sweeps entry_distance_atr, PT mult, SL mult, time window, max_move_from_open_atr.
Evaluates on full 2026 YTD data (no train/test split — watch for overfitting).

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/vwap_param_sweep.py
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
RESULTS_PATH = RBV1 / "diagnostics" / "vwap_sweep_results.json"

_DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "mnq_2026ytd_databento_5min_rth.h5",
    ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5",
]

def load_bars() -> pd.DataFrame:
    for p in _DATA_CANDIDATES:
        if p.exists():
            df = pd.read_hdf(str(p), key="bars_5min")
            if df.index.tz is None:
                df.index = df.index.tz_localize("US/Eastern")
            else:
                df.index = df.index.tz_convert("US/Eastern")
            return df
    raise FileNotFoundError("No 2026 YTD data found")

POINT_VALUE = 2.0
TICK_SIZE   = 0.25
COMMISSION  = 0.62
N_CONTRACTS = 5

def slip(price, direction, is_entry):
    return price + direction * TICK_SIZE * (1 if is_entry else -1)

def trade_pnl(entry, exit_, direction):
    return (exit_ - entry) * direction * N_CONTRACTS * POINT_VALUE - 2 * COMMISSION * N_CONTRACTS

def compute_atr(bars, period=14):
    prev = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"],
                    (bars["high"] - prev).abs(),
                    (bars["low"]  - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def build_vwap(bars):
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    tp_vol = tp * bars["volume"]
    dates = bars.index.map(lambda t: t.date())
    return tp_vol.groupby(dates).cumsum() / bars["volume"].groupby(dates).cumsum().replace(0, np.nan)

def run_vwap_only(
    bars, atr_s, vwap_s,
    entry_dist=1.0, max_dist=3.0,
    time_start="10:30", time_end="13:30",
    pt_mult=2.0, sl_mult=1.5,
    time_stop_bars=12, max_per_day=2,
    max_move_atr=2.0, long_only=True,
    max_daily_loss=-700.0,
):
    tstart = pd.Timestamp(f"2000-01-01 {time_start}").time()
    tend   = pd.Timestamp(f"2000-01-01 {time_end}").time()

    @dataclass
    class Pos:
        direction: int
        entry_price: float
        stop_loss: float
        profit_target: float
        time_stop_bar: int

    trades = []
    pos: Optional[Pos] = None
    cur_date = None
    daily_pnl_run = 0.0
    daily_pnl_map = {}
    vwap_today = 0
    equity = 50_000.0
    peak_equity = equity
    eq_curve = [equity]

    min_bars = 16
    for i in range(min_bars, len(bars)):
        bar  = bars.iloc[i]
        bt   = bars.index[i]
        bdate = bt.date()

        if cur_date is not None and bdate != cur_date:
            daily_pnl_map[cur_date] = daily_pnl_run
            daily_pnl_run = 0.0
            vwap_today = 0
        cur_date = bdate

        is_last = (i + 1 >= len(bars)) or (bars.index[i+1].date() != bdate)
        sess_close = is_last or (bt.hour == 15 and bt.minute >= 55)

        if pos is not None:
            h, l, c = bar["high"], bar["low"], bar["close"]
            exited, ex_p, reason = False, 0.0, ""
            if sess_close or i >= pos.time_stop_bar:
                exited, ex_p, reason = True, slip(c, pos.direction, False), "time_stop"
            elif pos.direction == 1:
                if l <= pos.stop_loss:
                    exited, ex_p, reason = True, slip(pos.stop_loss, 1, False), "stop_loss"
                elif h >= pos.profit_target:
                    exited, ex_p, reason = True, slip(pos.profit_target, 1, False), "profit_target"
            else:
                if h >= pos.stop_loss:
                    exited, ex_p, reason = True, slip(pos.stop_loss, -1, False), "stop_loss"
                elif l <= pos.profit_target:
                    exited, ex_p, reason = True, slip(pos.profit_target, -1, False), "profit_target"
            if exited:
                p = trade_pnl(pos.entry_price, ex_p, pos.direction)
                trades.append({"pnl": p, "reason": reason, "direction": pos.direction})
                daily_pnl_run += p
                equity += p
                peak_equity = max(peak_equity, equity)
                eq_curve.append(equity)
                pos = None

        if sess_close or pos is not None:
            continue
        if daily_pnl_run <= max_daily_loss or vwap_today >= max_per_day:
            continue

        bt_time = bt.time()
        if not (tstart <= bt_time <= tend):
            continue

        atr_val = float(atr_s.iloc[i])
        if np.isnan(atr_val) or atr_val <= 0:
            continue
        vwap = float(vwap_s.iloc[i])
        if np.isnan(vwap):
            continue

        close  = float(bar["close"])
        prev_c = float(bars["close"].iloc[i-1])
        dev    = (close - vwap) / atr_val

        if abs(dev) > max_dist:
            continue

        # Regime gate
        today_bars = bars[bars.index.map(lambda t: t.date()) == bdate]
        if not today_bars.empty and max_move_atr > 0:
            day_open = float(today_bars["open"].iloc[0])
            move_atr = (close - day_open) / atr_val
            if dev < 0 and move_atr < -max_move_atr:
                continue
            if dev > 0 and move_atr > max_move_atr:
                continue

        sig = None
        if dev <= -entry_dist and close > prev_c:
            sig = 1
        elif not long_only and dev >= entry_dist and close < prev_c:
            sig = -1

        if sig is not None:
            ep = slip(close, sig, True)
            pos = Pos(
                direction=sig,
                entry_price=ep,
                stop_loss=ep - sig * sl_mult * atr_val,
                profit_target=ep + sig * pt_mult * atr_val,
                time_stop_bar=i + time_stop_bars,
            )
            vwap_today += 1

    if cur_date and cur_date not in daily_pnl_map:
        daily_pnl_map[cur_date] = daily_pnl_run

    if not trades:
        return None

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total  = sum(t["pnl"] for t in trades)
    gp     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))
    daily  = pd.Series(daily_pnl_map)
    active = daily[daily != 0]
    sharpe = float(active.mean() / active.std() * np.sqrt(252)) if len(active) > 1 and active.std() > 0 else 0
    eq_s   = pd.Series(eq_curve)
    max_dd = float((eq_s - eq_s.cummax()).min())
    n_wks  = max(1, len(daily) / 5)

    return {
        "n": len(trades),
        "wr": len(wins) / len(trades),
        "total_pnl": round(total, 2),
        "weekly_pnl": round(total / n_wks, 2),
        "avg_win": round(gp / len(wins), 2) if wins else 0,
        "avg_loss": round(-gl / len(losses), 2) if losses else 0,
        "pf": round(gp / gl, 3) if gl > 0 else 99.0,
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd, 2),
    }


def main():
    bars  = load_bars()
    atr_s = compute_atr(bars)
    vwap_s = build_vwap(bars)

    print(f"Loaded {len(bars):,} bars  [{bars.index[0].date()} → {bars.index[-1].date()}]")
    print("Running VWAP parameter sweep...")

    grid = list(product(
        [0.75, 1.0, 1.25, 1.5],       # entry_dist_atr
        [1.5, 2.0, 2.5],              # pt_mult
        [1.0, 1.25, 1.5],             # sl_mult
        ["10:00", "10:30", "11:00"],  # time_start
        ["13:00", "13:30", "14:00"],  # time_end
        [1.5, 2.0, 3.0],             # max_move_from_open_atr
    ))

    print(f"Testing {len(grid):,} configurations...")

    results = []
    for (entry_dist, pt, sl, tstart, tend, max_move) in grid:
        if pt <= sl:
            continue
        r = run_vwap_only(
            bars, atr_s, vwap_s,
            entry_dist=entry_dist,
            pt_mult=pt, sl_mult=sl,
            time_start=tstart, time_end=tend,
            max_move_atr=max_move,
            long_only=True,
        )
        if r and r["n"] >= 15:  # need enough trades to be meaningful
            results.append({
                "entry_dist": entry_dist, "pt": pt, "sl": sl,
                "tstart": tstart, "tend": tend, "max_move": max_move,
                **r,
            })

    if not results:
        print("No results found.")
        return

    # Sort by Sharpe × WR (reward quality and frequency)
    results.sort(key=lambda x: x["sharpe"] * x["wr"], reverse=True)

    print(f"\nFound {len(results)} valid configs. Top 20 by Sharpe×WR:\n")
    print(f"  {'Dist':>5} {'PT':>4} {'SL':>4} {'Start':>6} {'End':>6} {'MaxMv':>6} "
          f"{'N':>5} {'WR':>7} {'$/wk':>8} {'MaxDD':>8} {'Sharpe':>8} {'PF':>6}")
    print(f"  {'-'*90}")
    for r in results[:20]:
        print(f"  {r['entry_dist']:>5.2f} {r['pt']:>4.1f} {r['sl']:>4.2f} "
              f"{r['tstart']:>6} {r['tend']:>6} {r['max_move']:>6.1f} "
              f"{r['n']:>5} {r['wr']:>7.1%} {r['weekly_pnl']:>8,.0f} "
              f"{r['max_dd']:>8,.0f} {r['sharpe']:>8.2f} {r['pf']:>6.2f}")

    best = results[0]
    print(f"\nBest config:")
    print(f"  Entry dist : {best['entry_dist']}x ATR")
    print(f"  PT / SL    : {best['pt']}x / {best['sl']}x ATR")
    print(f"  Window     : {best['tstart']} - {best['tend']}")
    print(f"  Max move   : {best['max_move']}x ATR from open")
    print(f"  Trades     : {best['n']}  WR={best['wr']:.1%}  $/wk=${best['weekly_pnl']:,.0f}")
    print(f"  Max DD     : ${best['max_dd']:,.0f}  Sharpe={best['sharpe']:.2f}  PF={best['pf']:.2f}")

    # Also show baseline (default params)
    baseline = run_vwap_only(bars, atr_s, vwap_s)
    print(f"\nBaseline (1.0 entry, PT=2.0, SL=1.5, 10:30-13:30, maxmv=2.0):")
    if baseline:
        print(f"  Trades={baseline['n']}  WR={baseline['wr']:.1%}  $/wk=${baseline['weekly_pnl']:,.0f}  "
              f"DD=${baseline['max_dd']:,.0f}  Sharpe={baseline['sharpe']:.2f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results[:50], f, indent=2)
    print(f"\nSaved top 50 → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
