"""Ivan Trades Fibonacci Strategy — Parameter Sweep.

Tests combinations of:
  - touch_tolerance_atr: how close price must come to level
  - min_wick_ratio: wick size as fraction of bar total range (pinbar quality)
  - min_body_confirm: close must be at least N*ATR from level (strong rejection)
  - long_only: test LONG-only vs bidirectional
  - entry_window: time cutoff for entries

Run:
    python rule_based_v1/diagnostics/ivan_fib_sweep.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "processed"
DIAG_DIR = ROOT / "rule_based_v1" / "diagnostics"
MNQ_PATH = DATA_DIR / "mnq_5min_aug25_mar26.h5"
RESULTS_PATH = DIAG_DIR / "ivan_fib_sweep_results.json"

for p in [str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

POINT_VALUE = 2.0
SLIP = 0.25
COMMISSION = 0.62
N_CONTRACTS = 2
KEY_LEVELS = [0.382, 0.500, 0.618]
ATR_PERIOD = 14
TREND_LEVEL = 0.500
PT_ATR = 3.0
SL_ATR = 1.5
ENTRY_START_MINUTE = 9 * 60 + 35


def load_bars() -> pd.DataFrame:
    with pd.HDFStore(str(MNQ_PATH), mode="r") as store:
        df = store["/bars_5min"].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    return df.sort_index()


def _calc_atr(bars: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = bars["close"].shift(1)
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def build_daily_data(df: pd.DataFrame) -> dict:
    atr_series = _calc_atr(df)
    dates = sorted(df.index.normalize().unique())
    rth_by_date = {d: df[df.index.normalize() == d] for d in dates
                   if len(df[df.index.normalize() == d]) > 0}
    date_list = sorted(rth_by_date.keys())
    daily: dict = {}

    for i, d in enumerate(date_list):
        if i == 0:
            continue
        prev_d = date_list[i - 1]
        if prev_d not in rth_by_date:
            continue
        prev_rth = rth_by_date[prev_d]
        pdh = float(prev_rth["high"].max())
        pdl = float(prev_rth["low"].min())
        fib_range = pdh - pdl
        if fib_range < 1e-6:
            continue

        fib_prices = {lvl: pdl + lvl * fib_range for lvl in KEY_LEVELS}
        trend_price = pdl + TREND_LEVEL * fib_range
        cur_bars = rth_by_date[d]
        if len(cur_bars) < 2:
            continue

        today_open = float(cur_bars["open"].iloc[0])
        trend = 1 if today_open > trend_price else -1

        atr_slice = atr_series.loc[:cur_bars.index[0]]
        atr_at_open = float(atr_slice.iloc[-2]) if len(atr_slice) >= 2 else fib_range / 2.0
        if np.isnan(atr_at_open) or atr_at_open <= 0:
            atr_at_open = fib_range / 2.0

        daily[d] = {
            "pdh": pdh, "pdl": pdl, "fib_range": fib_range,
            "fib_prices": fib_prices, "trend": trend,
            "today_open": today_open, "atr_at_open": atr_at_open,
            "cur_bars": cur_bars,
        }
    return daily


def simulate_day(
    day_data: dict,
    touch_tolerance_atr: float = 0.3,
    min_wick_ratio: float = 0.0,       # wick / bar_range >= this
    min_body_confirm_atr: float = 0.0, # close must be this far from level
    long_only: bool = False,
    entry_stop_minute: int = 12 * 60,
) -> Optional[dict]:
    fib_prices = day_data["fib_prices"]
    trend = day_data["trend"]
    atr_val = day_data["atr_at_open"]
    tolerance = touch_tolerance_atr * atr_val
    cur_bars = day_data["cur_bars"]
    pt_dist = PT_ATR * atr_val
    sl_dist = SL_ATR * atr_val

    if long_only and trend == -1:
        return None

    entry_price = entry_ts = direction = fib_triggered = fib_triggered_price = tp = sl = None

    for ts, bar in cur_bars.iterrows():
        bar_minute = ts.hour * 60 + ts.minute
        if bar_minute < ENTRY_START_MINUTE:
            continue
        if bar_minute >= entry_stop_minute:
            break

        bar_range = float(bar["high"]) - float(bar["low"])

        for level_pct, level_price in sorted(fib_prices.items(),
                                              key=lambda kv: abs(kv[1] - float(bar["close"]))):
            if trend == 1:
                touched = float(bar["low"]) <= level_price + tolerance
                rejected = float(bar["close"]) > level_price

                if touched and rejected:
                    # Wick quality: lower wick from level
                    lower_wick = level_price - float(bar["low"])
                    wick_ok = (bar_range < 1e-9) or (lower_wick / bar_range >= min_wick_ratio)
                    # Body confirmation: close is far enough above level
                    body_ok = (float(bar["close"]) - level_price) >= min_body_confirm_atr * atr_val

                    if wick_ok and body_ok:
                        entry_price = float(bar["close"]) + SLIP
                        tp = entry_price + pt_dist
                        sl = entry_price - sl_dist
                        direction = "LONG"
                        entry_ts = ts
                        fib_triggered = level_pct
                        fib_triggered_price = level_price
                        break
            else:
                touched = float(bar["high"]) >= level_price - tolerance
                rejected = float(bar["close"]) < level_price

                if touched and rejected:
                    upper_wick = float(bar["high"]) - level_price
                    wick_ok = (bar_range < 1e-9) or (upper_wick / bar_range >= min_wick_ratio)
                    body_ok = (level_price - float(bar["close"])) >= min_body_confirm_atr * atr_val

                    if wick_ok and body_ok:
                        entry_price = float(bar["close"]) - SLIP
                        tp = entry_price - pt_dist
                        sl = entry_price + sl_dist
                        direction = "SHORT"
                        entry_ts = ts
                        fib_triggered = level_pct
                        fib_triggered_price = level_price
                        break

        if entry_ts is not None:
            break

    if entry_ts is None:
        return None

    post_entry = cur_bars[cur_bars.index > entry_ts]
    exit_price = None
    exit_reason = None

    for ts2, bar2 in post_entry.iterrows():
        bar_minute2 = ts2.hour * 60 + ts2.minute
        if bar_minute2 >= entry_stop_minute:
            exit_price = float(bar2["open"]) + (SLIP if direction == "SHORT" else -SLIP)
            exit_reason = "time"
            break
        if direction == "LONG":
            if float(bar2["low"]) <= sl:
                exit_price = sl; exit_reason = "stop"; break
            if float(bar2["high"]) >= tp:
                exit_price = tp; exit_reason = "tp"; break
        else:
            if float(bar2["high"]) >= sl:
                exit_price = sl; exit_reason = "stop"; break
            if float(bar2["low"]) <= tp:
                exit_price = tp; exit_reason = "tp"; break

    if exit_price is None:
        if len(post_entry) > 0:
            last_bar = post_entry.iloc[-1]
            exit_price = float(last_bar["close"]) + (SLIP if direction == "SHORT" else -SLIP)
            exit_reason = "eod"
        else:
            exit_price = entry_price; exit_reason = "eod"

    pnl_pts = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
    pnl_usd = pnl_pts * POINT_VALUE * N_CONTRACTS - 2 * COMMISSION * N_CONTRACTS

    return {
        "date": str(entry_ts.date()), "direction": direction,
        "fib_level_pct": round(fib_triggered, 3),
        "exit_reason": exit_reason, "pnl_usd": round(pnl_usd, 2),
    }


def run_config(
    daily: dict,
    touch_tolerance_atr: float,
    min_wick_ratio: float,
    min_body_confirm_atr: float,
    long_only: bool,
    entry_stop_minute: int,
) -> dict:
    trades = []
    for d in sorted(daily.keys()):
        result = simulate_day(
            daily[d],
            touch_tolerance_atr=touch_tolerance_atr,
            min_wick_ratio=min_wick_ratio,
            min_body_confirm_atr=min_body_confirm_atr,
            long_only=long_only,
            entry_stop_minute=entry_stop_minute,
        )
        if result:
            trades.append(result)

    if not trades:
        return {"N": 0, "WR": 0.0, "AvgPnL": 0.0, "TotalPnL": 0.0,
                "Sharpe": 0.0, "MaxDD": 0.0}

    pnls = [t["pnl_usd"] for t in trades]
    wins = [p for p in pnls if p > 0]

    daily_pnl: dict[str, float] = {}
    for t in trades:
        daily_pnl[t["date"]] = daily_pnl.get(t["date"], 0.0) + t["pnl_usd"]
    dpnl = list(daily_pnl.values())
    mean_d = float(np.mean(dpnl)) if dpnl else 0.0
    std_d = float(np.std(dpnl, ddof=1)) if len(dpnl) > 1 else 1e-9
    sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 1e-9 else 0.0

    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity - peak).min())

    return {
        "N": len(trades),
        "WR": round(100.0 * len(wins) / len(trades), 1),
        "AvgPnL": round(float(np.mean(pnls)), 2),
        "TotalPnL": round(float(sum(pnls)), 2),
        "Sharpe": round(sharpe, 3),
        "MaxDD": round(max_dd, 2),
    }


def main():
    print("Loading MNQ 5-min bars...")
    df = load_bars()
    print(f"  {len(df):,} bars  {df.index[0].date()} → {df.index[-1].date()}")

    print("Building daily data...")
    daily = build_daily_data(df)
    print(f"  {len(daily)} days")

    # Grid sweep
    tolerances = [0.1, 0.2, 0.3]
    wick_ratios = [0.0, 0.2, 0.3, 0.4]     # min wick / bar_range
    body_confirms = [0.0, 0.05, 0.1]        # min close distance from level in ATR
    long_onlys = [False, True]
    stop_minutes = [11 * 60, 12 * 60]

    total = (len(tolerances) * len(wick_ratios) * len(body_confirms)
             * len(long_onlys) * len(stop_minutes))
    print(f"\nRunning {total} parameter combinations...")

    rows = []
    for tol, wr, bc, lo, sm in itertools.product(
        tolerances, wick_ratios, body_confirms, long_onlys, stop_minutes
    ):
        stats = run_config(daily, tol, wr, bc, lo, sm)
        rows.append({
            "tol": tol, "wick_ratio": wr, "body_confirm": bc,
            "long_only": lo, "stop_hr": sm // 60,
            **stats,
        })

    # Filter: N >= 15, sort by Sharpe
    qualified = [r for r in rows if r["N"] >= 15]
    qualified.sort(key=lambda x: x["Sharpe"], reverse=True)

    print(f"\n{'='*90}")
    print(f"  TOP 20 CONFIGS (N≥15, sorted by Sharpe)")
    print(f"{'='*90}")
    print(f"  {'tol':>4} {'wick':>5} {'body':>5} {'lo':>4} {'hr':>3}  "
          f"{'N':>4}  {'WR%':>6}  {'AvgPnL':>8}  {'Sharpe':>7}  {'MaxDD':>9}  {'TotalPnL':>10}")
    print("  " + "-" * 85)

    for r in qualified[:20]:
        print(
            f"  {r['tol']:>4.2f} {r['wick_ratio']:>5.2f} {r['body_confirm']:>5.2f} "
            f"{str(r['long_only']):>4}  {r['stop_hr']:>2}  "
            f"{r['N']:>4}  {r['WR']:>5.1f}%  "
            f"${r['AvgPnL']:>7.2f}  {r['Sharpe']:>7.3f}  "
            f"${r['MaxDD']:>8.2f}  ${r['TotalPnL']:>9.2f}"
        )

    if qualified:
        best = qualified[0]
        print(f"\nBest: tol={best['tol']}, wick={best['wick_ratio']}, body={best['body_confirm']}, "
              f"long_only={best['long_only']}, stop_hr={best['stop_hr']}")
        print(f"  N={best['N']}, WR={best['WR']}%, Sharpe={best['Sharpe']:.3f}, "
              f"MaxDD=${best['MaxDD']:.2f}, TotalPnL=${best['TotalPnL']:.2f}")
    else:
        print("\nNo config passed N≥15 gate.")
        best = None

    results = {"grid": rows, "top20": qualified[:20], "best": best}
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
