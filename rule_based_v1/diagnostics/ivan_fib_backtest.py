"""Ivan Trades Fibonacci Strategy — MNQ 5-min bar backtest.

Strategy: Trade wick rejections at prior-day Fibonacci retracement levels
(38.2%, 50%, 61.8%) during the morning session (09:35-12:00 ET).

Trend determined by today's open relative to the 50% Fib level:
  - Open above 50% → uptrend → look for LONG pullbacks to key levels
  - Open below 50% → downtrend → look for SHORT rallies to key levels

Entry: First qualifying wick rejection on a 5-min bar in the morning session.
Exit: PT=3.0x ATR, SL=1.5x ATR, time stop at 12:00 ET.

Run:
    python rule_based_v1/diagnostics/ivan_fib_backtest.py
"""

from __future__ import annotations

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
RESULTS_PATH = DIAG_DIR / "ivan_fib_results.json"

for p in [str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POINT_VALUE = 2.0       # $/point for MNQ
SLIP = 0.25             # slippage per side (points)
COMMISSION = 0.62       # $/side/contract
N_CONTRACTS = 2

KEY_LEVELS = [0.382, 0.500, 0.618]
TOUCH_TOL_ATR = 0.3     # bar must come within 0.3x ATR of level
ATR_PERIOD = 14
TREND_LEVEL = 0.500     # open above 50% Fib = uptrend
PT_ATR = 3.0
SL_ATR = 1.5
ENTRY_START_MINUTE = 9 * 60 + 35   # 09:35
ENTRY_STOP_MINUTE = 12 * 60        # 12:00 ET


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ATR helper
# ---------------------------------------------------------------------------

def _calc_atr(bars: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = bars["close"].shift(1)
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Build per-day structures
# ---------------------------------------------------------------------------

def build_daily_data(df: pd.DataFrame) -> dict:
    """Build per-day Fibonacci levels and session bars."""
    atr_series = _calc_atr(df)

    dates = sorted(df.index.normalize().unique())
    rth_by_date: dict = {}
    for d in dates:
        day_bars = df[df.index.normalize() == d]
        if len(day_bars) > 0:
            rth_by_date[d] = day_bars

    date_list = sorted(rth_by_date.keys())
    daily: dict = {}

    for i, d in enumerate(date_list):
        if i == 0:
            continue  # need prior day

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

        # ATR at session open (last value before today's first bar)
        atr_at_open = float(atr_series.loc[:cur_bars.index[0]].iloc[-2]) if len(atr_series.loc[:cur_bars.index[0]]) >= 2 else np.nan
        if np.isnan(atr_at_open) or atr_at_open <= 0:
            # Fallback: use prior-day range as proxy
            atr_at_open = fib_range / 2.0

        daily[d] = {
            "pdh": pdh,
            "pdl": pdl,
            "fib_range": fib_range,
            "fib_prices": fib_prices,
            "trend": trend,
            "trend_price": trend_price,
            "today_open": today_open,
            "atr_at_open": atr_at_open,
            "cur_bars": cur_bars,
        }

    return daily


# ---------------------------------------------------------------------------
# Single-day simulation
# ---------------------------------------------------------------------------

def simulate_day(day_data: dict) -> Optional[dict]:
    """Return a trade dict or None if no signal fires."""
    fib_prices = day_data["fib_prices"]
    trend = day_data["trend"]
    atr_val = day_data["atr_at_open"]
    tolerance = TOUCH_TOL_ATR * atr_val
    cur_bars = day_data["cur_bars"]

    pt_dist = PT_ATR * atr_val
    sl_dist = SL_ATR * atr_val

    entry_price = entry_ts = direction = fib_triggered = fib_triggered_price = tp = sl = None

    # Scan morning session bars for first qualifying signal
    for ts, bar in cur_bars.iterrows():
        bar_minute = ts.hour * 60 + ts.minute
        if bar_minute < ENTRY_START_MINUTE:
            continue
        if bar_minute >= ENTRY_STOP_MINUTE:
            break

        for level_pct, level_price in sorted(fib_prices.items(),
                                              key=lambda kv: abs(kv[1] - float(bar["close"]))):
            if trend == 1:
                touched = float(bar["low"]) <= level_price + tolerance
                rejected = float(bar["close"]) > level_price
                if touched and rejected:
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
                    entry_price = float(bar["close"]) - SLIP
                    tp = entry_price - pt_dist
                    sl = entry_price + sl_dist
                    direction = "SHORT"
                    entry_ts = ts
                    fib_triggered = level_pct
                    fib_triggered_price = level_price
                    break

        if entry_ts is not None:
            break  # Found signal — stop scanning

    if entry_ts is None:
        return None  # No signal today

    # Simulate exit
    post_entry = cur_bars[cur_bars.index > entry_ts]
    exit_price = None
    exit_reason = None

    for ts2, bar2 in post_entry.iterrows():
        bar_minute2 = ts2.hour * 60 + ts2.minute

        # Time stop at noon
        if bar_minute2 >= ENTRY_STOP_MINUTE:
            exit_price = float(bar2["open"]) + (SLIP if direction == "SHORT" else -SLIP)
            exit_reason = "time"
            break

        if direction == "LONG":
            if float(bar2["low"]) <= sl:
                exit_price = sl
                exit_reason = "stop"
                break
            if float(bar2["high"]) >= tp:
                exit_price = tp
                exit_reason = "tp"
                break
        else:
            if float(bar2["high"]) >= sl:
                exit_price = sl
                exit_reason = "stop"
                break
            if float(bar2["low"]) <= tp:
                exit_price = tp
                exit_reason = "tp"
                break

    if exit_price is None:
        if len(post_entry) > 0:
            last_bar = post_entry.iloc[-1]
            exit_price = float(last_bar["close"]) + (SLIP if direction == "SHORT" else -SLIP)
            exit_reason = "eod"
        else:
            exit_price = entry_price
            exit_reason = "eod"

    pnl_pts = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
    commission_total = 2 * COMMISSION * N_CONTRACTS
    pnl_usd = pnl_pts * POINT_VALUE * N_CONTRACTS - commission_total

    return {
        "date": str(entry_ts.date()),
        "direction": direction,
        "fib_level_pct": round(fib_triggered, 3),
        "fib_level_price": round(fib_triggered_price, 2),
        "pdh": round(day_data["pdh"], 2),
        "pdl": round(day_data["pdl"], 2),
        "trend": "up" if trend == 1 else "down",
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2),
        "tp": round(tp, 2),
        "sl": round(sl, 2),
        "exit_reason": exit_reason,
        "atr": round(atr_val, 2),
        "pnl_pts": round(pnl_pts, 4),
        "pnl_usd": round(pnl_usd, 2),
    }


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(daily: dict) -> list[dict]:
    trades = []
    for d in sorted(daily.keys()):
        result = simulate_day(daily[d])
        if result is not None:
            trades.append(result)
    return trades


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"N": 0, "WR": 0.0, "AvgPnL": 0.0, "TotalPnL": 0.0,
                "Sharpe": 0.0, "MaxDD": 0.0, "ProfitFactor": 0.0}

    pnls = [t["pnl_usd"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    daily_pnl: dict[str, float] = {}
    for t in trades:
        daily_pnl[t["date"]] = daily_pnl.get(t["date"], 0.0) + t["pnl_usd"]
    dpnl = list(daily_pnl.values())
    mean_d = float(np.mean(dpnl)) if dpnl else 0.0
    std_d = float(np.std(dpnl, ddof=1)) if len(dpnl) > 1 else 1e-9
    sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 1e-9 else 0.0

    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(dd.min())

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")

    return {
        "N": len(trades),
        "WR": round(100.0 * len(wins) / len(trades), 1),
        "AvgPnL": round(float(np.mean(pnls)), 2),
        "TotalPnL": round(float(sum(pnls)), 2),
        "Sharpe": round(sharpe, 3),
        "MaxDD": round(max_dd, 2),
        "ProfitFactor": round(pf, 3),
    }


def exit_breakdown(trades: list[dict]) -> dict:
    out = {}
    for reason in ("tp", "stop", "time", "eod"):
        subset = [t for t in trades if t["exit_reason"] == reason]
        pnls = [t["pnl_usd"] for t in subset]
        wins = [p for p in pnls if p > 0]
        out[reason] = {
            "N": len(subset),
            "WR": round(100.0 * len(wins) / len(subset), 1) if subset else 0.0,
            "AvgPnL": round(float(np.mean(pnls)), 2) if pnls else 0.0,
        }
    return out


def direction_breakdown(trades: list[dict]) -> dict:
    out = {}
    for d in ("LONG", "SHORT"):
        subset = [t for t in trades if t["direction"] == d]
        pnls = [t["pnl_usd"] for t in subset]
        wins = [p for p in pnls if p > 0]
        out[d] = {
            "N": len(subset),
            "WR": round(100.0 * len(wins) / len(subset), 1) if subset else 0.0,
            "AvgPnL": round(float(np.mean(pnls)), 2) if pnls else 0.0,
            "TotalPnL": round(float(sum(pnls)), 2) if pnls else 0.0,
        }
    return out


def fib_level_breakdown(trades: list[dict]) -> list[dict]:
    rows = []
    for lvl in KEY_LEVELS:
        subset = [t for t in trades if abs(t["fib_level_pct"] - lvl) < 0.01]
        pnls = [t["pnl_usd"] for t in subset]
        wins = [p for p in pnls if p > 0]
        rows.append({
            "level": f"{lvl:.1%}",
            "N": len(subset),
            "WR": round(100.0 * len(wins) / len(subset), 1) if subset else 0.0,
            "AvgPnL": round(float(np.mean(pnls)), 2) if pnls else 0.0,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading MNQ 5-min bars...")
    df = load_bars()
    print(f"  Loaded {len(df):,} bars  {df.index[0]} → {df.index[-1]}")

    print("Building daily Fib levels...")
    daily = build_daily_data(df)
    print(f"  {len(daily)} trading days with prior-day data")

    print("Running backtest...")
    trades = run_backtest(daily)
    print(f"  {len(trades)} trades found")

    stats = compute_stats(trades)
    exits = exit_breakdown(trades)
    dirs = direction_breakdown(trades)
    fibs = fib_level_breakdown(trades)

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  IVAN TRADES FIB STRATEGY — MNQ 5-min  ({df.index[0].date()} to {df.index[-1].date()})")
    print(f"  Config: levels={[f'{l:.1%}' for l in KEY_LEVELS]}, "
          f"PT={PT_ATR}×ATR, SL={SL_ATR}×ATR, tol={TOUCH_TOL_ATR}×ATR")
    print(f"{'='*65}")
    print(f"  Trades      : {stats['N']}")
    print(f"  Win Rate    : {stats['WR']:.1f}%")
    print(f"  Avg PnL     : ${stats['AvgPnL']:.2f}")
    print(f"  Total PnL   : ${stats['TotalPnL']:.2f}")
    print(f"  Sharpe      : {stats['Sharpe']:.3f}")
    print(f"  Max DD      : ${stats['MaxDD']:.2f}")
    print(f"  Profit Factor: {stats['ProfitFactor']:.3f}")

    print(f"\n--- Exit Breakdown ---")
    print(f"  {'Reason':<8} {'N':>4}  {'WR%':>6}  {'AvgPnL':>8}")
    for reason, s in exits.items():
        print(f"  {reason:<8} {s['N']:>4}  {s['WR']:>5.1f}%  ${s['AvgPnL']:>7.2f}")

    print(f"\n--- Direction Breakdown ---")
    print(f"  {'Dir':<6} {'N':>4}  {'WR%':>6}  {'AvgPnL':>8}  {'TotalPnL':>10}")
    for d, s in dirs.items():
        print(f"  {d:<6} {s['N']:>4}  {s['WR']:>5.1f}%  ${s['AvgPnL']:>7.2f}  ${s['TotalPnL']:>9.2f}")

    print(f"\n--- Fib Level Breakdown ---")
    print(f"  {'Level':<8} {'N':>4}  {'WR%':>6}  {'AvgPnL':>8}")
    for row in fibs:
        print(f"  {row['level']:<8} {row['N']:>4}  {row['WR']:>5.1f}%  ${row['AvgPnL']:>7.2f}")

    print(f"\n--- Sample Trades (first 10) ---")
    print(f"  {'Date':<12} {'Dir':<6} {'Fib':>6} {'Trend':<6} {'Entry':>8} {'Exit':>8} {'Reason':<7} {'PnL':>8}")
    print("  " + "-" * 70)
    for t in trades[:10]:
        print(
            f"  {t['date']:<12} {t['direction']:<6} {t['fib_level_pct']:.3f}  "
            f"{t['trend']:<6} {t['entry_price']:>8.2f} {t['exit_price']:>8.2f} "
            f"{t['exit_reason']:<7} ${t['pnl_usd']:>7.2f}"
        )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    results = {
        "config": {
            "key_levels": KEY_LEVELS,
            "touch_tolerance_atr": TOUCH_TOL_ATR,
            "pt_atr": PT_ATR,
            "sl_atr": SL_ATR,
            "n_contracts": N_CONTRACTS,
            "entry_window": "09:35-12:00 ET",
            "data": str(MNQ_PATH.name),
            "date_range": f"{df.index[0].date()} to {df.index[-1].date()}",
        },
        "stats": stats,
        "exit_breakdown": exits,
        "direction_breakdown": dirs,
        "fib_level_breakdown": fibs,
        "trades": trades,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    passed = (
        stats["Sharpe"] >= 1.5
        and stats["MaxDD"] >= -1500
        and stats["WR"] >= 45
        and stats["N"] >= 20
    )
    verdict = "PASS" if passed else "NEEDS TUNING"
    print(f"  Gate check (Sharpe≥1.5, DD≥-$1500, WR≥45%, N≥20): {verdict}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
