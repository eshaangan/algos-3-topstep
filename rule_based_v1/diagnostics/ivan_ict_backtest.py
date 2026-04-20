"""Ivan Trades + ICT Combined Strategy Backtest — MNQ 5-min.

Combines:
  IVAN: Prior 5-day swing Fibonacci retracements + BOC (Break of Candle) entry
  ICT:  Fair Value Gap confirmation + OTE zone (61.8-79%) + Market Structure filter
        + NY Kill Zone timing (08:30-11:00 ET)

SETUP (must satisfy ALL):
  1. Market structure bullish: recent swing shows HH+HL pattern (no downtrend entries)
  2. Price in OTE zone: 61.8% - 79% retracement of 5-day swing (tighter than Ivan alone)
  3. Signal candle touches OTE zone AND there is a Bullish FVG within ±ATR of the zone
     (FVG = bar[i-1].high < bar[i+1].low — institutional imbalance)
  4. NY Kill Zone: signal must fire 08:30-11:00 ET

ENTRY:
  BOC: Next bar breaks above signal candle high → LONG at signal candle high + slip

STOP / TARGET:
  Stop: Below signal candle low
  Target: ATR-based (3x ATR) or prior swing high — whichever closer

RUN:
    python rule_based_v1/diagnostics/ivan_ict_backtest.py
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
RESULTS_PATH = DIAG_DIR / "ivan_ict_results.json"

for p in [str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

POINT_VALUE = 2.0
SLIP = 0.25
COMMISSION = 0.62
N_CONTRACTS = 2
ATR_PERIOD = 14

# Ivan: 5-day swing lookback
SWING_DAYS = 5

# ICT OTE zone (Optimal Trade Entry): 61.8% - 79% retracement
OTE_LOW = 0.618
OTE_HIGH = 0.786      # 78.6% — ICT golden pocket upper bound

# ICT Kill Zone: NY open 08:30 – 11:00 ET
KILL_ZONE_START = 8 * 60 + 30
KILL_ZONE_END = 11 * 60

# Exit parameters
PT_ATR = 3.0
SL_ATR = 1.5
SL_BUFFER_ATR = 0.05

# Touch tolerance (ATR multiples)
TOUCH_TOL_ATR = 0.40

# Market structure: look back N swing points
MS_LOOKBACK_BARS = 50  # bars to scan for HH/HL pattern


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_combined() -> pd.DataFrame:
    """Load Aug 2025 – Apr 2026 combined MNQ 5-min RTH bars."""
    frames = []
    for fname in ["mnq_5min_aug25_mar26.h5", "mnq_2026ytd_databento_5min_rth.h5"]:
        path = DATA_DIR / fname
        if not path.exists():
            continue
        with pd.HDFStore(str(path), "r") as s:
            df = s["/bars_5min"].copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("US/Eastern")
        else:
            df.index = df.index.tz_convert("US/Eastern")
        frames.append(df.sort_index())
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def calc_atr(bars: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    pc = bars["close"].shift(1)
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - pc).abs(),
        (bars["low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# ICT: Fair Value Gap detection
# ---------------------------------------------------------------------------

def find_bullish_fvgs(bars: pd.DataFrame, lookback: int = 20) -> list[dict]:
    """Return list of unfilled bullish FVGs in the recent `lookback` bars.

    Bullish FVG: bars[i-1].high < bars[i+1].low
    Zone: [bars[i-1].high, bars[i+1].low]
    """
    fvgs = []
    recent = bars.iloc[-lookback - 2:]
    for i in range(1, len(recent) - 1):
        prev_high = float(recent["high"].iloc[i - 1])
        next_low = float(recent["low"].iloc[i + 1])
        if prev_high < next_low:
            # Displacement candle (middle) should be bullish
            if float(recent["close"].iloc[i]) > float(recent["open"].iloc[i]):
                fvgs.append({
                    "bottom": prev_high,
                    "top": next_low,
                    "ts": recent.index[i],
                })
    return fvgs


# ---------------------------------------------------------------------------
# ICT: Market structure — bullish requires HH + HL
# ---------------------------------------------------------------------------

def find_swing_highs_lows(bars: pd.DataFrame, n: int = 2) -> tuple[list, list]:
    """Identify swing highs and lows using n-bar pivot rule.

    Swing high: bar[i].high > max(bars[i-n:i] + bars[i+1:i+n+1])
    Swing low:  bar[i].low  < min(bars[i-n:i] + bars[i+1:i+n+1])
    """
    highs, lows = [], []
    for i in range(n, len(bars) - n):
        h = float(bars["high"].iloc[i])
        l = float(bars["low"].iloc[i])
        left_h = bars["high"].iloc[i - n:i].max()
        right_h = bars["high"].iloc[i + 1:i + n + 1].max()
        left_l = bars["low"].iloc[i - n:i].min()
        right_l = bars["low"].iloc[i + 1:i + n + 1].min()
        if h > left_h and h > right_h:
            highs.append((bars.index[i], h))
        if l < left_l and l < right_l:
            lows.append((bars.index[i], l))
    return highs, lows


def is_bullish_structure(bars: pd.DataFrame, lookback: int = MS_LOOKBACK_BARS) -> bool:
    """Return True if recent market structure shows HH + HL (bullish).

    Requires at least 2 swing highs and 2 swing lows, with the most recent
    swing high > previous swing high AND most recent swing low > previous swing low.
    """
    recent = bars.iloc[-lookback:]
    highs, lows = find_swing_highs_lows(recent, n=3)
    if len(highs) < 2 or len(lows) < 2:
        return True  # insufficient data — don't filter out

    # Higher High: most recent swing high > prior swing high
    hh = highs[-1][1] > highs[-2][1]
    # Higher Low: most recent swing low > prior swing low
    hl = lows[-1][1] > lows[-2][1]
    return hh and hl


# ---------------------------------------------------------------------------
# Daily simulation
# ---------------------------------------------------------------------------

def simulate_day(
    day_data: dict,
    all_bars: pd.DataFrame,
    atr_series: pd.Series,
) -> Optional[dict]:
    cur_bars = day_data["cur_bars"]
    fib_prices = day_data["fib_prices"]
    swing_low = day_data["swing_low"]
    swing_high = day_data["swing_high"]
    fib_range = day_data["fib_range"]

    # ATR at session open
    atr_slice = atr_series.loc[:cur_bars.index[0]]
    atr_val = float(atr_slice.iloc[-2]) if len(atr_slice) >= 2 else fib_range / 3.0
    if np.isnan(atr_val) or atr_val <= 0:
        atr_val = fib_range / 3.0

    tol = TOUCH_TOL_ATR * atr_val
    pt_dist = PT_ATR * atr_val
    sl_dist = SL_ATR * atr_val
    sl_buf = SL_BUFFER_ATR * atr_val

    # ICT: check market structure using bars up to session open
    pre_session = all_bars.loc[:cur_bars.index[0]].iloc[:-1]
    if len(pre_session) >= MS_LOOKBACK_BARS:
        if not is_bullish_structure(pre_session):
            return None  # bearish structure — skip LONG

    # ICT OTE: only trade the 61.8%–79% zone
    ote_low_price = swing_low + OTE_LOW * fib_range
    ote_high_price = swing_low + OTE_HIGH * fib_range

    signal_candle = None
    entry_price = entry_ts = fib_trig = fib_trig_price = tp = stop = None
    fvg_confirmed = False

    for ts, bar in cur_bars.iterrows():
        bm = ts.hour * 60 + ts.minute
        if bm < KILL_ZONE_START:
            continue
        if bm >= KILL_ZONE_END:
            break

        # Phase 2: We have a signal candle — look for BOC
        if signal_candle is not None:
            sc = signal_candle
            if float(bar["high"]) > sc["high"]:
                entry_price = sc["high"] + SLIP
                sd = min(entry_price - (sc["low"] - sl_buf), sl_dist)
                stop = entry_price - sd
                tp = min(entry_price + pt_dist, swing_high)  # cap at swing high
                entry_ts = ts
                fib_trig = sc["lvl"]
                fib_trig_price = sc["lvl_price"]
                fvg_confirmed = sc["fvg"]
                break
            elif float(bar["low"]) < sc["low"] - sl_buf:
                signal_candle = None

        # Phase 1: Look for signal candle in OTE zone with FVG confluence
        if signal_candle is None:
            # Only consider bars touching the OTE zone
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            for lvl, lvl_price in sorted(fib_prices.items(),
                                          key=lambda kv: abs(kv[1] - bar_close)):
                # Must be in the OTE zone (61.8-79%)
                if lvl < OTE_LOW - 0.01 or lvl > OTE_HIGH + 0.01:
                    continue

                touched = bar_low <= lvl_price + tol
                near = bar_close >= lvl_price - tol
                if not (touched and near):
                    continue

                # ICT FVG check: look for a bullish FVG near the OTE level in recent bars
                bars_up_to_signal = all_bars.loc[:ts]
                fvgs = find_bullish_fvgs(bars_up_to_signal, lookback=30)
                fvg_near = any(
                    f["bottom"] >= lvl_price - 2 * atr_val
                    and f["top"] <= lvl_price + 2 * atr_val
                    for f in fvgs
                )

                signal_candle = {
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "lvl": lvl,
                    "lvl_price": lvl_price,
                    "fvg": fvg_near,
                }
                break

        if entry_ts is not None:
            break

    if entry_ts is None:
        return None

    # Exit simulation
    post = cur_bars[cur_bars.index > entry_ts]
    exit_price = exit_reason = None
    for ts2, bar2 in post.iterrows():
        bm2 = ts2.hour * 60 + ts2.minute
        if bm2 >= KILL_ZONE_END:
            exit_price = float(bar2["open"]) - SLIP
            exit_reason = "time"
            break
        if float(bar2["low"]) <= stop:
            exit_price = stop; exit_reason = "stop"; break
        if float(bar2["high"]) >= tp:
            exit_price = tp; exit_reason = "tp"; break

    if exit_price is None:
        if len(post) > 0:
            exit_price = float(post.iloc[-1]["close"]) - SLIP
            exit_reason = "eod"
        else:
            exit_price = entry_price; exit_reason = "eod"

    pnl = (exit_price - entry_price) * POINT_VALUE * N_CONTRACTS - 2 * COMMISSION * N_CONTRACTS

    return {
        "date": str(entry_ts.date()),
        "direction": "LONG",
        "fib_level_pct": round(fib_trig, 3),
        "fvg_confirmed": fvg_confirmed,
        "entry": round(entry_price, 2),
        "exit": round(exit_price, 2),
        "tp": round(tp, 2),
        "sl": round(stop, 2),
        "reason": exit_reason,
        "atr": round(atr_val, 2),
        "pnl": round(pnl, 2),
    }


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, atr_series: pd.Series, target_year: Optional[int] = None) -> list[dict]:
    dates = sorted(df.index.normalize().unique())
    rth_by_date = {d: df[df.index.normalize() == d] for d in dates}
    date_list = sorted(rth_by_date.keys())

    trades = []
    for i, d in enumerate(date_list):
        if target_year and d.year != target_year:
            continue
        if i < SWING_DAYS:
            continue

        # Build swing from prior days
        prior = [rth_by_date[date_list[i - k - 1]]
                 for k in range(SWING_DAYS)
                 if date_list[i - k - 1] in rth_by_date]
        if not prior:
            continue
        swing_bars = pd.concat(prior).sort_index()
        sh = float(swing_bars["high"].max())
        sl = float(swing_bars["low"].min())
        fib_range = sh - sl
        if fib_range < 1e-6:
            continue

        # Swing direction filter: must be a bullish swing (low before high)
        lp = swing_bars["low"].idxmin()
        hp = swing_bars["high"].idxmax()
        if lp >= hp:  # bearish swing — skip
            continue

        # OTE levels only (61.8, 70.5, 78.6%)
        fib_prices = {lvl: sl + lvl * fib_range
                      for lvl in [0.618, 0.705, 0.786]}

        cur_bars = rth_by_date.get(d)
        if cur_bars is None or len(cur_bars) < 2:
            continue

        day_data = {
            "cur_bars": cur_bars,
            "fib_prices": fib_prices,
            "swing_low": sl,
            "swing_high": sh,
            "fib_range": fib_range,
        }

        result = simulate_day(day_data, df, atr_series)
        if result:
            trades.append(result)

    return trades


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def stats(trades: list[dict]) -> dict:
    if not trades:
        return {"N": 0, "WR": 0, "AvgPnL": 0, "TotalPnL": 0, "Sharpe": 0, "MaxDD": 0}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    dpnl: dict = {}
    for t in trades:
        dpnl[t["date"]] = dpnl.get(t["date"], 0) + t["pnl"]
    dv = list(dpnl.values())
    sh = float(np.mean(dv) / np.std(dv, ddof=1) * np.sqrt(252)) if len(dv) > 1 and np.std(dv, ddof=1) > 0 else 0
    eq = np.cumsum(pnls)
    mdd = float((eq - np.maximum.accumulate(eq)).min())
    return {
        "N": len(trades),
        "WR": round(100 * len(wins) / len(trades), 1),
        "AvgPnL": round(float(np.mean(pnls)), 2),
        "TotalPnL": round(float(sum(pnls)), 2),
        "Sharpe": round(sh, 3),
        "MaxDD": round(mdd, 2),
    }


def print_results(label: str, trades: list[dict]):
    s = stats(trades)
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  N={s['N']}, WR={s['WR']}%, AvgPnL=${s['AvgPnL']:.2f}, "
          f"TotalPnL=${s['TotalPnL']:.2f}")
    print(f"  Sharpe={s['Sharpe']:.3f}, MaxDD=${s['MaxDD']:.2f}")

    exits = {}
    for r in ("tp", "stop", "time", "eod"):
        ss = [t for t in trades if t["reason"] == r]
        ps = [t["pnl"] for t in ss]
        ws = [p for p in ps if p > 0]
        exits[r] = f"N={len(ss)} WR={100*len(ws)/len(ss):.0f}% avg=${np.mean(ps):.0f}" if ss else "N=0"
    for r, v in exits.items():
        print(f"    {r}: {v}")

    fvg_trades = [t for t in trades if t.get("fvg_confirmed")]
    no_fvg = [t for t in trades if not t.get("fvg_confirmed")]
    if fvg_trades:
        fp = [t["pnl"] for t in fvg_trades]
        print(f"  FVG-confirmed: N={len(fvg_trades)}, WR={100*sum(1 for p in fp if p>0)/len(fp):.0f}%, avg=${np.mean(fp):.0f}")
    if no_fvg:
        np_ = [t["pnl"] for t in no_fvg]
        print(f"  No FVG:        N={len(no_fvg)}, WR={100*sum(1 for p in np_ if p>0)/len(no_fvg):.0f}%, avg=${np.mean(np_):.0f}")

    if trades:
        print(f"\n  {'Date':<12} {'Fib':>6} {'FVG':>4} {'Entry':>9} {'Exit':>9} {'Reason':<7} {'PnL':>8}")
        print("  " + "-" * 62)
        for t in trades:
            print(f"  {t['date']:<12} {t['fib_level_pct']:.3f}  {'Y' if t.get('fvg_confirmed') else 'N':>3}  "
                  f"{t['entry']:>9.2f} {t['exit']:>9.2f} {t['reason']:<7} ${t['pnl']:>7.2f}")

    gate = s["Sharpe"] >= 1.5 and s["MaxDD"] >= -1500 and s["WR"] >= 45 and s["N"] >= 10
    print(f"\n  Gate (Sharpe≥1.5, DD≥-$1500, WR≥45%, N≥10): {'PASS ✓' if gate else 'FAIL'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading MNQ bars (Aug 2025 – Apr 2026)...")
    df = load_combined()
    print(f"  {len(df):,} bars  {df.index[0].date()} → {df.index[-1].date()}")

    atr_series = calc_atr(df)

    # Training period: Aug 2025 – Dec 2025
    print("\nRunning Aug–Dec 2025 (in-sample)...")
    is_trades = [t for t in run_backtest(df, atr_series) if t["date"] < "2026-01-01"]
    print_results("IVAN+ICT — Aug–Dec 2025 IN-SAMPLE", is_trades)

    # 2026 YTD OOS
    print("\nRunning 2026 YTD (out-of-sample)...")
    oos_trades = run_backtest(df, atr_series, target_year=2026)
    print_results("IVAN+ICT — 2026 YTD OOS", oos_trades)

    # Combined full period
    all_trades = run_backtest(df, atr_series)
    print_results("IVAN+ICT — FULL PERIOD (Aug 2025–Apr 2026)", all_trades)

    # Comparison: Ivan BOC (from prior results) vs Ivan+ICT
    print(f"\n{'='*65}")
    print(f"  COMPARISON: Ivan BOC alone vs Ivan+ICT")
    print(f"{'='*65}")
    print(f"  {'Metric':<20} {'Ivan BOC (full)':>20} {'Ivan+ICT (full)':>20}")
    print(f"  {'-'*60}")
    ivan_boc = {"N": 39, "WR": 53.8, "Sharpe": 4.78, "MaxDD": -539, "TotalPnL": 2797}
    s = stats(all_trades)
    for k in ("N", "WR", "Sharpe", "MaxDD", "TotalPnL"):
        b_val = ivan_boc[k]
        i_val = s.get(k, 0)
        print(f"  {k:<20} {str(b_val):>20} {str(i_val):>20}")

    # Save
    results = {
        "in_sample": {"trades": is_trades, "stats": stats(is_trades)},
        "oos_2026": {"trades": oos_trades, "stats": stats(oos_trades)},
        "full": {"trades": all_trades, "stats": stats(all_trades)},
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
