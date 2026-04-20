"""Ivan Trades — Break of Candle (BOC) Fibonacci Backtest.

Corrected strategy based on public research:

SETUP:
  1. Swing detection: find the most significant swing high/low in the prior N days
     (NOT just prior day — Ivan draws from meaningful institutional swings)
  2. Fibonacci retracements from swing low to swing high
  3. Daily bias: today's open above 50% Fib → uptrend, below → downtrend

ENTRY (Break of Candle):
  1. Price touches a key Fib level (38.2%, 50%, 61.8%) — this is the "signal candle"
  2. NEXT bar breaks ABOVE signal candle high → LONG entry
  3. NEXT bar breaks BELOW signal candle low → SHORT entry

STOP LOSS:
  - Just below the signal candle low (LONG) or above signal candle high (SHORT)
  - Plus a small ATR buffer

TARGET:
  - Fibonacci extension: 127.2% or 161.8% of the swing move
  - OR: prior swing high (LONG) / prior swing low (SHORT) as a cap

RUN:
    python rule_based_v1/diagnostics/ivan_boc_backtest.py
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
RESULTS_PATH = DIAG_DIR / "ivan_boc_results.json"

for p in [str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

POINT_VALUE = 2.0
SLIP = 0.25
COMMISSION = 0.62
N_CONTRACTS = 2
KEY_LEVELS = [0.382, 0.500, 0.618]
ATR_PERIOD = 14
ENTRY_START_MINUTE = 9 * 60 + 35
ENTRY_STOP_MINUTE = 12 * 60


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


def _calc_atr(bars: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = bars["close"].shift(1)
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Swing detection
# ---------------------------------------------------------------------------

def find_swing(bars: pd.DataFrame, lookback_days: int = 3) -> tuple[float, float]:
    """Find the most significant swing high and low over the prior N days.

    Returns (swing_low, swing_high).
    """
    # Use the actual high/low extremes over the lookback period
    swing_high = float(bars["high"].max())
    swing_low = float(bars["low"].min())
    return swing_low, swing_high


# ---------------------------------------------------------------------------
# Daily data builder
# ---------------------------------------------------------------------------

def build_daily_data(
    df: pd.DataFrame,
    swing_lookback_days: int = 5,
) -> dict:
    atr_series = _calc_atr(df)
    dates = sorted(df.index.normalize().unique())
    rth_by_date: dict = {d: df[df.index.normalize() == d]
                         for d in dates if len(df[df.index.normalize() == d]) > 0}
    date_list = sorted(rth_by_date.keys())
    daily: dict = {}

    for i, d in enumerate(date_list):
        if i < swing_lookback_days:
            continue

        # Prior N days for swing identification
        prior_dates = date_list[max(0, i - swing_lookback_days):i]
        prior_bars_list = [rth_by_date[pd_] for pd_ in prior_dates if pd_ in rth_by_date]
        if not prior_bars_list:
            continue
        prior_bars = pd.concat(prior_bars_list).sort_index()

        swing_low, swing_high = find_swing(prior_bars, swing_lookback_days)
        fib_range = swing_high - swing_low
        if fib_range < 1e-6:
            continue

        fib_prices = {lvl: swing_low + lvl * fib_range for lvl in KEY_LEVELS}

        # Fibonacci extensions (targets)
        # 127.2% and 161.8% of the fib move beyond the swing high (for LONG targets)
        ext_127 = swing_high + 0.272 * fib_range   # = swing_low + 1.272 * fib_range
        ext_162 = swing_high + 0.618 * fib_range   # = swing_low + 1.618 * fib_range
        # Mirror for SHORT targets (below swing_low)
        ext_127_short = swing_low - 0.272 * fib_range
        ext_162_short = swing_low - 0.618 * fib_range

        cur_bars = rth_by_date.get(d)
        if cur_bars is None or len(cur_bars) < 2:
            continue

        today_open = float(cur_bars["open"].iloc[0])
        trend_price = swing_low + 0.500 * fib_range
        trend = 1 if today_open > trend_price else -1

        # ATR at session open
        atr_slice = atr_series.loc[:cur_bars.index[0]]
        atr_val = float(atr_slice.iloc[-2]) if len(atr_slice) >= 2 else fib_range / 3.0
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = fib_range / 3.0

        daily[d] = {
            "swing_low": swing_low,
            "swing_high": swing_high,
            "fib_range": fib_range,
            "fib_prices": fib_prices,
            "ext_127": ext_127,
            "ext_162": ext_162,
            "ext_127_short": ext_127_short,
            "ext_162_short": ext_162_short,
            "trend": trend,
            "today_open": today_open,
            "atr_val": atr_val,
            "cur_bars": cur_bars,
        }

    return daily


# ---------------------------------------------------------------------------
# Break of Candle simulation
# ---------------------------------------------------------------------------

def simulate_day_boc(
    day_data: dict,
    touch_tolerance_atr: float = 0.25,
    sl_buffer_atr: float = 0.1,
    use_extension_target: bool = True,    # True=fib extension, False=ATR-based
    pt_atr: float = 3.0,                  # used only if use_extension_target=False
    sl_atr: float = 1.5,                  # also used for max cap
    long_only: bool = True,
    target_ext: float = 1.272,            # 127.2% or 161.8%
    min_signal_candle_body_atr: float = 0.0,  # signal candle body size filter
) -> Optional[dict]:
    """Simulate one trading day using the Break of Candle entry."""
    fib_prices = day_data["fib_prices"]
    trend = day_data["trend"]
    atr_val = day_data["atr_val"]
    cur_bars = day_data["cur_bars"]

    if long_only and trend == -1:
        return None

    tolerance = touch_tolerance_atr * atr_val
    sl_buffer = sl_buffer_atr * atr_val
    max_sl_dist = sl_atr * atr_val

    # Extension target price
    if target_ext >= 1.6:
        ext_long_tp = day_data["ext_162"]
        ext_short_tp = day_data["ext_162_short"]
    else:
        ext_long_tp = day_data["ext_127"]
        ext_short_tp = day_data["ext_127_short"]

    # State machine: look for signal candle → then BOC entry
    signal_candle: Optional[dict] = None
    entry_price = entry_ts = direction = fib_triggered = signal_level_price = None
    tp = sl = None

    bar_list = list(cur_bars.iterrows())

    for idx, (ts, bar) in enumerate(bar_list):
        bar_minute = ts.hour * 60 + ts.minute
        if bar_minute >= ENTRY_STOP_MINUTE:
            break

        if bar_minute < ENTRY_START_MINUTE:
            continue

        # ----------------------------------------------------------------
        # Phase 2: We have a signal candle — look for BOC entry
        # ----------------------------------------------------------------
        if signal_candle is not None:
            sc = signal_candle
            if sc["direction"] == 1:  # LONG — wait for break above signal candle high
                if float(bar["high"]) > sc["high"]:
                    entry_price = sc["high"] + SLIP
                    sl_raw = sc["low"] - sl_buffer
                    sl_dist = min(entry_price - sl_raw, max_sl_dist)
                    sl = entry_price - sl_dist

                    if use_extension_target:
                        tp = ext_long_tp
                        # Sanity check: TP must be above entry with at least 1R
                        if tp <= entry_price + sl_dist:
                            tp = entry_price + pt_atr * atr_val
                    else:
                        tp = entry_price + pt_atr * atr_val

                    direction = "LONG"
                    entry_ts = ts
                    fib_triggered = sc["fib_level_pct"]
                    signal_level_price = sc["fib_level_price"]
                    break
                elif float(bar["low"]) < sc["low"] - sl_buffer:
                    # Signal candle invalidated (price fell through it)
                    signal_candle = None
                    # Fall through to look for new signal below

            else:  # SHORT — wait for break below signal candle low
                if float(bar["low"]) < sc["low"]:
                    entry_price = sc["low"] - SLIP
                    sl_raw = sc["high"] + sl_buffer
                    sl_dist = min(sl_raw - entry_price, max_sl_dist)
                    sl = entry_price + sl_dist

                    if use_extension_target:
                        tp = ext_short_tp
                        if tp >= entry_price - sl_dist:
                            tp = entry_price - pt_atr * atr_val
                    else:
                        tp = entry_price - pt_atr * atr_val

                    direction = "SHORT"
                    entry_ts = ts
                    fib_triggered = sc["fib_level_pct"]
                    signal_level_price = sc["fib_level_price"]
                    break
                elif float(bar["high"]) > sc["high"] + sl_buffer:
                    signal_candle = None

        # ----------------------------------------------------------------
        # Phase 1: Look for signal candle (bar touching Fib level)
        # ----------------------------------------------------------------
        if signal_candle is None:
            bar_body = abs(float(bar["close"]) - float(bar["open"]))

            for level_pct, level_price in sorted(fib_prices.items(),
                                                  key=lambda kv: abs(kv[1] - float(bar["close"]))):
                if trend == 1:
                    # LONG: bar touches level from above (pullback to level)
                    touches_level = float(bar["low"]) <= level_price + tolerance
                    above_level = float(bar["close"]) > level_price - tolerance
                    body_ok = bar_body >= min_signal_candle_body_atr * atr_val
                    if touches_level and above_level and body_ok:
                        signal_candle = {
                            "direction": 1,
                            "high": float(bar["high"]),
                            "low": float(bar["low"]),
                            "close": float(bar["close"]),
                            "fib_level_pct": level_pct,
                            "fib_level_price": level_price,
                            "ts": ts,
                        }
                        break
                else:
                    # SHORT: bar touches level from below (rally to level)
                    touches_level = float(bar["high"]) >= level_price - tolerance
                    below_level = float(bar["close"]) < level_price + tolerance
                    body_ok = bar_body >= min_signal_candle_body_atr * atr_val
                    if touches_level and below_level and body_ok:
                        signal_candle = {
                            "direction": -1,
                            "high": float(bar["high"]),
                            "low": float(bar["low"]),
                            "close": float(bar["close"]),
                            "fib_level_pct": level_pct,
                            "fib_level_price": level_price,
                            "ts": ts,
                        }
                        break

    if entry_ts is None:
        return None

    # -----------------------------------------------------------------------
    # Simulate exit
    # -----------------------------------------------------------------------
    post_entry = cur_bars[cur_bars.index > entry_ts]
    exit_price = None
    exit_reason = None

    for ts2, bar2 in post_entry.iterrows():
        bar_minute2 = ts2.hour * 60 + ts2.minute
        if bar_minute2 >= ENTRY_STOP_MINUTE:
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
        "date": str(entry_ts.date()),
        "direction": direction,
        "fib_level_pct": round(fib_triggered, 3),
        "fib_level_price": round(signal_level_price, 2),
        "swing_low": round(day_data["swing_low"], 2),
        "swing_high": round(day_data["swing_high"], 2),
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
# Backtest runner & stats
# ---------------------------------------------------------------------------

def run_backtest(daily: dict, **kwargs) -> list[dict]:
    trades = []
    for d in sorted(daily.keys()):
        result = simulate_day_boc(daily[d], **kwargs)
        if result:
            trades.append(result)
    return trades


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
    max_dd = float((equity - peak).min())

    gp = sum(wins)
    gl = abs(sum(losses)) if losses else 1e-9
    pf = gp / gl if gl > 1e-9 else float("inf")

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


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def run_sweep(daily: dict) -> list[dict]:
    tolerances = [0.15, 0.25, 0.40]
    sl_buffers = [0.05, 0.15]
    long_onlys = [True, False]
    use_ext = [True, False]
    swing_lookbacks = [3, 5, 10]    # pre-built
    target_exts = [1.272, 1.618]

    rows = []
    for tol, slb, lo, ext_flag, te in itertools.product(
        tolerances, sl_buffers, long_onlys, use_ext, target_exts
    ):
        trades = run_backtest(daily, touch_tolerance_atr=tol, sl_buffer_atr=slb,
                              long_only=lo, use_extension_target=ext_flag,
                              target_ext=te)
        stats = compute_stats(trades)
        rows.append({
            "tol": tol, "sl_buf": slb, "long_only": lo,
            "use_ext": ext_flag, "target_ext": te, **stats
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading MNQ 5-min bars...")
    df = load_bars()
    print(f"  {len(df):,} bars  {df.index[0].date()} → {df.index[-1].date()}")

    all_results = {}

    for swing_days in [3, 5, 10]:
        print(f"\nBuilding daily data (swing_lookback={swing_days} days)...")
        daily = build_daily_data(df, swing_lookback_days=swing_days)
        print(f"  {len(daily)} days")

        print(f"  Running parameter sweep...")
        rows = run_sweep(daily)

        qualified = [r for r in rows if r["N"] >= 15]
        qualified.sort(key=lambda x: x["Sharpe"], reverse=True)

        print(f"\n  TOP 10 — swing_lookback={swing_days}d")
        print(f"  {'tol':>4} {'slb':>4} {'lo':>4} {'ext':>4} {'te':>5}  "
              f"{'N':>4}  {'WR%':>6}  {'AvgPnL':>8}  {'Sharpe':>7}  {'MaxDD':>9}  {'TotalPnL':>10}")
        print("  " + "-" * 80)
        for r in qualified[:10]:
            print(
                f"  {r['tol']:>4.2f} {r['sl_buf']:>4.2f} {str(r['long_only']):>4} "
                f"{str(r['use_ext']):>4} {r['target_ext']:>5.3f}  "
                f"{r['N']:>4}  {r['WR']:>5.1f}%  "
                f"${r['AvgPnL']:>7.2f}  {r['Sharpe']:>7.3f}  "
                f"${r['MaxDD']:>8.2f}  ${r['TotalPnL']:>9.2f}"
            )

        if qualified:
            best = qualified[0]
            all_results[f"swing_{swing_days}d"] = {
                "best": best,
                "top20": qualified[:20],
                "all": rows,
            }

    # -----------------------------------------------------------------------
    # Run best overall config with full trade output
    # -----------------------------------------------------------------------
    best_swing = 5
    print(f"\n{'='*70}")
    print(f"  BEST CONFIG DETAILED RUN — swing={best_swing}d")
    print(f"{'='*70}")

    daily5 = build_daily_data(df, swing_lookback_days=best_swing)
    best_key = f"swing_{best_swing}d"

    if best_key in all_results and all_results[best_key]["top20"]:
        best_cfg = all_results[best_key]["top20"][0]
        trades = run_backtest(
            daily5,
            touch_tolerance_atr=best_cfg["tol"],
            sl_buffer_atr=best_cfg["sl_buf"],
            long_only=best_cfg["long_only"],
            use_extension_target=best_cfg["use_ext"],
            target_ext=best_cfg["target_ext"],
        )
        stats = compute_stats(trades)
        exits = exit_breakdown(trades)

        print(f"  Config: tol={best_cfg['tol']}, slb={best_cfg['sl_buf']}, "
              f"lo={best_cfg['long_only']}, ext={best_cfg['use_ext']}, te={best_cfg['target_ext']}")
        print(f"  N={stats['N']}, WR={stats['WR']}%, Sharpe={stats['Sharpe']:.3f}, "
              f"MaxDD=${stats['MaxDD']:.2f}, TotalPnL=${stats['TotalPnL']:.2f}")

        print(f"\n  Exit breakdown:")
        for reason, s in exits.items():
            print(f"    {reason}: N={s['N']}, WR={s['WR']}%, AvgPnL=${s['AvgPnL']:.2f}")

        print(f"\n  Sample trades:")
        print(f"  {'Date':<12} {'Dir':<6} {'Fib':>5} {'Entry':>8} {'Exit':>8} {'Reason':<7} {'PnL':>8}")
        print("  " + "-" * 65)
        for t in trades[:15]:
            print(f"  {t['date']:<12} {t['direction']:<6} {t['fib_level_pct']:.3f}  "
                  f"{t['entry_price']:>8.2f} {t['exit_price']:>8.2f} "
                  f"{t['exit_reason']:<7} ${t['pnl_usd']:>7.2f}")

        # Gate check
        passed = (stats["Sharpe"] >= 1.5 and stats["MaxDD"] >= -1500
                  and stats["WR"] >= 45 and stats["N"] >= 20)
        print(f"\n  Gate (Sharpe≥1.5, DD≥-$1500, WR≥45%, N≥20): {'PASS' if passed else 'NEEDS TUNING'}")

        all_results["best_run"] = {"config": best_cfg, "stats": stats, "exits": exits, "trades": trades}
    else:
        print("  No qualifying configs found.")

    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {k: v for k, v in all_results.items() if k != "best_run"},
            f, indent=2, default=str
        )
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
