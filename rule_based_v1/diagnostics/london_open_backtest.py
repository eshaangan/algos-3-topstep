"""London Open Breakout — MNQ ETH data, 2026 YTD.

Strategy:
  1. Build the Asian session range: 8pm ET prior night → 2am ET
  2. At London open (2am ET), look for breakout above/below Asian range
  3. Enter on first bar that closes outside the range with momentum
  4. Stop: opposite side of Asian range (+/- buffer)
  5. Target: PT_MULT × range width  OR  time stop at 5am ET

Edge: London open often has the largest 1-hour move of the 24h session.
Filters: min_range_atr (filter slow Asian sessions), volume confirmation.

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/london_open_backtest.py
    python rule_based_v1/diagnostics/london_open_backtest.py --pt 2.0 --sl 1.0 --sweep
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

ROOT  = Path(__file__).resolve().parent.parent.parent
RBV1  = ROOT / "rule_based_v1"
RESULTS_PATH = RBV1 / "diagnostics" / "london_open_results.json"

ETH_PATH = ROOT / "data" / "processed" / "mnq_aug2025_apr2026_1min_eth.h5"
ETH_PATH2 = ROOT / "data" / "processed" / "mnq_2026ytd_databento_1min_eth.h5"

# ---------------------------------------------------------------------------
POINT_VALUE = 2.0
TICK_SIZE   = 0.25
COMMISSION  = 0.62   # per side per contract
SLIPPAGE_TICKS = 1

def slip(price, direction, is_entry):
    return price + direction * SLIPPAGE_TICKS * TICK_SIZE * (1 if is_entry else -1)

def trade_pnl(entry, exit_, direction, n):
    return (exit_ - entry) * direction * n * POINT_VALUE - 2 * COMMISSION * n

def compute_atr(bars, period=14):
    prev = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"],
                    (bars["high"] - prev).abs(),
                    (bars["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

# ---------------------------------------------------------------------------
def load_eth_bars() -> pd.DataFrame:
    for path, key in [(ETH_PATH2, "bars_1min_eth"), (ETH_PATH, "bars_5min_eth")]:
        if path.exists():
            df = pd.read_hdf(str(path), key=key)
            if df.index.tz is None:
                df.index = df.index.tz_localize("US/Eastern")
            else:
                df.index = df.index.tz_convert("US/Eastern")
            # Resample to 5-min if 1-min
            if "1min" in path.name:
                df = df.resample("5min").agg(
                    open=("open","first"), high=("high","max"),
                    low=("low","min"), close=("close","last"), volume=("volume","sum")
                ).dropna(subset=["open"])
            # Filter to 2026
            df = df[df.index >= pd.Timestamp("2026-01-01", tz="US/Eastern")]
            print(f"Loaded {len(df):,} 5-min ETH bars [{df.index[0].date()} → {df.index[-1].date()}]")
            return df
    raise FileNotFoundError("No ETH data found")

# ---------------------------------------------------------------------------
def build_asian_ranges(bars: pd.DataFrame) -> dict:
    """For each calendar date, compute the Asian session range (prior 8pm-2am ET).

    Returns: {date: {"high": float, "low": float, "close": float, "n_bars": int}}
    """
    asian_ranges = {}
    dates = sorted(set(bars.index.date))

    for d in dates:
        # Asian session = 8pm of PRIOR day through 2am of this day
        try:
            session_start = pd.Timestamp(f"{d} 00:00:00", tz="US/Eastern") - pd.Timedelta(hours=4)
            session_end   = pd.Timestamp(f"{d} 02:00:00", tz="US/Eastern")
        except Exception:
            # DST transition — skip this day
            continue

        asian_bars = bars[(bars.index >= session_start) & (bars.index < session_end)]
        if len(asian_bars) < 5:
            continue

        asian_ranges[d] = {
            "high":   float(asian_bars["high"].max()),
            "low":    float(asian_bars["low"].min()),
            "open":   float(asian_bars["open"].iloc[0]),
            "close":  float(asian_bars["close"].iloc[-1]),
            "n_bars": len(asian_bars),
            "volume": float(asian_bars["volume"].sum()),
        }

    return asian_ranges

# ---------------------------------------------------------------------------
@dataclass
class Position:
    direction: int
    entry_price: float
    stop_loss: float
    profit_target: float
    time_stop_bar: int
    atr_at_entry: float
    range_width: float

def run_london_backtest(
    bars: pd.DataFrame,
    atr_s: pd.Series,
    asian_ranges: dict,
    n_contracts: int = 5,
    pt_mult: float = 1.5,
    sl_mult: float = 1.0,
    min_range_atr: float = 0.3,
    max_range_atr: float = 4.0,
    london_start: str = "02:00",
    london_end: str = "05:00",
    long_only: bool = False,
    max_daily_loss: float = -700.0,
    time_stop_bars: int = 24,      # ~2 hours of 5-min bars
) -> tuple[list, list, list, dict]:

    ls_time = pd.Timestamp(f"2000-01-01 {london_start}").time()
    le_time = pd.Timestamp(f"2000-01-01 {london_end}").time()

    pos: Optional[Position] = None
    trades = []
    equity = 50_000.0
    peak_equity = equity
    eq_vals, eq_times = [equity], [bars.index[0]]
    cur_date = None
    daily_pnl_run = 0.0
    daily_pnl_map = {}
    traded_today = False

    min_bars = 20

    for i in range(min_bars, len(bars)):
        bar  = bars.iloc[i]
        bt   = bars.index[i]
        bdate = bt.date()

        if cur_date is not None and bdate != cur_date:
            daily_pnl_map[cur_date] = daily_pnl_run
            daily_pnl_run = 0.0
            traded_today = False
        cur_date = bdate

        # Time-stop: force close at end of London session
        is_london_end = bt.time() >= le_time
        is_session_close = bt.time() >= pd.Timestamp("2000-01-01 05:30").time()

        if pos is not None:
            h, l, c = bar["high"], bar["low"], bar["close"]
            exited, ex_p, reason = False, 0.0, ""
            if is_session_close or i >= pos.time_stop_bar:
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
                p = trade_pnl(pos.entry_price, ex_p, pos.direction, n_contracts)
                trades.append({"pnl": p, "reason": reason, "direction": pos.direction,
                               "entry": pos.entry_price, "exit": ex_p, "date": str(bdate)})
                daily_pnl_run += p
                equity += p
                peak_equity = max(peak_equity, equity)
                eq_vals.append(equity)
                eq_times.append(bt)
                pos = None

        # Entry window: 2am - london_end
        bt_time = bt.time()
        if not (ls_time <= bt_time <= le_time):
            continue
        if pos is not None or traded_today:
            continue
        if daily_pnl_run <= max_daily_loss:
            continue
        if bdate not in asian_ranges:
            continue

        ar = asian_ranges[bdate]
        asian_high = ar["high"]
        asian_low  = ar["low"]
        asian_range = asian_high - asian_low

        atr_val = float(atr_s.iloc[i])
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        # Filter: range must be meaningful but not extreme
        if asian_range < min_range_atr * atr_val:
            continue
        if asian_range > max_range_atr * atr_val:
            continue

        close = float(bar["close"])
        prev_c = float(bars["close"].iloc[i-1])

        sig = None
        if close > asian_high and close > prev_c:  # Bullish breakout
            sig = 1
        elif not long_only and close < asian_low and close < prev_c:  # Bearish breakout
            sig = -1

        if sig is not None:
            ep = slip(close, sig, True)
            # Stop at opposite side of Asian range
            if sig == 1:
                sl_price = asian_low - sl_mult * TICK_SIZE * 4  # small buffer
            else:
                sl_price = asian_high + sl_mult * TICK_SIZE * 4
            pt_price = ep + sig * pt_mult * asian_range

            pos = Position(
                direction=sig, entry_price=ep, stop_loss=sl_price,
                profit_target=pt_price, time_stop_bar=i + time_stop_bars,
                atr_at_entry=atr_val, range_width=asian_range,
            )
            traded_today = True

    if cur_date and cur_date not in daily_pnl_map:
        daily_pnl_map[cur_date] = daily_pnl_run

    return trades, eq_vals, eq_times, daily_pnl_map

# ---------------------------------------------------------------------------
def print_results(trades, eq_vals, eq_times, daily_pnl_map, n_contracts, label="London Open"):
    if not trades:
        print(f"  [{label}] No trades.")
        return {}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total  = sum(t["pnl"] for t in trades)
    gp     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))

    eq_s   = pd.Series(eq_vals, index=eq_times)
    max_dd = float((eq_s - eq_s.cummax()).min())
    daily  = pd.Series(daily_pnl_map)
    active = daily[daily != 0]
    n_wks  = max(1, len(daily) / 5)
    sharpe = float(active.mean() / active.std() * np.sqrt(252)) if len(active) > 1 and active.std() > 0 else 0

    reasons = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    reason_pct = {k: round(v / len(trades) * 100, 1) for k, v in reasons.items()}

    long_t  = [t for t in trades if t["direction"] == 1]
    short_t = [t for t in trades if t["direction"] == -1]

    print(f"\n{'='*60}")
    print(f"  {label}  |  {n_contracts} contracts")
    print(f"{'='*60}")
    print(f"  Trades       : {len(trades)}  ({len(trades)/n_wks:.1f}/week)")
    print(f"  Win Rate     : {len(wins)/len(trades):.1%}  ({len(wins)}W / {len(losses)}L)")
    print(f"  Total PnL    : ${total:,.0f}  (${total/n_wks:,.0f}/week)")
    print(f"  Avg Win      : ${gp/len(wins):,.0f}" if wins else "")
    print(f"  Avg Loss     : ${-gl/len(losses):,.0f}" if losses else "")
    print(f"  Profit Factor: {gp/gl:.2f}" if gl > 0 else "  PF: ∞")
    print(f"  Sharpe       : {sharpe:.2f}")
    print(f"  Max Drawdown : ${max_dd:,.0f}")
    print(f"  Exit reasons : {reason_pct}")
    if long_t:
        lw = [t for t in long_t if t["pnl"] > 0]
        print(f"  LONG trades  : {len(long_t)}, WR={len(lw)/len(long_t):.1%}")
    if short_t:
        sw = [t for t in short_t if t["pnl"] > 0]
        print(f"  SHORT trades : {len(short_t)}, WR={len(sw)/len(short_t):.1%}")

    return {
        "n_trades": len(trades),
        "win_rate": round(len(wins)/len(trades), 3),
        "total_pnl": round(total, 2),
        "weekly_pnl": round(total/n_wks, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(gp/gl, 3) if gl > 0 else None,
    }

# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", type=int, default=5)
    parser.add_argument("--pt", type=float, default=1.5)
    parser.add_argument("--sl", type=float, default=1.0)
    parser.add_argument("--both", action="store_true", help="Allow SHORT breakouts too")
    parser.add_argument("--sweep", action="store_true", help="Sweep PT/SL/min_range combos")
    args = parser.parse_args()

    bars = load_eth_bars()
    atr_s = compute_atr(bars)
    asian_ranges = build_asian_ranges(bars)
    print(f"Asian ranges computed for {len(asian_ranges)} days")

    if args.sweep:
        grid = list(product(
            [1.0, 1.5, 2.0, 2.5],   # PT mult of range
            [0.5, 0.75, 1.0],        # SL mult (small — Asian range IS the stop)
            [0.2, 0.3, 0.5],         # min_range_atr
            [False, True],           # long_only vs both
        ))
        print(f"\nSweeping {len(grid)} configurations...\n")
        results = []
        for pt, sl, min_rng, both_sides in grid:
            trades, eqv, eqt, dpm = run_london_backtest(
                bars, atr_s, asian_ranges,
                n_contracts=args.contracts,
                pt_mult=pt, sl_mult=sl, min_range_atr=min_rng,
                long_only=not both_sides,
            )
            if not trades or len(trades) < 10:
                continue
            wins = [t for t in trades if t["pnl"] > 0]
            total = sum(t["pnl"] for t in trades)
            n_wks = max(1, len(dpm) / 5)
            gl = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
            gp = sum(t["pnl"] for t in wins)
            eq_s = pd.Series(eqv, index=eqt)
            max_dd = float((eq_s - eq_s.cummax()).min())
            daily = pd.Series(dpm)
            active = daily[daily != 0]
            sharpe = float(active.mean() / active.std() * np.sqrt(252)) if len(active) > 1 and active.std() > 0 else 0
            results.append({
                "pt": pt, "sl": sl, "min_rng": min_rng, "both": both_sides,
                "n": len(trades), "wr": round(len(wins)/len(trades), 3),
                "weekly_pnl": round(total/n_wks, 0),
                "max_dd": round(max_dd, 0),
                "sharpe": round(sharpe, 2),
                "pf": round(gp/gl, 2) if gl > 0 else 99,
            })

        results.sort(key=lambda x: x["sharpe"] * x["wr"], reverse=True)
        print(f"{'PT':>5} {'SL':>5} {'MinR':>6} {'Both':>5} {'N':>5} {'WR':>7} "
              f"{'$/wk':>8} {'MaxDD':>8} {'Sharpe':>8} {'PF':>6}")
        print(f"  {'-'*65}")
        for r in results[:20]:
            print(f"  {r['pt']:>4.1f} {r['sl']:>5.2f} {r['min_rng']:>6.2f} "
                  f"{'Y' if r['both'] else 'N':>5} {r['n']:>5} {r['wr']:>7.1%} "
                  f"{r['weekly_pnl']:>8,.0f} {r['max_dd']:>8,.0f} "
                  f"{r['sharpe']:>8.2f} {r['pf']:>6.2f}")
        with open(RESULTS_PATH, "w") as f:
            json.dump(results[:30], f, indent=2)
        print(f"\nSaved → {RESULTS_PATH}")
        return

    # Single run
    trades, eqv, eqt, dpm = run_london_backtest(
        bars, atr_s, asian_ranges,
        n_contracts=args.contracts,
        pt_mult=args.pt, sl_mult=args.sl,
        long_only=not args.both,
    )
    r = print_results(trades, eqv, eqt, dpm, args.contracts)

    # Show trades
    print(f"\n  Individual trades:")
    for t in trades:
        d = "L" if t["direction"] == 1 else "S"
        print(f"    {t['date']}  {d}  entry={t['entry']:.2f}  exit={t['exit']:.2f}  "
              f"pnl=${t['pnl']:,.0f}  [{t['reason']}]")

    with open(RESULTS_PATH, "w") as f:
        json.dump(r, f, indent=2)
    print(f"\nSaved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
