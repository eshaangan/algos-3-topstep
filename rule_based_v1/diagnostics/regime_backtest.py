"""ORB Regime Backtest — Opening Range Window Comparison with 2 Trades/Day Cap.

Tests four OR window "regimes" on MNQ 5-min data (Aug 2025 – Feb 2026):

  Regime A: 15-min OR  (09:30–09:44, entry from 09:45)
  Regime B: 30-min OR  (09:30–09:55, entry from 10:00) ← current live config
  Regime C: 45-min OR  (09:30–10:09, entry from 10:10)
  Regime D: 60-min OR  (09:30–10:24, entry from 10:25)

All regimes use:
  - max_trades_per_day = 2
  - PT=2.0x ATR, SL=1.5x ATR, trailing disabled
  - time_stop = 24 bars (2 hours)
  - n=2 MNQ contracts
  - Topstep risk: max_daily_loss=$950, drawdown_buffer=$1,800

Run:
    python rule_based_v1/diagnostics/regime_backtest.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from engine.risk_manager import RiskManager, TradeRecord
from engine.signal_aggregator import SignalAggregator
from rules.opening_range import OpeningRangeBreakoutRule
from utils.indicators import atr

# ---------------------------------------------------------------------------
# Instrument constants
# ---------------------------------------------------------------------------
POINT_VALUE = 2.0       # MNQ: $2 per point
TICK_SIZE = 0.25
TICK_VALUE = 0.50
N_CONTRACTS = 2
COMMISSION_PER_SIDE = 0.62
SLIPPAGE_TICKS = 1

PT_ATR_MULT = 2.0
SL_ATR_MULT = 1.5
TRAILING_ACTIVATION_ATR = 999.0   # disabled
TRAILING_DISTANCE_ATR = 0.75
ATR_PERIOD = 14
TIME_STOP_BARS = 24               # 2 hours on 5-min bars

STARTING_EQUITY = 50_000.0
MAX_TRADES_PER_DAY = 2

# Topstep risk params
MAX_DAILY_LOSS = -950.0
DRAWDOWN_BUFFER = 1_800.0
PER_TRADE_MAX_LOSS = 1_000.0
COOLDOWN_BARS = 3
MAX_CONSECUTIVE_LOSSES = 10       # effectively disabled

MIN_RANGE_ATR = 0.3
ENTRY_CUTOFF_TIME = "12:00"       # same for all regimes

DATA_PATH = ROOT / "data" / "processed" / "mnq_bars_5min.h5"

# ---------------------------------------------------------------------------
# Regime definitions: (label, or_end_time, min_or_bars)
# ---------------------------------------------------------------------------
REGIMES = [
    ("15-min OR", "09:44", 3),   # bars: 09:30, 09:35, 09:40 → entry from 09:45
    ("30-min OR", "09:55", 5),   # bars: 09:30–09:50         → entry from 10:00
    ("45-min OR", "10:09", 8),   # bars: 09:30–10:05         → entry from 10:10
    ("60-min OR", "10:24", 11),  # bars: 09:30–10:20         → entry from 10:25
]


# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------
@dataclass
class Position:
    direction: int
    entry_price: float
    entry_bar_idx: int
    stop_loss: float
    profit_target: float
    time_stop_bar: int
    trailing_active: bool = False
    trailing_stop: float = 0.0
    peak_favorable: float = 0.0
    atr_at_entry: float = 0.0


# ---------------------------------------------------------------------------
# Helpers (identical to live_parity_backtest for consistency)
# ---------------------------------------------------------------------------

def _apply_slippage(price: float, direction: int, is_entry: bool) -> float:
    slippage = SLIPPAGE_TICKS * TICK_SIZE
    return price + slippage * direction if is_entry else price - slippage * direction


def _round_trip_commission() -> float:
    return 2 * COMMISSION_PER_SIDE * N_CONTRACTS


def _compute_pnl(entry_price: float, exit_price: float, direction: int) -> float:
    raw = (exit_price - entry_price) * direction * N_CONTRACTS * POINT_VALUE
    return raw - _round_trip_commission()


def _check_exit(position: Position, bar: pd.Series, bar_idx: int,
                is_session_close: bool) -> tuple[bool, float, str]:
    high, low, close = bar["high"], bar["low"], bar["close"]

    if is_session_close:
        return True, _apply_slippage(close, position.direction, False), "session_close"

    if bar_idx >= position.time_stop_bar:
        return True, _apply_slippage(close, position.direction, False), "time_stop"

    if position.direction == 1:
        if low <= position.stop_loss:
            return True, _apply_slippage(position.stop_loss, position.direction, False), "stop_loss"
        if position.trailing_active and low <= position.trailing_stop:
            return True, _apply_slippage(position.trailing_stop, position.direction, False), "trailing_stop"
        if high >= position.profit_target:
            return True, _apply_slippage(position.profit_target, position.direction, False), "profit_target"
        # Update trailing stop
        if not position.trailing_active:
            if high - position.entry_price >= TRAILING_ACTIVATION_ATR * position.atr_at_entry:
                position.trailing_active = True
                position.peak_favorable = high
                position.trailing_stop = high - TRAILING_DISTANCE_ATR * position.atr_at_entry
        elif high > position.peak_favorable:
            position.peak_favorable = high
            position.trailing_stop = high - TRAILING_DISTANCE_ATR * position.atr_at_entry
    else:
        if high >= position.stop_loss:
            return True, _apply_slippage(position.stop_loss, position.direction, False), "stop_loss"
        if position.trailing_active and high >= position.trailing_stop:
            return True, _apply_slippage(position.trailing_stop, position.direction, False), "trailing_stop"
        if low <= position.profit_target:
            return True, _apply_slippage(position.profit_target, position.direction, False), "profit_target"
        if not position.trailing_active:
            if position.entry_price - low >= TRAILING_ACTIVATION_ATR * position.atr_at_entry:
                position.trailing_active = True
                position.peak_favorable = low
                position.trailing_stop = low + TRAILING_DISTANCE_ATR * position.atr_at_entry
        elif low < position.peak_favorable:
            position.peak_favorable = low
            position.trailing_stop = low + TRAILING_DISTANCE_ATR * position.atr_at_entry

    return False, 0.0, ""


def _build_summary(trades: list[TradeRecord], equity_values: list[float],
                   equity_times: list, daily_pnl_records: dict) -> dict:
    if not trades:
        return {
            "num_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
            "avg_pnl": 0.0, "profit_factor": 0.0, "sharpe": 0.0,
            "max_drawdown": 0.0, "trades_per_day": 0.0, "exit_reasons": {},
        }

    total_pnl = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses))
    pf = gp / gl if gl > 0 else float("inf")

    eq = pd.Series(equity_values, index=equity_times)
    dd = (eq - eq.cummax()).min()

    daily = pd.Series(daily_pnl_records)
    sharpe = 0.0
    if len(daily) > 1 and daily.std() > 0:
        sharpe = daily.mean() / daily.std() * np.sqrt(252)

    num_days = max(1, len(daily_pnl_records))
    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t.exit_reason] += 1
    reason_pct = {k: round(v / len(trades) * 100, 1) for k, v in exit_reasons.items()}

    return {
        "num_trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / len(trades), 2),
        "profit_factor": round(pf, 3),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(dd, 2),
        "trades_per_day": round(len(trades) / num_days, 2),
        "exit_reasons": reason_pct,
    }


# ---------------------------------------------------------------------------
# Core simulation engine
# ---------------------------------------------------------------------------

def run_regime(bars: pd.DataFrame, or_end_time: str, min_or_bars: int,
               label: str) -> dict:
    """Run a single OR regime with max 2 trades/day."""

    orb_rule = OpeningRangeBreakoutRule(
        or_end_time=or_end_time,
        min_or_bars=min_or_bars,
        min_range_atr=MIN_RANGE_ATR,
        entry_cutoff_time=ENTRY_CUTOFF_TIME,
        atr_period=ATR_PERIOD,
    )
    aggregator = SignalAggregator(
        primary_rule=orb_rule,
        filter_rules=[],
        confirmation_rules=[],
        min_confirmations=0,
    )

    rm = RiskManager(
        contracts=N_CONTRACTS,
        point_value=POINT_VALUE,
        tick_size=TICK_SIZE,
        tick_value=TICK_VALUE,
        max_daily_loss=MAX_DAILY_LOSS,
        per_trade_max_loss=PER_TRADE_MAX_LOSS,
        max_consecutive_losses=MAX_CONSECUTIVE_LOSSES,
        cooldown_bars=COOLDOWN_BARS,
        drawdown_buffer=DRAWDOWN_BUFFER,
    )
    rm.reset_all(STARTING_EQUITY)

    atr_series = atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars_needed = aggregator.required_bars()

    position: Position | None = None
    trades: list[TradeRecord] = []
    equity_values: list[float] = []
    equity_times: list = []
    current_equity = STARTING_EQUITY
    current_date = None
    daily_pnl_records: dict = {}
    trades_today = 0          # ← 2-trades-per-day counter

    for i in range(min_bars_needed, len(bars)):
        bar = bars.iloc[i]
        bar_time = bars.index[i]
        bar_time_et = bar_time.tz_convert("US/Eastern") if bar_time.tzinfo else bar_time
        bar_date = bar_time_et.date()

        # Day rollover
        if current_date is not None and bar_date != current_date:
            daily_pnl_records[current_date] = rm.daily_pnl
            rm.reset_daily()
            trades_today = 0
        current_date = bar_date

        rm.tick_bar()

        # Session close detection
        is_last_bar = False
        if i + 1 < len(bars):
            next_et = bars.index[i + 1]
            if next_et.tzinfo:
                next_et = next_et.tz_convert("US/Eastern")
            is_last_bar = next_et.date() != bar_date
        else:
            is_last_bar = True
        is_near_close = bar_time_et.hour == 15 and bar_time_et.minute >= 55
        is_session_close = is_last_bar or is_near_close

        # Exit check
        if position is not None:
            exited, exit_price, reason = _check_exit(position, bar, i, is_session_close)
            if exited:
                pnl = _compute_pnl(position.entry_price, exit_price, position.direction)
                trade = TradeRecord(
                    entry_bar=position.entry_bar_idx,
                    exit_bar=i,
                    direction=position.direction,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    exit_reason=reason,
                )
                trades.append(trade)
                rm.record_trade(trade)
                current_equity += pnl
                position = None

        # Entry check: flat + not session close + under daily trade cap
        if position is None and not is_session_close and trades_today < MAX_TRADES_PER_DAY:
            can_trade, _ = rm.can_trade()
            if can_trade:
                lookback = bars.iloc[max(0, i - min_bars_needed + 1): i + 1]
                decision = aggregator.evaluate(lookback)
                if decision.should_trade:
                    cur_atr = atr_series.iloc[i]
                    if not (np.isnan(cur_atr) or cur_atr <= 0):
                        ep = _apply_slippage(bar["close"], decision.direction, True)
                        sl = rm.compute_stop_price(ep, decision.direction, cur_atr, SL_ATR_MULT)
                        pt = rm.compute_target_price(ep, decision.direction, cur_atr, PT_ATR_MULT)
                        position = Position(
                            direction=decision.direction,
                            entry_price=ep,
                            entry_bar_idx=i,
                            stop_loss=sl,
                            profit_target=pt,
                            time_stop_bar=i + TIME_STOP_BARS,
                            atr_at_entry=cur_atr,
                        )
                        trades_today += 1

        equity_values.append(current_equity)
        equity_times.append(bar_time)

    if current_date is not None:
        daily_pnl_records[current_date] = rm.daily_pnl

    return _build_summary(trades, equity_values, equity_times, daily_pnl_records)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_bars() -> pd.DataFrame:
    store = pd.HDFStore(str(DATA_PATH), mode="r")
    bars = store["bars_5min"]
    store.close()

    # RTH only: 09:30–16:00 ET
    if bars.index.tzinfo is None:
        bars.index = bars.index.tz_localize("UTC")
    bars_et = bars.tz_convert("US/Eastern")
    rth_mask = (
        (bars_et.index.hour > 9) |
        ((bars_et.index.hour == 9) & (bars_et.index.minute >= 30))
    ) & (bars_et.index.hour < 16)
    bars = bars_et.loc[rth_mask]

    print(f"Loaded {len(bars)} RTH bars: {bars.index[0]} → {bars.index[-1]}")
    return bars


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    bars = load_bars()

    print(f"\n{'='*70}")
    print(f"  ORB Regime Backtest  —  {len(bars)} bars  —  max {MAX_TRADES_PER_DAY} trades/day")
    print(f"  MNQ n={N_CONTRACTS}, PT={PT_ATR_MULT}x ATR, SL={SL_ATR_MULT}x ATR")
    print(f"  Max daily loss: ${MAX_DAILY_LOSS}, Drawdown buffer: ${DRAWDOWN_BUFFER}")
    print(f"{'='*70}\n")

    results = {}
    for label, or_end_time, min_or_bars in REGIMES:
        print(f"Running {label} (or_end={or_end_time}, min_or_bars={min_or_bars})...", end=" ", flush=True)
        r = run_regime(bars, or_end_time, min_or_bars, label)
        results[label] = r
        print(f"done — {r['num_trades']} trades, PnL=${r['total_pnl']:+,.0f}, "
              f"WR={r['win_rate']:.0%}, Sharpe={r['sharpe']:.2f}")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    header = f"{'Regime':<15} {'Trades':>7} {'T/Day':>6} {'WR':>7} {'PnL':>10} {'AvgPnL':>8} {'PF':>6} {'Sharpe':>7} {'MaxDD':>10}"
    print(header)
    print("-" * len(header))

    for label, r in results.items():
        print(
            f"{label:<15} {r['num_trades']:>7} {r['trades_per_day']:>6.2f} "
            f"{r['win_rate']:>6.1%} {r['total_pnl']:>+10,.0f} "
            f"{r['avg_pnl']:>+8.1f} {r['profit_factor']:>6.2f} "
            f"{r['sharpe']:>7.2f} {r['max_drawdown']:>+10,.0f}"
        )

    print()

    # Exit reason breakdown
    print("Exit reason breakdown (% of trades):")
    for label, r in results.items():
        reasons_str = "  ".join(f"{k}={v:.0f}%" for k, v in sorted(r["exit_reasons"].items()))
        print(f"  {label:<15}: {reasons_str}")

    # ---------------------------------------------------------------------------
    # Topstep combine assessment
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  TOPSTEP 50K COMBINE ASSESSMENT")
    print(f"{'='*70}")
    TARGET_PROFIT = 3_000.0
    MAX_DD = DRAWDOWN_BUFFER

    for label, r in results.items():
        pnl = r["total_pnl"]
        dd = abs(r["max_drawdown"])
        # Rough estimate: daily profit rate × 15 trading days needed to pass
        num_days = max(1, r["num_trades"] / max(0.01, r["trades_per_day"]))
        daily_rate = pnl / num_days if num_days > 0 else 0
        days_to_target = TARGET_PROFIT / daily_rate if daily_rate > 0 else float("inf")
        dd_ok = dd < MAX_DD
        status = "✓ DD OK" if dd_ok else "✗ DD TOO HIGH"
        print(f"  {label:<15}: daily_avg=${daily_rate:+.0f}/day, "
              f"est_days_to_pass={days_to_target:.0f}, "
              f"max_dd=${dd:,.0f} {status}")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    out_path = ROOT / "rule_based_v1" / "diagnostics" / "regime_backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
