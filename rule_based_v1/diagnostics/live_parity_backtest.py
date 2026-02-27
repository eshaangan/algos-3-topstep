"""Live Parity Backtest — ORB Strategy Variant Comparison.

Compares three variants of the MNQ ORB strategy on Jan 1 – Feb 27 2026 OOS data:

  A: Original   — time_stop_bars=24, one position at a time
  B: No TimeStop — time_stop_bars=9999 (effectively infinite), one position at a time
  C: Multi-Entry — time_stop_bars=9999, multiple concurrent positions (live bug behavior)

Run:
    python rule_based_v1/diagnostics/live_parity_backtest.py
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
# Instrument / strategy config  (best validated OOS config)
# ---------------------------------------------------------------------------
POINT_VALUE = 2.0
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

OR_END_TIME = "09:55"
MIN_OR_BARS = 4
MIN_RANGE_ATR = 0.3
ENTRY_CUTOFF_TIME = "12:00"

STARTING_EQUITY = 50_000.0
MAX_CONCURRENT_POSITIONS = 4       # Variant C cap

# Risk manager params (from validated OOS test)
MAX_DAILY_LOSS = -950.0
DRAWDOWN_BUFFER = 1800.0
PER_TRADE_MAX_LOSS = 1000.0
COOLDOWN_BARS = 3
MAX_CONSECUTIVE_LOSSES = 10        # effectively disabled for parity with original test


# ---------------------------------------------------------------------------
# Position dataclass (shared by A/B/C)
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
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_slippage(price: float, direction: int, is_entry: bool,
                    tick_size: float = TICK_SIZE,
                    slippage_ticks: int = SLIPPAGE_TICKS) -> float:
    slippage = slippage_ticks * tick_size
    if is_entry:
        return price + slippage * direction
    else:
        return price - slippage * direction


def _round_commission(contracts: int = N_CONTRACTS,
                      commission_per_side: float = COMMISSION_PER_SIDE) -> float:
    return 2 * commission_per_side * contracts


def _compute_trade_pnl(entry_price: float, exit_price: float,
                        direction: int,
                        contracts: int = N_CONTRACTS,
                        point_value: float = POINT_VALUE) -> float:
    raw = (exit_price - entry_price) * direction * contracts * point_value
    return raw - _round_commission(contracts)


def _check_exit(position: Position, bar: pd.Series, bar_idx: int,
                is_session_close: bool,
                time_stop_bars: int,
                trailing_activation_atr: float = TRAILING_ACTIVATION_ATR,
                trailing_distance_atr: float = TRAILING_DISTANCE_ATR
                ) -> tuple[bool, float, str]:
    high = bar["high"]
    low = bar["low"]
    close = bar["close"]

    if is_session_close:
        ep = _apply_slippage(close, position.direction, is_entry=False)
        return True, ep, "session_close"

    if bar_idx >= position.time_stop_bar:
        ep = _apply_slippage(close, position.direction, is_entry=False)
        return True, ep, "time_stop"

    if position.direction == 1:
        if low <= position.stop_loss:
            ep = _apply_slippage(position.stop_loss, position.direction, is_entry=False)
            return True, ep, "stop_loss"
        if position.trailing_active and low <= position.trailing_stop:
            ep = _apply_slippage(position.trailing_stop, position.direction, is_entry=False)
            return True, ep, "trailing_stop"
        if high >= position.profit_target:
            ep = _apply_slippage(position.profit_target, position.direction, is_entry=False)
            return True, ep, "profit_target"
        # Update trailing
        if not position.trailing_active:
            if high - position.entry_price >= trailing_activation_atr * position.atr_at_entry:
                position.trailing_active = True
                position.peak_favorable = high
                position.trailing_stop = high - trailing_distance_atr * position.atr_at_entry
        else:
            if high > position.peak_favorable:
                position.peak_favorable = high
                position.trailing_stop = high - trailing_distance_atr * position.atr_at_entry
    else:
        if high >= position.stop_loss:
            ep = _apply_slippage(position.stop_loss, position.direction, is_entry=False)
            return True, ep, "stop_loss"
        if position.trailing_active and high >= position.trailing_stop:
            ep = _apply_slippage(position.trailing_stop, position.direction, is_entry=False)
            return True, ep, "trailing_stop"
        if low <= position.profit_target:
            ep = _apply_slippage(position.profit_target, position.direction, is_entry=False)
            return True, ep, "profit_target"
        if not position.trailing_active:
            if position.entry_price - low >= trailing_activation_atr * position.atr_at_entry:
                position.trailing_active = True
                position.peak_favorable = low
                position.trailing_stop = low + trailing_distance_atr * position.atr_at_entry
        else:
            if low < position.peak_favorable:
                position.peak_favorable = low
                position.trailing_stop = low + trailing_distance_atr * position.atr_at_entry

    return False, 0.0, ""


def _make_risk_manager() -> RiskManager:
    return RiskManager(
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


def _make_aggregator() -> SignalAggregator:
    orb_rule = OpeningRangeBreakoutRule(
        or_end_time=OR_END_TIME,
        min_or_bars=MIN_OR_BARS,
        min_range_atr=MIN_RANGE_ATR,
        entry_cutoff_time=ENTRY_CUTOFF_TIME,
        atr_period=ATR_PERIOD,
    )
    return SignalAggregator(
        primary_rule=orb_rule,
        filter_rules=[],
        confirmation_rules=[],
        min_confirmations=0,
    )


def _build_results(trades: list[TradeRecord], equity_values: list[float],
                   equity_times: list, daily_pnl_records: dict) -> dict:
    if not trades:
        return {
            "num_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
            "avg_pnl": 0.0, "profit_factor": 0.0, "sharpe": 0.0,
            "max_drawdown": 0.0, "trades_per_day": 0.0,
            "exit_reasons": {},
        }

    total_pnl = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    equity_series = pd.Series(equity_values, index=equity_times)
    peak = equity_series.cummax()
    dd = (equity_series - peak).min()

    daily_pnl_series = pd.Series(daily_pnl_records)
    sharpe = 0.0
    if len(daily_pnl_series) > 1 and daily_pnl_series.std() > 0:
        sharpe = daily_pnl_series.mean() / daily_pnl_series.std() * np.sqrt(252)

    num_days = max(1, len(daily_pnl_records))
    trades_per_day = len(trades) / num_days

    exit_reasons: dict[str, int] = defaultdict(int)
    for t in trades:
        exit_reasons[t.exit_reason] += 1
    reason_pct = {k: round(v / len(trades) * 100, 1) for k, v in exit_reasons.items()}

    return {
        "num_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / len(trades), 2),
        "profit_factor": round(profit_factor, 3),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(dd, 2),
        "trades_per_day": round(trades_per_day, 2),
        "exit_reasons": reason_pct,
    }


# ---------------------------------------------------------------------------
# Variant A & B — single-position engine
# ---------------------------------------------------------------------------

def run_single_position(bars: pd.DataFrame, time_stop_bars: int,
                         label: str) -> dict:
    """Run single-position sequential backtest (Variants A and B)."""
    rm = _make_risk_manager()
    aggregator = _make_aggregator()
    rm.reset_all(STARTING_EQUITY)

    atr_series = atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars = aggregator.required_bars()

    position: Position | None = None
    trades: list[TradeRecord] = []
    equity_values: list[float] = []
    equity_times: list = []
    current_equity = STARTING_EQUITY
    current_date = None
    daily_pnl_records: dict = {}

    for i in range(min_bars, len(bars)):
        bar = bars.iloc[i]
        bar_time = bars.index[i]

        bar_time_et = bar_time.tz_convert("US/Eastern") if bar_time.tzinfo else bar_time
        bar_date = bar_time_et.date()

        if current_date is not None and bar_date != current_date:
            daily_pnl_records[current_date] = rm.daily_pnl
            rm.reset_daily()
        current_date = bar_date

        rm.tick_bar()

        # Determine session close
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
            should_exit, exit_price, reason = _check_exit(
                position, bar, i, is_session_close, time_stop_bars
            )
            if should_exit:
                pnl = _compute_trade_pnl(position.entry_price, exit_price, position.direction)
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

        # Entry check (only when flat)
        if position is None and not is_session_close:
            can_trade, _ = rm.can_trade()
            if can_trade:
                lookback = bars.iloc[max(0, i - min_bars + 1): i + 1]
                decision = aggregator.evaluate(lookback)
                if decision.should_trade:
                    cur_atr = atr_series.iloc[i]
                    if not (np.isnan(cur_atr) or cur_atr <= 0):
                        entry_price = _apply_slippage(bar["close"], decision.direction, is_entry=True)
                        stop_loss = rm.compute_stop_price(entry_price, decision.direction, cur_atr, SL_ATR_MULT)
                        profit_target = rm.compute_target_price(entry_price, decision.direction, cur_atr, PT_ATR_MULT)
                        position = Position(
                            direction=decision.direction,
                            entry_price=entry_price,
                            entry_bar_idx=i,
                            stop_loss=stop_loss,
                            profit_target=profit_target,
                            time_stop_bar=i + time_stop_bars,
                            atr_at_entry=cur_atr,
                        )

        equity_values.append(current_equity)
        equity_times.append(bar_time)

    if current_date is not None:
        daily_pnl_records[current_date] = rm.daily_pnl

    return _build_results(trades, equity_values, equity_times, daily_pnl_records)


# ---------------------------------------------------------------------------
# Variant C — multi-position engine (live bug behavior)
# ---------------------------------------------------------------------------

def run_multi_position(bars: pd.DataFrame, time_stop_bars: int = 9999) -> dict:
    """Run multi-position concurrent backtest (Variant C).

    A new entry fires whenever an ORB signal is detected, regardless of whether
    positions are already open. Up to MAX_CONCURRENT_POSITIONS at once.
    Daily risk is aggregated across all positions.
    """
    rm = _make_risk_manager()
    aggregator = _make_aggregator()
    rm.reset_all(STARTING_EQUITY)

    atr_series = atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars = aggregator.required_bars()

    positions: list[Position] = []
    trades: list[TradeRecord] = []
    equity_values: list[float] = []
    equity_times: list = []
    current_equity = STARTING_EQUITY
    current_date = None
    daily_pnl_records: dict = {}

    # Track daily loss outside rm (rm.daily_pnl accumulates per-trade)
    # Use rm.daily_pnl directly as the aggregate; record_trade() updates it.

    for i in range(min_bars, len(bars)):
        bar = bars.iloc[i]
        bar_time = bars.index[i]

        bar_time_et = bar_time.tz_convert("US/Eastern") if bar_time.tzinfo else bar_time
        bar_date = bar_time_et.date()

        if current_date is not None and bar_date != current_date:
            daily_pnl_records[current_date] = rm.daily_pnl
            rm.reset_daily()
            positions.clear()  # flatten all at day boundary (session_close caught earlier)
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

        # Exit check for ALL open positions
        surviving: list[Position] = []
        for pos in positions:
            should_exit, exit_price, reason = _check_exit(
                pos, bar, i, is_session_close, time_stop_bars
            )
            if should_exit:
                pnl = _compute_trade_pnl(pos.entry_price, exit_price, pos.direction)
                trade = TradeRecord(
                    entry_bar=pos.entry_bar_idx,
                    exit_bar=i,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    exit_reason=reason,
                )
                trades.append(trade)
                rm.record_trade(trade)
                current_equity += pnl
            else:
                surviving.append(pos)
        positions = surviving

        # Entry check — always attempt if below concurrent cap and not session close
        if not is_session_close and len(positions) < MAX_CONCURRENT_POSITIONS:
            # Check aggregate daily loss
            can_trade, _ = rm.can_trade()
            if can_trade:
                lookback = bars.iloc[max(0, i - min_bars + 1): i + 1]
                decision = aggregator.evaluate(lookback)
                if decision.should_trade:
                    cur_atr = atr_series.iloc[i]
                    if not (np.isnan(cur_atr) or cur_atr <= 0):
                        # Prevent re-entry on the same bar for the same direction
                        # if an identical position was just opened this bar
                        entry_price = _apply_slippage(bar["close"], decision.direction, is_entry=True)
                        stop_loss = rm.compute_stop_price(entry_price, decision.direction, cur_atr, SL_ATR_MULT)
                        profit_target = rm.compute_target_price(entry_price, decision.direction, cur_atr, PT_ATR_MULT)

                        # Deduplicate: skip if identical entry already open from this bar
                        is_dup = any(
                            p.entry_bar_idx == i and p.direction == decision.direction
                            for p in positions
                        )
                        if not is_dup:
                            positions.append(Position(
                                direction=decision.direction,
                                entry_price=entry_price,
                                entry_bar_idx=i,
                                stop_loss=stop_loss,
                                profit_target=profit_target,
                                time_stop_bar=i + time_stop_bars,
                                atr_at_entry=cur_atr,
                            ))

        equity_values.append(current_equity)
        equity_times.append(bar_time)

    if current_date is not None:
        daily_pnl_records[current_date] = rm.daily_pnl

    return _build_results(trades, equity_values, equity_times, daily_pnl_records)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _dollar(v: float) -> str:
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.0f}"


def print_comparison(results: dict[str, dict]) -> None:
    labels = list(results.keys())
    col_w = 17
    lbl_w = 22

    hdr = f"{'':>{lbl_w}}"
    for lbl in labels:
        hdr += f"  {lbl:>{col_w}}"
    print()
    print("VARIANT COMPARISON (MNQ ORB, Jan 1 – Feb 27 2026, 2 contracts)")
    print("=" * (lbl_w + (col_w + 2) * len(labels)))
    print(hdr)
    print("-" * (lbl_w + (col_w + 2) * len(labels)))

    rows = [
        ("Trades",          lambda r: str(r["num_trades"])),
        ("Win Rate",        lambda r: _pct(r["win_rate"])),
        ("Total PnL",       lambda r: _dollar(r["total_pnl"])),
        ("Avg PnL/trade",   lambda r: _dollar(r["avg_pnl"])),
        ("Profit Factor",   lambda r: f"{r['profit_factor']:.2f}"),
        ("Sharpe",          lambda r: f"{r['sharpe']:.2f}"),
        ("Max Drawdown",    lambda r: _dollar(r["max_drawdown"])),
        ("Trades/day (avg)",lambda r: f"{r['trades_per_day']:.1f}"),
    ]
    for row_label, fn in rows:
        line = f"{row_label:>{lbl_w}}"
        for lbl in labels:
            val = fn(results[lbl])
            line += f"  {val:>{col_w}}"
        print(line)

    print()
    print("EXIT REASON BREAKDOWN (%)")
    print("-" * (lbl_w + (col_w + 2) * len(labels)))
    all_reasons = set()
    for r in results.values():
        all_reasons.update(r["exit_reasons"].keys())
    for reason in sorted(all_reasons):
        line = f"{reason:>{lbl_w}}"
        for lbl in labels:
            pct = results[lbl]["exit_reasons"].get(reason, 0.0)
            line += f"  {f'{pct:.1f}%':>{col_w}}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    data_path = ROOT / "data" / "processed" / "jan_feb_2026_oos_test.h5"
    print(f"Loading data from: {data_path}")
    bars = pd.read_hdf(str(data_path), key="bars_5min")

    # RTH only
    if bars.index.tzinfo is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    else:
        bars.index = bars.index.tz_convert("US/Eastern")

    rth = bars[(bars.index.hour >= 9) & (bars.index.hour < 16)].copy()

    # Date filter
    rth = rth[(rth.index >= "2026-01-01") & (rth.index <= "2026-02-27")]
    print(f"Bars after RTH filter: {len(rth):,}  "
          f"({rth.index[0].date()} to {rth.index[-1].date()})")
    print()

    print("Running Variant A — Original (time_stop=24, single position)...")
    res_a = run_single_position(rth, time_stop_bars=24, label="A")

    print("Running Variant B — No TimeStop (time_stop=9999, single position)...")
    res_b = run_single_position(rth, time_stop_bars=9999, label="B")

    print("Running Variant C — No TimeStop + Multi-Entry (live bug)...")
    res_c = run_multi_position(rth, time_stop_bars=9999)
    print()

    results = {
        "A: Original      ": res_a,
        "B: No TimeStop   ": res_b,
        "C: Multi-Entry   ": res_c,
    }

    print_comparison(results)

    # Save JSON
    out_dir = RBV1 / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "live_parity_results.json"

    save_data = {
        "description": "MNQ ORB variant comparison, Jan 1 – Feb 27 2026 OOS",
        "config": {
            "or_end_time": OR_END_TIME,
            "min_or_bars": MIN_OR_BARS,
            "min_range_atr": MIN_RANGE_ATR,
            "entry_cutoff_time": ENTRY_CUTOFF_TIME,
            "pt_atr_mult": PT_ATR_MULT,
            "sl_atr_mult": SL_ATR_MULT,
            "n_contracts": N_CONTRACTS,
            "trailing_activation_atr": TRAILING_ACTIVATION_ATR,
            "commission_per_side": COMMISSION_PER_SIDE,
            "slippage_ticks": SLIPPAGE_TICKS,
            "atr_period": ATR_PERIOD,
            "point_value": POINT_VALUE,
            "tick_size": TICK_SIZE,
            "tick_value": TICK_VALUE,
        },
        "variants": {
            "A_original": res_a,
            "B_no_time_stop": res_b,
            "C_multi_entry": res_c,
        },
    }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
