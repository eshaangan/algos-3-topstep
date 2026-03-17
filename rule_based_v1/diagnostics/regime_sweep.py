"""ORB Fine-Grained Sweep — OR Window + PT/SL Grid.

Part 1: Fine OR window sweep around the 45-min winner (10:04–10:29 in 5-min steps)
Part 2: PT/SL grid on the best OR window

All configs use max_trades_per_day=2, n=2 MNQ contracts.

Run:
    python rule_based_v1/diagnostics/regime_sweep.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from itertools import product

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
# Constants
# ---------------------------------------------------------------------------
POINT_VALUE = 2.0
TICK_SIZE = 0.25
N_CONTRACTS = 2
COMMISSION_PER_SIDE = 0.62
SLIPPAGE_TICKS = 1

ATR_PERIOD = 14
TIME_STOP_BARS = 24
TRAILING_ACTIVATION_ATR = 999.0
TRAILING_DISTANCE_ATR = 0.75

STARTING_EQUITY = 50_000.0
MAX_TRADES_PER_DAY = 2
MAX_DAILY_LOSS = -950.0
DRAWDOWN_BUFFER = 1_800.0
PER_TRADE_MAX_LOSS = 1_000.0
COOLDOWN_BARS = 3
MAX_CONSECUTIVE_LOSSES = 10

MIN_RANGE_ATR = 0.3
ENTRY_CUTOFF = "12:00"

DATA_PATH = ROOT / "data" / "processed" / "mnq_bars_5min.h5"

# ---------------------------------------------------------------------------
# OR fine sweep configs: (label, or_end_time, min_or_bars)
# 5-min bar timestamps are the bar open. or_end_time="HH:MM" includes bars <= HH:MM.
# bars <= 10:04 → 09:30,35,40,45,50,55,10:00 = 7 bars → entry from 10:05
# ---------------------------------------------------------------------------
OR_SWEEP = [
    ("10:04 (7bar)",  "10:04",  7),
    ("10:09 (8bar)",  "10:09",  8),   # ← previous winner
    ("10:14 (9bar)",  "10:14",  9),
    ("10:19 (10bar)", "10:19", 10),
    ("10:24 (11bar)", "10:24", 11),
    ("10:29 (12bar)", "10:29", 12),
]

# PT/SL grid — tested on the best OR from Part 1
PT_MULTS = [1.5, 2.0, 2.5, 3.0]
SL_MULTS = [1.0, 1.5, 2.0]


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
# Helpers
# ---------------------------------------------------------------------------

def _slip(price: float, direction: int, is_entry: bool) -> float:
    s = SLIPPAGE_TICKS * TICK_SIZE
    return price + s * direction if is_entry else price - s * direction


def _pnl(entry: float, exit_: float, direction: int) -> float:
    raw = (exit_ - entry) * direction * N_CONTRACTS * POINT_VALUE
    return raw - 2 * COMMISSION_PER_SIDE * N_CONTRACTS


def _check_exit(pos: Position, bar: pd.Series, idx: int,
                session_close: bool, pt_mult: float, sl_mult: float) -> tuple[bool, float, str]:
    h, l, c = bar["high"], bar["low"], bar["close"]

    if session_close:
        return True, _slip(c, pos.direction, False), "session_close"
    if idx >= pos.time_stop_bar:
        return True, _slip(c, pos.direction, False), "time_stop"

    if pos.direction == 1:
        if l <= pos.stop_loss:
            return True, _slip(pos.stop_loss, 1, False), "stop_loss"
        if pos.trailing_active and l <= pos.trailing_stop:
            return True, _slip(pos.trailing_stop, 1, False), "trailing_stop"
        if h >= pos.profit_target:
            return True, _slip(pos.profit_target, 1, False), "profit_target"
        if not pos.trailing_active:
            if h - pos.entry_price >= TRAILING_ACTIVATION_ATR * pos.atr_at_entry:
                pos.trailing_active = True
                pos.peak_favorable = h
                pos.trailing_stop = h - TRAILING_DISTANCE_ATR * pos.atr_at_entry
        elif h > pos.peak_favorable:
            pos.peak_favorable = h
            pos.trailing_stop = h - TRAILING_DISTANCE_ATR * pos.atr_at_entry
    else:
        if h >= pos.stop_loss:
            return True, _slip(pos.stop_loss, -1, False), "stop_loss"
        if pos.trailing_active and h >= pos.trailing_stop:
            return True, _slip(pos.trailing_stop, -1, False), "trailing_stop"
        if l <= pos.profit_target:
            return True, _slip(pos.profit_target, -1, False), "profit_target"
        if not pos.trailing_active:
            if pos.entry_price - l >= TRAILING_ACTIVATION_ATR * pos.atr_at_entry:
                pos.trailing_active = True
                pos.peak_favorable = l
                pos.trailing_stop = l + TRAILING_DISTANCE_ATR * pos.atr_at_entry
        elif l < pos.peak_favorable:
            pos.peak_favorable = l
            pos.trailing_stop = l + TRAILING_DISTANCE_ATR * pos.atr_at_entry

    return False, 0.0, ""


def _summary(trades: list[TradeRecord], eq_vals: list, eq_times: list,
             daily_pnl: dict) -> dict:
    if not trades:
        return {"num_trades": 0, "win_rate": 0.0, "total_pnl": 0.0,
                "avg_pnl": 0.0, "profit_factor": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "trades_per_day": 0.0, "exit_reasons": {}}

    total = sum(t.pnl for t in trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses))
    pf = gp / gl if gl > 0 else float("inf")

    eq = pd.Series(eq_vals, index=eq_times)
    dd = (eq - eq.cummax()).min()

    daily = pd.Series(daily_pnl)
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0.0

    num_days = max(1, len(daily_pnl))
    reasons = defaultdict(int)
    for t in trades:
        reasons[t.exit_reason] += 1
    reason_pct = {k: round(v / len(trades) * 100, 1) for k, v in reasons.items()}

    return {
        "num_trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3),
        "total_pnl": round(total, 2),
        "avg_pnl": round(total / len(trades), 2),
        "profit_factor": round(pf, 3),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(dd, 2),
        "trades_per_day": round(len(trades) / num_days, 2),
        "exit_reasons": reason_pct,
    }


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run(bars: pd.DataFrame, or_end_time: str, min_or_bars: int,
        pt_mult: float, sl_mult: float) -> dict:

    orb = OpeningRangeBreakoutRule(
        or_end_time=or_end_time,
        min_or_bars=min_or_bars,
        min_range_atr=MIN_RANGE_ATR,
        entry_cutoff_time=ENTRY_CUTOFF,
        atr_period=ATR_PERIOD,
    )
    agg = SignalAggregator(primary_rule=orb, filter_rules=[], confirmation_rules=[], min_confirmations=0)

    rm = RiskManager(
        contracts=N_CONTRACTS, point_value=POINT_VALUE, tick_size=TICK_SIZE,
        tick_value=TICK_SIZE * 2,  # MNQ tick value = $0.50
        max_daily_loss=MAX_DAILY_LOSS, per_trade_max_loss=PER_TRADE_MAX_LOSS,
        max_consecutive_losses=MAX_CONSECUTIVE_LOSSES, cooldown_bars=COOLDOWN_BARS,
        drawdown_buffer=DRAWDOWN_BUFFER,
    )
    rm.reset_all(STARTING_EQUITY)

    atr_s = atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars_needed = agg.required_bars()

    pos: Position | None = None
    trades: list[TradeRecord] = []
    eq_vals, eq_times = [], []
    equity = STARTING_EQUITY
    cur_date = None
    daily_pnl: dict = {}
    trades_today = 0

    for i in range(min_bars_needed, len(bars)):
        bar = bars.iloc[i]
        bt = bars.index[i]
        bt_et = bt.tz_convert("US/Eastern") if bt.tzinfo else bt
        bdate = bt_et.date()

        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = rm.daily_pnl
            rm.reset_daily()
            trades_today = 0
        cur_date = bdate
        rm.tick_bar()

        # Session close
        is_last = (i + 1 >= len(bars)) or (
            (bars.index[i + 1].tz_convert("US/Eastern") if bars.index[i + 1].tzinfo
             else bars.index[i + 1]).date() != bdate
        )
        near_close = bt_et.hour == 15 and bt_et.minute >= 55
        sess_close = is_last or near_close

        # Exit
        if pos is not None:
            exited, exit_p, reason = _check_exit(pos, bar, i, sess_close, pt_mult, sl_mult)
            if exited:
                p = _pnl(pos.entry_price, exit_p, pos.direction)
                tr = TradeRecord(entry_bar=pos.entry_bar_idx, exit_bar=i,
                                 direction=pos.direction, entry_price=pos.entry_price,
                                 exit_price=exit_p, pnl=p, exit_reason=reason)
                trades.append(tr)
                rm.record_trade(tr)
                equity += p
                pos = None

        # Entry
        if pos is None and not sess_close and trades_today < MAX_TRADES_PER_DAY:
            ok, _ = rm.can_trade()
            if ok:
                lookback = bars.iloc[max(0, i - min_bars_needed + 1): i + 1]
                dec = agg.evaluate(lookback)
                if dec.should_trade:
                    cur_atr = atr_s.iloc[i]
                    if not (np.isnan(cur_atr) or cur_atr <= 0):
                        ep = _slip(bar["close"], dec.direction, True)
                        sl = rm.compute_stop_price(ep, dec.direction, cur_atr, sl_mult)
                        pt = rm.compute_target_price(ep, dec.direction, cur_atr, pt_mult)
                        pos = Position(direction=dec.direction, entry_price=ep,
                                       entry_bar_idx=i, stop_loss=sl, profit_target=pt,
                                       time_stop_bar=i + TIME_STOP_BARS, atr_at_entry=cur_atr)
                        trades_today += 1

        eq_vals.append(equity)
        eq_times.append(bt)

    if cur_date is not None:
        daily_pnl[cur_date] = rm.daily_pnl

    return _summary(trades, eq_vals, eq_times, daily_pnl)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_bars() -> pd.DataFrame:
    store = pd.HDFStore(str(DATA_PATH), mode="r")
    bars = store["bars_5min"]
    store.close()
    if bars.index.tzinfo is None:
        bars.index = bars.index.tz_localize("UTC")
    bars = bars.tz_convert("US/Eastern")
    rth = ((bars.index.hour > 9) | ((bars.index.hour == 9) & (bars.index.minute >= 30))) \
          & (bars.index.hour < 16)
    return bars.loc[rth]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    bars = load_bars()
    print(f"Loaded {len(bars)} RTH bars: {bars.index[0].date()} → {bars.index[-1].date()}")

    all_results = {}

    # =========================================================================
    # PART 1: Fine OR window sweep (PT=2.0, SL=1.5 fixed)
    # =========================================================================
    print(f"\n{'='*72}")
    print(f"  PART 1: Fine OR Window Sweep  (PT=2.0x, SL=1.5x, max {MAX_TRADES_PER_DAY} trades/day)")
    print(f"{'='*72}")

    or_results = {}
    for label, or_end, min_orb in OR_SWEEP:
        r = run(bars, or_end, min_orb, pt_mult=2.0, sl_mult=1.5)
        or_results[label] = r
        dd_ok = abs(r["max_drawdown"]) < DRAWDOWN_BUFFER
        flag = "✓" if dd_ok and r["total_pnl"] > 0 else "✗"
        print(f"  {flag} or_end={or_end}  trades={r['num_trades']:>3}  "
              f"WR={r['win_rate']:.0%}  PnL=${r['total_pnl']:>+7,.0f}  "
              f"Sharpe={r['sharpe']:>5.2f}  DD=${r['max_drawdown']:>+7,.0f}")

    all_results["or_sweep"] = or_results

    # Find best OR by Sharpe (with DD constraint)
    best_or_label = max(
        or_results,
        key=lambda k: or_results[k]["sharpe"] if abs(or_results[k]["max_drawdown"]) < DRAWDOWN_BUFFER else -999
    )
    best_or_end = OR_SWEEP[[x[0] for x in OR_SWEEP].index(best_or_label)][1]
    best_or_min_bars = OR_SWEEP[[x[0] for x in OR_SWEEP].index(best_or_label)][2]
    print(f"\n  → Best OR: {best_or_label} (or_end={best_or_end})")

    # =========================================================================
    # PART 2: PT/SL grid on best OR
    # =========================================================================
    print(f"\n{'='*72}")
    print(f"  PART 2: PT/SL Grid  (or_end={best_or_end}, max {MAX_TRADES_PER_DAY} trades/day)")
    print(f"{'='*72}")

    grid_header = f"  {'PT×':>5} {'SL×':>5} {'Trades':>7} {'WR':>6} {'PnL':>10} {'Avg':>8} {'PF':>6} {'Sharpe':>7} {'MaxDD':>10} {'OK?':>5}"
    print(grid_header)
    print("  " + "-" * (len(grid_header) - 2))

    grid_results = {}
    for pt_m, sl_m in product(PT_MULTS, SL_MULTS):
        r = run(bars, best_or_end, best_or_min_bars, pt_mult=pt_m, sl_mult=sl_m)
        key = f"PT{pt_m:.1f}_SL{sl_m:.1f}"
        grid_results[key] = r
        dd_ok = abs(r["max_drawdown"]) < DRAWDOWN_BUFFER
        flag = "✓" if dd_ok and r["total_pnl"] > 0 else "✗"
        print(f"  {pt_m:>5.1f} {sl_m:>5.1f} {r['num_trades']:>7} {r['win_rate']:>5.0%} "
              f"{r['total_pnl']:>+10,.0f} {r['avg_pnl']:>+8.1f} {r['profit_factor']:>6.2f} "
              f"{r['sharpe']:>7.2f} {r['max_drawdown']:>+10,.0f} {flag:>5}")

    all_results["ptsl_grid"] = grid_results

    # =========================================================================
    # Best overall combo
    # =========================================================================
    best_key = max(
        grid_results,
        key=lambda k: grid_results[k]["sharpe"]
        if abs(grid_results[k]["max_drawdown"]) < DRAWDOWN_BUFFER and grid_results[k]["total_pnl"] > 0
        else -999
    )
    br = grid_results[best_key]
    pt_best, sl_best = [float(x[2:]) for x in best_key.split("_")]

    print(f"\n{'='*72}")
    print(f"  BEST COMBO: or_end={best_or_end}, {best_key}")
    print(f"  Trades={br['num_trades']}, WR={br['win_rate']:.1%}, "
          f"PnL=${br['total_pnl']:+,.0f}, Sharpe={br['sharpe']:.2f}, "
          f"MaxDD=${br['max_drawdown']:+,.0f}")
    print(f"  Exit reasons: {br['exit_reasons']}")

    daily_avg = br["total_pnl"] / max(1, br["num_trades"] / max(0.01, br["trades_per_day"]))
    days_to_pass = 3_000 / daily_avg if daily_avg > 0 else float("inf")
    print(f"  ~${daily_avg:.0f}/day → est. {days_to_pass:.0f} trading days to $3k target")
    print(f"{'='*72}")

    # =========================================================================
    # Save
    # =========================================================================
    out = ROOT / "rule_based_v1" / "diagnostics" / "regime_sweep_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
