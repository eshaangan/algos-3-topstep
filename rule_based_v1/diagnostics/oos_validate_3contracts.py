"""OOS Validation — Current Live Config with 3 Contracts
=========================================================
Runs the exact deployed config on Jan–Feb 2026 OOS data:
  - or_end_time: 10:04  (7-bar OR window)
  - PT: 3.0x ATR, SL: 1.5x ATR
  - 3 MNQ contracts
  - max_trades_per_day: 2

Compare vs original 2-contract results to see impact of scaling up.

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/oos_validate_3contracts.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
# Config — mirrors deployed rules.yaml + risk.yaml
# ---------------------------------------------------------------------------
OR_END_TIME       = "10:04"
MIN_OR_BARS       = 7
PT_MULT           = 3.0
SL_MULT           = 1.5
ENTRY_CUTOFF      = "12:00"
MIN_RANGE_ATR     = 0.3
ATR_PERIOD        = 14
TIME_STOP_BARS    = 24
TRAILING_ACT_ATR  = 999.0   # disabled
TRAILING_DIST_ATR = 0.75

POINT_VALUE       = 2.0     # MNQ: $2/point
TICK_SIZE         = 0.25
TICK_VALUE        = 0.50    # $0.50/tick

MAX_TRADES_PER_DAY    = 2
MAX_DAILY_LOSS        = -950.0
DRAWDOWN_BUFFER       = 1_950.0   # MLL=$2,000, buffer=$50
PER_TRADE_MAX_LOSS    = 1_000.0
COOLDOWN_BARS         = 3
MAX_CONSECUTIVE_LOSSES = 10
STARTING_EQUITY       = 50_000.0
COMMISSION_PER_SIDE   = 0.62
SLIPPAGE_TICKS        = 1

OOS_PATH    = ROOT / "data" / "processed" / "jan_feb_2026_oos_test.h5"
RESULTS_OUT = ROOT / "rule_based_v1" / "diagnostics" / "oos_3contracts_results.json"

COMBINE = dict(profit_target=3_000, mll=2_000, daily_loss=950)


# ---------------------------------------------------------------------------
# Position
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


def _slip(price: float, direction: int, is_entry: bool) -> float:
    s = SLIPPAGE_TICKS * TICK_SIZE
    return price + s * direction if is_entry else price - s * direction


def _pnl(entry: float, exit_: float, direction: int, n: int) -> float:
    raw = (exit_ - entry) * direction * n * POINT_VALUE
    return raw - 2 * COMMISSION_PER_SIDE * n


def _check_exit(pos: Position, bar: pd.Series, idx: int, sess_close: bool) -> tuple[bool, float, str]:
    h, l, c = bar["high"], bar["low"], bar["close"]

    if sess_close:
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
        if not pos.trailing_active and h - pos.entry_price >= TRAILING_ACT_ATR * pos.atr_at_entry:
            pos.trailing_active = True
            pos.peak_favorable = h
            pos.trailing_stop = h - TRAILING_DIST_ATR * pos.atr_at_entry
        elif pos.trailing_active and h > pos.peak_favorable:
            pos.peak_favorable = h
            pos.trailing_stop = h - TRAILING_DIST_ATR * pos.atr_at_entry
    else:
        if h >= pos.stop_loss:
            return True, _slip(pos.stop_loss, -1, False), "stop_loss"
        if pos.trailing_active and h >= pos.trailing_stop:
            return True, _slip(pos.trailing_stop, -1, False), "trailing_stop"
        if l <= pos.profit_target:
            return True, _slip(pos.profit_target, -1, False), "profit_target"
        if not pos.trailing_active and pos.entry_price - l >= TRAILING_ACT_ATR * pos.atr_at_entry:
            pos.trailing_active = True
            pos.peak_favorable = l
            pos.trailing_stop = l + TRAILING_DIST_ATR * pos.atr_at_entry
        elif pos.trailing_active and l < pos.peak_favorable:
            pos.peak_favorable = l
            pos.trailing_stop = l + TRAILING_DIST_ATR * pos.atr_at_entry

    return False, 0.0, ""


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def run_backtest(bars: pd.DataFrame, n_contracts: int) -> dict:
    orb = OpeningRangeBreakoutRule(
        or_end_time=OR_END_TIME,
        min_or_bars=MIN_OR_BARS,
        min_range_atr=MIN_RANGE_ATR,
        entry_cutoff_time=ENTRY_CUTOFF,
        atr_period=ATR_PERIOD,
    )
    agg = SignalAggregator(primary_rule=orb, filter_rules=[], confirmation_rules=[], min_confirmations=0)

    rm = RiskManager(
        contracts=n_contracts, point_value=POINT_VALUE, tick_size=TICK_SIZE,
        tick_value=TICK_VALUE, max_daily_loss=MAX_DAILY_LOSS,
        per_trade_max_loss=PER_TRADE_MAX_LOSS,
        max_consecutive_losses=MAX_CONSECUTIVE_LOSSES,
        cooldown_bars=COOLDOWN_BARS, drawdown_buffer=DRAWDOWN_BUFFER,
    )
    rm.reset_all(STARTING_EQUITY)

    atr_s = atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars_needed = agg.required_bars()

    pos: Position | None = None
    trades: list[TradeRecord] = []
    eq_vals, eq_times = [STARTING_EQUITY], [bars.index[0]]
    equity = STARTING_EQUITY
    cur_date = None
    daily_pnl: dict = {}
    daily_trades: dict = {}
    trades_today = 0

    for i in range(min_bars_needed, len(bars)):
        bar = bars.iloc[i]
        bt = bars.index[i]
        bt_et = bt.tz_convert("US/Eastern") if bt.tzinfo else bt
        bdate = bt_et.date()

        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = rm.daily_pnl
            daily_trades[cur_date] = trades_today
            rm.reset_daily()
            trades_today = 0
        cur_date = bdate
        rm.tick_bar()

        is_last = (i + 1 >= len(bars)) or (
            (bars.index[i + 1].tz_convert("US/Eastern") if bars.index[i + 1].tzinfo
             else bars.index[i + 1]).date() != bdate
        )
        near_close = bt_et.hour == 15 and bt_et.minute >= 55
        sess_close = is_last or near_close

        if pos is not None:
            exited, exit_p, reason = _check_exit(pos, bar, i, sess_close)
            if exited:
                p = _pnl(pos.entry_price, exit_p, pos.direction, n_contracts)
                tr = TradeRecord(entry_bar=pos.entry_bar_idx, exit_bar=i,
                                 direction=pos.direction, entry_price=pos.entry_price,
                                 exit_price=exit_p, pnl=p, exit_reason=reason)
                trades.append(tr)
                rm.record_trade(tr)
                equity += p
                eq_vals.append(equity)
                eq_times.append(bt)
                pos = None

        if pos is None and not sess_close and trades_today < MAX_TRADES_PER_DAY:
            ok, _ = rm.can_trade()
            if ok:
                lookback = bars.iloc[max(0, i - min_bars_needed + 1): i + 1]
                dec = agg.evaluate(lookback)
                if dec.should_trade:
                    cur_atr = atr_s.iloc[i]
                    if not (np.isnan(cur_atr) or cur_atr <= 0):
                        ep = _slip(bar["close"], dec.direction, True)
                        sl = rm.compute_stop_price(ep, dec.direction, cur_atr, SL_MULT)
                        pt = rm.compute_target_price(ep, dec.direction, cur_atr, PT_MULT)
                        pos = Position(
                            direction=dec.direction, entry_price=ep, entry_bar_idx=i,
                            stop_loss=sl, profit_target=pt,
                            time_stop_bar=i + TIME_STOP_BARS, atr_at_entry=cur_atr,
                        )
                        trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = rm.daily_pnl
        daily_trades[cur_date] = trades_today

    # Summary
    if not trades:
        return {"n_contracts": n_contracts, "num_trades": 0}

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    total = sum(t.pnl for t in trades)
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses))

    eq = pd.Series(eq_vals, index=eq_times)
    max_dd = (eq - eq.cummax()).min()

    daily = pd.Series(daily_pnl)
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0.0

    reasons = defaultdict(int)
    for t in trades:
        reasons[t.exit_reason] += 1

    # Per-trade breakdown
    trade_list = [
        {"entry": t.entry_price, "exit": t.exit_price,
         "direction": "LONG" if t.direction == 1 else "SHORT",
         "pnl": round(t.pnl, 2), "exit_reason": t.exit_reason}
        for t in trades
    ]

    return {
        "n_contracts": n_contracts,
        "num_trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3),
        "total_pnl": round(total, 2),
        "avg_win": round(gp / len(wins), 2) if wins else 0,
        "avg_loss": round(-gl / len(losses), 2) if losses else 0,
        "profit_factor": round(gp / gl, 3) if gl > 0 else float("inf"),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "trades_per_day": round(len(trades) / max(1, len(daily_pnl)), 2),
        "exit_reasons": {k: round(v / len(trades) * 100, 1) for k, v in reasons.items()},
        "daily_pnl": {str(k): round(v, 2) for k, v in daily_pnl.items()},
        "trades": trade_list,
    }


# ---------------------------------------------------------------------------
# Monte Carlo — P(pass combine) in N days
# ---------------------------------------------------------------------------
def monte_carlo(trades: list[dict], n_paths: int = 10_000, horizon_days: int = 30) -> dict:
    pnls = [t["pnl"] for t in trades]
    if not pnls:
        return {}

    rng = np.random.default_rng(42)
    pnl_arr = np.array(pnls)
    trades_per_day = len(trades) / 30  # approx over OOS period

    n_trades_per_path = int(trades_per_day * horizon_days)
    paths = rng.choice(pnl_arr, size=(n_paths, max(1, n_trades_per_path)), replace=True)
    cum = np.cumsum(paths, axis=1)

    # Running max drawdown per path
    running_max = np.maximum.accumulate(cum, axis=1)
    drawdowns = cum - running_max
    worst_dd = drawdowns.min(axis=1)

    passed = ((cum[:, -1] >= COMBINE["profit_target"]) &
              (worst_dd >= -COMBINE["mll"]))

    days_to_pass = []
    for path in cum:
        hit = np.where(path >= COMBINE["profit_target"])[0]
        days_to_pass.append(int(hit[0] / trades_per_day) + 1 if len(hit) else horizon_days + 1)

    return {
        "p_pass": round(float(passed.mean()), 3),
        "median_days_to_pass": int(np.median(days_to_pass)),
        "p95_max_drawdown": round(float(np.percentile(worst_dd, 5)), 2),
        "expected_pnl_30d": round(float(cum[:, -1].mean()), 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load OOS data
    bars = pd.read_hdf(str(OOS_PATH), key="bars_5min")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    bars = bars.tz_convert("US/Eastern")
    rth = ((bars.index.hour > 9) | ((bars.index.hour == 9) & (bars.index.minute >= 30))) \
          & (bars.index.hour < 16)
    bars = bars.loc[rth]
    print(f"OOS data: {len(bars)} RTH bars  {bars.index[0].date()} → {bars.index[-1].date()}")

    print(f"\n{'='*60}")
    print(f"  Config: or_end={OR_END_TIME}  PT={PT_MULT}x  SL={SL_MULT}x")
    print(f"{'='*60}")

    # Run for 2 and 3 contracts to compare
    r2 = run_backtest(bars, n_contracts=2)
    r3 = run_backtest(bars, n_contracts=3)

    print(f"\n{'Metric':<22} {'2 contracts':>14} {'3 contracts':>14}")
    print("-" * 52)
    metrics = ["num_trades", "win_rate", "total_pnl", "avg_win", "avg_loss",
               "profit_factor", "sharpe", "max_drawdown", "trades_per_day"]
    for m in metrics:
        v2 = r2.get(m, "—")
        v3 = r3.get(m, "—")
        if m in ("win_rate",):
            print(f"  {m:<20} {v2:>13.1%} {v3:>13.1%}")
        elif isinstance(v2, float):
            print(f"  {m:<20} {v2:>13.2f} {v3:>13.2f}")
        else:
            print(f"  {m:<20} {v2!s:>14} {v3!s:>14}")

    print(f"\n  Exit reasons (3 contracts): {r3.get('exit_reasons', {})}")

    # Monte Carlo for 3 contracts
    mc = monte_carlo(r3.get("trades", []))
    if mc:
        print(f"\n{'='*60}")
        print(f"  Monte Carlo — 3 contracts (10k paths, 30-day horizon)")
        print(f"{'='*60}")
        print(f"  P(pass combine)     : {mc['p_pass']:.1%}")
        print(f"  Median days to pass : {mc['median_days_to_pass']}")
        print(f"  p95 max drawdown    : ${mc['p95_max_drawdown']:,.0f}")
        print(f"  Expected PnL/30d    : ${mc['expected_pnl_30d']:,.0f}")
        print(f"\n  Combine limits: PT=${COMBINE['profit_target']:,}  MLL=${COMBINE['mll']:,}")

    # Save
    output = {
        "config": {
            "or_end_time": OR_END_TIME, "min_or_bars": MIN_OR_BARS,
            "pt_mult": PT_MULT, "sl_mult": SL_MULT,
            "entry_cutoff": ENTRY_CUTOFF,
        },
        "2_contracts": {k: v for k, v in r2.items() if k != "trades"},
        "3_contracts": {k: v for k, v in r3.items() if k != "trades"},
        "monte_carlo_3c": mc,
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved → {RESULTS_OUT}")


if __name__ == "__main__":
    main()
