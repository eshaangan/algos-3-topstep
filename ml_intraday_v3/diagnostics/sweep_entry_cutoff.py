"""
Entry Cutoff + OR Window Sweep
==============================
Holds the confirmed best config fixed and sweeps:
  - entry_cutoff_time: 11:00 → 15:00 ET (what the user asked)
  - or_end_time: 09:44, 09:55, 10:14, 10:30 (shorter/wider OR windows)

All other params stay at confirmed best values:
  pt=2.0x ATR, sl=1.5x ATR, n=2 MES, trailing disabled.

Usage:
    cd "algos 3 topstep"
    python ml_intraday_v3/diagnostics/sweep_entry_cutoff.py
"""
from __future__ import annotations

import json
import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
_RBV1_DIR = _PROJECT_ROOT / "rule_based_v1"
_DIAG_DIR = _HERE.parent

for _p in [str(_PROJECT_ROOT), str(_RBV1_DIR), str(_DIAG_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.backtest_engine import BacktestEngine
from engine.signal_aggregator import SignalAggregator
from engine.risk_manager import RiskManager
from rules.opening_range import OpeningRangeBreakoutRule
from rules.time_of_day import TimeOfDayRule
from utils.data_loader import load_bars
from topstep_combine_simulator import TopstepCombineSimulator

logging.basicConfig(level=logging.WARNING)

DATA_PATH = _PROJECT_ROOT / "data" / "processed" / "jan_feb_2026_oos_test.h5"

# Confirmed best values — held fixed
BASE = dict(
    min_range_atr=0.3,
    pt_atr_mult=2.0,
    sl_atr_mult=1.5,
    n_contracts=2,
    trailing_activation_atr=999.0,
    trailing_distance_atr=0.75,
    atr_period=14,
    commission_per_side=0.62,
    slippage_ticks=1,
    time_stop_bars=24,
)

RISK = dict(
    point_value=5.0,
    tick_size=0.25,
    tick_value=1.25,
    max_daily_loss=-950.0,
    per_trade_max_loss=1000.0,
    max_consecutive_losses=3,
    cooldown_bars=3,
    flatten_minutes_before_close=5,
    drawdown_buffer=1800.0,
)

COMBINE = dict(
    account_size=50_000,
    profit_target=3_000,
    max_trailing_drawdown=2_000,
    max_daily_loss=1_000,
    consistency_pct=0.30,
    min_trading_days=5,
)

# Sweep axes
ENTRY_CUTOFFS = ["11:00", "12:00", "13:00", "14:00", "15:00"]
OR_WINDOWS    = ["09:44", "09:55", "10:14", "10:30"]   # 3, 6, 9, 13 bars


def run_config(bars, or_end, cutoff, simulator):
    primary = OpeningRangeBreakoutRule(
        or_end_time=or_end,
        min_or_bars=2,
        min_range_atr=BASE["min_range_atr"],
        entry_cutoff_time=cutoff,
        atr_period=BASE["atr_period"],
        use_close_for_signal=True,
    )
    agg = SignalAggregator(
        primary_rule=primary,
        filter_rules=[TimeOfDayRule(session_start="09:35", session_end="15:45",
                                    lunch_filter_enabled=False)],
        confirmation_rules=[],
        min_confirmations=0,
    )
    rm = RiskManager(
        contracts=BASE["n_contracts"],
        **{k: RISK[k] for k in RISK},
    )
    engine = BacktestEngine(
        aggregator=agg,
        risk_manager=rm,
        commission_per_side=BASE["commission_per_side"],
        slippage_ticks=BASE["slippage_ticks"],
        profit_target_atr=BASE["pt_atr_mult"],
        stop_loss_atr=BASE["sl_atr_mult"],
        time_stop_bars=BASE["time_stop_bars"],
        trailing_activation_atr=BASE["trailing_activation_atr"],
        trailing_distance_atr=BASE["trailing_distance_atr"],
        atr_period=BASE["atr_period"],
    )
    result = engine.run(bars, starting_equity=50_000.0)
    s = result.summary()

    trade_pnls = [t.pnl for t in result.trades]
    if len(trade_pnls) >= 5:
        mc = simulator.monte_carlo(
            trade_pnl_list=trade_pnls,
            n_paths=10_000,
            trades_per_day_range=(1, 4),
            max_days=40,
            seed=42,
        )
    else:
        mc = {"p_pass": 0.0, "median_days": None, "p95_max_drawdown": None,
              "p_fail_trailing": None}

    exit_counts = {}
    for t in result.trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    return {
        "or_end": or_end,
        "cutoff": cutoff,
        "n_trades": s["num_trades"],
        "win_rate": s.get("win_rate", 0.0),
        "total_pnl": s.get("total_pnl", 0.0),
        "avg_pnl": s.get("avg_trade_pnl", 0.0),
        "sharpe": s.get("sharpe_ratio", 0.0),
        "max_dd": s.get("max_drawdown", 0.0),
        "p_pass": mc["p_pass"],
        "med_days": mc.get("median_days"),
        "p95_dd": mc.get("p95_max_drawdown"),
        "p_fail_dd": mc.get("p_fail_trailing"),
        "exits": exit_counts,
    }


def main():
    print(f"Loading bars from {DATA_PATH}")
    bars = load_bars(str(DATA_PATH), key="bars_5min",
                     start_date="2026-01-01", end_date="2026-02-10")
    print(f"Loaded {len(bars):,} bars  {bars.index[0]} → {bars.index[-1]}\n")

    simulator = TopstepCombineSimulator(**COMBINE)
    results = []

    total = len(OR_WINDOWS) * len(ENTRY_CUTOFFS)
    for i, (or_end, cutoff) in enumerate(product(OR_WINDOWS, ENTRY_CUTOFFS), 1):
        print(f"  [{i:2d}/{total}] or_end={or_end}  cutoff={cutoff}", end="  ", flush=True)
        r = run_config(bars, or_end, cutoff, simulator)
        results.append(r)
        print(f"trades={r['n_trades']:2d}  wr={r['win_rate']:.1%}  "
              f"pnl=${r['total_pnl']:+.0f}  P(pass)={r['p_pass']:.1%}")

    # -----------------------------------------------------------------------
    # Print comparison tables
    # -----------------------------------------------------------------------
    CONFIRMED = {"or_end": "09:55", "cutoff": "12:00"}   # baseline

    print("\n" + "=" * 90)
    print("ENTRY CUTOFF SWEEP  (or_end fixed = 09:55 ET, confirmed best OR window)")
    print("=" * 90)
    print(f"{'Cutoff':>8}  {'Trades':>6}  {'WR':>6}  {'PnL':>8}  {'Avg':>7}  "
          f"{'Sharpe':>7}  {'MaxDD':>8}  {'P(pass)':>8}  {'MedDays':>8}  {'p95DD':>8}")
    print("-" * 90)
    base_row = None
    for r in results:
        if r["or_end"] != "09:55":
            continue
        marker = " ◄ CURRENT" if r["cutoff"] == "12:00" else ""
        if r["cutoff"] == "12:00":
            base_row = r
        print(f"  {r['cutoff']:>6}  {r['n_trades']:>6}  {r['win_rate']:>5.1%}  "
              f"${r['total_pnl']:>+7.0f}  ${r['avg_pnl']:>+6.0f}  "
              f"{r['sharpe']:>7.2f}  ${r['max_dd']:>+7.0f}  "
              f"{r['p_pass']:>7.1%}  {str(r['med_days'] or '—'):>8}  "
              f"${r['p95_dd'] or 0:>7.0f}{marker}")

    print("\n" + "=" * 90)
    print("OR WINDOW SWEEP  (cutoff fixed = 12:00 ET, to isolate that effect)")
    print("=" * 90)
    print(f"{'OR end':>8}  {'OR bars':>7}  {'Trades':>6}  {'WR':>6}  {'PnL':>8}  "
          f"{'Avg':>7}  {'Sharpe':>7}  {'P(pass)':>8}  {'MedDays':>8}")
    print("-" * 90)
    or_bars_map = {"09:44": 3, "09:55": 6, "10:14": 9, "10:30": 13}
    for r in results:
        if r["cutoff"] != "12:00":
            continue
        marker = " ◄ CURRENT" if r["or_end"] == "09:55" else ""
        print(f"  {r['or_end']:>6}  {or_bars_map[r['or_end']]:>7}  {r['n_trades']:>6}  "
              f"{r['win_rate']:>5.1%}  ${r['total_pnl']:>+7.0f}  ${r['avg_pnl']:>+6.0f}  "
              f"{r['sharpe']:>7.2f}  {r['p_pass']:>7.1%}  "
              f"{str(r['med_days'] or '—'):>8}{marker}")

    print("\n" + "=" * 90)
    print("BEST COMBINATIONS  (all OR windows × all cutoffs, ranked by P(pass))")
    print("=" * 90)
    ranked = sorted(results, key=lambda x: (-x["p_pass"], -x["sharpe"]))
    print(f"{'OR end':>8}  {'Cutoff':>8}  {'Trades':>6}  {'WR':>6}  {'PnL':>8}  "
          f"{'Sharpe':>7}  {'P(pass)':>8}  {'MedDays':>8}  {'p95DD':>8}")
    print("-" * 90)
    for r in ranked[:10]:
        marker = " ◄ CURRENT" if (r["or_end"] == "09:55" and r["cutoff"] == "12:00") else ""
        print(f"  {r['or_end']:>6}  {r['cutoff']:>8}  {r['n_trades']:>6}  "
              f"{r['win_rate']:>5.1%}  ${r['total_pnl']:>+7.0f}  "
              f"{r['sharpe']:>7.2f}  {r['p_pass']:>7.1%}  "
              f"{str(r['med_days'] or '—'):>8}  ${r['p95_dd'] or 0:>7.0f}{marker}")

    # Save
    out = _DIAG_DIR / "entry_cutoff_sweep_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
