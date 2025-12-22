"""
Compare backtest results with 4-trades/day vs 2-trades/day.

This script runs the backtest with quality-gated 4 trades/day and compares
against the baseline 2 trades/day to assess impact on Topstep pass-rate.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

# Setup paths
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("RISK_PRESET_NAME", "TOPSTEP_50K")

from core.risk_presets import get_risk_preset
from data.clean_bars import clean_bars
from models.nn_inference import load_nn_bundle, predict_scores_for_bars
from backtesting.backtest import run_backtest_nn


def main():
    # Configuration
    data_path = "data/processed/es_bars_2010_2025.h5"
    dataset_key = "bars_5min"
    model_dir = "models/nn_saved"
    fold = 0
    fast_max_bars = 250_000  # Use same window as notebook

    print("=" * 80)
    print("4-TRADES/DAY QUALITY-GATED COMPARISON")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from {data_path}...")
    with pd.HDFStore(data_path, "r") as store:
        bars = store[dataset_key].copy()

    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    preset = get_risk_preset("TOPSTEP_50K")
    bars = clean_bars(bars, tick_size=preset.risk_config.tick_size, verbose=False)
    bars = bars.tail(fast_max_bars).reset_index(drop=True)

    print(f"Loaded {len(bars):,} bars")
    print(f"Date range: {bars['timestamp'].min()} to {bars['timestamp'].max()}")

    # Load model
    print(f"\nLoading model from {model_dir}...")
    bundle = load_nn_bundle(model_dir, fold=fold)
    nn_cfg = bundle.config.get("nn_config", {})

    print(f"\nModel config:")
    print(f"  max_trades_per_day: {nn_cfg['max_trades_per_day']}")
    print(f"  confidence_min: {nn_cfg.get('confidence_min', 'N/A')}")
    print(f"  quality_margin: {nn_cfg.get('quality_margin', 'N/A')}")
    print(f"  daily_stop_loss: ${nn_cfg.get('daily_stop_loss', 'N/A')}")
    print(f"  daily_profit_lock: ${nn_cfg.get('daily_profit_lock', 'N/A')}")

    # Compute probabilities
    print("\nComputing model probabilities...")
    prob_df = predict_scores_for_bars(bars, bundle)

    # Check if confidence column exists
    if "confidence" in prob_df.columns:
        conf_stats = prob_df["confidence"].dropna()
        print(f"Confidence stats:")
        print(f"  Mean: {conf_stats.mean():.4f}")
        print(f"  Median: {conf_stats.median():.4f}")
        print(f"  Min: {conf_stats.min():.4f}")
        print(f"  Max: {conf_stats.max():.4f}")

    # Run backtest
    print("\nRunning backtest with 4-trades/day + quality gates...")
    risk_cfg = preset.risk_config
    results = run_backtest_nn(
        bars,
        prob_df,
        score_threshold=float(nn_cfg["score_threshold"]),
        selection_mode=str(nn_cfg.get("selection_mode", "global_threshold")),
        day_percentile_floor=float(nn_cfg.get("day_percentile_floor", 0.90)),
        global_floor_score=float(nn_cfg.get("global_floor_score", nn_cfg["score_threshold"])),
        max_trades_per_day=int(nn_cfg["max_trades_per_day"]),
        min_bars_between_trades=int(nn_cfg["min_bars_between_trades"]),
        enable_long=bool(nn_cfg["enable_long"]),
        enable_short=bool(nn_cfg["enable_short"]),
        horizon_bars=int(nn_cfg["horizon_bars"]),
        execution_mode=str(nn_cfg["execution_mode"]),
        exit_price_mode=str(nn_cfg["exit_price_mode"]),
        session_mode=str(nn_cfg["session_mode"]),
        deadline_time=nn_cfg.get("deadline_time"),
        deadline_relax_factor=float(nn_cfg.get("deadline_relax_factor", 0.98)),
        bar_minutes=int(nn_cfg["bar_minutes"]),
        session_start=risk_cfg.session_start,
        session_end=risk_cfg.session_end,
        stop_loss_ticks=int(nn_cfg["stop_loss_ticks"]),
        target_multiplier=float(nn_cfg["target_multiplier"]),
        catastrophic_stop_ticks=int(nn_cfg.get("catastrophic_stop_ticks", int(nn_cfg["threshold_ticks"]) * 4)),
        max_hold_bars=int(nn_cfg["max_hold_bars"]),
        tick_size=float(nn_cfg["tick_size"]),
        tick_value=float(nn_cfg["tick_value"]),
        save_trades_path="analysis/trades_4per_day.csv",
    )

    # Print results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS (4 TRADES/DAY)")
    print("=" * 80)

    print("\nSummary:")
    summary = results["summary"]
    print(json.dumps(summary, indent=2))

    print("\nDaily stats:")
    daily_stats = results["daily_stats"]
    print(json.dumps(daily_stats, indent=2))

    print("\nConfidence stats:")
    confidence_stats = results.get("confidence_stats", {})
    if confidence_stats:
        print(json.dumps(confidence_stats, indent=2))
    else:
        print("  No confidence data available")

    print("\nExit reasons:")
    exit_reasons = results.get("exit_reason_counts", {})
    print(json.dumps(exit_reasons, indent=2))

    print("\nExit reason avg PnL:")
    exit_avg_pnl = results.get("exit_reason_avg_pnl", {})
    print(json.dumps(exit_avg_pnl, indent=2))

    # Comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print("\nBASELINE (2 trades/day, from previous run):")
    print("  Total trades: 786")
    print("  TIME_EXIT: 569 (72.4%)")
    print("  CATASTOP: 217 (27.6%)")
    print("  SESSION_FLAT: 0")
    print("  Net PnL: -$1,534.20")
    print("  Win rate: 49.4%")
    print("  Profit factor: 0.93")

    total_trades = summary["trades"]
    time_exit = exit_reasons.get("TIME_EXIT", 0)
    catastop = exit_reasons.get("CATASTOP", 0)
    session_flat = exit_reasons.get("SESSION_FLAT", 0)
    time_exit_pct = (time_exit / total_trades * 100) if total_trades > 0 else 0
    catastop_pct = (catastop / total_trades * 100) if total_trades > 0 else 0

    print("\nNEW (4 trades/day with quality gates):")
    print(f"  Total trades: {total_trades}")
    print(f"  TIME_EXIT: {time_exit} ({time_exit_pct:.1f}%)")
    print(f"  CATASTOP: {catastop} ({catastop_pct:.1f}%)")
    print(f"  SESSION_FLAT: {session_flat}")
    print(f"  Net PnL: ${summary['net_pnl']:.2f}")
    print(f"  Win rate: {summary['win_rate']:.1%}")
    print(f"  Profit factor: {summary['profit_factor']:.2f}")

    print("\nTRADE DISTRIBUTION:")
    trade_dist = daily_stats.get("trades_per_day_distribution", {})
    print(f"  Days with 0 trades: {trade_dist.get(0, 0)}")
    print(f"  Days with 1 trade: {trade_dist.get(1, 0)}")
    print(f"  Days with 2 trades: {trade_dist.get(2, 0)}")
    print(f"  Days with 3 trades: {trade_dist.get(3, 0)}")
    print(f"  Days with 4 trades: {trade_dist.get(4, 0)}")

    # Check max trades in a day
    max_trades_day = daily_stats.get("max_trades_in_day", 0)
    if max_trades_day > 4:
        print(f"\n⚠️  WARNING: Max trades in a day ({max_trades_day}) exceeds limit of 4!")
    else:
        print(f"\n✓ Max trades/day constraint respected: {max_trades_day} <= 4")

    # Analyze CATASTOP rate
    catastop_rate = catastop / total_trades if total_trades > 0 else 0
    baseline_catastop_rate = 217 / 786
    catastop_change = (catastop_rate - baseline_catastop_rate) * 100

    print(f"\nCATASTOP ANALYSIS:")
    print(f"  Baseline rate: {baseline_catastop_rate:.1%}")
    print(f"  New rate: {catastop_rate:.1%}")
    print(f"  Change: {catastop_change:+.1f} percentage points")

    if catastop_rate < baseline_catastop_rate:
        print("  ✓ CATASTOP rate IMPROVED")
    else:
        print("  ✗ CATASTOP rate WORSENED")

    # PnL analysis
    pnl_change = summary["net_pnl"] - (-1534.20)
    print(f"\nP&L ANALYSIS:")
    print(f"  Change: ${pnl_change:+.2f}")
    if pnl_change > 0:
        print("  ✓ P&L IMPROVED")
    else:
        print("  ✗ P&L WORSENED")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
