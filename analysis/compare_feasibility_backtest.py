"""
Compare backtest results before/after implementing feasibility check.

This script runs the backtest with the NN model and shows the impact of the
RTH bars-left feasibility rule on SESSION_FLAT exits.
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
    fast_max_bars = 250_000  # Use same window as notebook for comparison

    print("=" * 80)
    print("FEASIBILITY CHECK IMPACT ANALYSIS")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from {data_path}...")
    with pd.HDFStore(data_path, "r") as store:
        bars = store[dataset_key].copy()

    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    preset = get_risk_preset("TOPSTEP_50K")
    bars = clean_bars(bars, tick_size=preset.risk_config.tick_size, verbose=False)

    # Use same window as notebook for comparison
    bars = bars.tail(fast_max_bars).reset_index(drop=True)

    print(f"Loaded {len(bars):,} bars")
    print(f"Date range: {bars['timestamp'].min()} to {bars['timestamp'].max()}")

    # Load model
    print(f"\nLoading model from {model_dir}...")
    bundle = load_nn_bundle(model_dir, fold=fold)
    nn_cfg = bundle.config.get("nn_config", {})

    # Compute probabilities
    print("Computing model probabilities...")
    prob_df = predict_scores_for_bars(bars, bundle)

    # Run backtest
    print("\nRunning backtest with feasibility check...")
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
        save_trades_path=None,
    )

    # Print results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS WITH FEASIBILITY CHECK")
    print("=" * 80)

    print("\nSummary:")
    print(json.dumps(results["summary"], indent=2))

    print("\nExit reasons:")
    exit_reasons = results.get("exit_reason_counts", {})
    print(json.dumps(exit_reasons, indent=2))

    print("\nExit reason avg PnL:")
    exit_avg_pnl = results.get("exit_reason_avg_pnl", {})
    print(json.dumps(exit_avg_pnl, indent=2))

    print("\nFeasibility stats:")
    feasibility_stats = results.get("feasibility_stats", {})
    print(json.dumps(feasibility_stats, indent=2))

    # Comparison summary
    print("\n" + "=" * 80)
    print("IMPACT SUMMARY")
    print("=" * 80)

    print("\nBEFORE (from notebook):")
    print("  Total trades: 489")
    print("  SESSION_FLAT exits: 407 (83.2%)")
    print("  TIME_EXIT: 62 (12.7%)")
    print("  CATASTOP: 20 (4.1%)")
    print("  SESSION_FLAT avg PnL: -$4.40")
    print("  Overall net PnL: -$2,017.05")
    print("  Win rate: 37.2%")
    print("  Profit factor: 0.66")

    total_trades = results["summary"]["trades"]
    session_flat = exit_reasons.get("SESSION_FLAT", 0)
    time_exit = exit_reasons.get("TIME_EXIT", 0)
    catastop = exit_reasons.get("CATASTOP", 0)
    session_flat_pct = (session_flat / total_trades * 100) if total_trades > 0 else 0
    time_exit_pct = (time_exit / total_trades * 100) if total_trades > 0 else 0

    print("\nAFTER (with feasibility check):")
    print(f"  Total trades: {total_trades}")
    print(f"  SESSION_FLAT exits: {session_flat} ({session_flat_pct:.1f}%)")
    print(f"  TIME_EXIT: {time_exit} ({time_exit_pct:.1f}%)")
    print(f"  CATASTOP: {catastop}")
    print(f"  SESSION_FLAT avg PnL: ${exit_avg_pnl.get('SESSION_FLAT', 0):.2f}")
    print(f"  Overall net PnL: ${results['summary']['net_pnl']:.2f}")
    print(f"  Win rate: {results['summary']['win_rate']:.1%}")
    print(f"  Profit factor: {results['summary']['profit_factor']:.2f}")
    print(f"  Rejected by feasibility: {feasibility_stats['rejected_by_feasibility']}")
    print(f"  Feasibility rejection rate: {feasibility_stats['feasibility_rejection_rate']:.2%}")

    print("\nCHANGES:")
    trades_delta = total_trades - 489
    session_flat_delta = session_flat - 407
    time_exit_delta = time_exit - 62
    pnl_delta = results["summary"]["net_pnl"] - (-2017.05)

    print(f"  Total trades: {trades_delta:+d} ({trades_delta/489*100:+.1f}%)")
    print(f"  SESSION_FLAT exits: {session_flat_delta:+d} ({session_flat_delta/407*100:+.1f}%)")
    print(f"  TIME_EXIT: {time_exit_delta:+d}")
    print(f"  Net PnL improvement: ${pnl_delta:+.2f}")

    if session_flat_delta < 0:
        print(f"\n✓ SUCCESS: Reduced SESSION_FLAT exits by {abs(session_flat_delta)} ({abs(session_flat_delta)/407*100:.1f}%)")
    else:
        print(f"\n✗ UNEXPECTED: SESSION_FLAT exits increased by {session_flat_delta}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
