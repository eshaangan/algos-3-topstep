"""
Tune dynamic catastrophic stop multiplier.

Tests multipliers: 2.0x, 2.5x, 3.0x, 3.5x
Reports optimal multiplier for CATASTOP reduction while preserving profitability.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

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
from analysis.monte_carlo_combine import simulate_combine


def run_with_multiplier(multiplier, bars, prob_df, bundle, preset):
    """Run backtest with specific ATR multiplier."""

    nn_cfg = bundle.config.get("nn_config", {})
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
        use_dynamic_catastop=True,
        catastop_atr_multiplier=multiplier,
        catastop_min_ticks=int(nn_cfg.get("catastop_min_ticks", 24)),
        catastop_max_ticks=int(nn_cfg.get("catastop_max_ticks", 72)),
        save_trades_path=None,
    )

    trades_df = pd.DataFrame(results["trades"])
    summary = results["summary"]
    exit_reasons = results.get("exit_reason_counts", {})
    exit_avg_pnl = results.get("exit_reason_avg_pnl", {})

    # Monte Carlo
    mc_results = simulate_combine(
        trades_df,
        starting_balance=risk_cfg.starting_balance,
        profit_target=5000.0,
        daily_loss_limit=risk_cfg.max_daily_loss,
        trailing_drawdown=risk_cfg.trailing_drawdown,
        runs=10_000,
        seed=42,
        max_days=252,
    )

    total = sum(exit_reasons.values())
    catastop_count = exit_reasons.get("CATASTOP", 0)
    catastop_rate = (catastop_count / total * 100) if total > 0 else 0
    timeexit_count = exit_reasons.get("TIME_EXIT", 0)

    return {
        "multiplier": multiplier,
        "trades": summary["trades"],
        "catastop_count": catastop_count,
        "catastop_rate": catastop_rate,
        "timeexit_count": timeexit_count,
        "catastop_avg_pnl": exit_avg_pnl.get("CATASTOP", 0),
        "timeexit_avg_pnl": exit_avg_pnl.get("TIME_EXIT", 0),
        "win_rate": summary["win_rate"],
        "profit_factor": summary["profit_factor"],
        "avg_pnl": summary["avg_pnl"],
        "net_pnl": summary["net_pnl"],
        "max_drawdown": summary["max_drawdown"],
        "pass_rate": mc_results["pass_rate"] * 100,
        "median_days": mc_results["days_to_pass"].get("p50", 0),
        "fail_trailing_dd": mc_results["fail_reasons"].get("trailing_drawdown", 0),
        "fail_max_days": mc_results["fail_reasons"].get("max_days", 0),
    }


def main():
    # Configuration
    data_path = "data/processed/es_bars_2010_2025.h5"
    dataset_key = "bars_5min"
    model_dir = "models/nn_saved"
    fold = 0
    fast_max_bars = 250_000

    print("=" * 100)
    print("DYNAMIC STOP MULTIPLIER TUNING")
    print("=" * 100)
    print("\nTesting multipliers: 2.0x, 2.5x, 3.0x, 3.5x")
    print("Objective: Minimize CATASTOP while preserving/improving profitability")

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

    # Load model
    print(f"Loading model from {model_dir}...")
    bundle = load_nn_bundle(model_dir, fold=fold)

    # Compute probabilities
    print("Computing model probabilities...")
    prob_df = predict_scores_for_bars(bars, bundle)

    # Run baseline (fixed stop)
    print("\n" + "=" * 100)
    print("BASELINE: Fixed 48-tick stop")
    print("=" * 100)

    nn_cfg = bundle.config.get("nn_config", {})
    risk_cfg = preset.risk_config

    baseline_results = run_backtest_nn(
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
        catastrophic_stop_ticks=48,  # Fixed
        max_hold_bars=int(nn_cfg["max_hold_bars"]),
        tick_size=float(nn_cfg["tick_size"]),
        tick_value=float(nn_cfg["tick_value"]),
        use_dynamic_catastop=False,
        save_trades_path=None,
    )

    baseline_trades = pd.DataFrame(baseline_results["trades"])
    baseline_summary = baseline_results["summary"]
    baseline_exit = baseline_results.get("exit_reason_counts", {})

    baseline_mc = simulate_combine(
        baseline_trades,
        starting_balance=risk_cfg.starting_balance,
        profit_target=5000.0,
        daily_loss_limit=risk_cfg.max_daily_loss,
        trailing_drawdown=risk_cfg.trailing_drawdown,
        runs=10_000,
        seed=42,
        max_days=252,
    )

    baseline_total = sum(baseline_exit.values())
    baseline_catastop_rate = (baseline_exit.get("CATASTOP", 0) / baseline_total * 100) if baseline_total > 0 else 0

    print(f"Trades: {baseline_summary['trades']}")
    print(f"CATASTOP: {baseline_exit.get('CATASTOP', 0)} ({baseline_catastop_rate:.1f}%)")
    print(f"Profit Factor: {baseline_summary['profit_factor']:.2f}")
    print(f"Avg PnL: ${baseline_summary['avg_pnl']:.2f}")
    print(f"Pass Rate: {baseline_mc['pass_rate']*100:.2f}%")

    # Test multipliers
    multipliers = [2.0, 2.5, 3.0, 3.5]
    results = []

    for mult in multipliers:
        print(f"\n" + "=" * 100)
        print(f"Testing {mult:.1f}x ATR multiplier...")
        print("=" * 100)

        result = run_with_multiplier(mult, bars, prob_df, bundle, preset)
        results.append(result)

        print(f"Trades: {result['trades']}")
        print(f"CATASTOP: {result['catastop_count']} ({result['catastop_rate']:.1f}%)")
        print(f"Profit Factor: {result['profit_factor']:.2f}")
        print(f"Avg PnL: ${result['avg_pnl']:.2f}")
        print(f"Pass Rate: {result['pass_rate']:.2f}%")

    # Comparison table
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)

    print(f"\n{'Multiplier':<12} {'Trades':<8} {'CATASTOP':<10} {'Rate%':<8} {'PF':<8} {'AvgPnL':<10} {'NetPnL':<12} {'PassRate%':<12}")
    print("-" * 100)

    # Baseline
    print(f"{'FIXED 48':<12} {baseline_summary['trades']:<8} {baseline_exit.get('CATASTOP', 0):<10} {baseline_catastop_rate:<8.1f} {baseline_summary['profit_factor']:<8.2f} ${baseline_summary['avg_pnl']:<9.2f} ${baseline_summary['net_pnl']:<11.2f} {baseline_mc['pass_rate']*100:<12.2f}")

    # Dynamic results
    for res in results:
        print(f"{res['multiplier']:.1f}x ATR    {res['trades']:<8} {res['catastop_count']:<10} {res['catastop_rate']:<8.1f} {res['profit_factor']:<8.2f} ${res['avg_pnl']:<9.2f} ${res['net_pnl']:<11.2f} {res['pass_rate']:<12.2f}")

    # Detailed comparison vs baseline
    print("\n" + "=" * 100)
    print("CHANGE vs BASELINE (Fixed 48-tick)")
    print("=" * 100)

    print(f"\n{'Multiplier':<12} {'CATASTOP Δ':<15} {'PF Δ':<12} {'AvgPnL Δ':<15} {'PassRate Δ':<15}")
    print("-" * 100)

    for res in results:
        catastop_delta = res['catastop_rate'] - baseline_catastop_rate
        pf_delta = res['profit_factor'] - baseline_summary['profit_factor']
        avgpnl_delta = res['avg_pnl'] - baseline_summary['avg_pnl']
        passrate_delta = res['pass_rate'] - (baseline_mc['pass_rate'] * 100)

        catastop_str = f"{catastop_delta:+.1f} pp"
        pf_str = f"{pf_delta:+.2f}"
        avgpnl_str = f"${avgpnl_delta:+.2f}"
        passrate_str = f"{passrate_delta:+.2f} pp"

        print(f"{res['multiplier']:.1f}x ATR    {catastop_str:<15} {pf_str:<12} {avgpnl_str:<15} {passrate_str:<15}")

    # Find optimal
    print("\n" + "=" * 100)
    print("OPTIMAL MULTIPLIER ANALYSIS")
    print("=" * 100)

    # Rank by different criteria
    by_catastop = sorted(results, key=lambda x: x['catastop_rate'])
    by_pf = sorted(results, key=lambda x: x['profit_factor'], reverse=True)
    by_pass_rate = sorted(results, key=lambda x: x['pass_rate'], reverse=True)

    print(f"\nLowest CATASTOP: {by_catastop[0]['multiplier']:.1f}x ({by_catastop[0]['catastop_rate']:.1f}%)")
    print(f"Highest Profit Factor: {by_pf[0]['multiplier']:.1f}x ({by_pf[0]['profit_factor']:.2f})")
    print(f"Highest Pass Rate: {by_pass_rate[0]['multiplier']:.1f}x ({by_pass_rate[0]['pass_rate']:.2f}%)")

    # Composite score (weighted combination)
    # Lower CATASTOP is better, higher PF is better, higher pass rate is better
    for res in results:
        # Normalize metrics (0-1 scale)
        catastop_norm = 1.0 - (res['catastop_rate'] / 100.0)  # Lower is better
        pf_norm = min(res['profit_factor'] / 2.0, 1.0)  # Higher is better, cap at 2.0
        pass_norm = res['pass_rate'] / 100.0  # Higher is better

        # Weighted score: 40% CATASTOP, 40% PF, 20% pass rate
        res['composite_score'] = (0.4 * catastop_norm) + (0.4 * pf_norm) + (0.2 * pass_norm)

    by_composite = sorted(results, key=lambda x: x['composite_score'], reverse=True)

    best = by_composite[0]
    print(f"\n🎯 RECOMMENDED MULTIPLIER: {best['multiplier']:.1f}x ATR")
    print(f"   CATASTOP: {best['catastop_rate']:.1f}% (vs {baseline_catastop_rate:.1f}% baseline)")
    print(f"   Profit Factor: {best['profit_factor']:.2f} (vs {baseline_summary['profit_factor']:.2f} baseline)")
    print(f"   Avg PnL: ${best['avg_pnl']:.2f} (vs ${baseline_summary['avg_pnl']:.2f} baseline)")
    print(f"   Pass Rate: {best['pass_rate']:.2f}% (vs {baseline_mc['pass_rate']*100:.2f}% baseline)")
    print(f"   Composite Score: {best['composite_score']:.3f}")

    # Save results
    output = {
        "baseline": {
            "stop_type": "fixed_48_ticks",
            "trades": baseline_summary["trades"],
            "catastop_rate": baseline_catastop_rate,
            "profit_factor": baseline_summary["profit_factor"],
            "avg_pnl": baseline_summary["avg_pnl"],
            "pass_rate": baseline_mc["pass_rate"] * 100,
        },
        "multiplier_tests": results,
        "recommended": {
            "multiplier": best["multiplier"],
            "composite_score": best["composite_score"],
        }
    }

    output_path = "analysis/multiplier_tuning_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
