"""
Run backtest and Monte Carlo with dynamic catastrophic stop sizing.

This script:
1. Runs backtest with dynamic stops (2.0x ATR, clamped 24-72 ticks)
2. Compares against baseline (fixed 48-tick stop)
3. Runs Monte Carlo simulation
4. Reports comprehensive comparison
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


def run_and_analyze(use_dynamic, label):
    """Run backtest with or without dynamic stops."""
    # Configuration
    data_path = "data/processed/es_bars_2010_2025.h5"
    dataset_key = "bars_5min"
    model_dir = "models/nn_saved"
    fold = 0
    fast_max_bars = 250_000

    print(f"\n{'=' * 100}")
    print(f"{label.upper()}")
    print("=" * 100)

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
    nn_cfg = bundle.config.get("nn_config", {})

    # Compute probabilities
    print("Computing model probabilities...")
    prob_df = predict_scores_for_bars(bars, bundle)

    # Run backtest
    print(f"\nRunning backtest ({'DYNAMIC' if use_dynamic else 'FIXED'} stops)...")
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
        use_dynamic_catastop=use_dynamic,
        catastop_atr_multiplier=float(nn_cfg.get("catastop_atr_multiplier", 2.0)),
        catastop_min_ticks=int(nn_cfg.get("catastop_min_ticks", 24)),
        catastop_max_ticks=int(nn_cfg.get("catastop_max_ticks", 72)),
        save_trades_path=None,
    )

    trades_df = pd.DataFrame(results["trades"])
    summary = results["summary"]
    daily_stats = results["daily_stats"]
    exit_reasons = results.get("exit_reason_counts", {})
    exit_avg_pnl = results.get("exit_reason_avg_pnl", {})
    conf_stats = results.get("confidence_stats", {})

    # Print results
    print("\n" + "=" * 100)
    print("BACKTEST RESULTS")
    print("=" * 100)

    print(f"\nSummary:")
    print(f"  Trades: {summary['trades']}")
    print(f"  Win rate: {summary['win_rate']:.1%}")
    print(f"  Profit factor: {summary['profit_factor']:.2f}")
    print(f"  Avg PnL: ${summary['avg_pnl']:.2f}")
    print(f"  Net PnL: ${summary['net_pnl']:.2f}")
    print(f"  Max drawdown: ${summary['max_drawdown']:.2f}")

    print(f"\nExit reasons:")
    total = sum(exit_reasons.values())
    for reason, count in exit_reasons.items():
        pct = count / total * 100 if total > 0 else 0
        avg_pnl = exit_avg_pnl.get(reason, 0)
        print(f"  {reason}: {count} ({pct:.1f}%), avg PnL: ${avg_pnl:.2f}")

    catastop_rate = (exit_reasons.get("CATASTOP", 0) / total * 100) if total > 0 else 0
    print(f"\nCATASTOP rate: {catastop_rate:.1f}%")

    # Run Monte Carlo
    print("\n" + "=" * 100)
    print("MONTE CARLO SIMULATION (10,000 runs)")
    print("=" * 100)

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

    print(f"\nPass rate: {mc_results['pass_rate']*100:.2f}%")
    print(f"Median days to pass: {mc_results['days_to_pass'].get('p50', 0):.1f}")

    print(f"\nFail reasons:")
    total_fails = sum(mc_results['fail_reasons'].values())
    for reason, count in mc_results['fail_reasons'].items():
        pct = count / total_fails * 100 if total_fails > 0 else 0
        print(f"  {reason}: {count} ({pct:.1f}%)")

    return {
        "label": label,
        "trades": summary["trades"],
        "win_rate": summary["win_rate"],
        "profit_factor": summary["profit_factor"],
        "avg_pnl": summary["avg_pnl"],
        "net_pnl": summary["net_pnl"],
        "catastop_count": exit_reasons.get("CATASTOP", 0),
        "catastop_rate": catastop_rate,
        "pass_rate": mc_results["pass_rate"] * 100,
        "median_days": mc_results["days_to_pass"].get("p50", 0),
        "fail_trailing_dd": mc_results["fail_reasons"].get("trailing_drawdown", 0),
        "fail_max_days": mc_results["fail_reasons"].get("max_days", 0),
        "fail_daily_loss": mc_results["fail_reasons"].get("daily_loss", 0),
    }


def main():
    print("=" * 100)
    print("DYNAMIC CATASTROPHIC STOP ANALYSIS")
    print("=" * 100)
    print("\nComparing FIXED vs DYNAMIC catastrophic stop sizing")
    print("Fixed: 48 ticks (12 MES points)")
    print("Dynamic: 2.0x ATR, clamped [24, 72] ticks ([6, 18] MES points)")

    # Run both variants
    baseline = run_and_analyze(use_dynamic=False, label="Baseline (Fixed 48-tick stop)")
    dynamic = run_and_analyze(use_dynamic=True, label="Dynamic (2.0x ATR, clamped 24-72)")

    # Comparison
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)

    metrics = [
        ("Trades", "trades", ""),
        ("CATASTOP Count", "catastop_count", ""),
        ("CATASTOP Rate", "catastop_rate", "%"),
        ("Win Rate", "win_rate", "%"),
        ("Profit Factor", "profit_factor", ""),
        ("Avg PnL", "avg_pnl", "$"),
        ("Net PnL", "net_pnl", "$"),
        ("Pass Rate", "pass_rate", "%"),
        ("Median Days to Pass", "median_days", " days"),
        ("Fail (Trailing DD)", "fail_trailing_dd", ""),
        ("Fail (Max Days)", "fail_max_days", ""),
        ("Fail (Daily Loss)", "fail_daily_loss", ""),
    ]

    print(f"\n{'Metric':<25} {'Baseline':<20} {'Dynamic':<20} {'Change':<20}")
    print("-" * 100)

    for name, key, suffix in metrics:
        base_val = baseline[key]
        dyn_val = dynamic[key]

        if key in ["win_rate"]:
            base_str = f"{base_val:.1%}"
            dyn_str = f"{dyn_val:.1%}"
            change = (dyn_val - base_val) * 100
            change_str = f"{change:+.1f} pp"
        elif key in ["catastop_rate", "pass_rate"]:
            base_str = f"{base_val:.1f}%"
            dyn_str = f"{dyn_val:.1f}%"
            change = dyn_val - base_val
            change_str = f"{change:+.1f} pp"
        elif key in ["profit_factor"]:
            base_str = f"{base_val:.2f}"
            dyn_str = f"{dyn_val:.2f}"
            change = dyn_val - base_val
            change_pct = (change / base_val * 100) if base_val != 0 else 0
            change_str = f"{change:+.2f} ({change_pct:+.1f}%)"
        elif suffix == "$":
            base_str = f"${base_val:.2f}"
            dyn_str = f"${dyn_val:.2f}"
            change = dyn_val - base_val
            change_str = f"${change:+.2f}"
        elif suffix == " days":
            base_str = f"{base_val:.1f} days"
            dyn_str = f"{dyn_val:.1f} days"
            change = dyn_val - base_val
            change_str = f"{change:+.1f} days"
        else:
            base_str = f"{base_val}"
            dyn_str = f"{dyn_val}"
            change = dyn_val - base_val
            change_pct = (change / base_val * 100) if base_val != 0 else 0
            change_str = f"{change:+.0f} ({change_pct:+.1f}%)"

        print(f"{name:<25} {base_str:<20} {dyn_str:<20} {change_str:<20}")

    # Key insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)

    catastop_improvement = baseline["catastop_rate"] - dynamic["catastop_rate"]
    pass_rate_improvement = dynamic["pass_rate"] - baseline["pass_rate"]
    pf_improvement = dynamic["profit_factor"] - baseline["profit_factor"]

    print(f"\n✓ CATASTOP reduction: {catastop_improvement:.1f} percentage points ({baseline['catastop_rate']:.1f}% → {dynamic['catastop_rate']:.1f}%)")
    print(f"✓ Pass rate change: {pass_rate_improvement:+.2f} percentage points ({baseline['pass_rate']:.2f}% → {dynamic['pass_rate']:.2f}%)")
    print(f"✓ Profit factor change: {pf_improvement:+.2f} ({baseline['profit_factor']:.2f} → {dynamic['profit_factor']:.2f})")

    if dynamic["pass_rate"] > 50:
        print("\n🎉 EXCELLENT: Pass rate > 50%, dynamic stops are HIGHLY effective!")
    elif dynamic["pass_rate"] > 20:
        print("\n✅ GOOD: Pass rate > 20%, dynamic stops show promise")
    elif dynamic["pass_rate"] > baseline["pass_rate"]:
        print("\n⚠️  IMPROVED: Pass rate increased but still low, may need further tuning")
    else:
        print("\n❌ UNCHANGED: No improvement in pass rate, dynamic stops may not be sufficient alone")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
