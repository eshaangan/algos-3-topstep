"""
Quick test of dynamic stops with diagnostics.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Setup
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ.setdefault("RISK_PRESET_NAME", "TOPSTEP_50K")

from core.risk_presets import get_risk_preset
from data.clean_bars import clean_bars
from models.nn_inference import load_nn_bundle, predict_scores_for_bars
from backtesting.backtest import run_backtest_nn

# Config
data_path = "data/processed/es_bars_2010_2025.h5"
dataset_key = "bars_5min"
model_dir = "models/nn_saved"
fold = 0
max_bars = 100_000  # Quick test

print("=" * 80)
print("DYNAMIC STOPS DIAGNOSTIC TEST")
print("=" * 80)

# Load data
print(f"\nLoading data from {data_path}...")
with pd.HDFStore(data_path, "r") as store:
    bars = store[dataset_key].copy()

bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
bars = bars.sort_values("timestamp").reset_index(drop=True)

preset = get_risk_preset("TOPSTEP_50K")
bars = clean_bars(bars, tick_size=preset.risk_config.tick_size, verbose=False)
bars = bars.tail(max_bars).reset_index(drop=True)
print(f"Loaded {len(bars):,} bars")

# Load model
print(f"\nLoading model from {model_dir}/fold_{fold}...")
bundle = load_nn_bundle(model_dir, fold=fold)
nn_cfg = bundle.config.get("nn_config", {})

# Print resolved config
print("\n" + "=" * 80)
print("RESOLVED CONFIGURATION")
print("=" * 80)
print(f"Artifact path: {model_dir}/fold_{fold}")
print(f"\nuse_dynamic_catastop: {nn_cfg.get('use_dynamic_catastop', False)}")
print(f"catastop_atr_multiplier: {nn_cfg.get('catastop_atr_multiplier', 2.0):.1f}x")
print(f"catastop_min_ticks: {nn_cfg.get('catastop_min_ticks', 24)}")
print(f"catastop_max_ticks: {nn_cfg.get('catastop_max_ticks', 72)}")

# Compute probabilities
print("\nComputing model probabilities...")
prob_df = predict_scores_for_bars(bars, bundle)

# Run backtest
print("\nRunning backtest...")
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
    use_dynamic_catastop=bool(nn_cfg.get("use_dynamic_catastop", True)),
    catastop_atr_multiplier=float(nn_cfg.get("catastop_atr_multiplier", 3.5)),
    catastop_min_ticks=int(nn_cfg.get("catastop_min_ticks", 24)),
    catastop_max_ticks=int(nn_cfg.get("catastop_max_ticks", 72)),
    save_trades_path=None,
)

# Print results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

summary = results["summary"]
print(f"\nTrades: {summary['trades']}")
print(f"Win Rate: {summary['win_rate']:.2%}")
print(f"Profit Factor: {summary['profit_factor']:.2f}")
print(f"Avg PnL: ${summary['avg_pnl']:.2f}")
print(f"Net PnL: ${summary['net_pnl']:.2f}")

print(f"\n⚠️  CATASTOP Rate: {results.get('catastop_rate', 0):.2f}%")

# Diagnostics
if "stop_diagnostics" in results and results["stop_diagnostics"]:
    print("\n" + "=" * 80)
    print("STOP DIAGNOSTICS")
    print("=" * 80)

    diag = results["stop_diagnostics"]

    if "stop_ticks_distribution" in diag:
        stop_dist = diag["stop_ticks_distribution"]
        print(f"\nStop ticks: p50={stop_dist['p50']:.0f}, p90={stop_dist['p90']:.0f}, mean={stop_dist['mean']:.1f}")

    if "atr_ticks_distribution" in diag:
        atr_dist = diag["atr_ticks_distribution"]
        print(f"ATR ticks:  p50={atr_dist['p50']:.1f}, p90={atr_dist['p90']:.1f}, mean={atr_dist['mean']:.1f}")

        # Verify ATR is in reasonable range
        if atr_dist['mean'] < 10 or atr_dist['mean'] > 50:
            print(f"⚠️  WARNING: ATR mean {atr_dist['mean']:.1f} outside expected range [10-50] for MES 5-min!")

    if "stop_atr_ratio_distribution" in diag:
        ratio_dist = diag["stop_atr_ratio_distribution"]
        print(f"Stop/ATR:   p50={ratio_dist['p50']:.2f}x, p90={ratio_dist['p90']:.2f}x, mean={ratio_dist['mean']:.2f}x")

        # Verify ratio is close to multiplier
        expected_ratio = nn_cfg.get('catastop_atr_multiplier', 3.5)
        if abs(ratio_dist['mean'] - expected_ratio) > 0.5:
            print(f"⚠️  WARNING: Mean ratio {ratio_dist['mean']:.2f}x differs from expected {expected_ratio:.1f}x!")

    if "atr_fallback_count" in diag:
        fallback = diag['atr_fallback_count']
        print(f"\nATR fallback: {fallback} trades ({fallback/summary['trades']*100:.1f}%)")

# Goal check
goal_met = results.get('catastop_rate', 100) < 15 and summary['profit_factor'] > 1.0
print("\n" + "=" * 80)
if goal_met:
    print("✅ GOAL MET: CATASTOP < 15% and PF > 1.0")
else:
    print("⚠️  NEEDS IMPROVEMENT: CATASTOP >= 15% or PF <= 1.0")
print("=" * 80)
