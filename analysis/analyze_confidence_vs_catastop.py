"""
Analyze relationship between confidence and CATASTOP rate.

This script loads the backtest trades and analyzes whether higher confidence
trades have lower CATASTOP rates, helping determine the optimal confidence_min.
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


def main():
    # Configuration
    data_path = "data/processed/es_bars_2010_2025.h5"
    dataset_key = "bars_5min"
    model_dir = "models/nn_saved"
    fold = 0
    fast_max_bars = 250_000

    print("=" * 80)
    print("CONFIDENCE vs CATASTOP ANALYSIS")
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

    # Load model
    print(f"\nLoading model from {model_dir}...")
    bundle = load_nn_bundle(model_dir, fold=fold)
    nn_cfg = bundle.config.get("nn_config", {})

    # Compute probabilities
    print("Computing model probabilities...")
    prob_df = predict_scores_for_bars(bars, bundle)

    # Run backtest with NO confidence filter to get all possible trades
    print("\nRunning backtest with NO confidence filter (confidence_min=0.0)...")
    risk_cfg = preset.risk_config

    # Temporarily override confidence_min to 0.0 to get all trades
    cfg_override = dict(nn_cfg)
    cfg_override["confidence_min"] = 0.0

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

    trades_df = pd.DataFrame(results["trades"])

    if trades_df.empty or "confidence" not in trades_df.columns:
        print("ERROR: No trades or no confidence data")
        return

    # Filter out trades with missing confidence
    trades_with_conf = trades_df[trades_df["confidence"].notna()].copy()
    print(f"\nTotal trades: {len(trades_df)}")
    print(f"Trades with confidence: {len(trades_with_conf)}")

    if trades_with_conf.empty:
        print("ERROR: No trades with confidence data")
        return

    # Analyze CATASTOP rate by confidence bucket
    print("\n" + "=" * 80)
    print("CATASTOP RATE BY CONFIDENCE BUCKET")
    print("=" * 80)

    confidence_bins = [0.00, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20, 1.0]
    trades_with_conf["conf_bucket"] = pd.cut(
        trades_with_conf["confidence"],
        bins=confidence_bins,
        include_lowest=True,
    )

    bucket_stats = []
    for bucket in trades_with_conf["conf_bucket"].cat.categories:
        bucket_trades = trades_with_conf[trades_with_conf["conf_bucket"] == bucket]
        if len(bucket_trades) == 0:
            continue

        catastop_trades = bucket_trades[bucket_trades["reason"] == "CATASTOP"]
        catastop_rate = len(catastop_trades) / len(bucket_trades) * 100.0
        avg_pnl = bucket_trades["pnl"].mean()
        win_rate = (bucket_trades["pnl"] > 0).sum() / len(bucket_trades) * 100.0

        bucket_stats.append({
            "bucket": str(bucket),
            "trades": len(bucket_trades),
            "catastop_rate": catastop_rate,
            "catastop_count": len(catastop_trades),
            "avg_pnl": avg_pnl,
            "win_rate": win_rate,
        })

    print(f"\n{'Confidence Bucket':<20} {'Trades':<8} {'CATASTOP %':<12} {'Avg PnL':<12} {'Win Rate %':<12}")
    print("-" * 80)
    for stat in bucket_stats:
        print(
            f"{stat['bucket']:<20} "
            f"{stat['trades']:<8} "
            f"{stat['catastop_rate']:<12.1f} "
            f"${stat['avg_pnl']:<11.2f} "
            f"{stat['win_rate']:<12.1f}"
        )

    # Find threshold for <10% CATASTOP
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS FOR <10% CATASTOP TARGET")
    print("=" * 80)

    thresholds = [0.00, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]
    for thresh in thresholds:
        filtered = trades_with_conf[trades_with_conf["confidence"] >= thresh]
        if len(filtered) == 0:
            continue

        catastop = filtered[filtered["reason"] == "CATASTOP"]
        catastop_rate = len(catastop) / len(filtered) * 100.0
        avg_pnl = filtered["pnl"].mean()
        win_rate = (filtered["pnl"] > 0).sum() / len(filtered) * 100.0

        status = "✓ MEETS TARGET" if catastop_rate < 10.0 else ""
        print(
            f"conf_min={thresh:.2f}: {len(filtered):>4} trades, "
            f"CATASTOP={catastop_rate:>5.1f}%, "
            f"avg_pnl=${avg_pnl:>7.2f}, "
            f"win_rate={win_rate:>5.1f}% "
            f"{status}"
        )

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS (all trades)")
    print("=" * 80)

    catastop_all = trades_with_conf[trades_with_conf["reason"] == "CATASTOP"]
    timeexit_all = trades_with_conf[trades_with_conf["reason"] == "TIME_EXIT"]

    print(f"\nCATASTOP trades: {len(catastop_all)} ({len(catastop_all)/len(trades_with_conf)*100:.1f}%)")
    print(f"  Avg confidence: {catastop_all['confidence'].mean():.4f}")
    print(f"  Median confidence: {catastop_all['confidence'].median():.4f}")
    print(f"  Avg PnL: ${catastop_all['pnl'].mean():.2f}")

    print(f"\nTIME_EXIT trades: {len(timeexit_all)} ({len(timeexit_all)/len(trades_with_conf)*100:.1f}%)")
    print(f"  Avg confidence: {timeexit_all['confidence'].mean():.4f}")
    print(f"  Median confidence: {timeexit_all['confidence'].median():.4f}")
    print(f"  Avg PnL: ${timeexit_all['pnl'].mean():.2f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
