"""
Analyze ATR distribution in training data to establish adaptive thresholds.

Usage:
    python -m ml_intraday_v3.analysis.analyze_training_atr \
        --run-dir runs/v3_2022_5m \
        --bar-size 5m
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_atr(bars_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Match pipeline ATR calculation exactly."""
    df = bars_df.sort_index()
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()

    return atr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bar-size", type=str, default="5m")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Load training bars
    bars_path = args.run_dir / f"bar_size={args.bar_size}" / "bars.parquet"
    print(f"Loading bars from: {bars_path}")
    bars = pd.read_parquet(bars_path)

    # Compute ATR
    atr = compute_atr(bars, period=14)

    # Remove NaN from initial warmup
    atr = atr.dropna()

    # Statistical summary
    stats = {
        "count": int(len(atr)),
        "mean": float(atr.mean()),
        "median": float(atr.median()),
        "std": float(atr.std()),
        "min": float(atr.min()),
        "max": float(atr.max()),
        "p05": float(atr.quantile(0.05)),
        "p25": float(atr.quantile(0.25)),
        "p50": float(atr.quantile(0.50)),
        "p75": float(atr.quantile(0.75)),
        "p95": float(atr.quantile(0.95)),
        "p99": float(atr.quantile(0.99)),
    }

    # CUSUM threshold distribution (0.8 × ATR)
    cusum_threshold = 0.8 * atr
    cusum_stats = {
        "mean": float(cusum_threshold.mean()),
        "median": float(cusum_threshold.median()),
        "p25": float(cusum_threshold.quantile(0.25)),
        "p50": float(cusum_threshold.quantile(0.50)),
        "p75": float(cusum_threshold.quantile(0.75)),
    }

    # Recommended thresholds
    recommendations = {
        "min_atr_conservative": float(0.7 * stats["median"]),  # 70% of median
        "min_atr_balanced": float(0.6 * stats["median"]),      # 60% of median
        "min_atr_aggressive": float(0.5 * stats["median"]),    # 50% of median
        "min_cusum_threshold": float(0.7 * cusum_stats["median"]),
    }

    # Print results
    print("\n=== ATR DISTRIBUTION (14-bar EMA) ===")
    for key, val in stats.items():
        print(f"{key:10s}: {val:8.2f} points")

    print("\n=== CUSUM THRESHOLD DISTRIBUTION (0.8 × ATR) ===")
    for key, val in cusum_stats.items():
        print(f"{key:10s}: {val:8.2f} points")

    print("\n=== RECOMMENDED MINIMUM THRESHOLDS ===")
    print(f"Conservative (70% median ATR): {recommendations['min_atr_conservative']:.2f} points")
    print(f"Balanced (60% median ATR):     {recommendations['min_atr_balanced']:.2f} points")
    print(f"Aggressive (50% median ATR):   {recommendations['min_atr_aggressive']:.2f} points")
    print(f"Min CUSUM threshold:           {recommendations['min_cusum_threshold']:.2f} points")

    # Save results
    output_path = args.output or (args.run_dir / f"bar_size={args.bar_size}" / "atr_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "atr_stats": stats,
        "cusum_stats": cusum_stats,
        "recommendations": recommendations,
        "bar_size": args.bar_size,
        "atr_period": 14,
        "cusum_multiplier": 0.8,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Optional: Plot distribution
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].hist(atr, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(stats['median'], color='red', linestyle='--', label=f"Median: {stats['median']:.2f}")
    axes[0].axvline(recommendations['min_atr_balanced'], color='green', linestyle='--',
                    label=f"Min (balanced): {recommendations['min_atr_balanced']:.2f}")
    axes[0].set_xlabel('ATR (points)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('ATR Distribution in Training Data')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(cusum_threshold, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(cusum_stats['median'], color='red', linestyle='--', label=f"Median: {cusum_stats['median']:.2f}")
    axes[1].axvline(recommendations['min_cusum_threshold'], color='green', linestyle='--',
                    label=f"Min threshold: {recommendations['min_cusum_threshold']:.2f}")
    axes[1].set_xlabel('CUSUM Threshold (points)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('CUSUM Threshold Distribution (0.8 × ATR)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path.parent / "atr_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Distribution plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
