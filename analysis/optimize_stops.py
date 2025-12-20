"""
Optimize stop/target parameters by testing multiple configurations.

This script tests different stop sizes to find the optimal configuration
that produces 40-50% label win rates (realistic for 1.5:1 R:R).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.labels import create_labels


def analyze_stop_size(bars_df: pd.DataFrame, stop_ticks: int, target_multiplier: float = 1.5) -> dict:
    """
    Test a specific stop/target configuration.

    Returns:
        Dictionary with win rates, expected values, and label distribution
    """
    print(f"\n{'='*60}")
    print(f"Testing: {stop_ticks} ticks stop, {target_multiplier:.1f}x target ({stop_ticks * target_multiplier:.0f} ticks)")
    print(f"{'='*60}")

    # Create labels with this configuration
    labels_df = create_labels(
        bars_df,
        lookback=100,
        stop_ticks=stop_ticks,
        target_multiplier=target_multiplier,
        max_hold_bars=12,
        tick_size=0.25,
        tick_value=1.25
    )

    # Calculate statistics
    long_wins = (labels_df['label_long'] == 1).sum()
    long_losses = (labels_df['label_long'] == 0).sum()
    long_neutral = (labels_df['label_long'] == 2).sum()
    long_total = long_wins + long_losses
    long_wr = long_wins / long_total if long_total > 0 else 0

    short_wins = (labels_df['label_short'] == 1).sum()
    short_losses = (labels_df['label_short'] == 0).sum()
    short_neutral = (labels_df['label_short'] == 2).sum()
    short_total = short_wins + short_losses
    short_wr = short_wins / short_total if short_total > 0 else 0

    # Calculate expected value per trade (in R-multiples)
    # Win: +1.5R, Loss: -1.0R
    long_ev = long_wr * target_multiplier - (1 - long_wr) * 1.0
    short_ev = short_wr * target_multiplier - (1 - short_wr) * 1.0

    # Calculate profit factor
    long_pf = (long_wr * target_multiplier) / ((1 - long_wr) * 1.0) if long_wr < 1 else float('inf')
    short_pf = (short_wr * target_multiplier) / ((1 - short_wr) * 1.0) if short_wr < 1 else float('inf')

    results = {
        'stop_ticks': stop_ticks,
        'target_ticks': int(stop_ticks * target_multiplier),
        'target_multiplier': target_multiplier,
        'long_wr': long_wr,
        'long_ev': long_ev,
        'long_pf': long_pf,
        'long_wins': long_wins,
        'long_losses': long_losses,
        'long_neutral': long_neutral,
        'short_wr': short_wr,
        'short_ev': short_ev,
        'short_pf': short_pf,
        'short_wins': short_wins,
        'short_losses': short_losses,
        'short_neutral': short_neutral,
        'avg_wr': (long_wr + short_wr) / 2,
        'avg_ev': (long_ev + short_ev) / 2,
    }

    return results


def main():
    print("="*60)
    print("STOP SIZE OPTIMIZATION EXPERIMENT")
    print("="*60)
    print("\nLoading data...")

    # Load data
    with pd.HDFStore("data/processed/mes_bars.h5", "r") as store:
        bars = store["bars_5min"]

    print(f"Loaded {len(bars):,} bars")
    print(f"Date range: {bars['timestamp'].min()} to {bars['timestamp'].max()}")

    # Test different stop sizes
    stop_sizes = [18, 20, 22, 24, 26, 28, 30]
    target_multiplier = 1.5

    results = []
    for stop_ticks in stop_sizes:
        result = analyze_stop_size(bars, stop_ticks, target_multiplier)
        results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Print summary table
    print("\n" + "="*100)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*100)
    print("\nTarget: 40-50% win rate with positive expected value")
    print()

    # Format and display results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.3f}'.format)

    summary = results_df[[
        'stop_ticks', 'target_ticks',
        'long_wr', 'long_ev', 'long_pf',
        'short_wr', 'short_ev', 'short_pf',
        'avg_wr', 'avg_ev'
    ]].copy()

    print(summary.to_string(index=False))

    # Find optimal configurations
    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)

    # Filter for win rates in target range (35-55%)
    valid = results_df[
        (results_df['avg_wr'] >= 0.35) &
        (results_df['avg_wr'] <= 0.55)
    ].copy()

    if len(valid) == 0:
        print("\n⚠️  WARNING: No configurations achieve 35-55% win rate!")
        print("Consider wider stops or different target multiplier.")
        # Find closest
        results_df['distance'] = abs(results_df['avg_wr'] - 0.45)
        closest = results_df.loc[results_df['distance'].idxmin()]
        print(f"\nClosest configuration: {closest['stop_ticks']:.0f} ticks")
        print(f"  Win Rate: {closest['avg_wr']*100:.1f}%")
        print(f"  Expected Value: {closest['avg_ev']:.3f}R")
    else:
        # Sort by expected value
        valid = valid.sort_values('avg_ev', ascending=False)
        best = valid.iloc[0]

        print(f"\n✅ OPTIMAL CONFIGURATION:")
        print(f"  Stop: {best['stop_ticks']:.0f} ticks ({best['stop_ticks'] * 0.25:.2f} points)")
        print(f"  Target: {best['target_ticks']:.0f} ticks ({best['target_ticks'] * 0.25:.2f} points)")
        print(f"  R:R Ratio: 1:{best['target_multiplier']:.1f}")
        print()
        print(f"  Long Win Rate: {best['long_wr']*100:.1f}%")
        print(f"  Long Expected Value: {best['long_ev']:.3f}R per trade")
        print(f"  Long Profit Factor: {best['long_pf']:.2f}")
        print()
        print(f"  Short Win Rate: {best['short_wr']*100:.1f}%")
        print(f"  Short Expected Value: {best['short_ev']:.3f}R per trade")
        print(f"  Short Profit Factor: {best['short_pf']:.2f}")
        print()
        print(f"  Average Win Rate: {best['avg_wr']*100:.1f}%")
        print(f"  Average Expected Value: {best['avg_ev']:.3f}R per trade")

        # Show all valid options
        if len(valid) > 1:
            print(f"\n📊 ALL VALID OPTIONS (sorted by expected value):")
            for idx, row in valid.iterrows():
                print(f"  {row['stop_ticks']:.0f} ticks: WR={row['avg_wr']*100:.1f}%, EV={row['avg_ev']:.3f}R, PF={(row['long_pf']+row['short_pf'])/2:.2f}")

    # Save results
    output_path = "analysis/stop_optimization_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Full results saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
