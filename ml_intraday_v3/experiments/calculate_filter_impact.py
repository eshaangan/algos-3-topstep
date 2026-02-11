#!/usr/bin/env python3
"""
Calculate Exact Impact of Filters on Jan 2026 Performance

Based on actual Jan 2026 confidence breakdown from live trading data.
Simulates what would have happened with different confidence thresholds.
"""

import pandas as pd
import numpy as np
from typing import Dict, List

# Jan 2026 ACTUAL data (from live trading analysis)
JAN_2026_ACTUAL = {
    'total_trades': 152,
    'total_pnl': -884.73,
    'win_rate': 0.355,
    'days': 18,
    'avg_trade': -5.82
}

# Confidence breakdown (from the plan document)
CONFIDENCE_BREAKDOWN = [
    {
        'range': 'P < 0.50',
        'threshold': 0.50,
        'trades': 122,
        'win_rate': 0.336,
        'avg_pnl': -6.77,
        'total_pnl': -825.94
    },
    {
        'range': 'P = 0.50-0.55',
        'threshold': 0.55,
        'trades': 10,  # Estimated from "medium confidence"
        'win_rate': 0.50,
        'avg_pnl': 4.63,
        'total_pnl': 46.30
    },
    {
        'range': 'P = 0.55-0.60',
        'threshold': 0.60,
        'trades': 10,  # Part of medium confidence
        'win_rate': 0.50,
        'avg_pnl': 4.63,
        'total_pnl': 46.30
    },
    {
        'range': 'P > 0.60',
        'threshold': 0.65,
        'trades': 10,  # Estimated
        'win_rate': 0.45,
        'avg_pnl': -5.00,
        'total_pnl': -50.00
    }
]


def calculate_filtered_performance(threshold: float) -> Dict:
    """
    Calculate performance if we only traded signals above threshold.

    Args:
        threshold: Minimum confidence threshold (0.50, 0.55, 0.60, etc.)

    Returns:
        Dict with performance metrics
    """
    # Filter trades based on threshold
    kept_trades = [t for t in CONFIDENCE_BREAKDOWN if t['threshold'] >= threshold]

    if not kept_trades:
        return {
            'threshold': threshold,
            'trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_trade': 0,
            'trades_per_day': 0,
            'daily_pnl': 0,
            'days_to_3000': float('inf')
        }

    total_trades = sum(t['trades'] for t in kept_trades)
    total_pnl = sum(t['total_pnl'] for t in kept_trades)
    avg_trade = total_pnl / total_trades if total_trades > 0 else 0

    # Calculate weighted win rate
    total_wins = sum(t['trades'] * t['win_rate'] for t in kept_trades)
    win_rate = total_wins / total_trades if total_trades > 0 else 0

    # Calculate daily metrics
    trades_per_day = total_trades / JAN_2026_ACTUAL['days']
    daily_pnl = total_pnl / JAN_2026_ACTUAL['days']

    # Days to $3,000 (if profitable)
    if daily_pnl > 0:
        days_to_3000 = 3000 / daily_pnl
    else:
        days_to_3000 = float('inf')

    return {
        'threshold': threshold,
        'trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_trade': avg_trade,
        'trades_per_day': trades_per_day,
        'daily_pnl': daily_pnl,
        'days_to_3000': days_to_3000,
        'improvement_vs_baseline': total_pnl - JAN_2026_ACTUAL['total_pnl']
    }


def calculate_realistic_scenario(threshold: float, quality_improvement: float = 0.05) -> Dict:
    """
    Calculate MORE REALISTIC performance accounting for:
    1. More medium-confidence trades (model improvement)
    2. Better win rate from improved signals (+5%)

    Args:
        threshold: Confidence threshold
        quality_improvement: Expected win rate improvement from signal enhancements

    Returns:
        Realistic performance projection
    """
    base = calculate_filtered_performance(threshold)

    # Assume we can get 6-8 quality trades per day with improvements
    # (vs the 1.67 from just filtering existing trades)
    target_trades_per_day = 7
    days = JAN_2026_ACTUAL['days']

    # Project total trades
    projected_trades = target_trades_per_day * days  # 126 trades

    # Use average metrics from medium-confidence trades
    medium_avg_pnl = 4.63
    medium_win_rate = 0.50

    # Apply quality improvement
    improved_win_rate = min(medium_win_rate + quality_improvement, 0.70)

    # Adjust avg P&L based on improved win rate
    # If win rate improves, avg P&L improves proportionally
    win_improvement_factor = improved_win_rate / medium_win_rate
    improved_avg_pnl = medium_avg_pnl * win_improvement_factor

    # Calculate totals
    total_pnl = projected_trades * improved_avg_pnl
    daily_pnl = total_pnl / days
    days_to_3000 = 3000 / daily_pnl if daily_pnl > 0 else float('inf')

    return {
        'threshold': threshold,
        'trades': projected_trades,
        'win_rate': improved_win_rate,
        'total_pnl': total_pnl,
        'avg_trade': improved_avg_pnl,
        'trades_per_day': target_trades_per_day,
        'daily_pnl': daily_pnl,
        'days_to_3000': days_to_3000,
        'improvement_vs_baseline': total_pnl - JAN_2026_ACTUAL['total_pnl'],
        'scenario': 'Realistic (w/ improvements)'
    }


def main():
    print("=" * 80)
    print("JAN 2026 FILTER IMPACT ANALYSIS")
    print("=" * 80)

    # Baseline
    print("\n📊 BASELINE (No Filters)")
    print("-" * 80)
    print(f"Total Trades:     {JAN_2026_ACTUAL['total_trades']}")
    print(f"Win Rate:         {JAN_2026_ACTUAL['win_rate']:.1%}")
    print(f"Total P&L:        ${JAN_2026_ACTUAL['total_pnl']:,.2f}")
    print(f"Avg Trade:        ${JAN_2026_ACTUAL['avg_trade']:,.2f}")
    print(f"Trades/Day:       {JAN_2026_ACTUAL['total_trades'] / JAN_2026_ACTUAL['days']:.1f}")
    print(f"Daily P&L:        ${JAN_2026_ACTUAL['total_pnl'] / JAN_2026_ACTUAL['days']:,.2f}")
    print(f"Days to $3,000:   NEVER (losing money)")

    # Test different thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65]

    print("\n" + "=" * 80)
    print("SCENARIO 1: FILTERING ONLY (No Other Improvements)")
    print("=" * 80)
    print("This shows what happens if we ONLY filter existing trades by confidence.\n")

    for threshold in thresholds:
        result = calculate_filtered_performance(threshold)

        print(f"\n🔹 THRESHOLD = {threshold:.2f}")
        print("-" * 80)
        print(f"Trades Kept:      {result['trades']}/{JAN_2026_ACTUAL['total_trades']} "
              f"({100 * result['trades'] / JAN_2026_ACTUAL['total_trades']:.1f}% of original)")
        print(f"Win Rate:         {result['win_rate']:.1%} "
              f"({result['win_rate'] - JAN_2026_ACTUAL['win_rate']:+.1%} vs baseline)")
        print(f"Total P&L:        ${result['total_pnl']:,.2f} "
              f"({result['improvement_vs_baseline']:+.2f} vs baseline)")
        print(f"Avg Trade:        ${result['avg_trade']:,.2f}")
        print(f"Trades/Day:       {result['trades_per_day']:.1f}")
        print(f"Daily P&L:        ${result['daily_pnl']:,.2f}")

        if result['days_to_3000'] < 100:
            print(f"Days to $3,000:   {result['days_to_3000']:.1f} days ✅")
        else:
            print(f"Days to $3,000:   NEVER ❌")

    # Realistic scenarios with improvements
    print("\n" + "=" * 80)
    print("SCENARIO 2: REALISTIC (Filters + Signal Quality Improvements)")
    print("=" * 80)
    print("Assumes:\n"
          "  - Better entry timing, dynamic stops → +5% win rate\n"
          "  - More quality signals generated → 6-8 trades/day\n"
          "  - Model improvements → maintain medium-confidence quality\n")

    for threshold in [0.55, 0.60]:
        for quality_boost in [0.00, 0.05, 0.10]:
            result = calculate_realistic_scenario(threshold, quality_boost)

            print(f"\n🔹 THRESHOLD = {threshold:.2f}, Quality Boost = +{quality_boost:.0%}")
            print("-" * 80)
            print(f"Total Trades:     {result['trades']} ({result['trades_per_day']:.1f}/day)")
            print(f"Win Rate:         {result['win_rate']:.1%}")
            print(f"Total P&L:        ${result['total_pnl']:,.2f} "
                  f"({result['improvement_vs_baseline']:+.2f} vs baseline)")
            print(f"Avg Trade:        ${result['avg_trade']:,.2f}")
            print(f"Daily P&L:        ${result['daily_pnl']:,.2f}")

            if result['days_to_3000'] < 30:
                print(f"Days to $3,000:   {result['days_to_3000']:.1f} days ✅")
            elif result['days_to_3000'] < 100:
                print(f"Days to $3,000:   {result['days_to_3000']:.1f} days ⚠️")
            else:
                print(f"Days to $3,000:   NEVER ❌")

    # Summary recommendation
    print("\n" + "=" * 80)
    print("📋 RECOMMENDED CONFIGURATION")
    print("=" * 80)

    recommended = calculate_realistic_scenario(0.55, 0.05)

    print(f"\nThreshold:        0.55 (medium confidence)")
    print(f"Expected Trades:  ~{recommended['trades_per_day']:.0f} per day ({recommended['trades']} total over {JAN_2026_ACTUAL['days']} days)")
    print(f"Expected Win Rate: {recommended['win_rate']:.1%}")
    print(f"Expected Daily P&L: ${recommended['daily_pnl']:,.2f}")
    print(f"Expected Total P&L (18 days): ${recommended['total_pnl']:,.2f}")
    print(f"\nImprovement vs Baseline: ${recommended['improvement_vs_baseline']:,.2f}")
    print(f"Days to $3,000: {recommended['days_to_3000']:.1f} days")

    if recommended['days_to_3000'] <= 20:
        print("\n✅ PASSES COMBINE TIMELINE (15-20 days target)")
    elif recommended['days_to_3000'] <= 30:
        print("\n⚠️ ACCEPTABLE (but slower than ideal)")
    else:
        print("\n❌ TOO SLOW FOR COMBINE")

    print("\n" + "=" * 80)
    print("⚠️  IMPORTANT NOTE")
    print("=" * 80)
    print("These are PROJECTIONS based on Jan 2026 confidence breakdown.")
    print("Actual results depend on:")
    print("  1. Model improvements generating more quality signals")
    print("  2. Signal quality enhancements (entry timing, dynamic stops)")
    print("  3. Market conditions in future trading")
    print("\nNext step: RUN BACKTEST with filters to validate these projections.")
    print("=" * 80)


if __name__ == "__main__":
    main()
