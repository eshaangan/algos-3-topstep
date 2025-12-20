"""
Test different probability thresholds to optimize trade frequency.

Goal: 1-2 trades/day while maintaining profitability.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
from features.engineer import add_features, select_features


def test_threshold(long_model, short_model, bars_df, feature_cols,
                   long_threshold, short_threshold, trading_days):
    """
    Simulate backtest with different thresholds (simplified).

    Returns trade frequency and estimated win rate.
    """
    # Get features
    features_df = add_features(bars_df, verbose=False)
    features_df = features_df.reset_index().rename(columns={"index": "idx"})

    valid_mask = features_df[feature_cols].notna().all(axis=1)
    valid_idx = features_df.loc[valid_mask, "idx"].values

    X = features_df.loc[valid_mask, feature_cols].values
    long_prob = long_model.predict_proba(X)[:, 1]
    short_prob = short_model.predict_proba(X)[:, 1]

    # Count trades that would be taken
    long_signals = (long_prob >= long_threshold).sum()
    short_signals = (short_prob >= short_threshold).sum()
    total_signals = long_signals + short_signals

    # Estimate actual trades (accounting for max hold, etc)
    # Rough estimate: ~60% of signals become trades
    estimated_trades = int(total_signals * 0.6)
    trades_per_day = estimated_trades / trading_days

    # Estimate win rates (higher threshold = higher WR typically)
    # This is approximate based on probability distribution
    long_avg_prob = long_prob[long_prob >= long_threshold].mean() if long_signals > 0 else 0
    short_avg_prob = short_prob[short_prob >= short_threshold].mean() if short_signals > 0 else 0

    # Simple heuristic: WR roughly scales with avg probability
    # At 0.65, we get 71% WR, so approximate from there
    est_long_wr = min(0.80, long_avg_prob * 1.1) if long_signals > 0 else 0
    est_short_wr = min(0.75, short_avg_prob * 1.05) if short_signals > 0 else 0

    weighted_wr = (long_signals * est_long_wr + short_signals * est_short_wr) / total_signals if total_signals > 0 else 0

    return {
        'long_threshold': long_threshold,
        'short_threshold': short_threshold,
        'long_signals': long_signals,
        'short_signals': short_signals,
        'total_signals': total_signals,
        'estimated_trades': estimated_trades,
        'trades_per_day': trades_per_day,
        'est_long_wr': est_long_wr,
        'est_short_wr': est_short_wr,
        'est_avg_wr': weighted_wr,
        'long_avg_prob': long_avg_prob,
        'short_avg_prob': short_avg_prob,
    }


def main():
    print("="*80)
    print("PROBABILITY THRESHOLD OPTIMIZATION FOR TRADE FREQUENCY")
    print("="*80)
    print("\nGoal: 1-2 trades/day while maintaining >65% win rate and >5 profit factor")
    print()

    # Load data
    print("Loading data and models...")
    with pd.HDFStore("data/processed/mes_bars.h5", "r") as store:
        bars = store["bars_5min"]

    # Load models
    model_dir = Path("models/saved_v3_optimized")
    long_model = joblib.load(model_dir / "model_long.joblib")
    short_model = joblib.load(model_dir / "model_short.joblib")

    with open(model_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_cols"]

    # Calculate trading days
    bars['date'] = pd.to_datetime(bars['timestamp']).dt.date
    trading_days = bars['date'].nunique()
    print(f"Total trading days: {trading_days}")
    print()

    # Test various threshold combinations
    threshold_tests = [
        # (long_threshold, short_threshold, description)
        (0.65, 0.65, "Current (baseline)"),
        (0.65, 0.60, "Lower short only"),
        (0.65, 0.55, "Much lower short"),
        (0.60, 0.60, "Lower both to 0.60"),
        (0.55, 0.55, "Lower both to 0.55"),
        (0.60, 0.55, "Long 0.60, Short 0.55"),
        (0.55, 0.50, "Long 0.55, Short 0.50"),
        (0.50, 0.50, "Lower both to 0.50"),
    ]

    results = []
    for long_thresh, short_thresh, desc in threshold_tests:
        result = test_threshold(
            long_model, short_model, bars, feature_cols,
            long_thresh, short_thresh, trading_days
        )
        result['description'] = desc
        results.append(result)

        print(f"{desc}:")
        print(f"  Thresholds: Long={long_thresh:.2f}, Short={short_thresh:.2f}")
        print(f"  Signals: {result['long_signals']:,} long, {result['short_signals']:,} short, {result['total_signals']:,} total")
        print(f"  Estimated trades: {result['estimated_trades']:,}")
        print(f"  Trades/day: {result['trades_per_day']:.2f}")
        print(f"  Est. Win Rates: Long={result['est_long_wr']*100:.1f}%, Short={result['est_short_wr']*100:.1f}%, Avg={result['est_avg_wr']*100:.1f}%")
        print()

    # Create summary table
    results_df = pd.DataFrame(results)

    print("="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.3f}'.format)

    summary = results_df[[
        'description', 'long_threshold', 'short_threshold',
        'trades_per_day', 'est_avg_wr', 'long_signals', 'short_signals'
    ]].copy()

    print(summary.to_string(index=False))
    print()

    # Find optimal
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()

    # Filter for 1-2 trades/day and good win rate
    optimal = results_df[
        (results_df['trades_per_day'] >= 1.0) &
        (results_df['trades_per_day'] <= 2.5) &
        (results_df['est_avg_wr'] >= 0.60)
    ].copy()

    if len(optimal) > 0:
        optimal = optimal.sort_values('trades_per_day', ascending=False)
        best = optimal.iloc[0]

        print(f"✅ RECOMMENDED CONFIGURATION:")
        print(f"  Long Threshold: {best['long_threshold']:.2f}")
        print(f"  Short Threshold: {best['short_threshold']:.2f}")
        print(f"  Expected Trades/Day: {best['trades_per_day']:.2f}")
        print(f"  Estimated Win Rate: {best['est_avg_wr']*100:.1f}%")
        print(f"  Long Signals: {best['long_signals']:,.0f}")
        print(f"  Short Signals: {best['short_signals']:,.0f}")
        print()

        # Calculate expected performance
        # With 2:1 R:R and estimated WR
        wr = best['est_avg_wr']
        ev_per_trade = wr * 2.0 - (1 - wr) * 1.0
        pf = (wr * 2.0) / ((1 - wr) * 1.0) if wr < 1 else float('inf')

        print(f"📊 EXPECTED PERFORMANCE:")
        print(f"  Expected Value: {ev_per_trade:.3f}R per trade")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Annual Trades: ~{best['trades_per_day'] * 252:.0f}")
        print()

        if len(optimal) > 1:
            print(f"📋 OTHER VALID OPTIONS:")
            for idx, row in optimal.iloc[1:].iterrows():
                print(f"  {row['description']}: {row['trades_per_day']:.2f} trades/day, {row['est_avg_wr']*100:.1f}% WR")
    else:
        print("⚠️  No configuration meets both criteria (1-2 trades/day AND >60% WR)")
        print("Consider relaxing constraints or using the closest option:")
        closest = results_df.iloc[(results_df['trades_per_day'] - 1.5).abs().argsort()[:3]]
        print()
        for idx, row in closest.iterrows():
            print(f"  {row['description']}: {row['trades_per_day']:.2f} trades/day, {row['est_avg_wr']*100:.1f}% WR")

    # Save results
    results_df.to_csv("analysis/threshold_optimization_results.csv", index=False)
    print()
    print("💾 Full results saved to: analysis/threshold_optimization_results.csv")


if __name__ == "__main__":
    main()
