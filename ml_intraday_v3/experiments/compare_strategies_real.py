"""
Compare signal selection strategies using REAL model predictions.

This script re-runs the best model configuration and compares:
- Solution 1: Fixed threshold (0.20, 0.25, 0.30)
- Solution 2: Percentile ranking (top 10%, 15%, 20%)

Using actual predictions instead of synthetic probabilities.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from comprehensive_grid_search_v2 import (
    load_base_configs,
    create_experiment_configs,
    run_single_fold,
)

def main():
    print("\n" + "="*100)
    print("REAL SIGNAL SELECTION STRATEGY COMPARISON")
    print("="*100)
    print("\nRe-running best model with actual predictions...")
    print("Model: PT=3.0, SL=3.5, Conservative, Top10 features, 6mo window")
    print("="*100 + "\n")

    # Configuration
    data_dir = Path("data")
    config_dir = Path("configs")

    # Load data
    data_file = data_dir / "MES_5min_Oct2024_Dec2025.parquet"
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        return

    bars_df = pd.read_parquet(data_file)
    bars_df = bars_df.sort_index()
    if bars_df.index.tz is None:
        bars_df.index = bars_df.index.tz_localize('UTC')

    # Load base configs
    base_configs = load_base_configs(config_dir)

    # Best model configuration
    exp_config = {
        "exp_id": "real_comparison_conservative_pt3.0_sl3.5",
        "phase": 1,
        "model_name": "conservative",
        "model_params": {
            "n_estimators": 100,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 300,
            "reg_alpha": 0.5,
            "reg_lambda": 0.5,
            "learning_rate": 0.05,
        },
        "feature_set_name": "top10",
        "feature_set": [
            "side",
            "autocorr_5",
            "vol_regime",
            "vol_20",
            "ema_ratio",
            "relative_volume",
            "lower_wick",
            "vol_forecast",
            "parkinson_vol",
            "ema_spread",
        ],
        "training_window_months": 6,
        "labeling": {"pt": 3.0, "sl": 3.5, "hz": 12},
        "sample_weight": "uniform",
        "calibration": None,
    }

    # Create experiment configs
    configs = create_experiment_configs(base_configs, exp_config)

    # Walk-forward validation folds
    folds = [
        {
            "fold": 1,
            "train_start": pd.Timestamp("2024-01-01", tz="UTC"),
            "train_end": pd.Timestamp("2024-06-30", tz="UTC"),
            "test_start": pd.Timestamp("2024-07-01", tz="UTC"),
            "test_end": pd.Timestamp("2024-12-31", tz="UTC"),
        },
        {
            "fold": 2,
            "train_start": pd.Timestamp("2024-03-01", tz="UTC"),
            "train_end": pd.Timestamp("2024-08-31", tz="UTC"),
            "test_start": pd.Timestamp("2024-09-01", tz="UTC"),
            "test_end": pd.Timestamp("2025-02-28", tz="UTC"),
        },
        {
            "fold": 3,
            "train_start": pd.Timestamp("2024-05-01", tz="UTC"),
            "train_end": pd.Timestamp("2024-10-31", tz="UTC"),
            "test_start": pd.Timestamp("2024-11-01", tz="UTC"),
            "test_end": pd.Timestamp("2025-04-30", tz="UTC"),
        },
    ]

    # Define strategies
    strategies = {
        'Threshold_0.20': {'type': 'threshold', 'value': 0.20},
        'Threshold_0.25': {'type': 'threshold', 'value': 0.25},
        'Threshold_0.30': {'type': 'threshold', 'value': 0.30},
        'Top_20%': {'type': 'percentile', 'value': 0.80},
        'Top_15%': {'type': 'percentile', 'value': 0.85},
        'Top_10%': {'type': 'percentile', 'value': 0.90},
    }

    results_summary = []

    for fold_info in folds:
        print(f"Running Fold {fold_info['fold']}...")

        # Split data
        train_data = bars_df[fold_info["train_start"]:fold_info["train_end"]]
        test_data = bars_df[fold_info["test_start"]:fold_info["test_end"]]

        # Run fold and get predictions
        fold_result = run_single_fold(
            train_data=train_data,
            test_data=test_data,
            exp_config=exp_config,
            configs=configs,
            fold_num=fold_info["fold"],
            return_predictions=True,  # Request predictions
        )

        if 'error' in fold_result:
            print(f"  ERROR: {fold_result['error']}")
            continue

        # Get real predictions and outcomes
        probs = fold_result['predictions']  # Array of probabilities
        outcomes = fold_result['outcomes']  # Array of actual outcomes (0=stop, 1=target)
        n_test = len(probs)

        print(f"  Fold {fold_info['fold']}: {n_test} signals, AUC={fold_result['test_auc']:.3f}")
        print(f"  Prob range: [{probs.min():.3f}, {probs.max():.3f}], mean={probs.mean():.3f}")

        # Evaluate each strategy
        for strategy_name, strategy in strategies.items():
            if strategy['type'] == 'threshold':
                selected = probs >= strategy['value']
            else:  # percentile
                threshold = np.percentile(probs, strategy['value'] * 100)
                selected = probs >= threshold

            n_selected = selected.sum()

            if n_selected == 0:
                print(f"    {strategy_name}: 0 signals selected")
                continue

            # Metrics for selected signals
            selected_outcomes = outcomes[selected]
            win_rate = selected_outcomes.mean()

            # Estimate daily stats
            signals_per_day = fold_result['est_signals_per_day']
            trades_per_day = n_selected / n_test * signals_per_day

            print(f"    {strategy_name}: {n_selected} signals ({n_selected/n_test*100:.1f}%), "
                  f"win_rate={win_rate:.1%}, trades/day={trades_per_day:.1f}")

            results_summary.append({
                'Strategy': strategy_name,
                'Fold': fold_info['fold'],
                'N_Signals': n_selected,
                'Pct_Signals': n_selected / n_test * 100,
                'Win_Rate': win_rate,
                'Trades_Day': trades_per_day,
                'Type': strategy['type'],
            })

    # Aggregate results
    if not results_summary:
        print("\nERROR: No results to aggregate!")
        return

    df = pd.DataFrame(results_summary)
    df_agg = df.groupby('Strategy').agg({
        'N_Signals': 'mean',
        'Pct_Signals': 'mean',
        'Win_Rate': 'mean',
        'Trades_Day': 'mean',
        'Type': 'first',
    }).reset_index()

    # Sort by type then by trades/day
    df_agg = df_agg.sort_values(['Type', 'Trades_Day'], ascending=[True, False])

    print("\n" + "="*100)
    print("AGGREGATED RESULTS (Across All Folds)")
    print("="*100)
    print(f"{'Strategy':<18} {'Type':<12} {'Signals':<12} {'Win Rate':<12} {'Trades/Day':<12}")
    print("-"*100)

    for _, row in df_agg.iterrows():
        print(f"{row['Strategy']:<18} {row['Type']:<12} {row['Pct_Signals']:<12.1f}% "
              f"{row['Win_Rate']:<12.1%} {row['Trades_Day']:<12.1f}")

    print("="*100)
    print("\nRECOMMENDATION:")
    print("-"*100)

    # Find strategy with ~8-15 trades/day and best win rate
    target_trades = df_agg[(df_agg['Trades_Day'] >= 8) & (df_agg['Trades_Day'] <= 15)]

    if len(target_trades) > 0:
        best = target_trades.loc[target_trades['Win_Rate'].idxmax()]
        print(f"✅ BEST: {best['Strategy']}")
        print(f"   - Trades/Day: {best['Trades_Day']:.1f}")
        print(f"   - Win Rate: {best['Win_Rate']:.1%}")
        print(f"   - Signals Selected: {best['Pct_Signals']:.1f}%")
        print(f"   - Type: {best['Type']}")

        if best['Type'] == 'threshold':
            print(f"\n   📝 Deploy with: primary_threshold = {float(best['Strategy'].split('_')[1])}")
        else:
            pct = best['Strategy'].split('_')[1].replace('%', '')
            print(f"\n   📝 Deploy with: Select top {pct}% of signals by probability")
    else:
        # Find closest to target
        df_agg['distance_from_10'] = abs(df_agg['Trades_Day'] - 10.0)
        closest = df_agg.loc[df_agg['distance_from_10'].idxmin()]

        print(f"⚠️  No strategy achieved exact target 8-15 trades/day")
        print(f"\nCLOSEST: {closest['Strategy']}")
        print(f"   - Trades/Day: {closest['Trades_Day']:.1f}")
        print(f"   - Win Rate: {closest['Win_Rate']:.1%}")
        print(f"   - Signals Selected: {closest['Pct_Signals']:.1f}%")

        if closest['Trades_Day'] < 8:
            print(f"\n   💡 Consider: Lower threshold to increase volume")
        elif closest['Trades_Day'] > 15:
            print(f"\n   💡 Consider: Higher threshold to reduce volume")

    print("\n" + "="*100)

    # Calculate expected daily P&L (simplified)
    print("\nEXPECTED PERFORMANCE ESTIMATES:")
    print("-"*100)

    # Assume PT=3.0 points = $15, SL=3.5 points = $17.50 per contract
    pt_dollars = 3.0 * 5  # $15
    sl_dollars = 3.5 * 5  # $17.50

    for _, row in df_agg.iterrows():
        avg_win = pt_dollars * row['Win_Rate']
        avg_loss = sl_dollars * (1 - row['Win_Rate'])
        expected_value_per_trade = avg_win - avg_loss
        expected_daily_pnl = expected_value_per_trade * row['Trades_Day']

        print(f"{row['Strategy']:<18} EV/trade: ${expected_value_per_trade:>6.2f}  "
              f"Daily PnL: ${expected_daily_pnl:>7.2f}")

    print("="*100 + "\n")

if __name__ == "__main__":
    main()
