#!/usr/bin/env python3
"""
Walk-Forward Validation - Week 2 Day 12-13

Purpose:
- Test model stability across multiple time periods
- Detect overfitting by validating on 6 different months
- Ensure edge is consistent across different market regimes

Success Criteria:
- Positive months: 4/6 (67%)
- Cumulative PnL: >$500
- Avg Sharpe: >0.8
- Max DD: <$1,500
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def walk_forward_test(
    data: pd.DataFrame,
    feature_cols: List[str],
    test_months: List[Tuple[str, str, str]],
    model_class=LogisticRegression,
    model_params: Dict = None
) -> pd.DataFrame:
    """
    Run walk-forward validation across multiple test months.

    Args:
        data: Full dataset with features and labels
        feature_cols: List of feature column names
        test_months: List of (month_name, start_date, end_date) tuples
        model_class: Model class to use
        model_params: Model initialization parameters

    Returns:
        DataFrame with results for each test month
    """

    if model_params is None:
        model_params = {
            'C': 1.0,
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs'
        }

    results = []

    for month_name, start_date, end_date in test_months:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {month_name} ({start_date} to {end_date})")
        logger.info(f"{'='*60}")

        # Split data
        train = data[data.index < start_date]
        test = data[(data.index >= start_date) & (data.index <= end_date)]

        if len(test) < 5:
            logger.warning(f"   Skipped: Only {len(test)} events in test set")
            continue

        # Prepare features
        X_train = train[feature_cols]
        y_train = train['y']
        w_train = train['w_final']

        # Drop NaN rows
        train_valid = ~X_train.isna().any(axis=1)
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        w_train = w_train[train_valid]

        X_test = test[feature_cols]
        y_test = test['y']

        # Drop NaN rows
        test_valid = ~X_test.isna().any(axis=1)
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]

        if len(X_test) == 0:
            logger.warning(f"   Skipped: No valid test samples after NaN filtering")
            continue

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = model_class(**model_params)
        model.fit(X_train_scaled, y_train, sample_weight=w_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = np.nan

        # Get actual returns from test data (using valid test indices)
        test_returns = test[test_valid]['ret_net']

        # Calculate PnL (assuming 1 contract per trade)
        pnl = test_returns.sum()
        avg_pnl = test_returns.mean()

        # Calculate Sharpe (annualized)
        if len(test_returns) > 1 and test_returns.std() > 0:
            sharpe = (test_returns.mean() / test_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = test_returns.cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_dd = drawdown.min()

        # Target rate
        target_rate = (y_test == 1).sum() / len(y_test)

        # Log results
        logger.info(f"   Train: {len(X_train):,} events")
        logger.info(f"   Test:  {len(X_test):,} events")
        logger.info(f"   Accuracy: {acc:.1%} | Balanced: {bal_acc:.1%} | AUC: {auc:.3f}")
        logger.info(f"   PnL: ${pnl:,.2f} | Avg: ${avg_pnl:.2f} | Sharpe: {sharpe:.2f}")
        logger.info(f"   Max DD: ${max_dd:,.2f} | Target Rate: {target_rate:.1%}")

        results.append({
            'month': month_name,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'auc': auc,
            'pnl': pnl,
            'avg_pnl': avg_pnl,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'target_rate': target_rate,
            'positive': pnl > 0
        })

    return pd.DataFrame(results)


def main():
    """Main execution."""

    # Paths
    project_root = Path(__file__).parent.parent.parent
    features_path = project_root / "runs" / "v3_2022_5m" / "bar_size=5m" / "features.parquet"
    events_path = project_root / "runs" / "v3_2022_5m" / "bar_size=5m" / "events.parquet"
    weights_path = project_root / "runs" / "v3_2022_5m" / "bar_size=5m" / "weights.parquet"

    # Create diagnostics directory
    diagnostics_dir = project_root / "ml_intraday_v3" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")

    # Load data
    features = pd.read_parquet(features_path)
    events = pd.read_parquet(events_path)
    weights = pd.read_parquet(weights_path)

    # Merge
    events_with_weights = events.merge(weights[['event_id', 'w_final']], on='event_id')
    events_with_weights['t0'] = pd.to_datetime(events_with_weights['t0'])
    events_with_weights = events_with_weights.set_index('t0').sort_index()

    data = features.join(events_with_weights[['y', 'w_final', 'side', 't1', 'ret_net']], how='inner')

    # Filter vertical barriers
    data = data[data['y'] != 0].copy()
    logger.info(f"   Filtered data: {len(data):,} events (no vertical barriers)")

    # Core features
    core_features = [
        'log_return_1', 'log_return_4', 'atr_14', 'vol_20', 'vol_regime',
        'ema_13', 'ema_21', 'ema_spread', 'minute_of_day_sin', 'minute_of_day_cos'
    ]
    feature_cols = [f for f in core_features if f in data.columns]
    logger.info(f"   Using {len(feature_cols)} features")

    # Define test months (use periods with enough data)
    test_months = [
        ('Aug 2024', '2024-08-01', '2024-08-31'),
        ('Sep 2024', '2024-09-01', '2024-09-30'),
        ('Oct 2024', '2024-10-01', '2024-10-31'),
        ('Nov 2024', '2024-11-01', '2024-11-30'),
        ('Dec 2024', '2024-12-01', '2024-12-31'),
        ('Jan 2025', '2025-01-01', '2025-01-31'),
    ]

    logger.info(f"\n📊 Running Walk-Forward Validation")
    logger.info(f"   Test Periods: {len(test_months)} months")
    logger.info(f"   Model: Logistic Regression (simple baseline)")

    # Run validation
    results_df = walk_forward_test(data, feature_cols, test_months)

    # Summary statistics
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY STATISTICS")
    logger.info(f"{'='*60}")

    positive_months = results_df['positive'].sum()
    total_months = len(results_df)
    cumulative_pnl = results_df['pnl'].sum()
    avg_accuracy = results_df['accuracy'].mean()
    avg_auc = results_df['auc'].mean()
    avg_sharpe = results_df['sharpe'].mean()
    max_dd = results_df['max_dd'].min()

    logger.info(f"Positive Months: {positive_months}/{total_months} ({100*positive_months/total_months:.1f}%)")
    logger.info(f"Cumulative PnL: ${cumulative_pnl:,.2f}")
    logger.info(f"Avg Accuracy: {avg_accuracy:.1%}")
    logger.info(f"Avg AUC: {avg_auc:.3f}")
    logger.info(f"Avg Sharpe: {avg_sharpe:.2f}")
    logger.info(f"Max DD (worst month): ${max_dd:,.2f}")

    # Decision criteria
    logger.info(f"\n{'='*60}")
    logger.info("GO/NO-GO DECISION (Week 2 Day 13)")
    logger.info(f"{'='*60}")

    criteria_met = []

    # Criterion 1: Positive months
    if positive_months >= 4:
        logger.info(f"✅ Positive Months: {positive_months}/6 (target: 4/6)")
        criteria_met.append(True)
    else:
        logger.warning(f"❌ Positive Months: {positive_months}/6 (target: 4/6)")
        criteria_met.append(False)

    # Criterion 2: Cumulative PnL
    if cumulative_pnl >= 500:
        logger.info(f"✅ Cumulative PnL: ${cumulative_pnl:,.2f} (target: $500+)")
        criteria_met.append(True)
    else:
        logger.warning(f"❌ Cumulative PnL: ${cumulative_pnl:,.2f} (target: $500+)")
        criteria_met.append(False)

    # Criterion 3: Avg Sharpe
    if avg_sharpe >= 0.8:
        logger.info(f"✅ Avg Sharpe: {avg_sharpe:.2f} (target: 0.8+)")
        criteria_met.append(True)
    else:
        logger.warning(f"⚠️  Avg Sharpe: {avg_sharpe:.2f} (target: 0.8+)")
        criteria_met.append(False)

    # Criterion 4: Max DD
    if max_dd >= -1500:
        logger.info(f"✅ Max DD: ${max_dd:,.2f} (target: >-$1,500)")
        criteria_met.append(True)
    else:
        logger.warning(f"❌ Max DD: ${max_dd:,.2f} (target: >-$1,500)")
        criteria_met.append(False)

    logger.info(f"\n{'='*60}")
    if all(criteria_met):
        logger.info("🎯 STRONG GO: All criteria met - proceed to Day 14 paper trading")
    elif sum(criteria_met) >= 2:
        logger.info("⚠️  CONDITIONAL GO: Some criteria met - review results carefully")
    else:
        logger.error("❌ NO-GO: Insufficient consistency across months")
    logger.info(f"{'='*60}")

    # Save results
    output = {
        'test_months': results_df.to_dict('records'),
        'summary': {
            'positive_months': int(positive_months),
            'total_months': int(total_months),
            'cumulative_pnl': float(cumulative_pnl),
            'avg_accuracy': float(avg_accuracy),
            'avg_auc': float(avg_auc),
            'avg_sharpe': float(avg_sharpe),
            'max_dd': float(max_dd)
        },
        'criteria': {
            'positive_months': bool(positive_months >= 4),
            'cumulative_pnl': bool(cumulative_pnl >= 500),
            'avg_sharpe': bool(avg_sharpe >= 0.8),
            'max_dd': bool(max_dd >= -1500)
        },
        'decision': 'GO' if all(criteria_met) else ('CONDITIONAL' if sum(criteria_met) >= 2 else 'NO-GO')
    }

    output_path = diagnostics_dir / "week2_walkforward_validation.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n💾 Saved results to {output_path}")

    # Save CSV
    csv_path = diagnostics_dir / "week2_walkforward_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"   Saved detailed results to {csv_path}")


if __name__ == "__main__":
    main()
