#!/usr/bin/env python3
"""
Simple Baseline Model - Logistic Regression
Week 1 Day 5-6: Prove LINEAR edge exists before adding complexity

Purpose:
- Test if 12 core features contain predictive signal
- Establish baseline for comparison with tree-based models
- Analyze feature importance via coefficients

Success Criteria:
- AUC > 0.58 → Linear edge exists, proceed
- AUC < 0.55 → No edge, need different features or approach
"""

import os
import sys
import logging
import yaml
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_data(bars_path: str, features_path: str, events_path: str, weights_path: str):
    """Load and merge bars, features, events, and weights."""
    logger.info("Loading data...")

    # Load bars (now from parquet)
    bars = pd.read_parquet(bars_path)
    if 'timestamp' in bars.columns:
        bars['timestamp'] = pd.to_datetime(bars['timestamp'])
        bars = bars.set_index('timestamp').sort_index()
    logger.info(f"   Loaded {len(bars):,} bars ({bars.index[0].date()} to {bars.index[-1].date()})")

    # Load features
    features = pd.read_parquet(features_path)
    if 'timestamp' not in features.index.names:
        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
            features = features.set_index('timestamp').sort_index()
        else:
            features.index = pd.to_datetime(features.index)
    logger.info(f"   Loaded {len(features):,} feature rows")

    # Load events
    events = pd.read_parquet(events_path)
    logger.info(f"   Loaded {len(events):,} events")

    # Load weights
    weights = pd.read_parquet(weights_path)
    logger.info(f"   Loaded {len(weights):,} weights")

    # Merge events and weights on event_id
    events_with_weights = events.merge(weights[['event_id', 'w_final']], on='event_id', how='left')

    # Set t0 as index for merging with features
    events_with_weights['t0'] = pd.to_datetime(events_with_weights['t0'])
    events_with_weights = events_with_weights.set_index('t0').sort_index()

    # Merge with features on timestamp (t0 for events)
    data = features.join(events_with_weights[['y', 'w_final', 'side', 't1']], how='inner')
    logger.info(f"   Merged dataset: {len(data):,} rows")

    return data

def filter_vertical_barriers(data: pd.DataFrame) -> pd.DataFrame:
    """Remove vertical barrier labels (y=0) to focus on actionable signals."""
    before = len(data)
    data = data[data['y'] != 0].copy()
    after = len(data)
    pct_removed = 100 * (before - after) / before
    logger.info(f"   Filtered vertical barriers: {before:,} → {after:,} ({pct_removed:.1f}% removed)")

    # Check label balance
    label_counts = data['y'].value_counts()
    logger.info(f"   Label distribution:")
    for label, count in label_counts.items():
        pct = 100 * count / len(data)
        label_name = "TARGET" if label == 1.0 else "STOP"
        logger.info(f"      {label_name} ({label:+.0f}): {count:,} ({pct:.1f}%)")

    return data

def select_core_features(data: pd.DataFrame) -> list:
    """Select core features for baseline model."""
    core_features = [
        'log_return_1',      # Single-bar momentum
        'log_return_4',      # 20-min momentum (5m bars)
        'atr_14',            # Volatility scaling
        'vol_20',            # Short-term volatility
        'vol_regime',        # Volatility regime
        'ema_13',            # Fast trend
        'ema_21',            # Slow trend
        'ema_spread',        # Trend direction
        'minute_of_day_sin', # Time cyclical
        'minute_of_day_cos'  # Time cyclical
    ]

    # Optional features (if available in data)
    optional_features = [
        'rsi_14',            # Momentum oscillator (if available)
        'macd_hist',         # Momentum divergence (if available)
    ]

    # Check which features are actually present
    available_features = [f for f in core_features if f in data.columns]
    available_optional = [f for f in optional_features if f in data.columns]
    missing_features = [f for f in core_features if f not in data.columns]
    missing_optional = [f for f in optional_features if f not in data.columns]

    logger.info(f"\n📊 Feature Selection:")
    logger.info(f"   Core features available: {len(available_features)}/10")
    if available_optional:
        logger.info(f"   Optional features available: {available_optional}")
    if missing_features:
        logger.error(f"   Missing CORE features: {missing_features}")
    if missing_optional:
        logger.info(f"   Missing optional features: {missing_optional} (will regenerate later)")

    # Return all available features (core + optional)
    all_available = available_features + available_optional
    logger.info(f"   Total features for baseline: {len(all_available)}")

    return all_available

def train_logistic_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """Train logistic regression baseline and return results."""

    logger.info("\n🔬 Training Logistic Regression Baseline...")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        class_weight='balanced',  # Handle any remaining imbalance
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )

    logger.info("   Fitting model with sample weights...")
    model.fit(X_train_scaled, y_train, sample_weight=w_train)

    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_prob_train = model.predict_proba(X_train_scaled)[:, 1]
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    results = {
        'train': {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'balanced_accuracy': balanced_accuracy_score(y_train, y_pred_train),
            'auc': roc_auc_score(y_train, y_prob_train),
            'precision': precision_score(y_train, y_pred_train, average='weighted'),
            'recall': recall_score(y_train, y_pred_train, average='weighted'),
            'f1': f1_score(y_train, y_pred_train, average='weighted')
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test),
            'auc': roc_auc_score(y_test, y_prob_test),
            'precision': precision_score(y_test, y_pred_test, average='weighted'),
            'recall': recall_score(y_test, y_pred_test, average='weighted'),
            'f1': f1_score(y_test, y_pred_test, average='weighted')
        },
        'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(),
        'classification_report': classification_report(y_test, y_pred_test, output_dict=True)
    }

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': model.coef_[0],
        'abs_coef': np.abs(model.coef_[0])
    }).sort_values('abs_coef', ascending=False)

    results['feature_importance'] = feature_importance.to_dict('records')

    # Log results
    logger.info("\n📈 Training Results:")
    logger.info(f"   Accuracy: {results['train']['accuracy']:.3f}")
    logger.info(f"   Balanced Accuracy: {results['train']['balanced_accuracy']:.3f}")
    logger.info(f"   AUC: {results['train']['auc']:.3f}")

    logger.info("\n📊 Test Results (Oct-Nov 2025 Holdout):")
    logger.info(f"   Accuracy: {results['test']['accuracy']:.3f}")
    logger.info(f"   Balanced Accuracy: {results['test']['balanced_accuracy']:.3f}")
    logger.info(f"   AUC: {results['test']['auc']:.3f}")
    logger.info(f"   Precision: {results['test']['precision']:.3f}")
    logger.info(f"   Recall: {results['test']['recall']:.3f}")
    logger.info(f"   F1 Score: {results['test']['f1']:.3f}")

    logger.info("\n🔝 Top 5 Features by Absolute Coefficient:")
    for i, row in feature_importance.head(5).iterrows():
        logger.info(f"   {row['feature']:20s}: {row['coefficient']:+.4f} (|{row['abs_coef']:.4f}|)")

    # Decision
    auc_test = results['test']['auc']
    acc_test = results['test']['accuracy']

    logger.info("\n🎯 Decision Criteria:")
    logger.info(f"   AUC: {auc_test:.3f} (target: 0.58+)")
    logger.info(f"   Accuracy: {acc_test:.3f} (target: 0.54+)")

    if auc_test >= 0.58 and acc_test >= 0.54:
        logger.info("   ✅ GO: Linear edge exists, proceed to Week 2")
    elif auc_test >= 0.55:
        logger.warning("   ⚠️  PARTIAL: Edge exists but weak, may need calibration")
    else:
        logger.error("   ❌ NO-GO: No linear edge detected, revisit features")

    return results, model, scaler, feature_importance

def plot_confusion_matrix(cm: np.ndarray, output_path: str):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['STOP (-1)', 'TARGET (+1)'],
                yticklabels=['STOP (-1)', 'TARGET (+1)'])
    plt.title('Confusion Matrix - Oct-Nov 2025 Holdout')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"   Saved confusion matrix to {output_path}")
    plt.close()

def main():
    """Main execution."""

    # Paths
    project_root = Path(__file__).parent.parent
    bars_path = project_root / "runs" / "v3_2022_5m" / "bar_size=5m" / "bars.parquet"
    features_path = project_root / "runs" / "v3_2022_5m" / "bar_size=5m" / "features.parquet"
    events_path = project_root / "runs" / "v3_2022_5m" / "bar_size=5m" / "events.parquet"
    weights_path = project_root / "runs" / "v3_2022_5m" / "bar_size=5m" / "weights.parquet"

    # Create diagnostics directory
    diagnostics_dir = project_root / "ml_intraday_v3" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_data(str(bars_path), str(features_path), str(events_path), str(weights_path))

    # Filter vertical barriers
    data = filter_vertical_barriers(data)

    # Select core features
    feature_cols = select_core_features(data)

    # Split train/test (adjusted for available data)
    train_start = pd.Timestamp('2022-01-01', tz='UTC')
    train_end = pd.Timestamp('2025-09-30 23:59:59', tz='UTC')
    test_start = pd.Timestamp('2025-10-01', tz='UTC')
    test_end = pd.Timestamp('2025-11-30 23:59:59', tz='UTC')

    train_data = data[(data.index >= train_start) & (data.index <= train_end)]
    test_data = data[(data.index >= test_start) & (data.index <= test_end)]

    logger.info(f"\n📅 Data Split:")
    logger.info(f"   Train: {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data):,} events)")
    logger.info(f"   Test:  {test_data.index[0].date()} to {test_data.index[-1].date()} ({len(test_data):,} events)")

    # Prepare X, y, w
    X_train = train_data[feature_cols]
    y_train = train_data['y']
    w_train = train_data['w_final']

    X_test = test_data[feature_cols]
    y_test = test_data['y']

    # Check for NaN values
    train_nan_count = X_train.isna().sum().sum()
    test_nan_count = X_test.isna().sum().sum()

    if train_nan_count > 0 or test_nan_count > 0:
        logger.warning(f"\n⚠️  NaN Values Detected:")
        logger.warning(f"   Train: {train_nan_count:,} NaN values")
        logger.warning(f"   Test: {test_nan_count:,} NaN values")
        logger.info("   Dropping rows with NaN values...")

        # Drop NaN rows from training
        train_valid_mask = ~X_train.isna().any(axis=1)
        X_train = X_train[train_valid_mask]
        y_train = y_train[train_valid_mask]
        w_train = w_train[train_valid_mask]

        # Drop NaN rows from test
        test_valid_mask = ~X_test.isna().any(axis=1)
        X_test = X_test[test_valid_mask]
        y_test = y_test[test_valid_mask]

        logger.info(f"   After filtering: Train={len(X_train):,}, Test={len(X_test):,}")

    # Train model
    results, model, scaler, feature_importance = train_logistic_baseline(
        X_train, y_train, w_train, X_test, y_test
    )

    # Save results
    results_path = diagnostics_dir / "week1_baseline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n💾 Saved results to {results_path}")

    # Save feature importance
    fi_path = diagnostics_dir / "week1_feature_importance.csv"
    feature_importance.to_csv(fi_path, index=False)
    logger.info(f"   Saved feature importance to {fi_path}")

    # Plot confusion matrix
    cm = np.array(results['confusion_matrix'])
    cm_path = diagnostics_dir / "week1_confusion_matrix.png"
    plot_confusion_matrix(cm, str(cm_path))

    logger.info("\n✅ Baseline training complete!")
    logger.info(f"   Check diagnostics in: {diagnostics_dir}")

if __name__ == "__main__":
    main()
