#!/usr/bin/env python3
"""
Model Comparison - Week 2 Day 10-11

Compare simple models on Dec 2025 holdout:
1. Logistic Regression (Week 1 baseline)
2. Shallow XGBoost (max_depth=2, n_estimators=50)
3. Shallow LightGBM (max_depth=3, n_estimators=100)

Decision Criteria:
- If LogReg within 2% of best → choose LogReg (simplicity wins)
- If XGBoost/LightGBM > LogReg by 3%+ → choose best performer
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)
from lightgbm import LGBMClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed - will skip XGBoost comparison")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def train_and_evaluate(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_scaler: bool = False
) -> dict:
    """Train model and return evaluation metrics."""

    logger.info(f"\n🔬 Training {model_name}...")

    # Optional scaling
    if use_scaler:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Train
    model.fit(X_train_scaled, y_train, sample_weight=w_train)

    # Predict
    y_pred_test = model.predict(X_test_scaled)
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred_test),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test),
        'auc': roc_auc_score(y_test, y_prob_test),
        'precision': precision_score(y_test, y_pred_test, average='weighted'),
        'recall': recall_score(y_test, y_pred_test, average='weighted'),
        'f1': f1_score(y_test, y_pred_test, average='weighted')
    }

    logger.info(f"   Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    logger.info(f"   AUC: {metrics['auc']:.3f}")

    return metrics, model


def main():
    """Run model comparison."""

    # Load data (use same paths as baseline)
    project_root = Path(__file__).parent.parent.parent
    bars_path = project_root / "runs" / "v3_2022_5m" / "bar_size=5m" / "bars.parquet"
    features_path = project_root / "runs" / "v3_2022_5m" / "bar_size=5m" / "features.parquet"
    events_path = project_root / "runs" / "v3_2022_5m" / "bar_size=5m" / "events.parquet"
    weights_path = project_root / "runs" / "v3_2022_5m" / "bar_size=5m" / "weights.parquet"

    # Create diagnostics directory
    diagnostics_dir = project_root / "ml_intraday_v3" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    features = pd.read_parquet(features_path)
    events = pd.read_parquet(events_path)
    weights = pd.read_parquet(weights_path)

    # Merge
    events_with_weights = events.merge(weights[['event_id', 'w_final']], on='event_id')
    events_with_weights['t0'] = pd.to_datetime(events_with_weights['t0'])
    events_with_weights = events_with_weights.set_index('t0').sort_index()

    data = features.join(events_with_weights[['y', 'w_final', 'side', 't1']], how='inner')

    # Filter vertical barriers
    data = data[data['y'] != 0].copy()
    logger.info(f"   Filtered data: {len(data):,} events")

    # Core features (same as baseline)
    core_features = [
        'log_return_1', 'log_return_4', 'atr_14', 'vol_20', 'vol_regime',
        'ema_13', 'ema_21', 'ema_spread', 'minute_of_day_sin', 'minute_of_day_cos'
    ]
    feature_cols = [f for f in core_features if f in data.columns]
    logger.info(f"Using {len(feature_cols)} features")

    # Split train/test (same as baseline)
    train_start = pd.Timestamp('2022-01-01', tz='UTC')
    train_end = pd.Timestamp('2025-09-30 23:59:59', tz='UTC')
    test_start = pd.Timestamp('2025-10-01', tz='UTC')
    test_end = pd.Timestamp('2025-11-30 23:59:59', tz='UTC')

    train_data = data[(data.index >= train_start) & (data.index <= train_end)]
    test_data = data[(data.index >= test_start) & (data.index <= test_end)]

    logger.info(f"\n📅 Train: {len(train_data):,} events | Test: {len(test_data):,} events")

    # Prepare data
    X_train = train_data[feature_cols]
    y_train = train_data['y']
    w_train = train_data['w_final']
    X_test = test_data[feature_cols]
    y_test = test_data['y']

    # Handle NaN values
    train_valid_mask = ~X_train.isna().any(axis=1)
    X_train = X_train[train_valid_mask]
    y_train = y_train[train_valid_mask]
    w_train = w_train[train_valid_mask]

    test_valid_mask = ~X_test.isna().any(axis=1)
    X_test = X_test[test_valid_mask]
    y_test = y_test[test_valid_mask]

    logger.info(f"After NaN filtering: Train={len(X_train)}, Test={len(X_test)}")

    # Models to compare
    models = {
        'Logistic Regression': (
            LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            ),
            True  # Use scaler
        ),
        'Shallow LightGBM': (
            LGBMClassifier(
                max_depth=3,
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1
            ),
            False  # No scaler needed
        )
    }

    # Add XGBoost if available
    if HAS_XGBOOST:
        models['Shallow XGBoost'] = (
            XGBClassifier(
                max_depth=2,
                n_estimators=50,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            False  # No scaler needed
        )

    # Train and evaluate all models
    results = []
    trained_models = {}

    for model_name, (model, use_scaler) in models.items():
        metrics, trained_model = train_and_evaluate(
            model_name, model, X_train, y_train, w_train, X_test, y_test, use_scaler
        )
        results.append(metrics)
        trained_models[model_name] = trained_model

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Find best model
    best_auc_idx = results_df['auc'].idxmax()
    best_model_name = results_df.loc[best_auc_idx, 'model']
    best_auc = results_df.loc[best_auc_idx, 'auc']

    logreg_auc = results_df[results_df['model'] == 'Logistic Regression']['auc'].values[0]

    logger.info("\n📊 Model Comparison Results:")
    logger.info(results_df.to_string(index=False))

    logger.info(f"\n🎯 Decision:")
    logger.info(f"   Best Model: {best_model_name} (AUC: {best_auc:.3f})")
    logger.info(f"   LogReg AUC: {logreg_auc:.3f}")

    auc_diff = best_auc - logreg_auc
    pct_diff = 100 * auc_diff / logreg_auc

    if pct_diff < 2.0:
        chosen_model = 'Logistic Regression'
        reason = f"LogReg within 2% of best ({pct_diff:.1f}%) - simplicity wins"
    elif pct_diff >= 3.0:
        chosen_model = best_model_name
        reason = f"Tree model {pct_diff:.1f}% better - choose best performer"
    else:
        chosen_model = 'Logistic Regression'
        reason = f"Tree model only {pct_diff:.1f}% better (< 3% threshold) - prefer simplicity"

    logger.info(f"   ✅ Chosen Model: {chosen_model}")
    logger.info(f"   Reason: {reason}")

    # Save results
    output = {
        'comparison_results': results,
        'best_model': best_model_name,
        'chosen_model': chosen_model,
        'reason': reason,
        'auc_difference_pct': pct_diff
    }

    output_path = diagnostics_dir / "week2_model_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n💾 Saved results to {output_path}")

if __name__ == "__main__":
    main()
