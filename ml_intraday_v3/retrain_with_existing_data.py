#!/usr/bin/env python3
"""
Retrain model using existing data - no API fetch needed.

Training: Oct 2024 - Nov 2025 (14 months)
Testing: Dec 2025 (held out)

This addresses distribution shift by training on recent data while
keeping the most recent month for validation.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import yaml
from sklearn.metrics import roc_auc_score, accuracy_score
from datetime import datetime

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("MODEL RETRAINING WITH EXISTING DATA")
    logger.info("="*80)
    
    # Step 1: Load existing data
    logger.info("\n📥 Loading existing data from data/processed/...")
    
    data_path = Path("../data/processed/mes_bars_databento_rth.h5")
    if not data_path.exists():
        logger.error(f"❌ Data file not found: {data_path}")
        sys.exit(1)
    
    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()
    
    logger.info(f"✅ Loaded {len(bars):,} bars")
    logger.info(f"   Range: {bars.index[0]} to {bars.index[-1]}")
    
    # Step 2: Split data - Train up to Nov 30, 2025; Test = Dec 2025
    logger.info("\n📊 Splitting data...")
    logger.info("   Training: Oct 2024 - Nov 2025 (14 months)")
    logger.info("   Testing: Dec 2025 (held out)")
    
    # Training period: Oct 1, 2024 to Nov 30, 2025
    train_start = pd.Timestamp('2024-10-01', tz='UTC')
    train_end = pd.Timestamp('2025-11-30 23:59:59', tz='UTC')
    
    # Test period: Dec 1-18, 2025
    test_start = pd.Timestamp('2025-12-01', tz='UTC')
    test_end = pd.Timestamp('2025-12-18 23:59:59', tz='UTC')
    
    # Filter to training period
    bars_train = bars[(bars.index >= train_start) & (bars.index <= train_end)]
    bars_test = bars[(bars.index >= test_start) & (bars.index <= test_end)]
    
    logger.info(f"   Train bars: {len(bars_train):,} ({bars_train.index[0].date()} to {bars_train.index[-1].date()})")
    logger.info(f"   Test bars: {len(bars_test):,} ({bars_test.index[0].date()} to {bars_test.index[-1].date()})")
    
    # Step 3: Generate features
    logger.info("\n🔧 Generating features...")
    
    config_path = Path("ml_intraday_v3/configs/features.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            feature_config = yaml.safe_load(f)
    else:
        feature_config = {
            "returns": {"lookback_bars": {"5m": [2, 4]}},
            "computation": {"eps": 1e-8}
        }
    
    from ml_intraday_v3.features.build import build_features
    
    # Generate features for full dataset (to avoid lookback issues at split)
    bars_full = bars[(bars.index >= train_start) & (bars.index <= test_end)]
    features_full = build_features(
        bars_df=bars_full,
        bar_size="5m",
        config=feature_config
    )
    
    logger.info(f"✅ Generated {len(features_full.columns)} features")
    
    # Check for 'side' feature
    if 'side' in features_full.columns:
        logger.info("   ✅ 'side' feature found (bidirectional model)")
        side_dist = features_full['side'].value_counts()
        logger.info(f"      LONG (1): {(features_full['side'] == 1).sum()} ({100*(features_full['side'] == 1).mean():.1f}%)")
        logger.info(f"      SHORT (-1): {(features_full['side'] == -1).sum()} ({100*(features_full['side'] == -1).mean():.1f}%)")
    else:
        logger.warning("   ⚠️  'side' feature NOT found!")
    
    # Step 4: Create labels
    logger.info("\n🏷️  Creating labels...")
    
    close_prices = bars_full['close']
    labels_full = (close_prices.shift(-1) > close_prices).astype(int)
    labels_full = labels_full.iloc[:-1]
    features_full = features_full.iloc[:-1]
    
    # Remove NaN
    if 'usable_for_training' in features_full.columns:
        valid_idx = features_full['usable_for_training'] & ~labels_full.isna()
        features_full = features_full[valid_idx].drop(
            columns=['usable_for_training', 'is_synthetic'], errors='ignore'
        )
    else:
        valid_idx = ~features_full.isna().any(axis=1) & ~labels_full.isna()
        features_full = features_full[valid_idx]
    
    labels_full = labels_full[valid_idx]
    
    # Split into train/test based on dates
    X_train = features_full[features_full.index <= train_end]
    y_train = labels_full[labels_full.index <= train_end]
    X_test = features_full[features_full.index >= test_start]
    y_test = labels_full[labels_full.index >= test_start]
    
    logger.info(f"✅ {len(y_train):,} train samples ({y_train.mean()*100:.1f}% positive)")
    logger.info(f"✅ {len(y_test):,} test samples ({y_test.mean()*100:.1f}% positive)")
    
    # Step 5: Preprocess
    logger.info("\n🔄 Preprocessing...")
    
    medians = X_train.median().values
    means = X_train.mean().values
    stds = X_train.std().values
    stds[stds == 0] = 1.0
    
    X_train_scaled = (X_train.values - means) / stds
    X_test_scaled = (X_test.values - means) / stds
    
    logger.info("   ✅ Standardized")
    
    # Step 6: Train model
    logger.info("\n🤖 Training LightGBM...")
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        callbacks=[lgb.log_evaluation(period=0)]
    )
    
    # Step 7: Evaluate
    logger.info("\n📊 Evaluating...")
    
    y_pred_train = model.predict_proba(X_train_scaled)[:, 1]
    y_pred_test = model.predict_proba(X_test_scaled)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_pred_train)
    test_auc = roc_auc_score(y_test, y_pred_test)
    train_acc = accuracy_score(y_train, y_pred_train > 0.5)
    test_acc = accuracy_score(y_test, y_pred_test > 0.5)
    
    logger.info(f"   Train AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}")
    logger.info(f"   Test AUC:  {test_auc:.4f}, Accuracy: {test_acc:.4f}")
    
    # Step 8: Create bundle
    logger.info("\n📦 Creating model bundle...")
    
    feature_columns = list(features_full.columns)
    has_side_feature = 'side' in feature_columns
    
    preprocessor_state = {
        'impute': 'median',
        'scaler': 'standard',
        'medians': medians.tolist(),
        'means': means.tolist(),
        'stds': stds.tolist(),
    }
    
    bundle = {
        'primary_model': model,
        'primary_preprocessor': preprocessor_state,
        'primary_feature_columns': feature_columns,
        'has_side_feature': has_side_feature,
        'thresholds': {'primary_threshold': 0.10},
        'meta_model': None,
        'meta_preprocessor': None,
        'meta_feature_columns': None,
        'metadata': {
            'created': datetime.now().isoformat(),
            'bar_size': '5m',
            'n_features': len(feature_columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'train_auc': float(train_auc),
            'test_auc': float(test_auc),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_period': f'{train_start.date()} to {train_end.date()}',
            'test_period': f'{test_start.date()} to {test_end.date()}',
            'rth_only': True,
            'training_method': 'Oct2024_Nov2025_retrain',
        }
    }
    
    logger.info(f"   has_side_feature: {bundle['has_side_feature']}")
    if has_side_feature:
        side_idx = feature_columns.index('side')
        logger.info(f"   ✅ 'side' at index {side_idx}")
    
    # Step 9: Save
    logger.info("\n💾 Saving...")
    
    output_dir = Path("ml_intraday_v3/models/saved")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_bundle_retrained_oct2024_nov2025.pkl"
    
    joblib.dump(bundle, output_path)
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"   ✅ Saved to: {output_path}")
    logger.info(f"   Size: {file_size_mb:.2f} MB")
    
    # Step 10: Summary
    logger.info("\n" + "="*80)
    logger.info("✅ RETRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\n📦 Model: {output_path}")
    logger.info(f"   Training: {len(X_train):,} samples from {train_start.date()} to {train_end.date()}")
    logger.info(f"   Testing: {len(X_test):,} samples from {test_start.date()} to {test_end.date()}")
    logger.info(f"   Test AUC: {test_auc:.4f}")
    logger.info(f"   Test Accuracy: {test_acc:.4f}")
    logger.info(f"   has_side_feature: {has_side_feature}")
    
    logger.info("\n🚀 Next: Run backtest validation on Dec 2025")
    logger.info("   python ml_intraday_v3/backtest_databento_recent.py \\")
    logger.info(f"       --model-bundle {output_path}")


if __name__ == "__main__":
    main()
