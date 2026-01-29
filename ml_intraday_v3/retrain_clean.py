#!/usr/bin/env python3
"""
Clean retraining with 'side' feature - NO DATA LEAKAGE
Trains on Oct 2024 - Nov 2025, tests on Dec 2025
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

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
    logger.info("CLEAN RETRAINING WITH 'SIDE' FEATURE")
    logger.info("="*80)
    logger.info("Training: Oct 2024 - Nov 2025")
    logger.info("Testing: Dec 2025 (held out)")
    logger.info("="*80)

    # Load data
    logger.info("\n📥 Loading data...")
    data_path = Path("data/processed/mes_bars_databento_rth.h5")
    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()

    # Split periods
    train_start = pd.Timestamp('2024-10-01', tz='UTC')
    train_end = pd.Timestamp('2025-11-30 23:59:59', tz='UTC')
    test_start = pd.Timestamp('2025-12-01', tz='UTC')
    test_end = pd.Timestamp('2025-12-18 23:59:59', tz='UTC')

    bars_train = bars[(bars.index >= train_start) & (bars.index <= train_end)]
    bars_test = bars[(bars.index >= test_start) & (bars.index <= test_end)]

    logger.info(f"   Train: {len(bars_train):,} bars ({bars_train.index[0].date()} to {bars_train.index[-1].date()})")
    logger.info(f"   Test:  {len(bars_test):,} bars ({bars_test.index[0].date()} to {bars_test.index[-1].date()})")

    # Generate features
    logger.info("\n🔧 Generating features...")
    config_path = Path("ml_intraday_v3/configs/features.yaml")
    with open(config_path, 'r') as f:
        feature_config = yaml.safe_load(f)

    from ml_intraday_v3.features.build import build_features

    features_train = build_features(bars_train, bar_size="5m", config=feature_config)
    features_test = build_features(bars_test, bar_size="5m", config=feature_config)

    logger.info(f"   Train features: {len(features_train):,} x {len(features_train.columns)}")
    logger.info(f"   Test features:  {len(features_test):,} x {len(features_test.columns)}")

    # Generate events for 'side' feature
    logger.info("\n🎯 Generating events for 'side' feature...")

    label_config_path = Path("ml_intraday_v3/configs/labeling.yaml")
    with open(label_config_path, 'r') as f:
        labeling_config = yaml.safe_load(f)

    exec_spec_path = Path("ml_intraday_v3/configs/execution_spec.yaml")
    with open(exec_spec_path, 'r') as f:
        execution_spec = yaml.safe_load(f)

    from ml_intraday_v3.labels.events import generate_events
    from ml_intraday_v3.labels.triple_barrier import apply_triplebarrier
    from ml_intraday_v3.core.instrument import InstrumentSpec

    # Create instrument spec for MES (Micro E-mini S&P 500)
    instrument_spec = InstrumentSpec(
        symbol="MES",
        tick_size_points=0.25,
        contract_multiplier_usd_per_point=5.0  # $5 per point for MES
    )

    events_train = generate_events(
        bars_df=bars_train,
        bar_size="5m",
        labeling_config=labeling_config,
        execution_spec=execution_spec
    )

    events_test = generate_events(
        bars_df=bars_test,
        bar_size="5m",
        labeling_config=labeling_config,
        execution_spec=execution_spec
    )

    logger.info(f"   Train events: {len(events_train):,} (LONG: {(events_train['side']==1).sum():,}, SHORT: {(events_train['side']==-1).sum():,})")
    logger.info(f"   Test events:  {len(events_test):,} (LONG: {(events_test['side']==1).sum():,}, SHORT: {(events_test['side']==-1).sum():,})")
    
    # Apply triple-barrier labeling to get 'y' labels
    logger.info("\n🏷️  Applying triple-barrier labeling...")
    
    events_train = apply_triplebarrier(
        bars_df=bars_train,
        events_df=events_train,
        bar_size="5m",
        labeling_config=labeling_config,
        execution_spec=execution_spec,
        instrument_spec=instrument_spec
    )
    
    events_test = apply_triplebarrier(
        bars_df=bars_test,
        events_df=events_test,
        bar_size="5m",
        labeling_config=labeling_config,
        execution_spec=execution_spec,
        instrument_spec=instrument_spec
    )
    
    logger.info(f"   Train labeled: {len(events_train):,} events, y distribution: {events_train['y'].value_counts().to_dict()}")
    logger.info(f"   Test labeled:  {len(events_test):,} events, y distribution: {events_test['y'].value_counts().to_dict()}")

    # Merge: Use events as base (has t0 and 'side'), lookup features at t0
    logger.info("\n🔗 Merging features at event times...")

    # Use the same approach as ml_intraday_v3/training/dataset.py:build_event_dataset()
    # Extract t0 timestamps from events and use reindex() for exact lookup
    
    # Get t0 timestamps (events should have 't0' column)
    if 't0' in events_train.columns:
        t0_train = events_train['t0'].tolist()
        t0_test = events_test['t0'].tolist()
    else:
        # If t0 is the index
        t0_train = events_train.index.tolist()
        t0_test = events_test.index.tolist()
    
    logger.info(f"   Events train: {len(t0_train):,} timestamps")
    logger.info(f"   Events test:  {len(t0_test):,} timestamps")
    logger.info(f"   Features train: {len(features_train):,} bars")
    logger.info(f"   Features test:  {len(features_test):,} bars")
    
    # Reindex features to event timestamps (exact lookup)
    features_at_t0_train = features_train.reindex(t0_train)
    features_at_t0_test = features_test.reindex(t0_test)
    
    # Check for missing timestamps
    missing_train = features_at_t0_train.isna().all(axis=1)
    missing_test = features_at_t0_test.isna().all(axis=1)
    
    if missing_train.any():
        logger.warning(f"   {missing_train.sum()} train event timestamps not found in features")
    if missing_test.any():
        logger.warning(f"   {missing_test.sum()} test event timestamps not found in features")
    
    # Combine: events metadata (side, y) + features at those timestamps
    dataset_train = pd.concat([
        events_train[['side', 'y']].reset_index(drop=True),
        features_at_t0_train.reset_index(drop=True)
    ], axis=1)
    
    dataset_test = pd.concat([
        events_test[['side', 'y']].reset_index(drop=True),
        features_at_t0_test.reset_index(drop=True)
    ], axis=1)

    logger.info(f"   Train: {len(dataset_train):,} samples with 'side'")
    logger.info(f"   Test:  {len(dataset_test):,} samples with 'side'")

    # Extract labels from events (they already have 'y' from triple-barrier labeling)
    logger.info("\n🏷️  Extracting labels from events...")

    y_train = dataset_train['y'].astype(int)
    y_test = dataset_test['y'].astype(int)
    
    # Remove NaN rows (from missing feature timestamps)
    valid_train = ~dataset_train.isna().any(axis=1)
    valid_test = ~dataset_test.isna().any(axis=1)

    dataset_train = dataset_train[valid_train]
    y_train = y_train[valid_train]
    dataset_test = dataset_test[valid_test]
    y_test = y_test[valid_test]
    
    # Drop 'y' from features (it's the target)
    dataset_train = dataset_train.drop(columns=['y'])
    dataset_test = dataset_test.drop(columns=['y'])

    # Drop metadata columns (keep 'side'!)
    drop_cols = ['is_synthetic', 'usable_for_training']
    dataset_train = dataset_train.drop(columns=drop_cols, errors='ignore')
    dataset_test = dataset_test.drop(columns=drop_cols, errors='ignore')

    logger.info(f"   Train: {len(y_train):,} labels ({y_train.mean()*100:.1f}% positive)")
    logger.info(f"   Test:  {len(y_test):,} labels ({y_test.mean()*100:.1f}% positive)")
    logger.info(f"   Features: {list(dataset_train.columns)}")

    # Train model
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

    model.fit(dataset_train, y_train)

    # Evaluate
    logger.info("\n📊 Evaluation:")

    y_pred_train = model.predict_proba(dataset_train)[:, 1]
    y_pred_test = model.predict_proba(dataset_test)[:, 1]
    
    # For multiclass (3 classes: -1, 0, 1), use multiclass AUC
    from sklearn.preprocessing import label_binarize
    
    # Check if truly multiclass or can be treated as binary
    unique_labels = sorted(y_train.unique())
    logger.info(f"   Unique labels: {unique_labels}")
    
    if len(unique_labels) == 3:
        # Multiclass AUC (one-vs-rest)
        train_auc = roc_auc_score(y_train, model.predict_proba(dataset_train), multi_class='ovr')
        test_auc = roc_auc_score(y_test, model.predict_proba(dataset_test), multi_class='ovr')
    else:
        # Binary AUC
        train_auc = roc_auc_score(y_train, y_pred_train)
        test_auc = roc_auc_score(y_test, y_pred_test)
    
    y_pred_train_class = model.predict(dataset_train)
    y_pred_test_class = model.predict(dataset_test)
    train_acc = accuracy_score(y_train, y_pred_train_class)
    test_acc = accuracy_score(y_test, y_pred_test_class)

    logger.info(f"   Train AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}")
    logger.info(f"   Test AUC:  {test_auc:.4f}, Accuracy: {test_acc:.4f}")
    
    # Show classification report
    logger.info("\n   Test Classification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred_test_class, target_names=['Stop', 'Vertical', 'Target'])}")

    # Feature importance
    logger.info("\n📈 Top 10 Features:")
    feature_importance = pd.Series(model.feature_importances_, index=dataset_train.columns).sort_values(ascending=False)
    for i, (feat, imp) in enumerate(feature_importance.head(10).items(), 1):
        logger.info(f"      {i:2d}. {feat:20s}: {imp:.0f}")

    # Create bundle
    logger.info("\n📦 Creating model bundle...")

    bundle = {
        'primary_model': model,
        'primary_feature_columns': list(dataset_train.columns),
        'has_side_feature': True,  # ✅ CRITICAL
        'primary_preprocessor': None,  # No preprocessing (features already scaled in build_features)
        'thresholds': {'primary_threshold': 0.10},
        'meta_model': None,
        'meta_preprocessor': None,
        'meta_feature_columns': None,
        'metadata': {
            'created': datetime.now().isoformat(),
            'training_method': 'Clean_Retraining_Oct2024_Nov2025',
            'bar_size': '5m',
            'n_features': len(dataset_train.columns),
            'n_train_samples': len(dataset_train),
            'n_test_samples': len(dataset_test),
            'train_auc': float(train_auc),
            'test_auc': float(test_auc),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_period': f'{train_start.date()} to {train_end.date()}',
            'test_period': f'{test_start.date()} to {test_end.date()}',
            'rth_only': True,
            'has_side_feature': True,
            'side_feature_included': True
        }
    }

    # Save
    logger.info("\n💾 Saving...")
    output_dir = Path("ml_intraday_v3/models/saved")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_bundle_retrained_clean.pkl"

    joblib.dump(bundle, output_path)

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"   ✅ Saved to: {output_path}")
    logger.info(f"   Size: {file_size_mb:.2f} MB")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("✅ CLEAN RETRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\n📦 Model: {output_path}")
    logger.info(f"   Training: {len(dataset_train):,} samples from {train_start.date()} to {train_end.date()}")
    logger.info(f"   Testing: {len(dataset_test):,} samples from {test_start.date()} to {test_end.date()}")
    logger.info(f"   Test AUC: {test_auc:.4f}")
    logger.info(f"   Test Accuracy: {test_acc:.4f}")
    logger.info(f"   has_side_feature: True")
    logger.info(f"   Features: {len(dataset_train.columns)} (including 'side')")

    logger.info("\n🎯 Key Points:")
    logger.info("   ✅ No data leakage (no label metadata)")
    logger.info("   ✅ 'side' feature from trend_scanning")
    logger.info("   ✅ Clean features only (no exit_price, ret_gross, etc.)")

    logger.info("\n🚀 Next: Replace model_bundle.pkl and run backtest")


if __name__ == "__main__":
    main()
