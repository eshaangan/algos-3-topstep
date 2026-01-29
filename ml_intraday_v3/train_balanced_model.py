#!/usr/bin/env python3
"""
Train Balanced Bidirectional Model

Simplified training script that:
1. Loads 2024-2025 data (no 2026 leak)
2. Generates balanced 50/50 LONG/SHORT events
3. Applies triple-barrier labeling
4. Trains LightGBM model with sample weighting
5. Saves model bundle for testing
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ml_intraday_v3.labels.events import generate_events, balance_events
from ml_intraday_v3.labels.triple_barrier import apply_triplebarrier
from ml_intraday_v3.core.instrument import InstrumentSpec
from ml_intraday_v3.features.build import build_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("BALANCED BIDIRECTIONAL MODEL TRAINING")
    logger.info("="*80)
    
    # Load data (2024-2025 only, no 2026)
    data_path = Path("data/processed/mes_bars_databento_rth.h5")
    logger.info(f"\n📊 Loading data from: {data_path}")
    
    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()
    
    logger.info(f"   Full dataset: {len(bars):,} bars ({bars.index[0].date()} to {bars.index[-1].date()})")
    
    # Split train/test (CRITICAL: no 2026 data)
    train_start = pd.Timestamp('2024-01-01', tz='UTC')
    train_end = pd.Timestamp('2025-11-30 23:59:59', tz='UTC')
    test_start = pd.Timestamp('2025-12-01', tz='UTC')
    test_end = pd.Timestamp('2025-12-31 23:59:59', tz='UTC')
    
    bars_train = bars[(bars.index >= train_start) & (bars.index <= train_end)]
    bars_test = bars[(bars.index >= test_start) & (bars.index <= test_end)]
    
    logger.info(f"\n📅 Train period: {bars_train.index[0].date()} to {bars_train.index[-1].date()} ({len(bars_train):,} bars)")
    logger.info(f"📅 Test period:  {bars_test.index[0].date()} to {bars_test.index[-1].date()} ({len(bars_test):,} bars)")
    logger.info(f"✅ No 2026 data in training set")
    
    # Load configs
    logger.info("\n⚙️  Loading configurations...")
    
    with open("ml_intraday_v3/configs/labeling.yaml") as f:
        labeling_config = yaml.safe_load(f)
    
    with open("ml_intraday_v3/configs/execution_spec.yaml") as f:
        execution_spec = yaml.safe_load(f)
    
    with open("ml_intraday_v3/configs/features.yaml") as f:
        feature_config = yaml.safe_load(f)
    
    with open("ml_intraday_v3/configs/training.yaml") as f:
        training_config = yaml.safe_load(f)
    
    # Create instrument spec for MES
    instrument_spec = InstrumentSpec(
        symbol="MES",
        tick_size_points=0.25,
        contract_multiplier_usd_per_point=5.0
    )
    
    # Generate events (train) - uses labeling.yaml: primary_labeling.event_policy (cusum recommended)
    event_policy = (labeling_config.get("primary_labeling") or {}).get("event_policy", "cusum")
    logger.info(f"\n🎯 Generating events (event_policy={event_policy})...")
    events_train = generate_events(
        bars_df=bars_train,
        bar_size="5m",
        labeling_config=labeling_config,
        execution_spec=execution_spec,
    )
    logger.info(f"   Train events: {len(events_train):,}")
    
    # Apply triple-barrier labeling
    logger.info("\n🏷️  Applying triple-barrier labeling...")
    events_train = apply_triplebarrier(
        bars_df=bars_train,
        events_df=events_train,
        bar_size="5m",
        labeling_config=labeling_config,
        execution_spec=execution_spec,
        instrument_spec=instrument_spec
    )
    logger.info(f"   Labeled events: {len(events_train):,}")
    logger.info(f"   Label distribution: {events_train['y'].value_counts().to_dict()}")
    
    # Check side distribution before balancing
    if 'side' in events_train.columns:
        long_pct_before = (events_train['side'] == 1).sum() / len(events_train) * 100
        logger.info(f"   Side distribution (before balance): {long_pct_before:.1f}% LONG")
    
    # Balance events (50/50 LONG/SHORT)
    logger.info("\n⚖️  Balancing LONG/SHORT events...")
    events_train = balance_events(
        events=events_train,
        target_long_ratio=0.50,
        method='undersample'
    )
    
    # Generate test events
    logger.info("\n🎯 Generating test events...")
    events_test = generate_events(
        bars_df=bars_test,
        bar_size="5m",
        labeling_config=labeling_config,
        execution_spec=execution_spec,
    )
    
    events_test = apply_triplebarrier(
        bars_df=bars_test,
        events_df=events_test,
        bar_size="5m",
        labeling_config=labeling_config,
        execution_spec=execution_spec,
        instrument_spec=instrument_spec
    )
    logger.info(f"   Test events: {len(events_test):,}, y distribution: {events_test['y'].value_counts().to_dict()}")
    
    # Build features
    logger.info("\n🔧 Building features...")
    features_train = build_features(bars_train, "5m", feature_config)
    features_test = build_features(bars_test, "5m", feature_config)
    logger.info(f"   Train features: {len(features_train):,} bars, {len(features_train.columns)} features")
    logger.info(f"   Test features:  {len(features_test):,} bars")
    
    # Merge events with features
    logger.info("\n🔗 Merging events with features...")
    
    # Get t0 timestamps
    if 't0' in events_train.columns:
        t0_train = events_train['t0'].tolist()
        t0_test = events_test['t0'].tolist()
    else:
        t0_train = events_train.index.tolist()
        t0_test = events_test.index.tolist()
    
    # Reindex features to event timestamps
    features_at_t0_train = features_train.reindex(t0_train)
    features_at_t0_test = features_test.reindex(t0_test)
    
    # Check for missing
    missing_train = features_at_t0_train.isna().all(axis=1)
    missing_test = features_at_t0_test.isna().all(axis=1)
    
    if missing_train.any():
        logger.warning(f"   {missing_train.sum()} train events have no features (will be dropped)")
    if missing_test.any():
        logger.warning(f"   {missing_test.sum()} test events have no features (will be dropped)")
    
    # Combine
    dataset_train = pd.concat([
        events_train[['side', 'y']].reset_index(drop=True),
        features_at_t0_train.reset_index(drop=True)
    ], axis=1)
    
    dataset_test = pd.concat([
        events_test[['side', 'y']].reset_index(drop=True),
        features_at_t0_test.reset_index(drop=True)
    ], axis=1)
    
    # Extract labels
    y_train = dataset_train['y'].astype(int)
    y_test = dataset_test['y'].astype(int)
    
    # Drop rows with missing features
    valid_train = ~dataset_train.isna().any(axis=1)
    valid_test = ~dataset_test.isna().any(axis=1)
    
    dataset_train = dataset_train[valid_train]
    y_train = y_train[valid_train]
    dataset_test = dataset_test[valid_test]
    y_test = y_test[valid_test]
    
    # Drop 'y' from features
    dataset_train = dataset_train.drop(columns=['y'])
    dataset_test = dataset_test.drop(columns=['y'])
    
    logger.info(f"   Final train set: {len(dataset_train):,} events")
    logger.info(f"   Final test set:  {len(dataset_test):,} events")
    
    # Feature list
    # INCLUDE 'side' feature so model can learn directional context
    feature_columns = [c for c in dataset_train.columns]
    X_train = dataset_train[feature_columns]
    X_test = dataset_test[feature_columns]
    
    logger.info(f"   Features: {len(feature_columns)}")
    logger.info(f"   Includes 'side' feature: {'side' in feature_columns}")
    
    # Train model
    logger.info("\n🤖 Training LightGBM model...")
    
    model_params = training_config.get('model', {}).get('params', {})
    logger.info(f"   Model parameters: {model_params}")
    
    model = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        random_state=42,
        verbose=-1,
        **model_params
    )
    
    # Sample weights (if available)
    sample_weight_col = training_config.get('sample_weight', {}).get('column', 'w_final')
    if sample_weight_col in events_train.columns:
        w_train = events_train.loc[valid_train, sample_weight_col].values
        logger.info(f"   Using sample weights: {sample_weight_col}")
    else:
        w_train = None
        logger.info(f"   No sample weights (column '{sample_weight_col}' not found)")
    
    model.fit(X_train, y_train, sample_weight=w_train)
    
    logger.info(f"   ✅ Model trained")
    
    # Evaluate
    logger.info("\n📊 Evaluating model...")
    
    unique_labels = sorted(y_train.unique())
    logger.info(f"   Unique labels: {unique_labels}")
    
    # Multiclass AUC
    if len(unique_labels) == 3:
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovr')
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    else:
        y_pred_train = model.predict_proba(X_train)[:, 1]
        y_pred_test = model.predict_proba(X_test)[:, 1]
        train_auc = roc_auc_score(y_train, y_pred_train)
        test_auc = roc_auc_score(y_test, y_pred_test)
    
    y_pred_train_class = model.predict(X_train)
    y_pred_test_class = model.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train_class)
    test_acc = accuracy_score(y_test, y_pred_test_class)
    
    logger.info(f"\n   Train AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}")
    logger.info(f"   Test AUC:  {test_auc:.4f}, Accuracy: {test_acc:.4f}")
    
    logger.info("\n   Test Classification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred_test_class, target_names=['Stop', 'Vertical', 'Target'])}")
    
    # Create preprocessor (from OLD_BASELINE for compatibility)
    logger.info("\n🔧 Creating preprocessor...")
    old_bundle_path = Path("ml_intraday_v3/models/saved/model_bundle_OLD_BASELINE.pkl")
    if old_bundle_path.exists():
        old_bundle = joblib.load(old_bundle_path)
        preprocessor = old_bundle.get('primary_preprocessor')
        logger.info("   Copied preprocessor from OLD_BASELINE")
        
        # Add 'side' stats if missing (OLD_BASELINE didn't have it)
        if len(preprocessor['means']) < len(feature_columns):
            logger.info("   Extending preprocessor for 'side' feature")
            # 'side' is usually the first column or last. Let's append if it's new.
            # But wait, X_train has side. We need new stats for the new feature set.
            # Since we are training a new model with new features, we should CREATE A NEW PREPROCESSOR
            # copying old one is risky if dimensions change.
            
            logger.info("   ⚠️ Feature count mismatch (old vs new). Creating FRESH preprocessor.")
            preprocessor = {
                'impute': 'median',
                'scaler': 'standard',
                'medians': X_train.median().values,
                'means': X_train.mean().values,
                'stds': X_train.std().values
            }
    else:
        # Create simple preprocessor
        preprocessor = {
            'impute': 'median',
            'scaler': 'standard',
            'medians': X_train.median().values,
            'means': X_train.mean().values,
            'stds': X_train.std().values
        }
        logger.info("   Created new preprocessor")
    
    # Save model bundle
    logger.info("\n💾 Saving model bundle...")
    
    bundle = {
        'primary_model': model,
        'primary_feature_columns': feature_columns,
        'primary_preprocessor': preprocessor,
        'has_side_feature': 'side' in feature_columns,
        'has_dual_model': False,
        'thresholds': {
            'primary_threshold': training_config.get('meta', {}).get('threshold_primary', 0.10)
        },
        'meta_model': None,
        'metadata': {
            'created': datetime.now().isoformat(),
            'training_method': 'Balanced_V3_Q1_2024_Q4_2025',
            'train_period': f"{train_start.date()} to {train_end.date()}",
            'test_period': f"{test_start.date()} to {test_end.date()}",
            'train_events': len(dataset_train),
            'train_long_pct': (dataset_train['side'] == 1).mean() * 100,
            'train_short_pct': (dataset_train['side'] == -1).mean() * 100,
            'test_events': len(dataset_test),
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'feature_count': len(feature_columns),
        }
    }
    
    output_dir = Path("ml_intraday_v3/models/saved")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_bundle_balanced_v3.pkl"
    
    joblib.dump(bundle, output_path)
    
    logger.info(f"   Saved to: {output_path}")
    logger.info(f"\n   Bundle metadata:")
    for key, value in bundle['metadata'].items():
        logger.info(f"     {key}: {value}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nModel: {output_path}")
    logger.info(f"Train events: {len(dataset_train):,} (balanced 50/50)")
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Has 'side' feature: {bundle['has_side_feature']}")
    logger.info(f"\nNext: Test on January 2026 data")
    logger.info("="*80)


if __name__ == "__main__":
    main()
