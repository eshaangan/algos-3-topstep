"""
Quick model bundle creation for live trading.

Uses batch feature generation for speed.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("QUICK MODEL BUNDLE CREATION")
    logger.info("="*80)

    # 1. Load data (use recent data only for speed)
    logger.info("\n📊 Loading data...")
    data_path = Path("../data/processed/mes_bars.h5")

    with pd.HDFStore(data_path, 'r') as store:
        bars = store['/bars_5min']

    # Set timestamp as index
    if 'timestamp' in bars.columns:
        bars = bars.set_index('timestamp')

    # Use last 20K bars for speed
    bars = bars.tail(20000).copy()

    logger.info(f"✅ Loaded {len(bars):,} bars (5min)")
    logger.info(f"   Range: {bars.index[0]} to {bars.index[-1]}")

    # 2. Load feature config
    logger.info("\n⚙️  Loading feature config...")
    config_path = Path("configs/features.yaml")

    if config_path.exists():
        with open(config_path, 'r') as f:
            feature_config = yaml.safe_load(f)
    else:
        # Use default config
        logger.warning("   No features.yaml found, using defaults")
        feature_config = {
            "returns": {"lookback_bars": {"5m": [2, 4]}},
            "computation": {"eps": 1e-8}
        }

    # 3. Build features using batch processor
    logger.info("\n🔧 Generating features...")
    from features.build import build_features

    features_df = build_features(
        bars_df=bars,
        bar_size="5m",
        config=feature_config
    )

    logger.info(f"✅ Generated {len(features_df.columns)} features")
    logger.info(f"   Features: {list(features_df.columns[:10])}...")

    # 4. Create simple labels (next bar direction)
    logger.info("\n🏷️  Creating labels...")

    close_prices = bars['close']
    labels = (close_prices.shift(-1) > close_prices).astype(int)
    labels = labels.iloc[:-1]  # Drop last row (no future bar)
    features_df = features_df.iloc[:-1]  # Match length

    # Remove NaN rows
    valid_idx = ~features_df.isna().any(axis=1) & ~labels.isna()
    features_df = features_df[valid_idx]
    labels = labels[valid_idx]

    logger.info(f"✅ {len(labels):,} valid samples")
    logger.info(f"   Positive rate: {labels.mean()*100:.1f}%")

    # 5. Train/test split
    logger.info("\n📈 Creating splits...")

    split_idx = int(len(features_df) * 0.8)
    X_train = features_df.iloc[:split_idx]
    y_train = labels.iloc[:split_idx]
    X_test = features_df.iloc[split_idx:]
    y_test = labels.iloc[split_idx:]

    logger.info(f"   Train: {len(X_train):,} samples")
    logger.info(f"   Test:  {len(X_test):,} samples")

    # 6. Preprocessing
    logger.info("\n🔄 Preprocessing...")

    medians = X_train.median().values
    means = X_train.mean().values
    stds = X_train.std().values
    stds[stds == 0] = 1.0

    X_train_scaled = (X_train.values - means) / stds
    X_test_scaled = (X_test.values - means) / stds

    # 7. Train model
    logger.info("\n🤖 Training model...")

    model = lgb.LGBMClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=50,
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

    # Evaluate
    y_pred_train = model.predict_proba(X_train_scaled)[:, 1]
    y_pred_test = model.predict_proba(X_test_scaled)[:, 1]

    train_auc = roc_auc_score(y_train, y_pred_train)
    test_auc = roc_auc_score(y_test, y_pred_test)
    train_acc = accuracy_score(y_train, y_pred_train > 0.5)
    test_acc = accuracy_score(y_test, y_pred_test > 0.5)

    logger.info(f"   Train AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
    logger.info(f"   Test AUC:  {test_auc:.4f}, Acc: {test_acc:.4f}")

    # 8. Create bundle
    logger.info("\n📦 Creating bundle...")

    feature_columns = list(features_df.columns)

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
        'thresholds': {
            'primary_threshold': 0.10,
        },
        'meta_model': None,
        'meta_preprocessor': None,
        'meta_feature_columns': None,
        'metadata': {
            'created': pd.Timestamp.now().isoformat(),
            'n_features': len(feature_columns),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'train_auc': float(train_auc),
            'test_auc': float(test_auc),
            'bar_size': '5m',
            'note': 'Quick bundle for testing - retrain with 1m bars for production'
        }
    }

    # 9. Save
    logger.info("\n💾 Saving bundle...")

    output_dir = Path("models/saved")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_bundle.pkl"

    joblib.dump(bundle, output_path)

    file_size_mb = output_path.stat().st_size / 1024 / 1024

    logger.info(f"✅ Saved to: {output_path}")
    logger.info(f"   Size: {file_size_mb:.2f} MB")

    # 10. Verify
    logger.info("\n✅ Verifying...")

    loaded = joblib.load(output_path)
    assert 'primary_model' in loaded
    assert 'primary_preprocessor' in loaded
    assert 'primary_feature_columns' in loaded

    logger.info("   Bundle verified!")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("✅ SUCCESS!")
    logger.info("="*80)
    logger.info(f"\nModel bundle created: {output_path}")
    logger.info(f"  - Features: {len(feature_columns)}")
    logger.info(f"  - Test AUC: {test_auc:.4f}")
    logger.info(f"  - Test Accuracy: {test_acc:.4f}")
    logger.info(f"\n⚠️  NOTE: This model uses 5min bars.")
    logger.info("   For production, retrain with 1min bars using the notebook.")
    logger.info("\n✅ Next: Run 'bash monday_startup.sh' to verify!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
