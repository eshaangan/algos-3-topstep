"""
Create model bundle for live trading.

This script trains a model and creates a bundle compatible with LiveModelPredictor.
"""

import json
import pickle
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_live_bundle():
    """
    Train model and create bundle for live trading.

    Creates a model_bundle.pkl file compatible with LiveModelPredictor.
    """

    logger.info("=" * 80)
    logger.info("CREATING LIVE TRADING MODEL BUNDLE")
    logger.info("=" * 80)

    # 1. Load data
    logger.info("\n1️⃣  Loading data...")
    data_path = Path("data/processed/mes_bars.h5")

    if not data_path.exists():
        # Try alternative paths
        alt_paths = [
            Path("../data/processed/mes_bars.h5"),
            Path("data/processed/es_bars_2010_2025.h5"),
        ]
        for p in alt_paths:
            if p.exists():
                data_path = p
                break

    if not data_path.exists():
        logger.error(f"❌ Data file not found: {data_path}")
        logger.error("Please ensure you have preprocessed data available.")
        return False

    logger.info(f"   Loading from: {data_path}")

    # Load bars
    with pd.HDFStore(data_path, 'r') as store:
        # Check available keys
        available_keys = store.keys()
        logger.info(f"   Available keys: {available_keys}")

        # Use first available key (usually bars_5min)
        if '/bars_5min' in available_keys:
            bars = store['/bars_5min']
        elif 'bars' in available_keys:
            bars = store['bars']
        else:
            bars = store[available_keys[0]]

    # Ensure datetime index
    if 'timestamp' in bars.columns:
        bars = bars.set_index('timestamp')
    elif not isinstance(bars.index, pd.DatetimeIndex):
        logger.warning("   ⚠️  No datetime index found, using integer index")

    logger.info(f"   ✅ Loaded {len(bars):,} bars")
    if isinstance(bars.index, pd.DatetimeIndex):
        logger.info(f"   Date range: {bars.index[0]} to {bars.index[-1]}")
    else:
        logger.info(f"   Index range: {bars.index[0]} to {bars.index[-1]}")

    # 2. Generate features
    logger.info("\n2️⃣  Generating features...")

    # Define expected feature columns based on LiveFeatureGenerator
    feature_columns = [
        # Returns
        'log_return_1', 'log_return_2', 'log_return_3', 'log_return_5',
        # Volatility
        'true_range', 'atr_14', 'vol_20', 'vol_regime', 'parkinson_vol', 'vol_forecast',
        # Moving averages
        'ema_8', 'ema_13', 'ema_21', 'ema_spread', 'ema_ratio',
        'sma_10', 'sma_20', 'sma_30', 'trend_strength',
        # Other technical
        'autocorr_5', 'bb_position', 'volume_imbalance', 'relative_volume',
        'price_vs_vwap', 'large_move',
        # Candle features
        'candle_body', 'candle_range', 'body_pct', 'upper_wick', 'lower_wick',
        # Time features
        'minute_of_day_sin', 'minute_of_day_cos', 'day_of_week',
    ]

    from live_trading.feature_generator import LiveFeatureGenerator

    feature_gen = LiveFeatureGenerator(feature_columns)

    # Generate features for each bar
    all_features = []
    all_labels = []
    all_indices = []

    logger.info("   Processing bars (this may take a few minutes)...")

    # Start from bar 100 to have enough history
    for i in range(100, len(bars)):
        if i % 1000 == 0:
            logger.info(f"   Progress: {i}/{len(bars)} bars ({100*i/len(bars):.1f}%)")

        # Get historical window
        hist_bars = bars.iloc[:i+1]

        try:
            # Generate features
            features = feature_gen.generate_features(hist_bars)

            # Simple label: 1 if next bar closes higher, 0 otherwise
            if i < len(bars) - 1:
                current_close = bars.iloc[i]['close']
                next_close = bars.iloc[i+1]['close']
                label = 1 if next_close > current_close else 0

                all_features.append(features)
                all_labels.append(label)
                all_indices.append(bars.index[i])
        except Exception as e:
            # Skip bars with errors (usually early bars with insufficient history)
            continue

    logger.info(f"   ✅ Generated features for {len(all_features):,} bars")

    # Convert to DataFrame
    features_df = pd.DataFrame(all_features, index=all_indices)
    labels_series = pd.Series(all_labels, index=all_indices)

    # Remove any NaN values
    valid_idx = ~features_df.isna().any(axis=1)
    features_df = features_df[valid_idx]
    labels_series = labels_series[valid_idx]

    logger.info(f"   ✅ {len(features_df):,} valid samples after NaN removal")
    logger.info(f"   Features: {list(features_df.columns)}")

    # 3. Train/test split
    logger.info("\n3️⃣  Creating train/test split...")

    # Use last 20% for test
    split_idx = int(len(features_df) * 0.8)

    X_train = features_df.iloc[:split_idx]
    y_train = labels_series.iloc[:split_idx]
    X_test = features_df.iloc[split_idx:]
    y_test = labels_series.iloc[split_idx:]

    logger.info(f"   Train: {len(X_train):,} samples ({y_train.mean()*100:.1f}% positive)")
    logger.info(f"   Test:  {len(X_test):,} samples ({y_test.mean()*100:.1f}% positive)")

    # 4. Preprocess data
    logger.info("\n4️⃣  Preprocessing features...")

    # Calculate preprocessing stats on training data only
    medians = X_train.median().values
    means = X_train.mean().values
    stds = X_train.std().values
    stds[stds == 0] = 1.0  # Avoid division by zero

    # Apply preprocessing
    X_train_scaled = (X_train.values - means) / stds
    X_test_scaled = (X_test.values - means) / stds

    logger.info(f"   ✅ Standardized features")

    # 5. Train model
    logger.info("\n5️⃣  Training LightGBM model...")

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
        callbacks=[lgb.log_evaluation(period=0)]  # Suppress output
    )

    # Evaluate
    from sklearn.metrics import roc_auc_score, accuracy_score

    y_pred_train = model.predict_proba(X_train_scaled)[:, 1]
    y_pred_test = model.predict_proba(X_test_scaled)[:, 1]

    train_auc = roc_auc_score(y_train, y_pred_train)
    test_auc = roc_auc_score(y_test, y_pred_test)

    train_acc = accuracy_score(y_train, y_pred_train > 0.5)
    test_acc = accuracy_score(y_test, y_pred_test > 0.5)

    logger.info(f"   ✅ Model trained")
    logger.info(f"   Train AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}")
    logger.info(f"   Test AUC:  {test_auc:.4f}, Accuracy: {test_acc:.4f}")

    # 6. Create bundle
    logger.info("\n6️⃣  Creating model bundle...")

    feature_columns = list(features_df.columns)

    # Create preprocessor state dict
    preprocessor_state = {
        'impute': 'median',
        'scaler': 'standard',
        'medians': medians.tolist(),
        'means': means.tolist(),
        'stds': stds.tolist(),
    }

    # Create bundle with keys expected by LiveModelPredictor
    bundle = {
        'primary_model': model,
        'primary_preprocessor': preprocessor_state,
        'primary_feature_columns': feature_columns,
        'thresholds': {
            'primary_threshold': 0.10,  # Can adjust this
        },
        'meta_model': None,
        'meta_preprocessor': None,
        'meta_feature_columns': None,
        'metadata': {
            'created': pd.Timestamp.now().isoformat(),
            'n_features': len(feature_columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'train_auc': float(train_auc),
            'test_auc': float(test_auc),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'data_path': str(data_path),
        }
    }

    logger.info(f"   ✅ Bundle created with {len(feature_columns)} features")

    # 7. Save bundle
    logger.info("\n7️⃣  Saving bundle...")

    output_dir = Path("models/saved")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "model_bundle.pkl"

    import joblib
    joblib.dump(bundle, output_path)

    logger.info(f"   ✅ Saved to: {output_path}")

    # Verify it can be loaded
    logger.info("\n8️⃣  Verifying bundle...")

    loaded_bundle = joblib.load(output_path)

    assert 'primary_model' in loaded_bundle, "Missing primary_model"
    assert 'primary_preprocessor' in loaded_bundle, "Missing primary_preprocessor"
    assert 'primary_feature_columns' in loaded_bundle, "Missing primary_feature_columns"

    logger.info(f"   ✅ Bundle verified successfully")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ MODEL BUNDLE CREATED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"\nBundle saved to: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info(f"\nModel performance:")
    logger.info(f"  - Test AUC: {test_auc:.4f}")
    logger.info(f"  - Test Accuracy: {test_acc:.4f}")
    logger.info(f"  - Features: {len(feature_columns)}")
    logger.info(f"  - Training samples: {len(X_train):,}")
    logger.info(f"\nNext step: Run monday_startup.sh to verify everything is ready!")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    success = create_live_bundle()
    if not success:
        exit(1)
