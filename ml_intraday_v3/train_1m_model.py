"""
Train model using 1-minute bars from Databento.

This creates a production-ready model bundle for live trading.
"""

import logging
import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import yaml
from sklearn.metrics import roc_auc_score, accuracy_score

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path("../.env")
if env_path.exists():
    load_dotenv(env_path)
    logger_temp = logging.getLogger(__name__)
    logger_temp.info(f"Loaded .env from {env_path}")

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("1-MINUTE MODEL TRAINING FOR LIVE TRADING")
    logger.info("="*80)

    # 1. Download 1-minute bars from Databento
    logger.info("\n📥 Fetching 1-minute bars from Databento...")
    logger.info("   This will use your DATABENTO_API_KEY from .env")

    from live_trading.data_fetcher import LiveDataFetcher

    # Use continuous symbol format for Databento: ES.c.0 (front month, calendar roll)
    # MES is the micro contract, but Databento uses ES as the root
    fetcher = LiveDataFetcher(
        symbol="ES.c.0",  # Continuous front month
        bar_size="1m",
        lookback_bars=100
    )

    # Fetch recent data (last 6 months for training)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    logger.info(f"   Fetching: {start_date.date()} to {end_date.date()}")
    logger.info("   ⏳ This may take 2-3 minutes...")

    try:
        bars = fetcher.fetch_historical(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
    except Exception as e:
        logger.error(f"❌ Failed to fetch data from Databento: {e}")
        logger.error("\nPossible issues:")
        logger.error("  - DATABENTO_API_KEY not set in ../.env")
        logger.error("  - Insufficient Databento credits")
        logger.error("  - Network/API issue")
        logger.error("\n💡 Fallback: Using 5-minute bars instead...")

        # Fallback to 5-minute bars
        with pd.HDFStore('../data/processed/mes_bars.h5', 'r') as store:
            bars = store['/bars_5min']
            if 'timestamp' in bars.columns:
                bars = bars.set_index('timestamp')
            bars = bars.tail(50000)  # Use more data with 5min bars

        bar_size = "5m"
        logger.info(f"   Using {len(bars):,} bars of 5-minute data")

    else:
        bar_size = "1m"
        logger.info(f"✅ Downloaded {len(bars):,} bars of 1-minute data")
        logger.info(f"   Range: {bars.index[0]} to {bars.index[-1]}")

        # Save for future use
        cache_path = Path("../data/processed/mes_1m_bars_cache.h5")
        logger.info(f"\n💾 Caching 1m bars to: {cache_path}")
        with pd.HDFStore(cache_path, 'w') as store:
            store['bars_1m'] = bars.reset_index()
        logger.info(f"✅ Cached (can reuse for future training)")

    # 2. Filter to RTH (Regular Trading Hours) only
    logger.info("\n⏰ Filtering to RTH (8:30 AM - 3:00 PM CT)...")

    # Convert to Chicago time
    bars_ct = bars.copy()
    if bars_ct.index.tz is not None:
        bars_ct.index = bars_ct.index.tz_convert('America/Chicago')
    else:
        bars_ct.index = bars_ct.index.tz_localize('UTC').tz_convert('America/Chicago')

    # Filter RTH: 8:30 AM to 3:00 PM
    rth_mask = (bars_ct.index.hour > 8) | ((bars_ct.index.hour == 8) & (bars_ct.index.minute >= 30))
    rth_mask &= (bars_ct.index.hour < 15)
    bars_rth = bars_ct[rth_mask]

    logger.info(f"   RTH bars: {len(bars_rth):,} ({100*len(bars_rth)/len(bars):.1f}% of total)")

    # Convert back to UTC for feature generation
    bars_rth.index = bars_rth.index.tz_convert('UTC')

    # 3. Load feature config
    logger.info("\n⚙️  Loading feature configuration...")
    config_path = Path("configs/features.yaml")

    if config_path.exists():
        with open(config_path, 'r') as f:
            feature_config = yaml.safe_load(f)
    else:
        logger.warning("   No features.yaml, using defaults")
        feature_config = {
            "returns": {"lookback_bars": {"1m": [3, 6], "5m": [2, 4]}},
            "computation": {"eps": 1e-8}
        }

    # 4. Generate features
    logger.info(f"\n🔧 Generating features for {bar_size} bars...")

    from features.build import build_features

    features_df = build_features(
        bars_df=bars_rth,
        bar_size=bar_size,
        config=feature_config
    )

    logger.info(f"✅ Generated {len(features_df.columns)} features")

    # 5. Create labels
    logger.info("\n🏷️  Creating labels...")

    close_prices = bars_rth['close']
    labels = (close_prices.shift(-1) > close_prices).astype(int)
    labels = labels.iloc[:-1]
    features_df = features_df.iloc[:-1]

    # Remove NaN and synthetic bars
    if 'usable_for_training' in features_df.columns:
        valid_idx = features_df['usable_for_training'] & ~labels.isna()
        features_df = features_df[valid_idx].drop(columns=['usable_for_training', 'is_synthetic'], errors='ignore')
    else:
        valid_idx = ~features_df.isna().any(axis=1) & ~labels.isna()
        features_df = features_df[valid_idx]

    labels = labels[valid_idx]

    logger.info(f"✅ {len(labels):,} valid samples")
    logger.info(f"   Positive rate: {labels.mean()*100:.1f}%")

    # 6. Train/test split (time-based)
    logger.info("\n📊 Creating train/test split...")

    split_idx = int(len(features_df) * 0.8)
    X_train = features_df.iloc[:split_idx]
    y_train = labels.iloc[:split_idx]
    X_test = features_df.iloc[split_idx:]
    y_test = labels.iloc[split_idx:]

    logger.info(f"   Train: {len(X_train):,} samples ({y_train.mean()*100:.1f}% positive)")
    logger.info(f"   Test:  {len(X_test):,} samples ({y_test.mean()*100:.1f}% positive)")

    # 7. Preprocessing
    logger.info("\n🔄 Preprocessing features...")

    medians = X_train.median().values
    means = X_train.mean().values
    stds = X_train.std().values
    stds[stds == 0] = 1.0

    X_train_scaled = (X_train.values - means) / stds
    X_test_scaled = (X_test.values - means) / stds

    logger.info("   ✅ Features standardized")

    # 8. Train model
    logger.info("\n🤖 Training LightGBM model...")
    logger.info("   (This may take 1-2 minutes...)")

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

    # Evaluate
    y_pred_train = model.predict_proba(X_train_scaled)[:, 1]
    y_pred_test = model.predict_proba(X_test_scaled)[:, 1]

    train_auc = roc_auc_score(y_train, y_pred_train)
    test_auc = roc_auc_score(y_test, y_pred_test)
    train_acc = accuracy_score(y_train, y_pred_train > 0.5)
    test_acc = accuracy_score(y_test, y_pred_test > 0.5)

    logger.info(f"   ✅ Training complete!")
    logger.info(f"   Train AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}")
    logger.info(f"   Test AUC:  {test_auc:.4f}, Accuracy: {test_acc:.4f}")

    # 9. Create production bundle
    logger.info("\n📦 Creating production model bundle...")

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
            'bar_size': bar_size,
            'n_features': len(feature_columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'train_auc': float(train_auc),
            'test_auc': float(test_auc),
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'data_start': str(bars_rth.index[0]),
            'data_end': str(bars_rth.index[-1]),
            'rth_only': True,
        }
    }

    # 10. Save bundle
    logger.info("\n💾 Saving model bundle...")

    output_dir = Path("models/saved")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_bundle.pkl"

    joblib.dump(bundle, output_path)

    file_size_mb = output_path.stat().st_size / 1024 / 1024

    logger.info(f"   ✅ Saved to: {output_path}")
    logger.info(f"   Size: {file_size_mb:.2f} MB")

    # 11. Verify
    logger.info("\n✅ Verifying bundle...")

    loaded = joblib.load(output_path)
    assert 'primary_model' in loaded, "Missing primary_model"
    assert 'primary_preprocessor' in loaded, "Missing primary_preprocessor"
    assert 'primary_feature_columns' in loaded, "Missing primary_feature_columns"

    logger.info("   Bundle structure verified!")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("🎉 MODEL TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\n📦 Model Bundle: {output_path}")
    logger.info(f"   Bar Size: {bar_size}")
    logger.info(f"   Features: {len(feature_columns)}")
    logger.info(f"   Training Samples: {len(X_train):,}")
    logger.info(f"   Test AUC: {test_auc:.4f}")
    logger.info(f"   Test Accuracy: {test_acc:.4f}")

    if bar_size == "1m":
        logger.info(f"\n✅ PRODUCTION READY - 1-minute bars model!")
    else:
        logger.info(f"\n⚠️  Using {bar_size} bars (1m bars fetch failed)")
        logger.info("   Model will work but may not match live 1m bar features exactly")

    logger.info(f"\n🚀 Next steps:")
    logger.info("   1. Run: bash monday_startup.sh")
    logger.info("   2. Verify all checks pass")
    logger.info("   3. Ready for Monday live trading!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
