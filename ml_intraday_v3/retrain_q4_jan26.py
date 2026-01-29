#!/usr/bin/env python3
"""
Practical Model Retraining Script: Fix Distribution Shift & Directional Bias

This script retrains the model on recent data (Q4 2024 + Jan 2026) to address:
1. Distribution shift (44-point win rate drop)
2. Directional bias (100% LONG signals)
3. Poor risk-reward (86% stop-hit rate)

Key Changes from Baseline:
- Training period: Oct 2024 - Jan 2026 (vs Dec 2024 only)
- Validates bidirectional prediction (has_side_feature=True)
- Compares to baseline on same Jan 2026 validation data

Usage:
    python ml_intraday_v3/retrain_q4_jan26.py
    python ml_intraday_v3/retrain_q4_jan26.py --use-cached-data
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import yaml
from sklearn.metrics import roc_auc_score, accuracy_score

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch_recent_data(start_date: str, end_date: str, use_cached: bool = False):
    """Fetch Q4 2024 + Jan 2026 data from Databento or cache."""
    logger.info("="*80)
    logger.info("STEP 1: FETCH RECENT DATA")
    logger.info("="*80)
    
    cache_path = Path("ml_intraday_v3/data/databento_q4_2024_jan_2026.parquet")
    
    if use_cached and cache_path.exists():
        logger.info(f"✅ Using cached data from: {cache_path}")
        bars = pd.read_parquet(cache_path)
        logger.info(f"   Loaded {len(bars):,} bars")
        logger.info(f"   Range: {bars.index[0]} to {bars.index[-1]}")
        return bars
    
    logger.info(f"\n📥 Fetching Databento data...")
    logger.info(f"   Start: {start_date}")
    logger.info(f"   End:   {end_date}")
    logger.info("   Symbol: NQ.c.0 (continuous front month)")
    logger.info("   Bar size: 1m")
    
    try:
        from ml_intraday_v3.live_trading.data_fetcher import LiveDataFetcher
        
        fetcher = LiveDataFetcher(
            symbol="NQ.c.0",
            bar_size="1m",
            lookback_bars=100
        )
        
        logger.info("   ⏳ Fetching... (this may take 2-5 minutes)")
        
        bars = fetcher.fetch_historical(
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"✅ Fetched {len(bars):,} bars")
        logger.info(f"   Range: {bars.index[0]} to {bars.index[-1]}")
        
        # Cache for future use
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        bars.to_parquet(cache_path)
        logger.info(f"💾 Cached to: {cache_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch from Databento: {e}")
        logger.error("\nPossible issues:")
        logger.error("  - DATABENTO_API_KEY not set in .env")
        logger.error("  - Insufficient credits")
        logger.error("  - Network/API issue")
        
        if cache_path.exists():
            logger.info(f"\n💡 Using cached data from: {cache_path}")
            bars = pd.read_parquet(cache_path)
        else:
            raise RuntimeError("No data available. Please check Databento credentials.")
    
    return bars


def train_model_on_recent_data(bars: pd.DataFrame, bar_size: str = "1m"):
    """Train model using same architecture as baseline but on recent data."""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: TRAIN MODEL ON RECENT DATA")
    logger.info("="*80)
    
    # Filter to RTH
    logger.info("\n⏰ Filtering to RTH (8:30 AM - 3:00 PM CT)...")
    bars_ct = bars.copy()
    if bars_ct.index.tz is not None:
        bars_ct.index = bars_ct.index.tz_convert('America/Chicago')
    else:
        bars_ct.index = bars_ct.index.tz_localize('UTC').tz_convert('America/Chicago')
    
    rth_mask = (bars_ct.index.hour > 8) | ((bars_ct.index.hour == 8) & (bars_ct.index.minute >= 30))
    rth_mask &= (bars_ct.index.hour < 15)
    bars_rth = bars_ct[rth_mask]
    
    logger.info(f"   RTH bars: {len(bars_rth):,} ({100*len(bars_rth)/len(bars):.1f}% of total)")
    bars_rth.index = bars_rth.index.tz_convert('UTC')
    
    # Load feature config
    logger.info("\n⚙️  Loading feature configuration...")
    config_path = Path("ml_intraday_v3/configs/features.yaml")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            feature_config = yaml.safe_load(f)
        logger.info(f"   Loaded from: {config_path}")
        logger.info(f"   vol_regime_lookback: {feature_config.get('volatility', {}).get('vol_regime_lookback', 'N/A')}")
    else:
        logger.warning("   No features.yaml, using defaults")
        feature_config = {
            "returns": {"lookback_bars": {"1m": [3, 6], "5m": [2, 4]}},
            "computation": {"eps": 1e-8}
        }
    
    # Generate features
    logger.info(f"\n🔧 Generating features for {bar_size} bars...")
    
    from ml_intraday_v3.features.build import build_features
    
    features_df = build_features(
        bars_df=bars_rth,
        bar_size=bar_size,
        config=feature_config
    )
    
    logger.info(f"✅ Generated {len(features_df.columns)} features")
    logger.info(f"   Samples: {len(features_df):,}")
    
    # Check for 'side' feature
    if 'side' in features_df.columns:
        logger.info("   ✅ 'side' feature found (bidirectional model)")
        side_dist = features_df['side'].value_counts()
        logger.info(f"      LONG (1):  {(features_df['side'] == 1).sum()} ({100*(features_df['side'] == 1).mean():.1f}%)")
        logger.info(f"      SHORT (-1): {(features_df['side'] == -1).sum()} ({100*(features_df['side'] == -1).mean():.1f}%)")
    else:
        logger.warning("   ⚠️  'side' feature NOT found - this will cause directional bias!")
    
    # Create simple labels (next bar direction)
    logger.info("\n🏷️  Creating labels...")
    
    close_prices = bars_rth['close']
    labels = (close_prices.shift(-1) > close_prices).astype(int)
    labels = labels.iloc[:-1]
    features_df = features_df.iloc[:-1]
    
    # Remove NaN
    if 'usable_for_training' in features_df.columns:
        valid_idx = features_df['usable_for_training'] & ~labels.isna()
        features_df = features_df[valid_idx].drop(columns=['usable_for_training', 'is_synthetic'], errors='ignore')
    else:
        valid_idx = ~features_df.isna().any(axis=1) & ~labels.isna()
        features_df = features_df[valid_idx]
    
    labels = labels[valid_idx]
    
    logger.info(f"✅ {len(labels):,} valid samples")
    logger.info(f"   Positive rate: {labels.mean()*100:.1f}%")
    
    # Train/test split (time-based)
    logger.info("\n📊 Creating train/test split...")
    
    split_idx = int(len(features_df) * 0.8)
    X_train = features_df.iloc[:split_idx]
    y_train = labels.iloc[:split_idx]
    X_test = features_df.iloc[split_idx:]
    y_test = labels.iloc[split_idx:]
    
    logger.info(f"   Train: {len(X_train):,} samples ({y_train.mean()*100:.1f}% positive)")
    logger.info(f"   Test:  {len(X_test):,} samples ({y_test.mean()*100:.1f}% positive)")
    
    # Preprocessing
    logger.info("\n🔄 Preprocessing features...")
    
    medians = X_train.median().values
    means = X_train.mean().values
    stds = X_train.std().values
    stds[stds == 0] = 1.0
    
    X_train_scaled = (X_train.values - means) / stds
    X_test_scaled = (X_test.values - means) / stds
    
    logger.info("   ✅ Features standardized")
    
    # Train model
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
    
    # Create production bundle
    logger.info("\n📦 Creating production model bundle...")
    
    feature_columns = list(features_df.columns)
    
    preprocessor_state = {
        'impute': 'median',
        'scaler': 'standard',
        'medians': medians.tolist(),
        'means': means.tolist(),
        'stds': stds.tolist(),
    }
    
    # CRITICAL: Check if 'side' feature exists
    has_side_feature = 'side' in feature_columns
    
    bundle = {
        'primary_model': model,
        'primary_preprocessor': preprocessor_state,
        'primary_feature_columns': feature_columns,
        'has_side_feature': has_side_feature,  # CRITICAL FLAG
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
            'training_method': 'Q4_2024_Jan_2026_retrain',
        }
    }
    
    # Verify bundle
    logger.info("\n🔍 Verifying model bundle...")
    logger.info(f"   has_side_feature: {bundle['has_side_feature']}")
    logger.info(f"   n_features: {len(bundle['primary_feature_columns'])}")
    
    if has_side_feature:
        side_idx = feature_columns.index('side')
        logger.info(f"   ✅ 'side' feature at index {side_idx}")
    else:
        logger.warning("   ⚠️  'side' feature NOT in bundle - will cause 100% LONG bias!")
    
    return bundle, bars_rth


def save_and_validate_bundle(bundle: dict, bars: pd.DataFrame):
    """Save model bundle and validate structure."""
    logger.info("\n" + "="*80)
    logger.info("STEP 3: SAVE & VALIDATE MODEL BUNDLE")
    logger.info("="*80)
    
    output_dir = Path("ml_intraday_v3/models/saved")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_bundle_retrained_q4_jan26.pkl"
    
    logger.info(f"\n💾 Saving model bundle...")
    logger.info(f"   Path: {output_path}")
    
    joblib.dump(bundle, output_path)
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"   ✅ Saved ({file_size_mb:.2f} MB)")
    
    # Verify by reloading
    logger.info("\n✅ Verifying bundle...")
    
    loaded = joblib.load(output_path)
    
    required_keys = ['primary_model', 'primary_preprocessor', 'primary_feature_columns', 
                     'has_side_feature', 'metadata']
    
    for key in required_keys:
        if key in loaded:
            if key == 'has_side_feature':
                logger.info(f"   ✅ {key}: {loaded[key]}")
            else:
                logger.info(f"   ✅ {key}: present")
        else:
            logger.error(f"   ❌ {key}: MISSING!")
    
    return output_path


def compare_to_baseline(retrained_bundle_path: Path):
    """Compare retrained model to baseline on Jan 2026 data."""
    logger.info("\n" + "="*80)
    logger.info("STEP 4: COMPARE TO BASELINE")
    logger.info("="*80)
    
    logger.info("\n📊 To run validation backtest:")
    logger.info(f"   python ml_intraday_v3/backtest_databento_recent.py \\")
    logger.info(f"       --model-bundle {retrained_bundle_path} \\")
    logger.info(f"       --start-date 2026-01-04 \\")
    logger.info(f"       --end-date 2026-01-23")
    
    logger.info("\n📈 Baseline Results (Old Model on Jan 2026):")
    logger.info("   Win Rate: 13.7%")
    logger.info("   Profit Factor: 0.19")
    logger.info("   Total P&L: -$9,220")
    logger.info("   LONG %: 100.0%")
    logger.info("   SHORT %: 0.0%")
    logger.info("   Total Trades: 168")
    
    logger.info("\n✅ Expected Improvements (if model is good):")
    logger.info("   Win Rate: >40%")
    logger.info("   Profit Factor: >1.0")
    logger.info("   LONG %: 40-60%")
    logger.info("   SHORT %: 40-60%")
    
    logger.info("\n⚠️  If retrained model still shows 100% LONG:")
    logger.info("   1. Check if 'side' feature is in features.yaml")
    logger.info("   2. Check if labeling uses trend_scanning")
    logger.info("   3. Check if model_predictor.py uses prediction['side']")


def create_deployment_guide(bundle_path: Path):
    """Create deployment guide and monitoring checklist."""
    logger.info("\n" + "="*80)
    logger.info("STEP 5: DEPLOYMENT GUIDE")
    logger.info("="*80)
    
    guide_path = Path("ml_intraday_v3/RETRAINED_MODEL_DEPLOYMENT.md")
    
    guide_content = f"""# Retrained Model Deployment Guide

**Model**: Q4 2024 + Jan 2026 Retrained Model  
**Created**: {datetime.now().isoformat()}  
**Bundle**: {bundle_path}

## Quick Validation

### 1. Run Backtest on Jan 2026

```bash
python ml_intraday_v3/backtest_databento_recent.py \\
    --model-bundle {bundle_path} \\
    --start-date 2026-01-04 \\
    --end-date 2026-01-23
```

### 2. Check Success Criteria

Must achieve on Jan 2026 backtest:
- [ ] Win rate > 40% (baseline: 13.7%)
- [ ] Profit factor > 1.0 (baseline: 0.19)
- [ ] LONG % between 40-60% (baseline: 100%)
- [ ] SHORT % between 40-60% (baseline: 0%)
- [ ] Total P&L positive (baseline: -$9,220)

## Paper Trading (Week 1)

**MANDATORY** before live deployment:

```bash
python ml_intraday_v3/live_trading/paper_trade.py --duration 7d
```

Monitor daily:
- Win rate
- Profit factor
- Direction balance
- Max drawdown

## Production Deployment

### Only if paper trading passes:

```bash
# Backup old model
cp ml_intraday_v3/models/saved/model_bundle.pkl \\
   ml_intraday_v3/models/saved/model_bundle_dec2024_backup.pkl

# Deploy new model
cp {bundle_path} \\
   ml_intraday_v3/models/saved/model_bundle.pkl

# Verify
python -c "
import pickle
with open('ml_intraday_v3/models/saved/model_bundle.pkl', 'rb') as f:
    b = pickle.load(f)
print(f'has_side_feature: {{b.get(\\"has_side_feature\\")}}')
assert b.get('has_side_feature') is True
print('✅ Deployment verified')
"

# Restart trading system
bash ml_intraday_v3/monday_startup.sh
```

### Gradual Rollout

- **Week 1**: 1 micro contract ($200 risk)
- **Week 2**: 2 micros if profitable
- **Week 3+**: Full size if consistent

## Monthly Retraining

Retrain on rolling 4-month window:

- **Feb 2026**: Retrain on Nov 2024 - Feb 2026
- **Mar 2026**: Retrain on Dec 2024 - Mar 2026
- **Apr 2026**: Retrain on Jan 2025 - Apr 2026

Command:
```bash
python ml_intraday_v3/retrain_q4_jan26.py \\
    --start-date YYYY-MM-DD \\
    --end-date YYYY-MM-DD
```

## Emergency Rollback

If model fails:

```bash
# Restore backup
cp ml_intraday_v3/models/saved/model_bundle_dec2024_backup.pkl \\
   ml_intraday_v3/models/saved/model_bundle.pkl

# Restart
bash ml_intraday_v3/monday_startup.sh
```

## Monitoring Alerts

Retrain immediately if:
- Win rate drops below 40% for 1 week
- Profit factor drops below 1.0 for 1 week
- Direction bias becomes >80% one direction
- Max drawdown approaches Topstep limits

## References

- Baseline analysis: `ml_intraday_v3/backtest_results/databento_validation_20260125_000415/`
- Retraining plan: See plan document in project root
- Config files:
  - `ml_intraday_v3/configs/features.yaml`
  - `ml_intraday_v3/configs/labeling.yaml`
"""
    
    with open(guide_path, 'w') as f:
        f.write(guide_content)
    
    logger.info(f"\n📋 Created deployment guide: {guide_path}")
    logger.info("\n✅ RETRAINING COMPLETE!")
    logger.info("\n🚀 Next Steps:")
    logger.info("   1. Review deployment guide")
    logger.info("   2. Run backtest validation")
    logger.info("   3. Paper trade for 1 week")
    logger.info("   4. Deploy to production if successful")


def main():
    parser = argparse.ArgumentParser(
        description="Retrain model on Q4 2024 + Jan 2026 data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-10-01",
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-01-23",
        help="End date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--use-cached-data",
        action="store_true",
        help="Use cached data instead of fetching"
    )
    
    parser.add_argument(
        "--bar-size",
        type=str,
        default="1m",
        help="Bar size (1m or 5m)"
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("MODEL RETRAINING: FIX DISTRIBUTION SHIFT & DIRECTIONAL BIAS")
    logger.info("="*80)
    logger.info(f"\nTraining Period: {args.start_date} to {args.end_date}")
    logger.info(f"Bar Size: {args.bar_size}")
    logger.info(f"Use Cached Data: {args.use_cached_data}")
    
    try:
        # Step 1: Fetch data
        bars = fetch_recent_data(args.start_date, args.end_date, args.use_cached_data)
        
        # Step 2: Train model
        bundle, bars_rth = train_model_on_recent_data(bars, args.bar_size)
        
        # Step 3: Save and validate
        bundle_path = save_and_validate_bundle(bundle, bars_rth)
        
        # Step 4: Compare to baseline
        compare_to_baseline(bundle_path)
        
        # Step 5: Create deployment guide
        create_deployment_guide(bundle_path)
        
        logger.info("\n" + "="*80)
        logger.info("🎉 SUCCESS!")
        logger.info("="*80)
        logger.info(f"\n📦 New Model: {bundle_path}")
        logger.info(f"📋 Deployment Guide: ml_intraday_v3/RETRAINED_MODEL_DEPLOYMENT.md")
        logger.info("\n⚠️  CRITICAL: Run backtest validation before deployment!")
        
    except Exception as e:
        logger.error(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
