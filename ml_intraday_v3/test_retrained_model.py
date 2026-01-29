#!/usr/bin/env python3
"""
Test retrained model on December 2025 data.

Simple direct test - load model, load Dec 2025 bars, run predictions, show metrics.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import yaml

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("TEST RETRAINED MODEL ON DEC 2025")
    logger.info("="*80)
    
    # Load model
    model_path = Path("ml_intraday_v3/models/saved/model_bundle_retrained_oct2024_nov2025.pkl")
    logger.info(f"\n📦 Loading model: {model_path}")
    bundle = joblib.load(model_path)
    
    logger.info(f"   Features: {len(bundle['primary_feature_columns'])}")
    logger.info(f"   has_side_feature: {bundle['has_side_feature']}")
    
    # Load Dec 2025 data
    logger.info("\n📥 Loading Dec 2025 data...")
    data_path = Path("data/processed/mes_bars_databento_rth.h5")
    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()
    
    # Filter to Dec 2025
    dec_bars = bars[(bars.index >= '2025-12-01') & (bars.index <= '2025-12-18')]
    logger.info(f"   Dec 2025: {len(dec_bars)} bars")
    
    # Generate features
    logger.info("\n🔧 Generating features...")
    config_path = Path("ml_intraday_v3/configs/features.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            feature_config = yaml.safe_load(f)
    else:
        feature_config = {"returns": {"lookback_bars": {"5m": [2, 4]}}}
    
    from ml_intraday_v3.features.build import build_features
    
    features = build_features(dec_bars, bar_size="5m", config=feature_config)
    
    # Remove NaN
    if 'usable_for_training' in features.columns:
        valid_idx = features['usable_for_training']
        features = features[valid_idx].drop(columns=['usable_for_training', 'is_synthetic'], errors='ignore')
    else:
        valid_idx = ~features.isna().any(axis=1)
        features = features[valid_idx]
    
    logger.info(f"   Valid samples: {len(features)}")
    
    # Create actual labels (next bar direction)
    close_prices = dec_bars.loc[features.index, 'close']
    actual_labels = (close_prices.shift(-1) > close_prices).astype(int)
    actual_labels = actual_labels.iloc[:-1]
    features = features.iloc[:-1]
    
    # Remove any remaining NaN
    valid_mask = ~actual_labels.isna()
    actual_labels = actual_labels[valid_mask]
    features = features[valid_mask]
    
    logger.info(f"   Final samples: {len(features)}")
    logger.info(f"   Actual positive rate: {actual_labels.mean()*100:.1f}%")
    
    # Make predictions
    logger.info("\n🎯 Making predictions...")
    
    # Preprocess
    preprocessor = bundle['primary_preprocessor']
    means = np.array(preprocessor['means'])
    stds = np.array(preprocessor['stds'])
    
    X_scaled = (features.values - means) / stds
    
    # Predict
    model = bundle['primary_model']
    probs = model.predict_proba(X_scaled)[:, 1]
    predictions = (probs > 0.5).astype(int)
    
    logger.info(f"   Predicted positive rate: {predictions.mean()*100:.1f}%")
    
    # Evaluate
    logger.info("\n📊 RESULTS:")
    logger.info("="*80)
    
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    
    accuracy = accuracy_score(actual_labels, predictions)
    try:
        auc = roc_auc_score(actual_labels, probs)
    except:
        auc = 0.5
    
    cm = confusion_matrix(actual_labels, predictions)
    
    logger.info(f"\n🎯 Accuracy: {accuracy*100:.2f}%")
    logger.info(f"📈 AUC: {auc:.4f}")
    logger.info(f"\n📊 Confusion Matrix:")
    logger.info(f"              Predicted")
    logger.info(f"              0      1")
    logger.info(f"   Actual 0: {cm[0,0]:4d}  {cm[0,1]:4d}")
    logger.info(f"   Actual 1: {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Calculate more metrics
    tp = cm[1,1]
    fp = cm[0,1]
    tn = cm[0,0]
    fn = cm[1,0]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"\n📈 Classification Metrics:")
    logger.info(f"   Precision: {precision*100:.2f}%")
    logger.info(f"   Recall: {recall*100:.2f}%")
    logger.info(f"   F1 Score: {f1:.4f}")
    
    # Probability distribution
    logger.info(f"\n📊 Prediction Distribution:")
    logger.info(f"   Mean probability: {probs.mean():.4f}")
    logger.info(f"   Median probability: {np.median(probs):.4f}")
    logger.info(f"   Min probability: {probs.min():.4f}")
    logger.info(f"   Max probability: {probs.max():.4f}")
    
    # Confidence-based analysis
    high_conf = probs > 0.6
    low_conf = probs < 0.4
    neutral = ~(high_conf | low_conf)
    
    logger.info(f"\n🎯 By Confidence Level:")
    if high_conf.sum() > 0:
        high_acc = accuracy_score(actual_labels[high_conf], predictions[high_conf])
        logger.info(f"   High (>0.6): {high_conf.sum()} samples, {high_acc*100:.1f}% accuracy")
    if low_conf.sum() > 0:
        low_acc = accuracy_score(actual_labels[low_conf], predictions[low_conf])
        logger.info(f"   Low (<0.4): {low_conf.sum()} samples, {low_acc*100:.1f}% accuracy")
    if neutral.sum() > 0:
        neu_acc = accuracy_score(actual_labels[neutral], predictions[neutral])
        logger.info(f"   Neutral (0.4-0.6): {neutral.sum()} samples, {neu_acc*100:.1f}% accuracy")
    
    logger.info("\n" + "="*80)
    logger.info("✅ TEST COMPLETE")
    logger.info("="*80)
    
    # Summary
    logger.info(f"\n📋 SUMMARY:")
    logger.info(f"   Model: Oct 2024 - Nov 2025 training")
    logger.info(f"   Test: Dec 2025 (held out)")
    logger.info(f"   Samples: {len(actual_labels)}")
    logger.info(f"   Accuracy: {accuracy*100:.2f}%")
    logger.info(f"   AUC: {auc:.4f}")
    logger.info(f"   Baseline (random): 50%")
    
    if accuracy > 0.52:
        logger.info(f"\n✅ Model beats random baseline!")
    else:
        logger.info(f"\n⚠️  Model near random - needs improvement")
    
    if not bundle['has_side_feature']:
        logger.info(f"\n⚠️  WARNING: Model doesn't have 'side' feature")
        logger.info(f"   This may cause directional bias in live trading")


if __name__ == "__main__":
    main()
