#!/usr/bin/env python3
"""
Quick test of retrained model on Dec 2025 data
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("TESTING RETRAINED MODEL ON DEC 2025")
    logger.info("="*80)

    # Load model bundle
    model_path = Path("ml_intraday_v3/models/saved/model_bundle.pkl")
    logger.info(f"\n📦 Loading model from: {model_path}")
    bundle = joblib.load(model_path)

    model = bundle['primary_model']
    feature_cols = bundle['primary_feature_columns']
    has_side = bundle.get('has_side_feature', False)

    logger.info(f"   Model: {type(model).__name__}")
    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   has_side_feature: {has_side}")
    logger.info(f"   'side' in features: {'side' in feature_cols}")

    # Load data
    logger.info(f"\n📥 Loading Dec 2025 data...")
    data_path = Path("data/processed/mes_bars_databento_rth.h5")
    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()

    # Filter to Dec 2025
    test_start = pd.Timestamp('2025-12-01', tz='UTC')
    test_end = pd.Timestamp('2025-12-18 23:59:59', tz='UTC')
    bars_test = bars[(bars.index >= test_start) & (bars.index <= test_end)]

    logger.info(f"   Loaded {len(bars_test):,} bars")
    logger.info(f"   Range: {bars_test.index[0].date()} to {bars_test.index[-1].date()}")

    # Generate features
    logger.info(f"\n🔧 Generating features...")
    config_path = Path("ml_intraday_v3/configs/features.yaml")
    with open(config_path, 'r') as f:
        feature_config = yaml.safe_load(f)

    from ml_intraday_v3.features.build import build_features
    features = build_features(bars_test, bar_size="5m", config=feature_config)

    logger.info(f"   Generated {len(features.columns)} features")

    # Check if we need to generate events for 'side' feature
    if has_side and 'side' not in features.columns:
        logger.info(f"\n🎯 Generating events for 'side' feature...")

        label_config_path = Path("ml_intraday_v3/configs/labeling.yaml")
        with open(label_config_path, 'r') as f:
            labeling_config = yaml.safe_load(f)

        exec_spec_path = Path("ml_intraday_v3/configs/execution_spec.yaml")
        with open(exec_spec_path, 'r') as f:
            execution_spec = yaml.safe_load(f)

        from ml_intraday_v3.labels.events import generate_events
        events = generate_events(
            bars_df=bars_test,
            bar_size="5m",
            labeling_config=labeling_config,
            execution_spec=execution_spec
        )

        logger.info(f"   Generated {len(events):,} events with 'side'")

        # Merge events with features (both indexed by timestamp)
        features = features.merge(events[['side']], left_index=True, right_index=True, how='inner')

        side_dist = features['side'].value_counts()
        logger.info(f"   Side distribution:")
        for side, count in side_dist.items():
            side_name = "LONG" if side == 1 else "SHORT" if side == -1 else "NEUTRAL"
            logger.info(f"      {side_name:8s} ({side:+2d}): {count:,} ({100*count/len(features):.1f}%)")

    # Align features with model expectations
    logger.info(f"\n🔄 Aligning features with model...")
    missing_cols = [c for c in feature_cols if c not in features.columns]
    extra_cols = [c for c in features.columns if c not in feature_cols]

    if missing_cols:
        logger.warning(f"   Missing {len(missing_cols)} features: {missing_cols[:5]}...")
        for col in missing_cols:
            features[col] = 0

    if extra_cols:
        logger.info(f"   Dropping {len(extra_cols)} extra features")

    X_test = features[feature_cols].fillna(0)

    logger.info(f"   Test samples: {len(X_test):,}")

    # Make predictions
    logger.info(f"\n🤖 Making predictions...")
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # Analyze predictions
    logger.info(f"\n📊 Prediction Analysis:")

    unique_preds, counts = np.unique(y_pred, return_counts=True)
    for pred, count in zip(unique_preds, counts):
        pred_int = int(pred)
        pred_name = {-1: "STOP", 0: "VERTICAL", 1: "TARGET"}.get(pred_int, f"UNKNOWN({pred_int})")
        logger.info(f"   {pred_name:10s} ({pred_int:+2d}): {count:,} ({100*count/len(y_pred):.1f}%)")

    # Check if 'side' is influencing predictions
    if 'side' in X_test.columns:
        logger.info(f"\n🎯 Side Feature Analysis:")
        side_values = X_test['side'].values
        long_pct = (side_values == 1).sum() / len(side_values) * 100
        short_pct = (side_values == -1).sum() / len(side_values) * 100
        logger.info(f"   LONG side:  {long_pct:.1f}%")
        logger.info(f"   SHORT side: {short_pct:.1f}%")

        # Prediction by side
        logger.info(f"\n   Predictions by side:")
        for side_val in [-1, 1]:
            side_name = "SHORT" if side_val == -1 else "LONG"
            mask = side_values == side_val
            if mask.sum() > 0:
                side_preds = y_pred[mask]
                target_pct = (side_preds == 1).sum() / len(side_preds) * 100
                logger.info(f"      {side_name:5s}: {target_pct:.1f}% predicted TARGET")

    # Summary
    logger.info(f"\n" + "="*80)
    logger.info(f"✅ MODEL TEST COMPLETE")
    logger.info(f"="*80)
    logger.info(f"\nKey Findings:")
    logger.info(f"   - Model has 'side' feature: {has_side}")
    logger.info(f"   - Tested on {len(X_test):,} samples from Dec 2025")
    logger.info(f"   - Predictions: {dict(zip(unique_preds, counts))}")

    if 'side' in X_test.columns:
        logger.info(f"   - Side distribution: {long_pct:.1f}% LONG, {short_pct:.1f}% SHORT")
    else:
        logger.warning(f"   - WARNING: 'side' feature not used in predictions!")


if __name__ == "__main__":
    main()
