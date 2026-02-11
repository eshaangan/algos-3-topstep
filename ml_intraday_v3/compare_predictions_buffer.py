#!/usr/bin/env python3
"""
Direct Prediction Comparison: 100-bar vs 300-bar Buffer

Tests if buffer size affects model predictions by:
1. Generating features with 100-bar rolling buffer
2. Generating features with 300-bar rolling buffer
3. Comparing prediction diversity and feature quality
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import yaml
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from ml_intraday_v3.features.build import build_features
from ml_intraday_v3.live_trading.model_predictor import LiveModelPredictor


def load_data():
    """Load Jan 2026 data."""
    data_path = Path("ml_intraday_v3/ml_intraday_v3/data/jan2026_mes/mes_jan2026_5m.parquet")
    df = pd.read_parquet(data_path)

    # Convert to Chicago time for RTH filtering
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('America/Chicago')

    # Filter to RTH only
    df_rth = df.between_time('08:30', '15:00')

    logger.info(f"Loaded {len(df)} total bars, {len(df_rth)} RTH bars")
    logger.info(f"Date range: {df_rth.index.min()} to {df_rth.index.max()}")

    return df_rth


def simulate_rolling_buffer(df, buffer_size, features_cfg):
    """
    Simulate rolling buffer feature generation.

    Returns DataFrame with features calculated using rolling buffer.
    """
    logger.info(f"\nSimulating {buffer_size}-bar rolling buffer...")

    all_features = []
    all_nan_counts = []

    for i in range(len(df)):
        # Get buffer (last N bars up to current)
        start_idx = max(0, i - buffer_size + 1)
        buffer = df.iloc[start_idx:i+1]

        if len(buffer) < 30:  # Skip very early bars
            continue

        # Build features on buffer
        features_df = build_features(buffer, bar_size="5m", config=features_cfg)

        # Get latest row features
        if len(features_df) > 0:
            latest_features = features_df.iloc[-1]
            nan_count = latest_features.isna().sum()

            all_features.append(latest_features)
            all_nan_counts.append(nan_count)

        if (i + 1) % 500 == 0:
            logger.info(f"  Processed {i+1}/{len(df)} bars...")

    features_df = pd.DataFrame(all_features)
    logger.info(f"Generated {len(features_df)} feature rows")

    # Analyze NaN counts
    nan_series = pd.Series(all_nan_counts)
    logger.info(f"\nNaN Statistics:")
    logger.info(f"  Mean NaN per row: {nan_series.mean():.1f}")
    logger.info(f"  Median NaN per row: {nan_series.median():.1f}")
    logger.info(f"  Max NaN per row: {nan_series.max():.0f}")
    logger.info(f"  Rows with 0 NaN: {(nan_series == 0).sum()} ({(nan_series == 0).sum() / len(nan_series) * 100:.1f}%)")
    logger.info(f"  Rows with >5 NaN: {(nan_series > 5).sum()} ({(nan_series > 5).sum() / len(nan_series) * 100:.1f}%)")

    # Check which columns are frequently NaN
    nan_freq = features_df.isna().mean()
    frequent_nan = nan_freq[nan_freq > 0.1]
    if len(frequent_nan) > 0:
        logger.info(f"\nFrequently NaN columns (>10% of rows):")
        for col, freq in frequent_nan.sort_values(ascending=False).items():
            logger.info(f"  {col}: {freq*100:.1f}%")

    return features_df, nan_series


def test_predictions(features_df, buffer_size):
    """Generate predictions and analyze diversity."""
    logger.info(f"\nTesting predictions with {buffer_size}-bar buffer...")

    # Load model
    model_path = Path("ml_intraday_v3/model_bundle_retrained_oct2024_nov2025.pkl")
    bundle = joblib.load(model_path)

    model = bundle['primary_model']
    feature_names = bundle['primary_feature_columns']
    preprocessor = bundle.get('primary_preprocessor', {})

    logger.info(f"  Model: {type(model).__name__}")
    logger.info(f"  Features: {len(feature_names)}")

    # Align features
    X = features_df[feature_names].values

    # Preprocess (handle NaN with median imputation like live trading)
    imputer_medians = np.nan_to_num(np.nanmedian(X, axis=0))

    X_clean = X.copy()
    nan_mask = np.isnan(X_clean)
    for col_idx in range(X_clean.shape[1]):
        X_clean[nan_mask[:, col_idx], col_idx] = imputer_medians[col_idx]

    # Standardize
    means = X_clean.mean(axis=0)
    stds = X_clean.std(axis=0) + 1e-8
    X_scaled = (X_clean - means) / stds

    # Predict
    probs = model.predict_proba(X_scaled)
    predictions = probs[:, 1]  # P(target reached)

    # Analyze diversity
    logger.info(f"\nPrediction Diversity:")
    logger.info(f"  Mean: {predictions.mean():.4f}")
    logger.info(f"  Std: {predictions.std():.4f}")
    logger.info(f"  Min: {predictions.min():.4f}")
    logger.info(f"  Max: {predictions.max():.4f}")
    logger.info(f"  Unique values: {len(np.unique(predictions))}")

    # Check for identical predictions (the bug we're fixing)
    value_counts = pd.Series(predictions).value_counts()
    if len(value_counts) == 1:
        logger.warning(f"  ⚠️  ALL PREDICTIONS IDENTICAL: {predictions[0]:.4f}")
    elif value_counts.iloc[0] > len(predictions) * 0.9:
        logger.warning(f"  ⚠️  {value_counts.iloc[0]/len(predictions)*100:.1f}% predictions are {value_counts.index[0]:.4f}")
    else:
        logger.info(f"  ✅ Predictions are diverse (largest cluster: {value_counts.iloc[0]/len(predictions)*100:.1f}%)")

    # Show distribution
    logger.info(f"\nPrediction Distribution:")
    for percentile in [10, 25, 50, 75, 90]:
        val = np.percentile(predictions, percentile)
        logger.info(f"  {percentile}th percentile: {val:.4f}")

    return predictions


def main():
    logger.info("="*80)
    logger.info("Buffer Size Impact on Predictions - Direct Test")
    logger.info("="*80)

    # Load data
    logger.info("\n1. Loading Jan 2026 RTH data...")
    df = load_data()

    # Load features config
    features_cfg_path = Path("ml_intraday_v3/configs/features.yaml")
    with open(features_cfg_path, 'r') as f:
        features_cfg = yaml.safe_load(f)

    # Test 100-bar buffer
    logger.info("\n" + "="*80)
    logger.info("2. Testing 100-Bar Buffer (Original - With NaN Issue)")
    logger.info("="*80)
    features_100, nan_counts_100 = simulate_rolling_buffer(df, 100, features_cfg)
    predictions_100 = test_predictions(features_100, 100)

    # Test 300-bar buffer
    logger.info("\n" + "="*80)
    logger.info("3. Testing 300-Bar Buffer (Fixed - Clean Features)")
    logger.info("="*80)
    features_300, nan_counts_300 = simulate_rolling_buffer(df, 300, features_cfg)
    predictions_300 = test_predictions(features_300, 300)

    # Compare
    logger.info("\n" + "="*80)
    logger.info("4. COMPARISON: 100-Bar vs 300-Bar")
    logger.info("="*80)

    logger.info(f"\nNaN Impact:")
    logger.info(f"  100-bar avg NaN/row: {nan_counts_100.mean():.1f}")
    logger.info(f"  300-bar avg NaN/row: {nan_counts_300.mean():.1f}")
    logger.info(f"  Improvement: {nan_counts_100.mean() - nan_counts_300.mean():.1f} fewer NaN/row")

    logger.info(f"\nPrediction Diversity:")
    logger.info(f"  100-bar std: {predictions_100.std():.6f}")
    logger.info(f"  300-bar std: {predictions_300.std():.6f}")

    if predictions_300.std() > predictions_100.std():
        improvement = (predictions_300.std() - predictions_100.std()) / predictions_100.std() * 100
        logger.info(f"  ✅ 300-bar is {improvement:.1f}% more diverse")
    else:
        logger.info(f"  ⚠️  300-bar is LESS diverse")

    logger.info(f"\n" + "="*80)
    logger.info("VERDICT")
    logger.info("="*80)

    if nan_counts_300.mean() < nan_counts_100.mean():
        logger.info("✅ 300-bar buffer reduces NaN features")
    if predictions_300.std() > predictions_100.std() * 1.1:
        logger.info("✅ 300-bar buffer increases prediction diversity")
    elif predictions_300.std() > predictions_100.std():
        logger.info("≈  300-bar buffer slightly increases prediction diversity")
    else:
        logger.info("❌ 300-bar buffer does NOT improve prediction diversity")

    logger.info("\n" + "="*80)


if __name__ == "__main__":
    main()
