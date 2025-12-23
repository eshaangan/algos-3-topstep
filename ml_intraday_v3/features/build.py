"""
Feature building module for V3 pipeline.

Computes all features causally (no lookahead) with deterministic ordering.
Handles NaN/synthetic bars explicitly per config policy.
"""

import logging
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from .registry import get_feature_registry, filter_registry_for_bar_size, FeatureSpec

logger = logging.getLogger(__name__)


def build_features(
    bars_df: pd.DataFrame,
    bar_size: Literal["1m", "5m"],
    config: dict,
) -> pd.DataFrame:
    """
    Build feature DataFrame from bars DataFrame.

    All features are computed causally - a feature at time t uses only
    data with timestamps <= t. No lookahead.

    Args:
        bars_df: OHLCV DataFrame with UTC DatetimeIndex
            Required columns: open, high, low, close, volume
            Optional columns: is_synthetic, minute_of_day, day_of_week
        bar_size: Bar size ("1m" or "5m")
        config: Features config dict (from features.yaml)

    Returns:
        DataFrame with features in deterministic column order
        Index matches bars_df index

    NaN Handling:
        - If bars have NaN OHLCV (synthetic bars), features will be NaN
        - Features with lookback have NaN for initial bars
        - Policy controlled by config.computation.nan_policy:
          - "keep_with_mask": Keep all rows, add usable_for_training mask
          - "drop": Drop rows with any NaN feature (NOT IMPLEMENTED)
    """
    if not isinstance(bars_df.index, pd.DatetimeIndex):
        raise ValueError("bars_df must have DatetimeIndex")

    logger.info(f"Building features for {len(bars_df)} bars (bar_size={bar_size})")

    # Get epsilon from config
    eps = config.get("computation", {}).get("eps", 1e-8)

    # Get registry for this bar size
    full_registry = get_feature_registry(config)
    registry = filter_registry_for_bar_size(full_registry, bar_size)

    logger.info(f"Computing {len(registry)} features for bar_size={bar_size}")

    # Sort bars by timestamp to ensure causal computation
    df = bars_df.sort_index()

    # Check required columns
    required_cols = ["open", "high", "low", "close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Initialize feature dict (will build in registry order)
    features = {}

    # -------------------------------------------------------------------------
    # 1. RETURNS
    # -------------------------------------------------------------------------
    logger.debug("Computing return features")

    # Single-bar log return (always computed)
    # log(close_t / close_{t-1}) = log(close_t) - log(close_{t-1})
    log_close = np.log(df["close"])
    features["log_return_1"] = log_close.diff(1)

    # Multi-bar log returns
    returns_config = config.get("returns", {})
    lookback_1m = returns_config.get("lookback_bars", {}).get("1m", [3, 6])
    lookback_5m = returns_config.get("lookback_bars", {}).get("5m", [2, 4])

    if bar_size == "1m":
        for k in lookback_1m:
            features[f"log_return_{k}"] = log_close.diff(k)
    elif bar_size == "5m":
        for k in lookback_5m:
            features[f"log_return_{k}"] = log_close.diff(k)

    # -------------------------------------------------------------------------
    # 2. VOLATILITY
    # -------------------------------------------------------------------------
    logger.debug("Computing volatility features")

    # True Range: max(high-low, |high-prev_close|, |low-prev_close|)
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    features["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR: EMA of true_range
    atr_period = config.get("volatility", {}).get("atr_period", 14)
    features[f"atr_{atr_period}"] = (
        features["true_range"].ewm(span=atr_period, adjust=False).mean()
    )

    # -------------------------------------------------------------------------
    # 3. TREND
    # -------------------------------------------------------------------------
    logger.debug("Computing trend features")

    ema_fast_period = config.get("trend", {}).get("ema_fast_period", 13)
    ema_slow_period = config.get("trend", {}).get("ema_slow_period", 34)

    # EMAs of close
    ema_fast = df["close"].ewm(span=ema_fast_period, adjust=False).mean()
    ema_slow = df["close"].ewm(span=ema_slow_period, adjust=False).mean()

    features[f"ema_{ema_fast_period}"] = ema_fast
    features[f"ema_{ema_slow_period}"] = ema_slow

    # EMA spread and ratio
    features["ema_spread"] = ema_fast - ema_slow
    features["ema_ratio"] = ema_fast / (ema_slow + eps)

    # -------------------------------------------------------------------------
    # 4. STRUCTURE (Candle features)
    # -------------------------------------------------------------------------
    if config.get("structure", {}).get("enabled", True):
        logger.debug("Computing candle structure features")

        features["candle_body"] = df["close"] - df["open"]
        features["candle_range"] = df["high"] - df["low"]

        # Body percentage (avoid division by zero)
        features["body_pct"] = features["candle_body"] / (
            features["candle_range"] + eps
        )

        # Wicks
        features["upper_wick"] = df["high"] - pd.concat(
            [df["open"], df["close"]], axis=1
        ).max(axis=1)
        features["lower_wick"] = pd.concat(
            [df["open"], df["close"]], axis=1
        ).min(axis=1) - df["low"]

    # -------------------------------------------------------------------------
    # 5. TIME (Cyclical encodings)
    # -------------------------------------------------------------------------
    if config.get("time", {}).get("enabled", True):
        logger.debug("Computing time features")

        minutes_per_session = config.get("time", {}).get("minutes_per_session", 1440)

        # Check if minute_of_day already exists in bars_df
        if "minute_of_day" in df.columns:
            minute_of_day = df["minute_of_day"]
        else:
            # Compute from index (UTC hour * 60 + minute)
            minute_of_day = df.index.hour * 60 + df.index.minute

        # Cyclical encoding
        angle = 2 * np.pi * minute_of_day / minutes_per_session
        features["minute_of_day_sin"] = np.sin(angle)
        features["minute_of_day_cos"] = np.cos(angle)

        # Day of week
        if "day_of_week" in df.columns:
            features["day_of_week"] = df["day_of_week"]
        else:
            # Monday=0, Sunday=6
            features["day_of_week"] = df.index.dayofweek

    # -------------------------------------------------------------------------
    # 6. META (Flags and masks)
    # -------------------------------------------------------------------------
    if config.get("output", {}).get("include_synthetic_flag", True):
        if "is_synthetic" in df.columns:
            features["is_synthetic"] = df["is_synthetic"]
        else:
            # No synthetic flag in input, create all False
            features["is_synthetic"] = pd.Series(False, index=df.index)

    # Usable for training mask
    nan_policy = config.get("computation", {}).get("nan_policy", "keep_with_mask")
    if nan_policy == "keep_with_mask":
        logger.debug("Computing usable_for_training mask")

        # Build feature DataFrame (without mask) to check for NaNs
        feature_cols_to_check = []
        for spec in registry:
            if spec.name in ["is_synthetic", "usable_for_training"]:
                continue  # Don't check meta columns
            feature_cols_to_check.append(spec.name)

        # usable_for_training = True if all checked features are non-NaN
        usable_mask = pd.Series(True, index=df.index)
        for col in feature_cols_to_check:
            if col in features:
                usable_mask &= ~features[col].isna()

        features["usable_for_training"] = usable_mask

    # -------------------------------------------------------------------------
    # Assemble DataFrame in registry order
    # -------------------------------------------------------------------------
    logger.debug("Assembling feature DataFrame in registry order")

    # Get column order from registry
    ordered_columns = [spec.name for spec in registry]

    # Build DataFrame with exact column ordering
    features_df = pd.DataFrame(index=df.index)
    for col in ordered_columns:
        if col in features:
            features_df[col] = features[col]
        else:
            logger.warning(f"Feature {col} registered but not computed")

    # Verify column order matches registry
    assert list(features_df.columns) == ordered_columns, (
        "Column order mismatch! "
        f"Expected: {ordered_columns}, Got: {list(features_df.columns)}"
    )

    logger.info(f"Built {len(features_df.columns)} features")
    logger.info(f"Feature columns: {list(features_df.columns)}")

    # Log NaN statistics
    nan_counts = features_df.isna().sum()
    if nan_counts.any():
        logger.info(f"NaN counts per feature:\n{nan_counts[nan_counts > 0]}")

    if "usable_for_training" in features_df.columns:
        n_usable = features_df["usable_for_training"].sum()
        n_total = len(features_df)
        pct_usable = 100 * n_usable / n_total if n_total > 0 else 0
        logger.info(
            f"Usable for training: {n_usable}/{n_total} ({pct_usable:.2f}%)"
        )

    return features_df
