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

    logger.debug(f"Building features for {len(bars_df)} bars (bar_size={bar_size})")

    # Get epsilon from config
    eps = config.get("computation", {}).get("eps", 1e-8)
    if isinstance(eps, str):
        try:
            eps = float(eps)
        except ValueError as exc:
            raise ValueError(f"Invalid eps value in config: {eps}") from exc

    # Get registry for this bar size
    full_registry = get_feature_registry(config)
    registry = filter_registry_for_bar_size(full_registry, bar_size)

    logger.debug(f"Computing {len(registry)} features for bar_size={bar_size}")

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

    # PHASE 3: Multi-horizon returns (match 12-24 bar label horizons)
    # These are critical for predicting multi-bar outcomes
    if config.get("returns", {}).get("enable_multi_horizon", True):
        multi_horizon = config.get("returns", {}).get("multi_horizon_bars", [6, 12, 24])
        for k in multi_horizon:
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

    # PHASE 3: Volatility regime indicators
    if config.get("volatility", {}).get("enable_regime_features", True):
        logger.debug("Computing volatility regime features")

        # Rolling volatility (20-bar window)
        returns_1 = features["log_return_1"]
        features["vol_20"] = returns_1.rolling(20).std()

        # Volatility regime (current vol vs median, configurable lookback)
        vol_regime_lookback = config.get("volatility", {}).get("vol_regime_lookback", 50)
        features["vol_regime"] = features["vol_20"] / (
            features["vol_20"].rolling(vol_regime_lookback).median() + eps
        )

        # Parkinson volatility (uses high-low range, more efficient estimator)
        hl_log_ratio = np.log(df["high"] / (df["low"] + eps))
        features["parkinson_vol"] = np.sqrt(
            (1 / (4 * np.log(2))) * (hl_log_ratio ** 2).rolling(20).mean()
        )

        # EWMA volatility forecast (decay factor 0.94 as per research)
        features["vol_forecast"] = returns_1.ewm(alpha=0.06).std()

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

    # PHASE 3: Advanced trend and mean reversion features
    if config.get("trend", {}).get("enable_advanced_features", True):
        logger.debug("Computing advanced trend features")

        # SMAs for trend strength
        sma_long_period = config.get("trend", {}).get("sma_long_period", 30)
        sma_20 = df["close"].rolling(20).mean()
        sma_long = df["close"].rolling(sma_long_period).mean()

        features["sma_20"] = sma_20
        features[f"sma_{sma_long_period}"] = sma_long

        # Trend strength (distance from long-term SMA, normalized)
        features["trend_strength"] = (df["close"] - sma_long) / (sma_long + eps)

        # Autocorrelation (lag-5 on 20-bar window) - detects mean reversion
        def autocorr_lag5(x):
            if len(x) < 6:
                return np.nan
            return x.autocorr(lag=5) if len(x.dropna()) > 6 else np.nan

        features["autocorr_5"] = df["close"].rolling(20).apply(autocorr_lag5, raw=False)

        # Price position relative to Bollinger Bands
        bb_std = df["close"].rolling(20).std()
        bb_upper = sma_20 + 2 * bb_std
        bb_lower = sma_20 - 2 * bb_std
        features["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + eps)

    # -------------------------------------------------------------------------
    # 4. MOMENTUM (MACD, RSI, Stochastic)
    # -------------------------------------------------------------------------
    if config.get("momentum", {}).get("enabled", True):
        logger.debug("Computing momentum features")
        
        # RSI
        rsi_period = config.get("momentum", {}).get("rsi_period", 14)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / (loss + eps)
        features[f"rsi_{rsi_period}"] = 100 - (100 / (1 + rs))
        
        # MACD
        macd_config = config.get("momentum", {}).get("macd", {})
        if macd_config.get("enabled", True):
            fast = macd_config.get("fast_period", 12)
            slow = macd_config.get("slow_period", 26)
            signal = macd_config.get("signal_period", 9)
            
            ema_fast_macd = df["close"].ewm(span=fast, adjust=False).mean()
            ema_slow_macd = df["close"].ewm(span=slow, adjust=False).mean()
            
            features["macd"] = ema_fast_macd - ema_slow_macd
            features["macd_signal"] = features["macd"].ewm(span=signal, adjust=False).mean()
            features["macd_hist"] = features["macd"] - features["macd_signal"]
            
        # Stochastic Oscillator
        stoch_config = config.get("momentum", {}).get("stochastic", {})
        if stoch_config.get("enabled", True):
            k_period = stoch_config.get("k_period", 14)
            d_period = stoch_config.get("d_period", 3)
            
            low_min = df["low"].rolling(window=k_period).min()
            high_max = df["high"].rolling(window=k_period).max()
            
            # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
            features[f"stoch_k_{k_period}"] = 100 * ((df["close"] - low_min) / (high_max - low_min + eps))
            
            # %D = SMA of %K
            features[f"stoch_d_{d_period}"] = features[f"stoch_k_{k_period}"].rolling(window=d_period).mean()

    # -------------------------------------------------------------------------
    # 5. VOLUME & MICROSTRUCTURE (Order flow proxies)
    # -------------------------------------------------------------------------
    if config.get("microstructure", {}).get("enabled", True) and "volume" in df.columns:
        logger.debug("Computing microstructure features")

        # Volume imbalance (buying vs selling pressure proxy)
        # Assumes: close > open = buying, close < open = selling
        features["volume_imbalance"] = (df["close"] - df["open"]) / (
            df["high"] - df["low"] + eps
        )

        # Volume-weighted price position (VWAP proxy)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol_price = (df["volume"] * typical_price).rolling(50).sum()
        cum_vol = df["volume"].rolling(50).sum()
        vwap = cum_vol_price / (cum_vol + eps)
        features["price_vs_vwap"] = (df["close"] - vwap) / (vwap + eps)

        # Relative volume (current vs 20-bar average)
        avg_volume = df["volume"].rolling(20).mean()
        features["relative_volume"] = df["volume"] / (avg_volume + eps)

        # Large move detection (exceeds 2× typical volatility)
        if "vol_20" in features:
            features["large_move"] = (
                features["log_return_1"].abs() > 2 * features["vol_20"]
            ).astype(int)

    # -------------------------------------------------------------------------
    # 5. STRUCTURE (Candle features)
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

    logger.debug(f"Built {len(features_df.columns)} features")
    logger.debug(f"Feature columns: {list(features_df.columns)}")

    # Log NaN statistics
    nan_counts = features_df.isna().sum()
    if nan_counts.any():
        logger.debug(f"NaN counts per feature:\n{nan_counts[nan_counts > 0]}")

    if "usable_for_training" in features_df.columns:
        n_usable = features_df["usable_for_training"].sum()
        n_total = len(features_df)
        pct_usable = 100 * n_usable / n_total if n_total > 0 else 0
        logger.debug(
            f"Usable for training: {n_usable}/{n_total} ({pct_usable:.2f}%)"
        )

    return features_df
