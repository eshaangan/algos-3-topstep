"""
Feature building module for V3 pipeline.

Computes all features causally (no lookahead) with deterministic ordering.
Handles NaN/synthetic bars explicitly per config policy.

Phase 1: Feature stationarity normalization (rolling z-score, price-distance ratios, ROC)
Phase 2: Momentum features (RSI, MACD, RSI divergence, VWAP momentum)
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from .registry import get_feature_registry, filter_registry_for_bar_size, FeatureSpec
from .multi_resolution import build_multi_resolution_features
from .fractional_diff import fracdiff_series
from .hmm_regime import build_hmm_regime_features
from .structural_breaks import build_structural_break_features
from .crt_tbs import build_crt_tbs_features

logger = logging.getLogger(__name__)


def _weighted_moving_average(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    weight_sum = weights.sum()
    return series.rolling(period).apply(
        lambda values: float(np.dot(values, weights) / weight_sum),
        raw=True,
    )


def _hhmm_to_minutes(value: str) -> int:
    hour_str, minute_str = str(value).split(":")
    return int(hour_str) * 60 + int(minute_str)


def _apply_normalization(features: dict, df: pd.DataFrame, config: dict) -> dict:
    """Apply stationarity normalization transforms to raw features.

    Phase 1: Converts non-stationary features into stationary versions:
    1. Rolling z-score for level-dependent features
    2. Distance-from-price ratios for price-level indicators
    3. Rate-of-change for volatility features
    4. ATR-normalization for spread features
    """
    norm_cfg = config.get("normalization", {})
    if not norm_cfg.get("enabled", False):
        return features

    eps = config.get("computation", {}).get("eps", 1e-8)
    # Ensure eps is float
    if isinstance(eps, str):
        try:
            eps = float(eps)
        except ValueError:
            eps = 1e-8
    zscore_lookback = norm_cfg.get("zscore_lookback", 50)

    # 1. Rolling z-score normalization
    zscore_features = norm_cfg.get("zscore_features", [])
    for feat_name in zscore_features:
        if feat_name not in features:
            continue
        raw = features[feat_name]
        roll_mean = raw.rolling(zscore_lookback, min_periods=1).mean()
        roll_std = raw.rolling(zscore_lookback, min_periods=1).std()
        features[feat_name] = (raw - roll_mean) / (roll_std + eps)

    # 2. Price-distance normalization: (close - indicator) / ATR
    price_dist_features = norm_cfg.get("price_distance_features", [])
    atr_period = config.get("volatility", {}).get("atr_period", 14)
    atr_key = f"atr_{atr_period}"
    if price_dist_features and atr_key in features:
        atr = features[atr_key]
        for feat_name in price_dist_features:
            if feat_name not in features:
                continue
            features[feat_name] = (df["close"] - features[feat_name]) / (atr + eps)

    # 3. Rate-of-change normalization: current / lagged - 1
    roc_features = norm_cfg.get("rate_of_change_features", {})
    for feat_name, shift_bars in roc_features.items():
        if feat_name not in features:
            continue
        raw = features[feat_name]
        lagged = raw.shift(int(shift_bars))
        features[feat_name] = (raw / (lagged + eps)) - 1.0

    # 4. Spread normalization by ATR
    spread_features = norm_cfg.get("spread_normalize_by_atr", [])
    if spread_features and atr_key in features:
        atr = features[atr_key]
        for feat_name in spread_features:
            if feat_name not in features:
                continue
            features[feat_name] = features[feat_name] / (atr + eps)

    return features


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

    # Multi-horizon returns (match 12-24 bar label horizons)
    if config.get("returns", {}).get("enable_multi_horizon", True):
        multi_horizon = config.get("returns", {}).get("multi_horizon_bars", [6, 12, 24])
        for k in multi_horizon:
            features[f"log_return_{k}"] = log_close.diff(k)

    # -------------------------------------------------------------------------
    # 2. VOLATILITY
    # -------------------------------------------------------------------------
    logger.debug("Computing volatility features")

    # True Range
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    features["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR: EMA of true_range
    atr_period = config.get("volatility", {}).get("atr_period", 14)
    features[f"atr_{atr_period}"] = (
        features["true_range"].ewm(span=atr_period, adjust=False, min_periods=1).mean()
    )

    # Volatility regime indicators
    if config.get("volatility", {}).get("enable_regime_features", True):
        logger.debug("Computing volatility regime features")

        returns_1 = features["log_return_1"]
        features["vol_20"] = returns_1.rolling(20).std()

        vol_regime_lookback = config.get("volatility", {}).get("vol_regime_lookback", 50)
        features["vol_regime"] = features["vol_20"] / (
            features["vol_20"].rolling(vol_regime_lookback).median() + eps
        )

        hl_log_ratio = np.log(df["high"] / (df["low"] + eps))
        features["parkinson_vol"] = np.sqrt(
            (1 / (4 * np.log(2))) * (hl_log_ratio ** 2).rolling(20).mean()
        )

        features["vol_forecast"] = returns_1.ewm(alpha=0.06, min_periods=1).std()

    # -------------------------------------------------------------------------
    # 3. TREND
    # -------------------------------------------------------------------------
    logger.debug("Computing trend features")

    ema_fast_period = config.get("trend", {}).get("ema_fast_period", 13)
    ema_slow_period = config.get("trend", {}).get("ema_slow_period", 34)

    ema_fast = df["close"].ewm(span=ema_fast_period, adjust=False, min_periods=1).mean()
    ema_slow = df["close"].ewm(span=ema_slow_period, adjust=False, min_periods=1).mean()

    features[f"ema_{ema_fast_period}"] = ema_fast
    features[f"ema_{ema_slow_period}"] = ema_slow

    features["ema_spread"] = ema_fast - ema_slow
    features["ema_ratio"] = ema_fast / (ema_slow + eps)

    # Advanced trend and mean reversion features
    if config.get("trend", {}).get("enable_advanced_features", True):
        logger.debug("Computing advanced trend features")

        sma_long_period = config.get("trend", {}).get("sma_long_period", 30)
        sma_20 = df["close"].rolling(20).mean()
        sma_long = df["close"].rolling(sma_long_period).mean()

        features["sma_20"] = sma_20
        features[f"sma_{sma_long_period}"] = sma_long

        features["trend_strength"] = (df["close"] - sma_long) / (sma_long + eps)

        def autocorr_lag5(x):
            if len(x) < 6:
                return np.nan
            return x.autocorr(lag=5) if len(x.dropna()) > 6 else np.nan

        features["autocorr_5"] = df["close"].rolling(20).apply(autocorr_lag5, raw=False)

        bb_std = df["close"].rolling(20).std()
        bb_upper = sma_20 + 2 * bb_std
        bb_lower = sma_20 - 2 * bb_std
        features["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + eps)

        anchor_cfg = config.get("trend", {}).get("anchor_context", {})
        if anchor_cfg.get("enabled", False):
            fast_wma_period = int(anchor_cfg.get("fast_wma_period", 13))
            mid_wma_period = int(anchor_cfg.get("mid_wma_period", 48))
            anchor_wma_period = int(anchor_cfg.get("anchor_wma_period", 200))

            wma_fast = _weighted_moving_average(df["close"], fast_wma_period)
            wma_mid = _weighted_moving_average(df["close"], mid_wma_period)
            wma_anchor = _weighted_moving_average(df["close"], anchor_wma_period)

            features[f"close_vs_wma_{anchor_wma_period}_atr"] = (
                (df["close"] - wma_anchor) / (features[f"atr_{atr_period}"] + eps)
            )
            bullish_ladder = (
                (df["close"] > wma_fast).astype(int)
                + (wma_fast > wma_mid).astype(int)
                + (wma_mid > wma_anchor).astype(int)
            )
            bearish_ladder = (
                (df["close"] < wma_fast).astype(int)
                + (wma_fast < wma_mid).astype(int)
                + (wma_mid < wma_anchor).astype(int)
            )
            features["wma_ladder_score"] = bullish_ladder - bearish_ladder

    # -------------------------------------------------------------------------
    # 4. MOMENTUM (MACD, RSI, Stochastic, RSI Divergence, VWAP Momentum)
    # -------------------------------------------------------------------------
    if config.get("momentum", {}).get("enabled", True):
        logger.debug("Computing momentum features")

        # RSI
        rsi_period = config.get("momentum", {}).get("rsi_period", 14)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / (loss + eps)
        rsi = 100 - (100 / (1 + rs))
        features[f"rsi_{rsi_period}"] = rsi

        # MACD
        macd_config = config.get("momentum", {}).get("macd", {})
        if macd_config.get("enabled", True):
            fast = macd_config.get("fast_period", 12)
            slow = macd_config.get("slow_period", 26)
            signal = macd_config.get("signal_period", 9)

            ema_fast_macd = df["close"].ewm(span=fast, adjust=False, min_periods=1).mean()
            ema_slow_macd = df["close"].ewm(span=slow, adjust=False, min_periods=1).mean()

            features["macd"] = ema_fast_macd - ema_slow_macd
            features["macd_signal"] = features["macd"].ewm(span=signal, adjust=False, min_periods=1).mean()
            features["macd_hist"] = features["macd"] - features["macd_signal"]

        # Stochastic Oscillator
        stoch_config = config.get("momentum", {}).get("stochastic", {})
        if stoch_config.get("enabled", True):
            k_period = stoch_config.get("k_period", 14)
            d_period = stoch_config.get("d_period", 3)

            low_min = df["low"].rolling(window=k_period).min()
            high_max = df["high"].rolling(window=k_period).max()

            features[f"stoch_k_{k_period}"] = 100 * ((df["close"] - low_min) / (high_max - low_min + eps))
            features[f"stoch_d_{d_period}"] = features[f"stoch_k_{k_period}"].rolling(window=d_period).mean()

        # RSI Divergence (Phase 2): detects momentum exhaustion
        rsi_div_cfg = config.get("momentum", {}).get("rsi_divergence", {})
        if rsi_div_cfg.get("enabled", False):
            div_lookback = rsi_div_cfg.get("lookback", 5)
            price_change = df["close"].diff(div_lookback)
            rsi_change = rsi.diff(div_lookback)
            # Divergence = price going one way, RSI going the other
            features["rsi_divergence"] = (
                (np.sign(price_change) != np.sign(rsi_change)) & price_change.notna()
            ).astype(int)

        # VWAP Momentum (Phase 2): rate of change of VWAP deviation
        # Note: computed after microstructure section below if price_vs_vwap not yet available
        vwap_mom_cfg = config.get("momentum", {}).get("vwap_momentum", {})
        _vwap_mom_enabled = vwap_mom_cfg.get("enabled", False)

    # -------------------------------------------------------------------------
    # 5. VOLUME & MICROSTRUCTURE (Order flow proxies)
    # -------------------------------------------------------------------------
    if config.get("microstructure", {}).get("enabled", True) and "volume" in df.columns:
        logger.debug("Computing microstructure features")

        features["volume_imbalance"] = (df["close"] - df["open"]) / (
            df["high"] - df["low"] + eps
        )

        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol_price = (df["volume"] * typical_price).rolling(50).sum()
        cum_vol = df["volume"].rolling(50).sum()
        vwap = cum_vol_price / (cum_vol + eps)
        features["price_vs_vwap"] = (df["close"] - vwap) / (vwap + eps)

        avg_volume = df["volume"].rolling(20).mean()
        features["relative_volume"] = df["volume"] / (avg_volume + eps)

        if "vol_20" in features:
            features["large_move"] = (
                features["log_return_1"].abs() > 2 * features["vol_20"]
            ).astype(int)

        vwap_ctx_cfg = config.get("microstructure", {}).get("vwap_context", {})
        if vwap_ctx_cfg.get("enabled", False):
            zscore_lookback = int(vwap_ctx_cfg.get("zscore_lookback", 20))
            vwap_std = features["price_vs_vwap"].rolling(zscore_lookback).std()
            features["vwap_zscore"] = features["price_vs_vwap"] / (vwap_std + eps)
            signed_excess = pd.Series(
                np.maximum(features["vwap_zscore"].abs() - 1.0, 0.0),
                index=df.index,
            )
            features["vwap_band_distance"] = signed_excess * np.sign(features["vwap_zscore"])

    # Compute VWAP momentum now that price_vs_vwap is available
    vwap_mom_cfg = config.get("momentum", {}).get("vwap_momentum", {})
    if vwap_mom_cfg.get("enabled", False) and "price_vs_vwap" in features:
        vwap_lookback = vwap_mom_cfg.get("lookback", 5)
        features["vwap_momentum"] = features["price_vs_vwap"] - features["price_vs_vwap"].shift(vwap_lookback)

    # -------------------------------------------------------------------------
    # 6. STRUCTURE (Candle features)
    # -------------------------------------------------------------------------
    if config.get("structure", {}).get("enabled", True):
        logger.debug("Computing candle structure features")

        features["candle_body"] = df["close"] - df["open"]
        features["candle_range"] = df["high"] - df["low"]

        features["body_pct"] = features["candle_body"] / (
            features["candle_range"] + eps
        )

        features["upper_wick"] = df["high"] - pd.concat(
            [df["open"], df["close"]], axis=1
        ).max(axis=1)
        features["lower_wick"] = pd.concat(
            [df["open"], df["close"]], axis=1
        ).min(axis=1) - df["low"]

    crt_tbs_cfg = config.get("crt_tbs", {}) or {}
    if crt_tbs_cfg.get("enabled", False):
        logger.debug("Computing CRT/TBS features")
        atr_period = config.get("volatility", {}).get("atr_period", 14)
        crt_tbs_df = build_crt_tbs_features(
            df,
            atr=features.get(f"atr_{atr_period}"),
            range_lookback=int(crt_tbs_cfg.get("range_lookback", 20)),
            eps=eps,
        )
        for col in crt_tbs_df.columns:
            features[col] = crt_tbs_df[col]

    # -------------------------------------------------------------------------
    # 7. TIME (Cyclical encodings)
    # -------------------------------------------------------------------------
    if config.get("time", {}).get("enabled", True):
        logger.debug("Computing time features")

        minutes_per_session = config.get("time", {}).get("minutes_per_session", 1440)

        if "minute_of_day" in df.columns:
            minute_of_day = df["minute_of_day"]
        else:
            minute_of_day = df.index.hour * 60 + df.index.minute

        angle = 2 * np.pi * minute_of_day / minutes_per_session
        features["minute_of_day_sin"] = np.sin(angle)
        features["minute_of_day_cos"] = np.cos(angle)

        if "day_of_week" in df.columns:
            features["day_of_week"] = df["day_of_week"]
        else:
            features["day_of_week"] = df.index.dayofweek

        session_cfg = config.get("time", {}).get("session_windows", {})
        if session_cfg.get("enabled", False):
            tz_name = session_cfg.get("timezone", "America/Chicago")
            local_index = df.index.tz_localize("UTC") if df.index.tz is None else df.index
            local_minutes = (
                local_index.tz_convert(tz_name).hour * 60
                + local_index.tz_convert(tz_name).minute
            )

            open_minutes = _hhmm_to_minutes(session_cfg.get("rth_open_time", "09:30"))
            close_minutes = _hhmm_to_minutes(session_cfg.get("rth_close_time", "15:55"))
            opening_window_minutes = int(session_cfg.get("opening_window_minutes", 45))
            midday_start = _hhmm_to_minutes(session_cfg.get("midday_start_time", "11:30"))
            midday_end = _hhmm_to_minutes(session_cfg.get("midday_end_time", "13:30"))
            closing_window_minutes = int(session_cfg.get("closing_window_minutes", 60))

            features["is_opening_window"] = (
                (local_minutes >= open_minutes)
                & (local_minutes < open_minutes + opening_window_minutes)
            ).astype(int)
            features["is_midday_window"] = (
                (local_minutes >= midday_start)
                & (local_minutes < midday_end)
            ).astype(int)
            features["is_closing_window"] = (
                (local_minutes >= close_minutes - closing_window_minutes)
                & (local_minutes <= close_minutes)
            ).astype(int)
            for col in ["is_opening_window", "is_midday_window", "is_closing_window"]:
                features[col] = pd.Series(features[col], index=df.index)

    # -------------------------------------------------------------------------
    # Optional structural break features
    # -------------------------------------------------------------------------
    if config.get("structural_breaks", {}).get("enabled", False):
        sb = build_structural_break_features(
            close=df["close"],
            returns=features.get("log_return_1"),
        )
        for col in sb.columns:
            features[col] = sb[col]

    # -------------------------------------------------------------------------
    # Optional HMM regime features
    # -------------------------------------------------------------------------
    hmm_cfg = config.get("hmm_regime", {})
    if hmm_cfg.get("enabled", False):
        hmm_n_states = int(hmm_cfg.get("n_states", 2))
        hmm_df = build_hmm_regime_features(
            close=df["close"],
            n_states=hmm_n_states,
            min_train_samples=int(hmm_cfg.get("min_train_samples", 252)),
            refit_every=int(hmm_cfg.get("refit_every", 21)),
            rolling_window_size=int(hmm_cfg.get("rolling_window_size", 252)),
            covariance_type=str(hmm_cfg.get("covariance_type", "full")),
            n_iter=int(hmm_cfg.get("n_iter", 100)),
            tol=float(hmm_cfg.get("tol", 1e-4)),
        )
        neutral_prob = 1.0 / float(max(hmm_n_states, 1))
        for col in hmm_df.columns:
            s = hmm_df[col].ffill()
            if col == "hmm_state":
                features[col] = s.fillna(-1)
            else:
                features[col] = s.fillna(neutral_prob)

    # -------------------------------------------------------------------------
    # PHASE 1: NORMALIZATION (apply after all raw features computed)
    # Converts non-stationary features to stationary versions
    # -------------------------------------------------------------------------
    features = _apply_normalization(features, df, config)

    # -------------------------------------------------------------------------
    # Optional fractional differentiation for selected feature columns
    # -------------------------------------------------------------------------
    frac_cfg = config.get("fractional_diff", {})
    if frac_cfg.get("enabled", False):
        d_val = float(frac_cfg.get("d", 0.4))
        threshold = float(frac_cfg.get("threshold", 1e-5))
        apply_to = frac_cfg.get("apply_to", [])
        for col in apply_to:
            if col in features:
                features[col] = fracdiff_series(features[col], d=d_val, threshold=threshold)
            else:
                logger.debug("fractional_diff skipped missing feature column: %s", col)

    # -------------------------------------------------------------------------
    # 8. META (Flags and masks)
    # -------------------------------------------------------------------------
    if config.get("output", {}).get("include_synthetic_flag", True):
        if "is_synthetic" in df.columns:
            features["is_synthetic"] = df["is_synthetic"]
        else:
            features["is_synthetic"] = pd.Series(False, index=df.index)

    # Usable for training mask
    nan_policy = config.get("computation", {}).get("nan_policy", "keep_with_mask")
    if nan_policy == "keep_with_mask":
        logger.debug("Computing usable_for_training mask")

        feature_cols_to_check = []
        for spec in registry:
            if spec.name in ["is_synthetic", "usable_for_training"]:
                continue
            feature_cols_to_check.append(spec.name)

        usable_mask = pd.Series(True, index=df.index)
        for col in feature_cols_to_check:
            if col in features:
                usable_mask &= ~features[col].isna()

        features["usable_for_training"] = usable_mask

    # -------------------------------------------------------------------------
    # Assemble DataFrame in registry order
    # -------------------------------------------------------------------------
    logger.debug("Assembling feature DataFrame in registry order")

    ordered_columns = [spec.name for spec in registry]

    features_df = pd.DataFrame(index=df.index)
    for col in ordered_columns:
        if col in features:
            features_df[col] = features[col]
        else:
            logger.warning(f"Feature {col} registered but not computed; filling with NaN")
            features_df[col] = np.nan

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

    # -------------------------------------------------------------------------
    # 9. Optional multi-resolution features (5m -> higher timeframes)
    # -------------------------------------------------------------------------
    if bar_size == "5m" and config.get("multi_resolution", {}).get("enabled", False):
        mr_df = build_multi_resolution_features(
            bars_df=df,
            base_index=features_df.index,
            config=config,
            feature_builder=build_features,
        )
        if not mr_df.empty:
            features_df = features_df.join(mr_df, how="left")
            # Maintain deterministic ordering for appended columns.
            base_cols = [spec.name for spec in registry]
            mr_cols = sorted([c for c in features_df.columns if c not in base_cols])
            features_df = features_df[base_cols + mr_cols]

            # Update mask to include multi-resolution feature availability.
            if "usable_for_training" in features_df.columns:
                mr_nonmeta = [
                    c for c in mr_cols if c not in {"is_synthetic", "usable_for_training"}
                ]
                if mr_nonmeta:
                    base_mask = features_df["usable_for_training"].astype(bool)
                    extra_mask = ~features_df[mr_nonmeta].isna().any(axis=1)
                    features_df["usable_for_training"] = base_mask & extra_mask

    return features_df
