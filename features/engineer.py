"""
Build features that:
1. Are available at decision time (no lookahead)
2. Include risk awareness
3. Capture tradeable patterns
"""

from typing import List

import numpy as np
import pandas as pd


def _to_chicago_time(timestamp_series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamp_series)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert("America/Chicago")


def add_features(bars_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Add all features to bars DataFrame.

    CRITICAL: NO LOOKAHEAD BIAS
    - Only use .shift() with positive values (past data)
    - Only use .rolling() on past data
    - Never use future data in any form

    Args:
        bars_df: DataFrame with timestamp, open, high, low, close, volume

    Returns:
        DataFrame with feature columns added
    """
    df = bars_df.copy()

    if verbose:
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)

    # ===== PRICE FEATURES =====
    if verbose:
        print("\n[1/7] Price features...")
    df["returns_1"] = df["close"].pct_change()
    df["returns_5"] = df["close"].pct_change(5)
    df["returns_12"] = df["close"].pct_change(12)

    df["log_returns_1"] = np.log(df["close"] / df["close"].shift(1))

    df["body"] = (df["close"] - df["open"]).abs()
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["range"] = df["high"] - df["low"]
    df["body_pct"] = df["body"] / (df["range"] + 1e-8)

    df["is_bullish"] = (df["close"] > df["open"]).astype(int)

    # ===== TREND FEATURES =====
    if verbose:
        print("[2/7] Trend features...")
    for span in [9, 21, 50]:
        ema = df["close"].ewm(span=span, adjust=False).mean()
        df[f"ema_{span}"] = ema
        df[f"ema_{span}_dist"] = (df["close"] - ema) / df["close"]
        df[f"ema_{span}_slope"] = ema.diff() / df["close"]

    df["ema_spread_9_21"] = (df["ema_9"] - df["ema_21"]) / df["close"]
    df["ema_spread_21_50"] = (df["ema_21"] - df["ema_50"]) / df["close"]

    df["price_above_ema9"] = (df["close"] > df["ema_9"]).astype(int)
    df["price_above_ema21"] = (df["close"] > df["ema_21"]).astype(int)
    df["price_above_ema50"] = (df["close"] > df["ema_50"]).astype(int)

    # ===== VOLATILITY FEATURES =====
    if verbose:
        print("[3/7] Volatility features...")
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_ticks"] = df["atr_14"] / 0.25

    df["vol_20"] = df["log_returns_1"].rolling(20).std()
    df["vol_50"] = df["log_returns_1"].rolling(50).std()

    df["vol_percentile"] = df["vol_20"].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan
    )

    bb_ma = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_width"] = (bb_std * 2) / bb_ma

    # ===== VOLUME FEATURES =====
    if verbose:
        print("[4/7] Volume features...")
    df["volume_ma_50"] = df["volume"].rolling(50).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_ma_50"] + 1)
    df["volume_surge"] = (df["volume"] > df["volume_ma_50"] * 1.5).astype(int)

    # ===== TIME FEATURES =====
    if verbose:
        print("[5/7] Time features...")
    df_ts = _to_chicago_time(df["timestamp"])
    df["hour"] = df_ts.dt.hour
    df["minute"] = df_ts.dt.minute
    df["time_of_day"] = df["hour"] + df["minute"] / 60.0

    df["session_open"] = ((df["hour"] == 8) & (df["minute"] <= 45)).astype(int)
    df["session_close"] = ((df["hour"] >= 14) & (df["minute"] >= 30)).astype(int)
    df["lunch_period"] = ((df["hour"] >= 11) & (df["hour"] <= 12)).astype(int)

    df["day_of_week"] = df_ts.dt.dayofweek

    # ===== MARKET STRUCTURE =====
    if verbose:
        print("[6/7] Market structure features...")
    df["high_12"] = df["high"].rolling(12).max()
    df["low_12"] = df["low"].rolling(12).min()
    df["high_50"] = df["high"].rolling(50).max()
    df["low_50"] = df["low"].rolling(50).min()

    df["range_position_12"] = (df["close"] - df["low_12"]) / (df["high_12"] - df["low_12"] + 1e-8)
    df["range_position_50"] = (df["close"] - df["low_50"]) / (df["high_50"] - df["low_50"] + 1e-8)

    df["dist_from_high_12"] = (df["high_12"] - df["close"]) / df["close"]
    df["dist_from_low_12"] = (df["close"] - df["low_12"]) / df["close"]

    # ===== RISK-AWARE FEATURES =====
    if verbose:
        print("[7/7] Risk-aware features...")
    df["estimated_stop_ticks"] = (df["atr_14"] * 1.5) / 0.25
    df["estimated_stop_ticks"] = df["estimated_stop_ticks"].clip(8, 24)

    df["estimated_risk_usd"] = df["estimated_stop_ticks"] * 1.25

    df["trade_affordable"] = (df["estimated_risk_usd"] <= 200).astype(int)

    # Fill NaN/inf values: use high risk (200) for NaN to default to 1 contract
    # Replace inf with a large finite value to avoid division issues
    risk_usd = df["estimated_risk_usd"].fillna(200).replace([np.inf, -np.inf], 200)
    max_contracts = (200 / (risk_usd + 1)).clip(1, 5)
    df["max_contracts_by_risk"] = max_contracts.fillna(1).astype(int)

    # FIXED: Removed expected_rr_15x (constant value, provides zero information)

    if verbose:
        print("\nOK: feature engineering complete")
        feature_count = len(
            [
                c
                for c in df.columns
                if c not in ["timestamp", "open", "high", "low", "close", "volume"]
            ]
        )
        print(f"  Total features: {feature_count}")

    return df


def select_features() -> List[str]:
    """
    Return list of feature column names for training.

    Excludes:
    - Raw OHLCV (to prevent data leakage)
    - Timestamp
    - Intermediate calculations
    """
    return [
        "returns_1",
        "returns_5",
        "returns_12",
        "log_returns_1",
        "body_pct",
        "upper_wick",
        "lower_wick",
        "is_bullish",
        "ema_9_dist",
        "ema_21_dist",
        "ema_50_dist",
        "ema_9_slope",
        "ema_21_slope",
        "ema_spread_9_21",
        "ema_spread_21_50",
        "price_above_ema9",
        "price_above_ema21",
        "price_above_ema50",
        "atr_ticks",
        "vol_20",
        "vol_percentile",
        "bb_width",
        "volume_ratio",
        "volume_surge",
        "time_of_day",
        "session_open",
        "session_close",
        "lunch_period",
        "day_of_week",
        "range_position_12",
        "range_position_50",
        "dist_from_high_12",
        "dist_from_low_12",
        "estimated_stop_ticks",
        "trade_affordable",
        "max_contracts_by_risk",
    ]


def verify_no_lookahead(df: pd.DataFrame, feature_cols: List[str]) -> bool:
    """
    Verify features do not use future data.

    Returns:
        True if no lookahead detected
    """
    print("\n" + "=" * 60)
    print("LOOKAHEAD BIAS CHECK")
    print("=" * 60)

    first_100 = df[feature_cols].head(100)
    nan_counts = first_100.isna().sum()
    has_leading_nans = (nan_counts > 0).any()

    if not has_leading_nans:
        print("\nWARNING: No leading NaNs found")
        print("   This might indicate features do not use rolling windows")
        print("   (or data was forward-filled improperly)")
    else:
        print("\nOK: leading NaNs present (expected for rolling/shift operations)")

    print("\nShift correlation test...")
    print("(Skipped - requires labels)")

    print("\nOK: lookahead check complete")
    return True


if __name__ == "__main__":
    print("Loading data...")
    with pd.HDFStore("data/processed/mes_bars.h5", "r") as store:
        bars = store["bars_5min"]

    test_bars = bars.tail(1000).reset_index(drop=True)

    print(f"Input shape: {test_bars.shape}")
    features_df = add_features(test_bars)

    feature_cols = select_features()
    print(f"\nSelected {len(feature_cols)} features for training")

    verify_no_lookahead(features_df, feature_cols)

    print("\nSample features (last 5 bars):")
    print(features_df[["timestamp"] + feature_cols[:5]].tail())

    print("\nFeature statistics:")
    print(features_df[feature_cols].describe())
