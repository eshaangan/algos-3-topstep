"""
Fixed-horizon market-centric labels for ML training.

This module replaces TP/SL simulation labels with pure market movement labels
to reduce overfitting and improve generalization.

Key differences from labels.py:
- No execution simulation (no TP/SL/max-hold logic in labels)
- Fixed-horizon future returns (H bars ahead)
- Threshold-based binary classification (X ticks movement)
- Separates label definition from execution strategy

This approach prevents the model from memorizing strategy-specific patterns
and forces it to learn generalizable market dynamics.
"""

from __future__ import annotations

from datetime import time
from typing import Literal

import numpy as np
import pandas as pd


def make_fixed_horizon_labels(
    df: pd.DataFrame,
    horizon_bars: int,
    threshold_ticks: int,
    tick_size: float,
) -> pd.DataFrame:
    """
    Create market-centric fixed-horizon labels.

    Args:
        df: Bars DataFrame with a 'close' column.
        horizon_bars: Number of bars to look ahead.
        threshold_ticks: Movement threshold in ticks.
        tick_size: Tick size for the instrument.

    Returns:
        DataFrame with idx, ret_ticks, y_long, y_short.
    """
    if "close" not in df.columns:
        raise ValueError("df must contain 'close' column")
    if horizon_bars < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
    if threshold_ticks < 1:
        raise ValueError(f"threshold_ticks must be >= 1, got {threshold_ticks}")
    if tick_size <= 0:
        raise ValueError(f"tick_size must be > 0, got {tick_size}")

    close = df["close"].astype(float)
    future_close = close.shift(-horizon_bars)
    ret_ticks = (future_close - close) / tick_size

    y_long = (ret_ticks >= float(threshold_ticks)).astype(int)
    y_short = (ret_ticks <= -float(threshold_ticks)).astype(int)

    labels = pd.DataFrame(
        {
            "idx": np.arange(len(df)),
            "ret_ticks": ret_ticks,
            "y_long": y_long,
            "y_short": y_short,
        }
    )

    labels = labels.dropna(subset=["ret_ticks"]).copy()
    return labels


def _to_chicago(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert timestamp to Chicago timezone."""
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/Chicago")


def create_fixed_horizon_labels(
    bars_df: pd.DataFrame,
    *,
    horizon_bars: int = 12,
    threshold_ticks: int = 10,
    tick_size: float = 0.25,
    session_start: time = time(8, 30),
    session_end: time = time(14, 55),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Create fixed-horizon market-centric labels.

    For each bar at time t, compute the future return H bars ahead and
    create binary labels based on whether the return exceeds ±X ticks.

    CRITICAL: Labels use ONLY future market data (no execution simulation).
    Features must be causal (use only past data).

    Args:
        bars_df: DataFrame with OHLCV bars (must have 'close', 'timestamp' columns)
        horizon_bars: Number of bars to look ahead (H). Examples:
            - 6 bars @ 5min = 30 minutes
            - 12 bars @ 5min = 60 minutes
            - 24 bars @ 5min = 120 minutes
        threshold_ticks: Minimum movement in ticks to classify as directional (X)
            - 10 ticks @ 0.25/tick = 2.5 points on MES
            - 12 ticks = 3.0 points
            - 8 ticks = 2.0 points
        tick_size: Size of one tick (0.25 for MES)
        session_start: Only create labels during regular trading hours
        session_end: End of trading session
        verbose: Print label statistics

    Returns:
        DataFrame with columns:
            - idx: Bar index (matches bars_df index)
            - timestamp: Bar timestamp
            - future_return_ticks: Return H bars ahead in ticks
            - label_long: 1 if future return >= +threshold_ticks, else 0
            - label_short: 1 if future return <= -threshold_ticks, else 0

    Label Logic:
        future_close = close[t + H]
        future_return_ticks = (future_close - close[t]) / tick_size

        Long label (bullish):
            y_long = 1 if future_return_ticks >= +threshold_ticks else 0

        Short label (bearish):
            y_short = 1 if future_return_ticks <= -threshold_ticks else 0

        Both can be 0 (sideways), but never both 1 (exclusive classes).

    Example:
        bars_df has 10,000 bars at 5-minute resolution
        horizon_bars = 12 (60 minutes ahead)
        threshold_ticks = 10 (2.5 MES points)

        For bar at index 1000:
            current_close = bars_df.iloc[1000]['close']
            future_close = bars_df.iloc[1012]['close']  # 12 bars ahead
            future_return_ticks = (future_close - current_close) / 0.25

            If future_return_ticks >= +10: label_long = 1, label_short = 0
            If future_return_ticks <= -10: label_long = 0, label_short = 1
            If -10 < future_return_ticks < +10: label_long = 0, label_short = 0

    Causality:
        - Features at time t use data from [t-lookback, t] (past only)
        - Labels at time t use data at [t+H] (future only)
        - No overlap: features never see label information

    Why This Approach:
        1. Reduces overfitting: model can't memorize execution strategy
        2. Portable: same labels work for different execution strategies
        3. Simpler: no simulation complexity, just pure price movement
        4. Generalizable: learns market dynamics, not strategy artifacts
    """
    if verbose:
        print("\n" + "=" * 60)
        print("FIXED-HORIZON MARKET LABELS V2")
        print("=" * 60)
        print(f"\nHorizon: {horizon_bars} bars")
        print(f"Threshold: {threshold_ticks} ticks ({threshold_ticks * tick_size:.2f} points)")
        print(f"Session: {session_start} - {session_end} CT")

    # Validate inputs
    required_cols = ["close", "timestamp"]
    missing = [col for col in required_cols if col not in bars_df.columns]
    if missing:
        raise ValueError(f"bars_df missing required columns: {missing}")

    if horizon_bars < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
    if threshold_ticks < 1:
        raise ValueError(f"threshold_ticks must be >= 1, got {threshold_ticks}")
    if tick_size <= 0:
        raise ValueError(f"tick_size must be > 0, got {tick_size}")

    # Compute future returns
    # CRITICAL: .shift(-H) moves data H rows UP (gets future data)
    # Example: close[100].shift(-12) = close[112] (12 bars ahead)
    close = bars_df["close"].values
    future_close = pd.Series(close).shift(-horizon_bars).values

    # Compute return in ticks
    future_return_ticks = (future_close - close) / tick_size

    # Create binary labels based on threshold
    label_long = (future_return_ticks >= threshold_ticks).astype(int)
    label_short = (future_return_ticks <= -threshold_ticks).astype(int)

    # Build label DataFrame
    labels_df = pd.DataFrame(
        {
            "idx": np.arange(len(bars_df)),
            "timestamp": bars_df["timestamp"].values,
            "future_return_ticks": future_return_ticks,
            "label_long": label_long,
            "label_short": label_short,
        }
    )

    # Filter to regular trading hours
    labels_df["bar_time"] = labels_df["timestamp"].apply(_to_chicago)
    labels_df["is_rth"] = labels_df["bar_time"].apply(
        lambda t: session_start <= t.time() < session_end
    )

    # Remove bars outside RTH
    n_before = len(labels_df)
    labels_df = labels_df[labels_df["is_rth"]].copy()
    n_after = len(labels_df)

    if verbose:
        print(f"\nFiltered to RTH: {n_before:,} → {n_after:,} bars ({n_after/n_before*100:.1f}%)")

    # Remove rows with NaN (last H bars will have NaN future_close)
    n_before = len(labels_df)
    labels_df = labels_df.dropna(subset=["future_return_ticks"]).copy()
    n_after = len(labels_df)

    if verbose:
        print(f"Removed NaN (last {horizon_bars} bars): {n_before:,} → {n_after:,} bars")

    # Drop helper columns
    labels_df = labels_df.drop(columns=["bar_time", "is_rth"])

    if verbose:
        print("\n" + "=" * 60)
        print("LABEL DISTRIBUTION")
        print("=" * 60)

        # Long labels
        long_positive = (labels_df["label_long"] == 1).sum()
        long_negative = (labels_df["label_long"] == 0).sum()
        long_positive_pct = long_positive / len(labels_df) * 100
        print(f"\nLONG (bullish >= +{threshold_ticks} ticks):")
        print(f"  Positive: {long_positive:6,} ({long_positive_pct:5.1f}%)")
        print(f"  Negative: {long_negative:6,} ({100-long_positive_pct:5.1f}%)")

        # Short labels
        short_positive = (labels_df["label_short"] == 1).sum()
        short_negative = (labels_df["label_short"] == 0).sum()
        short_positive_pct = short_positive / len(labels_df) * 100
        print(f"\nSHORT (bearish <= -{threshold_ticks} ticks):")
        print(f"  Positive: {short_positive:6,} ({short_positive_pct:5.1f}%)")
        print(f"  Negative: {short_negative:6,} ({100-short_positive_pct:5.1f}%)")

        # Sideways (both 0)
        sideways = ((labels_df["label_long"] == 0) & (labels_df["label_short"] == 0)).sum()
        sideways_pct = sideways / len(labels_df) * 100
        print(f"\nSIDEWAYS (|return| < {threshold_ticks} ticks):")
        print(f"  Count: {sideways:6,} ({sideways_pct:5.1f}%)")

        # Sanity checks
        print("\n" + "=" * 60)
        print("SANITY CHECKS")
        print("=" * 60)

        # Check for both labels = 1 (should never happen)
        both_positive = ((labels_df["label_long"] == 1) & (labels_df["label_short"] == 1)).sum()
        if both_positive > 0:
            print(f"\n⚠️  WARNING: {both_positive} bars have both long=1 AND short=1 (impossible!)")

        # Check label balance
        if long_positive_pct < 10 or long_positive_pct > 40:
            print(f"\n⚠️  WARNING: Long positive rate {long_positive_pct:.1f}% outside 10-40% range")
            print("   Consider adjusting threshold_ticks or horizon_bars")

        if short_positive_pct < 10 or short_positive_pct > 40:
            print(f"\n⚠️  WARNING: Short positive rate {short_positive_pct:.1f}% outside 10-40% range")
            print("   Consider adjusting threshold_ticks or horizon_bars")

        # Check sideways rate
        if sideways_pct > 70:
            print(f"\n⚠️  WARNING: {sideways_pct:.1f}% sideways bars (threshold too high)")
            print(f"   Consider reducing threshold_ticks from {threshold_ticks}")
        elif sideways_pct < 30:
            print(f"\n⚠️  WARNING: Only {sideways_pct:.1f}% sideways bars (threshold too low)")
            print(f"   Consider increasing threshold_ticks from {threshold_ticks}")

        # Return statistics
        print(f"\nFuture return statistics (ticks):")
        print(f"  Mean: {labels_df['future_return_ticks'].mean():.2f}")
        print(f"  Std:  {labels_df['future_return_ticks'].std():.2f}")
        print(f"  Min:  {labels_df['future_return_ticks'].min():.2f}")
        print(f"  25%:  {labels_df['future_return_ticks'].quantile(0.25):.2f}")
        print(f"  50%:  {labels_df['future_return_ticks'].quantile(0.50):.2f}")
        print(f"  75%:  {labels_df['future_return_ticks'].quantile(0.75):.2f}")
        print(f"  Max:  {labels_df['future_return_ticks'].max():.2f}")

        print("\n" + "=" * 60 + "\n")

    return labels_df


def get_label_splits(
    labels_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    *,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    horizon_bars: int = 12,
    lookback_bars: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Create walk-forward train/val/test splits with proper purge/embargo.

    Splits must account for:
    1. Feature lookback (features need L bars of history)
    2. Label horizon (labels look H bars ahead)
    3. Purge gap between splits (prevent label leakage)

    Args:
        labels_df: Label DataFrame from create_fixed_horizon_labels()
        bars_df: Original bars DataFrame
        train_fraction: Fraction of data for training (default 0.6)
        val_fraction: Fraction for validation (default 0.2)
        test_fraction: Fraction for test (default 0.2)
        horizon_bars: Label horizon H (for embargo calculation)
        lookback_bars: Feature lookback L (for purge calculation)
        verbose: Print split details

    Returns:
        dict with:
            "train": (start_idx, end_idx)
            "val": (start_idx, end_idx)
            "test": (start_idx, end_idx)
            "embargo_bars": purge/embargo gap size
            "train_timestamps": {"start": ..., "end": ...}
            "val_timestamps": {"start": ..., "end": ...}
            "test_timestamps": {"start": ..., "end": ...}

    Embargo Logic:
        Between train and val, we need a gap to prevent label leakage:
        - Features at train_end use data from [train_end - lookback, train_end]
        - Labels at train_end use data at train_end + horizon
        - Val must start AFTER train_end + horizon + safety_margin

        embargo_bars = horizon_bars + lookback_bars + 1

        Example with horizon=12, lookback=100:
            embargo = 12 + 100 + 1 = 113 bars

            If train ends at bar 10000:
                - Last train feature uses bars [9900, 10000]
                - Last train label uses bar 10012
                - Val must start at bar 10113 or later

    Walk-Forward Timeline:
        |--- Train ---|--Embargo--|--- Val ---|--Embargo--|--- Test ---|
        0            T1           T2          V1          V2           N

        T1 = train_end (based on train_fraction)
        T2 = T1 + embargo_bars (start of val)
        V1 = val_end (based on val_fraction)
        V2 = V1 + embargo_bars (start of test)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("WALK-FORWARD SPLITS WITH EMBARGO")
        print("=" * 60)

    # Validate fractions
    if not abs(train_fraction + val_fraction + test_fraction - 1.0) < 0.01:
        raise ValueError(f"Fractions must sum to 1.0, got {train_fraction + val_fraction + test_fraction:.2f}")

    # Calculate embargo gap
    embargo_bars = horizon_bars + lookback_bars + 1

    if verbose:
        print(f"\nEmbargo calculation:")
        print(f"  Horizon bars: {horizon_bars}")
        print(f"  Lookback bars: {lookback_bars}")
        print(f"  Safety margin: 1")
        print(f"  Total embargo: {embargo_bars} bars")

    # Get total bars (use bars_df length, not labels_df)
    n_bars = len(bars_df)

    # Calculate split points
    # Train: [lookback, train_end]
    train_start = lookback_bars
    train_end = int(train_start + (n_bars - lookback_bars) * train_fraction)

    # Val: [train_end + embargo, val_end]
    val_start = train_end + embargo_bars
    val_end = int(val_start + (n_bars - lookback_bars) * val_fraction)

    # Test: [val_end + embargo, test_end]
    test_start = val_end + embargo_bars
    test_end = n_bars

    if verbose:
        print(f"\nSplit indices (bars_df indices):")
        print(f"  Train: [{train_start:6,}, {train_end:6,}] = {train_end - train_start:6,} bars")
        print(f"  Val:   [{val_start:6,}, {val_end:6,}] = {val_end - val_start:6,} bars")
        print(f"  Test:  [{test_start:6,}, {test_end:6,}] = {test_end - test_start:6,} bars")

        # Get timestamps
        train_start_ts = bars_df.iloc[train_start]["timestamp"]
        train_end_ts = bars_df.iloc[train_end - 1]["timestamp"]
        val_start_ts = bars_df.iloc[val_start]["timestamp"]
        val_end_ts = bars_df.iloc[val_end - 1]["timestamp"]
        test_start_ts = bars_df.iloc[test_start]["timestamp"]
        test_end_ts = bars_df.iloc[test_end - 1]["timestamp"]

        print(f"\nTimestamps:")
        print(f"  Train: {train_start_ts} → {train_end_ts}")
        print(f"  Val:   {val_start_ts} → {val_end_ts}")
        print(f"  Test:  {test_start_ts} → {test_end_ts}")

        # Sanity checks
        print("\n" + "=" * 60)
        print("SANITY CHECKS")
        print("=" * 60)

        # Check for gaps
        train_val_gap = val_start - train_end
        val_test_gap = test_start - val_end

        if train_val_gap != embargo_bars:
            print(f"\n⚠️  WARNING: Train-Val gap {train_val_gap} != embargo {embargo_bars}")
        else:
            print(f"\n✓ Train-Val embargo: {train_val_gap} bars")

        if val_test_gap != embargo_bars:
            print(f"⚠️  WARNING: Val-Test gap {val_test_gap} != embargo {embargo_bars}")
        else:
            print(f"✓ Val-Test embargo: {val_test_gap} bars")

        # Check temporal ordering
        if train_end_ts >= val_start_ts:
            print(f"\n⚠️  ERROR: Train end ({train_end_ts}) >= Val start ({val_start_ts})")
        if val_end_ts >= test_start_ts:
            print(f"⚠️  ERROR: Val end ({val_end_ts}) >= Test start ({test_start_ts})")

        print("\n" + "=" * 60 + "\n")

    return {
        "train": (train_start, train_end),
        "val": (val_start, val_end),
        "test": (test_start, test_end),
        "embargo_bars": embargo_bars,
        "train_timestamps": {
            "start": str(bars_df.iloc[train_start]["timestamp"]),
            "end": str(bars_df.iloc[train_end - 1]["timestamp"]),
        },
        "val_timestamps": {
            "start": str(bars_df.iloc[val_start]["timestamp"]),
            "end": str(bars_df.iloc[val_end - 1]["timestamp"]),
        },
        "test_timestamps": {
            "start": str(bars_df.iloc[test_start]["timestamp"]),
            "end": str(bars_df.iloc[test_end - 1]["timestamp"]),
        },
    }


if __name__ == "__main__":
    """Test the new labeling approach on MES data."""
    print("Loading MES data...")
    with pd.HDFStore("data/processed/mes_bars.h5", "r") as store:
        bars = store["bars_5min"]

    # Test on recent data (last 5000 bars ~ 2 months)
    test_bars = bars.tail(5000).reset_index(drop=True)

    print(f"\nCreating fixed-horizon labels...")
    print(f"Test dataset: {len(test_bars):,} bars")

    # Create labels with 12-bar horizon (60 min) and 10-tick threshold (2.5 points)
    labels = create_fixed_horizon_labels(
        test_bars,
        horizon_bars=12,
        threshold_ticks=10,
        tick_size=0.25,
        verbose=True,
    )

    print("\nSample labels:")
    print(labels.head(20))

    # Test split creation
    splits = get_label_splits(
        labels_df=labels,
        bars_df=test_bars,
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        horizon_bars=12,
        lookback_bars=100,
        verbose=True,
    )

    print("\nSplits:")
    for split_name, (start, end) in [("train", splits["train"]), ("val", splits["val"]), ("test", splits["test"])]:
        print(f"{split_name}: {start} → {end}")
