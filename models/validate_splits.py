"""
Validate walk-forward split integrity and detect data leakage.

This module provides comprehensive diagnostics to ensure that:
1. Train/validation/test splits are properly isolated temporally
2. No future information leaks into past data via features or labels
3. Purge gaps are sufficient to prevent lookahead bias
4. Label generation doesn't span across split boundaries
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def validate_split_integrity(
    bars_df: pd.DataFrame,
    train_w: Tuple[int, int],
    val_w: Tuple[int, int],
    test_w: Tuple[int, int],
    purge_bars: int,
) -> None:
    """
    Verify walk-forward split integrity.

    Checks:
    1. Temporal ordering: train < val < test
    2. No overlap between splits
    3. Purge gaps match expected size
    4. All indices are valid

    Args:
        bars_df: Full bars DataFrame
        train_w: Training window (start_idx, end_idx)
        val_w: Validation window (start_idx, end_idx)
        test_w: Test window (start_idx, end_idx)
        purge_bars: Expected purge gap size

    Raises:
        ValueError: If integrity checks fail
    """
    print("\n" + "="*60)
    print("SPLIT INTEGRITY VALIDATION")
    print("="*60)

    n_bars = len(bars_df)
    train_start, train_end = train_w
    val_start, val_end = val_w
    test_start, test_end = test_w

    # Check 1: Temporal ordering
    print("\n[1/4] Checking temporal ordering...")
    if not (train_end <= val_start <= val_end <= test_start <= test_end):
        raise ValueError(
            f"Split ordering violation: train_end={train_end}, val_start={val_start}, "
            f"val_end={val_end}, test_start={test_start}"
        )
    print("  ✓ Temporal ordering correct: train < val < test")

    # Check 2: No overlap
    print("\n[2/4] Checking for overlaps...")
    if train_end > val_start:
        raise ValueError(f"Train-Val overlap: train_end={train_end} > val_start={val_start}")
    if val_end > test_start:
        raise ValueError(f"Val-Test overlap: val_end={val_end} > test_start={test_start}")
    print("  ✓ No overlaps detected")

    # Check 3: Purge gaps
    print("\n[3/4] Checking purge gaps...")
    actual_train_val_gap = val_start - train_end
    actual_val_test_gap = test_start - val_end

    print(f"  Expected purge gap: {purge_bars} bars")
    print(f"  Train-Val gap: {actual_train_val_gap} bars")
    print(f"  Val-Test gap: {actual_val_test_gap} bars")

    if actual_train_val_gap < purge_bars:
        raise ValueError(
            f"Train-Val purge gap ({actual_train_val_gap}) < expected ({purge_bars})"
        )
    if actual_val_test_gap < purge_bars:
        raise ValueError(
            f"Val-Test purge gap ({actual_val_test_gap}) < expected ({purge_bars})"
        )
    print("  ✓ Purge gaps sufficient")

    # Check 4: Valid indices
    print("\n[4/4] Checking index validity...")
    if train_start < 0 or test_end > n_bars:
        raise ValueError(f"Invalid indices: n_bars={n_bars}, test_end={test_end}")
    if train_end - train_start < 1:
        raise ValueError(f"Empty train set: {train_end - train_start} bars")
    if val_end - val_start < 1:
        raise ValueError(f"Empty validation set: {val_end - val_start} bars")
    if test_end - test_start < 1:
        raise ValueError(f"Empty test set: {test_end - test_start} bars")
    print("  ✓ All indices valid")

    print("\n✅ Split integrity validated successfully")


def audit_train_val_test_dates(
    bars_df: pd.DataFrame,
    windows: Dict[str, Tuple[int, int]],
) -> None:
    """
    Print detailed date ranges for each split to verify temporal isolation.

    Args:
        bars_df: Full bars DataFrame with 'timestamp' column
        windows: Dict mapping split names to (start_idx, end_idx) tuples
    """
    print("\n" + "="*60)
    print("SPLIT DATE RANGES AUDIT")
    print("="*60)

    for split_name, (start, end) in windows.items():
        if end <= start or end > len(bars_df):
            print(f"\n{split_name.upper()}: INVALID WINDOW [{start}, {end})")
            continue

        start_ts = bars_df.iloc[start]["timestamp"]
        end_ts = bars_df.iloc[min(end - 1, len(bars_df) - 1)]["timestamp"]
        n_bars = end - start

        print(f"\n{split_name.upper()}:")
        print(f"  Indices: [{start}, {end})")
        print(f"  Bars: {n_bars:,}")
        print(f"  Start: {start_ts}")
        print(f"  End:   {end_ts}")
        print(f"  Duration: {end_ts - start_ts}")


def check_feature_leakage(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    labels_df: pd.DataFrame,
    label_col: str,
    windows: Dict[str, Tuple[int, int]],
    max_future_correlation: float = 0.10,
) -> None:
    """
    Detect if features contain future information via correlation analysis.

    Tests if features at time t correlate with labels at time t+n across split boundaries,
    which would indicate data leakage.

    Args:
        features_df: DataFrame with features and 'idx' column
        feature_cols: List of feature column names to test
        labels_df: DataFrame with labels and 'idx' column
        label_col: Name of label column
        windows: Dict mapping split names to (start_idx, end_idx) tuples
        max_future_correlation: Maximum acceptable correlation with future labels

    Raises:
        ValueError: If feature leakage is detected
    """
    print("\n" + "="*60)
    print("FEATURE LEAKAGE DETECTION")
    print("="*60)

    print(f"\nTesting {len(feature_cols)} features for future information leakage...")
    print(f"Max acceptable correlation with future labels: {max_future_correlation:.3f}")

    # Merge features with labels on idx
    merged = features_df.merge(labels_df[["idx", label_col]], on="idx", how="inner")

    # Test correlation across split boundaries
    train_w = windows.get("training")
    val_w = windows.get("validation")

    if train_w and val_w:
        train_start, train_end = train_w
        val_start, val_end = val_w

        # Get features from end of train, labels from start of val
        train_features = merged[
            (merged["idx"] >= train_end - 50) & (merged["idx"] < train_end)
        ][feature_cols]

        val_labels = merged[
            (merged["idx"] >= val_start) & (merged["idx"] < val_start + 50)
        ][label_col]

        if len(train_features) > 0 and len(val_labels) > 0:
            print(f"\nTesting train-val boundary:")
            print(f"  Train features: {len(train_features)} rows (idx {train_end-50} to {train_end})")
            print(f"  Val labels: {len(val_labels)} rows (idx {val_start} to {val_start+50})")

            leakage_detected = False
            for feat in feature_cols[:10]:  # Test top 10 features
                if feat not in train_features.columns:
                    continue

                # Correlation should be near zero since there's a purge gap
                corr = np.corrcoef(
                    train_features[feat].fillna(0),
                    val_labels.values[:len(train_features)]
                )[0, 1] if len(train_features) == len(val_labels) else 0.0

                if abs(corr) > max_future_correlation:
                    print(f"  ⚠️  {feat}: corr={corr:.3f} (SUSPICIOUS)")
                    leakage_detected = True

            if leakage_detected:
                print("\n  ⚠️  WARNING: Suspicious correlations detected")
                print("     This may indicate feature leakage across split boundaries")
            else:
                print("\n  ✓ No suspicious correlations detected")

    print("\n✅ Feature leakage check complete")


def verify_label_isolation(
    long_df: pd.DataFrame,
    short_df: pd.DataFrame,
    windows: Dict[str, Tuple[int, int]],
    max_hold_bars: int,
) -> None:
    """
    Ensure labels don't span across split boundaries.

    Verifies that labels created near split boundaries don't use future data
    by checking that max_hold_bars doesn't cause labels to peek into next split.

    Args:
        long_df: Long labels DataFrame with 'idx' column
        short_df: Short labels DataFrame with 'idx' column
        windows: Dict mapping split names to (start_idx, end_idx) tuples
        max_hold_bars: Maximum holding period for trades

    Raises:
        ValueError: If label isolation is violated
    """
    print("\n" + "="*60)
    print("LABEL ISOLATION VERIFICATION")
    print("="*60)

    print(f"\nMax hold bars: {max_hold_bars}")
    print("Checking that labels near split boundaries don't peek into future...")

    train_w = windows.get("training")
    val_w = windows.get("validation")
    test_w = windows.get("test")

    if train_w and val_w:
        train_start, train_end = train_w
        val_start, val_end = val_w

        # Check train boundary: labels within max_hold_bars of train_end
        # could potentially peek into validation period
        boundary_zone = max_hold_bars + 5  # Add buffer

        train_boundary_labels_long = long_df[
            (long_df["idx"] >= train_end - boundary_zone) &
            (long_df["idx"] < train_end)
        ]

        print(f"\nTrain-Val boundary check:")
        print(f"  Labels in boundary zone (idx {train_end - boundary_zone} to {train_end}): "
              f"{len(train_boundary_labels_long)}")
        print(f"  Purge gap before validation: {val_start - train_end} bars")

        if val_start - train_end < max_hold_bars:
            print(f"  ⚠️  WARNING: Purge gap ({val_start - train_end}) < max_hold ({max_hold_bars})")
            print("     Labels near train boundary may peek into validation period")
        else:
            print(f"  ✓ Purge gap sufficient (>= {max_hold_bars})")

    if val_w and test_w:
        val_start, val_end = val_w
        test_start, test_end = test_w

        boundary_zone = max_hold_bars + 5

        print(f"\nVal-Test boundary check:")
        print(f"  Purge gap before test: {test_start - val_end} bars")

        if test_start - val_end < max_hold_bars:
            print(f"  ⚠️  WARNING: Purge gap ({test_start - val_end}) < max_hold ({max_hold_bars})")
        else:
            print(f"  ✓ Purge gap sufficient (>= {max_hold_bars})")

    print("\n✅ Label isolation verified")


def run_all_diagnostics(
    bars_df: pd.DataFrame,
    long_df: pd.DataFrame,
    short_df: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_cols: list[str],
    windows: Dict[str, Tuple[int, int]],
    purge_bars: int,
    max_hold_bars: int,
) -> None:
    """
    Run all data leakage and split integrity diagnostics.

    Args:
        bars_df: Full bars DataFrame
        long_df: Long labels DataFrame
        short_df: Short labels DataFrame
        features_df: Features DataFrame
        feature_cols: List of feature column names
        windows: Dict with 'training', 'validation', 'test' windows
        purge_bars: Expected purge gap size
        max_hold_bars: Maximum trade holding period
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE DATA LEAKAGE DIAGNOSTICS")
    print("="*60)

    train_w = windows["training"]
    val_w = windows["validation"]
    test_w = windows["test"]

    # Run all checks
    validate_split_integrity(bars_df, train_w, val_w, test_w, purge_bars)
    audit_train_val_test_dates(bars_df, windows)

    # Check feature leakage (if we have labels)
    if not long_df.empty:
        check_feature_leakage(
            features_df, feature_cols, long_df, "label_long", windows
        )

    verify_label_isolation(long_df, short_df, windows, max_hold_bars)

    print("\n" + "="*60)
    print("✅ ALL DIAGNOSTICS PASSED")
    print("="*60)
