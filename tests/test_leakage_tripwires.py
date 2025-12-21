"""
Leakage tripwire tests: detect data leakage in ML pipeline.

These tests are designed to FAIL if there's data leakage.
If leakage exists, the model will still perform well even when
we intentionally break the causal relationship.

Run these tests before deploying any model to production.

Usage:
    python tests/test_leakage_tripwires.py --data-path data/processed/mes_bars.h5
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.simple_config import RISK_CONFIG, TRAINING_CONFIG
from data.clean_bars import clean_bars
from features.engineer import add_features, get_recommended_features
from features.labels_v2 import create_fixed_horizon_labels


def load_test_data(h5_path: str, n_bars: int = 5000) -> pd.DataFrame:
    """Load small subset of data for testing."""
    with pd.HDFStore(h5_path, "r") as store:
        bars = store["bars_5min"].tail(n_bars).copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars = clean_bars(bars, tick_size=RISK_CONFIG.tick_size, verbose=False)
    return bars


def prepare_baseline_dataset(
    bars_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare baseline dataset for testing.

    Returns:
        X_train, y_train, X_test, y_test (all long labels)
    """
    # Features
    features_df = add_features(bars_df, verbose=False)
    features_df = features_df.reset_index().rename(columns={"index": "idx"})
    feature_cols = get_recommended_features()

    # Labels
    labels_df = create_fixed_horizon_labels(
        bars_df,
        horizon_bars=TRAINING_CONFIG.horizon_bars,
        threshold_ticks=TRAINING_CONFIG.threshold_ticks,
        tick_size=RISK_CONFIG.tick_size,
        session_start=RISK_CONFIG.session_start,
        session_end=RISK_CONFIG.session_end,
        verbose=False,
    )

    # Merge
    df = labels_df.merge(features_df, on="idx", how="inner")
    df = df.dropna(subset=feature_cols + ["label_long"])

    # Simple train/test split (60/40)
    n = len(df)
    train_end = int(n * 0.6)

    train_df = df.iloc[:train_end]
    test_df = df.iloc[train_end:]

    X_train = train_df[feature_cols].values
    y_train = train_df["label_long"].values.astype(int)
    X_test = test_df[feature_cols].values
    y_test = test_df["label_long"].values.astype(int)

    return X_train, y_train, X_test, y_test


def train_simple_model(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42) -> RandomForestClassifier:
    """Train a simple RF model for testing."""
    model = RandomForestClassifier(
        n_estimators=100,  # Smaller for speed
        max_depth=TRAINING_CONFIG.rf_max_depth,
        min_samples_leaf=TRAINING_CONFIG.rf_min_samples_leaf,
        min_samples_split=TRAINING_CONFIG.rf_min_samples_split,
        max_features=TRAINING_CONFIG.rf_max_features,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def test_label_shuffle(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Test 1: Label Shuffle Test

    Randomly shuffle labels to break the feature-label relationship.
    If model still performs well (AUC > 0.6), there's likely data leakage.

    Expected:
        - Baseline AUC > 0.55 (model learns something)
        - Shuffled AUC ~ 0.50 (random performance)
        - If shuffled AUC > 0.60 → FAIL (leakage suspected)
    """
    print("\n" + "=" * 60)
    print("TEST 1: LABEL SHUFFLE (Leakage Tripwire)")
    print("=" * 60)
    print("\nPurpose: Verify model performance collapses when labels are randomized")
    print("Expected: AUC drops to ~0.50 (random)")

    # Baseline model
    print("\nTraining baseline model...")
    baseline_model = train_simple_model(X_train, y_train, seed=42)
    baseline_prob = baseline_model.predict_proba(X_test)[:, 1]

    if len(np.unique(y_test)) == 2:
        baseline_auc = roc_auc_score(y_test, baseline_prob)
    else:
        baseline_auc = 0.5

    print(f"Baseline AUC: {baseline_auc:.4f}")

    # Shuffled labels model
    print("\nTraining model with shuffled labels...")
    y_train_shuffled = y_train.copy()
    np.random.RandomState(123).shuffle(y_train_shuffled)

    shuffled_model = train_simple_model(X_train, y_train_shuffled, seed=43)
    shuffled_prob = shuffled_model.predict_proba(X_test)[:, 1]

    if len(np.unique(y_test)) == 2:
        shuffled_auc = roc_auc_score(y_test, shuffled_prob)
    else:
        shuffled_auc = 0.5

    print(f"Shuffled AUC: {shuffled_auc:.4f}")

    # Evaluation
    delta = baseline_auc - shuffled_auc
    print(f"\nΔ AUC (baseline - shuffled): {delta:.4f}")

    # Thresholds
    MIN_BASELINE_AUC = 0.55  # Model should learn something
    MAX_SHUFFLED_AUC = 0.60  # Shuffled should be near random (0.50 ± 0.10)
    MIN_DELTA_AUC = 0.05  # Baseline should be significantly better

    passed = True
    reasons = []

    if baseline_auc < MIN_BASELINE_AUC:
        passed = False
        reasons.append(f"Baseline AUC too low ({baseline_auc:.4f} < {MIN_BASELINE_AUC})")

    if shuffled_auc > MAX_SHUFFLED_AUC:
        passed = False
        reasons.append(f"Shuffled AUC too high ({shuffled_auc:.4f} > {MAX_SHUFFLED_AUC}) - LEAKAGE SUSPECTED!")

    if delta < MIN_DELTA_AUC:
        passed = False
        reasons.append(f"Δ AUC too small ({delta:.4f} < {MIN_DELTA_AUC}) - model may not be learning")

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    if passed:
        print("✅ PASSED: Label shuffle test")
        print("   Model performance collapsed as expected when labels randomized")
    else:
        print("❌ FAILED: Label shuffle test")
        for reason in reasons:
            print(f"   - {reason}")

    return {
        "test": "label_shuffle",
        "passed": passed,
        "baseline_auc": baseline_auc,
        "shuffled_auc": shuffled_auc,
        "delta_auc": delta,
        "reasons": reasons,
    }


def test_time_shift(bars_df: pd.DataFrame) -> Dict:
    """
    Test 2: Time Shift Test

    Shift features forward by 1 bar relative to labels (features see 1 bar into future).
    If model still performs well, features contain future information (leakage).

    Expected:
        - Baseline AUC > 0.55 (model learns)
        - Time-shifted AUC ~ 0.50 (random) OR < baseline
        - If time-shifted AUC > baseline → FAIL (features leak future info)
    """
    print("\n" + "=" * 60)
    print("TEST 2: TIME SHIFT (Leakage Tripwire)")
    print("=" * 60)
    print("\nPurpose: Verify features don't contain future information")
    print("Expected: AUC collapses when features are shifted forward 1 bar")

    # Prepare baseline
    features_df = add_features(bars_df, verbose=False)
    features_df = features_df.reset_index().rename(columns={"index": "idx"})
    feature_cols = get_recommended_features()

    labels_df = create_fixed_horizon_labels(
        bars_df,
        horizon_bars=TRAINING_CONFIG.horizon_bars,
        threshold_ticks=TRAINING_CONFIG.threshold_ticks,
        tick_size=RISK_CONFIG.tick_size,
        session_start=RISK_CONFIG.session_start,
        session_end=RISK_CONFIG.session_end,
        verbose=False,
    )

    # Merge
    df_baseline = labels_df.merge(features_df, on="idx", how="inner")
    df_baseline = df_baseline.dropna(subset=feature_cols + ["label_long"])

    # Time-shifted: shift features forward by 1 bar (idx + 1)
    features_df_shifted = features_df.copy()
    features_df_shifted["idx"] = features_df_shifted["idx"] - 1  # Shift index down (features see future)

    df_shifted = labels_df.merge(features_df_shifted, on="idx", how="inner")
    df_shifted = df_shifted.dropna(subset=feature_cols + ["label_long"])

    # Train/test split
    n_base = len(df_baseline)
    n_shift = len(df_shifted)

    if n_base < 100 or n_shift < 100:
        return {
            "test": "time_shift",
            "passed": False,
            "reasons": ["Insufficient data after time shift"],
        }

    # Use same split proportion
    train_end_base = int(n_base * 0.6)
    train_end_shift = int(n_shift * 0.6)

    X_train_base = df_baseline.iloc[:train_end_base][feature_cols].values
    y_train_base = df_baseline.iloc[:train_end_base]["label_long"].values.astype(int)
    X_test_base = df_baseline.iloc[train_end_base:][feature_cols].values
    y_test_base = df_baseline.iloc[train_end_base:]["label_long"].values.astype(int)

    X_train_shift = df_shifted.iloc[:train_end_shift][feature_cols].values
    y_train_shift = df_shifted.iloc[:train_end_shift]["label_long"].values.astype(int)
    X_test_shift = df_shifted.iloc[train_end_shift:][feature_cols].values
    y_test_shift = df_shifted.iloc[train_end_shift:]["label_long"].values.astype(int)

    # Baseline model
    print("\nTraining baseline model...")
    baseline_model = train_simple_model(X_train_base, y_train_base, seed=42)
    baseline_prob = baseline_model.predict_proba(X_test_base)[:, 1]

    if len(np.unique(y_test_base)) == 2:
        baseline_auc = roc_auc_score(y_test_base, baseline_prob)
    else:
        baseline_auc = 0.5

    print(f"Baseline AUC: {baseline_auc:.4f}")

    # Time-shifted model
    print("\nTraining model with time-shifted features (features see 1 bar ahead)...")
    shifted_model = train_simple_model(X_train_shift, y_train_shift, seed=43)
    shifted_prob = shifted_model.predict_proba(X_test_shift)[:, 1]

    if len(np.unique(y_test_shift)) == 2:
        shifted_auc = roc_auc_score(y_test_shift, shifted_prob)
    else:
        shifted_auc = 0.5

    print(f"Time-shifted AUC: {shifted_auc:.4f}")

    # Evaluation
    delta = shifted_auc - baseline_auc
    print(f"\nΔ AUC (shifted - baseline): {delta:.4f}")

    # Thresholds
    MIN_BASELINE_AUC = 0.55
    MAX_DELTA_AUC = 0.10  # Shifted should not be significantly better than baseline

    passed = True
    reasons = []

    if baseline_auc < MIN_BASELINE_AUC:
        passed = False
        reasons.append(f"Baseline AUC too low ({baseline_auc:.4f} < {MIN_BASELINE_AUC})")

    if delta > MAX_DELTA_AUC:
        passed = False
        reasons.append(
            f"Time-shifted AUC > baseline by {delta:.4f} - FEATURES LEAK FUTURE INFO!"
        )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    if passed:
        print("✅ PASSED: Time shift test")
        print("   Features do not contain future information")
    else:
        print("❌ FAILED: Time shift test")
        for reason in reasons:
            print(f"   - {reason}")

    return {
        "test": "time_shift",
        "passed": passed,
        "baseline_auc": baseline_auc,
        "shifted_auc": shifted_auc,
        "delta_auc": delta,
        "reasons": reasons,
    }


def test_split_integrity(bars_df: pd.DataFrame) -> Dict:
    """
    Test 3: Split Integrity Test

    Verify train/val/test splits have proper embargo and no temporal overlap.

    Checks:
        - Train end < Val start (with embargo gap)
        - Val end < Test start (with embargo gap)
        - Embargo >= horizon + lookback + 1
        - No timestamp overlap
    """
    print("\n" + "=" * 60)
    print("TEST 3: SPLIT INTEGRITY (Leakage Tripwire)")
    print("=" * 60)
    print("\nPurpose: Verify walk-forward splits have proper embargo gaps")
    print("Expected: No temporal overlap between train/val/test")

    # Calculate expected embargo
    embargo_bars = TRAINING_CONFIG.horizon_bars + TRAINING_CONFIG.lookback_bars + 1

    # Create labels
    labels_df = create_fixed_horizon_labels(
        bars_df,
        horizon_bars=TRAINING_CONFIG.horizon_bars,
        threshold_ticks=TRAINING_CONFIG.threshold_ticks,
        tick_size=RISK_CONFIG.tick_size,
        session_start=RISK_CONFIG.session_start,
        session_end=RISK_CONFIG.session_end,
        verbose=False,
    )

    # Simple split
    n = len(bars_df)
    train_frac = 0.5
    val_frac = 0.2
    test_frac = 0.3

    train_start = TRAINING_CONFIG.lookback_bars
    train_end = int(train_start + (n - train_start) * train_frac)

    val_start = train_end + embargo_bars
    val_end = int(val_start + (n - train_start) * val_frac)

    test_start = val_end + embargo_bars
    test_end = n

    print(f"\nEmbargo calculation:")
    print(f"  horizon_bars: {TRAINING_CONFIG.horizon_bars}")
    print(f"  lookback_bars: {TRAINING_CONFIG.lookback_bars}")
    print(f"  Expected embargo: {embargo_bars}")

    print(f"\nSplit indices:")
    print(f"  Train: [{train_start}, {train_end}]")
    print(f"  Val:   [{val_start}, {val_end}]")
    print(f"  Test:  [{test_start}, {test_end}]")

    # Timestamps
    train_ts_start = bars_df.iloc[train_start]["timestamp"]
    train_ts_end = bars_df.iloc[train_end - 1]["timestamp"]
    val_ts_start = bars_df.iloc[val_start]["timestamp"]
    val_ts_end = bars_df.iloc[val_end - 1]["timestamp"]
    test_ts_start = bars_df.iloc[test_start]["timestamp"]
    test_ts_end = bars_df.iloc[test_end - 1]["timestamp"]

    print(f"\nTimestamps:")
    print(f"  Train: {train_ts_start} → {train_ts_end}")
    print(f"  Val:   {val_ts_start} → {val_ts_end}")
    print(f"  Test:  {test_ts_start} → {test_ts_end}")

    # Checks
    train_val_gap = val_start - train_end
    val_test_gap = test_start - val_end

    passed = True
    reasons = []

    # Check 1: Embargo gaps
    if train_val_gap != embargo_bars:
        passed = False
        reasons.append(f"Train-Val gap ({train_val_gap}) != embargo ({embargo_bars})")

    if val_test_gap != embargo_bars:
        passed = False
        reasons.append(f"Val-Test gap ({val_test_gap}) != embargo ({embargo_bars})")

    # Check 2: Temporal ordering
    if train_ts_end >= val_ts_start:
        passed = False
        reasons.append(f"Train end ({train_ts_end}) >= Val start ({val_ts_start})")

    if val_ts_end >= test_ts_start:
        passed = False
        reasons.append(f"Val end ({val_ts_end}) >= Test start ({test_ts_start})")

    # Check 3: No negative gaps
    if train_val_gap < 0:
        passed = False
        reasons.append(f"Negative train-val gap: {train_val_gap}")

    if val_test_gap < 0:
        passed = False
        reasons.append(f"Negative val-test gap: {val_test_gap}")

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    if passed:
        print("✅ PASSED: Split integrity test")
        print("   All splits have proper embargo gaps and temporal ordering")
    else:
        print("❌ FAILED: Split integrity test")
        for reason in reasons:
            print(f"   - {reason}")

    return {
        "test": "split_integrity",
        "passed": passed,
        "train_val_gap": train_val_gap,
        "val_test_gap": val_test_gap,
        "expected_embargo": embargo_bars,
        "reasons": reasons,
    }


def main():
    parser = argparse.ArgumentParser(description="Run leakage tripwire tests")
    parser.add_argument("--data-path", default="data/processed/mes_bars.h5")
    parser.add_argument("--n-bars", type=int, default=5000, help="Number of bars to test (smaller = faster)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("LEAKAGE TRIPWIRE TESTS")
    print("=" * 60)
    print("\nThese tests are designed to FAIL if there's data leakage.")
    print("Run before deploying any model to production.")
    print("\n" + "=" * 60)

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    print(f"Using last {args.n_bars} bars for testing (smaller = faster)")
    bars = load_test_data(args.data_path, n_bars=args.n_bars)
    print(f"Loaded {len(bars):,} bars")

    # Prepare baseline dataset
    print("\nPreparing baseline dataset...")
    X_train, y_train, X_test, y_test = prepare_baseline_dataset(bars)
    print(f"Train: {len(X_train):,} samples")
    print(f"Test:  {len(X_test):,} samples")

    # Run tests
    results = []

    # Test 1: Label Shuffle
    results.append(test_label_shuffle(X_train, y_train, X_test, y_test))

    # Test 2: Time Shift
    results.append(test_time_shift(bars))

    # Test 3: Split Integrity
    results.append(test_split_integrity(bars))

    # Summary
    print("\n\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = all(r["passed"] for r in results)

    for r in results:
        status = "✅ PASSED" if r["passed"] else "❌ FAILED"
        print(f"\n{r['test']:20s}: {status}")
        if not r["passed"]:
            for reason in r.get("reasons", []):
                print(f"  - {reason}")

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nNo data leakage detected. Safe to proceed.")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nData leakage suspected. DO NOT deploy this model.")
        print("Review failed tests and fix leakage before proceeding.")
    print("=" * 60 + "\n")

    # Exit code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
