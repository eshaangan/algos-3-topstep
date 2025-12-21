"""
Train ML classifiers using V2 approach: fixed-horizon market labels + pure ML strategy.

Key differences from train.py:
- Uses labels_v2 (market-centric) instead of TP/SL simulation labels
- No heuristic gates (trend/hour/lunch/ATR filters)
- Frequency control via score quantile + daily trade budget
- Lockbox split for final evaluation
- Simpler, more robust to overfitting
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from core.simple_config import RISK_CONFIG, TRAINING_CONFIG
from data.clean_bars import clean_bars
from features.engineer import add_features, get_recommended_features, select_features
from features.labels_v2 import create_fixed_horizon_labels, get_label_splits


def load_bars(h5_path: str) -> pd.DataFrame:
    """Load and clean bars from HDF5."""
    with pd.HDFStore(h5_path, "r") as store:
        bars = store["bars_5min"].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars, _ = clean_bars(bars, tick_size=RISK_CONFIG.tick_size, verbose=True)
    return bars


def prepare_features_and_labels(
    bars_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Prepare features and labels for training.

    Returns:
        features_df: Features with idx column
        labels_df: Labels with idx, label_long, label_short columns
        feature_cols: List of feature column names
    """
    print("\n" + "=" * 60)
    print("PREPARING FEATURES AND LABELS")
    print("=" * 60)

    # 1. Generate features
    features_df = add_features(bars_df, verbose=True)
    features_df = features_df.reset_index().rename(columns={"index": "idx"})

    # 2. Feature selection
    feature_selection_mode = getattr(TRAINING_CONFIG, "feature_selection_mode", "recommended")
    if feature_selection_mode == "recommended":
        feature_cols = get_recommended_features()
        print(f"\n✓ Using recommended {len(feature_cols)} features (top ~75-80% importance)")
    else:
        feature_cols = select_features()
        print(f"\n✓ Using all {len(feature_cols)} features")

    # 3. Generate labels using V2 approach (fixed-horizon market labels)
    labels_df = create_fixed_horizon_labels(
        bars_df,
        horizon_bars=TRAINING_CONFIG.horizon_bars,
        threshold_ticks=TRAINING_CONFIG.threshold_ticks,
        tick_size=RISK_CONFIG.tick_size,
        session_start=RISK_CONFIG.session_start,
        session_end=RISK_CONFIG.session_end,
        verbose=True,
    )

    print(f"\n✓ Generated {len(labels_df):,} labeled bars")
    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Labels: label_long, label_short (fixed {TRAINING_CONFIG.horizon_bars}-bar horizon)")

    return features_df, labels_df, feature_cols


def create_walk_forward_splits(
    bars_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> Dict[str, Tuple[int, int]]:
    """
    Create walk-forward splits with proper embargo and lockbox.

    Returns:
        windows: {"train": (start, end), "val": (start, end), "test": (start, end), "lockbox": (start, end)}
    """
    print("\n" + "=" * 60)
    print("CREATING WALK-FORWARD SPLITS WITH LOCKBOX")
    print("=" * 60)

    # Calculate embargo (horizon + lookback + 1)
    embargo_bars = TRAINING_CONFIG.horizon_bars + TRAINING_CONFIG.lookback_bars + 1

    # Total bars
    n_bars = len(bars_df)

    # Allocate splits
    # Train: 50%, Val: 20%, Test: 20%, Lockbox: 10%
    train_frac = TRAINING_CONFIG.train_fraction
    val_frac = TRAINING_CONFIG.val_fraction
    test_frac = TRAINING_CONFIG.test_fraction
    lockbox_frac = TRAINING_CONFIG.lockbox_fraction

    # Validate fractions
    total = train_frac + val_frac + test_frac + lockbox_frac
    if not abs(total - 1.0) < 0.01:
        raise ValueError(f"Fractions must sum to 1.0, got {total:.2f}")

    # Calculate split indices with embargo gaps
    train_start = TRAINING_CONFIG.lookback_bars
    train_end = int(train_start + (n_bars - train_start) * train_frac)

    val_start = train_end + embargo_bars
    val_end = int(val_start + (n_bars - train_start) * val_frac)

    test_start = val_end + embargo_bars
    test_end = int(test_start + (n_bars - train_start) * test_frac)

    lockbox_start = test_end + embargo_bars
    lockbox_end = n_bars

    print(f"\nEmbargo gap: {embargo_bars} bars")
    print(f"  = horizon({TRAINING_CONFIG.horizon_bars}) + lookback({TRAINING_CONFIG.lookback_bars}) + 1")

    print(f"\nSplit indices:")
    print(f"  Train:   [{train_start:6,}, {train_end:6,}] = {train_end - train_start:6,} bars")
    print(f"  Val:     [{val_start:6,}, {val_end:6,}] = {val_end - val_start:6,} bars")
    print(f"  Test:    [{test_start:6,}, {test_end:6,}] = {test_end - test_start:6,} bars")
    print(f"  Lockbox: [{lockbox_start:6,}, {lockbox_end:6,}] = {lockbox_end - lockbox_start:6,} bars")

    # Get timestamps
    def get_timestamp_range(start: int, end: int) -> Tuple[str, str]:
        if start >= len(bars_df) or end > len(bars_df):
            return ("N/A", "N/A")
        start_ts = bars_df.iloc[start]["timestamp"]
        end_ts = bars_df.iloc[min(end - 1, len(bars_df) - 1)]["timestamp"]
        return (str(start_ts), str(end_ts))

    train_ts = get_timestamp_range(train_start, train_end)
    val_ts = get_timestamp_range(val_start, val_end)
    test_ts = get_timestamp_range(test_start, test_end)
    lockbox_ts = get_timestamp_range(lockbox_start, lockbox_end)

    print(f"\nTimestamps:")
    print(f"  Train:   {train_ts[0]} → {train_ts[1]}")
    print(f"  Val:     {val_ts[0]} → {val_ts[1]}")
    print(f"  Test:    {test_ts[0]} → {test_ts[1]}")
    print(f"  Lockbox: {lockbox_ts[0]} → {lockbox_ts[1]}")

    # Sanity checks
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    checks = [
        (val_start - train_end == embargo_bars, f"Train-Val embargo: {val_start - train_end} == {embargo_bars}"),
        (test_start - val_end == embargo_bars, f"Val-Test embargo: {test_start - val_end} == {embargo_bars}"),
        (lockbox_start - test_end == embargo_bars, f"Test-Lockbox embargo: {lockbox_start - test_end} == {embargo_bars}"),
    ]

    for passed, msg in checks:
        print(f"  {'✓' if passed else '⚠️ '} {msg}")

    print("\n" + "=" * 60 + "\n")

    return {
        "train": (train_start, train_end),
        "val": (val_start, val_end),
        "test": (test_start, test_end),
        "lockbox": (lockbox_start, lockbox_end),
        "embargo_bars": embargo_bars,
        "train_timestamps": {"start": train_ts[0], "end": train_ts[1]},
        "val_timestamps": {"start": val_ts[0], "end": val_ts[1]},
        "test_timestamps": {"start": test_ts[0], "end": test_ts[1]},
        "lockbox_timestamps": {"start": lockbox_ts[0], "end": lockbox_ts[1]},
    }


def slice_dataset(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_cols: List[str],
    window: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice dataset for a given window.

    Returns:
        X: Feature matrix
        y_long: Long labels
        y_short: Short labels
    """
    start, end = window

    # Merge features and labels on idx
    df = labels_df.merge(features_df, on="idx", how="inner")

    # Filter to window
    df = df[(df["idx"] >= start) & (df["idx"] < end)].copy()

    # Drop NaN
    df = df.dropna(subset=feature_cols + ["label_long", "label_short"])

    X = df[feature_cols].values
    y_long = df["label_long"].values.astype(int)
    y_short = df["label_short"].values.astype(int)

    return X, y_long, y_short


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # ROC AUC only if both classes present
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.0

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn)})

    return metrics


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    direction: str,
    seed: int,
) -> Tuple[RandomForestClassifier, Dict[str, Dict[str, float]]]:
    """Train a Random Forest model and evaluate on all splits."""
    print(f"\nTraining {direction.upper()} model...")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")

    # Class distribution
    print(f"\n  Train label distribution:")
    print(f"    Positive: {(y_train == 1).sum():,} ({(y_train == 1).mean() * 100:.1f}%)")
    print(f"    Negative: {(y_train == 0).sum():,} ({(y_train == 0).mean() * 100:.1f}%)")

    # Train model
    model = RandomForestClassifier(
        n_estimators=TRAINING_CONFIG.rf_n_estimators,
        max_depth=TRAINING_CONFIG.rf_max_depth,
        min_samples_leaf=TRAINING_CONFIG.rf_min_samples_leaf,
        min_samples_split=TRAINING_CONFIG.rf_min_samples_split,
        max_features=TRAINING_CONFIG.rf_max_features,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Get predictions
    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = {
        "training": classification_metrics(y_train, train_prob),
        "validation": classification_metrics(y_val, val_prob),
        "test": classification_metrics(y_test, test_prob),
    }

    print(f"\n  Metrics:")
    for split, m in metrics.items():
        print(f"    {split:10s}: Acc={m['accuracy']:.3f}, ROC-AUC={m['roc_auc']:.3f}, Precision={m['precision']:.3f}, Recall={m['recall']:.3f}")

    return model, metrics


def compute_score_threshold(
    long_model: RandomForestClassifier,
    short_model: RandomForestClassifier,
    X_train: np.ndarray,
) -> float:
    """
    Compute score threshold from training set quantile.

    Score = max(p_long, p_short)
    Threshold = quantile(score, score_quantile)

    This controls trade frequency: higher quantile = fewer trades.
    """
    print("\n" + "=" * 60)
    print("COMPUTING SCORE THRESHOLD FROM TRAIN QUANTILE")
    print("=" * 60)

    # Get probabilities
    p_long = long_model.predict_proba(X_train)[:, 1]
    p_short = short_model.predict_proba(X_train)[:, 1] if TRAINING_CONFIG.enable_short else np.zeros_like(p_long)

    # Compute score
    score = np.maximum(p_long, p_short)

    # Compute threshold
    threshold = np.quantile(score, TRAINING_CONFIG.score_quantile)

    print(f"\nScore quantile: {TRAINING_CONFIG.score_quantile} (top {(1 - TRAINING_CONFIG.score_quantile) * 100:.1f}%)")
    print(f"Score threshold: {threshold:.4f}")
    print(f"\nScore statistics (train):")
    print(f"  Mean:   {score.mean():.4f}")
    print(f"  Std:    {score.std():.4f}")
    print(f"  Min:    {score.min():.4f}")
    print(f"  25%:    {np.quantile(score, 0.25):.4f}")
    print(f"  50%:    {np.quantile(score, 0.50):.4f}")
    print(f"  75%:    {np.quantile(score, 0.75):.4f}")
    print(f"  95%:    {np.quantile(score, 0.95):.4f}")
    print(f"  99%:    {np.quantile(score, 0.99):.4f}")
    print(f"  Max:    {score.max():.4f}")

    # Estimate trade count
    n_above_threshold = (score >= threshold).sum()
    pct_above = n_above_threshold / len(score) * 100

    print(f"\nEstimated signals (train):")
    print(f"  Bars above threshold: {n_above_threshold:,} ({pct_above:.2f}%)")
    print(f"  Bars in train: {len(score):,}")
    print(f"  Average bars per signal: {len(score) / max(1, n_above_threshold):.1f}")

    print("\n" + "=" * 60 + "\n")

    return float(threshold)


def train_models_v2(
    bars_df: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Train models using V2 approach.

    Returns:
        results: Dict with models, metrics, windows, score_threshold, etc.
    """
    # 1. Prepare features and labels
    features_df, labels_df, feature_cols = prepare_features_and_labels(bars_df)

    # 2. Create walk-forward splits
    windows = create_walk_forward_splits(bars_df, labels_df)

    # 3. Slice datasets
    X_train, y_train_long, y_train_short = slice_dataset(
        features_df, labels_df, feature_cols, windows["train"]
    )
    X_val, y_val_long, y_val_short = slice_dataset(
        features_df, labels_df, feature_cols, windows["val"]
    )
    X_test, y_test_long, y_test_short = slice_dataset(
        features_df, labels_df, feature_cols, windows["test"]
    )
    X_lockbox, y_lockbox_long, y_lockbox_short = slice_dataset(
        features_df, labels_df, feature_cols, windows["lockbox"]
    )

    # 4. Train long model
    long_model, long_metrics = train_model(
        X_train, y_train_long,
        X_val, y_val_long,
        X_test, y_test_long,
        direction="long",
        seed=seed,
    )

    # 5. Train short model (if enabled)
    if TRAINING_CONFIG.enable_short:
        short_model, short_metrics = train_model(
            X_train, y_train_short,
            X_val, y_val_short,
            X_test, y_test_short,
            direction="short",
            seed=seed + 1,
        )
    else:
        print("\n⚠️  Short model disabled (enable_short=False)")
        short_model = None
        short_metrics = {"training": {}, "validation": {}, "test": {}}

    # 6. Compute score threshold from train quantile
    score_threshold = compute_score_threshold(long_model, short_model or long_model, X_train)

    # 7. Lockbox evaluation (optional - only if needed)
    print("\n" + "=" * 60)
    print("LOCKBOX EVALUATION (FINAL HOLDOUT)")
    print("=" * 60)
    print("\nLockbox is reserved for FINAL evaluation only.")
    print("DO NOT use lockbox metrics for tuning or model selection.")

    lockbox_long_prob = long_model.predict_proba(X_lockbox)[:, 1]
    lockbox_long_metrics = classification_metrics(y_lockbox_long, lockbox_long_prob)

    if TRAINING_CONFIG.enable_short and short_model is not None:
        lockbox_short_prob = short_model.predict_proba(X_lockbox)[:, 1]
        lockbox_short_metrics = classification_metrics(y_lockbox_short, lockbox_short_prob)
    else:
        lockbox_short_metrics = {}

    print(f"\nLockbox {len(X_lockbox):,} samples:")
    print(f"  Long:  Acc={lockbox_long_metrics.get('accuracy', 0):.3f}, ROC-AUC={lockbox_long_metrics.get('roc_auc', 0):.3f}")
    if lockbox_short_metrics:
        print(f"  Short: Acc={lockbox_short_metrics.get('accuracy', 0):.3f}, ROC-AUC={lockbox_short_metrics.get('roc_auc', 0):.3f}")

    print("\n" + "=" * 60 + "\n")

    return {
        "long_model": long_model,
        "short_model": short_model,
        "feature_cols": feature_cols,
        "metrics": {
            "long": long_metrics,
            "short": short_metrics,
        },
        "lockbox_metrics": {
            "long": lockbox_long_metrics,
            "short": lockbox_short_metrics,
        },
        "windows": windows,
        "score_threshold": score_threshold,
        "labels_v2": {
            "horizon_bars": TRAINING_CONFIG.horizon_bars,
            "threshold_ticks": TRAINING_CONFIG.threshold_ticks,
        },
        "policy_v2": {
            "score_threshold": score_threshold,
            "score_quantile": TRAINING_CONFIG.score_quantile,
            "max_trades_per_day": TRAINING_CONFIG.max_trades_per_day,
            "min_bars_between_trades": TRAINING_CONFIG.min_bars_between_trades,
            "enable_long": TRAINING_CONFIG.enable_long,
            "enable_short": TRAINING_CONFIG.enable_short,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V2: Pure ML with fixed-horizon labels")
    parser.add_argument("--data-path", default="data/processed/mes_bars.h5")
    parser.add_argument("--output-dir", default="models/saved_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save", action="store_true", help="Don't save models (dry run)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TRAIN V2: PURE ML STRATEGY")
    print("=" * 60)
    print("\nApproach:")
    print("  ✓ Fixed-horizon market labels (no TP/SL simulation)")
    print("  ✓ Pure ML entries (no heuristic gates)")
    print("  ✓ Score-based frequency control (quantile + daily budget)")
    print("  ✓ Walk-forward with lockbox")
    print("\n" + "=" * 60)

    # Load data
    bars = load_bars(args.data_path)
    print(f"\nLoaded {len(bars):,} bars from {args.data_path}")

    # Train models
    results = train_models_v2(bars, seed=args.seed)

    # Prepare metadata
    risk_dict = asdict(RISK_CONFIG)
    training_dict = asdict(TRAINING_CONFIG)

    # Convert time objects to strings
    if "session_start" in risk_dict and isinstance(risk_dict["session_start"], time):
        risk_dict["session_start"] = risk_dict["session_start"].isoformat()
    if "session_end" in risk_dict and isinstance(risk_dict["session_end"], time):
        risk_dict["session_end"] = risk_dict["session_end"].isoformat()

    metadata = {
        "created": pd.Timestamp.utcnow().isoformat(),
        "version": "v2",
        "approach": "pure_ml_fixed_horizon_labels",
        "feature_cols": results["feature_cols"],
        "config": {
            "risk": risk_dict,
            "training": training_dict,
        },
        "labels_v2": results["labels_v2"],
        "policy_v2": results["policy_v2"],
        "metrics": results["metrics"],
        "lockbox_metrics": results["lockbox_metrics"],
        "windows": {
            "train": results["windows"]["train"],
            "val": results["windows"]["val"],
            "test": results["windows"]["test"],
            "lockbox": results["windows"]["lockbox"],
            "embargo_bars": results["windows"]["embargo_bars"],
        },
        "window_timestamps": {
            "train": results["windows"]["train_timestamps"],
            "val": results["windows"]["val_timestamps"],
            "test": results["windows"]["test_timestamps"],
            "lockbox": results["windows"]["lockbox_timestamps"],
        },
        "data": {
            "bars": len(bars),
            "start_timestamp": bars["timestamp"].iloc[0].isoformat() if len(bars) else None,
            "end_timestamp": bars["timestamp"].iloc[-1].isoformat() if len(bars) else None,
        },
    }

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nPolicy V2:")
    print(json.dumps(results["policy_v2"], indent=2))
    print("\nScore Threshold:")
    print(f"  {results['score_threshold']:.4f} (quantile {TRAINING_CONFIG.score_quantile})")

    if args.no_save:
        print("\n--no-save flag: models not saved")
        return

    # Save models
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(results["long_model"], output_dir / "model_long.joblib")
    if results["short_model"] is not None:
        joblib.dump(results["short_model"], output_dir / "model_short.joblib")

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved models to {output_dir}")


if __name__ == "__main__":
    main()
