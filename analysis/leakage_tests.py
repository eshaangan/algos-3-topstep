"""
Leakage tripwire tests for the tiny MLP pipeline.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from core.simple_config import NN_CONFIG, RISK_CONFIG
from data.clean_bars import clean_bars
from features.engineer import add_features, get_recommended_features
from features.labels_v2 import make_fixed_horizon_labels
from models.validate_splits import validate_split_integrity


def _to_chicago_series(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts, utc=True)
    return ts.dt.tz_convert("America/Chicago")


def load_bars(h5_path: str) -> pd.DataFrame:
    with pd.HDFStore(h5_path, "r") as store:
        bars = store["bars_5min"].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars, _ = clean_bars(bars, tick_size=RISK_CONFIG.tick_size, verbose=False)
    return bars


def build_dataset(bars_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    features_df = add_features(bars_df, verbose=False)
    features_df = features_df.reset_index().rename(columns={"index": "idx"})
    feature_cols = get_recommended_features()

    labels_df = make_fixed_horizon_labels(
        bars_df,
        horizon_bars=NN_CONFIG.horizon_bars,
        threshold_ticks=NN_CONFIG.threshold_ticks,
        tick_size=RISK_CONFIG.tick_size,
    )

    bar_times = _to_chicago_series(bars_df["timestamp"])
    rth_mask = (bar_times.dt.time >= RISK_CONFIG.session_start) & (bar_times.dt.time < RISK_CONFIG.session_end)
    rth_df = pd.DataFrame({"idx": np.arange(len(bars_df)), "is_rth": rth_mask.values})
    labels_df = labels_df.merge(rth_df, on="idx", how="left")
    labels_df = labels_df[labels_df["is_rth"]].drop(columns=["is_rth"]).copy()

    merged = labels_df.merge(features_df, on="idx", how="inner")
    merged = merged.dropna(subset=feature_cols + ["y_long", "y_short"])
    return merged, feature_cols


def build_windows(n_bars: int) -> Dict[str, Tuple[int, int]]:
    embargo = max(NN_CONFIG.embargo_bars, NN_CONFIG.horizon_bars + NN_CONFIG.feature_lookback)
    start = NN_CONFIG.feature_lookback
    usable = n_bars - start
    train_end = int(start + usable * NN_CONFIG.train_fraction)
    val_start = train_end + embargo
    val_end = int(val_start + usable * NN_CONFIG.val_fraction)
    test_start = val_end + embargo
    test_end = n_bars

    return {
        "train": (start, train_end),
        "val": (val_start, val_end),
        "test": (test_start, test_end),
        "embargo": embargo,
    }


def _fit_lr(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=200, solver="lbfgs")
    model.fit(X_train, y_train.astype(int))
    return model


def _auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, prob))


def _edge(y_true: np.ndarray, prob: np.ndarray) -> float:
    signed = (2 * y_true.astype(float) - 1.0)
    return float(np.mean((prob - 0.5) * signed))


def _prepare_split(
    merged: pd.DataFrame,
    feature_cols: List[str],
    window: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    start, end = window
    df = merged[(merged["idx"] >= start) & (merged["idx"] < end)].copy()
    X = df[feature_cols].values.astype(np.float32)
    y = df["y_long"].values.astype(np.int64)
    return X, y


def run_tripwire_tests(h5_path: str) -> None:
    bars = load_bars(h5_path)
    merged, feature_cols = build_dataset(bars)

    windows = build_windows(len(bars))
    embargo = windows["embargo"]
    validate_split_integrity(
        bars,
        windows["train"],
        windows["val"],
        windows["test"],
        purge_bars=embargo,
    )

    X_train, y_train = _prepare_split(merged, feature_cols, windows["train"])
    X_val, y_val = _prepare_split(merged, feature_cols, windows["val"])
    X_test, y_test = _prepare_split(merged, feature_cols, windows["test"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("\n[Tripwire A] Label shuffle test")
    shuffled = y_train.copy()
    rng = np.random.default_rng(42)
    rng.shuffle(shuffled)
    model = _fit_lr(X_train, shuffled)
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    val_auc = _auc(y_val, val_prob)
    test_auc = _auc(y_test, test_prob)
    val_edge = _edge(y_val, val_prob)
    test_edge = _edge(y_test, test_prob)
    print(f"  Val AUC:  {val_auc:.3f}")
    print(f"  Test AUC: {test_auc:.3f}")
    print(f"  Val edge:  {val_edge:.4f}")
    print(f"  Test edge: {test_edge:.4f}")
    if val_auc > 0.60 or test_auc > 0.60:
        raise ValueError("Label shuffle test failed: AUC too high (possible leakage).")
    if abs(val_edge) > 0.02 or abs(test_edge) > 0.02:
        raise ValueError("Label shuffle test failed: edge too large (possible leakage).")

    print("\n[Tripwire B] Time-shift test (+1 bar features)")
    shifted = merged.copy()
    shifted[feature_cols] = shifted[feature_cols].shift(-1)
    shifted = shifted.dropna(subset=feature_cols + ["y_long"])
    X_train_s, y_train_s = _prepare_split(shifted, feature_cols, windows["train"])
    X_val_s, y_val_s = _prepare_split(shifted, feature_cols, windows["val"])
    X_test_s, y_test_s = _prepare_split(shifted, feature_cols, windows["test"])

    scaler_s = StandardScaler()
    X_train_s = scaler_s.fit_transform(X_train_s)
    X_val_s = scaler_s.transform(X_val_s)
    X_test_s = scaler_s.transform(X_test_s)

    model_s = _fit_lr(X_train_s, y_train_s)
    val_prob_s = model_s.predict_proba(X_val_s)[:, 1]
    test_prob_s = model_s.predict_proba(X_test_s)[:, 1]
    val_auc_s = _auc(y_val_s, val_prob_s)
    test_auc_s = _auc(y_test_s, test_prob_s)
    val_edge_s = _edge(y_val_s, val_prob_s)
    test_edge_s = _edge(y_test_s, test_prob_s)
    print(f"  Val AUC:  {val_auc_s:.3f}")
    print(f"  Test AUC: {test_auc_s:.3f}")
    print(f"  Val edge:  {val_edge_s:.4f}")
    print(f"  Test edge: {test_edge_s:.4f}")
    if val_auc_s > 0.60 or test_auc_s > 0.60:
        raise ValueError("Time-shift test failed: AUC too high (possible leakage).")
    if abs(val_edge_s) > 0.02 or abs(test_edge_s) > 0.02:
        raise ValueError("Time-shift test failed: edge too large (possible leakage).")

    print("\n[Tripwire C] Split integrity: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage tripwire tests")
    parser.add_argument("--data-path", default="data/processed/mes_bars.h5")
    args = parser.parse_args()
    run_tripwire_tests(args.data_path)


if __name__ == "__main__":
    main()
