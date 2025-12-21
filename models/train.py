"""
Train ML classifiers for long and short trade outcomes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path if not already installed as package
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
from features.labels import create_per_bar_trade_labels, create_sequential_trade_labels
from models.validate_splits import run_all_diagnostics


def load_bars(h5_path: str) -> pd.DataFrame:
    with pd.HDFStore(h5_path, "r") as store:
        bars = store["bars_5min"].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars = clean_bars(bars, tick_size=RISK_CONFIG.tick_size, verbose=True)
    return bars


def _prepare_datasets(bars_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    features_df = add_features(bars_df)
    features_df = features_df.reset_index().rename(columns={"index": "idx"})

    # Feature selection based on config
    feature_selection_mode = getattr(TRAINING_CONFIG, "feature_selection_mode", "all")
    if feature_selection_mode == "recommended":
        feature_cols = get_recommended_features()
        print(f"\n✓ Using recommended {len(feature_cols)} features (top ~75-80% importance)")
    elif feature_selection_mode == "all":
        feature_cols = select_features()
        print(f"\n✓ Using all {len(feature_cols)} features")
    else:  # "auto" mode
        # Will be handled by two-stage training in train_models()
        feature_cols = select_features()
        print(f"\n✓ Starting with all {len(feature_cols)} features (auto-selection enabled)")

    label_mode = (getattr(TRAINING_CONFIG, "label_mode", "sequential") or "sequential").strip().lower()
    if label_mode not in {"sequential", "per_bar"}:
        raise ValueError(f"Unsupported label_mode={label_mode!r} (expected 'sequential' or 'per_bar')")

    label_fn = create_sequential_trade_labels if label_mode == "sequential" else create_per_bar_trade_labels

    # Generate trade labels per direction.
    long_labels = label_fn(
        bars_df,
        direction="long",
        lookback=TRAINING_CONFIG.lookback_bars,
        stop_ticks=TRAINING_CONFIG.stop_loss_ticks,
        target_multiplier=TRAINING_CONFIG.target_multiplier,
        max_hold_bars=TRAINING_CONFIG.max_hold_bars,
        tick_size=RISK_CONFIG.tick_size,
        tick_value=RISK_CONFIG.tick_value,
        session_start=RISK_CONFIG.session_start,
        session_end=RISK_CONFIG.session_end,
    ).rename(columns={"label": "label_long"})

    short_labels = label_fn(
        bars_df,
        direction="short",
        lookback=TRAINING_CONFIG.lookback_bars,
        stop_ticks=TRAINING_CONFIG.stop_loss_ticks,
        target_multiplier=TRAINING_CONFIG.target_multiplier,
        max_hold_bars=TRAINING_CONFIG.max_hold_bars,
        tick_size=RISK_CONFIG.tick_size,
        tick_value=RISK_CONFIG.tick_value,
        session_start=RISK_CONFIG.session_start,
        session_end=RISK_CONFIG.session_end,
    ).rename(columns={"label": "label_short"})

    long_df = long_labels.merge(features_df, on="idx", how="inner").dropna(subset=feature_cols).reset_index(drop=True)
    short_df = short_labels.merge(features_df, on="idx", how="inner").dropna(subset=feature_cols).reset_index(drop=True)

    return long_df, short_df, feature_cols


def _walk_forward_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    purge_bars: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train/val/test fractions must sum to 1.0")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_stop = max(0, train_end - purge_bars)
    val_start = min(n, train_end + purge_bars)
    val_stop = max(val_start, val_end - purge_bars)
    test_start = min(n, val_end + purge_bars)

    train = df.iloc[:train_stop].copy()
    val = df.iloc[val_start:val_stop].copy()
    test = df.iloc[test_start:].copy()

    return train, val, test


def _trade_stats(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    trade_mask = y_prob >= threshold
    trades = y_true[trade_mask]
    total = len(trades)
    wins = int((trades == 1).sum())
    losses = int((trades == 0).sum())

    if total == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "max_drawdown_r": 0.0,
        }

    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

    gross_win = wins * TRAINING_CONFIG.target_multiplier
    gross_loss = losses * 1.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    pnl_r = np.where(trades == 1, TRAINING_CONFIG.target_multiplier, -1.0)
    equity = np.cumsum(pnl_r)
    peaks = np.maximum.accumulate(equity)
    drawdowns = peaks - equity
    max_dd_r = float(np.max(drawdowns)) if len(drawdowns) else 0.0

    expectancy_r = float(pnl_r.mean()) if len(pnl_r) else 0.0

    return {
        "trades": int(total),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "expectancy_r": float(expectancy_r),
        "max_drawdown_r": float(max_dd_r),
    }


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn)})
    return metrics


def _train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    trade_threshold: float,
    seed: int,
) -> Tuple[RandomForestClassifier, Dict[str, Dict[str, float]]]:
    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values.astype(int)
    X_val = val_df[feature_cols].values
    y_val = val_df[label_col].values.astype(int)
    X_test = test_df[feature_cols].values
    y_test = test_df[label_col].values.astype(int)

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

    # Get predictions for all splits to detect overfitting
    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "training": {
            **_classification_metrics(y_train, train_prob),
            **_trade_stats(y_train, train_prob, trade_threshold),
        },
        "validation": {
            **_classification_metrics(y_val, val_prob),
            **_trade_stats(y_val, val_prob, trade_threshold),
        },
        "test": {
            **_classification_metrics(y_test, test_prob),
            **_trade_stats(y_test, test_prob, trade_threshold),
        },
    }

    return model, metrics


def train_models(
    bars_df: pd.DataFrame,
    seed: int = 42,
) -> Dict[str, object]:
    long_df, short_df, feature_cols = _prepare_datasets(bars_df)

    # Purge/embargo needs to cover both feature lookback and label horizon.
    purge_bars = TRAINING_CONFIG.lookback_bars + TRAINING_CONFIG.max_hold_bars + 1

    # Bar-based windows ensure labels never peek across train/val/test boundaries.
    train_w, val_w, test_w = _walk_forward_split_bars(
        bars_df,
        TRAINING_CONFIG.train_fraction,
        TRAINING_CONFIG.val_fraction,
        TRAINING_CONFIG.test_fraction,
        purge_bars,
    )

    # Run comprehensive data leakage diagnostics
    print("\n" + "="*60)
    print("RUNNING DATA LEAKAGE DIAGNOSTICS")
    print("="*60)
    try:
        features_df = add_features(bars_df, verbose=False)
        features_df = features_df.reset_index().rename(columns={"index": "idx"})

        run_all_diagnostics(
            bars_df=bars_df,
            long_df=long_df,
            short_df=short_df,
            features_df=features_df,
            feature_cols=feature_cols,
            windows={"training": train_w, "validation": val_w, "test": test_w},
            purge_bars=purge_bars,
            max_hold_bars=TRAINING_CONFIG.max_hold_bars,
        )
    except Exception as e:
        print(f"\n⚠️  DATA LEAKAGE DIAGNOSTIC FAILED: {e}")
        print("Please investigate and fix before proceeding with training.")
        raise

    def _slice_by_window(df: pd.DataFrame, window: tuple[int, int]) -> pd.DataFrame:
        start, end = (int(window[0]), int(window[1]))
        out = df[(df["idx"] >= start) & (df["idx"] < end)].copy()
        return out.reset_index(drop=True)

    long_train = _slice_by_window(long_df, train_w)
    long_val = _slice_by_window(long_df, val_w)
    long_test = _slice_by_window(long_df, test_w)

    short_train = _slice_by_window(short_df, train_w)
    short_val = _slice_by_window(short_df, val_w)
    short_test = _slice_by_window(short_df, test_w)

    if min(len(long_train), len(long_val), len(long_test)) == 0:
        raise ValueError("Not enough long samples after split; adjust fractions or dataset size")
    if min(len(short_train), len(short_val), len(short_test)) == 0:
        raise ValueError("Not enough short samples after split; adjust fractions or dataset size")

    long_model, long_metrics = _train_model(
        long_train,
        long_val,
        long_test,
        feature_cols,
        "label_long",
        TRAINING_CONFIG.min_probability_long,
        seed,
    )

    short_model, short_metrics = _train_model(
        short_train,
        short_val,
        short_test,
        feature_cols,
        "label_short",
        TRAINING_CONFIG.min_probability_short,
        seed + 1,
    )

    # Tune thresholds on validation window using the real backtest logic.
    tuned_policy = _tune_policy_with_backtest(
        bars_df,
        long_model,
        short_model,
        feature_cols,
        val_w,
    )

    backtest_eval = _evaluate_splits_with_backtest(
        bars_df,
        long_model,
        short_model,
        feature_cols,
        {"training": train_w, "validation": val_w, "test": test_w},
        tuned_policy,
    )

    return {
        "long_model": long_model,
        "short_model": short_model,
        "feature_cols": feature_cols,
        "metrics": {"long": long_metrics, "short": short_metrics},
        "backtest_metrics": backtest_eval,
        "policy": tuned_policy,
        "windows": {"training": train_w, "validation": val_w, "test": test_w},
        "dataset_rows": int(len(long_df) + len(short_df)),
        "label_rows": {"long": len(long_df), "short": len(short_df)},
    }


def _walk_forward_split_bars(
    bars_df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    purge_bars: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    Walk-forward split on contiguous bars with an embargo gap between segments.

    Returns:
        (train_window, val_window, test_window) where each is (start_idx, end_idx).
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train/val/test fractions must sum to 1.0")
    n = len(bars_df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_stop = max(0, train_end - purge_bars)
    val_start = min(n, train_end + purge_bars)
    val_stop = max(val_start, val_end - purge_bars)
    test_start = min(n, val_end + purge_bars)

    return (0, train_stop), (val_start, val_stop), (test_start, n)


def _evaluate_splits_with_backtest(
    bars_df: pd.DataFrame,
    long_model: object,
    short_model: object,
    feature_cols: List[str],
    windows: Dict[str, tuple[int, int]],
    policy: Dict[str, object],
) -> Dict[str, Dict[str, object]]:
    from backtesting.backtest import compute_probabilities, run_backtest

    out: Dict[str, Dict[str, object]] = {}
    prob_df = compute_probabilities(bars_df, long_model, short_model, feature_cols)
    for name, (start, end) in windows.items():
        if end <= start:
            out[name] = {"summary": {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_pnl": 0.0}}
            continue
        result = run_backtest(
            bars_df,
            long_model,
            short_model,
            feature_cols,
            prob_df=prob_df,
            start_idx=start,
            end_idx=end,
            min_probability_long=float(policy["min_probability_long"]),
            min_probability_short=float(policy["min_probability_short"]),
            enable_long=bool(policy["enable_long"]),
            enable_short=bool(policy["enable_short"]),
            blocked_hours=policy.get("blocked_hours"),
            allowed_hours=policy.get("allowed_hours"),
            exclude_lunch=policy.get("exclude_lunch"),
            require_trend_long=policy.get("require_trend_long"),
            require_trend_short=policy.get("require_trend_short"),
            min_atr_ticks=policy.get("min_atr_ticks"),
            max_atr_ticks=policy.get("max_atr_ticks"),
        )
        out[name] = result
    return out


def _tune_policy_with_backtest(
    bars_df: pd.DataFrame,
    long_model: object,
    short_model: object,
    feature_cols: List[str],
    validation_window: tuple[int, int],
) -> Dict[str, object]:
    """
    Choose thresholds using the actual backtest engine on the validation window.

    This directly optimizes the objective we care about (net P&L / DD) under
    real execution constraints (session, max hold, one-position-at-a-time).
    """
    from backtesting.backtest import compute_probabilities, run_backtest

    enable_long = bool(TRAINING_CONFIG.enable_long)

    # Long-only policy search (shorts are off by default).
    enable_short_candidates = [False]

    start, end = validation_window
    start = int(start)
    end = int(end)
    if end - start < 2:
        return {
            "enable_long": enable_long,
            "enable_short": False,
            "min_probability_long": TRAINING_CONFIG.min_probability_long,
            "min_probability_short": TRAINING_CONFIG.min_probability_short,
        }

    # Objective: find a profitable validation policy under REAL execution.
    min_trades = 20
    min_profit_factor = 1.00
    min_win_rate = 0.50
    max_drawdown = float(TRAINING_CONFIG.max_drawdown)
    min_trades_half = 5

    prob_df = compute_probabilities(bars_df, long_model, short_model, feature_cols)

    # Adaptive threshold grid based on validation probability distribution.
    val_probs = prob_df.iloc[start:end]["long_prob"].dropna().astype(float)
    if val_probs.empty:
        return {
            "enable_long": False,
            "enable_short": False,
            "min_probability_long": TRAINING_CONFIG.min_probability_long,
            "min_probability_short": TRAINING_CONFIG.min_probability_short,
        }

    quantiles = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    long_grid = np.unique(np.round(val_probs.quantile(quantiles).values, 3))
    # Keep within [0, 1] and add a couple of fixed points for stability.
    long_grid = np.unique(np.clip(np.concatenate([long_grid, [0.5, 0.65]]), 0.0, 1.0))
    short_grid = np.array([0.50, 0.55, 0.60, 0.65])

    # Candidate gating configurations (keep small + interpretable).
    gate_candidates: List[Dict[str, object]] = []
    atr_ranges = [(None, None), (6.0, 40.0)]
    for blocked in ([], [14], [13, 14]):
        for exclude_lunch in (False, True):
            for trend in (False, True):
                for min_atr, max_atr in atr_ranges:
                    gate_candidates.append(
                        {
                            "blocked_hours": blocked,
                            "allowed_hours": None,
                            "exclude_lunch": exclude_lunch,
                            "require_trend_long": trend,
                            "require_trend_short": False,
                            "min_atr_ticks": min_atr,
                            "max_atr_ticks": max_atr,
                        }
                    )

    # Use a small stability check: require that the policy is not only profitable in aggregate
    # on validation, but also not clearly losing in either half of the validation window.
    mid = start + (end - start) // 2

    def _passes_stability(half: Dict[str, object]) -> bool:
        trades_h = int(half.get("trades", 0) or 0)
        if trades_h < min_trades_half:
            return True  # too few trades to judge; don't over-penalize
        return float(half.get("net_pnl", 0.0) or 0.0) > 0 and float(half.get("profit_factor", 0.0) or 0.0) > 1.0

    best_profitable = None  # (score, long_thr, short_thr, trades, pf, gates, enable_short)

    for gates in gate_candidates:
        for enable_short in enable_short_candidates:
            for long_thr in long_grid:
                if not enable_short:
                    short_thr = 1.0  # effectively disabled
                    full = run_backtest(
                        bars_df,
                        long_model,
                        short_model,
                        feature_cols,
                        prob_df=prob_df,
                        start_idx=start,
                        end_idx=end,
                        min_probability_long=float(long_thr),
                        min_probability_short=float(short_thr),
                        enable_long=enable_long,
                        enable_short=False,
                        blocked_hours=gates.get("blocked_hours"),
                        allowed_hours=gates.get("allowed_hours"),
                        exclude_lunch=gates.get("exclude_lunch"),
                        require_trend_long=gates.get("require_trend_long"),
                        require_trend_short=False,
                        min_atr_ticks=gates.get("min_atr_ticks"),
                        max_atr_ticks=gates.get("max_atr_ticks"),
                    )["summary"]

                    first_half = run_backtest(
                        bars_df,
                        long_model,
                        short_model,
                        feature_cols,
                        prob_df=prob_df,
                        start_idx=start,
                        end_idx=mid,
                        min_probability_long=float(long_thr),
                        min_probability_short=float(short_thr),
                        enable_long=enable_long,
                        enable_short=False,
                        blocked_hours=gates.get("blocked_hours"),
                        allowed_hours=gates.get("allowed_hours"),
                        exclude_lunch=gates.get("exclude_lunch"),
                        require_trend_long=gates.get("require_trend_long"),
                        require_trend_short=False,
                        min_atr_ticks=gates.get("min_atr_ticks"),
                        max_atr_ticks=gates.get("max_atr_ticks"),
                    )["summary"]
                    second_half = run_backtest(
                        bars_df,
                        long_model,
                        short_model,
                        feature_cols,
                        prob_df=prob_df,
                        start_idx=mid,
                        end_idx=end,
                        min_probability_long=float(long_thr),
                        min_probability_short=float(short_thr),
                        enable_long=enable_long,
                        enable_short=False,
                        blocked_hours=gates.get("blocked_hours"),
                        allowed_hours=gates.get("allowed_hours"),
                        exclude_lunch=gates.get("exclude_lunch"),
                        require_trend_long=gates.get("require_trend_long"),
                        require_trend_short=False,
                        min_atr_ticks=gates.get("min_atr_ticks"),
                        max_atr_ticks=gates.get("max_atr_ticks"),
                    )["summary"]

                    trades = int(full.get("trades", 0) or 0)
                    win_rate = float(full.get("win_rate", 0.0) or 0.0)
                    profit_factor = float(full.get("profit_factor", 0.0) or 0.0)
                    net_pnl = float(full.get("net_pnl", 0.0) or 0.0)
                    dd = float(full.get("max_drawdown", 0.0) or 0.0)

                    if (
                        trades >= min_trades
                        and net_pnl > 0
                        and win_rate >= min_win_rate
                        and profit_factor >= min_profit_factor
                        and dd <= max_drawdown
                        and _passes_stability(first_half)
                        and _passes_stability(second_half)
                    ):
                        score = net_pnl - 0.25 * dd
                        cand = (score, long_thr, short_thr, trades, profit_factor, gates, False)
                        if best_profitable is None or cand[0] > best_profitable[0]:
                            best_profitable = cand
                    continue

                for short_thr in short_grid:
                    full = run_backtest(
                        bars_df,
                        long_model,
                        short_model,
                        feature_cols,
                        prob_df=prob_df,
                        start_idx=start,
                        end_idx=end,
                        min_probability_long=float(long_thr),
                        min_probability_short=float(short_thr),
                        enable_long=enable_long,
                        enable_short=True,
                        blocked_hours=gates.get("blocked_hours"),
                        allowed_hours=gates.get("allowed_hours"),
                        exclude_lunch=gates.get("exclude_lunch"),
                        require_trend_long=gates.get("require_trend_long"),
                        require_trend_short=gates.get("require_trend_short"),
                        min_atr_ticks=gates.get("min_atr_ticks"),
                        max_atr_ticks=gates.get("max_atr_ticks"),
                    )["summary"]

                    first_half = run_backtest(
                        bars_df,
                        long_model,
                        short_model,
                        feature_cols,
                        prob_df=prob_df,
                        start_idx=start,
                        end_idx=mid,
                        min_probability_long=float(long_thr),
                        min_probability_short=float(short_thr),
                        enable_long=enable_long,
                        enable_short=True,
                        blocked_hours=gates.get("blocked_hours"),
                        allowed_hours=gates.get("allowed_hours"),
                        exclude_lunch=gates.get("exclude_lunch"),
                        require_trend_long=gates.get("require_trend_long"),
                        require_trend_short=gates.get("require_trend_short"),
                        min_atr_ticks=gates.get("min_atr_ticks"),
                        max_atr_ticks=gates.get("max_atr_ticks"),
                    )["summary"]
                    second_half = run_backtest(
                        bars_df,
                        long_model,
                        short_model,
                        feature_cols,
                        prob_df=prob_df,
                        start_idx=mid,
                        end_idx=end,
                        min_probability_long=float(long_thr),
                        min_probability_short=float(short_thr),
                        enable_long=enable_long,
                        enable_short=True,
                        blocked_hours=gates.get("blocked_hours"),
                        allowed_hours=gates.get("allowed_hours"),
                        exclude_lunch=gates.get("exclude_lunch"),
                        require_trend_long=gates.get("require_trend_long"),
                        require_trend_short=gates.get("require_trend_short"),
                        min_atr_ticks=gates.get("min_atr_ticks"),
                        max_atr_ticks=gates.get("max_atr_ticks"),
                    )["summary"]

                    trades = int(full.get("trades", 0) or 0)
                    win_rate = float(full.get("win_rate", 0.0) or 0.0)
                    profit_factor = float(full.get("profit_factor", 0.0) or 0.0)
                    net_pnl = float(full.get("net_pnl", 0.0) or 0.0)
                    dd = float(full.get("max_drawdown", 0.0) or 0.0)

                    if (
                        trades >= min_trades
                        and net_pnl > 0
                        and win_rate >= min_win_rate
                        and profit_factor >= min_profit_factor
                        and dd <= max_drawdown
                        and _passes_stability(first_half)
                        and _passes_stability(second_half)
                    ):
                        score = net_pnl - 0.25 * dd
                        cand = (score, long_thr, short_thr, trades, profit_factor, gates, True)
                        if best_profitable is None or cand[0] > best_profitable[0]:
                            best_profitable = cand

    if best_profitable is None:
        return {
            "enable_long": False,
            "enable_short": False,
            "min_probability_long": TRAINING_CONFIG.min_probability_long,
            "min_probability_short": TRAINING_CONFIG.min_probability_short,
        }

    _, long_thr, short_thr, _, _, gates, enable_short = best_profitable
    enable_short = False
    if gates:
        gates = dict(gates)
        gates["require_trend_short"] = False
    return {
        "enable_long": enable_long,
        "enable_short": bool(enable_short),
        "min_probability_long": float(long_thr),
        "min_probability_short": float(short_thr),
        **(gates or {}),
    }


def _gate_trade_stats(metrics: Dict[str, Dict[str, float]]) -> Dict[str, bool]:
    test_stats = metrics.get("test", {})
    if not test_stats:
        return {"passed": False}

    win_rate = test_stats.get("win_rate", 0.0)
    profit_factor = test_stats.get("profit_factor", 0.0)
    max_dd_r = test_stats.get("max_drawdown_r", 0.0)
    max_dd_usd = max_dd_r * RISK_CONFIG.fixed_risk_per_trade

    passed = (
        win_rate >= TRAINING_CONFIG.min_win_rate
        and profit_factor >= TRAINING_CONFIG.min_profit_factor
        and max_dd_usd <= TRAINING_CONFIG.max_drawdown
    )

    return {
        "passed": passed,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown_usd": max_dd_usd,
    }


def _gate_backtest_summary(summary: Dict[str, object]) -> Dict[str, object]:
    trades = int(summary.get("trades", 0) or 0)
    win_rate = float(summary.get("win_rate", 0.0) or 0.0)
    profit_factor = float(summary.get("profit_factor", 0.0) or 0.0)
    max_drawdown = float(summary.get("max_drawdown", 0.0) or 0.0)

    passed = (
        trades > 0
        and win_rate >= TRAINING_CONFIG.min_win_rate
        and profit_factor >= TRAINING_CONFIG.min_profit_factor
        and max_drawdown <= TRAINING_CONFIG.max_drawdown
    )

    return {
        "passed": passed,
        "trades": trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML classifiers for TopstepX strategy")
    parser.add_argument("--data-path", default="data/processed/mes_bars.h5")
    parser.add_argument("--output-dir", default="models/saved")
    parser.add_argument("--recent-bars", type=int, default=None)
    parser.add_argument("--recent-days", type=int, default=None)
    parser.add_argument("--label-mode", choices=["per_bar", "sequential"], default=None)
    parser.add_argument("--stop-loss-ticks", type=int, default=None)
    parser.add_argument("--target-multiplier", type=float, default=None)
    parser.add_argument("--max-hold-bars", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    if args.label_mode:
        TRAINING_CONFIG.label_mode = args.label_mode
    if args.stop_loss_ticks is not None:
        TRAINING_CONFIG.stop_loss_ticks = int(args.stop_loss_ticks)
    if args.target_multiplier is not None:
        TRAINING_CONFIG.target_multiplier = float(args.target_multiplier)
    if args.max_hold_bars is not None:
        TRAINING_CONFIG.max_hold_bars = int(args.max_hold_bars)

    bars = load_bars(args.data_path)

    if args.recent_days:
        cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=args.recent_days)
        bars = bars[bars["timestamp"] >= cutoff].reset_index(drop=True)

    if args.recent_bars:
        bars = bars.tail(args.recent_bars).reset_index(drop=True)

    results = train_models(bars, seed=args.seed)

    long_gate_trade = _gate_trade_stats(results["metrics"]["long"])
    short_gate_trade = _gate_trade_stats(results["metrics"]["short"])

    test_bt = (results.get("backtest_metrics", {}).get("test", {}) or {}).get("summary", {})
    long_gate_backtest = _gate_backtest_summary(test_bt if isinstance(test_bt, dict) else {})
    # If shorts are disabled, treat short gate as N/A rather than fail.
    short_gate_backtest = {
        "passed": None if not TRAINING_CONFIG.enable_short else False,
        "note": "shorts disabled" if not TRAINING_CONFIG.enable_short else "shorts enabled but not separately gated",
    }

    # Convert config to dict and serialize time objects to strings
    risk_dict = asdict(RISK_CONFIG)
    training_dict = asdict(TRAINING_CONFIG)
    
    # Convert time objects to strings for JSON serialization
    if "session_start" in risk_dict and isinstance(risk_dict["session_start"], time):
        risk_dict["session_start"] = risk_dict["session_start"].isoformat()
    if "session_end" in risk_dict and isinstance(risk_dict["session_end"], time):
        risk_dict["session_end"] = risk_dict["session_end"].isoformat()
    
    metadata = {
        "created": pd.Timestamp.utcnow().isoformat(),
        "feature_cols": results["feature_cols"],
        "config": {
            "risk": risk_dict,
            "training": training_dict,
        },
        "policy": results.get("policy", {}),
        "metrics": results["metrics"],
        "backtest_metrics": {
            "windows": results.get("windows", {}),
            "window_timestamps": {
                name: {
                    "start_timestamp": (
                        bars["timestamp"].iloc[int(w[0])].isoformat()
                        if w and len(bars) and 0 <= int(w[0]) < len(bars)
                        else None
                    ),
                    "end_timestamp": (
                        bars["timestamp"].iloc[min(int(w[1]) - 1, len(bars) - 1)].isoformat()
                        if w and len(bars) and int(w[1]) > 0
                        else None
                    ),
                }
                for name, w in (results.get("windows", {}) or {}).items()
            },
            "training": (results.get("backtest_metrics", {}).get("training", {}) or {}).get("summary", {}),
            "validation": (results.get("backtest_metrics", {}).get("validation", {}) or {}).get("summary", {}),
            "test": (results.get("backtest_metrics", {}).get("test", {}) or {}).get("summary", {}),
        },
        "gates": {
            "trade_stats": {"long": long_gate_trade, "short": short_gate_trade},
            "backtest_test": {"long": long_gate_backtest, "short": short_gate_backtest},
        },
        "data": {
            "bars": len(bars),
            "start_timestamp": bars["timestamp"].iloc[0].isoformat() if len(bars) else None,
            "end_timestamp": bars["timestamp"].iloc[-1].isoformat() if len(bars) else None,
            "dataset_rows": results["dataset_rows"],
            "label_rows": results["label_rows"],
        },
    }

    print("\nTraining complete.")
    print(json.dumps({"policy": results.get("policy", {})}, indent=2))
    print(json.dumps(metadata["gates"], indent=2))

    if args.no_save:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(results["long_model"], output_dir / "model_long.joblib")
    joblib.dump(results["short_model"], output_dir / "model_short.joblib")

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved models to {output_dir}")


if __name__ == "__main__":
    main()
