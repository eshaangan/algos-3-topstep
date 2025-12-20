"""
Train ML classifiers for long and short trade outcomes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import time
from pathlib import Path
from typing import Dict, List, Tuple

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
from features.engineer import add_features, select_features
from features.labels import create_labels


def load_bars(h5_path: str) -> pd.DataFrame:
    with pd.HDFStore(h5_path, "r") as store:
        bars = store["bars_5min"].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars, _ = clean_bars(bars, tick_size=RISK_CONFIG.tick_size, verbose=True)
    return bars


def _prepare_dataset(bars_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    features_df = add_features(bars_df)
    labels_df = create_labels(
        bars_df,
        lookback=TRAINING_CONFIG.lookback_bars,
        stop_ticks=TRAINING_CONFIG.stop_loss_ticks,
        target_multiplier=TRAINING_CONFIG.target_multiplier,
        max_hold_bars=TRAINING_CONFIG.max_hold_bars,
        tick_size=RISK_CONFIG.tick_size,
        tick_value=RISK_CONFIG.tick_value,
    )

    features_df = features_df.reset_index().rename(columns={"index": "idx"})
    feature_cols = select_features()

    dataset = labels_df.merge(features_df, on="idx", how="inner")
    dataset = dataset.dropna(subset=feature_cols).reset_index(drop=True)

    return dataset, feature_cols


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
    dataset, feature_cols = _prepare_dataset(bars_df)

    # Purge/embargo needs to cover both feature lookback and label horizon.
    purge_bars = TRAINING_CONFIG.lookback_bars + TRAINING_CONFIG.max_hold_bars + 1

    long_df = dataset[dataset["label_long"] != 2].copy()
    short_df = dataset[dataset["label_short"] != 2].copy()

    long_train, long_val, long_test = _walk_forward_split(
        long_df,
        TRAINING_CONFIG.train_fraction,
        TRAINING_CONFIG.val_fraction,
        TRAINING_CONFIG.test_fraction,
        purge_bars,
    )

    short_train, short_val, short_test = _walk_forward_split(
        short_df,
        TRAINING_CONFIG.train_fraction,
        TRAINING_CONFIG.val_fraction,
        TRAINING_CONFIG.test_fraction,
        purge_bars,
    )

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

    # Trade-identical evaluation uses contiguous BAR splits (not sample rows).
    train_w, val_w, test_w = _walk_forward_split_bars(
        bars_df,
        TRAINING_CONFIG.train_fraction,
        TRAINING_CONFIG.val_fraction,
        TRAINING_CONFIG.test_fraction,
        purge_bars,
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
        "dataset_rows": len(dataset),
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
    from backtesting.backtest import run_backtest

    out: Dict[str, Dict[str, object]] = {}
    for name, (start, end) in windows.items():
        if end <= start:
            out[name] = {"summary": {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "net_pnl": 0.0}}
            continue
        result = run_backtest(
            bars_df,
            long_model,
            short_model,
            feature_cols,
            start_idx=start,
            end_idx=end,
            min_probability_long=float(policy["min_probability_long"]),
            min_probability_short=float(policy["min_probability_short"]),
            enable_long=bool(policy["enable_long"]),
            enable_short=bool(policy["enable_short"]),
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
    from backtesting.backtest import run_backtest

    enable_long = bool(TRAINING_CONFIG.enable_long)

    # Default: shorts off unless explicitly enabled in config.
    enable_short = bool(TRAINING_CONFIG.enable_short)

    start, end = validation_window
    start = int(start)
    end = int(end)
    if end - start < 2:
        return {
            "enable_long": enable_long,
            "enable_short": enable_short,
            "min_probability_long": TRAINING_CONFIG.min_probability_long,
            "min_probability_short": TRAINING_CONFIG.min_probability_short,
        }

    # Objective: maximize net P&L, break ties by lower max DD, require minimum trades.
    min_trades = 15
    long_grid = np.round(np.arange(0.50, 0.91, 0.05), 2)
    short_grid = np.round(np.arange(0.50, 0.91, 0.05), 2)

    best = None

    for long_thr in long_grid:
        if not enable_short:
            short_thr = 1.0  # effectively disabled
            res = run_backtest(
                bars_df,
                long_model,
                short_model,
                feature_cols,
                start_idx=start,
                end_idx=end,
                min_probability_long=float(long_thr),
                min_probability_short=float(short_thr),
                enable_long=enable_long,
                enable_short=False,
            )["summary"]
            trades = int(res.get("trades", 0))
            if trades < min_trades:
                continue
            score = (float(res.get("net_pnl", 0.0)), -float(res.get("max_drawdown", 0.0)))
            cand = (score, long_thr, short_thr, res)
            if best is None or cand[0] > best[0]:
                best = cand
            continue

        for short_thr in short_grid:
            res = run_backtest(
                bars_df,
                long_model,
                short_model,
                feature_cols,
                start_idx=start,
                end_idx=end,
                min_probability_long=float(long_thr),
                min_probability_short=float(short_thr),
                enable_long=enable_long,
                enable_short=True,
            )["summary"]
            trades = int(res.get("trades", 0))
            if trades < min_trades:
                continue
            score = (float(res.get("net_pnl", 0.0)), -float(res.get("max_drawdown", 0.0)))
            cand = (score, long_thr, short_thr, res)
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return {
            "enable_long": enable_long,
            "enable_short": enable_short,
            "min_probability_long": TRAINING_CONFIG.min_probability_long,
            "min_probability_short": TRAINING_CONFIG.min_probability_short,
        }

    _, long_thr, short_thr, _ = best
    return {
        "enable_long": enable_long,
        "enable_short": enable_short,
        "min_probability_long": float(long_thr),
        "min_probability_short": float(short_thr),
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

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
