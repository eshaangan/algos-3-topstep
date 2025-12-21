"""
Train tiny MLP models on fixed-horizon market labels with walk-forward splits.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from dataclasses import asdict
from datetime import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.selection import resolve_score_quantile
from core.simple_config import NN_CONFIG, RISK_CONFIG, TRAINING_CONFIG
from data.clean_bars import clean_bars
from features.engineer import add_features, get_recommended_features, select_features
from features.labels_aligned import make_aligned_fixed_horizon_labels
from models.nn_model import TinyMLP


def _to_chicago_series(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts, utc=True)
    return ts.dt.tz_convert("America/Chicago")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_bars(h5_path: str, *, dataset_key: str = "bars_5min", max_bars: Optional[int] = None) -> pd.DataFrame:
    with pd.HDFStore(h5_path, "r") as store:
        if dataset_key not in store:
            raise KeyError(f"Dataset key {dataset_key!r} not found in H5.")
        bars = store[dataset_key].copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars = clean_bars(bars, tick_size=RISK_CONFIG.tick_size, verbose=True)
    if max_bars is not None:
        max_bars = int(max_bars)
        if max_bars <= 0:
            raise ValueError("max_bars must be > 0")
        bars = bars.tail(max_bars).reset_index(drop=True)
    return bars


def build_features(bars_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    features_df = add_features(bars_df, verbose=True)
    features_df = features_df.reset_index(drop=True)
    if "idx" not in features_df.columns:
        features_df.insert(0, "idx", np.arange(len(features_df)))

    feature_selection_mode = getattr(NN_CONFIG, "feature_selection_mode", "recommended")
    if feature_selection_mode == "all":
        feature_cols = select_features()
    else:
        feature_cols = get_recommended_features()

    return features_df, feature_cols


def build_labels(
    bars_df: pd.DataFrame,
    *,
    horizon_bars: int,
    threshold_ticks: int,
    tick_size: float,
    session_mode: str,
    session_start: time,
    session_end: time,
) -> pd.DataFrame:
    labels_df = make_aligned_fixed_horizon_labels(
        bars_df,
        horizon_bars=horizon_bars,
        threshold_ticks=threshold_ticks,
        tick_size=tick_size,
        entry_price_col="open",
        exit_price_col="close",
    )

    if session_mode.upper() == "RTH":
        bar_times = _to_chicago_series(bars_df["timestamp"])
        rth_mask = (bar_times.dt.time >= session_start) & (bar_times.dt.time < session_end)
        rth_df = pd.DataFrame({"idx": np.arange(len(bars_df)), "is_rth": rth_mask.values})

        labels_df = labels_df.merge(rth_df, on="idx", how="left")
        labels_df = labels_df[labels_df["is_rth"]].drop(columns=["is_rth"]).copy()

    return labels_df


def _embargo_bars(horizon_bars: int, feature_lookback: int, embargo_bars: int) -> int:
    min_embargo = horizon_bars + feature_lookback + 1
    if embargo_bars <= 0:
        return min_embargo
    return max(int(embargo_bars), min_embargo)


def build_walk_forward_windows(
    n_bars: int,
    *,
    feature_lookback: int,
    horizon_bars: int,
    embargo_bars: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    folds: int,
) -> Tuple[List[Dict[str, Tuple[int, int]]], int]:
    """
    Build walk-forward windows with an embargo gap between train/val/test.

    Embargo is enforced to be at least horizon_bars + feature_lookback + 1.
    """
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("train/val/test fractions must sum to 1.0")

    embargo = _embargo_bars(horizon_bars, feature_lookback, embargo_bars)
    train_start = feature_lookback
    usable = n_bars - train_start
    if usable <= 0:
        raise ValueError("Not enough bars for requested feature_lookback")

    # Account for embargo gaps: we need 2 embargo gaps (train-val and val-test)
    # So reduce usable space by 2*embargo before applying fractions
    effective_usable = usable - 2 * embargo
    if effective_usable <= 0:
        raise ValueError(f"Not enough bars after accounting for embargo gaps (usable={usable}, embargo={embargo})")

    train_span = int(effective_usable * train_fraction)
    val_span = int(effective_usable * val_fraction)
    test_span = int(effective_usable * test_fraction)

    if train_span <= 0 or val_span <= 0 or test_span <= 0:
        raise ValueError("Train/val/test spans must all be > 0")

    windows: List[Dict[str, Tuple[int, int]]] = []
    step = val_span if folds > 1 else 0

    for fold in range(max(1, folds)):
        train_end = train_start + train_span + fold * step
        val_start = train_end + embargo
        val_end = val_start + val_span
        test_start = val_end + embargo
        test_end = test_start + test_span

        if test_end > n_bars:
            break

        windows.append(
            {
                "train": (train_start, train_end),
                "val": (val_start, val_end),
                "test": (test_start, test_end),
            }
        )

    if not windows:
        raise ValueError("No valid folds could be generated with the current configuration")

    return windows, embargo


def slice_window(
    merged_df: pd.DataFrame,
    feature_cols: List[str],
    window: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    start, end = window
    df = merged_df[(merged_df["idx"] >= start) & (merged_df["idx"] < end)].copy()
    df = df.dropna(subset=feature_cols + ["y_long", "y_short"])

    X = df[feature_cols].values.astype(np.float32)
    y_long = df["y_long"].values.astype(np.float32)
    y_short = df["y_short"].values.astype(np.float32)

    return X, y_long, y_short


def _predict_logits(model: TinyMLP, X: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    logits = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            out = model(batch)
            logits.append(out.detach().cpu().numpy())
    if not logits:
        return np.array([], dtype=np.float32)
    return np.concatenate(logits, axis=0)


def _train_direction(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    input_dim: int,
    device: torch.device,
    seed: int,
    hidden_dims: tuple[int, int],
    max_epochs: int = 60,
    patience: int = 5,
    batch_size: int = 256,
) -> Tuple[TinyMLP, Dict[str, float]]:
    set_seeds(seed)
    model = TinyMLP(input_dim, hidden_dims=hidden_dims).to(device)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val_loss = float("inf")
    patience_left = patience

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_val).to(device))
            val_loss = loss_fn(val_logits, torch.from_numpy(y_val).to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {"val_loss": float(best_val_loss)}
    return model, metrics


def _fit_calibrator(logits: np.ndarray, y_true: np.ndarray) -> Optional[LogisticRegression]:
    if logits.size == 0 or len(np.unique(y_true)) < 2:
        return None
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(logits.reshape(-1, 1), y_true.astype(int))
    return calibrator


def _apply_calibrator(calibrator: Optional[LogisticRegression], logits: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return 1.0 / (1.0 + np.exp(-logits))
    return calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]


def _auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, prob))


def _log_loss(y_true: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(log_loss(y_true, prob, labels=[0, 1]))
    except ValueError:
        return 0.0


def train_fold(
    fold_idx: int,
    merged_df: pd.DataFrame,
    feature_cols: List[str],
    windows: Dict[str, Tuple[int, int]],
    *,
    device: torch.device,
    seed: int,
    score_quantile: float,
    hidden_dims: tuple[int, int],
    max_epochs: int,
) -> Dict[str, object]:
    X_train, y_train_long, y_train_short = slice_window(merged_df, feature_cols, windows["train"])
    X_val, y_val_long, y_val_short = slice_window(merged_df, feature_cols, windows["val"])
    X_test, y_test_long, y_test_short = slice_window(merged_df, feature_cols, windows["test"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    input_dim = X_train.shape[1]

    long_model, long_metrics = _train_direction(
        X_train,
        y_train_long,
        X_val,
        y_val_long,
        input_dim=input_dim,
        device=device,
        seed=seed,
        hidden_dims=hidden_dims,
        max_epochs=max_epochs,
    )

    if NN_CONFIG.enable_short:
        short_model, short_metrics = _train_direction(
            X_train,
            y_train_short,
            X_val,
            y_val_short,
            input_dim=input_dim,
            device=device,
            seed=seed + 1,
            hidden_dims=hidden_dims,
            max_epochs=max_epochs,
        )
    else:
        short_model = None
        short_metrics = {"val_loss": 0.0}

    long_train_logits = _predict_logits(long_model, X_train, device)
    short_train_logits = (
        _predict_logits(short_model, X_train, device) if short_model is not None else np.zeros_like(long_train_logits)
    )
    long_val_logits = _predict_logits(long_model, X_val, device)
    short_val_logits = (
        _predict_logits(short_model, X_val, device) if short_model is not None else np.zeros_like(long_val_logits)
    )
    long_test_logits = _predict_logits(long_model, X_test, device)
    short_test_logits = (
        _predict_logits(short_model, X_test, device) if short_model is not None else np.zeros_like(long_test_logits)
    )

    long_cal = _fit_calibrator(long_val_logits, y_val_long)
    short_cal = _fit_calibrator(short_val_logits, y_val_short) if short_model is not None else None

    long_train_prob = _apply_calibrator(long_cal, long_train_logits)
    if short_model is not None:
        short_train_prob = _apply_calibrator(short_cal, short_train_logits)
    else:
        short_train_prob = np.zeros_like(long_train_prob)
    train_score = np.maximum(long_train_prob, short_train_prob)

    score_threshold = float(np.quantile(train_score, score_quantile))
    score_percentiles = {
        "p50": float(np.quantile(train_score, 0.50)),
        "p90": float(np.quantile(train_score, 0.90)),
        "p95": float(np.quantile(train_score, 0.95)),
        "p97": float(np.quantile(train_score, 0.97)),
        "p98": float(np.quantile(train_score, 0.98)),
        "p99": float(np.quantile(train_score, 0.99)),
    }

    long_val_prob = _apply_calibrator(long_cal, long_val_logits)
    short_val_prob = _apply_calibrator(short_cal, short_val_logits) if short_model is not None else np.zeros_like(long_val_prob)
    long_test_prob = _apply_calibrator(long_cal, long_test_logits)
    short_test_prob = (
        _apply_calibrator(short_cal, short_test_logits) if short_model is not None else np.zeros_like(long_test_prob)
    )

    metrics = {
        "long": {
            "train_auc": _auc(y_train_long, long_train_prob),
            "val_auc": _auc(y_val_long, long_val_prob),
            "test_auc": _auc(y_test_long, long_test_prob),
            "val_log_loss": _log_loss(y_val_long, long_val_prob),
            "test_log_loss": _log_loss(y_test_long, long_test_prob),
        },
        "short": {
            "train_auc": _auc(y_train_short, short_train_prob) if short_model is not None else 0.0,
            "val_auc": _auc(y_val_short, short_val_prob) if short_model is not None else 0.0,
            "test_auc": _auc(y_test_short, short_test_prob) if short_model is not None else 0.0,
            "val_log_loss": _log_loss(y_val_short, short_val_prob) if short_model is not None else 0.0,
            "test_log_loss": _log_loss(y_test_short, short_test_prob) if short_model is not None else 0.0,
        },
        "threshold": {
            "score_quantile": score_quantile,
            "score_threshold": score_threshold,
        },
        "score_percentiles": score_percentiles,
    }

    return {
        "fold_idx": fold_idx,
        "scaler": scaler,
        "long_model": long_model,
        "short_model": short_model,
        "long_calibrator": long_cal,
        "short_calibrator": short_cal,
        "metrics": metrics,
        "score_threshold": score_threshold,
        "score_percentiles": score_percentiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny MLP models with fixed-horizon labels")
    parser.add_argument("--data-path", default="/mnt/data/es_bars_2010_2025.h5")
    parser.add_argument("--dataset-key", default="bars_5min")
    parser.add_argument("--output-dir", dest="output_dir", default="models/nn_saved")
    parser.add_argument("--out-dir", dest="output_dir")
    parser.add_argument("--seed", type=int, default=NN_CONFIG.seed)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.set_defaults(output_dir="models/nn_saved")
    args = parser.parse_args()

    fast_max_bars = 250_000
    fast_epochs = 5
    fast_folds = 2
    fast_hidden_dims = (16, 8)
    full_hidden_dims = (32, 16)

    folds = args.folds if args.folds is not None else (fast_folds if args.fast else NN_CONFIG.folds)
    max_epochs = args.epochs if args.epochs is not None else (fast_epochs if args.fast else 60)
    max_bars = args.max_bars if args.max_bars is not None else (fast_max_bars if args.fast else None)
    hidden_dims = fast_hidden_dims if args.fast else full_hidden_dims

    set_seeds(args.seed)

    print("\n" + "=" * 60)
    print("NN TRAINING: TINY MLP")
    print("=" * 60)
    if args.fast:
        print("FAST MODE ENABLED")
    print(f"Dataset key: {args.dataset_key}")
    if max_bars is not None:
        print(f"Max bars: {max_bars:,}")
    print(f"Epochs: {max_epochs}")
    print(f"Folds: {folds}")
    print(f"Hidden dims: {hidden_dims}")

    bars = load_bars(args.data_path, dataset_key=args.dataset_key, max_bars=max_bars)
    print(f"\nLoaded {len(bars):,} bars from {args.data_path}")

    features_df, feature_cols = build_features(bars)
    labels_df = build_labels(
        bars,
        horizon_bars=NN_CONFIG.horizon_bars,
        threshold_ticks=NN_CONFIG.threshold_ticks,
        tick_size=RISK_CONFIG.tick_size,
        session_mode=NN_CONFIG.session_mode,
        session_start=RISK_CONFIG.session_start,
        session_end=RISK_CONFIG.session_end,
    )
    assert not labels_df[["y_long", "y_short"]].isna().any().any(), "Label NaNs detected"

    merged = labels_df.merge(features_df, on="idx", how="inner")

    windows_list, embargo = build_walk_forward_windows(
        n_bars=len(bars),
        feature_lookback=NN_CONFIG.feature_lookback,
        horizon_bars=NN_CONFIG.horizon_bars,
        embargo_bars=NN_CONFIG.embargo_bars,
        train_fraction=NN_CONFIG.train_fraction,
        val_fraction=NN_CONFIG.val_fraction,
        test_fraction=NN_CONFIG.test_fraction,
        folds=folds,
    )
    min_embargo = NN_CONFIG.horizon_bars + NN_CONFIG.feature_lookback + 1
    assert embargo >= min_embargo, "Embargo must be >= feature_lookback + horizon_bars + 1"

    score_quantile = resolve_score_quantile(
        score_quantile=NN_CONFIG.score_quantile,
        auto_score_quantile=NN_CONFIG.auto_score_quantile,
        target_trades_per_day=NN_CONFIG.target_trades_per_day,
        session_mode=NN_CONFIG.session_mode,
        session_start=RISK_CONFIG.session_start,
        session_end=RISK_CONFIG.session_end,
        bar_minutes=5,
        min_quantile=NN_CONFIG.score_quantile_min,
        max_quantile=NN_CONFIG.score_quantile_max,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Embargo bars: {embargo}")
    print(f"Folds: {len(windows_list)}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for fold_idx, windows in enumerate(windows_list):
        print(f"\n--- Fold {fold_idx} ---")
        print(f"Train window: {windows['train']}")
        print(f"Val window:   {windows['val']}")
        print(f"Test window:  {windows['test']}")

        results = train_fold(
            fold_idx,
            merged,
            feature_cols,
            windows,
            device=device,
            seed=args.seed + fold_idx * 10,
            score_quantile=score_quantile,
            hidden_dims=hidden_dims,
            max_epochs=max_epochs,
        )

        print(
            f"Fold {fold_idx} AUC (train/val/test) - "
            f"Long: {results['metrics']['long']['train_auc']:.3f}/"
            f"{results['metrics']['long']['val_auc']:.3f}/"
            f"{results['metrics']['long']['test_auc']:.3f} | "
            f"Short: {results['metrics']['short']['train_auc']:.3f}/"
            f"{results['metrics']['short']['val_auc']:.3f}/"
            f"{results['metrics']['short']['test_auc']:.3f}"
        )
        print(
            f"Score threshold: {results['score_threshold']:.4f} "
            f"(quantile {score_quantile})"
        )
        print(f"Score percentiles: {json.dumps(results['score_percentiles'], indent=2)}")

        if args.no_save:
            continue

        fold_dir = output_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        torch.save(results["long_model"].state_dict(), fold_dir / "model_long.pt")
        if results["short_model"] is not None:
            torch.save(results["short_model"].state_dict(), fold_dir / "model_short.pt")
        joblib.dump(results["scaler"], fold_dir / "scaler.pkl")

        if results["long_calibrator"] is not None:
            joblib.dump(results["long_calibrator"], fold_dir / "calibrator_long.pkl")
        if results["short_calibrator"] is not None:
            joblib.dump(results["short_calibrator"], fold_dir / "calibrator_short.pkl")

        risk_dict = asdict(RISK_CONFIG)
        if isinstance(risk_dict.get("session_start"), time):
            risk_dict["session_start"] = risk_dict["session_start"].isoformat()
        if isinstance(risk_dict.get("session_end"), time):
            risk_dict["session_end"] = risk_dict["session_end"].isoformat()

        nn_cfg = asdict(NN_CONFIG)
        nn_cfg["embargo_bars"] = embargo
        nn_cfg["score_threshold"] = results["score_threshold"]
        nn_cfg["score_quantile"] = score_quantile
        nn_cfg["selection_mode"] = NN_CONFIG.selection_mode
        nn_cfg["day_percentile_floor"] = NN_CONFIG.day_percentile_floor
        nn_cfg["global_floor_score"] = results["score_percentiles"].get("p90", results["score_threshold"])
        catastrophic_stop_ticks = NN_CONFIG.catastrophic_stop_ticks
        if catastrophic_stop_ticks is None:
            catastrophic_stop_ticks = int(NN_CONFIG.threshold_ticks) * 4
        nn_cfg["catastrophic_stop_ticks"] = int(catastrophic_stop_ticks)
        nn_cfg["max_hold_bars"] = NN_CONFIG.horizon_bars
        nn_cfg["model_hidden_dims"] = list(hidden_dims)
        nn_cfg["label_version"] = "aligned_fixed_horizon_v1"
        nn_cfg["label_entry_price_col"] = "open"
        nn_cfg["label_exit_price_col"] = "close"
        nn_cfg["stop_loss_ticks"] = TRAINING_CONFIG.stop_loss_ticks
        nn_cfg["target_multiplier"] = TRAINING_CONFIG.target_multiplier
        nn_cfg["tick_size"] = RISK_CONFIG.tick_size
        nn_cfg["tick_value"] = RISK_CONFIG.tick_value
        nn_cfg["bar_minutes"] = 5
        if isinstance(nn_cfg.get("deadline_time"), time):
            nn_cfg["deadline_time"] = nn_cfg["deadline_time"].isoformat()

        metadata = {
            "created": pd.Timestamp.utcnow().isoformat(),
            "model_type": "tiny_mlp",
            "bar_minutes": 5,
            "fold_idx": fold_idx,
            "feature_cols": feature_cols,
            "nn_config": nn_cfg,
            "risk_config": risk_dict,
            "windows": windows,
            "metrics": results["metrics"],
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn.__version__,
                "torch": torch.__version__,
            },
            "training_meta": {
                "dataset_key": args.dataset_key,
                "max_bars": max_bars,
                "fast_mode": bool(args.fast),
                "epochs": max_epochs,
                "folds": folds,
                "hidden_dims": list(hidden_dims),
            },
        }

        with open(fold_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved fold {fold_idx} artifacts to {fold_dir}")


if __name__ == "__main__":
    main()
