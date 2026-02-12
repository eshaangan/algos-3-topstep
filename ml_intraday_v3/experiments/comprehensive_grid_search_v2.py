"""
Comprehensive experiment runner with purged CV + embargo.

This replaces fixed train/test date windows with event-level,
time-safe validation and supports multiple model families.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from ml_intraday_v3.core.instrument import InstrumentSpec
from ml_intraday_v3.features.build import build_features
from ml_intraday_v3.labels.events import balance_events, generate_events
from ml_intraday_v3.labels.triple_barrier import apply_triplebarrier
from ml_intraday_v3.validation.purged_cv import build_purged_kfold_splits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_base_configs(config_dir: Path) -> Dict:
    configs: Dict[str, dict] = {}
    with open(config_dir / "labeling.yaml", "r") as f:
        configs["labeling"] = yaml.safe_load(f)
    with open(config_dir / "execution_spec.yaml", "r") as f:
        configs["execution"] = yaml.safe_load(f)
    with open(config_dir / "features.yaml", "r") as f:
        configs["features"] = yaml.safe_load(f)
    return configs


def apply_experiment_overrides(base_configs: Dict, exp_config: Dict) -> Dict:
    configs = deepcopy(base_configs)

    labeling = exp_config.get("labeling", {})
    if labeling:
        tb = configs["labeling"]["primary_labeling"]["triple_barrier"]
        if "pt" in labeling:
            tb["pt_multipliers"] = [float(labeling["pt"])]
        if "sl" in labeling:
            tb["sl_multipliers"] = [float(labeling["sl"])]
        if "hz" in labeling:
            hz_val = int(labeling["hz"])
            tb["horizon_bars"]["5m"] = [hz_val]
            configs["execution"]["holding_constraints"]["max_holding_bars"]["5m"] = hz_val

    features_cfg = exp_config.get("features_config")
    if features_cfg:
        configs["features"].update(features_cfg)

    session_mode = exp_config.get("session_mode")
    if session_mode == "rth_only":
        configs["execution"]["session_mode"] = "rth_only"
    elif session_mode == "eth_only":
        configs["execution"]["session_mode"] = "eth_only"

    return configs


def filter_session(df: pd.DataFrame, session_mode: str | None) -> pd.DataFrame:
    if not session_mode or session_mode == "rth_eth":
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    ts_ny = df.index.tz_convert("America/New_York") if df.index.tz else df.index.tz_localize("UTC").tz_convert("America/New_York")
    in_rth = (
        ((ts_ny.hour > 9) | ((ts_ny.hour == 9) & (ts_ny.minute >= 30)))
        & ((ts_ny.hour < 16) | ((ts_ny.hour == 16) & (ts_ny.minute == 0)))
    )
    if session_mode == "rth_only":
        return df[in_rth]
    if session_mode == "eth_only":
        return df[~in_rth]
    return df


def create_sample_weights(events_df: pd.DataFrame, method: str) -> pd.Series:
    n_samples = len(events_df)
    if n_samples == 0:
        return pd.Series(dtype=float)

    if "event_id" not in events_df.columns:
        raise ValueError("events_df must include event_id for weight alignment")

    if method == "uniform":
        return pd.Series(np.ones(n_samples), index=events_df["event_id"].to_numpy(), dtype=float)

    if method.startswith("time_decay_"):
        lambda_val = float(method.split("_")[-1])
        timestamps = pd.to_datetime(events_df["t0"], utc=True).astype("int64").to_numpy()
        t_min, t_max = timestamps.min(), timestamps.max()
        t_norm = (timestamps - t_min) / (t_max - t_min + 1)
        weights = np.exp(lambda_val * t_norm)
        return pd.Series(
            weights * n_samples / weights.sum(),
            index=events_df["event_id"].to_numpy(),
            dtype=float,
        )

    if method == "class_balanced":
        if "y" not in events_df.columns:
            raise ValueError("class_balanced weighting requires labeled events_df with column 'y'")
        y = (events_df["y"] == 1).astype(int).to_numpy()
        pos = max(int(y.sum()), 1)
        neg = max(int((1 - y).sum()), 1)
        w_pos = 0.5 * n_samples / pos
        w_neg = 0.5 * n_samples / neg
        return pd.Series(
            np.where(y == 1, w_pos, w_neg),
            index=events_df["event_id"].to_numpy(),
            dtype=float,
        )

    raise ValueError(f"Unknown sample weight method: {method}")


def fit_with_sample_weights(model, X_train: pd.DataFrame, y_train: pd.Series, sample_weights: np.ndarray):
    """
    Fit estimators while routing sample weights for Pipeline models.
    """
    if sample_weights is None or len(sample_weights) == 0:
        model.fit(X_train, y_train)
        return model

    # Calibrated model wrapping a pipeline/base estimator.
    if isinstance(model, CalibratedClassifierCV):
        base_estimator = model.estimator
        if isinstance(base_estimator, Pipeline):
            try:
                # CalibratedClassifierCV forwards fit kwargs to estimator.fit.
                model.fit(X_train, y_train, model__sample_weight=sample_weights)
                return model
            except TypeError:
                # Fallback: run without weights rather than fail the experiment.
                model.fit(X_train, y_train)
                return model
        model.fit(X_train, y_train, sample_weight=sample_weights)
        return model

    # Plain pipeline estimator.
    if isinstance(model, Pipeline):
        model.fit(X_train, y_train, model__sample_weight=sample_weights)
        return model

    # Plain estimator.
    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model


def build_model(exp_config: Dict):
    model_kind = exp_config.get("model_kind", "lightgbm")
    params = deepcopy(exp_config.get("model_params", {}))

    if model_kind == "lightgbm":
        return LGBMClassifier(objective="binary", verbose=-1, force_col_wise=True, **params)
    if model_kind == "random_forest":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    if model_kind == "extra_trees":
        return ExtraTreesClassifier(random_state=42, n_jobs=-1, **params)
    if model_kind == "logreg":
        base = LogisticRegression(max_iter=2000, solver="saga", n_jobs=-1, **params)
        return Pipeline([("scale", StandardScaler()), ("model", base)])
    if model_kind == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", MLPClassifier(random_state=42, max_iter=500, early_stopping=True, **params)),
            ]
        )
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def coerce_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure model inputs are numeric (LightGBM rejects object dtypes).
    """
    X_out = X.copy()
    for col in X_out.columns:
        if pd.api.types.is_bool_dtype(X_out[col]):
            X_out[col] = X_out[col].astype(float)
        elif pd.api.types.is_object_dtype(X_out[col]):
            X_out[col] = pd.to_numeric(X_out[col], errors="coerce")
    return X_out


def run_single_fold(
    all_df: pd.DataFrame,
    events_df: pd.DataFrame,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    exp_config: Dict,
    fold: Dict,
    fold_num: int,
) -> Dict:
    train_events = events_df[events_df["event_id"].isin(fold["train_event_ids"])].copy()
    test_events = events_df[events_df["event_id"].isin(fold["test_event_ids"])].copy()

    if train_events.empty or test_events.empty:
        return {"fold": fold_num, "error": "Empty train or test events"}

    # Optional balancing on training events only.
    balance_method = exp_config.get("balance_method", "undersample")
    target_ratio = float(exp_config.get("target_long_ratio", 0.5))
    if balance_method in {"undersample", "oversample"}:
        train_events = balance_events(train_events, target_long_ratio=target_ratio, method=balance_method)

    # Event-level features: align on event t0 then remap index to event_id.
    train_x = features_df.loc[train_events["t0"]].copy()
    train_x.index = train_events["event_id"].to_numpy()
    train_x["side"] = train_events["side"].to_numpy()

    test_x = features_df.loc[test_events["t0"]].copy()
    test_x.index = test_events["event_id"].to_numpy()
    test_x["side"] = test_events["side"].to_numpy()

    train_df = train_x.join(labels_df[["y"]], how="inner")
    test_df = test_x.join(labels_df[["y"]], how="inner")

    if len(train_df) < 50 or len(test_df) < 10:
        return {
            "fold": fold_num,
            "error": f"Insufficient samples (train={len(train_df)}, test={len(test_df)})",
        }

    feature_set = exp_config.get("feature_set")
    if feature_set:
        missing = [c for c in feature_set if c not in train_df.columns]
        if missing:
            return {"fold": fold_num, "error": f"Missing features: {missing[:5]}"}
        feature_cols = feature_set
    else:
        feature_cols = [c for c in train_df.columns if c not in {"y", "usable_for_training"}]

    X_train = coerce_numeric_features(train_df[feature_cols])
    X_test = coerce_numeric_features(test_df[feature_cols])
    y_train = (train_df["y"] == 1).astype(int)
    y_test = (test_df["y"] == 1).astype(int)

    weight_source = train_events[train_events["event_id"].isin(train_df.index)].copy()
    weight_source = weight_source.set_index("event_id").join(train_df[["y"]], how="left").reset_index()
    weight_series = create_sample_weights(weight_source, exp_config.get("sample_weight", "uniform"))
    sample_weights = weight_series.reindex(train_df.index).to_numpy(dtype=float)

    model = build_model(exp_config)
    calibration = exp_config.get("calibration")
    if calibration in {"sigmoid", "isotonic"}:
        model = CalibratedClassifierCV(model, method=calibration, cv=3)

    model = fit_with_sample_weights(model, X_train, y_train, sample_weights)

    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, test_proba)
    train_auc = roc_auc_score(y_train, train_proba)
    pr_auc = average_precision_score(y_test, test_proba)
    brier = brier_score_loss(y_test, test_proba)

    test_proba_clip = np.clip(test_proba, 1e-6, 1 - 1e-6)
    xent = log_loss(y_test, test_proba_clip)
    pct_above_055 = float((test_proba > 0.55).mean())
    pct_above_060 = float((test_proba > 0.60).mean())
    mean_prob = float(test_proba.mean())
    std_prob = float(test_proba.std())

    est_signals_per_day = len(test_events) / max((len(all_df.loc[test_events["t0"].min():test_events["t0"].max()]) / 80), 1.0)
    est_trades_per_day = est_signals_per_day * pct_above_055

    return {
        "fold": fold_num,
        "train_auc": float(train_auc),
        "test_auc": float(test_auc),
        "pr_auc": float(pr_auc),
        "brier": float(brier),
        "log_loss": float(xent),
        "train_test_gap": float(train_auc - test_auc),
        "pct_signals_above_055": pct_above_055,
        "pct_signals_above_060": pct_above_060,
        "mean_test_prob": mean_prob,
        "std_test_prob": std_prob,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_train_events": int(len(train_events)),
        "n_test_events": int(len(test_events)),
        "est_signals_per_day": float(est_signals_per_day),
        "est_trades_per_day": float(est_trades_per_day),
    }


def run_experiment(exp_config: Dict, data_path: Path, config_dir: Path) -> Dict:
    exp_id = exp_config["exp_id"]
    logger.info(f"=== Experiment {exp_id} ===")
    start = time.time()

    base_configs = load_base_configs(config_dir)
    configs = apply_experiment_overrides(base_configs, exp_config)

    data_file = data_path / "MES_5min_Oct2024_Dec2025.parquet"
    df = pd.read_parquet(data_file).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df = filter_session(df, exp_config.get("session_mode"))
    if df.empty:
        return {"exp_id": exp_id, "status": "SKIPPED", "error": "No bars after session filter", "config": exp_config}

    events = generate_events(
        bars_df=df,
        bar_size="5m",
        labeling_config=configs["labeling"],
        execution_spec=configs["execution"],
    )
    if events.empty:
        return {"exp_id": exp_id, "status": "FAILED", "error": "No events generated", "config": exp_config}

    # Optional training window truncation.
    train_window_months = exp_config.get("training_window_months")
    if train_window_months:
        max_t0 = pd.to_datetime(events["t0"], utc=True).max()
        min_t0 = max_t0 - pd.DateOffset(months=int(train_window_months))
        events = events[pd.to_datetime(events["t0"], utc=True) >= min_t0].copy()

    if len(events) < 200:
        return {"exp_id": exp_id, "status": "FAILED", "error": f"Insufficient events after filters: {len(events)}", "config": exp_config}

    instrument_spec = InstrumentSpec(symbol="MES", tick_size_points=0.25, contract_multiplier_usd_per_point=5.0)
    labels = apply_triplebarrier(
        bars_df=df,
        events_df=events,
        bar_size="5m",
        labeling_config=configs["labeling"],
        execution_spec=configs["execution"],
        instrument_spec=instrument_spec,
    )
    if labels.empty:
        return {"exp_id": exp_id, "status": "FAILED", "error": "No labels generated", "config": exp_config}

    # Keep only events with labels.
    labeled_event_ids = set(labels["event_id"].tolist())
    events = events[events["event_id"].isin(labeled_event_ids)].copy()
    labels = labels.set_index("event_id")

    features = build_features(df, "5m", configs["features"])

    n_splits = int(exp_config.get("cv_n_splits", 5))
    embargo_bars = int(exp_config.get("cv_embargo_bars", 12))
    folds = build_purged_kfold_splits(events, df.index, n_splits=n_splits, embargo_bars=embargo_bars)

    fold_results = []
    for i, fold in enumerate(folds, 1):
        try:
            fold_result = run_single_fold(
                all_df=df,
                events_df=events,
                features_df=features,
                labels_df=labels,
                exp_config=exp_config,
                fold=fold,
                fold_num=i,
            )
            fold_results.append(fold_result)
        except Exception as exc:
            fold_results.append({"fold": i, "error": str(exc)})
            logger.exception("Fold %s failed", i)

    successful = [f for f in fold_results if "error" not in f]
    if not successful:
        return {
            "exp_id": exp_id,
            "config": exp_config,
            "status": "FAILED",
            "error": "All folds failed",
            "folds": fold_results,
            "runtime_seconds": time.time() - start,
        }

    summary = {
        "median_test_auc": float(np.median([f["test_auc"] for f in successful])),
        "median_train_auc": float(np.median([f["train_auc"] for f in successful])),
        "median_pr_auc": float(np.median([f["pr_auc"] for f in successful])),
        "median_brier": float(np.median([f["brier"] for f in successful])),
        "median_log_loss": float(np.median([f["log_loss"] for f in successful])),
        "mean_train_test_gap": float(np.mean([f["train_test_gap"] for f in successful])),
        "std_test_auc": float(np.std([f["test_auc"] for f in successful])),
        "median_pct_signals_above_055": float(np.median([f["pct_signals_above_055"] for f in successful])),
        "median_pct_signals_above_060": float(np.median([f["pct_signals_above_060"] for f in successful])),
        "mean_test_prob": float(np.mean([f["mean_test_prob"] for f in successful])),
        "mean_std_prob": float(np.mean([f["std_test_prob"] for f in successful])),
        "median_est_trades_per_day": float(np.median([f["est_trades_per_day"] for f in successful])),
        "n_successful_folds": len(successful),
        "n_total_folds": len(fold_results),
    }

    return {
        "exp_id": exp_id,
        "config": exp_config,
        "status": "SUCCESS",
        "runtime_seconds": float(time.time() - start),
        "folds": fold_results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive grid search experiment")
    parser.add_argument("--config", type=str, required=True, help="Experiment config JSON")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument("--config-dir", type=str, required=True, help="Config directory")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        exp_config = json.load(f)

    result = run_experiment(exp_config, Path(args.data_dir), Path(args.config_dir))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
