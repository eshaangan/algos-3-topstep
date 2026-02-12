"""
Promote a completed exhaustive experiment to a deployable model bundle.

Usage:
  python -m ml_intraday_v3.experiments.promote_top_exhaustive_model \
    --result-json /tmp/phasefull_results/result_exhaustive_exp_00336.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml_intraday_v3.core.instrument import InstrumentSpec
from ml_intraday_v3.experiments.comprehensive_grid_search_v2 import (
    apply_experiment_overrides,
    build_model,
    coerce_numeric_features,
    create_sample_weights,
    fit_with_sample_weights,
    load_base_configs,
)
from ml_intraday_v3.features.build import build_features
from ml_intraday_v3.labels.events import balance_events, generate_events
from ml_intraday_v3.labels.triple_barrier import apply_triplebarrier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_preprocessor_state(X: pd.DataFrame) -> dict:
    medians = X.median(numeric_only=False).to_numpy(dtype=float)
    X_imp = X.fillna(X.median(numeric_only=False))
    means = X_imp.mean(numeric_only=False).to_numpy(dtype=float)
    stds = X_imp.std(numeric_only=False).to_numpy(dtype=float)
    stds = np.where(stds <= 1e-12, 1.0, stds)
    return {
        "impute": "median",
        "scaler": "standard",
        "medians": medians.tolist(),
        "means": means.tolist(),
        "stds": stds.tolist(),
    }


def _apply_preprocessor(X: pd.DataFrame, pre: dict) -> pd.DataFrame:
    X_imp = X.fillna(pd.Series(pre["medians"], index=X.columns))
    X_scaled = (X_imp - np.array(pre["means"])) / (np.array(pre["stds"]) + 1e-8)
    return pd.DataFrame(X_scaled, index=X.index, columns=X.columns)


def promote(result_json: Path, data_dir: Path, config_dir: Path, output_bundle: Path) -> dict:
    with open(result_json, "r") as f:
        result = json.load(f)

    if result.get("status") != "SUCCESS":
        raise ValueError(f"Result is not SUCCESS: {result_json}")

    exp_config = result["config"]
    exp_id = exp_config["exp_id"]
    logger.info("Promoting experiment: %s", exp_id)

    base_configs = load_base_configs(config_dir)
    configs = apply_experiment_overrides(base_configs, exp_config)

    data_file = data_dir / "MES_5min_Oct2024_Dec2025.parquet"
    df = pd.read_parquet(data_file).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    events = generate_events(
        bars_df=df,
        bar_size="5m",
        labeling_config=configs["labeling"],
        execution_spec=configs["execution"],
    )
    if events.empty:
        raise ValueError("No events generated")

    train_window_months = exp_config.get("training_window_months")
    if train_window_months:
        max_t0 = pd.to_datetime(events["t0"], utc=True).max()
        min_t0 = max_t0 - pd.DateOffset(months=int(train_window_months))
        events = events[pd.to_datetime(events["t0"], utc=True) >= min_t0].copy()

    balance_method = exp_config.get("balance_method", "undersample")
    target_ratio = float(exp_config.get("target_long_ratio", 0.5))
    if balance_method in {"undersample", "oversample"}:
        events = balance_events(events, target_long_ratio=target_ratio, method=balance_method)

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
        raise ValueError("No labels generated")

    labels = labels.set_index("event_id")
    events = events[events["event_id"].isin(labels.index)].copy()

    features = build_features(df, "5m", configs["features"])
    X = features.loc[events["t0"]].copy()
    X.index = events["event_id"].to_numpy()
    X["side"] = events["side"].to_numpy()

    dataset = X.join(labels[["y"]], how="inner")
    if len(dataset) < 200:
        raise ValueError(f"Insufficient rows after alignment: {len(dataset)}")

    feature_set = exp_config.get("feature_set")
    if feature_set:
        feature_cols = feature_set
    else:
        feature_cols = [c for c in dataset.columns if c not in {"y", "usable_for_training"}]

    X_raw = coerce_numeric_features(dataset[feature_cols])
    y = (dataset["y"] == 1).astype(int)

    weight_source = events[events["event_id"].isin(dataset.index)].copy()
    weight_source = weight_source.set_index("event_id").join(dataset[["y"]], how="left").reset_index()
    weight_series = create_sample_weights(weight_source, exp_config.get("sample_weight", "uniform"))
    sample_weights = weight_series.reindex(dataset.index).to_numpy(dtype=float)

    preprocessor_state = _build_preprocessor_state(X_raw)
    X_train = _apply_preprocessor(X_raw, preprocessor_state)

    model = build_model(exp_config)
    calibration = exp_config.get("calibration")
    if calibration in {"sigmoid", "isotonic"}:
        from sklearn.calibration import CalibratedClassifierCV

        model = CalibratedClassifierCV(model, method=calibration, cv=3)

    model = fit_with_sample_weights(model, X_train, y, sample_weights)
    train_proba = model.predict_proba(X_train)[:, 1]
    train_auc = float(roc_auc_score(y, train_proba))

    output_bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "primary_model": model,
        "primary_preprocessor": preprocessor_state,
        "primary_feature_columns": feature_cols,
        "has_side_feature": "side" in feature_cols,
        "thresholds": {"primary_threshold": 0.10},
        "meta_model": None,
        "meta_preprocessor": None,
        "meta_feature_columns": None,
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_result_json": str(result_json),
            "source_exp_id": exp_id,
            "source_summary": result.get("summary", {}),
            "n_train_samples_full": int(len(X_train)),
            "n_features": int(len(feature_cols)),
            "train_auc_full": train_auc,
            "training_window_months": exp_config.get("training_window_months"),
        },
    }
    joblib.dump(bundle, output_bundle)

    promoted_config_path = output_bundle.with_suffix(".config.json")
    with open(promoted_config_path, "w") as f:
        json.dump(exp_config, f, indent=2)

    logger.info("Bundle saved: %s", output_bundle)
    logger.info("Config saved: %s", promoted_config_path)
    logger.info("Full-train AUC: %.4f", train_auc)

    return {
        "bundle_path": str(output_bundle),
        "config_path": str(promoted_config_path),
        "train_auc_full": train_auc,
        "n_samples": int(len(X_train)),
        "n_features": int(len(feature_cols)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote exhaustive experiment to deployable model bundle")
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("ml_intraday_v3/data"))
    parser.add_argument("--config-dir", type=Path, default=Path("ml_intraday_v3/configs"))
    parser.add_argument(
        "--output-bundle",
        type=Path,
        default=Path("ml_intraday_v3/models/saved/model_bundle_phasefull_best.pkl"),
    )
    args = parser.parse_args()

    stats = promote(
        result_json=args.result_json,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        output_bundle=args.output_bundle,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

