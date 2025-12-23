"""
Training loop for primary model using CV splits.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from run_manifest import hash_content

from .dataset import build_event_dataset, build_meta_dataset
from .preprocess import FoldPreprocessor
from .metrics import compute_metrics
from .schema import write_training_schema

logger = logging.getLogger(__name__)


def _load_schema_hash(schema_path: Path, name: str) -> str:
    if not schema_path.exists():
        raise FileNotFoundError(f"{name} not found: {schema_path}")
    with open(schema_path, "r") as f:
        schema = json.load(f)
    if "schema_hash" not in schema:
        raise ValueError(f"{name} missing schema_hash: {schema_path}")
    return schema["schema_hash"]


def _binary_target(y, positive_label):
    return (np.asarray(y) == positive_label).astype(int)


def train_on_splits(
    run_dir: Path | str,
    bar_size: str,
    training_config: dict,
    cv_kind: str,
    training_dir_override: Path | str | None = None,
) -> dict:
    """
    Train baseline model on CV splits and write artifacts.

    Returns dict with training_dir, schema_path, schema_hash, n_splits.
    """
    run_dir = Path(run_dir)
    bar_dir = run_dir / f"bar_size={bar_size}"
    if training_dir_override:
        training_dir = Path(training_dir_override)
    else:
        training_dir = bar_dir / "training" / cv_kind
    training_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_event_dataset(run_dir, bar_size, training_config)
    feature_cols = [
        c
        for c in dataset.columns
        if c not in ["event_id", "t0", "y", "w_final"]
    ]

    dataset = dataset.set_index("event_id", drop=False)

    events_path = bar_dir / "events.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"events.parquet not found: {events_path}")
    events_df = pd.read_parquet(events_path)

    cv_path = bar_dir / "cv_splits.json"
    if not cv_path.exists():
        raise FileNotFoundError(f"cv_splits.json not found: {cv_path}")
    with open(cv_path, "r") as f:
        cv_data = json.load(f)

    if cv_kind == "purged_kfold":
        splits = cv_data.get("purged_kfold", [])
        split_id_key = "fold"
        prefix = "fold"
    elif cv_kind == "cpcv":
        splits = cv_data.get("cpcv", [])
        split_id_key = "path_id"
        prefix = "path"
    else:
        raise ValueError(f"Unsupported cv_kind: {cv_kind}")

    if not splits:
        raise ValueError(f"No splits found for cv_kind={cv_kind}")

    cfg = training_config
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    model_cfg = cfg.get("model", {})
    model_kind = model_cfg.get("kind", "logreg")
    if model_kind != "logreg":
        raise ValueError(f"Unsupported model kind: {model_kind}")

    model_params = model_cfg.get("params", {})
    training_config_hash = hash_content(training_config)
    target_cfg = cfg.get("target", {})
    target_col = target_cfg.get("column", "y")
    positive_label = target_cfg.get("positive_label", 1)

    weight_cfg = cfg.get("sample_weight", {})
    weight_enabled = bool(weight_cfg.get("enabled", False))

    threshold = cfg.get("eval", {}).get("threshold", 0.5)
    write_predictions = bool(cfg.get("output", {}).get("write_predictions", True))
    write_model = bool(cfg.get("output", {}).get("write_model", True))

    meta_cfg = cfg.get("meta", {})
    meta_enabled = bool(meta_cfg.get("enabled", False))
    meta_threshold = float(meta_cfg.get("threshold_meta", 0.5))
    meta_model_cfg = meta_cfg.get("model", {})
    meta_model_params = meta_model_cfg.get("params", {})
    meta_write_preds = bool(
        meta_cfg.get("output", {}).get("write_meta_predictions", True)
    )
    meta_write_model = bool(
        meta_cfg.get("output", {}).get("write_meta_model", True)
    )

    split_metrics = []
    meta_split_metrics = []

    for split in splits:
        split_id = split.get(split_id_key)
        train_ids = split.get("train_event_ids", [])
        test_ids = split.get("test_event_ids", [])

        train_df = dataset.loc[train_ids].sort_index()
        test_df = dataset.loc[test_ids].sort_index()

        pre = FoldPreprocessor(feature_cols, cfg).fit(train_df)
        X_train, y_train, w_train = pre.transform(train_df)
        X_test, y_test, w_test = pre.transform(test_df)

        y_train_bin = _binary_target(
            train_df[target_col].to_numpy(), positive_label
        )
        y_test_bin = _binary_target(
            test_df[target_col].to_numpy(), positive_label
        )

        if len(np.unique(y_train_bin)) < 2:
            raise ValueError(
                f"Split {split_id} has single-class training labels"
            )

        model = LogisticRegression(
            C=model_params.get("C", 1.0),
            penalty=model_params.get("penalty", "l2"),
            solver=model_params.get("solver", "lbfgs"),
            max_iter=model_params.get("max_iter", 200),
            class_weight=model_params.get("class_weight"),
            random_state=seed,
        )

        model.fit(
            X_train,
            y_train_bin,
            sample_weight=w_train if weight_enabled else None,
        )

        y_prob = model.predict_proba(X_test)[:, 1]
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        metrics = compute_metrics(
            y_test_bin,
            y_prob,
            threshold=threshold,
            sample_weight=w_test if weight_enabled else None,
        )
        tn, fp, fn, tp = confusion_matrix(
            y_test_bin, y_pred, labels=[0, 1]
        ).ravel()

        split_dir = training_dir / f"{prefix}_{split_id}"
        split_dir.mkdir(parents=True, exist_ok=True)

        metrics_payload = {
            "split_id": split_id,
            "cv_kind": cv_kind,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "pos_rate_train": float(y_train_bin.mean()),
            "pos_rate_test": float(y_test_bin.mean()),
            "base_rate_train": float(y_train_bin.mean()),
            "base_rate_test": float(y_test_bin.mean()),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "threshold": float(threshold),
            },
            "metrics": metrics,
        }
        with open(split_dir / "metrics.json", "w") as f:
            json.dump(metrics_payload, f, indent=2)

        if write_predictions:
            weights = (
                w_test if weight_enabled else np.ones(len(test_df), dtype=float)
            )
            preds_df = pd.DataFrame(
                {
                    "event_id": test_df["event_id"].to_numpy(),
                    "y_true": y_test_bin,
                    "y_prob": y_prob,
                    "y_pred": y_pred,
                    "weight": weights,
                }
            )
            preds_df.to_parquet(split_dir / "preds.parquet")

        if meta_enabled:
            primary_train_preds = pd.DataFrame(
                {
                    "event_id": train_df["event_id"].to_numpy(),
                    "y_prob": y_prob_train,
                }
            )
            primary_test_preds = pd.DataFrame(
                {
                    "event_id": test_df["event_id"].to_numpy(),
                    "y_prob": y_prob,
                }
            )

            meta_train_df, meta_feature_cols = build_meta_dataset(
                primary_preds_df=primary_train_preds,
                base_event_dataset_df=dataset.reset_index(drop=True),
                events_df=events_df,
                config=cfg,
            )
            meta_test_df, _ = build_meta_dataset(
                primary_preds_df=primary_test_preds,
                base_event_dataset_df=dataset.reset_index(drop=True),
                events_df=events_df,
                config=cfg,
            )

            meta_payload = {
                "split_id": split_id,
                "cv_kind": cv_kind,
                "n_test_events": int(len(test_df)),
                "n_proposed_trades_test": int(len(meta_test_df)),
                "n_meta_positive_test": int(meta_test_df["y"].sum())
                if not meta_test_df.empty
                else 0,
                "acceptance_rate": None,
                "metrics": None,
                "reason": None,
            }

            if meta_train_df.empty or meta_test_df.empty:
                meta_payload["reason"] = "no_proposed_trades"
                if meta_write_preds:
                    empty_preds = pd.DataFrame(
                        columns=[
                            "event_id",
                            "m_true",
                            "p_meta",
                            "m_pred",
                            "take_trade",
                            "p_primary",
                            "weight",
                        ]
                    )
                    empty_preds.to_parquet(split_dir / "meta_preds.parquet")
                with open(split_dir / "meta_metrics.json", "w") as f:
                    json.dump(meta_payload, f, indent=2)
                meta_split_metrics.append(meta_payload)
            else:
                meta_pre = FoldPreprocessor(meta_feature_cols, cfg).fit(
                    meta_train_df
                )
                X_meta_train, y_meta_train, w_meta_train = meta_pre.transform(
                    meta_train_df
                )
                X_meta_test, y_meta_test, w_meta_test = meta_pre.transform(
                    meta_test_df
                )

                if len(np.unique(y_meta_train)) < 2:
                    meta_payload["reason"] = "single_class_train"
                    with open(split_dir / "meta_metrics.json", "w") as f:
                        json.dump(meta_payload, f, indent=2)
                    meta_split_metrics.append(meta_payload)
                else:
                    meta_model = LogisticRegression(
                        C=meta_model_params.get("C", 1.0),
                        penalty=meta_model_params.get("penalty", "l2"),
                        solver=meta_model_params.get("solver", "lbfgs"),
                        max_iter=meta_model_params.get("max_iter", 200),
                        class_weight=meta_model_params.get("class_weight"),
                        random_state=seed,
                    )
                    meta_model.fit(
                        X_meta_train,
                        y_meta_train,
                        sample_weight=w_meta_train
                        if weight_enabled
                        else None,
                    )

                    p_meta = meta_model.predict_proba(X_meta_test)[:, 1]
                    m_pred = (p_meta >= meta_threshold).astype(int)
                    meta_metrics = compute_metrics(
                        y_meta_test,
                        p_meta,
                        threshold=meta_threshold,
                        sample_weight=w_meta_test if weight_enabled else None,
                    )
                    tn, fp, fn, tp = confusion_matrix(
                        y_meta_test, m_pred, labels=[0, 1]
                    ).ravel()
                    meta_payload.update(
                        {
                            "metrics": meta_metrics,
                            "acceptance_rate": float(m_pred.mean()),
                            "confusion_matrix": {
                                "tn": int(tn),
                                "fp": int(fp),
                                "fn": int(fn),
                                "tp": int(tp),
                                "threshold": float(meta_threshold),
                            },
                        }
                    )
                    with open(split_dir / "meta_metrics.json", "w") as f:
                        json.dump(meta_payload, f, indent=2)

                    if meta_write_preds:
                        weights_meta = (
                            w_meta_test
                            if weight_enabled
                            else np.ones(len(meta_test_df), dtype=float)
                        )
                        meta_preds = pd.DataFrame(
                            {
                                "event_id": meta_test_df["event_id"].to_numpy(),
                                "m_true": y_meta_test,
                                "p_meta": p_meta,
                                "m_pred": m_pred,
                                "take_trade": m_pred,
                                "p_primary": meta_test_df[
                                    "p_primary"
                                ].to_numpy(),
                                "weight": weights_meta,
                            }
                        )
                        meta_preds.to_parquet(
                            split_dir / "meta_preds.parquet"
                        )

                    if meta_write_model:
                        with open(split_dir / "meta_model.pkl", "wb") as f:
                            pickle.dump(meta_model, f)

                    meta_split_metrics.append(meta_payload)

        if write_model:
            bundle = {
                "model": model,
                "feature_columns": feature_cols,
                "preprocessor": pre.state(),
                "training_config_hash": training_config_hash,
            }
            with open(split_dir / "bundle.pkl", "wb") as f:
                pickle.dump(bundle, f)
            with open(split_dir / "model.pkl", "wb") as f:
                pickle.dump(model, f)

        split_metrics.append(metrics_payload)

    summary = {
        "cv_kind": cv_kind,
        "n_splits": len(split_metrics),
        "metrics_by_split": split_metrics,
        "metrics_mean": {},
    }

    metric_keys = split_metrics[0]["metrics"].keys()
    for key in metric_keys:
        vals = [
            m["metrics"][key]
            for m in split_metrics
            if m["metrics"][key] is not None
        ]
        summary["metrics_mean"][key] = (
            float(np.mean(vals)) if vals else None
        )

    with open(training_dir / "summary.json", "w") as f:
        if meta_enabled:
            summary["meta"] = {
                "enabled": True,
                "metrics_by_split": meta_split_metrics,
                "metrics_mean": {},
            }
            metric_keys = []
            for m in meta_split_metrics:
                if m.get("metrics"):
                    metric_keys = list(m["metrics"].keys())
                    break
            for key in metric_keys:
                vals = [
                    m["metrics"][key]
                    for m in meta_split_metrics
                    if m.get("metrics") and m["metrics"][key] is not None
                ]
                summary["meta"]["metrics_mean"][key] = (
                    float(np.mean(vals)) if vals else None
                )
        json.dump(summary, f, indent=2)

    feature_schema_hash = _load_schema_hash(
        bar_dir / "feature_schema.json", "feature_schema"
    )
    label_schema_hash = _load_schema_hash(
        bar_dir / "label_schema.json", "label_schema"
    )
    weight_schema_hash = _load_schema_hash(
        bar_dir / "weight_schema.json", "weight_schema"
    )
    cv_schema_hash = _load_schema_hash(bar_dir / "cv_schema.json", "cv_schema")

    schema_path = training_dir / "training_schema.json"
    schema_hash = write_training_schema(
        output_path=schema_path,
        training_config=training_config,
        feature_schema_hash=feature_schema_hash,
        label_schema_hash=label_schema_hash,
        weight_schema_hash=weight_schema_hash,
        cv_schema_hash=cv_schema_hash,
        model_kind=model_kind,
        model_params=model_params,
        seed=seed,
        cv_kind=cv_kind,
        n_splits=len(split_metrics),
        meta_enabled=meta_enabled,
        meta_config=meta_cfg,
        code_version="1.0.0",
    )

    return {
        "training_dir": training_dir,
        "training_schema_path": schema_path,
        "training_schema_hash": schema_hash,
        "n_splits": len(split_metrics),
    }
