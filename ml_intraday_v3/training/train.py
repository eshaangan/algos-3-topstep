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
from sklearn.metrics import confusion_matrix, roc_auc_score
import lightgbm as lgb

from ml_intraday_v3.run_manifest import hash_content
from .rare_events import RelogitClassifier

from .dataset import build_event_dataset, build_meta_dataset
from .preprocess import FoldPreprocessor
from .metrics import compute_metrics, compute_multiclass_metrics
from .schema import write_training_schema
from ml_intraday_v3.features.calibration import MulticlassIsotonicCalibrator

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


def _multiclass_target(y, classes):
    classes = list(classes)
    mapping = {c: i for i, c in enumerate(classes)}
    y_arr = np.asarray(y)
    out = np.full(len(y_arr), -1, dtype=int)
    for i, val in enumerate(y_arr):
        out[i] = mapping.get(val, -1)
    if (out < 0).any():
        missing = sorted(set(y_arr[out < 0].tolist()))
        raise ValueError(f"Found labels not in target.classes: {missing}")
    return out


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

    cfg = training_config
    target_cfg = cfg.get("target", {})
    target_col = target_cfg.get("column", "y")
    positive_label = target_cfg.get("positive_label", 1)
    target_mode = target_cfg.get("mode", "binary")
    target_classes = target_cfg.get("classes", [-1, 0, 1])
    flip_sign = bool(target_cfg.get("flip_sign", False))
    flip_ret_net = bool(target_cfg.get("flip_ret_net", False))
    side_filter = target_cfg.get("side_filter")

    events_path = bar_dir / "events.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"events.parquet not found: {events_path}")
    events_df = pd.read_parquet(events_path)
    events_df_for_meta = events_df
    if flip_sign:
        if target_mode != "multiclass":
            raise ValueError("target.flip_sign is only supported for multiclass targets")
        if target_classes != [-1, 0, 1]:
            raise ValueError("target.flip_sign expects target.classes == [-1, 0, 1]")
        dataset = dataset.copy()
        dataset[target_col] = -dataset[target_col].astype(int)
        events_df_for_meta = events_df.copy()
        events_df_for_meta["y"] = -events_df_for_meta["y"].astype(int)
        if flip_ret_net and "ret_net" in events_df_for_meta.columns:
            events_df_for_meta["ret_net"] = -events_df_for_meta["ret_net"].astype(float)
        logger.info(
            "Applied target.flip_sign to training labels%s",
            " (ret_net inverted)" if flip_ret_net else "",
        )
    if side_filter is not None:
        if "side" not in dataset.columns:
            raise ValueError("target.side_filter specified but dataset has no side column")
        side_filter = int(side_filter)
        dataset = dataset[dataset["side"] == side_filter].copy()
        if "side" in events_df_for_meta.columns:
            events_df_for_meta = events_df_for_meta[
                events_df_for_meta["side"] == side_filter
            ].copy()

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

    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    model_cfg = cfg.get("model", {})
    model_kind = model_cfg.get("kind", "logreg")
    if model_kind not in ["logreg", "lgbm"]:
        raise ValueError(f"Unsupported model kind: {model_kind}. Must be 'logreg' or 'lgbm'.")

    model_params = model_cfg.get("params", {})
    training_config_hash = hash_content(training_config)
    weight_cfg = cfg.get("sample_weight", {})
    weight_enabled = bool(weight_cfg.get("enabled", False))

    threshold = cfg.get("eval", {}).get("threshold", 0.5)
    write_predictions = bool(cfg.get("output", {}).get("write_predictions", True))
    write_model = bool(cfg.get("output", {}).get("write_model", True))

    # Check if rare events corrections are enabled (only applies to logreg)
    rare_events_cfg = cfg.get("rare_events", {})
    rare_events_enabled = bool(rare_events_cfg.get("enabled", False))

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

    # Calibration configuration
    calib_cfg = cfg.get("calibration", {})
    calib_enabled = bool(calib_cfg.get("enabled", False))
    calib_method = calib_cfg.get("method", "isotonic")
    calib_fraction = float(calib_cfg.get("calibration_fraction", 0.20))
    calib_recalc_score = bool(calib_cfg.get("recalibrate_score_ev", True))
    calib_save = bool(calib_cfg.get("save_calibrator", True))
    calib_seed = int(calib_cfg.get("calibration_seed", 42))

    split_metrics = []
    meta_split_metrics = []

    for split in splits:
        split_id = split.get(split_id_key)
        train_ids = split.get("train_event_ids", [])
        test_ids = split.get("test_event_ids", [])

        avail_ids = set(dataset.index.to_list())
        orig_train_count = len(train_ids)
        orig_test_count = len(test_ids)
        train_ids = [eid for eid in train_ids if eid in avail_ids]
        test_ids = [eid for eid in test_ids if eid in avail_ids]

        dropped_train = orig_train_count - len(train_ids)
        dropped_test = orig_test_count - len(test_ids)
        if dropped_train or dropped_test:
            logger.warning(
                "Dropping %d train and %d test event_ids not present in dataset",
                dropped_train,
                dropped_test,
            )

        if not train_ids or not test_ids:
            raise ValueError(
                f"Split {split_id} has no usable events after filtering "
                f"(train={len(train_ids)}, test={len(test_ids)})"
            )

        train_df = dataset.loc[train_ids].sort_index()
        test_df = dataset.loc[test_ids].sort_index()

        pre = FoldPreprocessor(feature_cols, cfg).fit(train_df)
        X_train, y_train, w_train = pre.transform(train_df)
        X_test, y_test, w_test = pre.transform(test_df)

        # Initialize calibrator as None (will be set if calibration is enabled for multiclass)
        calibrator = None
        calib_report = None
        proba_test_raw = None

        if target_mode == "multiclass":
            if model_kind != "lgbm":
                raise ValueError("multiclass target requires model.kind == 'lgbm'")
            y_train_mc = _multiclass_target(
                train_df[target_col].to_numpy(), target_classes
            )
            y_test_mc = _multiclass_target(
                test_df[target_col].to_numpy(), target_classes
            )
            if len(np.unique(y_train_mc)) < 2:
                raise ValueError(
                    f"Split {split_id} has single-class training labels"
                )

            labels_idx = list(range(len(target_classes)))

            # Calibration: split training data for calibration fitting
            calibrator = None
            if calib_enabled and target_mode == "multiclass":
                n_train = len(X_train)
                n_calib = int(n_train * calib_fraction)
                n_model = n_train - n_calib

                # Deterministic shuffle for reproducibility
                rng = np.random.RandomState(calib_seed + split_id)
                perm = rng.permutation(n_train)

                model_idx = perm[:n_model]
                calib_idx = perm[n_model:]

                X_model = X_train[model_idx]
                y_model = y_train_mc[model_idx]
                w_model = w_train[model_idx] if weight_enabled else None

                X_calib = X_train[calib_idx]
                y_calib = y_train_mc[calib_idx]

                # Get original class labels for calibration
                y_calib_orig = np.asarray([target_classes[i] for i in y_calib])

                logger.info(
                    f"Split {split_id}: Using {n_model} samples for training, "
                    f"{n_calib} for calibration"
                )
            else:
                X_model = X_train
                y_model = y_train_mc
                w_model = w_train if weight_enabled else None

            model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=len(target_classes),
                n_estimators=model_params.get("n_estimators", 500),
                learning_rate=model_params.get("learning_rate", 0.05),
                num_leaves=model_params.get("num_leaves", 31),
                max_depth=model_params.get("max_depth", 6),
                min_child_samples=model_params.get("min_child_samples", 100),
                subsample=model_params.get("subsample", 0.8),
                subsample_freq=model_params.get("subsample_freq", 5),
                colsample_bytree=model_params.get("colsample_bytree", 0.8),
                reg_alpha=model_params.get("reg_alpha", 0.1),
                reg_lambda=model_params.get("reg_lambda", 0.1),
                random_state=seed,
                verbose=-1,
                force_col_wise=True,
            )
            model.fit(
                X_model,
                y_model,
                sample_weight=w_model,
            )

            # Get raw (uncalibrated) predictions
            proba_test_raw = model.predict_proba(X_test)
            proba_train = model.predict_proba(X_train)

            # Fit and apply isotonic calibration
            if calib_enabled:
                proba_calib = model.predict_proba(X_calib)

                calibrator = MulticlassIsotonicCalibrator(classes=target_classes)
                calibrator.fit(proba_calib, y_calib_orig)

                # Apply calibration to test predictions
                proba_test = calibrator.transform(proba_test_raw)

                # Generate calibration report
                calib_report = calibrator.get_calibration_report(proba_calib, y_calib_orig)
                logger.info(
                    f"Split {split_id} calibration: "
                    f"ECE improved from {calib_report['summary']['mean_ece_before']:.4f} "
                    f"to {calib_report['summary']['mean_ece_after']:.4f}"
                )
            else:
                proba_test = proba_test_raw

            y_pred_mc = np.argmax(proba_test, axis=1).astype(int)

            metrics = compute_multiclass_metrics(
                y_test_mc,
                proba_test,
                labels=labels_idx,
                sample_weight=w_test if weight_enabled else None,
            )

            # Optional comparability metric: AUC(target vs rest) if target label exists
            auc_target = None
            if 1 in target_classes:
                target_idx = target_classes.index(1)
                y_bin = (y_test_mc == target_idx).astype(int)
                if len(np.unique(y_bin)) >= 2:
                    auc_target = roc_auc_score(
                        y_bin,
                        proba_test[:, target_idx],
                        sample_weight=w_test if weight_enabled else None,
                    )
            metrics["roc_auc_target_vs_rest"] = float(auc_target) if auc_target is not None else None

            cm = confusion_matrix(
                y_test_mc, y_pred_mc, labels=labels_idx
            )
            cm_payload = {
                "labels_idx": labels_idx,
                "labels": [int(x) for x in target_classes],
                "matrix": cm.astype(int).tolist(),
            }
        else:
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

            # PHASE 5: Support both LogisticRegression and LightGBM
            if model_kind == "logreg":
                if rare_events_enabled:
                    # Use RelogitClassifier with King & Zeng corrections
                    model = RelogitClassifier(
                        tau=rare_events_cfg.get("tau"),  # None = auto-estimate
                        use_sample_weights=rare_events_cfg.get("use_sample_weights", True),
                        weight_method=rare_events_cfg.get("weight_method", "king_zeng"),
                        correction_method=rare_events_cfg.get("correction_method", "king_zeng"),
                        C=model_params.get("C", 1.0),
                        penalty=model_params.get("penalty", "l2"),
                        solver=model_params.get("solver", "lbfgs"),
                        max_iter=model_params.get("max_iter", 200),
                        random_state=seed,
                    )
                else:
                    # Standard LogisticRegression
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
            elif model_kind == "lgbm":
                model = lgb.LGBMClassifier(
                    objective="binary",
                    n_estimators=model_params.get("n_estimators", 500),
                    learning_rate=model_params.get("learning_rate", 0.05),
                    num_leaves=model_params.get("num_leaves", 31),
                    max_depth=model_params.get("max_depth", 6),
                    min_child_samples=model_params.get("min_child_samples", 100),
                    subsample=model_params.get("subsample", 0.8),
                    subsample_freq=model_params.get("subsample_freq", 5),
                    colsample_bytree=model_params.get("colsample_bytree", 0.8),
                    reg_alpha=model_params.get("reg_alpha", 0.1),
                    reg_lambda=model_params.get("reg_lambda", 0.1),
                    random_state=seed,
                    verbose=-1,
                    force_col_wise=True,
                )
                model.fit(
                    X_train,
                    y_train_bin,
                    sample_weight=w_train if weight_enabled else None,
                )

            y_prob = model.predict_proba(X_test)[:, 1]
            y_prob_train = model.predict_proba(X_train)[:, 1]

            # Store uncorrected probabilities if using rare events corrections
            y_prob_uncorrected = None
            if model_kind == "logreg" and rare_events_enabled and hasattr(model, 'lr_model_'):
                # Get uncorrected probabilities from the underlying LogisticRegression model
                y_prob_uncorrected = model.lr_model_.predict_proba(X_test)[:, 1]

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
            cm_payload = {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "threshold": float(threshold),
            }

        split_dir = training_dir / f"{prefix}_{split_id}"
        split_dir.mkdir(parents=True, exist_ok=True)

        metrics_payload = {
            "split_id": split_id,
            "cv_kind": cv_kind,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_dropped_train": int(dropped_train),
            "n_dropped_test": int(dropped_test),
            "target_mode": target_mode,
            "confusion_matrix": cm_payload,
            "metrics": metrics,
        }
        if target_mode == "binary":
            metrics_payload["pos_rate_train"] = float(y_train_bin.mean())
            metrics_payload["pos_rate_test"] = float(y_test_bin.mean())
            metrics_payload["base_rate_train"] = float(y_train_bin.mean())
            metrics_payload["base_rate_test"] = float(y_test_bin.mean())
        with open(split_dir / "metrics.json", "w") as f:
            json.dump(metrics_payload, f, indent=2)

        if write_predictions:
            weights = (
                w_test if weight_enabled else np.ones(len(test_df), dtype=float)
            )
            if target_mode == "multiclass":
                # Provide explicit per-class probs + EV-style score for backtest decisions.
                classes = list(target_classes)
                p_target = None
                p_stop = None
                p_vertical = None
                if 1 in classes:
                    p_target = proba_test[:, classes.index(1)]
                if -1 in classes:
                    p_stop = proba_test[:, classes.index(-1)]
                if 0 in classes:
                    p_vertical = proba_test[:, classes.index(0)]
                score_ev = (
                    (p_target if p_target is not None else 0.0)
                    - (p_stop if p_stop is not None else 0.0)
                )
                y_pred_orig = np.asarray([classes[i] for i in y_pred_mc], dtype=int)
                y_true_orig = test_df[target_col].to_numpy().astype(int)

                preds_data = {
                    "event_id": test_df["event_id"].to_numpy(),
                    "y_true": y_true_orig,
                    "y_pred": y_pred_orig,
                    "p_target": p_target,
                    "p_stop": p_stop,
                    "p_vertical": p_vertical,
                    "score_ev": score_ev,
                    "weight": weights,
                }

                # Add raw (uncalibrated) probabilities if calibration was applied
                if calib_enabled and calibrator is not None:
                    preds_data["p_target_raw"] = proba_test_raw[:, classes.index(1)] if 1 in classes else None
                    preds_data["p_stop_raw"] = proba_test_raw[:, classes.index(-1)] if -1 in classes else None
                    preds_data["p_vertical_raw"] = proba_test_raw[:, classes.index(0)] if 0 in classes else None
                    # Also save raw score_ev for comparison
                    p_target_raw = preds_data.get("p_target_raw")
                    p_stop_raw = preds_data.get("p_stop_raw")
                    preds_data["score_ev_raw"] = (
                        (p_target_raw if p_target_raw is not None else 0.0)
                        - (p_stop_raw if p_stop_raw is not None else 0.0)
                    )

                preds_df = pd.DataFrame(preds_data)
            else:
                preds_data = {
                    "event_id": test_df["event_id"].to_numpy(),
                    "y_true": y_test_bin,
                    "p_target": y_prob,  # Renamed from y_prob for consistency with multiclass
                    "y_pred": y_pred,
                    "weight": weights,
                }
                # Add uncorrected probabilities if rare events corrections were applied
                if y_prob_uncorrected is not None:
                    preds_data["p_target_uncorrected"] = y_prob_uncorrected
                preds_df = pd.DataFrame(preds_data)
            preds_df.to_parquet(split_dir / "preds.parquet")

        if meta_enabled:
            primary_train_preds = pd.DataFrame(
                {
                    "event_id": train_df["event_id"].to_numpy(),
                    "y_prob": y_prob_train if target_mode == "binary" else proba_train[:, list(target_classes).index(1)] if 1 in target_classes else proba_train[:, 0],
                }
            )
            primary_test_preds = pd.DataFrame(
                {
                    "event_id": test_df["event_id"].to_numpy(),
                    "y_prob": y_prob if target_mode == "binary" else proba_test[:, list(target_classes).index(1)] if 1 in target_classes else proba_test[:, 0],
                }
            )

            meta_train_df, meta_feature_cols = build_meta_dataset(
                primary_preds_df=primary_train_preds,
                base_event_dataset_df=dataset.reset_index(drop=True),
                events_df=events_df_for_meta,
                config=cfg,
            )
            meta_test_df, _ = build_meta_dataset(
                primary_preds_df=primary_test_preds,
                base_event_dataset_df=dataset.reset_index(drop=True),
                events_df=events_df_for_meta,
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
                    # PHASE 5: Support both LogisticRegression and LightGBM for meta-model
                    meta_model_kind = meta_model_cfg.get("kind", "logreg")
                    if meta_model_kind == "logreg":
                        meta_model = LogisticRegression(
                            C=meta_model_params.get("C", 1.0),
                            penalty=meta_model_params.get("penalty", "l2"),
                            solver=meta_model_params.get("solver", "lbfgs"),
                            max_iter=meta_model_params.get("max_iter", 200),
                            class_weight=meta_model_params.get("class_weight"),
                            random_state=seed,
                        )
                    elif meta_model_kind == "lgbm":
                        meta_model = lgb.LGBMClassifier(
                            objective="binary",
                            n_estimators=meta_model_params.get("n_estimators", 300),
                            learning_rate=meta_model_params.get("learning_rate", 0.05),
                            num_leaves=meta_model_params.get("num_leaves", 15),
                            max_depth=meta_model_params.get("max_depth", 4),
                            min_child_samples=meta_model_params.get("min_child_samples", 50),
                            subsample=meta_model_params.get("subsample", 0.8),
                            subsample_freq=meta_model_params.get("subsample_freq", 5),
                            colsample_bytree=meta_model_params.get("colsample_bytree", 0.8),
                            reg_alpha=meta_model_params.get("reg_alpha", 0.1),
                            reg_lambda=meta_model_params.get("reg_lambda", 0.1),
                            random_state=seed,
                            verbose=-1,
                            force_col_wise=True,
                        )
                    else:
                        raise ValueError(f"Unsupported meta model kind: {meta_model_kind}")

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

        # Save calibrator if calibration was applied
        if calib_enabled and calib_save and calibrator is not None:
            with open(split_dir / "calibrator.pkl", "wb") as f:
                pickle.dump(calibrator, f)
            # Also save calibration report
            if calib_report is not None:
                with open(split_dir / "calibration_report.json", "w") as f:
                    # Convert numpy types to native Python types for JSON
                    def convert_to_json_serializable(obj):
                        if isinstance(obj, (np.integer, np.floating)):
                            return float(obj)
                        elif isinstance(obj, dict):
                            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_to_json_serializable(i) for i in obj]
                        return obj
                    json.dump(convert_to_json_serializable(calib_report), f, indent=2)
            logger.info(f"Saved calibrator to {split_dir / 'calibrator.pkl'}")

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
