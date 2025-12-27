"""
Walk-forward runner.
"""

from __future__ import annotations

import json
import logging
import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from run_manifest import hash_content
from training.dataset import build_event_dataset, build_meta_dataset
from training.preprocess import FoldPreprocessor
from training.metrics import compute_metrics
from backtesting_v3 import run_backtest
from core.instrument import (
    InstrumentSpec,
    load_instrument_from_execution_spec,
    validate_risk_config_no_instrument_economics,
)

from .windows import compute_walkforward_windows
from .metrics import aggregate_window_metrics
from .schema import write_walkforward_schema

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _load_manifest(run_dir: Path) -> dict | None:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _load_schema_hash(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("schema_hash", "")


def _select_config(
    manifest: dict | None, name: str, default_path: Path
) -> tuple[dict, dict]:
    provenance = {"name": name}
    config = {}

    if manifest:
        for entry in manifest.get("configs", []):
            if entry.get("name") != name:
                continue
            if "content" in entry and entry["content"] is not None:
                config = entry["content"]
                provenance.update(
                    {
                        "source": "manifest_content",
                        "path": entry.get("path"),
                        "content_hash": entry.get("content_hash"),
                    }
                )
                return config, provenance
            if entry.get("path"):
                path = Path(entry["path"])
                if path.exists():
                    config = _load_yaml(path)
                    provenance.update(
                        {
                            "source": "manifest_path",
                            "path": str(path),
                            "content_hash": entry.get("content_hash"),
                        }
                    )
                    return config, provenance

    config = _load_yaml(default_path)
    provenance.update(
        {
            "source": "default_path",
            "path": str(default_path),
            "content_hash": None,
            "fallback_used": True,
        }
    )
    return config, provenance


def _binary_target(y, positive_label):
    return (np.asarray(y) == positive_label).astype(int)


def _train_primary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    training_cfg: dict,
) -> dict:
    feature_cols = [
        c for c in train_df.columns if c not in ["event_id", "t0", "y", "w_final"]
    ]
    pre = FoldPreprocessor(feature_cols, training_cfg).fit(train_df)
    X_train, y_train, w_train = pre.transform(train_df)
    X_test, y_test, w_test = pre.transform(test_df)

    target_cfg = training_cfg.get("target", {})
    positive_label = target_cfg.get("positive_label", 1)
    y_train_bin = _binary_target(y_train, positive_label)
    y_test_bin = _binary_target(y_test, positive_label)

    if len(np.unique(y_train_bin)) < 2:
        raise ValueError("Single-class training labels in window")

    model_cfg = training_cfg.get("model", {})
    params = model_cfg.get("params", {})

    model = LogisticRegression(
        C=params.get("C", 1.0),
        penalty=params.get("penalty", "l2"),
        solver=params.get("solver", "lbfgs"),
        max_iter=params.get("max_iter", 200),
        class_weight=params.get("class_weight"),
        random_state=training_cfg.get("seed", 42),
    )
    weight_cfg = training_cfg.get("sample_weight", {})
    use_weight = bool(weight_cfg.get("enabled", False))
    model.fit(X_train, y_train_bin, sample_weight=w_train if use_weight else None)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_prob_train = model.predict_proba(X_train)[:, 1]
    threshold = float(training_cfg.get("eval", {}).get("threshold", 0.5))
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(
        y_test_bin, y_prob, threshold=threshold, sample_weight=w_test if use_weight else None
    )

    return {
        "model": model,
        "preprocessor": pre,
        "feature_cols": feature_cols,
        "y_test": y_test_bin,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "y_prob_train": y_prob_train,
        "w_test": w_test if use_weight else np.ones(len(test_df), dtype=float),
    }


def _train_meta(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    events_df: pd.DataFrame,
    training_cfg: dict,
    primary_probs_train: np.ndarray,
    primary_probs_test: np.ndarray,
) -> dict:
    meta_cfg = training_cfg.get("meta", {})
    if not meta_cfg.get("enabled", False):
        return {"enabled": False}

    primary_train_preds = pd.DataFrame(
        {"event_id": train_df["event_id"].to_numpy(), "y_prob": primary_probs_train}
    )
    primary_test_preds = pd.DataFrame(
        {"event_id": test_df["event_id"].to_numpy(), "y_prob": primary_probs_test}
    )

    meta_train_df, meta_feature_cols = build_meta_dataset(
        primary_train_preds, train_df, events_df, training_cfg
    )
    meta_test_df, _ = build_meta_dataset(
        primary_test_preds, test_df, events_df, training_cfg
    )

    if meta_train_df.empty or meta_test_df.empty:
        return {
            "enabled": True,
            "skipped": True,
            "reason": "no_proposed_trades",
            "meta_test_df": meta_test_df,
        }

    pre = FoldPreprocessor(meta_feature_cols, training_cfg).fit(meta_train_df)
    X_train, y_train, w_train = pre.transform(meta_train_df)
    X_test, y_test, w_test = pre.transform(meta_test_df)

    if len(np.unique(y_train)) < 2:
        return {
            "enabled": True,
            "skipped": True,
            "reason": "single_class_train",
            "meta_test_df": meta_test_df,
        }

    params = meta_cfg.get("model", {}).get("params", {})
    model = LogisticRegression(
        C=params.get("C", 1.0),
        penalty=params.get("penalty", "l2"),
        solver=params.get("solver", "lbfgs"),
        max_iter=params.get("max_iter", 200),
        class_weight=params.get("class_weight"),
        random_state=training_cfg.get("seed", 42),
    )
    weight_cfg = training_cfg.get("sample_weight", {})
    use_weight = bool(weight_cfg.get("enabled", False))
    model.fit(X_train, y_train, sample_weight=w_train if use_weight else None)

    threshold = float(meta_cfg.get("threshold_meta", 0.5))
    p_meta = model.predict_proba(X_test)[:, 1]
    m_pred = (p_meta >= threshold).astype(int)
    metrics = compute_metrics(
        y_test, p_meta, threshold=threshold, sample_weight=w_test if use_weight else None
    )

    return {
        "enabled": True,
        "skipped": False,
        "model": model,
        "preprocessor": pre,
        "feature_cols": meta_feature_cols,
        "meta_test_df": meta_test_df,
        "p_meta": p_meta,
        "m_pred": m_pred,
        "metrics": metrics,
        "w_test": w_test if use_weight else np.ones(len(meta_test_df), dtype=float),
    }


def run_walkforward(run_dir: Path | str, walkforward_config_path: Path | str) -> dict:
    run_dir = Path(run_dir)
    wf_config_path = Path(walkforward_config_path)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    if not wf_config_path.exists():
        raise FileNotFoundError(f"Walkforward config not found: {wf_config_path}")

    wf_cfg = _load_yaml(wf_config_path)
    schedule = wf_cfg.get("schedule", {})
    retrain_policy = wf_cfg.get("retrain_policy", {})
    eval_cfg = wf_cfg.get("evaluation", {})
    bar_sizes = eval_cfg.get("bar_sizes", ["1m", "5m"])

    training_cfg, training_prov = _select_config(
        _load_manifest(run_dir),
        "training",
        Path("ml_intraday_v3/configs/training.yaml"),
    )
    backtest_cfg, backtest_prov = _select_config(
        _load_manifest(run_dir),
        "backtest",
        Path("ml_intraday_v3/configs/backtest.yaml"),
    )
    execution_spec, exec_prov = _select_config(
        _load_manifest(run_dir),
        "execution_spec",
        Path("ml_intraday_v3/configs/execution_spec.yaml"),
    )
    risk_cfg, risk_prov = _select_config(
        _load_manifest(run_dir),
        "risk",
        Path("ml_intraday_v3/configs/risk.yaml"),
    )
    validate_risk_config_no_instrument_economics(risk_cfg)
    instrument_spec = InstrumentSpec.from_execution_spec(execution_spec)

    thresholds = eval_cfg.get("thresholds", {})
    use_meta = bool(eval_cfg.get("use_meta", False))

    base_backtest_cfg = deepcopy(backtest_cfg)
    base_backtest_cfg.setdefault("decision", {})
    if "primary_threshold" in thresholds:
        base_backtest_cfg["decision"]["primary_threshold"] = float(
            thresholds["primary_threshold"]
        )
    if "meta_threshold" in thresholds:
        base_backtest_cfg["decision"]["meta_threshold"] = float(
            thresholds["meta_threshold"]
        )
    base_backtest_cfg["decision"]["use_meta"] = use_meta

    wf_root = run_dir / "walkforward"
    wf_root.mkdir(parents=True, exist_ok=True)

    per_bar = {}
    for bar_size in bar_sizes:
        bar_dir = run_dir / f"bar_size={bar_size}"
        bars_path = bar_dir / "bars.parquet"
        events_path = bar_dir / "events.parquet"
        if not bars_path.exists() or not events_path.exists():
            continue

        bars_df = pd.read_parquet(bars_path)
        events_df = pd.read_parquet(events_path)
        events_df["t0"] = pd.to_datetime(events_df["t0"])
        label_schema_path = bar_dir / "label_schema.json"
        if not label_schema_path.exists():
            raise FileNotFoundError(f"label_schema.json not found: {label_schema_path}")
        with open(label_schema_path, "r") as f:
            label_schema = json.load(f)

        dataset = build_event_dataset(run_dir, bar_size, training_cfg)
        dataset = dataset.sort_values("t0").reset_index(drop=True)

        windows = compute_walkforward_windows(
            bars_df.index,
            int(schedule.get("train_window_days", 60)),
            int(schedule.get("test_window_days", 5)),
            int(schedule.get("step_days", 5)),
            timezone="America/Chicago",
            max_date=schedule.get("max_date"),
            expanding=bool(schedule.get("expanding", True)),
        )

        min_train_events = int(schedule.get("min_train_events", 0))
        max_model_age = int(retrain_policy.get("max_model_age_days", 0))
        retrain_on_drift = bool(retrain_policy.get("retrain_on_drift", False))

        last_model = None
        last_train_end = None
        last_train_range = None

        bar_out_dir = wf_root / f"bar_size={bar_size}"
        bar_out_dir.mkdir(parents=True, exist_ok=True)

        feature_schema_hash = _load_schema_hash(
            bar_dir / "feature_schema.json"
        )
        label_schema_hash = _load_schema_hash(
            bar_dir / "label_schema.json"
        )
        weight_schema_hash = _load_schema_hash(
            bar_dir / "weight_schema.json"
        )
        backtest_schema_hash = _load_schema_hash(
            bar_dir / "backtests" / "purged_kfold" / "backtest_schema.json"
        )

        window_metrics = []
        for idx, window in enumerate(windows):
            train_start = pd.Timestamp(window["train_start"])
            train_end = pd.Timestamp(window["train_end"])
            test_start = pd.Timestamp(window["test_start"])
            test_end = pd.Timestamp(window["test_end"])

            train_mask = (dataset["t0"] >= train_start) & (
                dataset["t0"] <= train_end
            )
            test_mask = (dataset["t0"] >= test_start) & (
                dataset["t0"] <= test_end
            )

            train_df = dataset.loc[train_mask].reset_index(drop=True)
            test_df = dataset.loc[test_mask].reset_index(drop=True)

            window_dir = bar_out_dir / f"window_{idx}"
            window_dir.mkdir(parents=True, exist_ok=True)

            metrics_payload = {
                "window_id": idx,
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "status": "ok",
            }

            if len(train_df) < min_train_events or test_df.empty:
                metrics_payload["status"] = "skipped"
                metrics_payload["reason"] = "insufficient_train_or_test"
                window_metrics.append(metrics_payload)
                with open(window_dir / "metrics.json", "w") as f:
                    json.dump(metrics_payload, f, indent=2)
                continue

            reuse_model = False
            if last_model is not None and last_train_end is not None and max_model_age > 0:
                age_days = (test_start - last_train_end).days
                if age_days <= max_model_age and not retrain_on_drift:
                    reuse_model = True

            if reuse_model:
                primary = last_model
                bundle = None
                metrics_payload["reused_model"] = True

                feature_cols = primary["feature_cols"]
                pre = primary["preprocessor"]
                X_test, y_test, w_test = pre.transform(test_df)
                X_train, y_train, w_train = pre.transform(train_df)

                target_cfg = training_cfg.get("target", {})
                positive_label = target_cfg.get("positive_label", 1)
                y_test_bin = _binary_target(y_test, positive_label)

                y_prob = primary["model"].predict_proba(X_test)[:, 1]
                y_prob_train = primary["model"].predict_proba(X_train)[:, 1]
                threshold = float(training_cfg.get("eval", {}).get("threshold", 0.5))
                y_pred = (y_prob >= threshold).astype(int)

                primary = {
                    "model": primary["model"],
                    "preprocessor": pre,
                    "feature_cols": feature_cols,
                    "y_test": y_test_bin,
                    "y_pred": y_pred,
                    "w_test": w_test,
                }
                meta = _train_meta(
                    train_df,
                    test_df,
                    events_df,
                    training_cfg,
                    primary_probs_train=y_prob_train,
                    primary_probs_test=y_prob,
                )
            else:
                primary = _train_primary(train_df, test_df, training_cfg)
                y_prob = primary["y_prob"]
                y_prob_train = primary["y_prob_train"]

                meta = _train_meta(
                    train_df,
                    test_df,
                    events_df,
                    training_cfg,
                    primary_probs_train=y_prob_train,
                    primary_probs_test=y_prob,
                )

                bundle = None
                last_model = {
                    "model": primary["model"],
                    "preprocessor": primary["preprocessor"],
                    "feature_cols": primary["feature_cols"],
                }
                last_train_end = train_end
                last_train_range = {
                    "start": train_start.isoformat(),
                    "end": train_end.isoformat(),
                }

            primary_preds = pd.DataFrame(
                {
                    "event_id": test_df["event_id"].to_numpy(),
                    "y_prob": y_prob,
                    "y_pred": primary["y_pred"],
                    "y_true": primary["y_test"],
                    "weight": primary["w_test"],
                }
            )
            primary_preds.to_parquet(window_dir / "preds.parquet")

            meta_preds = None
            if meta.get("enabled", False):
                if meta.get("skipped", False):
                    meta_preds = pd.DataFrame(
                        {
                            "event_id": meta.get("meta_test_df", pd.DataFrame()).get("event_id", pd.Series(dtype=int)),
                            "p_meta": pd.Series(dtype=float),
                        }
                    )
                    meta_preds.to_parquet(window_dir / "meta_preds.parquet")
                else:
                    meta_test_df = meta["meta_test_df"]
                    meta_preds = pd.DataFrame(
                        {
                            "event_id": meta_test_df["event_id"].to_numpy(),
                            "m_true": meta_test_df["y"].to_numpy(),
                            "p_meta": meta["p_meta"],
                            "m_pred": meta["m_pred"],
                            "p_primary": meta_test_df["p_primary"].to_numpy(),
                            "weight": meta["w_test"],
                        }
                    )
                    meta_preds.to_parquet(window_dir / "meta_preds.parquet")
                    with open(window_dir / "meta_metrics.json", "w") as f:
                        json.dump(meta["metrics"], f, indent=2)

            test_event_ids = set(test_df["event_id"].tolist())
            test_events = events_df[events_df["event_id"].isin(test_event_ids)].copy()
            test_events = test_events.sort_values("t0")

            local_backtest_cfg = deepcopy(base_backtest_cfg)
            if use_meta and meta_preds is None:
                local_backtest_cfg["decision"]["use_meta"] = False

            trades_df, equity_df, backtest_metrics = run_backtest(
                events_df=test_events,
                bars_df=bars_df,
                primary_preds_df=primary_preds,
                meta_preds_df=meta_preds if local_backtest_cfg["decision"]["use_meta"] else None,
                execution_spec=execution_spec,
                instrument_spec=instrument_spec,
                label_schema=label_schema,
                risk_cfg=risk_cfg,
                backtest_cfg=local_backtest_cfg,
                bar_size=bar_size,
            )

            trades_df.to_parquet(window_dir / "trades.parquet")

            metrics_payload["backtest"] = backtest_metrics
            with open(window_dir / "metrics.json", "w") as f:
                json.dump(metrics_payload, f, indent=2)

            if bundle is None:
                training_range = last_train_range or {
                    "start": train_start.isoformat(),
                    "end": train_end.isoformat(),
                }
                bundle = {
                    "primary_model": primary["model"],
                    "primary_preprocessor": primary["preprocessor"].state(),
                    "primary_feature_columns": primary["feature_cols"],
                    "meta_model": meta.get("model")
                    if meta.get("enabled") and not meta.get("skipped", False)
                    else None,
                    "meta_preprocessor": meta.get("preprocessor").state()
                    if meta.get("enabled") and not meta.get("skipped", False)
                    else None,
                    "meta_feature_columns": meta.get("feature_cols", []),
                    "thresholds": thresholds,
                    "training_range": training_range,
                    "config_hashes": {
                        "training": hash_content(training_cfg),
                        "walkforward": hash_content(wf_cfg),
                        "backtest": hash_content(backtest_cfg),
                        "execution_spec": hash_content(execution_spec),
                        "risk": hash_content(risk_cfg),
                        "instrument": hash_content(execution_spec.get("instrument", {})),
                        "feature_schema": feature_schema_hash,
                        "label_schema": label_schema_hash,
                        "weight_schema": weight_schema_hash,
                        "backtest_schema": backtest_schema_hash,
                    },
                    "provenance": {
                        "training_config": training_prov,
                        "backtest_config": backtest_prov,
                        "execution_spec": exec_prov,
                        "risk_config": risk_prov,
                        "instrument_config": {
                            "source": exec_prov.get("source"),
                            "path": exec_prov.get("path"),
                            "content_hash": hash_content(execution_spec.get("instrument", {})),
                            "derived_from": "execution_spec",
                        },
                    },
                }
                manifest = _load_manifest(run_dir)
                if manifest:
                    bundle["git_state"] = manifest.get("git_state", {})

            with open(window_dir / "model_bundle.pkl", "wb") as f:
                pickle.dump(bundle, f)

            window_metrics.append(
                {
                    "window_id": idx,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "total_pnl_usd": backtest_metrics.get("total_pnl_usd"),
                    "trades_count": backtest_metrics.get("trades_count"),
                    "skipped_count": backtest_metrics.get("skipped_count"),
                }
            )

        summary = aggregate_window_metrics(window_metrics)
        summary_path = bar_out_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "bar_size": bar_size,
                    "n_windows": len(window_metrics),
                    "metrics_by_window": window_metrics,
                    "metrics_mean": summary,
                },
                f,
                indent=2,
            )

        schema_path = bar_out_dir / "walkforward_schema.json"
        schema_hash = write_walkforward_schema(
            output_path=schema_path,
            config_snapshot=wf_cfg,
            summary={
                "bar_size": bar_size,
                "n_windows": len(window_metrics),
                "thresholds": thresholds,
                "use_meta": use_meta,
            },
            code_version="1.0.0",
        )

        per_bar[bar_size] = {
            "walkforward_dir": str(bar_out_dir.relative_to(run_dir)),
            "walkforward_schema_path": str(schema_path.relative_to(run_dir)),
            "walkforward_schema_hash": schema_hash,
            "n_windows": len(window_metrics),
        }

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if "per_bar_artifacts" not in manifest:
            if "per_bar_size_artifacts" in manifest:
                manifest["per_bar_artifacts"] = manifest.get(
                    "per_bar_size_artifacts", {}
                )
            else:
                manifest["per_bar_artifacts"] = {}
        for bar_size, artifacts in per_bar.items():
            if bar_size not in manifest["per_bar_artifacts"]:
                manifest["per_bar_artifacts"][bar_size] = {}
            manifest["per_bar_artifacts"][bar_size].update(artifacts)
        manifest["per_bar_size_artifacts"] = manifest.get("per_bar_artifacts", {})
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    return {"walkforward_dir": str(wf_root), "per_bar": per_bar}
