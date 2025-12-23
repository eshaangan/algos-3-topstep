"""
Tests for training scaffolding.
"""

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training import build_event_dataset, build_meta_dataset, FoldPreprocessor
from cli import build_train_command


def test_dataset_join_at_t0_failfast_on_mismatch(tmp_path):
    run_dir = tmp_path / "run_train"
    bar_dir = run_dir / "bar_size=1m"
    bar_dir.mkdir(parents=True)

    index = pd.date_range("2025-01-01 09:30:00", periods=3, freq="1min")
    features_df = pd.DataFrame(
        {"f1": [1.0, 2.0, 3.0]}, index=index
    )
    features_df.to_parquet(bar_dir / "features.parquet")

    events_df = pd.DataFrame(
        {
            "event_id": [0],
            "t0": [index[0] + pd.Timedelta(seconds=30)],
            "t1": [index[1]],
            "y": [1],
        }
    )
    events_df.to_parquet(bar_dir / "events.parquet")

    with open(bar_dir / "feature_schema.json", "w") as f:
        json.dump({"feature_columns": ["f1"], "schema_hash": "feat"}, f)

    try:
        build_event_dataset(run_dir, "1m", training_config={"sample_weight": {"enabled": False}})
        assert False, "Expected ValueError for missing t0 in features index"
    except ValueError as e:
        assert "t0 timestamps not found" in str(e)


def test_preprocess_fit_on_train_only():
    train_df = pd.DataFrame(
        {"event_id": [0, 1], "t0": [0, 1], "y": [1, 0], "w_final": [1, 1], "f1": [0.0, 2.0]}
    )
    test_df = pd.DataFrame(
        {"event_id": [2], "t0": [2], "y": [1], "w_final": [1], "f1": [100.0]}
    )
    cfg = {"preprocessing": {"impute": "median", "scaler": "standard"}}
    pre = FoldPreprocessor(["f1"], cfg).fit(train_df)

    assert np.isclose(pre.means_[0], train_df["f1"].mean())


def test_train_writes_metrics_and_preds(tmp_path):
    run_dir = tmp_path / "run_train"
    bar_dir = run_dir / "bar_size=1m"
    bar_dir.mkdir(parents=True)

    index = pd.date_range("2025-01-01 09:30:00", periods=4, freq="1min")
    features_df = pd.DataFrame(
        {
            "f1": [0.0, 1.0, 2.0, 3.0],
            "f2": [1.0, 0.5, -0.5, -1.0],
            "usable_for_training": [True, True, True, True],
        },
        index=index,
    )
    features_df.to_parquet(bar_dir / "features.parquet")

    events_df = pd.DataFrame(
        {
            "event_id": [0, 1, 2, 3],
            "t0": [index[0], index[1], index[2], index[3]],
            "t1": [index[1], index[2], index[3], index[3]],
            "y": [1, 0, 1, 0],
        }
    )
    events_df.to_parquet(bar_dir / "events.parquet")

    weights_df = pd.DataFrame(
        {"event_id": [0, 1, 2, 3], "w_final": [1.0, 1.0, 1.0, 1.0]}
    )
    weights_df.to_parquet(bar_dir / "weights.parquet")

    cv_splits = {
        "bar_size": "1m",
        "purged_kfold": [
            {
                "fold": 0,
                "train_event_ids": [0, 1, 2],
                "test_event_ids": [3],
                "test_interval": {"start": index[3].isoformat(), "end": index[3].isoformat()},
                "purge": {"n_purged": 0, "n_embargoed": 0},
                "params": {"n_splits": 1, "embargo_bars": 0},
            }
        ],
    }
    with open(bar_dir / "cv_splits.json", "w") as f:
        json.dump(cv_splits, f, indent=2)

    with open(bar_dir / "feature_schema.json", "w") as f:
        json.dump({"feature_columns": ["f1", "f2", "usable_for_training"], "schema_hash": "feat"}, f)
    with open(bar_dir / "label_schema.json", "w") as f:
        json.dump({"schema_hash": "label"}, f)
    with open(bar_dir / "weight_schema.json", "w") as f:
        json.dump({"schema_hash": "weight"}, f)
    with open(bar_dir / "cv_schema.json", "w") as f:
        json.dump({"schema_hash": "cv"}, f)

    manifest = {
        "run_id": "unit_test_run",
        "timestamp": "2025-01-01T00:00:00Z",
        "bar_sizes": ["1m"],
        "configs": [],
        "per_bar_artifacts": {},
    }
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    training_config = {
        "model": {"kind": "logreg", "params": {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 50, "class_weight": None}},
        "features": {"use_columns": "all", "drop_meta_columns": ["usable_for_training"]},
        "target": {"column": "y", "positive_label": 1},
        "sample_weight": {"column": "w_final", "enabled": True},
        "preprocessing": {"scaler": "standard", "impute": "median"},
        "eval": {"threshold": 0.5, "metrics": ["accuracy"]},
        "output": {"write_predictions": True, "write_model": False},
        "seed": 42,
    }
    training_path = tmp_path / "training.yaml"
    with open(training_path, "w") as f:
        json.dump(training_config, f)

    args = SimpleNamespace(
        run_dir=str(run_dir),
        training_config=str(training_path),
        cv_kind="purged_kfold",
    )

    build_train_command(args)

    split_dir = bar_dir / "training" / "purged_kfold" / "fold_0"
    assert (split_dir / "metrics.json").exists()
    assert (split_dir / "preds.parquet").exists()
    assert (bar_dir / "training" / "purged_kfold" / "summary.json").exists()
    assert (bar_dir / "training" / "purged_kfold" / "training_schema.json").exists()


def test_meta_dataset_filters_on_primary_threshold():
    base_df = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "t0": pd.date_range("2025-01-01 09:30:00", periods=3, freq="1min"),
            "y": [1, 0, 1],
            "w_final": [1.0, 1.0, 1.0],
            "f1": [0.1, 0.2, 0.3],
        }
    )
    events_df = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "y": [1, 0, 1],
        }
    )
    primary_preds = pd.DataFrame(
        {"event_id": [0, 1, 2], "y_prob": [0.4, 0.6, 0.8]}
    )
    cfg = {
        "meta": {
            "enabled": True,
            "threshold_primary": 0.5,
            "target": {"kind": "y_positive"},
            "features": {
                "include_primary_prob": True,
                "include_primary_logit": False,
                "include_original_features": True,
            },
        }
    }

    meta_df, _ = build_meta_dataset(primary_preds, base_df, events_df, cfg)
    assert set(meta_df["event_id"]) == {1, 2}


def test_meta_labels_constructed_correctly():
    base_df = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "t0": pd.date_range("2025-01-01 09:30:00", periods=3, freq="1min"),
            "y": [1, 1, 0],
            "w_final": [1.0, 1.0, 1.0],
            "f1": [0.1, 0.2, 0.3],
        }
    )
    events_df = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "y": [1, 1, 0],
            "ret_net": [1.0, -0.5, 2.0],
        }
    )
    primary_preds = pd.DataFrame(
        {"event_id": [0, 1, 2], "y_prob": [0.9, 0.9, 0.9]}
    )
    cfg = {
        "meta": {
            "enabled": True,
            "threshold_primary": 0.1,
            "target": {"kind": "y_and_ret_net"},
            "features": {
                "include_primary_prob": True,
                "include_primary_logit": True,
                "include_original_features": False,
            },
        }
    }

    meta_df, _ = build_meta_dataset(primary_preds, base_df, events_df, cfg)
    assert meta_df["y"].tolist() == [1, 0, 0]


def test_train_writes_meta_artifacts(tmp_path):
    run_dir = tmp_path / "run_train_meta"
    bar_dir = run_dir / "bar_size=1m"
    bar_dir.mkdir(parents=True)

    index = pd.date_range("2025-01-01 09:30:00", periods=4, freq="1min")
    features_df = pd.DataFrame(
        {
            "f1": [0.0, 1.0, 2.0, 3.0],
            "f2": [1.0, 0.5, -0.5, -1.0],
            "usable_for_training": [True, True, True, True],
        },
        index=index,
    )
    features_df.to_parquet(bar_dir / "features.parquet")

    events_df = pd.DataFrame(
        {
            "event_id": [0, 1, 2, 3],
            "t0": [index[0], index[1], index[2], index[3]],
            "t1": [index[1], index[2], index[3], index[3]],
            "y": [1, 0, 1, 0],
            "ret_net": [1.0, -0.5, 0.5, -0.2],
        }
    )
    events_df.to_parquet(bar_dir / "events.parquet")

    weights_df = pd.DataFrame(
        {"event_id": [0, 1, 2, 3], "w_final": [1.0, 1.0, 1.0, 1.0]}
    )
    weights_df.to_parquet(bar_dir / "weights.parquet")

    cv_splits = {
        "bar_size": "1m",
        "purged_kfold": [
            {
                "fold": 0,
                "train_event_ids": [0, 1, 2],
                "test_event_ids": [3],
                "test_interval": {"start": index[3].isoformat(), "end": index[3].isoformat()},
                "purge": {"n_purged": 0, "n_embargoed": 0},
                "params": {"n_splits": 1, "embargo_bars": 0},
            }
        ],
    }
    with open(bar_dir / "cv_splits.json", "w") as f:
        json.dump(cv_splits, f, indent=2)

    with open(bar_dir / "feature_schema.json", "w") as f:
        json.dump({"feature_columns": ["f1", "f2", "usable_for_training"], "schema_hash": "feat"}, f)
    with open(bar_dir / "label_schema.json", "w") as f:
        json.dump({"schema_hash": "label"}, f)
    with open(bar_dir / "weight_schema.json", "w") as f:
        json.dump({"schema_hash": "weight"}, f)
    with open(bar_dir / "cv_schema.json", "w") as f:
        json.dump({"schema_hash": "cv"}, f)

    manifest = {
        "run_id": "unit_test_run",
        "timestamp": "2025-01-01T00:00:00Z",
        "bar_sizes": ["1m"],
        "configs": [],
        "per_bar_artifacts": {},
    }
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    training_config = {
        "model": {"kind": "logreg", "params": {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 50, "class_weight": None}},
        "features": {"use_columns": "all", "drop_meta_columns": ["usable_for_training"]},
        "target": {"column": "y", "positive_label": 1},
        "sample_weight": {"column": "w_final", "enabled": True},
        "preprocessing": {"scaler": "standard", "impute": "median"},
        "eval": {"threshold": 0.5, "metrics": ["accuracy"]},
        "output": {"write_predictions": True, "write_model": False},
        "seed": 42,
        "meta": {
            "enabled": True,
            "threshold_primary": 0.0,
            "target": {"kind": "y_positive"},
            "features": {
                "include_primary_prob": True,
                "include_primary_logit": True,
                "include_original_features": True,
            },
            "model": {"kind": "logreg", "params": {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 50, "class_weight": None}},
            "output": {"write_meta_predictions": True, "write_meta_model": True},
        },
    }
    training_path = tmp_path / "training.yaml"
    with open(training_path, "w") as f:
        json.dump(training_config, f)

    args = SimpleNamespace(
        run_dir=str(run_dir),
        training_config=str(training_path),
        cv_kind="purged_kfold",
    )

    build_train_command(args)

    split_dir = bar_dir / "training" / "purged_kfold" / "fold_0"
    assert (split_dir / "meta_metrics.json").exists()
    assert (split_dir / "meta_preds.parquet").exists()
    assert (split_dir / "meta_model.pkl").exists()
