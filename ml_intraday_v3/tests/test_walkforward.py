"""
Tests for walk-forward evaluation.
"""

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from walkforward.windows import compute_walkforward_windows
from cli import run_walkforward_command


def test_window_generation_no_overlap():
    index = pd.date_range("2025-01-01", periods=15, freq="1D", tz="UTC")
    windows = compute_walkforward_windows(
        index, train_window_days=5, test_window_days=2, step_days=2
    )
    assert windows
    for window in windows:
        assert window["train_end"] < window["test_start"]


def test_walkforward_writes_window_artifacts(tmp_path):
    run_dir = tmp_path / "run_wf"
    bar_dir = run_dir / "bar_size=1m"
    bar_dir.mkdir(parents=True)

    index = pd.date_range("2025-01-01", periods=10, freq="1D", tz="UTC")
    bars_df = pd.DataFrame({"close": range(100, 110)}, index=index)
    bars_df.to_parquet(bar_dir / "bars.parquet")

    features_df = pd.DataFrame({"f1": range(10)}, index=index)
    features_df.to_parquet(bar_dir / "features.parquet")

    events_df = pd.DataFrame(
        {
            "event_id": [0, 1, 2, 3],
            "t0": [index[0], index[1], index[2], index[3]],
            "t1": [index[1], index[2], index[3], index[4]],
            "entry_time": [index[0], index[1], index[2], index[3]],
            "entry_price": [100.0, 101.0, 102.0, 103.0],
            "exit_price": [101.0, 102.0, 103.0, 104.0],
            "y": [0, 1, 0, 1],
        }
    )
    events_df.to_parquet(bar_dir / "events.parquet")

    label_schema = {"schema_version": "1.0.0", "cost_mode": "gross_in_events"}
    with open(bar_dir / "label_schema.json", "w") as f:
        json.dump(label_schema, f)

    weights_df = pd.DataFrame(
        {"event_id": [0, 1, 2, 3], "w_final": [1.0, 1.0, 1.0, 1.0]}
    )
    weights_df.to_parquet(bar_dir / "weights.parquet")

    training_cfg = {
        "model": {"kind": "logreg", "params": {"C": 1.0}},
        "features": {"use_columns": "all", "drop_meta_columns": []},
        "target": {"column": "y", "positive_label": 1},
        "sample_weight": {"column": "w_final", "enabled": False},
        "preprocessing": {"scaler": "none", "impute": "none"},
        "eval": {"threshold": 0.5},
        "meta": {"enabled": False},
        "seed": 42,
    }

    manifest = {
        "run_id": "unit_test_run",
        "timestamp": "2025-01-01T00:00:00Z",
        "bar_sizes": ["1m"],
        "configs": [
            {
                "name": "training",
                "path": "ml_intraday_v3/configs/training.yaml",
                "content_hash": "test",
                "content": training_cfg,
            }
        ],
        "per_bar_artifacts": {},
    }
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    wf_cfg = {
        "schedule": {
            "train_window_days": 2,
            "test_window_days": 1,
            "step_days": 1,
            "min_train_events": 1,
        },
        "retrain_policy": {"max_model_age_days": 7, "retrain_on_drift": False},
        "evaluation": {
            "bar_sizes": ["1m"],
            "thresholds": {"primary_threshold": 0.5, "meta_threshold": 0.5},
            "use_meta": False,
        },
    }
    wf_path = tmp_path / "walkforward.yaml"
    with open(wf_path, "w") as f:
        json.dump(wf_cfg, f)

    args = SimpleNamespace(run_dir=str(run_dir), walkforward_config=str(wf_path))
    run_walkforward_command(args)

    window_dir = run_dir / "walkforward" / "bar_size=1m" / "window_0"
    assert (window_dir / "model_bundle.pkl").exists()
    assert (window_dir / "preds.parquet").exists()
    assert (window_dir / "trades.parquet").exists()
    assert (window_dir / "metrics.json").exists()
