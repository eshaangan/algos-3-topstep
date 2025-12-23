"""
Tests for leakage-safe validation (purged CV + CPCV).
"""

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation import build_purged_kfold_splits, build_cpcv_paths
from cli import build_cv_command


def test_purged_kfold_removes_overlaps():
    index = pd.date_range("2025-01-01 09:30:00", periods=6, freq="1min")
    events_df = pd.DataFrame(
        {
            "event_id": [0, 1, 2, 3],
            "t0": [index[0], index[1], index[2], index[5]],
            "t1": [index[4], index[1], index[2], index[5]],
        }
    )

    splits = build_purged_kfold_splits(
        events_df=events_df,
        bars_index=index,
        n_splits=2,
        embargo_bars=0,
    )

    fold1 = splits[1]
    assert 0 not in fold1["train_event_ids"]
    assert 1 in fold1["train_event_ids"]


def test_embargo_removes_near_future_t0():
    index = pd.date_range("2025-01-01 09:30:00", periods=6, freq="1min")
    events_df = pd.DataFrame(
        {
            "event_id": [0, 1, 2, 3],
            "t0": [index[0], index[2], index[3], index[5]],
            "t1": [index[1], index[2], index[3], index[5]],
        }
    )

    splits = build_purged_kfold_splits(
        events_df=events_df,
        bars_index=index,
        n_splits=2,
        embargo_bars=1,
    )

    fold0 = splits[0]
    assert 2 not in fold0["train_event_ids"]
    assert 3 in fold0["train_event_ids"]


def test_embargo_anchors_to_next_bar_off_grid():
    index = pd.date_range("2025-01-01 09:30:00", periods=6, freq="1min")
    events_df = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "t0": [index[0], index[1], index[4]],
            "t1": [index[2] + pd.Timedelta(seconds=30), index[1], index[4]],
        }
    )

    splits = build_purged_kfold_splits(
        events_df=events_df,
        bars_index=index,
        n_splits=2,
        embargo_bars=1,
    )

    fold0 = splits[0]
    assert 2 not in fold0["train_event_ids"]


def test_cpcv_combined_interval_purge_applied():
    index = pd.date_range("2025-01-01 09:30:00", periods=7, freq="1min")
    events_df = pd.DataFrame(
        {
            "event_id": [0, 1, 2, 3, 4, 5],
            "t0": [
                index[0],
                index[1],
                index[3],
                index[4],
                index[5],
                index[6],
            ],
            "t1": [
                index[0],
                index[5],
                index[3],
                index[4],
                index[5],
                index[6],
            ],
        }
    )

    base_folds = build_purged_kfold_splits(
        events_df=events_df,
        bars_index=index,
        n_splits=3,
        embargo_bars=0,
    )

    paths = build_cpcv_paths(
        base_folds=base_folds,
        events_df=events_df,
        bars_index=index,
        K=3,
        test_groups=2,
        embargo_bars=0,
        max_paths=None,
        selection="lexicographic",
    )

    path = next(p for p in paths if p["test_folds"] == [1, 2])
    assert 1 not in path["train_event_ids"]


def test_build_cv_writes_artifacts_and_updates_manifest(tmp_path):
    run_dir = tmp_path / "run_cv"
    run_dir.mkdir()

    index = pd.date_range("2025-01-01 09:30:00", periods=6, freq="1min")
    bars_df = pd.DataFrame(
        {
            "open": [100.0] * 6,
            "high": [100.5] * 6,
            "low": [99.5] * 6,
            "close": [100.0] * 6,
            "is_synthetic": [False] * 6,
        },
        index=index,
    )

    events_df = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "t0": [index[0], index[2], index[4]],
            "t1": [index[1], index[3], index[5]],
            "y": [1, -1, 0],
        }
    )

    for bar_size in ["1m", "5m"]:
        bar_dir = run_dir / f"bar_size={bar_size}"
        bar_dir.mkdir()
        bars_df.to_parquet(bar_dir / "bars.parquet")
        events_df.to_parquet(bar_dir / "events.parquet")

    manifest = {
        "run_id": "unit_test_run",
        "timestamp": "2025-01-01T00:00:00Z",
        "bar_sizes": ["1m", "5m"],
        "configs": [],
        "per_bar_artifacts": {},
    }
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    validation_config = {
        "purged_cv": {
            "n_splits": 2,
            "embargo": {
                "enabled": True,
                "embargo_bars": {"1m": 1, "5m": 1},
            },
        },
        "cpcv": {"enabled": False},
    }
    validation_path = tmp_path / "validation.yaml"
    with open(validation_path, "w") as f:
        json.dump(validation_config, f)

    args = SimpleNamespace(
        run_dir=str(run_dir),
        validation_config=str(validation_path),
    )

    build_cv_command(args)

    for bar_size in ["1m", "5m"]:
        bar_dir = run_dir / f"bar_size={bar_size}"
        assert (bar_dir / "cv_splits.json").exists()
        assert (bar_dir / "cv_schema.json").exists()

    with open(run_dir / "run_manifest.json", "r") as f:
        updated = json.load(f)

    for bar_size in ["1m", "5m"]:
        artifacts = updated["per_bar_artifacts"][bar_size]
        assert "cv_splits_path" in artifacts
        assert "cv_schema_path" in artifacts
        assert "cv_schema_hash" in artifacts
        assert artifacts["n_splits"] == 2
