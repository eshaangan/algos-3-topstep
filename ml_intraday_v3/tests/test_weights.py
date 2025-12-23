"""
Tests for sample weight computation.
"""

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import logging

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from weights import (
    map_event_intervals_to_index,
    compute_concurrency,
    compute_uniqueness_weights,
    compute_magnitude_weights,
)
from cli import build_weights_command


def test_uniqueness_weights_simple_overlap():
    index = pd.date_range("2025-01-01 09:30:00", periods=5, freq="1min")
    events_df = pd.DataFrame(
        {
            "event_id": [0, 1, 2],
            "t0": [index[0], index[1], index[4]],
            "t1": [index[2], index[3], index[4]],
        }
    )

    start_idx, end_idx = map_event_intervals_to_index(events_df, index)
    concurrency = compute_concurrency(len(index), start_idx, end_idx)

    inv_conc = np.zeros_like(concurrency, dtype=float)
    inv_conc[concurrency > 0] = 1.0 / concurrency[concurrency > 0]
    inv_prefix = np.concatenate([[0.0], np.cumsum(inv_conc)])

    w = compute_uniqueness_weights(start_idx, end_idx, inv_prefix)

    assert np.isclose(w[0], 2.0 / 3.0)
    assert np.isclose(w[1], 2.0 / 3.0)
    assert np.isclose(w[2], 1.0)


def test_magnitude_weights_prefers_ret_net():
    events_df = pd.DataFrame(
        {
            "ret_net": [1.0, -2.0, 0.5],
            "ret_gross": [10.0, 10.0, 10.0],
        }
    )
    w = compute_magnitude_weights(events_df)
    assert np.allclose(w.to_numpy(), np.array([1.0, 2.0, 0.5]))


def test_build_weights_writes_artifacts_and_updates_manifest(tmp_path):
    run_dir = tmp_path / "run_weights"
    run_dir.mkdir()

    timestamps = pd.date_range(
        "2025-01-01 09:30:00", periods=6, freq="1min"
    )
    bars_df = pd.DataFrame(
        {
            "open": [100.0] * 6,
            "high": [100.5] * 6,
            "low": [99.5] * 6,
            "close": [100.0] * 6,
            "is_synthetic": [False] * 6,
        },
        index=timestamps,
    )

    events_df = pd.DataFrame(
        {
            "event_id": [0, 1],
            "t0": [timestamps[0], timestamps[1]],
            "t1": [timestamps[3], timestamps[4]],
            "ret_net": [1.0, -0.5],
            "y": [1, -1],
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

    labeling_config = {
        "sample_weights": {
            "uniqueness": {"enabled": True, "method": "lopez_de_prado"},
            "magnitude": {"enabled": False},
            "weight_formula": {
                "uniqueness_exponent": 1.0,
                "magnitude_exponent": 0.0,
            },
        }
    }
    labeling_path = tmp_path / "labeling.yaml"
    with open(labeling_path, "w") as f:
        json.dump(labeling_config, f)

    args = SimpleNamespace(
        run_dir=str(run_dir),
        labeling_config=str(labeling_path),
    )

    build_weights_command(args)

    for bar_size in ["1m", "5m"]:
        bar_dir = run_dir / f"bar_size={bar_size}"
        assert (bar_dir / "weights.parquet").exists()
        assert (bar_dir / "weight_schema.json").exists()

    with open(run_dir / "run_manifest.json", "r") as f:
        updated = json.load(f)

    for bar_size in ["1m", "5m"]:
        artifacts = updated["per_bar_artifacts"][bar_size]
        assert "weights_path" in artifacts
        assert "weight_schema_path" in artifacts
        assert "weight_schema_hash" in artifacts
        assert artifacts["n_weighted_events"] == 2


def test_weights_deterministic():
    index = pd.date_range("2025-01-01 09:30:00", periods=10, freq="1min", tz="UTC")
    events_df = pd.DataFrame(
        {
            "event_id": [0, 1, 2, 3],
            "t0": [index[0], index[2], index[3], index[6]],
            "t1": [index[4], index[5], index[7], index[9]],
        }
    )

    start_idx_1, end_idx_1 = map_event_intervals_to_index(events_df, index)
    concurrency_1 = compute_concurrency(len(index), start_idx_1, end_idx_1)
    inv_1 = np.zeros_like(concurrency_1, dtype=float)
    inv_1[concurrency_1 > 0] = 1.0 / concurrency_1[concurrency_1 > 0]
    prefix_1 = np.concatenate([[0.0], np.cumsum(inv_1)])
    w_1 = compute_uniqueness_weights(start_idx_1, end_idx_1, prefix_1)

    start_idx_2, end_idx_2 = map_event_intervals_to_index(events_df, index)
    concurrency_2 = compute_concurrency(len(index), start_idx_2, end_idx_2)
    inv_2 = np.zeros_like(concurrency_2, dtype=float)
    inv_2[concurrency_2 > 0] = 1.0 / concurrency_2[concurrency_2 > 0]
    prefix_2 = np.concatenate([[0.0], np.cumsum(inv_2)])
    w_2 = compute_uniqueness_weights(start_idx_2, end_idx_2, prefix_2)

    assert np.array_equal(start_idx_1, start_idx_2)
    assert np.array_equal(end_idx_1, end_idx_2)
    assert np.allclose(w_1, w_2)


def test_magnitude_weights_clips_when_enabled():
    events_df = pd.DataFrame({"ret_net": [1.0, 2.0, 1000.0, 3.0]})
    w = compute_magnitude_weights(events_df, clip_quantiles=(0.0, 0.75))
    expected_high = events_df["ret_net"].abs().quantile(0.75)

    assert w.iloc[2] <= expected_high
    assert (w >= 0.0).all()


def test_uniqueness_skips_or_errors_on_misalignment(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    run_dir = tmp_path / "run_weights_misaligned"
    run_dir.mkdir()

    index = pd.date_range(
        "2025-01-01 09:30:00", periods=6, freq="1min"
    )
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
            "event_id": [0, 1],
            "t0": [index[1] + pd.Timedelta(seconds=30), index[2]],
            "t1": [index[4], index[5]],
            "ret_net": [1.0, 2.0],
            "y": [1, 1],
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

    labeling_config = {
        "sample_weights": {
            "uniqueness": {"enabled": True, "method": "lopez_de_prado"},
            "magnitude": {"enabled": False},
            "weight_formula": {
                "uniqueness_exponent": 1.0,
                "magnitude_exponent": 0.0,
            },
        }
    }
    labeling_path = tmp_path / "labeling.yaml"
    with open(labeling_path, "w") as f:
        json.dump(labeling_config, f)

    args = SimpleNamespace(
        run_dir=str(run_dir),
        labeling_config=str(labeling_path),
    )

    build_weights_command(args)

    for bar_size in ["1m", "5m"]:
        weights_df = pd.read_parquet(
            run_dir / f"bar_size={bar_size}" / "weights.parquet"
        )
        assert len(weights_df) == 1

    assert "Skipping 1 events with invalid t0/t1 alignment" in caplog.text
