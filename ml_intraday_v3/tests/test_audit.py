"""
Smoke test for audit harness.
"""

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli import run_audit_command


def test_audit_smoke(tmp_path):
    run_dir = tmp_path / "run_audit"
    bar_dir = run_dir / "bar_size=1m"
    bar_dir.mkdir(parents=True)

    index = pd.date_range("2025-01-01 09:30:00", periods=4, freq="1min")
    bars_df = pd.DataFrame(
        {"open": [100, 101, 102, 103], "high": [101, 102, 103, 104], "low": [99, 100, 101, 102], "close": [100, 101, 102, 103]},
        index=index,
    )
    bars_df.to_parquet(bar_dir / "bars.parquet")

    features_df = pd.DataFrame({"f1": [1.0, 1.1, 1.2, 1.3]}, index=index)
    features_df.to_parquet(bar_dir / "features.parquet")

    events_df = pd.DataFrame(
        {
            "event_id": [0, 1],
            "t0": [index[0], index[1]],
            "t1": [index[2], index[3]],
            "y": [1, 0],
        }
    )
    events_df.to_parquet(bar_dir / "events.parquet")

    weights_df = pd.DataFrame(
        {
            "event_id": [0, 1],
            "w_uniqueness": [1.0, 1.0],
            "w_magnitude": [1.0, 1.0],
            "w_final": [1.0, 1.0],
        }
    )
    weights_df.to_parquet(bar_dir / "weights.parquet")

    cv_splits = {
        "bar_size": "1m",
        "purged_kfold": [
            {
                "fold": 0,
                "train_event_ids": [0],
                "test_event_ids": [1],
                "test_interval": {
                    "start": index[1].isoformat(),
                    "end": index[3].isoformat(),
                },
                "purge": {"n_purged": 0, "n_embargoed": 0},
                "params": {"n_splits": 1, "embargo_bars": 0},
            }
        ],
    }
    with open(bar_dir / "cv_splits.json", "w") as f:
        json.dump(cv_splits, f, indent=2)

    backtest_dir = bar_dir / "backtests" / "purged_kfold"
    split_dir = backtest_dir / "fold_0"
    split_dir.mkdir(parents=True)
    trades_df = pd.DataFrame(
        {
            "event_id": [1],
            "entry_ts": [index[1]],
            "exit_ts": [index[2]],
            "pnl_points": [1.0],
            "pnl_usd": [5.0],
            "costs_usd": [0.0],
            "executed": [True],
            "exit_reason": ["event_exit"],
            "cost_mode": ["price_minus_costs"],
        }
    )
    trades_df.to_parquet(split_dir / "trades.parquet")
    equity_df = pd.DataFrame({"timestamp": [index[2]], "equity": [50005.0], "pnl_usd": [5.0]})
    equity_df.to_parquet(split_dir / "equity.parquet")

    backtest_schema = {
        "schema_version": "1.0.0",
        "backtest_config": {"session": {"flatten_time_chicago": "15:55"}},
        "cost_mode_policy": "price_minus_costs",
    }
    with open(backtest_dir / "backtest_schema.json", "w") as f:
        json.dump(backtest_schema, f, indent=2)

    manifest = {
        "run_id": "unit_test_run",
        "timestamp": "2025-01-01T00:00:00Z",
        "bar_sizes": ["1m"],
        "configs": [],
        "per_bar_artifacts": {},
    }
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    args = SimpleNamespace(run_dir=str(run_dir), strict="false")
    run_audit_command(args)

    report_path = bar_dir / "audit_report.json"
    assert report_path.exists()
    with open(report_path, "r") as f:
        report = json.load(f)
    assert report["bar_size"] == "1m"
    assert "checks" in report
