"""
Smoke tests for experiment runner.
"""

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from cli import run_experiments_command


def test_experiments_smoke(tmp_path):
    run_dir = tmp_path / "run_exp"
    bar_dir = run_dir / "bar_size=1m"
    bar_dir.mkdir(parents=True)

    index = pd.date_range("2025-01-01 09:30:00", periods=4, freq="1min")
    bars_df = pd.DataFrame(
        {
            "open": [100.0, 100.1, 100.2, 100.3],
            "high": [100.2, 100.2, 100.3, 100.4],
            "low": [99.9, 100.0, 100.1, 100.2],
            "close": [100.1, 100.2, 100.3, 100.4],
        },
        index=index,
    )
    bars_df.to_parquet(bar_dir / "bars.parquet")

    events_df = pd.DataFrame(
        {
            "event_id": [0, 1],
            "t0": [index[0], index[1]],
            "t1": [index[2], index[3]],
            "entry_time": [index[1], index[2]],
            "entry_price": [100.1, 100.2],
            "t_touch": [index[2], index[3]],
            "exit_price": [100.3, 100.4],
            "y": [1, 0],
        }
    )
    events_df.to_parquet(bar_dir / "events.parquet")

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

    train_dir = bar_dir / "training" / "purged_kfold" / "fold_0"
    train_dir.mkdir(parents=True)
    preds = pd.DataFrame(
        {
            "event_id": [1],
            "y_prob": [0.9],
            "y_true": [0],
            "y_pred": [1],
            "weight": [1.0],
        }
    )
    preds.to_parquet(train_dir / "preds.parquet")

    manifest = {
        "run_id": "unit_test_run",
        "timestamp": "2025-01-01T00:00:00Z",
        "bar_sizes": ["1m"],
        "configs": [],
        "per_bar_artifacts": {},
    }
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    backtest_cfg = {
        "decision": {
            "use_meta": False,
            "primary_threshold": 0.5,
            "meta_threshold": 0.5,
            "require_meta_for_trade": True,
        },
        "sizing": {"contracts": 1, "max_concurrent_positions": 1},
        "session": {"flatten_time_chicago": "15:55"},
        "outputs": {"write_trade_log": True, "write_equity_curve": True},
    }
    backtest_path = tmp_path / "backtest.yaml"
    with open(backtest_path, "w") as f:
        json.dump(backtest_cfg, f)

    training_cfg = {"model": {"kind": "logreg", "params": {"C": 1.0}}}
    training_path = tmp_path / "training.yaml"
    with open(training_path, "w") as f:
        json.dump(training_cfg, f)

    exec_cfg = {
        "fill_model": {"fill_price": "next_bar_open"},
        "costs": {"slippage_ticks": {"1m": 0.0}, "commission_per_contract": 0.0},
    }
    exec_path = tmp_path / "execution_spec.yaml"
    with open(exec_path, "w") as f:
        json.dump(exec_cfg, f)

    risk_cfg = {
        "topstep": {"starting_balance": 50000, "contract_multiplier": 5},
        "daily_loss_limit": {"enabled": False},
        "trailing_drawdown": {"enabled": False},
        "intraday_controls": {
            "max_trades_per_day": 10,
            "min_seconds_between_trades": 0,
            "max_consecutive_losses": 10,
        },
    }
    risk_path = tmp_path / "risk.yaml"
    with open(risk_path, "w") as f:
        json.dump(risk_cfg, f)

    grid_cfg = {
        "cv_kind": "purged_kfold",
        "training_config": str(training_path),
        "backtest_config": str(backtest_path),
        "execution_spec": str(exec_path),
        "risk_config": str(risk_path),
        "grid": {"primary_threshold": [0.4, 0.6]},
        "diagnostics": {"compute_pbo": False, "compute_dsr": False},
    }
    grid_path = tmp_path / "experiment_grid.yaml"
    with open(grid_path, "w") as f:
        json.dump(grid_cfg, f)

    args = SimpleNamespace(run_dir=str(run_dir), grid_config=str(grid_path))
    run_experiments_command(args)

    exp_root = run_dir / "experiments"
    exp_dirs = [p for p in exp_root.iterdir() if p.is_dir()]
    assert exp_dirs
    exp_dir = exp_dirs[0]
    assert (exp_dir / "config_snapshot.json").exists()
    assert (exp_dir / "results.parquet").exists()
    assert (exp_dir / "pbo.json").exists()
    assert (exp_dir / "dsr.json").exists()

    results = pd.read_parquet(exp_dir / "results.parquet")
    assert len(results) == 2
    leaderboard = exp_root / "leaderboard_bar_size=1m.parquet"
    assert leaderboard.exists()
