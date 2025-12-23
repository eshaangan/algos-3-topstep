"""
Tests for offline backtest components.
"""

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting_v3.decisions import decide_trades
from backtesting_v3.risk import RiskManager
from backtesting_v3.fills import apply_forced_flatten, FillResult
from backtesting_v3.simulator import run_backtest
from core.instrument import load_instrument_from_execution_spec
from cli import build_backtest_command

INSTRUMENT_SPEC = load_instrument_from_execution_spec(
    Path(__file__).parent.parent / "configs" / "execution_spec.yaml"
)

def test_decision_meta_filters_trades():
    events_df = pd.DataFrame({"event_id": [1, 2], "t0": [0, 1]})
    primary = pd.DataFrame({"event_id": [1, 2], "y_prob": [0.6, 0.6]})
    meta = pd.DataFrame({"event_id": [1, 2], "p_meta": [0.4, 0.6]})
    cfg = {
        "decision": {
            "use_meta": True,
            "primary_threshold": 0.5,
            "meta_threshold": 0.5,
            "require_meta_for_trade": True,
        }
    }
    decided = decide_trades(events_df, primary, meta, cfg)
    assert decided.loc[decided["event_id"] == 1, "accept"].item() is False
    assert decided.loc[decided["event_id"] == 2, "accept"].item() is True


def test_risk_daily_loss_gate_skips_trades():
    risk_cfg = {
        "topstep": {"starting_balance": 1000},
        "daily_loss_limit": {
            "enabled": True,
            "max_daily_loss": 100,
            "reset_time": "17:00",
            "reset_timezone": "America/Chicago",
            "breach_action": "halt_trading",
        },
        "trailing_drawdown": {"enabled": False},
        "intraday_controls": {"max_trades_per_day": 10, "min_seconds_between_trades": 0, "max_consecutive_losses": 10},
    }
    rm = RiskManager(risk_cfg)
    entry_ts = pd.Timestamp("2025-01-01 15:00:00", tz="UTC")
    rm.record_trade(entry_ts, entry_ts, pnl_usd=-150)
    can_trade, reason = rm.can_trade(entry_ts + pd.Timedelta(minutes=1))
    assert can_trade is False
    assert reason in ["risk_daily_loss", "halted"]


def test_forced_flatten_closes_position():
    index = pd.date_range("2025-01-01 20:00:00", periods=5, freq="1min", tz="UTC")
    bars_df = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=index)
    fill = FillResult(
        entry_ts=index[0],
        entry_px=100.0,
        exit_ts=index[4],
        exit_px=104.0,
        exit_reason="event_exit",
    )
    flattened = apply_forced_flatten(fill, bars_df, "14:01")
    assert flattened.exit_ts == index[1]
    assert flattened.exit_px == 101.0


def test_build_backtest_writes_artifacts_and_updates_manifest(tmp_path):
    run_dir = tmp_path / "run_bt"
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

    label_schema = {"schema_version": "1.0.0", "cost_mode": "gross_in_events"}
    with open(bar_dir / "label_schema.json", "w") as f:
        json.dump(label_schema, f)

    cv_splits = {
        "bar_size": "1m",
        "purged_kfold": [
            {
                "fold": 0,
                "train_event_ids": [0],
                "test_event_ids": [1],
                "test_interval": {"start": index[1].isoformat(), "end": index[3].isoformat()},
                "purge": {"n_purged": 0, "n_embargoed": 0},
                "params": {"n_splits": 1, "embargo_bars": 0},
            }
        ],
    }
    with open(bar_dir / "cv_splits.json", "w") as f:
        json.dump(cv_splits, f, indent=2)

    train_dir = bar_dir / "training" / "purged_kfold" / "fold_0"
    train_dir.mkdir(parents=True)
    preds = pd.DataFrame({"event_id": [1], "y_prob": [0.9], "y_true": [0], "y_pred": [1], "weight": [1.0]})
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
        "decision": {"use_meta": False, "primary_threshold": 0.5, "meta_threshold": 0.5, "require_meta_for_trade": True},
        "sizing": {"contracts": 1, "max_concurrent_positions": 1},
        "session": {"flatten_time_chicago": "15:55"},
        "outputs": {"write_trade_log": True, "write_equity_curve": True},
    }
    backtest_path = tmp_path / "backtest.yaml"
    with open(backtest_path, "w") as f:
        json.dump(backtest_cfg, f)

    exec_cfg = {
        "instrument": {
            "symbol": "MES",
            "tick_size_points": 0.25,
            "contract_multiplier_usd_per_point": 5.0,
        },
        "fill_model": {"fill_price": "next_bar_open"},
        "costs": {"slippage_ticks": {"1m": 0.0}, "commission_per_contract": 0.0},
    }
    exec_path = tmp_path / "execution_spec.yaml"
    with open(exec_path, "w") as f:
        json.dump(exec_cfg, f)

    risk_cfg = {
        "topstep": {"starting_balance": 50000},
        "daily_loss_limit": {"enabled": False},
        "trailing_drawdown": {"enabled": False},
        "intraday_controls": {"max_trades_per_day": 10, "min_seconds_between_trades": 0, "max_consecutive_losses": 10},
    }
    risk_path = tmp_path / "risk.yaml"
    with open(risk_path, "w") as f:
        json.dump(risk_cfg, f)

    args = SimpleNamespace(
        run_dir=str(run_dir),
        training_dir=str(run_dir),
        backtest_config=str(backtest_path),
        execution_spec=str(exec_path),
        risk_config=str(risk_path),
        cv_kind="purged_kfold",
    )
    build_backtest_command(args)

    split_dir = bar_dir / "backtests" / "purged_kfold" / "fold_0"
    assert (split_dir / "trades.parquet").exists()
    assert (split_dir / "equity.parquet").exists()
    assert (split_dir / "backtest_metrics.json").exists()

    with open(run_dir / "run_manifest.json", "r") as f:
        updated = json.load(f)
    artifacts = updated["per_bar_artifacts"]["1m"]
    assert "backtest_dir" in artifacts
    assert "backtest_schema_path" in artifacts
    assert "backtest_schema_hash" in artifacts


def test_mtm_daily_loss_forces_exit_early():
    index = pd.date_range("2025-01-01 09:30:00", periods=4, freq="1min")
    bars_df = pd.DataFrame(
        {"close": [100.0, 90.0, 90.0, 90.0]}, index=index
    )
    events_df = pd.DataFrame(
        {
            "event_id": [1],
            "t0": [index[0]],
            "t1": [index[-1]],
            "entry_time": [index[0]],
            "entry_price": [100.0],
            "t_touch": [index[-1]],
            "exit_price": [90.0],
        }
    )
    primary_preds = pd.DataFrame({"event_id": [1], "y_prob": [0.9]})
    exec_spec = {
        "instrument": {
            "symbol": "MES",
            "tick_size_points": 0.25,
            "contract_multiplier_usd_per_point": 5.0,
        },
        "fill_model": {"fill_price": "next_bar_open"},
        "costs": {"slippage_ticks": {"1m": 0.0}, "commission_per_contract": 0.0},
    }
    risk_cfg = {
        "topstep": {"starting_balance": 1000},
        "daily_loss_limit": {"enabled": True, "max_daily_loss": 5, "pnl_calculation": "realized_and_unrealized", "reset_time": "17:00", "reset_timezone": "America/Chicago"},
        "trailing_drawdown": {"enabled": False},
        "intraday_controls": {"max_trades_per_day": 10, "min_seconds_between_trades": 0, "max_consecutive_losses": 10},
    }
    backtest_cfg = {
        "decision": {
            "use_meta": False,
            "primary_threshold": 0.5,
            "meta_threshold": 0.5,
            "require_meta_for_trade": True,
        }
    }

    trades_df, _, metrics = run_backtest(
        events_df=events_df,
        bars_df=bars_df,
        primary_preds_df=primary_preds,
        meta_preds_df=None,
        execution_spec=exec_spec,
        instrument_spec=INSTRUMENT_SPEC,
        label_schema={"cost_mode": "gross_in_events"},
        risk_cfg=risk_cfg,
        backtest_cfg=backtest_cfg,
        bar_size="1m",
    )
    assert trades_df.loc[0, "exit_ts"] == index[1]
    assert trades_df.loc[0, "liquidation_reason"] == "daily_loss_breach"
    assert metrics["mtm_daily_loss_liquidations"] == 1


def test_mtm_trailing_dd_forces_exit_early():
    index = pd.date_range("2025-01-01 09:30:00", periods=4, freq="1min")
    bars_df = pd.DataFrame(
        {"close": [100.0, 90.0, 90.0, 90.0]}, index=index
    )
    events_df = pd.DataFrame(
        {
            "event_id": [2],
            "t0": [index[0]],
            "t1": [index[-1]],
            "entry_time": [index[0]],
            "entry_price": [100.0],
            "t_touch": [index[-1]],
            "exit_price": [90.0],
        }
    )
    primary_preds = pd.DataFrame({"event_id": [2], "y_prob": [0.9]})
    exec_spec = {
        "instrument": {
            "symbol": "MES",
            "tick_size_points": 0.25,
            "contract_multiplier_usd_per_point": 5.0,
        },
        "fill_model": {"fill_price": "next_bar_open"},
        "costs": {"slippage_ticks": {"1m": 0.0}, "commission_per_contract": 0.0},
    }
    risk_cfg = {
        "topstep": {"starting_balance": 1000},
        "daily_loss_limit": {"enabled": False},
        "trailing_drawdown": {"enabled": True, "max_drawdown": 5, "pnl_calculation": "realized_and_unrealized", "hwm_update_policy": "end_of_day"},
        "intraday_controls": {"max_trades_per_day": 10, "min_seconds_between_trades": 0, "max_consecutive_losses": 10},
    }
    backtest_cfg = {
        "decision": {
            "use_meta": False,
            "primary_threshold": 0.5,
            "meta_threshold": 0.5,
            "require_meta_for_trade": True,
        }
    }

    trades_df, _, metrics = run_backtest(
        events_df=events_df,
        bars_df=bars_df,
        primary_preds_df=primary_preds,
        meta_preds_df=None,
        execution_spec=exec_spec,
        instrument_spec=INSTRUMENT_SPEC,
        label_schema={"cost_mode": "gross_in_events"},
        risk_cfg=risk_cfg,
        backtest_cfg=backtest_cfg,
        bar_size="1m",
    )
    assert trades_df.loc[0, "exit_ts"] == index[1]
    assert trades_df.loc[0, "liquidation_reason"] == "trailing_dd_breach"
    assert metrics["mtm_trailing_dd_liquidations"] == 1


def test_backtest_does_not_double_count_costs_when_events_ret_net():
    index = pd.date_range("2025-01-01 09:30:00", periods=2, freq="1min")
    bars_df = pd.DataFrame({"close": [100.0, 100.0]}, index=index)
    events_df = pd.DataFrame(
        {
            "event_id": [1],
            "t0": [index[0]],
            "t1": [index[1]],
            "entry_time": [index[0]],
            "entry_price": [100.0],
            "t_touch": [index[1]],
            "exit_price": [100.0],
            "ret_net": [-0.5],
        }
    )
    primary_preds = pd.DataFrame({"event_id": [1], "y_prob": [0.9]})
    exec_spec = {
        "instrument": {
            "symbol": "MES",
            "tick_size_points": 0.25,
            "contract_multiplier_usd_per_point": 5.0,
        },
        "fill_model": {"fill_price": "next_bar_open"},
        "costs": {"slippage_ticks": {"1m": 2.0}, "commission_per_contract": 0.0},
    }
    backtest_cfg = {"decision": {"use_meta": False, "primary_threshold": 0.5}}
    risk_cfg = {
        "topstep": {"starting_balance": 1000},
        "daily_loss_limit": {"enabled": False},
        "trailing_drawdown": {"enabled": False},
        "intraday_controls": {"max_trades_per_day": 10, "min_seconds_between_trades": 0, "max_consecutive_losses": 10},
    }

    trades_df, _, _ = run_backtest(
        events_df=events_df,
        bars_df=bars_df,
        primary_preds_df=primary_preds,
        meta_preds_df=None,
        execution_spec=exec_spec,
        instrument_spec=INSTRUMENT_SPEC,
        label_schema={"cost_mode": "net_in_events"},
        risk_cfg=risk_cfg,
        backtest_cfg=backtest_cfg,
        bar_size="1m",
    )
    pnl_usd = trades_df.loc[0, "pnl_usd"]
    expected = events_df.loc[0, "ret_net"] * INSTRUMENT_SPEC.point_value_usd
    assert pnl_usd == expected
