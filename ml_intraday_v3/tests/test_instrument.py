"""
Tests for instrument config and pnl identity usage.
"""

import json

import pandas as pd
import pytest

from ml_intraday_v3.core.instrument import (
    load_instrument_from_execution_spec,
    InstrumentSpec,
    validate_risk_config_no_instrument_economics,
)
from ml_intraday_v3.audit.checks_accounting import check_pnl_identity


def test_instrument_spec_validates_point_value(tmp_path):
    bad_cfg = tmp_path / "execution_spec.yaml"
    bad_cfg.write_text(
        "\n".join(
            [
                "instrument:",
                "  symbol: MES",
                "  tick_size_points: 0.0",
                "  contract_multiplier_usd_per_point: 5.0",
            ]
        )
    )
    with pytest.raises(ValueError, match="tick_size_points"):
        load_instrument_from_execution_spec(bad_cfg)


def test_risk_config_disallows_instrument_economics_by_default():
    risk_cfg = {
        "topstep": {
            "starting_balance": 50000,
            "contract_multiplier": 5,
            "tick_value": 1.25,
        }
    }
    with pytest.raises(ValueError, match="Instrument economics must live in execution_spec.yaml"):
        validate_risk_config_no_instrument_economics(risk_cfg)

    risk_cfg["topstep"]["allow_instrument_override"] = True
    validate_risk_config_no_instrument_economics(risk_cfg)


def test_pnl_identity_uses_explicit_point_value_not_inferred(tmp_path):
    bar_dir = tmp_path / "bar_size=1m"
    trades_dir = bar_dir / "backtests" / "purged_kfold" / "fold_0"
    trades_dir.mkdir(parents=True)

    label_schema = {"schema_version": "1.0.0", "cost_mode": "gross_in_events"}
    with open(bar_dir / "label_schema.json", "w") as f:
        json.dump(label_schema, f, indent=2)
    backtest_schema = {
        "schema_version": "1.0.0",
        "pnl_mode": "compute_from_prices_then_subtract_costs",
    }
    backtest_dir = bar_dir / "backtests" / "purged_kfold"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    with open(backtest_dir / "backtest_schema.json", "w") as f:
        json.dump(backtest_schema, f, indent=2)

    trades_df = pd.DataFrame(
        {
            "event_id": [1],
            "pnl_points": [1.0],
            "pnl_usd": [5.0],
            "costs_usd": [0.0],
            "executed": [True],
            "cost_mode": ["price_minus_costs"],
        }
    )
    trades_df.to_parquet(trades_dir / "trades.parquet")

    instrument_spec = InstrumentSpec(
        symbol="MES",
        tick_size_points=0.25,
        contract_multiplier_usd_per_point=10.0,
        currency="USD",
    )

    result = check_pnl_identity(bar_dir, instrument_spec=instrument_spec)
    assert result["status"] == "FAIL"
    assert result["violations"] == 1
