"""
Accounting checks for cost handling and pnl identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _load_trades(backtest_dir: Path) -> pd.DataFrame:
    trade_files = list(backtest_dir.glob("*/trades.parquet"))
    if not trade_files:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in trade_files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _find_backtest_dir(bar_dir: Path) -> Path | None:
    base = bar_dir / "backtests"
    if not base.exists():
        return None
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    # Prefer purged_kfold if present.
    for name in ["purged_kfold", "cpcv"]:
        candidate = base / name
        if candidate.exists():
            return candidate
    return subdirs[0]


def check_cost_mode(
    bar_dir: Path, events_df: pd.DataFrame
) -> dict:
    backtest_dir = _find_backtest_dir(bar_dir)
    if backtest_dir is None:
        return {"status": "SKIP", "reason": "no_backtests"}

    trades_df = _load_trades(backtest_dir)
    if trades_df.empty:
        return {"status": "SKIP", "reason": "no_trades"}

    executed = trades_df[trades_df.get("executed", False)].copy()
    if executed.empty:
        return {"status": "SKIP", "reason": "no_executed_trades"}

    if "event_id" not in executed.columns:
        return {"status": "FAIL", "reason": "missing_event_id"}

    joined = executed.merge(
        events_df[["event_id", "ret_net"]]
        if "ret_net" in events_df.columns
        else events_df[["event_id"]],
        on="event_id",
        how="left",
    )

    cost_mode = joined.get("cost_mode")
    cost_mode_policy = None
    schema_path = backtest_dir / "backtest_schema.json"
    if schema_path.exists():
        import json

        with open(schema_path, "r") as f:
            schema = json.load(f)
        cost_mode_policy = schema.get("cost_mode_policy")

    issues = 0
    if cost_mode_policy == "event_ret_net_preferred" and "ret_net" in joined.columns:
        mask = (
            joined.get("exit_reason", "") == "event_exit"
        ) & joined["ret_net"].notna()
        if "cost_mode" in joined.columns:
            issues = int((joined.loc[mask, "cost_mode"] != "event_ret_net").sum())
        else:
            issues = int(mask.sum())

    if "costs_usd" in joined.columns and "cost_mode" in joined.columns:
        costs_nonzero = joined.loc[
            joined["cost_mode"] == "event_ret_net", "costs_usd"
        ]
        if not costs_nonzero.empty:
            issues += int((costs_nonzero.abs() > 1e-6).sum())

    status = "PASS" if issues == 0 else "FAIL"
    return {
        "status": status,
        "issues": issues,
        "cost_mode_policy": cost_mode_policy,
    }


def check_pnl_identity(
    bar_dir: Path,
    instrument_params: dict,
    tolerance: float = 1e-6,
) -> dict:
    backtest_dir = _find_backtest_dir(bar_dir)
    if backtest_dir is None:
        return {"status": "SKIP", "reason": "no_backtests"}

    trades_df = _load_trades(backtest_dir)
    if trades_df.empty:
        return {"status": "SKIP", "reason": "no_trades"}

    executed = trades_df[trades_df.get("executed", False)].copy()
    if executed.empty:
        return {"status": "SKIP", "reason": "no_executed_trades"}

    if "pnl_points" not in executed.columns or "pnl_usd" not in executed.columns:
        return {"status": "SKIP", "reason": "missing_pnl_columns"}

    tick_size = float(instrument_params.get("tick_size_points", 0.0))
    tick_value = float(instrument_params.get("tick_value_usd", 0.0))
    contract_multiplier = instrument_params.get("contract_multiplier")

    if tick_size <= 0.0 or tick_value <= 0.0:
        return {
            "status": "SKIP",
            "reason": "invalid_instrument_params",
            "instrument_params": instrument_params,
        }

    point_value = tick_value / tick_size
    if contract_multiplier is not None:
        contract_multiplier = float(contract_multiplier)

    cost_mode = (
        executed["cost_mode"].mode().iloc[0]
        if "cost_mode" in executed.columns
        else None
    )
    pnl_semantics = "unknown"
    if cost_mode in ["price_minus_costs", "event_ret_net"]:
        pnl_semantics = "net_of_costs"

    if pnl_semantics == "net_of_costs":
        expected = executed["pnl_points"] * point_value
        identity_formula = "pnl_usd ≈ pnl_points * point_value"
    else:
        if "costs_usd" not in executed.columns:
            return {
                "status": "SKIP",
                "reason": "missing_costs_usd_for_gross_check",
                "pnl_points_semantics": pnl_semantics,
            }
        expected = executed["pnl_points"] * point_value - executed["costs_usd"]
        identity_formula = "pnl_usd ≈ pnl_points * point_value - costs_usd"

    diff = (executed["pnl_usd"] - expected).abs()
    violations = int((diff > tolerance).sum())
    status = "PASS" if violations == 0 else "FAIL"

    params_check = None
    if contract_multiplier is not None:
        params_check = {
            "contract_multiplier": contract_multiplier,
            "point_value": point_value,
            "consistent": abs(point_value - contract_multiplier) <= 1e-6,
        }

    return {
        "status": status,
        "violations": violations,
        "point_value": point_value,
        "tolerance": tolerance,
        "identity_formula_used": identity_formula,
        "pnl_points_semantics": pnl_semantics,
        "instrument_params": instrument_params,
        "instrument_consistency": params_check,
    }
