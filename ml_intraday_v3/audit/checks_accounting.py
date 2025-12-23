"""
Accounting checks for cost handling and pnl identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json

import numpy as np
import pandas as pd

from core.instrument import InstrumentSpec


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


def _load_label_schema(bar_dir: Path) -> dict | None:
    path = bar_dir / "label_schema.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _load_backtest_schema(backtest_dir: Path) -> dict | None:
    path = backtest_dir / "backtest_schema.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


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
    label_schema = _load_label_schema(bar_dir)
    backtest_schema = _load_backtest_schema(backtest_dir)
    if label_schema is None:
        return {"status": "FAIL", "reason": "missing_label_schema"}
    if backtest_schema is None:
        return {"status": "FAIL", "reason": "missing_backtest_schema"}

    cost_mode_policy = backtest_schema.get("cost_mode_policy")
    label_cost_mode = label_schema.get("cost_mode")
    if label_cost_mode not in ["net_in_events", "gross_in_events"]:
        return {"status": "FAIL", "reason": "invalid_label_cost_mode"}

    expected_pnl_mode = (
        "use_events_ret_net"
        if label_cost_mode == "net_in_events"
        else "compute_from_prices_then_subtract_costs"
    )
    pnl_mode = backtest_schema.get("pnl_mode")
    if pnl_mode != expected_pnl_mode:
        return {
            "status": "FAIL",
            "reason": "pnl_mode_mismatch",
            "label_cost_mode": label_cost_mode,
            "expected_pnl_mode": expected_pnl_mode,
            "backtest_pnl_mode": pnl_mode,
        }

    issues = 0
    if label_cost_mode == "net_in_events":
        if "ret_net" not in joined.columns or joined["ret_net"].isna().all():
            return {
                "status": "FAIL",
                "reason": "ret_net_missing_for_net_mode",
            }
        mask = (joined.get("exit_reason", "") == "event_exit") & joined[
            "ret_net"
        ].notna()
        if "cost_mode" in joined.columns:
            issues = int(
                (joined.loc[mask, "cost_mode"] != "event_ret_net").sum()
            )
        else:
            issues = int(mask.sum())

    status = "PASS" if issues == 0 else "FAIL"
    return {
        "status": status,
        "issues": issues,
        "cost_mode_policy": cost_mode_policy,
        "label_cost_mode": label_cost_mode,
        "pnl_mode": pnl_mode,
    }


def check_pnl_identity(
    bar_dir: Path,
    instrument_spec: InstrumentSpec,
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

    label_schema = _load_label_schema(bar_dir)
    backtest_schema = _load_backtest_schema(backtest_dir)
    if label_schema is None or backtest_schema is None:
        return {"status": "FAIL", "reason": "missing_schema_for_identity"}

    label_cost_mode = label_schema.get("cost_mode")
    pnl_mode = backtest_schema.get("pnl_mode")
    if label_cost_mode not in ["net_in_events", "gross_in_events"]:
        return {"status": "FAIL", "reason": "invalid_label_cost_mode"}
    expected_pnl_mode = (
        "use_events_ret_net"
        if label_cost_mode == "net_in_events"
        else "compute_from_prices_then_subtract_costs"
    )
    if pnl_mode != expected_pnl_mode:
        return {
            "status": "FAIL",
            "reason": "pnl_mode_mismatch",
            "label_cost_mode": label_cost_mode,
            "expected_pnl_mode": expected_pnl_mode,
            "backtest_pnl_mode": pnl_mode,
        }

    if "pnl_points" not in executed.columns or "pnl_usd" not in executed.columns:
        return {"status": "SKIP", "reason": "missing_pnl_columns"}
    if "costs_usd" not in executed.columns:
        return {"status": "SKIP", "reason": "missing_costs_usd"}

    tick_size = float(instrument_spec.tick_size_points)
    tick_value = float(instrument_spec.tick_value_usd)
    point_value = float(instrument_spec.contract_multiplier_usd_per_point)

    expected = executed["pnl_points"] * point_value - executed["costs_usd"]
    identity_formula = "pnl_usd ≈ pnl_points * point_value - costs_usd"

    diff = (executed["pnl_usd"] - expected).abs()
    violations = int((diff > tolerance).sum())
    status = "PASS" if violations == 0 else "FAIL"

    return {
        "status": status,
        "violations": violations,
        "point_value": point_value,
        "tolerance": tolerance,
        "identity_formula_used": identity_formula,
        "pnl_points_semantics": "gross",
        "instrument": {
            "symbol": instrument_spec.symbol,
            "tick_size": tick_size,
            "tick_value_usd": tick_value,
            "contract_multiplier_usd_per_point": point_value,
            "currency": instrument_spec.currency,
        },
    }
