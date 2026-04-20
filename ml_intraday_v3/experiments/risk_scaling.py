"""
Scale Topstep-style risk YAML for multi-contract backtests.

When sizing.contracts > 1, dollar PnL swings scale roughly linearly. Fixed $500 / $1500
limits cause early halts unless limits scale with the reference contract count.

Use ``scale_risk_config_for_contracts`` for offline experiments only; live accounts
must still obey broker rule text and your funded combine parameters.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _scale_num(value: Any, factor: float) -> Any:
    if value is None:
        return None
    try:
        return float(value) * factor
    except (TypeError, ValueError):
        return value


def scale_risk_config_for_contracts(
    risk_cfg: dict,
    contracts: int,
    *,
    base_contracts: int = 1,
) -> dict:
    """
    Deep-copy risk_cfg and scale dollar limits by contracts / base_contracts.

    Also sets position_limits max_contracts_per_position and max_total_contracts
    to ``contracts`` when those keys exist (mirrors intended position size).

    Non-numeric fields (times, actions, booleans) are unchanged.
    """
    if base_contracts <= 0:
        raise ValueError("base_contracts must be positive")
    if contracts <= 0:
        raise ValueError("contracts must be positive")
    factor = float(contracts) / float(base_contracts)
    out = deepcopy(risk_cfg)

    daily = out.get("daily_loss_limit", {}) or {}
    if "max_daily_loss" in daily:
        daily["max_daily_loss"] = _scale_num(daily["max_daily_loss"], factor)
    out["daily_loss_limit"] = daily

    dd = out.get("trailing_drawdown", {}) or {}
    if "max_drawdown" in dd:
        dd["max_drawdown"] = _scale_num(dd["max_drawdown"], factor)
    out["trailing_drawdown"] = dd

    pos = out.get("position_limits", {}) or {}
    if pos:
        pos["max_contracts_per_position"] = int(contracts)
        pos["max_total_contracts"] = int(contracts)
        if "max_notional_exposure" in pos and pos["max_notional_exposure"] is not None:
            pos["max_notional_exposure"] = _scale_num(pos["max_notional_exposure"], factor)
    out["position_limits"] = pos

    rm = out.get("risk_management", {}) or {}
    for key in (
        "daily_loss_limit_usd",
        "daily_loss_warning_usd",
        "daily_loss_critical_usd",
        "max_drawdown_from_hwm",
        "drawdown_warning",
        "daily_profit_lock_usd",
    ):
        if key in rm and rm[key] is not None:
            rm[key] = _scale_num(rm[key], factor)
    if "base_position_size" in rm:
        rm["base_position_size"] = int(contracts)
    out["risk_management"] = rm

    return out
