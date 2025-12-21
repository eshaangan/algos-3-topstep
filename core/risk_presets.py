"""
Risk presets and selection helper for backtest/live/analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Optional

from core.simple_config import RiskConfig, RISK_CONFIG as DEFAULT_RISK_CONFIG

# Single switch used by backtest/live/analysis.
RISK_PRESET_NAME: str = os.getenv("RISK_PRESET_NAME", "DEFAULT")


@dataclass(frozen=True)
class RiskPreset:
    name: str
    risk_config: RiskConfig
    profit_target: float
    consistency_limit: Optional[float] = None


TOPSTEP_50K_RISK_CONFIG = RiskConfig(
    starting_balance=50000.0,
    max_daily_loss=1000.0,
    trailing_drawdown=2000.0,
    fixed_risk_per_trade=DEFAULT_RISK_CONFIG.fixed_risk_per_trade,
    max_contracts=5,
    tick_size=0.25,
    tick_value=1.25,
    session_start=DEFAULT_RISK_CONFIG.session_start,
    session_end=DEFAULT_RISK_CONFIG.session_end,
)

TOPSTEP_50K_PROFIT_TARGET = 3000.0
TOPSTEP_50K_CONSISTENCY_LIMIT = 1500.0

DEFAULT_RISK_PRESET = RiskPreset(
    name="DEFAULT",
    risk_config=DEFAULT_RISK_CONFIG,
    profit_target=0.0,
    consistency_limit=None,
)

TOPSTEP_50K_PRESET = RiskPreset(
    name="TOPSTEP_50K",
    risk_config=TOPSTEP_50K_RISK_CONFIG,
    profit_target=TOPSTEP_50K_PROFIT_TARGET,
    consistency_limit=TOPSTEP_50K_CONSISTENCY_LIMIT,
)

_PRESETS: Dict[str, RiskPreset] = {
    DEFAULT_RISK_PRESET.name: DEFAULT_RISK_PRESET,
    TOPSTEP_50K_PRESET.name: TOPSTEP_50K_PRESET,
}


def get_risk_config(preset_name: str) -> RiskConfig:
    preset = _PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(f"Unknown risk preset: {preset_name}")
    return preset.risk_config


def get_risk_preset(preset_name: str) -> RiskPreset:
    preset = _PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(f"Unknown risk preset: {preset_name}")
    return preset
