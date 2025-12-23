"""
Instrument specification loader and validation.

Execution spec is the single source of truth for instrument economics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    tick_size_points: float
    contract_multiplier_usd_per_point: float
    currency: str = "USD"

    def validate(self) -> None:
        if self.tick_size_points <= 0:
            raise ValueError("tick_size_points must be > 0")
        if self.contract_multiplier_usd_per_point <= 0:
            raise ValueError("contract_multiplier_usd_per_point must be > 0")

    @property
    def tick_value_usd(self) -> float:
        return self.tick_size_points * self.contract_multiplier_usd_per_point

    @property
    def point_value_usd(self) -> float:
        return self.contract_multiplier_usd_per_point

    @staticmethod
    def from_execution_spec(execution_spec: dict) -> "InstrumentSpec":
        instrument = execution_spec.get("instrument")
        if not instrument:
            raise ValueError(
                "execution_spec.instrument missing; define instrument in execution_spec.yaml"
            )
        required = ["symbol", "tick_size_points", "contract_multiplier_usd_per_point"]
        missing = [k for k in required if k not in instrument]
        if missing:
            raise ValueError(f"Missing instrument fields: {missing}")
        spec = InstrumentSpec(
            symbol=str(instrument["symbol"]),
            tick_size_points=float(instrument["tick_size_points"]),
            contract_multiplier_usd_per_point=float(
                instrument["contract_multiplier_usd_per_point"]
            ),
            currency=str(instrument.get("currency", "USD")),
        )
        spec.validate()
        return spec


def load_instrument_from_execution_spec(
    execution_spec_path: Path | str,
) -> InstrumentSpec:
    """
    Load execution_spec YAML and extract instrument spec.
    """
    path = Path(execution_spec_path)
    if not path.exists():
        raise FileNotFoundError(f"Execution spec not found: {path}")
    with open(path, "r") as f:
        execution_spec = yaml.safe_load(f) or {}
    return InstrumentSpec.from_execution_spec(execution_spec)


def validate_risk_config_no_instrument_economics(risk_cfg: dict) -> None:
    topstep = risk_cfg.get("topstep", {}) if isinstance(risk_cfg, dict) else {}
    allow_override = bool(topstep.get("allow_instrument_override", False))
    forbidden = []
    for key in ["contract_multiplier", "tick_value"]:
        if key in topstep:
            forbidden.append(key)
    if forbidden and not allow_override:
        raise ValueError(
            "Instrument economics must live in execution_spec.yaml instrument: {...}. "
            "Remove topstep.contract_multiplier/tick_value from risk.yaml."
        )
