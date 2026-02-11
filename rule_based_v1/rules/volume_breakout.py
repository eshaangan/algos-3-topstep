"""Volume Breakout Filter - FILTER rule.

PASS: volume >= min_ratio * avg AND < max_ratio * avg.
VETO: volume too low (no conviction) OR spike too high (news/reversal risk).
"""

import numpy as np
import pandas as pd

from rules.base import BaseRule, RuleSignal
from utils.indicators import volume_ratio


class VolumeBreakoutRule(BaseRule):
    def __init__(
        self,
        lookback: int = 20,
        min_ratio: float = 1.2,
        max_ratio: float = 2.0,
    ):
        super().__init__(name="volume_breakout", role="filter")
        self.lookback = lookback
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def required_bars(self) -> int:
        return self.lookback + 1

    def evaluate(self, bars: pd.DataFrame) -> RuleSignal:
        if len(bars) < self.required_bars():
            return RuleSignal.no_signal(self.name, "Insufficient bars")

        vol_ratio = volume_ratio(bars["volume"], self.lookback)
        current_ratio = vol_ratio.iloc[-1]

        if np.isnan(current_ratio):
            return RuleSignal(
                direction=0, strength=0.0, rule_name=self.name,
                reason="NaN volume ratio",
                metadata={"vetoed": True},
            )

        if current_ratio < self.min_ratio:
            return RuleSignal(
                direction=0, strength=0.0, rule_name=self.name,
                reason=f"Volume too low: {current_ratio:.2f}x avg (need {self.min_ratio}x)",
                metadata={"vetoed": True, "volume_ratio": current_ratio},
            )

        if current_ratio >= self.max_ratio:
            return RuleSignal(
                direction=0, strength=0.0, rule_name=self.name,
                reason=f"Volume spike: {current_ratio:.2f}x avg (max {self.max_ratio}x)",
                metadata={"vetoed": True, "volume_ratio": current_ratio},
            )

        # Volume is in acceptable range - pass
        return RuleSignal(
            direction=0, strength=0.0, rule_name=self.name,
            reason=f"Volume OK: {current_ratio:.2f}x avg",
            metadata={"vetoed": False, "volume_ratio": current_ratio},
        )
