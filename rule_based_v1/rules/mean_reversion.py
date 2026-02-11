"""Mean Reversion Rule - CONFIRMATION signal.

Confirms LONG when price is near lower Bollinger Band or RSI is oversold.
Confirms SHORT when price is near upper Bollinger Band or RSI is overbought.
"""

import numpy as np
import pandas as pd

from rules.base import BaseRule, RuleSignal
from utils.indicators import bollinger_position, rsi


class MeanReversionRule(BaseRule):
    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        long_bb_threshold: float = 0.3,
        short_bb_threshold: float = 0.7,
        rsi_period: int = 14,
        rsi_long_threshold: float = 35.0,
        rsi_short_threshold: float = 65.0,
    ):
        super().__init__(name="mean_reversion", role="confirmation")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.long_bb_threshold = long_bb_threshold
        self.short_bb_threshold = short_bb_threshold
        self.rsi_period = rsi_period
        self.rsi_long_threshold = rsi_long_threshold
        self.rsi_short_threshold = rsi_short_threshold

    def required_bars(self) -> int:
        return max(self.bb_period, self.rsi_period) + 5

    def evaluate(self, bars: pd.DataFrame) -> RuleSignal:
        if len(bars) < self.required_bars():
            return RuleSignal.no_signal(self.name, "Insufficient bars")

        close = bars["close"]
        bb_pos = bollinger_position(close, self.bb_period, self.bb_std).iloc[-1]
        rsi_val = rsi(close, self.rsi_period).iloc[-1]

        if np.isnan(bb_pos) and np.isnan(rsi_val):
            return RuleSignal.no_signal(self.name, "NaN indicators")

        # Check LONG confirmation
        bb_long = (not np.isnan(bb_pos)) and bb_pos < self.long_bb_threshold
        rsi_long = (not np.isnan(rsi_val)) and rsi_val < self.rsi_long_threshold

        if bb_long or rsi_long:
            reasons = []
            if bb_long:
                reasons.append(f"BB position={bb_pos:.2f}")
            if rsi_long:
                reasons.append(f"RSI={rsi_val:.1f}")
            strength = 0.7 if (bb_long and rsi_long) else 0.5
            return RuleSignal(
                direction=1, strength=strength, rule_name=self.name,
                reason=f"Oversold confirmation: {', '.join(reasons)}",
                metadata={"bb_position": bb_pos, "rsi": rsi_val},
            )

        # Check SHORT confirmation
        bb_short = (not np.isnan(bb_pos)) and bb_pos > self.short_bb_threshold
        rsi_short = (not np.isnan(rsi_val)) and rsi_val > self.rsi_short_threshold

        if bb_short or rsi_short:
            reasons = []
            if bb_short:
                reasons.append(f"BB position={bb_pos:.2f}")
            if rsi_short:
                reasons.append(f"RSI={rsi_val:.1f}")
            strength = 0.7 if (bb_short and rsi_short) else 0.5
            return RuleSignal(
                direction=-1, strength=strength, rule_name=self.name,
                reason=f"Overbought confirmation: {', '.join(reasons)}",
                metadata={"bb_position": bb_pos, "rsi": rsi_val},
            )

        return RuleSignal.no_signal(
            self.name,
            f"Neutral: BB={bb_pos:.2f}, RSI={rsi_val:.1f}"
        )
