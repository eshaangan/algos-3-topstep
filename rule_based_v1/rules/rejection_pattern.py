"""Rejection Pattern Rule - CONFIRMATION signal.

Confirms via candle wick analysis. A long lower wick confirms LONG (buyers rejected lower prices).
A long upper wick confirms SHORT (sellers rejected higher prices).
"""

import numpy as np
import pandas as pd

from rules.base import BaseRule, RuleSignal
from utils.indicators import candle_body, upper_wick, lower_wick


class RejectionPatternRule(BaseRule):
    def __init__(self, min_wick_body_ratio: float = 1.5):
        super().__init__(name="rejection_pattern", role="confirmation")
        self.min_wick_body_ratio = min_wick_body_ratio

    def required_bars(self) -> int:
        return 1

    def evaluate(self, bars: pd.DataFrame) -> RuleSignal:
        if len(bars) < 1:
            return RuleSignal.no_signal(self.name, "No bars")

        last = bars.iloc[-1]
        body = candle_body(bars["open"], bars["close"]).iloc[-1]
        u_wick = upper_wick(bars["high"], bars["open"], bars["close"]).iloc[-1]
        l_wick = lower_wick(bars["low"], bars["open"], bars["close"]).iloc[-1]

        if np.isnan(body) or np.isnan(u_wick) or np.isnan(l_wick):
            return RuleSignal.no_signal(self.name, "NaN candle data")

        # Avoid division by zero for doji candles - use absolute wick size
        if body < 0.01:
            # Doji: check if one wick dominates
            if l_wick > u_wick and l_wick > 0.5:
                return RuleSignal(
                    direction=1, strength=0.4, rule_name=self.name,
                    reason=f"Doji with long lower wick={l_wick:.2f}",
                    metadata={"body": body, "upper_wick": u_wick, "lower_wick": l_wick},
                )
            if u_wick > l_wick and u_wick > 0.5:
                return RuleSignal(
                    direction=-1, strength=0.4, rule_name=self.name,
                    reason=f"Doji with long upper wick={u_wick:.2f}",
                    metadata={"body": body, "upper_wick": u_wick, "lower_wick": l_wick},
                )
            return RuleSignal.no_signal(self.name, "Doji without dominant wick")

        lower_ratio = l_wick / body
        upper_ratio = u_wick / body

        # LONG confirmation: long lower wick = buying rejection
        if lower_ratio >= self.min_wick_body_ratio and lower_ratio > upper_ratio:
            strength = min(1.0, lower_ratio / 3.0)
            return RuleSignal(
                direction=1, strength=strength, rule_name=self.name,
                reason=f"Lower wick rejection: ratio={lower_ratio:.2f}",
                metadata={"body": body, "upper_wick": u_wick, "lower_wick": l_wick,
                          "lower_ratio": lower_ratio, "upper_ratio": upper_ratio},
            )

        # SHORT confirmation: long upper wick = selling rejection
        if upper_ratio >= self.min_wick_body_ratio and upper_ratio > lower_ratio:
            strength = min(1.0, upper_ratio / 3.0)
            return RuleSignal(
                direction=-1, strength=strength, rule_name=self.name,
                reason=f"Upper wick rejection: ratio={upper_ratio:.2f}",
                metadata={"body": body, "upper_wick": u_wick, "lower_wick": l_wick,
                          "lower_ratio": lower_ratio, "upper_ratio": upper_ratio},
            )

        return RuleSignal.no_signal(
            self.name,
            f"No rejection: lower_ratio={lower_ratio:.2f}, upper_ratio={upper_ratio:.2f}"
        )
