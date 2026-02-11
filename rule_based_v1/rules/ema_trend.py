"""EMA Trend Rule - PRIMARY signal generator.

LONG: EMA-fast crosses above EMA-slow, spread > min_spread * ATR, slow EMA slope positive.
SHORT: EMA-fast crosses below EMA-slow, spread > min_spread * ATR, slow EMA slope negative.
Strength proportional to spread magnitude relative to ATR.
"""

import numpy as np
import pandas as pd

from rules.base import BaseRule, RuleSignal
from utils.indicators import ema, atr, ema_slope


class EMATrendRule(BaseRule):
    def __init__(
        self,
        fast_period: int = 13,
        slow_period: int = 34,
        min_spread_atr_ratio: float = 0.3,
        slope_lookback: int = 3,
        atr_period: int = 14,
    ):
        super().__init__(name="ema_trend", role="primary")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.min_spread_atr_ratio = min_spread_atr_ratio
        self.slope_lookback = slope_lookback
        self.atr_period = atr_period

    def required_bars(self) -> int:
        # Need enough bars for the slowest indicator to stabilize
        return self.slow_period + self.slope_lookback + 10

    def evaluate(self, bars: pd.DataFrame) -> RuleSignal:
        if len(bars) < self.required_bars():
            return RuleSignal.no_signal(self.name, "Insufficient bars")

        close = bars["close"]
        high = bars["high"]
        low = bars["low"]

        fast_ema = ema(close, self.fast_period)
        slow_ema = ema(close, self.slow_period)
        current_atr = atr(high, low, close, self.atr_period)
        slow_slope = ema_slope(close, self.slow_period, self.slope_lookback)

        # Current values
        spread = fast_ema.iloc[-1] - slow_ema.iloc[-1]
        prev_spread = fast_ema.iloc[-2] - slow_ema.iloc[-2]
        atr_val = current_atr.iloc[-1]
        slope_val = slow_slope.iloc[-1]

        if np.isnan(spread) or np.isnan(atr_val) or atr_val <= 0:
            return RuleSignal.no_signal(self.name, "NaN indicators")

        normalized_spread = abs(spread) / atr_val

        # Check for crossover (spread changed sign)
        crossover_up = prev_spread <= 0 and spread > 0
        crossover_down = prev_spread >= 0 and spread < 0

        # LONG: fast crossed above slow, spread sufficient, slow slope up
        if crossover_up and normalized_spread >= self.min_spread_atr_ratio and slope_val > 0:
            strength = min(1.0, normalized_spread / 1.0)  # Caps at 1.0 when spread = 1x ATR
            return RuleSignal(
                direction=1,
                strength=strength,
                rule_name=self.name,
                reason=f"EMA crossover UP: spread={normalized_spread:.2f}x ATR, slope={slope_val:.4f}",
                metadata={
                    "spread": spread,
                    "normalized_spread": normalized_spread,
                    "atr": atr_val,
                    "slope": slope_val,
                },
            )

        # SHORT: fast crossed below slow, spread sufficient, slow slope down
        if crossover_down and normalized_spread >= self.min_spread_atr_ratio and slope_val < 0:
            strength = min(1.0, normalized_spread / 1.0)
            return RuleSignal(
                direction=-1,
                strength=strength,
                rule_name=self.name,
                reason=f"EMA crossover DOWN: spread={normalized_spread:.2f}x ATR, slope={slope_val:.4f}",
                metadata={
                    "spread": spread,
                    "normalized_spread": normalized_spread,
                    "atr": atr_val,
                    "slope": slope_val,
                },
            )

        return RuleSignal.no_signal(
            self.name,
            f"No crossover (spread={normalized_spread:.2f}x ATR)"
        )
