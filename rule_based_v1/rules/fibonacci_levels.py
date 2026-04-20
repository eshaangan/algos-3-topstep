"""Fibonacci Break-of-Candle Rule — Ivan Trades-style Fib pullback entry."""

from __future__ import annotations

import pandas as pd

from rule_based_v1.rules.base import BaseRule, RuleSignal
from rule_based_v1.utils.indicators import atr as calc_atr


class FibonacciLevelsRule(BaseRule):
    """Primary rule: BOC (Break of Candle) entries at Fibonacci retracement levels.

    Two-bar pattern based on Ivan Trades' published entry mechanism:
      Bar N-1 (signal candle): price touches a key Fibonacci retracement level.
        In uptrend → LONG candidate: bar low touches level, close stays near/above it.
        In downtrend → SHORT candidate: bar high touches level, close stays near/below it.
      Bar N (current bar): triggers entry when it breaks the signal candle.
        LONG: current bar high > signal candle high → enter LONG.
        SHORT: current bar low < signal candle low → enter SHORT.

    Swing identification:
      Uses the prior `swing_lookback_days` calendar days of RTH bars to find
      the swing high and low. This captures institutional swing levels rather
      than just the previous day's extremes.

    Fibonacci levels:
      Computed as retracements from swing_low to swing_high.
      Key levels: 38.2%, 50.0%, 61.8%.

    Parameters
    ----------
    key_levels : list[float]
        Fibonacci ratios to watch.
    touch_tolerance_atr : float
        Proximity (in ATR) required for a bar to count as "touching" a level.
    swing_lookback_bars : int
        Number of bars used to identify the anchor swing. Default ~100 = ~5 RTH days.
    trend_level : float
        Fib ratio used to determine trend. Open above → uptrend (look for LONG pullbacks).
    long_only : bool
        If True, only emit LONG signals. SHORT side lacks edge on MNQ.
    """

    def __init__(
        self,
        key_levels: list | None = None,
        touch_tolerance_atr: float = 0.40,
        swing_lookback_bars: int = 100,    # ~5 RTH days of 5-min bars
        trend_level: float = 0.500,
        long_only: bool = True,
    ):
        super().__init__(name="fibonacci_levels", role="primary")
        self.key_levels = key_levels if key_levels is not None else [0.382, 0.500, 0.618]
        self.touch_tolerance_atr = touch_tolerance_atr
        self.swing_lookback_bars = swing_lookback_bars
        self.trend_level = trend_level
        self.long_only = long_only

    def required_bars(self) -> int:
        # Need swing window + signal candle + BOC bar, plus 14 for ATR warmup
        return self.swing_lookback_bars + 16

    def evaluate(self, bars: pd.DataFrame) -> RuleSignal:
        if len(bars) < self.required_bars():
            return RuleSignal.no_signal(self.name, "Insufficient bars")

        # ATR on recent bars
        atr_val = float(calc_atr(bars["high"], bars["low"], bars["close"]).iloc[-1])
        if pd.isna(atr_val) or atr_val <= 0:
            return RuleSignal.no_signal(self.name, "ATR unavailable")

        # Swing detection: exclude the signal candle (bars[-2]) and BOC bar (bars[-1])
        # so the anchor swing is purely historical.
        swing_window = bars.iloc[-self.swing_lookback_bars - 2 : -2]
        if len(swing_window) < 10:
            return RuleSignal.no_signal(self.name, "Insufficient swing window")

        swing_high = float(swing_window["high"].max())
        swing_low = float(swing_window["low"].min())
        fib_range = swing_high - swing_low

        if fib_range < 1e-6:
            return RuleSignal.no_signal(self.name, "Fib range too small")

        fib_prices = {lvl: swing_low + lvl * fib_range for lvl in self.key_levels}

        # Swing direction: low before high = bullish swing → look for LONG pullbacks.
        low_pos = swing_window.index.get_loc(swing_window["low"].idxmin())
        high_pos = swing_window.index.get_loc(swing_window["high"].idxmax())
        trend = 1 if low_pos < high_pos else -1

        if self.long_only and trend == -1:
            return RuleSignal.no_signal(self.name, "Downtrend skipped (long_only)")

        # Signal candle = bars[-2], current bar = bars[-1]
        signal_bar = bars.iloc[-2]
        current_bar = bars.iloc[-1]
        tolerance = self.touch_tolerance_atr * atr_val

        # Check if signal_bar touched any Fib level
        for level_pct, level_price in sorted(
            fib_prices.items(),
            key=lambda kv: abs(kv[1] - float(signal_bar["close"])),
        ):
            if trend == 1:
                # Signal candle: low touches level, close stays near/above it
                sc_touched = float(signal_bar["low"]) <= level_price + tolerance
                sc_near = float(signal_bar["close"]) >= level_price - tolerance
                if not (sc_touched and sc_near):
                    continue

                # BOC entry: current bar breaks above signal candle high
                if float(current_bar["high"]) > float(signal_bar["high"]):
                    strength = min(1.0, max(0.3, (float(current_bar["high"]) - level_price) / atr_val))
                    return RuleSignal(
                        direction=1,
                        strength=strength,
                        rule_name=self.name,
                        reason=(
                            f"BOC LONG {level_pct:.1%} ({level_price:.2f}): "
                            f"signal_low={signal_bar['low']:.2f} "
                            f"break_high={current_bar['high']:.2f}"
                        ),
                        metadata={
                            "fib_level_pct": level_pct,
                            "fib_level_price": round(level_price, 4),
                            "swing_high": round(swing_high, 4),
                            "swing_low": round(swing_low, 4),
                            "signal_candle_high": round(float(signal_bar["high"]), 4),
                            "signal_candle_low": round(float(signal_bar["low"]), 4),
                            "trend": "up",
                            "atr": round(atr_val, 4),
                        },
                    )

            else:
                # Signal candle: high touches level, close stays near/below it
                sc_touched = float(signal_bar["high"]) >= level_price - tolerance
                sc_near = float(signal_bar["close"]) <= level_price + tolerance
                if not (sc_touched and sc_near):
                    continue

                # BOC entry: current bar breaks below signal candle low
                if float(current_bar["low"]) < float(signal_bar["low"]):
                    strength = min(1.0, max(0.3, (level_price - float(current_bar["low"])) / atr_val))
                    return RuleSignal(
                        direction=-1,
                        strength=strength,
                        rule_name=self.name,
                        reason=(
                            f"BOC SHORT {level_pct:.1%} ({level_price:.2f}): "
                            f"signal_high={signal_bar['high']:.2f} "
                            f"break_low={current_bar['low']:.2f}"
                        ),
                        metadata={
                            "fib_level_pct": level_pct,
                            "fib_level_price": round(level_price, 4),
                            "swing_high": round(swing_high, 4),
                            "swing_low": round(swing_low, 4),
                            "signal_candle_high": round(float(signal_bar["high"]), 4),
                            "signal_candle_low": round(float(signal_bar["low"]), 4),
                            "trend": "down",
                            "atr": round(atr_val, 4),
                        },
                    )

        return RuleSignal.no_signal(self.name, "No BOC signal")
