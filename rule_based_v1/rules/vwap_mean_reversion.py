"""VWAP Mean Reversion Rule.

Fades price deviations from session VWAP back toward the mean.
Active window: 10:30-13:30 ET (after ORB resolves, before afternoon drift).

Signal conditions:
  LONG:  close < vwap - entry_distance_atr * ATR  AND  close > prev_close (recovering)
  SHORT: close > vwap + entry_distance_atr * ATR  AND  close < prev_close (declining)

Skips extreme dislocations (> max_distance_atr) — these are trend days, not mean reversion.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from rules.base import BaseRule, RuleSignal

logger = logging.getLogger(__name__)


def _session_vwap(bars: pd.DataFrame) -> pd.Series:
    """Compute cumulative session VWAP, resetting at the start of each day."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    tp_vol = typical * bars["volume"]
    dates = bars.index.map(lambda t: t.date())

    cum_tp_vol = tp_vol.groupby(dates).cumsum()
    cum_vol = bars["volume"].groupby(dates).cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


class VWAPMeanReversionRule(BaseRule):
    """Mean reversion toward session VWAP in the midday window."""

    def __init__(
        self,
        entry_distance_atr: float = 1.0,
        max_distance_atr: float = 3.0,
        atr_period: int = 14,
        time_start: str = "10:30",
        time_end: str = "13:30",
        long_only: bool = False,
        session_timezone: str = "US/Eastern",
        max_move_from_open_atr: float = 1.0,
    ):
        super().__init__(name="vwap_mean_reversion", role="primary")
        self.entry_distance_atr = entry_distance_atr
        self.max_distance_atr = max_distance_atr
        self.atr_period = atr_period
        self.time_start = pd.Timestamp(f"2000-01-01 {time_start}").time()
        self.time_end = pd.Timestamp(f"2000-01-01 {time_end}").time()
        self.long_only = long_only
        self.session_timezone = session_timezone
        self.max_move_from_open_atr = max_move_from_open_atr

    def required_bars(self) -> int:
        return self.atr_period + 2

    def evaluate(self, bars: pd.DataFrame) -> RuleSignal:
        if len(bars) < self.required_bars():
            return RuleSignal.no_signal(self.name, "Insufficient bars")

        # Time window check on latest bar
        last_ts = bars.index[-1]
        try:
            if last_ts.tzinfo is None:
                last_ts_et = last_ts
            else:
                last_ts_et = last_ts.tz_convert(self.session_timezone)
        except Exception:
            last_ts_et = last_ts

        bar_time = last_ts_et.time()
        if not (self.time_start <= bar_time <= self.time_end):
            return RuleSignal.no_signal(self.name, f"Outside window ({bar_time})")

        # ATR
        prev_close = bars["close"].shift(1)
        tr = pd.concat([
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_val = float(tr.ewm(span=self.atr_period, adjust=False).mean().iloc[-1])

        if np.isnan(atr_val) or atr_val <= 0:
            return RuleSignal.no_signal(self.name, "Invalid ATR")

        # Session VWAP
        try:
            vwap_series = _session_vwap(bars)
        except Exception as e:
            return RuleSignal.no_signal(self.name, f"VWAP error: {e}")

        vwap = float(vwap_series.iloc[-1])
        if np.isnan(vwap):
            return RuleSignal.no_signal(self.name, "VWAP is NaN (no volume?)")

        close = float(bars["close"].iloc[-1])
        prev_c = float(bars["close"].iloc[-2])
        deviation = close - vwap
        deviation_atr = deviation / atr_val

        # Skip extreme dislocations (trend day, not reversion)
        if abs(deviation_atr) > self.max_distance_atr:
            return RuleSignal.no_signal(
                self.name,
                f"Extreme dislocation {deviation_atr:.2f}x ATR > max {self.max_distance_atr}x"
            )

        # Regime gate: skip if price has already trended far from the day's open.
        # e.g. market gaps up then sells off 1×ATR → trend day, not a mean reversion setup.
        if self.max_move_from_open_atr > 0:
            today = last_ts_et.date()
            today_bars = bars[bars.index.map(lambda t: t.date()) == today]
            if not today_bars.empty:
                day_open = float(today_bars["open"].iloc[0])
                move_from_open_atr = (close - day_open) / atr_val
                # LONG: blocked if price already fell hard from open (downtrend day)
                if deviation_atr < 0 and move_from_open_atr < -self.max_move_from_open_atr:
                    return RuleSignal.no_signal(
                        self.name,
                        f"Trend day: price {move_from_open_atr:.2f}x ATR below open — not fading"
                    )
                # SHORT: blocked if price already rallied hard from open (uptrend day)
                if deviation_atr > 0 and move_from_open_atr > self.max_move_from_open_atr:
                    return RuleSignal.no_signal(
                        self.name,
                        f"Trend day: price {move_from_open_atr:.2f}x ATR above open — not fading"
                    )

        if deviation_atr <= -self.entry_distance_atr and close > prev_c:
            # Below VWAP, recovering → LONG
            strength = min(1.0, abs(deviation_atr) / (self.entry_distance_atr + 1.0))
            logger.debug(
                f"VWAP LONG: close={close:.2f} vwap={vwap:.2f} dev={deviation_atr:.2f}x ATR"
            )
            return RuleSignal(
                direction=1,
                strength=strength,
                rule_name=self.name,
                reason=f"Below VWAP by {abs(deviation_atr):.2f}x ATR, recovering",
                metadata={"vwap": vwap, "deviation_atr": deviation_atr, "atr": atr_val},
            )

        if not self.long_only and deviation_atr >= self.entry_distance_atr and close < prev_c:
            # Above VWAP, declining → SHORT
            strength = min(1.0, abs(deviation_atr) / (self.entry_distance_atr + 1.0))
            logger.debug(
                f"VWAP SHORT: close={close:.2f} vwap={vwap:.2f} dev={deviation_atr:.2f}x ATR"
            )
            return RuleSignal(
                direction=-1,
                strength=strength,
                rule_name=self.name,
                reason=f"Above VWAP by {deviation_atr:.2f}x ATR, declining",
                metadata={"vwap": vwap, "deviation_atr": deviation_atr, "atr": atr_val},
            )

        return RuleSignal.no_signal(
            self.name,
            f"No trigger: dev={deviation_atr:.2f}x ATR, close_vs_prev={'up' if close > prev_c else 'down'}"
        )
