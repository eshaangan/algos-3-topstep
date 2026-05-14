"""Opening Range Breakout Rule - PRIMARY signal generator.

Establishes the opening range (first N minutes of RTH), then signals
a breakout when price closes above (LONG) or below (SHORT) that range.

Entry logic:
  - LONG : current bar close > range_high AND bar is after OR window closes
  - SHORT: current bar close < range_low  AND bar is after OR window closes

Additional filters applied inside this rule:
  - range_width >= min_range_atr * ATR  (range must be meaningful)
  - current time must be before entry_cutoff (avoid late-day noise)
  - signal is only triggered on the FIRST bar that breaks the range
    (prev close was inside range); subsequent continuation bars are skipped.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import date, time

from rules.base import BaseRule, RuleSignal
from utils.indicators import atr


class OpeningRangeBreakoutRule(BaseRule):
    """Opening Range Breakout primary signal.

    Parameters
    ----------
    session_timezone : str
        "US/Eastern" (default): session_start / or_end / entry_cutoff are U.S. Eastern wall times.
        "Asia/Tokyo": same fields are interpreted as Japan local times; the opening range is
        grouped by Tokyo calendar date (Nikkei cash session morning).
    session_start_time : str
        First bar included in the opening range window, format "HH:MM" in session_timezone.
    or_end_time : str
        When the OR window ends, format "HH:MM" in session_timezone.
    min_or_bars : int
        Minimum bars inside the opening range to consider the range valid.
        Default 4 (requires at least 20 min of data on 5-min bars).
    min_range_atr : float
        Minimum range width as a fraction of ATR. Range must be at least
        this wide to avoid choppy/flat open conditions. Default 0.5.
    entry_cutoff_time : str
        Latest time an ORB entry is allowed, format "HH:MM" Eastern.
        Default "12:00" to avoid afternoon noise.
    atr_period : int
        ATR period for volatility-normalised range filter. Default 14.
    use_close_for_signal : bool
        If True, trigger on close > range_high (conservative, confirmed break).
        If False, trigger on high > range_high (aggressive, first-touch entry).
        Default True.
    long_only : bool
        If True, only take LONG breakout signals. SHORT signals are suppressed.
        Default False.
    max_or_range_atr : float
        Skip ORB signal if OR range > this multiple of ATR. Avoids entering on
        abnormally wide-range opens where breakouts fail more often.
        Default 999 (disabled). Recommended: 4.0.
    """

    def __init__(
        self,
        session_timezone: str = "US/Eastern",
        session_start_time: str = "09:30",
        or_end_time: str = "10:00",
        min_or_bars: int = 4,
        min_range_atr: float = 0.5,
        entry_cutoff_time: str = "12:00",
        atr_period: int = 14,
        use_close_for_signal: bool = True,
        long_only: bool = False,
        max_or_range_atr: float = 999.0,
    ):
        super().__init__(name="opening_range_breakout", role="primary")
        self.session_timezone = session_timezone
        self.session_start_time = session_start_time
        self.or_end_time = or_end_time
        self.min_or_bars = min_or_bars
        self.min_range_atr = min_range_atr
        self.entry_cutoff_time = entry_cutoff_time
        self.atr_period = atr_period
        self.use_close_for_signal = use_close_for_signal
        self.long_only = long_only
        self.max_or_range_atr = max_or_range_atr

        # Parse time strings in the active session clock (Eastern or Tokyo)
        h_ss, m_ss = map(int, session_start_time.split(":"))
        self._session_start = time(h_ss, m_ss)
        h_or, m_or = map(int, or_end_time.split(":"))
        self._or_end = time(h_or, m_or)
        h_cut, m_cut = map(int, entry_cutoff_time.split(":"))
        self._entry_cutoff = time(h_cut, m_cut)

    def required_bars(self) -> int:
        # Need enough lookback for ATR + at least 1 full session's opening range.
        # For 5-min bars: 1 trading day ≈ 78 bars. Use 90 to be safe.
        return max(90, self.atr_period * 2)

    def evaluate(self, bars: pd.DataFrame) -> RuleSignal:
        if len(bars) < self.required_bars():
            return RuleSignal.no_signal(self.name, "Insufficient bars")

        if self.session_timezone == "Asia/Tokyo":
            return self._evaluate_tokyo_session(bars)
        return self._evaluate_us_eastern_session(bars)

    def _evaluate_us_eastern_session(self, bars: pd.DataFrame) -> RuleSignal:
        current_ts = bars.index[-1]
        if current_ts.tzinfo is not None:
            current_ts_et = current_ts.tz_convert("US/Eastern")
        else:
            current_ts_et = pd.Timestamp(current_ts).tz_localize("US/Eastern")

        current_time = current_ts_et.time()

        if current_time <= self._or_end:
            return RuleSignal.no_signal(
                self.name,
                f"Still in OR window: {current_time} <= {self._or_end} "
                f"(OR={self.session_start_time}-{self.or_end_time} ET)",
            )

        if current_time > self._entry_cutoff:
            return RuleSignal.no_signal(
                self.name, f"Past entry cutoff: {current_time} > {self._entry_cutoff}"
            )

        current_date = current_ts_et.date()
        session_open = pd.Timestamp(
            f"{current_date} {self.session_start_time}:00", tz="US/Eastern"
        )
        or_close_ts = pd.Timestamp(
            f"{current_date} {self.or_end_time}:00", tz="US/Eastern"
        )

        or_mask = (bars.index >= session_open) & (bars.index <= or_close_ts)
        or_bars = bars.loc[or_mask]

        return self._finalize_or_signal(bars, or_bars, "ET")

    def _evaluate_tokyo_session(self, bars: pd.DataFrame) -> RuleSignal:
        """Opening range aligned to Tokyo cash-session clock (calendar date in Asia/Tokyo)."""
        tz_name = "Asia/Tokyo"
        idx = bars.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        current_ts = idx[-1]
        current_jp = current_ts.tz_convert(tz_name)
        current_time = current_jp.time()
        session_day: date = current_jp.date()

        if current_time <= self._or_end:
            return RuleSignal.no_signal(
                self.name,
                f"Still in OR window (Tokyo): {current_time} <= {self._or_end} "
                f"(OR={self.session_start_time}-{self.or_end_time} JST)",
            )

        if current_time > self._entry_cutoff:
            return RuleSignal.no_signal(
                self.name,
                f"Past entry cutoff (Tokyo): {current_time} > {self._entry_cutoff}",
            )

        ts_jp = idx.tz_convert(tz_name)
        or_mask = []
        for t in ts_jp:
            d_ok = t.date() == session_day
            tt = t.time()
            in_or = self._session_start <= tt <= self._or_end
            or_mask.append(d_ok and in_or)
        or_bars = bars.loc[or_mask]
        return self._finalize_or_signal(bars, or_bars, "JST")

    def _finalize_or_signal(
        self, bars: pd.DataFrame, or_bars: pd.DataFrame, clock_label: str
    ) -> RuleSignal:

        if len(or_bars) < self.min_or_bars:
            return RuleSignal.no_signal(
                self.name,
                f"Insufficient OR bars: {len(or_bars)} < {self.min_or_bars} ({clock_label})",
            )

        # --- Compute opening range ---
        range_high = float(or_bars["high"].max())
        range_low = float(or_bars["low"].min())
        range_width = range_high - range_low

        # --- ATR filter: range must be wide enough to be meaningful ---
        atr_series = atr(bars["high"], bars["low"], bars["close"], self.atr_period)
        atr_val = float(atr_series.iloc[-1])

        if np.isnan(atr_val) or atr_val <= 0:
            return RuleSignal.no_signal(self.name, "Invalid ATR")

        if range_width < self.min_range_atr * atr_val:
            return RuleSignal.no_signal(
                self.name,
                f"Range too narrow: {range_width:.2f} < {self.min_range_atr} × ATR({atr_val:.2f})",
            )

        if range_width > self.max_or_range_atr * atr_val:
            return RuleSignal.no_signal(
                self.name,
                f"OR too wide: {range_width:.2f} > {self.max_or_range_atr} × ATR({atr_val:.2f}) — skipping volatile open",
            )

        # --- Breakout detection ---
        current_close = float(bars["close"].iloc[-1])
        current_high = float(bars["high"].iloc[-1])
        current_low = float(bars["low"].iloc[-1])

        # Previous bar close (for first-break detection)
        prev_close = float(bars["close"].iloc[-2]) if len(bars) >= 2 else current_close

        if self.use_close_for_signal:
            signal_high = current_close
            signal_low = current_close
            prev_signal_high = prev_close
            prev_signal_low = prev_close
        else:
            signal_high = current_high
            signal_low = current_low
            prev_signal_high = float(bars["high"].iloc[-2]) if len(bars) >= 2 else current_high
            prev_signal_low = float(bars["low"].iloc[-2]) if len(bars) >= 2 else current_low

        # LONG: first bar where price clears the range high
        if signal_high > range_high and prev_signal_high <= range_high:
            excess = (signal_high - range_high) / atr_val
            strength = min(1.0, excess / 0.5)
            return RuleSignal(
                direction=1,
                strength=max(0.1, strength),
                rule_name=self.name,
                reason=(
                    f"ORB LONG: {'close' if self.use_close_for_signal else 'high'}={signal_high:.2f} "
                    f"> range_high={range_high:.2f} "
                    f"(range={range_width:.2f}, ATR={atr_val:.2f})"
                ),
                metadata={
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_width": range_width,
                    "atr": atr_val,
                    "or_bars_count": len(or_bars),
                    "breakout_excess_atr": round(excess, 3),
                },
            )

        # SHORT: first bar where price breaks below the range low
        if self.long_only:
            return RuleSignal.no_signal(self.name, "long_only mode: SHORT suppressed")

        if signal_low < range_low and prev_signal_low >= range_low:
            excess = (range_low - signal_low) / atr_val
            strength = min(1.0, excess / 0.5)
            return RuleSignal(
                direction=-1,
                strength=max(0.1, strength),
                rule_name=self.name,
                reason=(
                    f"ORB SHORT: {'close' if self.use_close_for_signal else 'low'}={signal_low:.2f} "
                    f"< range_low={range_low:.2f} "
                    f"(range={range_width:.2f}, ATR={atr_val:.2f})"
                ),
                metadata={
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_width": range_width,
                    "atr": atr_val,
                    "or_bars_count": len(or_bars),
                    "breakout_excess_atr": round(excess, 3),
                },
            )

        return RuleSignal.no_signal(
            self.name,
            f"Inside range or continuation: close={current_close:.2f}, "
            f"range=[{range_low:.2f}, {range_high:.2f}]",
        )
