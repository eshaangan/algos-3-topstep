"""
Daily trade budget with min spacing between entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd


def _to_chicago(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/Chicago")


@dataclass
class DailyTradeBudget:
    max_trades_per_day: int
    min_bars_between_trades: int
    bar_minutes: int = 5

    trades_today: int = 0
    last_trade_time: Optional[pd.Timestamp] = None
    last_trade_idx: Optional[int] = None
    current_day: Optional[date] = None

    def _reset_if_new_day(self, ts: pd.Timestamp) -> None:
        day = _to_chicago(ts).date()
        if self.current_day != day:
            self.current_day = day
            self.trades_today = 0
            self.last_trade_time = None
            self.last_trade_idx = None

    def can_take(self, ts: pd.Timestamp, bar_index: Optional[int] = None) -> bool:
        self._reset_if_new_day(ts)
        if self.trades_today >= self.max_trades_per_day:
            return False

        if bar_index is not None and self.last_trade_idx is not None:
            return (bar_index - self.last_trade_idx) >= self.min_bars_between_trades

        if self.last_trade_time is None:
            return True

        min_delta = timedelta(minutes=self.min_bars_between_trades * self.bar_minutes)
        return (_to_chicago(ts) - _to_chicago(self.last_trade_time)) >= min_delta

    def register_trade(self, ts: pd.Timestamp, bar_index: Optional[int] = None) -> None:
        self._reset_if_new_day(ts)
        self.trades_today += 1
        self.last_trade_time = ts
        if bar_index is not None:
            self.last_trade_idx = bar_index
