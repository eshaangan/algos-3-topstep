"""
Selection helpers for ML-only trade entry decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
import logging
from typing import Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _to_chicago(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/Chicago")


def _time_to_minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def bars_per_day(
    *,
    session_mode: str,
    session_start: time,
    session_end: time,
    bar_minutes: int,
) -> int:
    if bar_minutes <= 0:
        return 0
    if session_mode.upper() == "24H":
        return int((24 * 60) / bar_minutes)
    start_min = _time_to_minutes(session_start)
    end_min = _time_to_minutes(session_end)
    span = max(0, end_min - start_min)
    if span == 0:
        return 0
    return max(1, int(span / bar_minutes))


def resolve_score_quantile(
    *,
    score_quantile: float,
    auto_score_quantile: bool,
    target_trades_per_day: int,
    session_mode: str,
    session_start: time,
    session_end: time,
    bar_minutes: int,
    min_quantile: float,
    max_quantile: float,
) -> float:
    if not auto_score_quantile:
        return float(score_quantile)
    bars = bars_per_day(
        session_mode=session_mode,
        session_start=session_start,
        session_end=session_end,
        bar_minutes=bar_minutes,
    )
    if bars <= 0:
        return float(score_quantile)
    target = max(1, int(target_trades_per_day))
    raw = 1.0 - (target / float(bars))
    return float(min(max(raw, min_quantile), max_quantile))


def in_session(
    ts: pd.Timestamp,
    *,
    session_mode: str,
    session_start: time,
    session_end: time,
) -> bool:
    if session_mode.upper() == "24H":
        return True
    local_ts = _to_chicago(ts)
    return session_start <= local_ts.time() < session_end


@dataclass
class DailyTopNSelector:
    max_trades_per_day: int
    min_bars_between_trades: int
    score_threshold: float
    bar_minutes: int = 5
    session_mode: str = "RTH"
    session_start: time = time(8, 30)
    session_end: time = time(14, 55)
    deadline_time: time = time(11, 30)
    deadline_relax_factor: float = 0.98

    trades_today: int = 0
    last_trade_time: Optional[pd.Timestamp] = None
    last_trade_idx: Optional[int] = None
    current_day: Optional[date] = None
    top_candidates: list[dict] = field(default_factory=list)
    last_rejection_reason: Optional[str] = None

    def _reset_if_new_day(self, ts: pd.Timestamp) -> None:
        day = _to_chicago(ts).date()
        if self.current_day != day:
            self.current_day = day
            self.trades_today = 0
            self.last_trade_time = None
            self.last_trade_idx = None
            self.top_candidates.clear()

    def _relaxed_floor(self) -> float:
        relax = max(0.0, float(self.deadline_relax_factor))
        return max(0.0, self.score_threshold * relax)

    def _current_floor(self, ts: pd.Timestamp) -> float:
        if self.trades_today == 0 and _to_chicago(ts).time() >= self.deadline_time:
            return min(self.score_threshold, self._relaxed_floor())
        return self.score_threshold

    def _update_top_candidates(self, score: float, bar_index: int, direction: str) -> None:
        candidate_floor = min(self.score_threshold, self._relaxed_floor())
        if score < candidate_floor:
            return
        self.top_candidates.append(
            {"score": float(score), "idx": int(bar_index), "direction": direction}
        )
        self.top_candidates.sort(key=lambda row: row["score"], reverse=True)
        self.top_candidates = self.top_candidates[: max(1, self.max_trades_per_day)]

    def _is_in_top_n(self, bar_index: int, direction: str) -> bool:
        for row in self.top_candidates:
            if row["idx"] == int(bar_index) and row["direction"] == direction:
                return True
        return False

    def _passes_spacing(self, ts: pd.Timestamp, bar_index: Optional[int]) -> bool:
        if self.last_trade_time is None and self.last_trade_idx is None:
            return True
        if bar_index is not None and self.last_trade_idx is not None:
            return (bar_index - self.last_trade_idx) >= self.min_bars_between_trades
        if self.last_trade_time is None:
            return True
        min_delta = timedelta(minutes=self.min_bars_between_trades * self.bar_minutes)
        return (_to_chicago(ts) - _to_chicago(self.last_trade_time)) >= min_delta

    def should_enter(
        self,
        ts: pd.Timestamp,
        *,
        score: Optional[float],
        direction: str,
        bar_index: Optional[int] = None,
    ) -> bool:
        self.last_rejection_reason = None
        self._reset_if_new_day(ts)

        if score is None or (isinstance(score, float) and np.isnan(score)):
            self.last_rejection_reason = "floor"
            return False

        if self.trades_today >= self.max_trades_per_day:
            self.last_rejection_reason = "budget"
            return False

        if not self._passes_spacing(ts, bar_index):
            self.last_rejection_reason = "spacing"
            return False

        if bar_index is not None:
            self._update_top_candidates(float(score), int(bar_index), direction)

        floor = self._current_floor(ts)
        if score < floor:
            self.last_rejection_reason = "floor"
            return False

        if bar_index is not None and not self._is_in_top_n(int(bar_index), direction):
            self.last_rejection_reason = "budget"
            return False

        return True

    def register_trade(self, ts: pd.Timestamp, *, bar_index: Optional[int] = None) -> None:
        self._reset_if_new_day(ts)
        self.trades_today += 1
        self.last_trade_time = ts
        if bar_index is not None:
            self.last_trade_idx = int(bar_index)

    def log_rejection(self, prefix: str = "Signal rejected") -> None:
        if self.last_rejection_reason and LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("%s: rejected_by=%s", prefix, self.last_rejection_reason)


@dataclass
class DayAdaptiveTopNSelector:
    max_trades_per_day: int
    min_bars_between_trades: int
    day_percentile_floor: float
    global_floor_score: float
    bar_minutes: int = 5
    session_mode: str = "RTH"
    session_start: time = time(8, 30)
    session_end: time = time(14, 55)
    deadline_time: time = time(11, 30)
    deadline_relax_factor: float = 0.98

    trades_today: int = 0
    last_trade_time: Optional[pd.Timestamp] = None
    last_trade_idx: Optional[int] = None
    current_day: Optional[date] = None
    top_candidates: list[dict] = field(default_factory=list)
    day_scores: list[float] = field(default_factory=list)
    last_rejection_reason: Optional[str] = None

    def _reset_if_new_day(self, ts: pd.Timestamp) -> None:
        day = _to_chicago(ts).date()
        if self.current_day != day:
            self.current_day = day
            self.trades_today = 0
            self.last_trade_time = None
            self.last_trade_idx = None
            self.top_candidates.clear()
            self.day_scores.clear()

    def _passes_spacing(self, ts: pd.Timestamp, bar_index: Optional[int]) -> bool:
        if self.last_trade_time is None and self.last_trade_idx is None:
            return True
        if bar_index is not None and self.last_trade_idx is not None:
            return (bar_index - self.last_trade_idx) >= self.min_bars_between_trades
        if self.last_trade_time is None:
            return True
        min_delta = timedelta(minutes=self.min_bars_between_trades * self.bar_minutes)
        return (_to_chicago(ts) - _to_chicago(self.last_trade_time)) >= min_delta

    def _update_day_scores(self, score: float) -> None:
        self.day_scores.append(float(score))

    def _current_day_percentile(self) -> float:
        if not self.day_scores:
            return float(self.global_floor_score)
        q = min(max(float(self.day_percentile_floor), 0.0), 1.0)
        return float(np.quantile(self.day_scores, q))

    def _current_floor(self, ts: pd.Timestamp) -> float:
        day_floor = max(self._current_day_percentile(), float(self.global_floor_score))
        if self.trades_today == 0 and _to_chicago(ts).time() >= self.deadline_time:
            relaxed = max(0.0, day_floor * float(self.deadline_relax_factor))
            return max(float(self.global_floor_score), relaxed)
        return day_floor

    def _update_top_candidates(self, score: float, bar_index: int, direction: str) -> None:
        candidate_floor = float(self.global_floor_score)
        if score < candidate_floor:
            return
        self.top_candidates.append(
            {"score": float(score), "idx": int(bar_index), "direction": direction}
        )
        self.top_candidates.sort(key=lambda row: row["score"], reverse=True)
        self.top_candidates = self.top_candidates[: max(1, self.max_trades_per_day)]

    def _is_in_top_n(self, bar_index: int, direction: str) -> bool:
        for row in self.top_candidates:
            if row["idx"] == int(bar_index) and row["direction"] == direction:
                return True
        return False

    def should_enter(
        self,
        ts: pd.Timestamp,
        *,
        score: Optional[float],
        direction: str,
        bar_index: Optional[int] = None,
    ) -> bool:
        self.last_rejection_reason = None
        self._reset_if_new_day(ts)

        if score is None or (isinstance(score, float) and np.isnan(score)):
            self.last_rejection_reason = "floor"
            return False

        score_val = float(score)
        self._update_day_scores(score_val)

        if self.trades_today >= self.max_trades_per_day:
            self.last_rejection_reason = "budget"
            return False

        if not self._passes_spacing(ts, bar_index):
            self.last_rejection_reason = "spacing"
            return False

        if bar_index is not None:
            self._update_top_candidates(score_val, int(bar_index), direction)

        floor = self._current_floor(ts)
        if score_val < floor:
            self.last_rejection_reason = "floor"
            return False

        if bar_index is not None and not self._is_in_top_n(int(bar_index), direction):
            self.last_rejection_reason = "budget"
            return False

        return True

    def register_trade(self, ts: pd.Timestamp, *, bar_index: Optional[int] = None) -> None:
        self._reset_if_new_day(ts)
        self.trades_today += 1
        self.last_trade_time = ts
        if bar_index is not None:
            self.last_trade_idx = int(bar_index)

    def log_rejection(self, prefix: str = "Signal rejected") -> None:
        if self.last_rejection_reason and LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("%s: rejected_by=%s", prefix, self.last_rejection_reason)
