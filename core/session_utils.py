"""
Session utilities for RTH feasibility checks.

Prevents entries that cannot reach their planned time-exit within the same RTH session.
This eliminates SESSION_FLAT exits caused by entering too late in the trading day.
"""

from __future__ import annotations

from datetime import time, timedelta
from typing import Optional

import pandas as pd


def _to_chicago(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert timestamp to Chicago timezone."""
    ts = pd.to_datetime(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("America/Chicago")


def is_in_rth(
    ts: pd.Timestamp,
    *,
    rth_start_time: time,
    rth_end_time: time,
) -> bool:
    """
    Check if timestamp falls within RTH session.

    Args:
        ts: Timestamp to check (UTC or timezone-aware)
        rth_start_time: RTH session start time (Chicago time)
        rth_end_time: RTH session end time (Chicago time)

    Returns:
        True if timestamp is in RTH session
    """
    local_ts = _to_chicago(ts)
    return rth_start_time <= local_ts.time() < rth_end_time


def bars_left_in_rth(
    ts: pd.Timestamp,
    *,
    bar_minutes: int,
    rth_end_time: time,
) -> int:
    """
    Compute number of full bars remaining AFTER the current bar timestamp until session end.

    Assumptions:
        - Bars are timestamped by their START time (e.g., a 9:30-9:35 bar has timestamp 9:30)
        - The current bar at timestamp `ts` is already in progress
        - We count how many COMPLETE bars can fit between (ts + bar_minutes) and rth_end_time

    Example:
        - bar_minutes = 5
        - rth_end_time = 15:00 (3:00 PM CT)
        - ts = 14:40 (2:40 PM CT)
        - Current bar: 14:40-14:45
        - Next bar: 14:45-14:50
        - Following bar: 14:50-14:55
        - Last bar: 14:55-15:00 (ends exactly at 15:00)
        - bars_left = 3 (bars starting at 14:45, 14:50, 14:55)

    Args:
        ts: Current bar timestamp (bar start time)
        bar_minutes: Bar size in minutes (e.g., 5)
        rth_end_time: RTH session end time (Chicago time)

    Returns:
        Number of complete bars remaining after current bar
    """
    local_ts = _to_chicago(ts)

    # Convert session end to datetime on same day
    session_end_dt = local_ts.replace(
        hour=rth_end_time.hour,
        minute=rth_end_time.minute,
        second=0,
        microsecond=0,
    )

    # Time remaining from END of current bar to session end
    current_bar_end = local_ts + timedelta(minutes=bar_minutes)
    time_remaining = session_end_dt - current_bar_end

    if time_remaining.total_seconds() <= 0:
        return 0

    # Number of complete bars that fit in remaining time
    minutes_remaining = time_remaining.total_seconds() / 60.0
    bars_remaining = int(minutes_remaining / bar_minutes)

    return max(0, bars_remaining)


def is_entry_feasible(
    ts: pd.Timestamp,
    *,
    horizon_bars: int,
    bar_minutes: int,
    rth_end_time: time,
    execution_mode: str = "time_exit",
    session_mode: str = "RTH",
) -> bool:
    """
    Check if entry at current bar is feasible for completing time-exit within RTH session.

    Entry mechanics:
        - Signal bar: bar at timestamp `ts` (we see the signal here)
        - Entry bar: NEXT bar at `ts + bar_minutes` (we enter at open of next bar)
        - Exit bar: `horizon_bars` bars after entry (we exit at close of this bar)
        - Required: entry_idx + horizon_bars must complete before session end

    Example with horizon_bars=12, bar_minutes=5:
        - Signal at 14:40 (bar index i)
        - Entry at 14:45 (bar index i+1)
        - Exit at 15:45 (bar index i+1+12)
        - If RTH ends at 15:00, this is NOT feasible (15:45 > 15:00)

    Rule:
        We need at least (horizon_bars + 1) bars remaining after the signal bar:
        - 1 bar for entry (next bar open)
        - horizon_bars for holding the position
        Total: signal_bar, entry_bar, hold_bar_1, ..., hold_bar_horizon_bars

    Args:
        ts: Current signal bar timestamp
        horizon_bars: Number of bars to hold position (e.g., 12)
        bar_minutes: Bar size in minutes (e.g., 5)
        rth_end_time: RTH session end time (Chicago time)
        execution_mode: Execution mode ("time_exit" or other)
        session_mode: Session mode ("RTH" or "24H")

    Returns:
        True if entry is feasible, False otherwise
    """
    # Only apply feasibility check for RTH + time_exit mode
    if session_mode.upper() != "RTH":
        return True

    if execution_mode != "time_exit":
        return True

    # Check how many bars are left after current signal bar
    bars_left = bars_left_in_rth(
        ts,
        bar_minutes=bar_minutes,
        rth_end_time=rth_end_time,
    )

    # We need:
    # - 1 bar to enter (next bar)
    # - horizon_bars to hold
    # Total: horizon_bars + 1
    required_bars = horizon_bars + 1

    return bars_left >= required_bars


def get_last_feasible_entry_time(
    *,
    horizon_bars: int,
    bar_minutes: int,
    rth_start_time: time,
    rth_end_time: time,
) -> time:
    """
    Compute the last signal bar time that allows a feasible time-exit entry.

    This is useful for debugging and validation.

    Mechanics:
        - Signal at bar i (at time T)
        - Entry at bar i+1 (at time T + bar_minutes)
        - Exit at bar i+1+horizon_bars (at time T + (horizon_bars+1) * bar_minutes)
        - The exit bar must start BEFORE session_end

    Args:
        horizon_bars: Number of bars to hold position
        bar_minutes: Bar size in minutes
        rth_start_time: RTH session start time
        rth_end_time: RTH session end time

    Returns:
        Last feasible signal bar time (Chicago time)
    """
    # The exit bar starts at: signal_time + (horizon_bars + 1) * bar_minutes
    # This must be < session_end
    # Therefore: signal_time < session_end - (horizon_bars + 1) * bar_minutes
    # Last valid signal time is one bar before that cutoff
    required_minutes = (horizon_bars + 1) * bar_minutes

    # Last signal time = session_end - required_minutes - bar_minutes
    # Actually, we want the last bar whose next bar + horizon_bars is still in session
    # So: last_signal_end + (horizon_bars + 1) * bar_minutes <= session_end
    # last_signal_end <= session_end - (horizon_bars + 1) * bar_minutes
    # last_signal_start = last_signal_end - bar_minutes
    # last_signal_start = session_end - (horizon_bars + 1) * bar_minutes - bar_minutes
    # last_signal_start = session_end - (horizon_bars + 2) * bar_minutes

    # Wait, let me reconsider. The exit bar at i+1+horizon_bars must START before session_end.
    # If signal bar starts at T, exit bar starts at T + (horizon_bars + 1) * bar_minutes
    # We need: T + (horizon_bars + 1) * bar_minutes < session_end
    # T < session_end - (horizon_bars + 1) * bar_minutes
    # Last valid T is the largest multiple of bar_minutes less than that cutoff

    session_end_minutes = rth_end_time.hour * 60 + rth_end_time.minute
    cutoff_minutes = session_end_minutes - (horizon_bars + 1) * bar_minutes

    # Round down to nearest bar boundary
    last_signal_minutes = (cutoff_minutes // bar_minutes) * bar_minutes - bar_minutes

    last_signal_hour = last_signal_minutes // 60
    last_signal_minute = last_signal_minutes % 60

    return time(last_signal_hour, last_signal_minute)
