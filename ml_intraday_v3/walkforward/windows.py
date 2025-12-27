"""
Walk-forward window generation.
"""

from __future__ import annotations

from datetime import date
from typing import List, Dict, Optional

import pandas as pd


def compute_walkforward_windows(
    bars_index: pd.Index,
    train_window_days: int,
    test_window_days: int,
    step_days: int,
    timezone: str = "America/Chicago",
    max_date: Optional[str] = None,
    expanding: bool = True,
) -> List[Dict]:
    """
    Compute rolling walk-forward windows using Chicago session dates.

    Args:
        bars_index: DatetimeIndex of bars
        train_window_days: Initial training window size in days
        test_window_days: Test window size in days
        step_days: Days to step forward between windows
        timezone: Timezone for session dates
        max_date: Maximum date for test windows (for holdout; format: "YYYY-MM-DD")
        expanding: If True, use expanding window (train grows). If False, use rolling window.

    Returns:
        List of window dictionaries with train/test date ranges
    """
    if not isinstance(bars_index, pd.DatetimeIndex) or bars_index.empty:
        return []

    idx = bars_index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local = idx.tz_convert(timezone)

    dates = pd.Series(local.date).drop_duplicates().tolist()

    # Apply max_date filter for holdout
    if max_date is not None:
        max_dt = pd.to_datetime(max_date).date()
        dates = [d for d in dates if d <= max_dt]

    n_days = len(dates)
    if n_days < (train_window_days + test_window_days):
        return []

    windows = []
    window_idx = 0

    for start in range(0, n_days - (train_window_days + test_window_days) + 1, step_days):
        if expanding:
            # Expanding window: train always starts at beginning
            train_dates = dates[0 : start + train_window_days]
        else:
            # Rolling window: fixed-size training window
            train_dates = dates[start : start + train_window_days]

        test_dates = dates[
            start + train_window_days : start + train_window_days + test_window_days
        ]

        train_mask = pd.Series(local.date).isin(train_dates).to_numpy()
        test_mask = pd.Series(local.date).isin(test_dates).to_numpy()

        if not train_mask.any() or not test_mask.any():
            continue

        train_start = idx[train_mask].min()
        train_end = idx[train_mask].max()
        test_start = idx[test_mask].min()
        test_end = idx[test_mask].max()

        windows.append(
            {
                "window_id": window_idx,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_dates": [str(d) for d in train_dates],
                "test_dates": [str(d) for d in test_dates],
                "train_days": len(train_dates),
                "test_days": len(test_dates),
            }
        )
        window_idx += 1

    return windows
