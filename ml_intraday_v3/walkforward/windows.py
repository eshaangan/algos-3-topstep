"""
Walk-forward window generation.
"""

from __future__ import annotations

from typing import List, Dict

import pandas as pd


def compute_walkforward_windows(
    bars_index: pd.Index,
    train_window_days: int,
    test_window_days: int,
    step_days: int,
    timezone: str = "America/Chicago",
) -> List[Dict]:
    """
    Compute rolling walk-forward windows using Chicago session dates.
    """
    if not isinstance(bars_index, pd.DatetimeIndex) or bars_index.empty:
        return []

    idx = bars_index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local = idx.tz_convert(timezone)

    dates = pd.Series(local.date).drop_duplicates().tolist()
    n_days = len(dates)
    if n_days < (train_window_days + test_window_days):
        return []

    windows = []
    for start in range(0, n_days - (train_window_days + test_window_days) + 1, step_days):
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
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_dates": [str(d) for d in train_dates],
                "test_dates": [str(d) for d in test_dates],
            }
        )

    return windows
