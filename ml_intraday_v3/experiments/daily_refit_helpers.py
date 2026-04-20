"""
Chicago session-date helpers for daily rolling refit OOS experiments.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Tuple

import pandas as pd

CHICAGO_TZ = "America/Chicago"


def chicago_session_dates_in_range(
    bars_index: pd.DatetimeIndex,
    start: date,
    end: date,
) -> list[date]:
    """
    Unique Chicago calendar dates present in the bar index, in range [start, end] inclusive.
    """
    if bars_index.empty:
        return []
    idx = bars_index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    local_dates = idx.tz_convert(CHICAGO_TZ).date
    in_range = {d for d in local_dates if start <= d <= end}
    return sorted(in_range)


def slice_bars_for_daily_refit(
    bars: pd.DataFrame,
    session_date: date,
    lookback_days: int,
    gap_days: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Optional[pd.Timestamp],
    Optional[pd.Timestamp],
    Optional[pd.Timestamp],
    Optional[pd.Timestamp],
]:
    """
    Split bars into train (strictly before first bar of session_date) and test (that session only).

    Train window: [train_start, train_end] inclusive, where
      train_end = first_test_ts - gap_days - 1 second
      train_start = train_end - lookback_days (calendar-day timedelta, matching standalone).

    Returns
    -------
    bars_train, bars_test, train_start, train_end, test_start, test_end
    All boundary timestamps use the same timezone as bars.index.
    """
    if bars.empty:
        raise ValueError("bars is empty")

    idx = bars.index
    if idx.tz is None:
        raise ValueError("bars index must be timezone-aware or localizable")

    dseries = pd.Series(idx.tz_convert(CHICAGO_TZ).date, index=bars.index)
    mask_d = (dseries == session_date).to_numpy()
    bars_test = bars.loc[mask_d].sort_index()
    if bars_test.empty:
        return bars.iloc[0:0].copy(), bars_test, None, None, None, None

    first_test_ts = bars_test.index.min()
    last_test_ts = bars_test.index.max()
    train_end = first_test_ts - pd.Timedelta(days=int(gap_days)) - pd.Timedelta(seconds=1)
    train_start = train_end - pd.Timedelta(days=int(lookback_days))
    bars_train = bars.loc[(bars.index >= train_start) & (bars.index <= train_end)].copy().sort_index()

    return bars_train, bars_test, train_start, train_end, first_test_ts, last_test_ts


def assert_train_before_test(bars_train: pd.DataFrame, bars_test: pd.DataFrame) -> None:
    if bars_train.empty or bars_test.empty:
        return
    if bars_train.index.max() >= bars_test.index.min():
        raise AssertionError(
            f"Lookahead risk: train max {bars_train.index.max()} >= test min {bars_test.index.min()}"
        )
