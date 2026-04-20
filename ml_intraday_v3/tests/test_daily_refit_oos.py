"""Unit tests for daily refit OOS session-date helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from ml_intraday_v3.experiments.daily_refit_helpers import (
    assert_train_before_test,
    chicago_session_dates_in_range,
    slice_bars_for_daily_refit,
)


def _utc_ts(*args, **kwargs):
    return pd.Timestamp(datetime(*args, tzinfo=timezone.utc), **kwargs)


def test_chicago_session_dates_in_range_sorted_unique():
    # 2024-01-02 06:00 UTC = 2024-01-02 00:00 Chicago (CST); add same calendar day later
    idx = pd.DatetimeIndex(
        [
            _utc_ts(2024, 1, 2, 6, 0),
            _utc_ts(2024, 1, 2, 16, 0),
            _utc_ts(2024, 1, 3, 6, 0),
        ]
    )
    out = chicago_session_dates_in_range(idx, date(2024, 1, 1), date(2024, 1, 31))
    assert out == [date(2024, 1, 2), date(2024, 1, 3)]


def test_chicago_session_dates_empty_index():
    assert chicago_session_dates_in_range(pd.DatetimeIndex([]), date(2024, 1, 1), date(2024, 1, 31)) == []


def test_slice_bars_train_strictly_before_test():
    # Two Chicago session days; train on D1, test on D2 with 180d lookback, gap 0
    t0 = _utc_ts(2024, 6, 3, 13, 30)  # Monday RTH-ish UTC
    bars = pd.DataFrame({"open": [1.0, 2.0, 3.0]}, index=[t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(days=1)])
    d_test = pd.Timestamp(t0 + pd.Timedelta(days=1)).tz_convert("America/Chicago").date()
    tr, te, ts, te_train, tstart, tend = slice_bars_for_daily_refit(
        bars, session_date=d_test, lookback_days=180, gap_days=0
    )
    assert len(te) == 1
    assert len(tr) == 2
    assert_train_before_test(tr, te)
    assert tr.index.max() < te.index.min()


def test_slice_bars_empty_test_returns_none_bounds():
    bars = pd.DataFrame({"open": [1.0]}, index=[_utc_ts(2024, 6, 3, 13, 30)])
    tr, te, a, b, c, d = slice_bars_for_daily_refit(
        bars, session_date=date(2024, 7, 1), lookback_days=30, gap_days=0
    )
    assert tr.empty and te.empty
    assert a is None and b is None and c is None and d is None


def test_assert_train_before_test_raises_on_overlap():
    idx = pd.DatetimeIndex([_utc_ts(2024, 1, 1, 10), _utc_ts(2024, 1, 1, 11)])
    train = pd.DataFrame({"x": [1]}, index=[idx[1]])
    test = pd.DataFrame({"x": [2]}, index=[idx[0]])
    with pytest.raises(AssertionError, match="Lookahead"):
        assert_train_before_test(train, test)
