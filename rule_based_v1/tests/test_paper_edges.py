from __future__ import annotations

from datetime import time

import pandas as pd
import pytest

from rule_based_v1.validation.research_close_momentum import (
    build_close_momentum_events,
    normalise_bars,
)
from rule_based_v1.validation.research_treasury_auction import build_auction_events


def _bars(index: pd.DatetimeIndex, prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.1 for p in prices],
            "low": [p - 0.1 for p in prices],
            "close": prices,
        },
        index=index,
    )


def test_close_momentum_direction_uses_only_morning_window():
    idx = pd.to_datetime(
        ["2025-01-02 09:30", "2025-01-02 10:00", "2025-01-02 15:30", "2025-01-02 15:55"]
    ).tz_localize("America/New_York")
    bars = _bars(idx, [100.0, 101.0, 102.0, 103.0])
    first = build_close_momentum_events(bars, commission_per_side=0, slippage_ticks=0)
    assert first.loc[0, "direction"] == 1
    assert first.loc[0, "points"] == pytest.approx(1.0)

    changed = bars.copy()
    changed.loc[idx[-1], ["open", "high", "low", "close"]] = [90.0, 90.1, 89.9, 90.0]
    second = build_close_momentum_events(changed, commission_per_side=0, slippage_ticks=0)
    assert second.loc[0, "direction"] == 1
    assert second.loc[0, "points"] < 0


def test_close_momentum_requires_nonoverlapping_windows():
    idx = pd.date_range("2025-01-02 09:30", periods=4, freq="2h", tz="America/New_York")
    bars = _bars(idx, [100, 101, 102, 103])
    with pytest.raises(ValueError):
        build_close_momentum_events(bars, signal_end=time(15, 45), trade_start=time(15, 30))


def test_naive_bars_require_explicit_timezone():
    raw = _bars(pd.date_range("2025-01-02", periods=4, freq="5min"), [1, 2, 3, 4])
    with pytest.raises(ValueError):
        normalise_bars(raw)


def test_treasury_auction_has_short_then_long_direction():
    idx = pd.to_datetime(
        [
            "2025-01-15 15:00:00+00:00",
            "2025-01-15 18:00:00+00:00",
            "2025-01-15 18:05:00+00:00",
            "2025-01-15 21:04:00+00:00",
        ],
        utc=True,
    )
    bars = _bars(idx, [110.0, 109.0, 108.0, 109.5])
    auctions = pd.DataFrame(
        {
            "auction_close": ["2025-01-15T18:00:00Z"],
            "result_release": ["2025-01-15T18:04:00Z"],
            "security_term": ["5-Year"],
        }
    )
    events = build_auction_events(
        bars,
        auctions,
        security_terms=["5-Year"],
        result_buffer_minutes=1,
        commission_per_side=0,
        slippage_ticks=0,
        point_value=1,
        tick_value=1,
    )
    assert len(events) == 1
    assert events.loc[0, "pre_points"] == pytest.approx(1.0)
    assert events.loc[0, "post_points"] == pytest.approx(1.5)
    assert events.loc[0, "pnl"] == pytest.approx(2.5)


def test_treasury_auction_filters_maturity_without_lookahead_default():
    idx = pd.to_datetime(
        ["2025-01-15 15:00Z", "2025-01-15 18:00Z", "2025-01-15 18:05Z", "2025-01-15 21:04Z"],
        utc=True,
    )
    bars = _bars(idx, [110, 109, 108, 109])
    auctions = pd.DataFrame(
        {
            "auction_close": ["2025-01-15T18:00:00Z"],
            "result_release": ["2025-01-15T18:04:00Z"],
            "security_term": ["10-Year"],
        }
    )
    events = build_auction_events(bars, auctions, security_terms=["5-Year"])
    assert events.empty
