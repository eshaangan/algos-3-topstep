from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from rule_based_v1.validation.research_opening_ofi import (
    event_from_signal,
    normalise_databento_mbp1,
    opening_signal,
    session_prices,
)


def test_normalises_native_databento_schema():
    raw = pd.DataFrame(
        {
            "bid_px_00": [100.0],
            "ask_px_00": [100.25],
            "bid_sz_00": [3],
            "ask_sz_00": [4],
        },
        index=pd.to_datetime(["2026-01-05 14:30:00Z"]),
    )
    out = normalise_databento_mbp1(raw)
    assert set(out) == {"recv_ns", "bid_px", "ask_px", "bid_sz", "ask_sz"}
    assert out.iloc[0]["recv_ns"] == raw.index[0].value


def test_opening_signal_excludes_bar_at_093015():
    features = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2026-01-05 14:30:01Z", "2026-01-05 14:30:14Z", "2026-01-05 14:30:15Z"]
            ),
            "ofi": [2.0, 4.0, 999.0],
            "avg_depth": [3.0, 3.0, 3.0],
        }
    )
    assert opening_signal(features) == 2.0


def test_session_prices_use_frozen_closes():
    idx = pd.to_datetime(["2026-01-05 14:31:00Z", "2026-01-05 14:36:00Z"])
    bars = pd.DataFrame({"close": [100.25, 102.0]}, index=idx)
    assert session_prices(bars, date(2026, 1, 5)) == (100.25, 102.0)


def test_event_applies_direction_and_measured_cost():
    event = event_from_signal(date(2026, 1, 5), -1.0, 100.0, 98.0, "development", "test")
    assert event is not None
    assert event["direction"] == -1
    assert event["gross_pnl"] == 4.0
    assert np.isclose(event["net_pnl"], 1.95)
