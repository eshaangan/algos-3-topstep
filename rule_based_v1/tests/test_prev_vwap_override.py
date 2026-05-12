"""Tests for bearish PrevVWAP ORB override scoring."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from live.runner import LiveRunner


def _runner() -> LiveRunner:
    runner = LiveRunner.__new__(LiveRunner)
    runner.rules_cfg = {
        "opening_range_breakout": {
            "session_start_time": "09:30",
            "or_end_time": "10:04",
            "min_or_bars": 7,
            "atr_period": 14,
        }
    }
    runner.prev_vwap_override_cfg = {
        "enabled": True,
        "min_score": 4,
        "require_breakout": True,
        "min_gap_up_pct": 0.001,
        "min_drive_from_open_atr": 0.75,
        "min_or_range_atr": 0.8,
        "min_breakout_excess_atr": 0.15,
        "min_volume_ratio": 1.1,
    }
    return runner


def _bars(*, breakout_close: float = 1022.0, breakout_volume: float = 220.0) -> pd.DataFrame:
    prior_idx = pd.date_range("2026-04-21 14:00", periods=20, freq="5min", tz="US/Eastern")
    prior = pd.DataFrame(
        {
            "open": np.full(len(prior_idx), 999.0),
            "high": np.full(len(prior_idx), 1001.0),
            "low": np.full(len(prior_idx), 998.0),
            "close": np.full(len(prior_idx), 1000.0),
            "volume": np.full(len(prior_idx), 100.0),
        },
        index=prior_idx,
    )

    or_idx = pd.date_range("2026-04-22 09:30", periods=7, freq="5min", tz="US/Eastern")
    or_highs = np.array([1006.0, 1008.0, 1010.0, 1011.0, 1012.0, 1014.0, 1015.0])
    current_or = pd.DataFrame(
        {
            "open": np.array([1002.0, 1005.0, 1007.0, 1008.0, 1009.0, 1011.0, 1013.0]),
            "high": or_highs,
            "low": np.full(len(or_idx), 1001.0),
            "close": or_highs - 1.0,
            "volume": np.full(len(or_idx), 100.0),
        },
        index=or_idx,
    )

    breakout = pd.DataFrame(
        {
            "open": [1015.0],
            "high": [1024.0],
            "low": [1014.0],
            "close": [breakout_close],
            "volume": [breakout_volume],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-04-22 10:05", tz="US/Eastern")]),
    )
    return pd.concat([prior, current_or, breakout])


def test_prev_vwap_override_allows_strong_gap_drive_breakout():
    assert _runner()._prev_vwap_bearish_override_allows_long(_bars()) is True


def test_prev_vwap_override_requires_actual_breakout():
    assert _runner()._prev_vwap_bearish_override_allows_long(_bars(breakout_close=1015.0)) is False


def test_prev_vwap_override_can_be_disabled():
    runner = _runner()
    runner.prev_vwap_override_cfg["enabled"] = False
    assert runner._prev_vwap_bearish_override_allows_long(_bars()) is False
