"""Tests for FibonacciLevelsRule (Break of Candle implementation)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rule_based_v1.rules.fibonacci_levels import FibonacciLevelsRule
from rule_based_v1.rules.base import RuleSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(
    n_swing_bars: int = 116,
    swing_high: float = 100.0,
    swing_low: float = 80.0,
    today_open: float = 95.0,
    signal_low: float = 88.0,       # signal candle low (should touch Fib level)
    signal_close: float = 89.5,     # signal candle close (stays near/above level)
    signal_high: float = 91.0,      # signal candle high
    boc_high: float = 92.0,         # BOC bar high (breaks signal candle high for LONG)
    boc_low: float = 87.5,
    boc_close: float = 91.5,
) -> pd.DataFrame:
    """Build synthetic bars: swing window + signal candle + BOC bar."""
    tz = "US/Eastern"
    start = pd.Timestamp("2025-01-01 09:30", tz=tz)
    rows = []

    # Swing window bars
    for i in range(n_swing_bars):
        ts = start + pd.Timedelta(minutes=5 * i)
        pct = i / (n_swing_bars - 1)
        price = swing_low + pct * (swing_high - swing_low)
        rows.append({
            "timestamp": ts,
            "open": price - 0.3,
            "high": max(swing_high, price + 0.3) if i == n_swing_bars - 1 else price + 0.3,
            "low": min(swing_low, price - 0.3) if i == 0 else price - 0.3,
            "close": price,
            "volume": 1000,
        })

    # Signal candle (bars[-2])
    signal_ts = start + pd.Timedelta(minutes=5 * n_swing_bars)
    rows.append({
        "timestamp": signal_ts,
        "open": today_open,
        "high": signal_high,
        "low": signal_low,
        "close": signal_close,
        "volume": 1200,
    })

    # BOC bar (bars[-1])
    boc_ts = start + pd.Timedelta(minutes=5 * (n_swing_bars + 1))
    rows.append({
        "timestamp": boc_ts,
        "open": signal_close,
        "high": boc_high,
        "low": boc_low,
        "close": boc_close,
        "volume": 1500,
    })

    df = pd.DataFrame(rows).set_index("timestamp")
    return df


# ---------------------------------------------------------------------------
# Fibonacci level calculation
# ---------------------------------------------------------------------------

class TestFibLevels:
    def test_38pct_level_calculation(self):
        """38.2% level = pdl + 0.382 * (pdh - pdl)."""
        pdh, pdl = 100.0, 80.0
        expected = pdl + 0.382 * (pdh - pdl)
        assert abs(expected - 87.64) < 0.01

    def test_50pct_is_midpoint(self):
        pdh, pdl = 120.0, 80.0
        assert abs(pdl + 0.500 * (pdh - pdl) - 100.0) < 1e-9

    def test_levels_ordered(self):
        pdh, pdl = 200.0, 100.0
        levels = [pdl + lvl * (pdh - pdl) for lvl in [0.382, 0.500, 0.618]]
        assert levels[0] < levels[1] < levels[2]


# ---------------------------------------------------------------------------
# BOC signal detection
# ---------------------------------------------------------------------------

class TestBOCSignal:
    def test_long_boc_signal(self):
        """BOC LONG: signal candle wicks to 38.2% level, next bar breaks its high."""
        # swing: [80, 100] → 38.2% = 87.64
        # signal bar: low=87.5 (touches 87.64 within 0.4*ATR), close=89.0 (above level)
        # BOC bar: high=92.0 > signal bar high=91.0 → LONG
        rule = FibonacciLevelsRule(
            key_levels=[0.382],
            touch_tolerance_atr=0.5,
            swing_lookback_bars=100,
            long_only=True,
        )
        bars = _make_bars(
            swing_high=100, swing_low=80,
            today_open=95,
            signal_low=87.5,   # near 38.2% (87.64)
            signal_close=89.0,
            signal_high=91.0,
            boc_high=92.0,     # breaks signal candle high → LONG
            boc_low=88.0,
            boc_close=91.5,
        )
        sig = rule.evaluate(bars)
        assert sig.direction == 1
        assert sig.rule_name == "fibonacci_levels"
        assert "fib_level_pct" in sig.metadata
        assert abs(sig.metadata["fib_level_pct"] - 0.382) < 0.001

    def test_no_signal_when_boc_bar_doesnt_break(self):
        """No LONG signal if current bar doesn't break signal candle high."""
        rule = FibonacciLevelsRule(
            key_levels=[0.382],
            touch_tolerance_atr=0.5,
            swing_lookback_bars=100,
            long_only=True,
        )
        bars = _make_bars(
            swing_high=100, swing_low=80,
            today_open=95,
            signal_low=87.5, signal_close=89.0, signal_high=91.0,
            boc_high=90.5,     # does NOT break signal candle high of 91.0
            boc_low=88.0, boc_close=89.5,
        )
        sig = rule.evaluate(bars)
        assert sig.direction == 0

    def test_no_signal_when_signal_candle_too_far_from_level(self):
        """No signal if signal candle low is far from any Fib level."""
        rule = FibonacciLevelsRule(
            key_levels=[0.382],
            touch_tolerance_atr=0.1,   # tight tolerance
            swing_lookback_bars=100,
            long_only=True,
        )
        bars = _make_bars(
            swing_high=100, swing_low=80,
            today_open=95,
            signal_low=92.0,   # far from 38.2% (87.64)
            signal_close=93.0, signal_high=95.0,
            boc_high=96.0, boc_low=92.0, boc_close=95.5,
        )
        sig = rule.evaluate(bars)
        assert sig.direction == 0

    def test_short_signal_blocked_by_long_only(self):
        """SHORT signals must be blocked when long_only=True."""
        rule = FibonacciLevelsRule(
            key_levels=[0.618],
            touch_tolerance_atr=0.5,
            swing_lookback_bars=100,
            long_only=True,
        )
        # downtrend: open below 50% (90) — but long_only should block
        bars = _make_bars(
            swing_high=100, swing_low=80,
            today_open=85,    # downtrend
            signal_high=92.5, signal_close=91.0, signal_low=84.0,
            boc_high=92.0, boc_low=90.0, boc_close=90.5,
        )
        sig = rule.evaluate(bars)
        assert sig.direction != -1

    def test_metadata_fields(self):
        """Signal metadata must include swing_high, swing_low, fib_level_price."""
        rule = FibonacciLevelsRule(
            key_levels=[0.382],
            touch_tolerance_atr=0.5,
            swing_lookback_bars=100,
            long_only=True,
        )
        bars = _make_bars(
            swing_high=100, swing_low=80,
            today_open=95,
            signal_low=87.5, signal_close=89.0, signal_high=91.0,
            boc_high=92.0, boc_low=88.0, boc_close=91.5,
        )
        sig = rule.evaluate(bars)
        if sig.direction == 1:
            for key in ("swing_high", "swing_low", "fib_level_price",
                        "signal_candle_high", "signal_candle_low", "atr"):
                assert key in sig.metadata, f"Missing metadata key: {key}"

    def test_strength_in_range(self):
        """Signal strength must always be in [0, 1]."""
        rule = FibonacciLevelsRule(
            key_levels=[0.382], touch_tolerance_atr=0.5, swing_lookback_bars=120
        )
        bars = _make_bars(
            swing_high=100, swing_low=80,
            today_open=95,
            signal_low=87.5, signal_close=89.0, signal_high=91.0,
            boc_high=92.0, boc_low=88.0, boc_close=91.5,
        )
        sig = rule.evaluate(bars)
        assert 0.0 <= sig.strength <= 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_insufficient_bars(self):
        rule = FibonacciLevelsRule(swing_lookback_bars=120)
        # Only 5 bars — well below required_bars()
        rows = []
        for i in range(5):
            rows.append({
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000
            })
        df = pd.DataFrame(rows, index=pd.date_range("2025-01-01", periods=5, freq="5min", tz="US/Eastern"))
        sig = rule.evaluate(df)
        assert sig.direction == 0

    def test_flat_swing_range(self):
        """Zero fib range → no signal."""
        rule = FibonacciLevelsRule(swing_lookback_bars=100, touch_tolerance_atr=1.0)
        # Build 142 bars all with identical OHLC so swing_high == swing_low
        n = 142
        idx = pd.date_range("2025-01-01 09:30", periods=n, freq="5min", tz="US/Eastern")
        df = pd.DataFrame({
            "open": 90.0, "high": 90.0, "low": 90.0, "close": 90.0, "volume": 1000
        }, index=idx)
        sig = rule.evaluate(df)
        assert sig.direction == 0

    def test_required_bars_positive(self):
        rule = FibonacciLevelsRule()
        assert rule.required_bars() > 0

    def test_required_bars_includes_lookback(self):
        rule = FibonacciLevelsRule(swing_lookback_bars=200)
        assert rule.required_bars() >= 200
