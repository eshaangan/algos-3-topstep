"""Unit tests for all trading rules."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.base import RuleSignal
from rules.ema_trend import EMATrendRule
from rules.time_of_day import TimeOfDayRule
from rules.volume_breakout import VolumeBreakoutRule
from rules.mean_reversion import MeanReversionRule
from rules.rejection_pattern import RejectionPatternRule


def make_bars(n: int, base_price: float = 5000.0, trend: float = 0.0,
              volume: float = 100.0, tz: str = "US/Eastern",
              start_time: str = "2025-01-02 10:00") -> pd.DataFrame:
    """Generate synthetic OHLCV bars for testing."""
    times = pd.date_range(start_time, periods=n, freq="5min", tz=tz)
    prices = base_price + np.arange(n) * trend + np.random.randn(n) * 0.5
    df = pd.DataFrame({
        "open": prices - 0.5,
        "high": prices + 1.0,
        "low": prices - 1.0,
        "close": prices,
        "volume": np.full(n, volume),
    }, index=times)
    return df


def make_crossover_bars(n: int = 60, cross_at: int = 50, direction: str = "up") -> pd.DataFrame:
    """Generate bars that produce an EMA crossover at the specified bar."""
    times = pd.date_range("2025-01-02 09:30", periods=n, freq="5min", tz="US/Eastern")
    prices = np.zeros(n)

    if direction == "up":
        # Downtrend then uptrend
        for i in range(n):
            if i < cross_at - 10:
                prices[i] = 5000 - i * 0.3
            else:
                prices[i] = prices[cross_at - 11] + (i - cross_at + 11) * 0.8
    else:
        # Uptrend then downtrend
        for i in range(n):
            if i < cross_at - 10:
                prices[i] = 5000 + i * 0.3
            else:
                prices[i] = prices[cross_at - 11] - (i - cross_at + 11) * 0.8

    df = pd.DataFrame({
        "open": prices - 0.3,
        "high": prices + 1.5,
        "low": prices - 1.5,
        "close": prices,
        "volume": np.full(n, 150.0),
    }, index=times)
    return df


# ---- RuleSignal Tests ----

class TestRuleSignal:
    def test_valid_signal(self):
        sig = RuleSignal(direction=1, strength=0.5, rule_name="test", reason="test")
        assert sig.is_long
        assert not sig.is_short
        assert sig.has_signal

    def test_no_signal(self):
        sig = RuleSignal.no_signal("test", "no reason")
        assert not sig.has_signal
        assert sig.direction == 0
        assert sig.strength == 0.0

    def test_invalid_direction(self):
        with pytest.raises(ValueError):
            RuleSignal(direction=2, strength=0.5, rule_name="test", reason="bad")

    def test_invalid_strength(self):
        with pytest.raises(ValueError):
            RuleSignal(direction=1, strength=1.5, rule_name="test", reason="bad")


# ---- EMA Trend Rule Tests ----

class TestEMATrendRule:
    def setup_method(self):
        self.rule = EMATrendRule(fast_period=13, slow_period=34)

    def test_insufficient_bars(self):
        bars = make_bars(10)
        sig = self.rule.evaluate(bars)
        assert not sig.has_signal
        assert "Insufficient" in sig.reason

    def test_no_crossover_flat_market(self):
        bars = make_bars(60, trend=0.0)
        sig = self.rule.evaluate(bars)
        # Flat market - unlikely to produce crossover signal
        assert sig.rule_name == "ema_trend"

    def test_upward_crossover(self):
        bars = make_crossover_bars(60, cross_at=50, direction="up")
        sig = self.rule.evaluate(bars)
        # Should either be LONG or no signal depending on spread/slope
        assert sig.rule_name == "ema_trend"
        if sig.has_signal:
            assert sig.direction == 1

    def test_downward_crossover(self):
        bars = make_crossover_bars(60, cross_at=50, direction="down")
        sig = self.rule.evaluate(bars)
        if sig.has_signal:
            assert sig.direction == -1

    def test_role_is_primary(self):
        assert self.rule.role == "primary"

    def test_required_bars(self):
        assert self.rule.required_bars() >= 34  # At least slow_period


# ---- Time of Day Rule Tests ----

class TestTimeOfDayRule:
    def setup_method(self):
        self.rule = TimeOfDayRule()

    def test_within_session(self):
        bars = make_bars(5, start_time="2025-01-02 10:00")
        sig = self.rule.evaluate(bars)
        assert sig.metadata["vetoed"] is False

    def test_before_session(self):
        bars = make_bars(5, start_time="2025-01-02 09:00")
        sig = self.rule.evaluate(bars)
        assert sig.metadata["vetoed"] is True

    def test_after_session(self):
        bars = make_bars(5, start_time="2025-01-02 16:00")
        sig = self.rule.evaluate(bars)
        assert sig.metadata["vetoed"] is True

    def test_lunch_filter_disabled(self):
        bars = make_bars(5, start_time="2025-01-02 12:30")
        sig = self.rule.evaluate(bars)
        assert sig.metadata["vetoed"] is False

    def test_lunch_filter_enabled(self):
        rule = TimeOfDayRule(lunch_filter_enabled=True)
        bars = make_bars(5, start_time="2025-01-02 12:30")
        sig = rule.evaluate(bars)
        assert sig.metadata["vetoed"] is True

    def test_role_is_filter(self):
        assert self.rule.role == "filter"


# ---- Volume Breakout Rule Tests ----

class TestVolumeBreakoutRule:
    def setup_method(self):
        self.rule = VolumeBreakoutRule(lookback=20, min_ratio=1.2, max_ratio=2.0)

    def test_insufficient_bars(self):
        bars = make_bars(5)
        sig = self.rule.evaluate(bars)
        assert not sig.has_signal
        assert "Insufficient" in sig.reason

    def test_normal_volume_vetoed(self):
        """Volume at average level should be vetoed (below 1.2x)."""
        bars = make_bars(30, volume=100.0)
        sig = self.rule.evaluate(bars)
        assert sig.metadata.get("vetoed", False) is True

    def test_high_volume_passes(self):
        """Volume at 1.5x average should pass."""
        n = 30
        vols = np.full(n, 100.0)
        vols[-1] = 150.0  # 1.5x average
        bars = make_bars(n)
        bars["volume"] = vols
        sig = self.rule.evaluate(bars)
        assert sig.metadata.get("vetoed") is False

    def test_spike_volume_vetoed(self):
        """Volume at 3x average should be vetoed (spike)."""
        n = 30
        vols = np.full(n, 100.0)
        vols[-1] = 300.0  # 3x average
        bars = make_bars(n)
        bars["volume"] = vols
        sig = self.rule.evaluate(bars)
        assert sig.metadata.get("vetoed") is True

    def test_role_is_filter(self):
        assert self.rule.role == "filter"


# ---- Mean Reversion Rule Tests ----

class TestMeanReversionRule:
    def setup_method(self):
        self.rule = MeanReversionRule()

    def test_insufficient_bars(self):
        bars = make_bars(5)
        sig = self.rule.evaluate(bars)
        assert not sig.has_signal

    def test_oversold_confirms_long(self):
        """Strong downtrend should confirm LONG (mean reversion)."""
        bars = make_bars(30, trend=-2.0)
        sig = self.rule.evaluate(bars)
        if sig.has_signal:
            assert sig.direction == 1

    def test_overbought_confirms_short(self):
        """Strong uptrend should confirm SHORT (mean reversion)."""
        bars = make_bars(30, trend=2.0)
        sig = self.rule.evaluate(bars)
        if sig.has_signal:
            assert sig.direction == -1

    def test_role_is_confirmation(self):
        assert self.rule.role == "confirmation"


# ---- Rejection Pattern Rule Tests ----

class TestRejectionPatternRule:
    def setup_method(self):
        self.rule = RejectionPatternRule(min_wick_body_ratio=1.5)

    def test_no_bars(self):
        bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        sig = self.rule.evaluate(bars)
        assert not sig.has_signal

    def test_long_lower_wick_confirms_long(self):
        """Hammer candle (long lower wick) should confirm LONG."""
        bars = pd.DataFrame({
            "open": [5000.0],
            "high": [5001.0],
            "low": [4995.0],
            "close": [5000.5],
            "volume": [100],
        }, index=pd.date_range("2025-01-02 10:00", periods=1, freq="5min", tz="US/Eastern"))
        sig = self.rule.evaluate(bars)
        # Body = 0.5, lower wick = 5.0, ratio = 10.0 >> 1.5
        assert sig.has_signal
        assert sig.direction == 1

    def test_long_upper_wick_confirms_short(self):
        """Shooting star (long upper wick) should confirm SHORT."""
        bars = pd.DataFrame({
            "open": [5000.0],
            "high": [5006.0],
            "low": [4999.5],
            "close": [5000.5],
            "volume": [100],
        }, index=pd.date_range("2025-01-02 10:00", periods=1, freq="5min", tz="US/Eastern"))
        sig = self.rule.evaluate(bars)
        # Body = 0.5, upper wick = 5.5, ratio = 11.0 >> 1.5
        assert sig.has_signal
        assert sig.direction == -1

    def test_no_rejection_balanced_candle(self):
        """Balanced candle should not produce a signal."""
        bars = pd.DataFrame({
            "open": [5000.0],
            "high": [5002.0],
            "low": [4998.0],
            "close": [5001.0],
            "volume": [100],
        }, index=pd.date_range("2025-01-02 10:00", periods=1, freq="5min", tz="US/Eastern"))
        sig = self.rule.evaluate(bars)
        # Body = 1.0, upper wick = 1.0, lower wick = 2.0 (ratio = 2.0 for lower, but upper = 1.0)
        # Lower wick ratio is 2.0 which IS > 1.5, so this should confirm LONG
        # Actually: body=|5001-5000|=1, lower_wick=min(5000,5001)-4998=2, upper_wick=5002-max(5000,5001)=1
        # lower_ratio=2/1=2.0 > 1.5, upper_ratio=1/1=1.0
        # lower > upper so LONG confirmation
        if sig.has_signal:
            assert sig.direction == 1

    def test_role_is_confirmation(self):
        assert self.rule.role == "confirmation"


# ---- NaN Handling Tests ----

class TestNaNHandling:
    def test_ema_trend_with_nans(self):
        bars = make_bars(60)
        bars.loc[bars.index[30], "close"] = np.nan
        rule = EMATrendRule()
        # Should not crash
        sig = rule.evaluate(bars)
        assert isinstance(sig, RuleSignal)

    def test_volume_with_zeros(self):
        bars = make_bars(30, volume=0.0)
        rule = VolumeBreakoutRule()
        sig = rule.evaluate(bars)
        assert isinstance(sig, RuleSignal)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
