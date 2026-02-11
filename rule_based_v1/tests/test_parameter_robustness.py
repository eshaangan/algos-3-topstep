"""Parameter robustness tests - verify strategy survives ±20% parameter changes."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.ema_trend import EMATrendRule
from rules.time_of_day import TimeOfDayRule
from rules.volume_breakout import VolumeBreakoutRule
from rules.mean_reversion import MeanReversionRule
from rules.rejection_pattern import RejectionPatternRule
from rules.base import RuleSignal


def make_test_bars(n: int = 80) -> pd.DataFrame:
    """Generate realistic test bars."""
    np.random.seed(42)
    times = pd.date_range("2025-01-02 09:30", periods=n, freq="5min", tz="US/Eastern")
    prices = 5000.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": prices - 0.3,
        "high": prices + np.abs(np.random.randn(n)) * 1.5,
        "low": prices - np.abs(np.random.randn(n)) * 1.5,
        "close": prices,
        "volume": 100 + np.random.randint(0, 100, n),
    }, index=times)


class TestEMATrendRobustness:
    """EMA Trend rule should not break with ±20% parameter changes."""

    @pytest.mark.parametrize("fast_period", [10, 11, 13, 15, 16])
    def test_fast_period_variation(self, fast_period):
        rule = EMATrendRule(fast_period=fast_period, slow_period=34)
        bars = make_test_bars()
        sig = rule.evaluate(bars)
        assert isinstance(sig, RuleSignal)

    @pytest.mark.parametrize("slow_period", [27, 30, 34, 38, 41])
    def test_slow_period_variation(self, slow_period):
        rule = EMATrendRule(fast_period=13, slow_period=slow_period)
        bars = make_test_bars()
        sig = rule.evaluate(bars)
        assert isinstance(sig, RuleSignal)

    @pytest.mark.parametrize("spread_ratio", [0.24, 0.27, 0.30, 0.33, 0.36])
    def test_spread_threshold_variation(self, spread_ratio):
        rule = EMATrendRule(min_spread_atr_ratio=spread_ratio)
        bars = make_test_bars()
        sig = rule.evaluate(bars)
        assert isinstance(sig, RuleSignal)


class TestVolumeBreakoutRobustness:
    @pytest.mark.parametrize("min_ratio", [0.96, 1.08, 1.20, 1.32, 1.44])
    def test_min_ratio_variation(self, min_ratio):
        rule = VolumeBreakoutRule(min_ratio=min_ratio, max_ratio=2.0)
        bars = make_test_bars()
        sig = rule.evaluate(bars)
        assert isinstance(sig, RuleSignal)

    @pytest.mark.parametrize("max_ratio", [1.6, 1.8, 2.0, 2.2, 2.4])
    def test_max_ratio_variation(self, max_ratio):
        rule = VolumeBreakoutRule(min_ratio=1.2, max_ratio=max_ratio)
        bars = make_test_bars()
        sig = rule.evaluate(bars)
        assert isinstance(sig, RuleSignal)


class TestMeanReversionRobustness:
    @pytest.mark.parametrize("long_bb", [0.24, 0.27, 0.30, 0.33, 0.36])
    def test_bb_threshold_variation(self, long_bb):
        rule = MeanReversionRule(long_bb_threshold=long_bb, short_bb_threshold=1.0 - long_bb)
        bars = make_test_bars()
        sig = rule.evaluate(bars)
        assert isinstance(sig, RuleSignal)

    @pytest.mark.parametrize("rsi_long", [28, 31, 35, 38, 42])
    def test_rsi_threshold_variation(self, rsi_long):
        rule = MeanReversionRule(rsi_long_threshold=rsi_long, rsi_short_threshold=100 - rsi_long)
        bars = make_test_bars()
        sig = rule.evaluate(bars)
        assert isinstance(sig, RuleSignal)


class TestRejectionPatternRobustness:
    @pytest.mark.parametrize("ratio", [1.2, 1.35, 1.5, 1.65, 1.8])
    def test_wick_ratio_variation(self, ratio):
        rule = RejectionPatternRule(min_wick_body_ratio=ratio)
        bars = make_test_bars()
        sig = rule.evaluate(bars)
        assert isinstance(sig, RuleSignal)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
