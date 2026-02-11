"""Integration tests for signal aggregation."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.base import BaseRule, RuleSignal
from engine.signal_aggregator import SignalAggregator, TradeDecision


class MockRule(BaseRule):
    """Mock rule for testing aggregator logic."""
    def __init__(self, name: str, role: str, signal: RuleSignal | None = None):
        super().__init__(name=name, role=role)
        self._signal = signal or RuleSignal.no_signal(name)

    def evaluate(self, bars: pd.DataFrame) -> RuleSignal:
        return self._signal

    def required_bars(self) -> int:
        return 1

    def set_signal(self, signal: RuleSignal):
        self._signal = signal


class TestSignalAggregator:
    def _make_bars(self, n=10):
        times = pd.date_range("2025-01-02 10:00", periods=n, freq="5min", tz="US/Eastern")
        return pd.DataFrame({
            "open": np.full(n, 5000),
            "high": np.full(n, 5001),
            "low": np.full(n, 4999),
            "close": np.full(n, 5000),
            "volume": np.full(n, 100),
        }, index=times)

    def test_no_primary_signal_no_trade(self):
        primary = MockRule("ema", "primary")
        agg = SignalAggregator(primary_rule=primary)
        decision = agg.evaluate(self._make_bars())
        assert not decision.should_trade

    def test_primary_signal_no_filters_no_confirmations_fails(self):
        """Primary signal alone is not enough without confirmations."""
        primary = MockRule("ema", "primary", RuleSignal(1, 0.8, "ema", "crossover"))
        agg = SignalAggregator(primary_rule=primary, min_confirmations=1)
        decision = agg.evaluate(self._make_bars())
        assert not decision.should_trade  # No confirmations

    def test_primary_plus_confirmation_trades(self):
        primary = MockRule("ema", "primary", RuleSignal(1, 0.8, "ema", "crossover"))
        confirm = MockRule("mr", "confirmation", RuleSignal(1, 0.5, "mr", "oversold"))
        agg = SignalAggregator(primary_rule=primary, confirmation_rules=[confirm])
        decision = agg.evaluate(self._make_bars())
        assert decision.should_trade
        assert decision.direction == 1
        assert decision.confidence > 0

    def test_filter_veto_blocks_trade(self):
        primary = MockRule("ema", "primary", RuleSignal(1, 0.8, "ema", "crossover"))
        filter_rule = MockRule("time", "filter",
            RuleSignal(0, 0.0, "time", "outside session", {"vetoed": True}))
        confirm = MockRule("mr", "confirmation", RuleSignal(1, 0.5, "mr", "oversold"))
        agg = SignalAggregator(primary_rule=primary, filter_rules=[filter_rule],
                               confirmation_rules=[confirm])
        decision = agg.evaluate(self._make_bars())
        assert not decision.should_trade
        assert len(decision.veto_reasons) > 0

    def test_filter_pass_allows_trade(self):
        primary = MockRule("ema", "primary", RuleSignal(1, 0.8, "ema", "crossover"))
        filter_rule = MockRule("time", "filter",
            RuleSignal(0, 0.0, "time", "session active", {"vetoed": False}))
        confirm = MockRule("mr", "confirmation", RuleSignal(1, 0.5, "mr", "oversold"))
        agg = SignalAggregator(primary_rule=primary, filter_rules=[filter_rule],
                               confirmation_rules=[confirm])
        decision = agg.evaluate(self._make_bars())
        assert decision.should_trade

    def test_wrong_direction_confirmation_doesnt_count(self):
        """Confirmation in opposite direction should not count."""
        primary = MockRule("ema", "primary", RuleSignal(1, 0.8, "ema", "crossover UP"))
        # Confirmation says SHORT but primary says LONG
        confirm = MockRule("mr", "confirmation", RuleSignal(-1, 0.5, "mr", "overbought"))
        agg = SignalAggregator(primary_rule=primary, confirmation_rules=[confirm])
        decision = agg.evaluate(self._make_bars())
        assert not decision.should_trade  # No matching confirmations

    def test_multiple_confirmations_boost_confidence(self):
        primary = MockRule("ema", "primary", RuleSignal(-1, 0.6, "ema", "crossover DOWN"))
        c1 = MockRule("mr", "confirmation", RuleSignal(-1, 0.5, "mr", "overbought"))
        c2 = MockRule("rej", "confirmation", RuleSignal(-1, 0.4, "rej", "upper wick"))
        agg = SignalAggregator(primary_rule=primary, confirmation_rules=[c1, c2])
        decision = agg.evaluate(self._make_bars())
        assert decision.should_trade
        assert decision.direction == -1
        # Confidence should be boosted beyond primary's 0.6
        assert decision.confidence > 0.6

    def test_min_confirmations_zero_always_trades(self):
        primary = MockRule("ema", "primary", RuleSignal(1, 0.8, "ema", "crossover"))
        agg = SignalAggregator(primary_rule=primary, min_confirmations=0)
        decision = agg.evaluate(self._make_bars())
        assert decision.should_trade

    def test_role_validation(self):
        with pytest.raises(ValueError):
            SignalAggregator(primary_rule=MockRule("bad", "filter"))

    def test_required_bars(self):
        primary = MockRule("ema", "primary")
        agg = SignalAggregator(primary_rule=primary)
        assert agg.required_bars() >= 1

    def test_decision_summary(self):
        primary = MockRule("ema", "primary", RuleSignal(1, 0.8, "ema", "crossover"))
        confirm = MockRule("mr", "confirmation", RuleSignal(1, 0.5, "mr", "oversold"))
        agg = SignalAggregator(primary_rule=primary, confirmation_rules=[confirm])
        decision = agg.evaluate(self._make_bars())
        summary = decision.summary
        assert "LONG" in summary

    def test_no_trade_summary(self):
        primary = MockRule("ema", "primary")
        agg = SignalAggregator(primary_rule=primary)
        decision = agg.evaluate(self._make_bars())
        assert "NO TRADE" in decision.summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
