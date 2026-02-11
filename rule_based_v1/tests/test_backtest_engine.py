"""Backtest engine correctness tests with hand-crafted scenarios."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.base import BaseRule, RuleSignal
from engine.signal_aggregator import SignalAggregator
from engine.risk_manager import RiskManager
from engine.backtest_engine import BacktestEngine, BacktestResult


class AlwaysLongRule(BaseRule):
    """Test rule that always signals LONG."""
    def __init__(self):
        super().__init__(name="always_long", role="primary")
    def evaluate(self, bars):
        return RuleSignal(1, 0.8, "always_long", "test signal")
    def required_bars(self):
        return 5


class AlwaysConfirmRule(BaseRule):
    """Test confirmation that always confirms."""
    def __init__(self):
        super().__init__(name="always_confirm", role="confirmation")
    def evaluate(self, bars):
        return RuleSignal(1, 0.5, "always_confirm", "confirmed")
    def required_bars(self):
        return 1


class NeverSignalRule(BaseRule):
    """Test rule that never signals."""
    def __init__(self):
        super().__init__(name="never", role="primary")
    def evaluate(self, bars):
        return RuleSignal.no_signal("never")
    def required_bars(self):
        return 1


def make_trending_bars(n: int, direction: float = 1.0, volatility: float = 2.0) -> pd.DataFrame:
    """Create bars with a clear trend for testing exits."""
    times = pd.date_range("2025-01-02 09:35", periods=n, freq="5min", tz="US/Eastern")
    prices = 5000.0 + np.arange(n) * direction * 0.5
    noise = np.random.RandomState(42).randn(n) * 0.2

    return pd.DataFrame({
        "open": prices + noise - 0.3,
        "high": prices + volatility,
        "low": prices - volatility,
        "close": prices + noise,
        "volume": np.full(n, 150.0),
    }, index=times)


class TestBacktestEngine:
    def test_no_signals_no_trades(self):
        agg = SignalAggregator(primary_rule=NeverSignalRule(), min_confirmations=0)
        rm = RiskManager()
        engine = BacktestEngine(agg, rm)
        bars = make_trending_bars(100)
        result = engine.run(bars)
        assert result.num_trades == 0

    def test_basic_trade_execution(self):
        """Should produce trades with always-long + always-confirm."""
        agg = SignalAggregator(
            primary_rule=AlwaysLongRule(),
            confirmation_rules=[AlwaysConfirmRule()],
        )
        rm = RiskManager(max_daily_loss=-2000.0, max_consecutive_losses=100)
        engine = BacktestEngine(agg, rm, time_stop_bars=10)
        bars = make_trending_bars(100, direction=1.0)
        result = engine.run(bars)
        assert result.num_trades > 0

    def test_commission_applied(self):
        """Trades should have commission deducted."""
        agg = SignalAggregator(
            primary_rule=AlwaysLongRule(),
            confirmation_rules=[AlwaysConfirmRule()],
        )
        rm = RiskManager(max_daily_loss=-5000.0, max_consecutive_losses=100)
        engine = BacktestEngine(agg, rm, commission_per_side=0.62, time_stop_bars=5)
        bars = make_trending_bars(50, direction=0.0, volatility=0.1)  # Flat
        result = engine.run(bars)
        if result.num_trades > 0:
            # With no trend, trades should mostly lose money (commission + slippage)
            total_commission = result.num_trades * 2 * 0.62
            # PnL should be negative due to costs on flat market
            assert result.total_pnl < total_commission

    def test_stop_loss_triggered(self):
        """Downtrend should trigger stop losses for LONG trades."""
        agg = SignalAggregator(
            primary_rule=AlwaysLongRule(),
            confirmation_rules=[AlwaysConfirmRule()],
        )
        rm = RiskManager(max_daily_loss=-5000.0, max_consecutive_losses=100,
                         cooldown_bars=0)
        engine = BacktestEngine(agg, rm, stop_loss_atr=1.0, time_stop_bars=50)
        bars = make_trending_bars(100, direction=-2.0, volatility=3.0)
        result = engine.run(bars)
        stop_trades = [t for t in result.trades if t.exit_reason == "stop_loss"]
        assert len(stop_trades) > 0

    def test_profit_target_triggered(self):
        """Strong uptrend should trigger profit targets for LONG trades."""
        agg = SignalAggregator(
            primary_rule=AlwaysLongRule(),
            confirmation_rules=[AlwaysConfirmRule()],
        )
        rm = RiskManager(max_daily_loss=-5000.0, max_consecutive_losses=100,
                         cooldown_bars=0)
        engine = BacktestEngine(agg, rm, profit_target_atr=1.0, stop_loss_atr=3.0,
                                time_stop_bars=50)
        bars = make_trending_bars(100, direction=3.0, volatility=2.0)
        result = engine.run(bars)
        target_trades = [t for t in result.trades if t.exit_reason == "profit_target"]
        assert len(target_trades) > 0

    def test_time_stop_triggered(self):
        """Time stop should exit after N bars."""
        agg = SignalAggregator(
            primary_rule=AlwaysLongRule(),
            confirmation_rules=[AlwaysConfirmRule()],
        )
        rm = RiskManager(max_daily_loss=-5000.0, max_consecutive_losses=100,
                         cooldown_bars=0)
        engine = BacktestEngine(agg, rm, time_stop_bars=5,
                                profit_target_atr=100.0, stop_loss_atr=100.0)
        bars = make_trending_bars(100, direction=0.0, volatility=0.5)
        result = engine.run(bars)
        time_stops = [t for t in result.trades if t.exit_reason == "time_stop"]
        assert len(time_stops) > 0

    def test_equity_curve_monotone_with_starting(self):
        """Equity curve should start at starting_equity."""
        agg = SignalAggregator(primary_rule=NeverSignalRule(), min_confirmations=0)
        rm = RiskManager()
        engine = BacktestEngine(agg, rm)
        bars = make_trending_bars(50)
        result = engine.run(bars, starting_equity=50000.0)
        assert result.equity_curve.iloc[0] == 50000.0

    def test_backtest_result_metrics(self):
        """BacktestResult should compute correct metrics."""
        from engine.risk_manager import TradeRecord
        trades = [
            TradeRecord(0, 5, 1, 5000, 5010, 50.0, "profit_target"),
            TradeRecord(10, 15, 1, 5000, 4990, -50.0, "stop_loss"),
            TradeRecord(20, 25, 1, 5000, 5020, 100.0, "profit_target"),
        ]
        result = BacktestResult(
            trades=trades,
            equity_curve=pd.Series([50000, 50050, 50000, 50100]),
            daily_pnl=pd.Series({"2025-01-02": 100.0}),
        )
        assert result.total_pnl == 100.0
        assert result.num_trades == 3
        assert abs(result.win_rate - 2/3) < 0.01
        assert result.profit_factor == 150.0 / 50.0
        assert result.max_consecutive_losses == 1

    def test_session_close_flatten(self):
        """Positions should be flattened at session close."""
        agg = SignalAggregator(
            primary_rule=AlwaysLongRule(),
            confirmation_rules=[AlwaysConfirmRule()],
        )
        rm = RiskManager(max_daily_loss=-5000.0, max_consecutive_losses=100)
        engine = BacktestEngine(agg, rm, time_stop_bars=1000,
                                profit_target_atr=100.0, stop_loss_atr=100.0)
        # Create bars spanning to end of day
        times = pd.date_range("2025-01-02 15:30", periods=10, freq="5min", tz="US/Eastern")
        bars_eod = pd.DataFrame({
            "open": np.full(10, 5000),
            "high": np.full(10, 5001),
            "low": np.full(10, 4999),
            "close": np.full(10, 5000),
            "volume": np.full(10, 100),
        }, index=times)
        # Prepend warmup bars
        warmup = make_trending_bars(20, direction=0.0)
        bars = pd.concat([warmup, bars_eod])
        result = engine.run(bars)
        close_trades = [t for t in result.trades if t.exit_reason == "session_close"]
        assert len(close_trades) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
