"""Risk management edge case tests."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.risk_manager import RiskManager, TradeRecord


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager(
            max_daily_loss=-400.0,
            per_trade_max_loss=100.0,
            max_consecutive_losses=3,
            cooldown_bars=5,
            drawdown_buffer=500.0,
        )
        self.rm.reset_all(starting_equity=50000.0)

    def test_initial_state_can_trade(self):
        can, reason = self.rm.can_trade()
        assert can
        assert reason == "OK"

    def test_daily_loss_limit(self):
        # Record a big loss
        trade = TradeRecord(0, 1, 1, 5000, 4920, -400.0, "stop_loss")
        self.rm.record_trade(trade)
        can, reason = self.rm.can_trade()
        assert not can
        assert "Daily loss" in reason

    def test_consecutive_losses_circuit_breaker(self):
        for i in range(3):
            trade = TradeRecord(i, i+1, 1, 5000, 4990, -50.0, "stop_loss")
            self.rm.record_trade(trade)
            # Advance bars past cooldown
            for _ in range(10):
                self.rm.tick_bar()
        can, reason = self.rm.can_trade()
        assert not can
        assert "Circuit breaker" in reason

    def test_win_resets_consecutive_losses(self):
        # Two losses
        for i in range(2):
            trade = TradeRecord(i, i+1, 1, 5000, 4990, -50.0, "stop_loss")
            self.rm.record_trade(trade)
            for _ in range(10):
                self.rm.tick_bar()
        # One win
        trade = TradeRecord(3, 4, 1, 5000, 5020, 100.0, "profit_target")
        self.rm.record_trade(trade)
        for _ in range(10):
            self.rm.tick_bar()
        can, reason = self.rm.can_trade()
        assert can

    def test_cooldown_bars(self):
        trade = TradeRecord(0, 1, 1, 5000, 4990, -50.0, "stop_loss")
        self.rm.record_trade(trade)
        # Immediately after trade, cooldown should block
        can, reason = self.rm.can_trade()
        assert not can
        assert "Cooldown" in reason

        # After enough bars, should be OK
        for _ in range(6):
            self.rm.tick_bar()
        can, reason = self.rm.can_trade()
        assert can

    def test_per_trade_max_loss_cap(self):
        # Stop distance should be capped at per_trade_max_loss
        stop = self.rm.compute_stop_price(5000.0, 1, atr=30.0, stop_loss_atr=1.5)
        # Raw stop: 5000 - 45 = 4955 ($225 loss)
        # Capped: 5000 - 20 = 4980 ($100 loss)
        assert stop == 4980.0

    def test_stop_not_capped_when_under_limit(self):
        stop = self.rm.compute_stop_price(5000.0, 1, atr=10.0, stop_loss_atr=1.5)
        # Raw stop: 5000 - 15 = 4985 ($75 loss < $100 limit)
        assert stop == 4985.0

    def test_short_stop_price(self):
        stop = self.rm.compute_stop_price(5000.0, -1, atr=10.0, stop_loss_atr=1.5)
        # SHORT: 5000 + 15 = 5015
        assert stop == 5015.0

    def test_target_price_long(self):
        target = self.rm.compute_target_price(5000.0, 1, atr=10.0, profit_target_atr=2.0)
        assert target == 5020.0

    def test_target_price_short(self):
        target = self.rm.compute_target_price(5000.0, -1, atr=10.0, profit_target_atr=2.0)
        assert target == 4980.0

    def test_drawdown_buffer(self):
        # Lose $500 to hit drawdown buffer
        trade = TradeRecord(0, 1, 1, 5000, 4900, -500.0, "stop_loss")
        self.rm.record_trade(trade)
        for _ in range(10):
            self.rm.tick_bar()
        # Need to reset halt from daily loss first to test drawdown
        # Actually daily loss would trigger first at -400
        # Let's test with smaller losses that don't hit daily limit
        self.rm.reset_all(starting_equity=50000.0)
        self.rm.max_daily_loss = -1000.0  # Raise daily limit

        # Two days of losses totaling $500
        trade1 = TradeRecord(0, 1, 1, 5000, 4950, -250.0, "stop_loss")
        self.rm.record_trade(trade1)
        self.rm.reset_daily()  # New day
        trade2 = TradeRecord(2, 3, 1, 5000, 4950, -250.0, "stop_loss")
        self.rm.record_trade(trade2)
        for _ in range(10):
            self.rm.tick_bar()

        can, reason = self.rm.can_trade()
        assert not can
        assert "Drawdown" in reason

    def test_daily_reset(self):
        # Trip circuit breaker
        for i in range(3):
            trade = TradeRecord(i, i+1, 1, 5000, 4990, -50.0, "stop_loss")
            self.rm.record_trade(trade)
        self.rm.reset_daily()
        can, reason = self.rm.can_trade()
        assert can

    def test_session_stats(self):
        trade1 = TradeRecord(0, 1, 1, 5000, 5020, 100.0, "profit_target")
        trade2 = TradeRecord(2, 3, -1, 5000, 5010, -50.0, "stop_loss")
        self.rm.record_trade(trade1)
        self.rm.record_trade(trade2)
        stats = self.rm.session_stats
        assert stats["total_trades"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["daily_pnl"] == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
