"""
Tests for risk manager equity floor enforcement.

Validates that equity never goes below zero and trading halts on liquidation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_intraday_v3.backtesting_v3.risk import RiskManager


def create_risk_manager(starting_capital=1000.0, max_daily_loss=500.0, max_drawdown=800.0, hwm_update_policy="end_of_day"):
    """Helper to create RiskManager with simplified parameters."""
    risk_cfg = {
        "topstep": {
            "starting_balance": starting_capital,
        },
        "daily_loss_limit": {
            "enabled": True,
            "max_daily_loss": max_daily_loss,
        },
        "trailing_drawdown": {
            "enabled": True,
            "max_drawdown": max_drawdown,
            "hwm_update_policy": hwm_update_policy,
        },
    }
    return RiskManager(risk_cfg)


class TestEquityFloorPreventsNegative:
    """Tests that equity floor prevents negative balance."""

    def test_equity_floor_on_large_loss(self):
        """Test that equity is capped at zero when loss exceeds balance."""
        # Initialize risk manager with $1000 starting capital
        rm = create_risk_manager(
            starting_capital=1000.0,
            max_daily_loss=500.0,
            max_drawdown=800.0,
        )

        # Simulate a catastrophic loss of -$2000 (exceeds starting capital)
        entry_ts = pd.Timestamp("2020-01-01 10:00", tz="UTC")
        exit_ts = pd.Timestamp("2020-01-01 10:05", tz="UTC")

        rm.record_trade(entry_ts, exit_ts, pnl_usd=-2000.0)

        # Verify equity floored at zero (not -$1000)
        assert rm.equity == 0.0, "Equity should be floored at zero"
        assert rm.equity >= 0.0, "Equity should never be negative"

    def test_equity_floor_on_exact_balance_loss(self):
        """Test equity floor when loss exactly equals balance."""
        rm = create_risk_manager(starting_capital=1000.0)

        entry_ts = pd.Timestamp("2020-01-01 10:00", tz="UTC")
        exit_ts = pd.Timestamp("2020-01-01 10:05", tz="UTC")

        rm.record_trade(entry_ts, exit_ts, pnl_usd=-1000.0)

        # Verify equity is exactly zero
        assert rm.equity == 0.0, "Equity should be zero after full loss"

    def test_normal_losses_work_correctly(self):
        """Test that normal losses (within equity) work as before."""
        rm = create_risk_manager(starting_capital=1000.0)

        entry_ts = pd.Timestamp("2020-01-01 10:00", tz="UTC")
        exit_ts = pd.Timestamp("2020-01-01 10:05", tz="UTC")

        # Normal loss of -$100
        rm.record_trade(entry_ts, exit_ts, pnl_usd=-100.0)

        # Verify equity updated correctly
        assert rm.equity == 900.0, "Normal losses should reduce equity normally"

    def test_multiple_losses_floor_at_zero(self):
        """Test multiple losses eventually floor at zero."""
        rm = create_risk_manager(starting_capital=1000.0, max_daily_loss=2000.0)

        timestamps = pd.date_range("2020-01-01 10:00", periods=5, freq="5min", tz="UTC")

        # Series of losses: -300, -300, -300, -300, -300 = -1500 total
        for i in range(5):
            entry_ts = timestamps[i]
            exit_ts = timestamps[i] + pd.Timedelta(minutes=5)
            rm.record_trade(entry_ts, exit_ts, pnl_usd=-300.0)

        # After 4 losses, equity should be near zero
        # Fifth loss should be capped to prevent negative equity
        assert rm.equity >= 0.0, "Equity should never go negative"
        assert rm.equity == 0.0, "Equity should be floored at zero after excessive losses"

    def test_equity_floor_with_gains_then_catastrophic_loss(self):
        """Test equity floor after initial gains followed by large loss."""
        rm = create_risk_manager(starting_capital=1000.0)

        timestamps = pd.date_range("2020-01-01 10:00", periods=3, freq="5min", tz="UTC")

        # First trade: +$500 gain
        rm.record_trade(timestamps[0], timestamps[0] + pd.Timedelta(minutes=5), pnl_usd=500.0)
        assert rm.equity == 1500.0

        # Second trade: -$2000 loss (catastrophic)
        rm.record_trade(timestamps[1], timestamps[1] + pd.Timedelta(minutes=5), pnl_usd=-2000.0)

        # Equity should be floored at zero (not -$500)
        assert rm.equity == 0.0, "Equity should be floored at zero"


class TestLiquidationHaltsTrading:
    """Tests that liquidation halts trading."""

    def test_halted_after_liquidation(self):
        """Test that halted_today is set after liquidation."""
        rm = create_risk_manager(starting_capital=1000.0)

        entry_ts = pd.Timestamp("2020-01-01 10:00", tz="UTC")
        exit_ts = pd.Timestamp("2020-01-01 10:05", tz="UTC")

        # Catastrophic loss
        rm.record_trade(entry_ts, exit_ts, pnl_usd=-2000.0)

        # Verify trading halted
        assert rm.halted_today is True, "Trading should be halted after liquidation"
        assert rm.equity == 0.0, "Equity should be zero"

    def test_halted_status_after_liquidation(self):
        """Test that halted_today is True after liquidation."""
        rm = create_risk_manager(starting_capital=1000.0)

        # First liquidate the account
        entry_ts = pd.Timestamp("2020-01-01 10:00", tz="UTC")
        exit_ts = pd.Timestamp("2020-01-01 10:05", tz="UTC")
        rm.record_trade(entry_ts, exit_ts, pnl_usd=-2000.0)

        # Verify halted flag is set
        assert rm.halted_today is True, "halted_today should be True after liquidation"
        assert rm.equity == 0.0, "Equity should be zero"

    def test_not_halted_on_normal_loss(self):
        """Test that normal losses don't trigger halt."""
        rm = create_risk_manager(starting_capital=1000.0, max_daily_loss=500.0)

        entry_ts = pd.Timestamp("2020-01-01 10:00", tz="UTC")
        exit_ts = pd.Timestamp("2020-01-01 10:05", tz="UTC")

        # Normal loss (within limits)
        rm.record_trade(entry_ts, exit_ts, pnl_usd=-100.0)

        # Verify not halted
        assert rm.halted_today is False, "Normal losses should not halt trading"
        assert rm.equity == 900.0


class TestDailyPnLTracking:
    """Tests that daily PnL is tracked correctly with equity floor."""

    def test_daily_pnl_capped_on_liquidation(self):
        """Test that daily PnL reflects capped loss on liquidation."""
        rm = create_risk_manager(starting_capital=1000.0, max_daily_loss=2000.0)

        entry_ts = pd.Timestamp("2020-01-01 10:00", tz="UTC")
        exit_ts = pd.Timestamp("2020-01-01 10:05", tz="UTC")

        # Catastrophic loss of -$2000 (but only $1000 available)
        rm.record_trade(entry_ts, exit_ts, pnl_usd=-2000.0)

        # Daily PnL should reflect actual loss (-$1000), not requested loss (-$2000)
        assert rm.daily_pnl == -1000.0, "Daily PnL should reflect capped loss"
        assert rm.equity == 0.0, "Equity should be zero"

    def test_daily_pnl_normal_tracking(self):
        """Test that daily PnL tracks normally without liquidation."""
        rm = create_risk_manager(starting_capital=1000.0)

        timestamps = pd.date_range("2020-01-01 10:00", periods=3, freq="5min", tz="UTC")

        rm.record_trade(timestamps[0], timestamps[0] + pd.Timedelta(minutes=5), pnl_usd=100.0)
        rm.record_trade(timestamps[1], timestamps[1] + pd.Timedelta(minutes=5), pnl_usd=-50.0)
        rm.record_trade(timestamps[2], timestamps[2] + pd.Timedelta(minutes=5), pnl_usd=25.0)

        assert rm.daily_pnl == 75.0, "Daily PnL should sum all trades"
        assert rm.equity == 1075.0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_starting_capital(self):
        """Test behavior with zero starting capital."""
        rm = create_risk_manager(starting_capital=0.0)

        # Verify equity is zero
        assert rm.equity == 0.0, "Equity should be zero with zero starting capital"

    def test_winning_trades_after_near_liquidation(self):
        """Test that winning trades after near-liquidation work correctly."""
        # Set high daily loss limit to avoid triggering it
        rm = create_risk_manager(starting_capital=1000.0, max_daily_loss=2000.0, max_drawdown=2000.0)

        timestamps = pd.date_range("2020-01-01 10:00", periods=3, freq="5min", tz="UTC")

        # Near-liquidation loss
        rm.record_trade(timestamps[0], timestamps[0] + pd.Timedelta(minutes=5), pnl_usd=-950.0)
        assert rm.equity == 50.0

        # Winning trade brings back equity
        rm.record_trade(timestamps[1], timestamps[1] + pd.Timedelta(minutes=5), pnl_usd=500.0)
        assert rm.equity == 550.0

    def test_exact_zero_equity_halts(self):
        """Test that exactly zero equity triggers halt."""
        rm = create_risk_manager(starting_capital=1000.0)

        entry_ts = pd.Timestamp("2020-01-01 10:00", tz="UTC")
        exit_ts = pd.Timestamp("2020-01-01 10:05", tz="UTC")

        # Exact capital loss
        rm.record_trade(entry_ts, exit_ts, pnl_usd=-1000.0)

        assert rm.equity == 0.0
        assert rm.halted_today is True, "Should halt when equity reaches exactly zero"


class TestHWMUpdateWithEquityFloor:
    """Tests high-water mark updates with equity floor."""

    def test_hwm_not_updated_after_liquidation(self):
        """Test that HWM doesn't get corrupted after liquidation."""
        rm = create_risk_manager(starting_capital=1000.0, hwm_update_policy="real_time")

        timestamps = pd.date_range("2020-01-01 10:00", periods=3, freq="5min", tz="UTC")

        # Initial gain
        rm.record_trade(timestamps[0], timestamps[0] + pd.Timedelta(minutes=5), pnl_usd=500.0)
        assert rm.hwm == 1500.0

        # Catastrophic loss
        rm.record_trade(timestamps[1], timestamps[1] + pd.Timedelta(minutes=5), pnl_usd=-2000.0)

        # HWM should remain at peak (1500), equity at zero
        assert rm.hwm == 1500.0, "HWM should remain at peak"
        assert rm.equity == 0.0, "Equity should be zero"


def test_apply_broker_snapshot_from_topstep():
    rm = create_risk_manager(starting_capital=50000.0)
    ts = pd.Timestamp("2020-01-01 10:00", tz="UTC")
    rm.apply_broker_snapshot(equity=51201.09, daily_pnl=63.77, timestamp=ts)
    assert rm.equity == 51201.09
    assert rm.daily_pnl == 63.77
    assert rm.hwm >= 51201.09


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
