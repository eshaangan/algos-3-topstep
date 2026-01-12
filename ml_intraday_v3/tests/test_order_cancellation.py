"""
Unit tests for order cancellation logic in LiveExecutionEngine.

Tests verify that when one bracket order (stop or target) fills, the other is cancelled.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime

from ml_intraday_v3.live_trading.execution_engine import LiveExecutionEngine


@pytest.fixture
def sample_bars():
    """Create sample bar data for testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=20, freq='1min'),
        'open': [5000.0] * 20,
        'high': [5010.0] * 20,
        'low': [4990.0] * 20,
        'close': [5000.0] * 20,
        'volume': [100] * 20,
    }).set_index('timestamp')


class TestChildOrderIDStorage:
    """Test that child order IDs are properly stored."""

    @patch('ml_intraday_v3.live_trading.execution_engine.ProjectXClient')
    def test_live_mode_stores_child_order_ids(self, MockClient, monkeypatch, sample_bars):
        """Test that live orders store stop and target order IDs."""
        # Setup environment
        monkeypatch.setenv("TOPSTEPX_USERNAME", "test_user")
        monkeypatch.setenv("TOPSTEPX_PROJECTX_API_KEY", "test_key")
        monkeypatch.setenv("TOPSTEPX_ACCOUNT_ID", "12345")

        # Mock client
        mock_client = Mock()
        mock_client._account_id = 12345

        # Mock order placement responses
        mock_entry_order = Mock()
        mock_entry_order.order_id = "ENTRY_123"
        mock_stop_order = Mock()
        mock_stop_order.order_id = "STOP_456"
        mock_target_order = Mock()
        mock_target_order.order_id = "TARGET_789"

        mock_client.place_order = Mock(side_effect=[mock_entry_order, mock_stop_order, mock_target_order])
        MockClient.return_value = mock_client

        # Create engine
        engine = LiveExecutionEngine(
            risk_cfg={
                'position_limits': {'max_concurrent_positions': 5},
                'max_contracts_per_trade': 1,
                'max_daily_loss_usd': 1000,
                'max_trailing_loss_usd': 2000,
            },
            execution_spec={
                'symbol': 'MES',
                'instrument': {'symbol': 'MES', 'tick_size': 0.25, 'tick_value': 1.25},
            },
            label_schema={'target_type': 'atr', 'atr_multiplier': 3.0},
            dry_run=False,
        )

        # Execute signal
        timestamp = pd.Timestamp('2025-01-01 10:00:00')
        success, reason = engine.execute_signal(
            timestamp=timestamp,
            direction="LONG",
            prediction={'score_ev': 0.5},
            bars_df=sample_bars,
            contracts=1,
        )

        assert success is True
        assert len(engine.open_positions) == 1

        # Verify child order IDs are stored
        position = engine.open_positions[0]
        assert position['order_id'] == "ENTRY_123"
        assert position['stop_order_id'] == "STOP_456"
        assert position['target_order_id'] == "TARGET_789"

    def test_dry_run_creates_placeholder_order_ids(self, monkeypatch, sample_bars):
        """Test that dry run mode creates placeholder order IDs."""
        # Setup environment (not strictly needed for dry_run but helps avoid warnings)
        monkeypatch.setenv("TOPSTEPX_USERNAME", "test_user")
        monkeypatch.setenv("TOPSTEPX_PROJECTX_API_KEY", "test_key")
        monkeypatch.setenv("TOPSTEPX_ACCOUNT_ID", "12345")

        # Create engine in dry run mode
        engine = LiveExecutionEngine(
            risk_cfg={
                'position_limits': {'max_concurrent_positions': 5},
                'max_contracts_per_trade': 1,
                'max_daily_loss_usd': 1000,
                'max_trailing_loss_usd': 2000,
            },
            execution_spec={
                'symbol': 'MES',
                'instrument': {'symbol': 'MES', 'tick_size': 0.25, 'tick_value': 1.25},
            },
            label_schema={'target_type': 'atr', 'atr_multiplier': 3.0},
            dry_run=True,  # Dry run mode
        )

        timestamp = pd.Timestamp('2025-01-01 10:00:00')
        success, reason = engine.execute_signal(
            timestamp=timestamp,
            direction="SHORT",
            prediction={'score_ev': 0.6},
            bars_df=sample_bars,
            contracts=1,
        )

        assert success is True
        position = engine.open_positions[0]

        # Verify placeholder IDs are created
        assert position['stop_order_id'] is not None
        assert position['target_order_id'] is not None
        assert "DRY_STOP_" in position['stop_order_id']
        assert "DRY_TARGET_" in position['target_order_id']


class TestOrderCancellationOnFill:
    """Test that remaining orders are cancelled when one fills."""

    def test_stop_fill_cancels_target(self, monkeypatch):
        """Test that when stop fills, target order is cancelled."""
        # Setup environment
        monkeypatch.setenv("TOPSTEPX_USERNAME", "test_user")
        monkeypatch.setenv("TOPSTEPX_PROJECTX_API_KEY", "test_key")
        monkeypatch.setenv("TOPSTEPX_ACCOUNT_ID", "12345")

        # Create engine in dry run mode to avoid API
        engine = LiveExecutionEngine(
            risk_cfg={
                'position_limits': {'max_concurrent_positions': 5},
                'max_contracts_per_trade': 1,
                'max_daily_loss_usd': 1000,
                'max_trailing_loss_usd': 2000,
            },
            execution_spec={
                'symbol': 'MES',
                'instrument': {'symbol': 'MES', 'tick_size': 0.25, 'tick_value': 1.25},
            },
            label_schema={'target_type': 'atr', 'atr_multiplier': 3.0},
            dry_run=False,  # Set to False to test cancellation logic
        )

        # Mock the client
        mock_client = Mock()
        mock_client._account_id = 12345
        mock_client.cancel_order = Mock()
        engine.client = mock_client

        # Setup: Create a position with bracket orders
        engine.open_positions = [{
            'order_id': 'ENTRY_123',
            'stop_order_id': 'STOP_456',
            'target_order_id': 'TARGET_789',
            'entry_ts': pd.Timestamp('2025-01-01 10:00:00'),
            'direction': 'LONG',
            'contracts': 1,
            'entry_price': 5000.0,
            'stop_price': 4990.0,
            'target_price': 5030.0,
            'prediction': {'score_ev': 0.5},
        }]

        # Mock: Stop order filled (not in open orders), target still open
        mock_order = Mock()
        mock_order.order_id = 'TARGET_789'
        mock_client.search_open_orders = Mock(return_value=[mock_order])

        # Execute update_positions
        current_bar = pd.Series({
            'timestamp': pd.Timestamp('2025-01-01 10:01:00'),
            'open': 4995.0,
            'high': 5000.0,
            'low': 4985.0,
            'close': 4988.0,
        })

        closed_positions = engine.update_positions(
            current_time=pd.Timestamp('2025-01-01 10:01:00'),
            current_bar=current_bar,
        )

        # Verify: Target order was cancelled
        mock_client.cancel_order.assert_called_once_with(order_id='TARGET_789')

        # Verify: Position was closed
        assert len(closed_positions) == 1
        assert closed_positions[0]['exit_reason'] == 'stop'
        assert len(engine.open_positions) == 0

    def test_target_fill_cancels_stop(self, monkeypatch):
        """Test that when target fills, stop order is cancelled."""
        # Setup environment
        monkeypatch.setenv("TOPSTEPX_USERNAME", "test_user")
        monkeypatch.setenv("TOPSTEPX_PROJECTX_API_KEY", "test_key")
        monkeypatch.setenv("TOPSTEPX_ACCOUNT_ID", "12345")

        # Create engine
        engine = LiveExecutionEngine(
            risk_cfg={
                'position_limits': {'max_concurrent_positions': 5},
                'max_contracts_per_trade': 1,
                'max_daily_loss_usd': 1000,
                'max_trailing_loss_usd': 2000,
            },
            execution_spec={
                'symbol': 'MES',
                'instrument': {'symbol': 'MES', 'tick_size': 0.25, 'tick_value': 1.25},
            },
            label_schema={'target_type': 'atr', 'atr_multiplier': 3.0},
            dry_run=False,
        )

        # Mock the client
        mock_client = Mock()
        mock_client._account_id = 12345
        mock_client.cancel_order = Mock()
        engine.client = mock_client

        # Setup: Create a position with bracket orders
        engine.open_positions = [{
            'order_id': 'ENTRY_123',
            'stop_order_id': 'STOP_456',
            'target_order_id': 'TARGET_789',
            'entry_ts': pd.Timestamp('2025-01-01 10:00:00'),
            'direction': 'SHORT',
            'contracts': 1,
            'entry_price': 5000.0,
            'stop_price': 5030.0,
            'target_price': 4970.0,
            'prediction': {'score_ev': 0.5},
        }]

        # Mock: Target order filled (not in open orders), stop still open
        mock_order = Mock()
        mock_order.order_id = 'STOP_456'
        mock_client.search_open_orders = Mock(return_value=[mock_order])

        # Execute update_positions
        current_bar = pd.Series({
            'timestamp': pd.Timestamp('2025-01-01 10:01:00'),
            'open': 4980.0,
            'high': 4985.0,
            'low': 4965.0,
            'close': 4968.0,
        })

        closed_positions = engine.update_positions(
            current_time=pd.Timestamp('2025-01-01 10:01:00'),
            current_bar=current_bar,
        )

        # Verify: Stop order was cancelled
        mock_client.cancel_order.assert_called_once_with(order_id='STOP_456')

        # Verify: Position was closed
        assert len(closed_positions) == 1
        assert closed_positions[0]['exit_reason'] == 'target'
        assert len(engine.open_positions) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
