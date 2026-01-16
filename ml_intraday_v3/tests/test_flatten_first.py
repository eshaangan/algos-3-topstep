"""
Unit tests for direction change detection and flatten behavior.

Tests the PRIMARY FIX: flatten_all_positions() on direction changes.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parents[1]
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from live_trading.execution_engine import LiveExecutionEngine


class TestFlattenFirst(unittest.TestCase):
    """Test position direction management and flatten behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.risk_cfg = {
            'position_limits': {
                'max_concurrent_positions': 5,
                'max_total_contracts': 10,
                'max_contracts_per_position': 3,
            },
            'margin': {
                'initial_margin_per_contract': 500,
            },
        }
        self.execution_spec = {
            'instrument': {
                'tick_size_points': 0.25,
                'contract_multiplier_usd_per_point': 5.0,
            },
            'costs': {
                'commission_per_contract': 0.62,
                'slippage_ticks': {'1m': 1.0},
            },
        }
        self.label_schema = {
            'stop_multiple': 1.0,
            'target_multiple': 2.0,
        }

        # Create engine in dry_run mode
        self.engine = LiveExecutionEngine(
            risk_cfg=self.risk_cfg,
            execution_spec=self.execution_spec,
            label_schema=self.label_schema,
            dry_run=True,
        )

    def test_get_net_position_direction_long(self):
        """Test net position calculation for LONG positions."""
        # Add 2 LONG positions
        self.engine.open_positions = [
            {'direction': 'LONG', 'contracts': 1, 'order_id': '1'},
            {'direction': 'LONG', 'contracts': 1, 'order_id': '2'},
        ]
        self.assertEqual(self.engine.get_net_position_direction(), "LONG")

    def test_get_net_position_direction_short(self):
        """Test net position calculation for SHORT positions."""
        # Add 2 SHORT positions
        self.engine.open_positions = [
            {'direction': 'SHORT', 'contracts': 1, 'order_id': '1'},
            {'direction': 'SHORT', 'contracts': 1, 'order_id': '2'},
        ]
        self.assertEqual(self.engine.get_net_position_direction(), "SHORT")

    def test_get_net_position_direction_flat(self):
        """Test net position calculation for FLAT (no positions)."""
        self.engine.open_positions = []
        self.assertEqual(self.engine.get_net_position_direction(), "FLAT")

    def test_get_net_position_direction_net_long(self):
        """Test net position calculation for LONG 2 + SHORT 1 = LONG 1."""
        self.engine.open_positions = [
            {'direction': 'LONG', 'contracts': 2, 'order_id': '1'},
            {'direction': 'SHORT', 'contracts': 1, 'order_id': '2'},
        ]
        self.assertEqual(self.engine.get_net_position_direction(), "LONG")

    def test_get_net_position_direction_net_flat(self):
        """Test net position calculation for LONG 1 + SHORT 1 = FLAT."""
        self.engine.open_positions = [
            {'direction': 'LONG', 'contracts': 1, 'order_id': '1'},
            {'direction': 'SHORT', 'contracts': 1, 'order_id': '2'},
        ]
        self.assertEqual(self.engine.get_net_position_direction(), "FLAT")

    def test_flatten_all_positions_cancels_brackets(self):
        """Test that flatten_all_positions() cancels all bracket orders."""
        # Add positions with bracket orders
        self.engine.open_positions = [
            {
                'direction': 'LONG',
                'contracts': 1,
                'order_id': '1',
                'stop_order_id': 'stop_1',
                'target_order_id': 'target_1',
                'entry_price': 5000.0,
                'entry_ts': pd.Timestamp('2024-01-01 10:00:00'),
            },
            {
                'direction': 'LONG',
                'contracts': 1,
                'order_id': '2',
                'stop_order_id': 'stop_2',
                'target_order_id': 'target_2',
                'entry_price': 5005.0,
                'entry_ts': pd.Timestamp('2024-01-01 10:05:00'),
            },
        ]

        # Call flatten
        success = self.engine.flatten_all_positions(
            current_time=pd.Timestamp('2024-01-01 10:10:00'),
            current_price=5010.0,
            reason="test_flatten"
        )

        # Verify all positions cleared
        self.assertTrue(success)
        self.assertEqual(len(self.engine.open_positions), 0)

        # Verify trades recorded
        self.assertEqual(len(self.engine.trade_log), 2)
        self.assertEqual(self.engine.trade_log[0]['action'], 'EXIT')
        self.assertEqual(self.engine.trade_log[0]['reason'], 'test_flatten')

    def test_direction_change_triggers_flatten(self):
        """
        CRITICAL TEST: Verify opposite direction signal triggers flatten.

        Setup: LONG 2 contracts (2 entries, 4 bracket orders)
        Action: SHORT signal arrives
        Expected: Flatten triggered, positions closed, awaiting confirmation
        """
        # Add 2 LONG positions
        self.engine.open_positions = [
            {
                'direction': 'LONG',
                'contracts': 1,
                'order_id': '1',
                'stop_order_id': 'stop_1',
                'target_order_id': 'target_1',
                'entry_price': 5000.0,
                'entry_ts': pd.Timestamp('2024-01-01 10:00:00'),
            },
            {
                'direction': 'LONG',
                'contracts': 1,
                'order_id': '2',
                'stop_order_id': 'stop_2',
                'target_order_id': 'target_2',
                'entry_price': 5005.0,
                'entry_ts': pd.Timestamp('2024-01-01 10:05:00'),
            },
        ]

        # Create bars DataFrame
        bars = pd.DataFrame({
            'close': [5010.0],
            'high': [5015.0],
            'low': [5005.0],
        })

        # Create SHORT signal
        prediction = {'score_ev': 0.5, 'direction': 'SHORT'}

        # Execute SHORT signal (should trigger flatten)
        success, reason = self.engine.execute_signal(
            timestamp=pd.Timestamp('2024-01-01 10:10:00'),
            direction='SHORT',
            prediction=prediction,
            bars_df=bars,
            contracts=1,
        )

        # Verify flatten was triggered
        self.assertFalse(success)  # Signal rejected after flatten
        self.assertEqual(reason, 'direction_changed_awaiting_confirmation')

        # Verify all positions were flattened
        self.assertEqual(len(self.engine.open_positions), 0)

        # Verify trades recorded (2 exit trades from flatten)
        exit_trades = [t for t in self.engine.trade_log if t['action'] == 'EXIT']
        self.assertEqual(len(exit_trades), 2)

    def test_same_direction_allows_pyramiding(self):
        """Test that same direction signal does NOT trigger flatten (pyramiding)."""
        # Add 1 LONG position
        self.engine.open_positions = [
            {
                'direction': 'LONG',
                'contracts': 1,
                'order_id': '1',
                'stop_order_id': 'stop_1',
                'target_order_id': 'target_1',
                'entry_price': 5000.0,
                'entry_ts': pd.Timestamp('2024-01-01 10:00:00'),
            },
        ]

        # Create bars DataFrame
        bars = pd.DataFrame({
            'close': [5010.0] * 14,  # Need 14 bars for ATR
            'high': [5015.0] * 14,
            'low': [5005.0] * 14,
        })

        # Create LONG signal (same direction)
        prediction = {'score_ev': 0.5, 'direction': 'LONG'}

        # Execute LONG signal (should NOT flatten, should add position)
        success, reason = self.engine.execute_signal(
            timestamp=pd.Timestamp('2024-01-01 10:10:00'),
            direction='LONG',
            prediction=prediction,
            bars_df=bars,
            contracts=1,
        )

        # Verify signal succeeded (pyramiding allowed)
        self.assertTrue(success)
        self.assertEqual(reason, 'executed')

        # Verify we now have 2 LONG positions
        self.assertEqual(len(self.engine.open_positions), 2)
        self.assertEqual(self.engine.get_net_position_direction(), 'LONG')

    def test_short_to_long_triggers_flatten(self):
        """Test SHORT → LONG direction change triggers flatten."""
        # Add 1 SHORT position
        self.engine.open_positions = [
            {
                'direction': 'SHORT',
                'contracts': 1,
                'order_id': '1',
                'stop_order_id': 'stop_1',
                'target_order_id': 'target_1',
                'entry_price': 5000.0,
                'entry_ts': pd.Timestamp('2024-01-01 10:00:00'),
            },
        ]

        # Create bars DataFrame
        bars = pd.DataFrame({
            'close': [4990.0],
            'high': [4995.0],
            'low': [4985.0],
        })

        # Create LONG signal (opposite direction)
        prediction = {'score_ev': 0.5, 'direction': 'LONG'}

        # Execute LONG signal (should trigger flatten)
        success, reason = self.engine.execute_signal(
            timestamp=pd.Timestamp('2024-01-01 10:10:00'),
            direction='LONG',
            prediction=prediction,
            bars_df=bars,
            contracts=1,
        )

        # Verify flatten was triggered
        self.assertFalse(success)
        self.assertEqual(reason, 'direction_changed_awaiting_confirmation')

        # Verify all positions were flattened
        self.assertEqual(len(self.engine.open_positions), 0)

    def test_flatten_from_flat_does_nothing(self):
        """Test calling flatten when already FLAT does nothing."""
        # No positions
        self.engine.open_positions = []

        # Call flatten
        success = self.engine.flatten_all_positions(
            current_time=pd.Timestamp('2024-01-01 10:10:00'),
            current_price=5010.0,
            reason="test_flatten"
        )

        # Should succeed without errors
        self.assertTrue(success)
        self.assertEqual(len(self.engine.open_positions), 0)


if __name__ == '__main__':
    unittest.main()
