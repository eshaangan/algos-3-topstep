"""
Critical tests for high-confidence threshold direction change logic.

These tests validate the NEW behavior where weak opposing signals are rejected
and only strong opposing signals trigger position flattening.

MISSING TESTS IDENTIFIED BY VALIDATION REPORT - MUST IMPLEMENT
"""

import unittest
from unittest.mock import Mock, MagicMock
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).resolve().parents[1]
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from live_trading.execution_engine import LiveExecutionEngine


class TestHighConfidenceThreshold(unittest.TestCase):
    """Test high-confidence threshold filtering for opposing signals."""

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

    def _create_engine_with_threshold(self, enabled=True, threshold=0.20):
        """Helper to create engine with specific threshold config."""
        config = {
            'direction_change': {
                'enabled': enabled,
                'high_confidence_threshold': threshold,
            }
        }
        return LiveExecutionEngine(
            risk_cfg=self.risk_cfg,
            execution_spec=self.execution_spec,
            label_schema=self.label_schema,
            dry_run=True,
            config=config,
        )

    def test_weak_opposing_signal_rejected(self):
        """
        CRITICAL: Weak opposing signal (score_ev < 0.20) should be REJECTED.

        Setup: LONG position
        Signal: SHORT with score_ev = 0.16 (< 0.20)
        Expected: Signal rejected, LONG position kept
        Reason: "opposing_signal_too_weak"
        """
        engine = self._create_engine_with_threshold(enabled=True, threshold=0.20)

        # Add LONG position
        engine.open_positions = [
            {
                'direction': 'LONG',
                'contracts': 1,
                'order_id': '1',
                'entry_price': 5000.0,
                'entry_ts': pd.Timestamp('2024-01-01 10:00:00'),
            },
        ]

        # Create bars DataFrame
        bars = pd.DataFrame({
            'close': [5010.0],
            'high': [5015.0],
            'low': [5005.0],
        })

        # Create WEAK SHORT signal (score_ev = 0.16 < 0.20)
        prediction = {'score_ev': -0.16}  # Negative for SHORT

        # Execute SHORT signal
        success, reason = engine.execute_signal(
            timestamp=pd.Timestamp('2024-01-01 10:10:00'),
            direction='SHORT',
            prediction=prediction,
            bars_df=bars,
            contracts=1,
        )

        # Verify signal was REJECTED
        self.assertFalse(success)
        self.assertEqual(reason, 'opposing_signal_too_weak')

        # Verify LONG position is STILL OPEN
        self.assertEqual(len(engine.open_positions), 1)
        self.assertEqual(engine.open_positions[0]['direction'], 'LONG')

    def test_strong_opposing_signal_triggers_flatten(self):
        """
        CRITICAL: Strong opposing signal (score_ev >= 0.20) should FLATTEN.

        Setup: LONG position
        Signal: SHORT with score_ev = 0.25 (>= 0.20)
        Expected: LONG flattened, signal rejected pending confirmation
        Reason: "direction_changed_awaiting_confirmation"
        """
        engine = self._create_engine_with_threshold(enabled=True, threshold=0.20)

        # Add LONG position
        engine.open_positions = [
            {
                'direction': 'LONG',
                'contracts': 1,
                'order_id': '1',
                'entry_price': 5000.0,
                'entry_ts': pd.Timestamp('2024-01-01 10:00:00'),
            },
        ]

        # Create bars DataFrame
        bars = pd.DataFrame({
            'close': [5010.0],
            'high': [5015.0],
            'low': [5005.0],
        })

        # Create STRONG SHORT signal (score_ev = 0.25 >= 0.20)
        prediction = {'score_ev': -0.25}  # Negative for SHORT

        # Execute SHORT signal
        success, reason = engine.execute_signal(
            timestamp=pd.Timestamp('2024-01-01 10:10:00'),
            direction='SHORT',
            prediction=prediction,
            bars_df=bars,
            contracts=1,
        )

        # Verify flatten was triggered
        self.assertFalse(success)
        self.assertEqual(reason, 'direction_changed_awaiting_confirmation')

        # Verify all positions were FLATTENED
        self.assertEqual(len(engine.open_positions), 0)

        # Verify trades recorded (1 exit from flatten)
        exit_trades = [t for t in engine.trade_log if t['action'] == 'EXIT']
        self.assertEqual(len(exit_trades), 1)
        self.assertIn('direction_change', exit_trades[0]['reason'])

    def test_threshold_boundary_exact(self):
        """
        Signal at exact threshold (score_ev = 0.20) should trigger flatten.
        """
        engine = self._create_engine_with_threshold(enabled=True, threshold=0.20)

        # Add SHORT position
        engine.open_positions = [
            {
                'direction': 'SHORT',
                'contracts': 1,
                'order_id': '1',
                'entry_price': 5000.0,
                'entry_ts': pd.Timestamp('2024-01-01 10:00:00'),
            },
        ]

        # Create bars
        bars = pd.DataFrame({
            'close': [4990.0],
            'high': [4995.0],
            'low': [4985.0],
        })

        # Create LONG signal with EXACT threshold
        prediction = {'score_ev': 0.20}  # Exactly at threshold

        # Execute LONG signal
        success, reason = engine.execute_signal(
            timestamp=pd.Timestamp('2024-01-01 10:10:00'),
            direction='LONG',
            prediction=prediction,
            bars_df=bars,
            contracts=1,
        )

        # Verify flatten triggered (>= threshold)
        self.assertFalse(success)
        self.assertEqual(reason, 'direction_changed_awaiting_confirmation')
        self.assertEqual(len(engine.open_positions), 0)

    def test_threshold_boundary_just_below(self):
        """
        Signal just below threshold (score_ev = 0.199) should be rejected.
        """
        engine = self._create_engine_with_threshold(enabled=True, threshold=0.20)

        # Add LONG position
        engine.open_positions = [
            {
                'direction': 'LONG',
                'contracts': 1,
                'order_id': '1',
                'entry_price': 5000.0,
                'entry_ts': pd.Timestamp('2024-01-01 10:00:00'),
            },
        ]

        # Create bars
        bars = pd.DataFrame({
            'close': [5010.0],
            'high': [5015.0],
            'low': [5005.0],
        })

        # Create SHORT signal JUST BELOW threshold
        prediction = {'score_ev': -0.199}  # 0.001 below threshold

        # Execute SHORT signal
        success, reason = engine.execute_signal(
            timestamp=pd.Timestamp('2024-01-01 10:10:00'),
            direction='SHORT',
            prediction=prediction,
            bars_df=bars,
            contracts=1,
        )

        # Verify signal REJECTED
        self.assertFalse(success)
        self.assertEqual(reason, 'opposing_signal_too_weak')

        # Verify position KEPT
        self.assertEqual(len(engine.open_positions), 1)


class TestDirectionChangeConfiguration(unittest.TestCase):
    """Test configuration loading and defaults."""

    def setUp(self):
        """Set up test fixtures."""
        self.risk_cfg = {
            'position_limits': {
                'max_concurrent_positions': 5,
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

    def test_direction_change_config_enabled(self):
        """Test direction change enabled via config."""
        config = {
            'direction_change': {
                'enabled': True,
                'high_confidence_threshold': 0.20,
            }
        }
        engine = LiveExecutionEngine(
            risk_cfg=self.risk_cfg,
            execution_spec=self.execution_spec,
            label_schema=self.label_schema,
            dry_run=True,
            config=config,
        )

        self.assertTrue(engine.direction_change_enabled)
        self.assertEqual(engine.direction_change_threshold, 0.20)

    def test_direction_change_config_disabled(self):
        """Test direction change disabled via config."""
        config = {
            'direction_change': {
                'enabled': False,
                'high_confidence_threshold': 0.20,
            }
        }
        engine = LiveExecutionEngine(
            risk_cfg=self.risk_cfg,
            execution_spec=self.execution_spec,
            label_schema=self.label_schema,
            dry_run=True,
            config=config,
        )

        self.assertFalse(engine.direction_change_enabled)

    def test_direction_change_threshold_configurable(self):
        """Test custom threshold from config."""
        config = {
            'direction_change': {
                'enabled': True,
                'high_confidence_threshold': 0.15,  # Custom value
            }
        }
        engine = LiveExecutionEngine(
            risk_cfg=self.risk_cfg,
            execution_spec=self.execution_spec,
            label_schema=self.label_schema,
            dry_run=True,
            config=config,
        )

        self.assertEqual(engine.direction_change_threshold, 0.15)

    def test_direction_change_config_defaults(self):
        """Test default values when config missing."""
        # No config provided
        engine = LiveExecutionEngine(
            risk_cfg=self.risk_cfg,
            execution_spec=self.execution_spec,
            label_schema=self.label_schema,
            dry_run=True,
        )

        # Should use defaults
        self.assertTrue(engine.direction_change_enabled)  # Default: enabled
        self.assertEqual(engine.direction_change_threshold, 0.20)  # Default threshold


class TestDirectionChangeEdgeCases(unittest.TestCase):
    """Test edge cases for direction change logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.risk_cfg = {
            'position_limits': {
                'max_concurrent_positions': 5,
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

        config = {
            'direction_change': {
                'enabled': True,
                'high_confidence_threshold': 0.20,
            }
        }
        self.engine = LiveExecutionEngine(
            risk_cfg=self.risk_cfg,
            execution_spec=self.execution_spec,
            label_schema=self.label_schema,
            dry_run=True,
            config=config,
        )

    def test_score_ev_zero(self):
        """Test signal with score_ev = 0 (neutral)."""
        # Add LONG position
        self.engine.open_positions = [
            {
                'direction': 'LONG',
                'contracts': 1,
                'order_id': '1',
                'entry_price': 5000.0,
                'entry_ts': pd.Timestamp('2024-01-01 10:00:00'),
            },
        ]

        # Create bars
        bars = pd.DataFrame({
            'close': [5000.0],
            'high': [5005.0],
            'low': [4995.0],
        })

        # Create neutral signal
        prediction = {'score_ev': 0.0}

        # Execute SHORT signal (direction determined by negative score_ev)
        success, reason = self.engine.execute_signal(
            timestamp=pd.Timestamp('2024-01-01 10:10:00'),
            direction='SHORT',
            prediction=prediction,
            bars_df=bars,
            contracts=1,
        )

        # Verify rejected (0.0 < 0.20)
        self.assertFalse(success)
        self.assertEqual(reason, 'opposing_signal_too_weak')

    def test_score_ev_uses_absolute_value(self):
        """
        Test that threshold check uses absolute value of score_ev.

        For SHORT signals, score_ev is typically negative.
        abs(score_ev) should be compared to threshold.
        """
        # Add LONG position
        self.engine.open_positions = [
            {
                'direction': 'LONG',
                'contracts': 1,
                'order_id': '1',
                'entry_price': 5000.0,
                'entry_ts': pd.Timestamp('2024-01-01 10:00:00'),
            },
        ]

        # Create bars
        bars = pd.DataFrame({
            'close': [5010.0],
            'high': [5015.0],
            'low': [5005.0],
        })

        # Create SHORT signal with negative score_ev
        # abs(-0.25) = 0.25 >= 0.20, should flatten
        prediction = {'score_ev': -0.25}

        # Execute SHORT signal
        success, reason = self.engine.execute_signal(
            timestamp=pd.Timestamp('2024-01-01 10:10:00'),
            direction='SHORT',
            prediction=prediction,
            bars_df=bars,
            contracts=1,
        )

        # Verify flatten triggered (abs value check)
        self.assertFalse(success)
        self.assertEqual(reason, 'direction_changed_awaiting_confirmation')
        self.assertEqual(len(self.engine.open_positions), 0)


if __name__ == '__main__':
    unittest.main()
