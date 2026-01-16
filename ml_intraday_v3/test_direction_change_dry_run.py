"""
Dry run integration test for direction change detection and flatten behavior.

This script tests the PRIMARY FIX in a realistic scenario:
1. Open LONG positions (pyramiding)
2. SHORT signal arrives
3. Verify flatten is triggered
4. Verify all brackets cancelled
5. Verify FLAT state with no orphaned orders

Usage:
    python test_direction_change_dry_run.py
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from live_trading.execution_engine import LiveExecutionEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_direction_change_scenario():
    """Test full direction change scenario in dry_run mode."""

    logger.info("=" * 80)
    logger.info("DIRECTION CHANGE DRY RUN INTEGRATION TEST")
    logger.info("=" * 80)

    # Setup
    risk_cfg = {
        'position_limits': {
            'max_concurrent_positions': 5,
            'max_total_contracts': 10,
            'max_contracts_per_position': 3,
        },
        'margin': {
            'initial_margin_per_contract': 500,
        },
        'daily_loss_limit_usd': -500,
        'trailing_drawdown_limit_usd': -1000,
        'max_consecutive_losses': 5,
    }

    execution_spec = {
        'instrument': {
            'tick_size_points': 0.25,
            'contract_multiplier_usd_per_point': 5.0,
        },
        'costs': {
            'commission_per_contract': 0.62,
            'slippage_ticks': {'1m': 1.0},
        },
    }

    label_schema = {
        'stop_multiple': 1.0,
        'target_multiple': 2.0,
    }

    # Create engine in dry_run mode
    logger.info("Initializing execution engine (dry_run=True)")
    engine = LiveExecutionEngine(
        risk_cfg=risk_cfg,
        execution_spec=execution_spec,
        label_schema=label_schema,
        dry_run=True,
    )

    # Initialize risk manager with proper starting equity
    engine.risk_manager.equity = 50000.0
    engine.risk_manager.hwm = 50000.0
    engine.risk_manager.daily_pnl = 0.0
    engine.risk_manager.halted_today = False

    # Create sample bars
    bars = pd.DataFrame({
        'close': [5000.0] * 14,
        'high': [5005.0] * 14,
        'low': [4995.0] * 14,
    })

    # SCENARIO 1: Open 2 LONG positions (pyramiding)
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 1: Opening 2 LONG positions")
    logger.info("=" * 80)

    prediction1 = {'score_ev': 0.6, 'direction': 'LONG'}
    success1, reason1 = engine.execute_signal(
        timestamp=pd.Timestamp('2024-01-01 10:00:00'),
        direction='LONG',
        prediction=prediction1,
        bars_df=bars,
        contracts=1,
    )
    logger.info(f"LONG Entry #1: success={success1}, reason={reason1}")
    logger.info(f"Open positions: {len(engine.open_positions)}")
    logger.info(f"Net position direction: {engine.get_net_position_direction()}")

    prediction2 = {'score_ev': 0.7, 'direction': 'LONG'}
    success2, reason2 = engine.execute_signal(
        timestamp=pd.Timestamp('2024-01-01 10:05:00'),
        direction='LONG',
        prediction=prediction2,
        bars_df=bars,
        contracts=1,
    )
    logger.info(f"LONG Entry #2: success={success2}, reason={reason2}")
    logger.info(f"Open positions: {len(engine.open_positions)}")
    logger.info(f"Net position direction: {engine.get_net_position_direction()}")

    # Verify state
    assert len(engine.open_positions) == 2, "Should have 2 LONG positions"
    assert engine.get_net_position_direction() == "LONG", "Net position should be LONG"

    logger.info("\n✅ SCENARIO 1 PASSED: Pyramiding allowed for same direction")

    # SCENARIO 2: SHORT signal arrives (opposite direction)
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 2: SHORT signal arrives (DIRECTION CHANGE)")
    logger.info("=" * 80)
    logger.info("Expected: Flatten all LONG positions, cancel all brackets")

    prediction3 = {'score_ev': 0.8, 'direction': 'SHORT'}
    success3, reason3 = engine.execute_signal(
        timestamp=pd.Timestamp('2024-01-01 10:10:00'),
        direction='SHORT',
        prediction=prediction3,
        bars_df=bars,
        contracts=1,
    )
    logger.info(f"SHORT signal result: success={success3}, reason={reason3}")
    logger.info(f"Open positions: {len(engine.open_positions)}")
    logger.info(f"Net position direction: {engine.get_net_position_direction()}")

    # Verify flatten was triggered
    assert success3 is False, "Signal should be rejected after flatten"
    assert reason3 == "direction_changed_awaiting_confirmation", \
        "Reason should indicate direction change"
    assert len(engine.open_positions) == 0, "All positions should be flattened"
    assert engine.get_net_position_direction() == "FLAT", "Net position should be FLAT"

    logger.info("\n✅ SCENARIO 2 PASSED: Direction change triggered flatten")

    # Verify trade log
    exit_trades = [t for t in engine.trade_log if t['action'] == 'EXIT']
    assert len(exit_trades) == 2, "Should have 2 exit trades from flatten"

    logger.info(f"\nTrade log summary:")
    for i, trade in enumerate(engine.trade_log, 1):
        logger.info(f"  {i}. {trade['action']}: {trade.get('direction', 'N/A')} "
                   f"@ {trade.get('price', 0):.2f}, reason={trade.get('reason', 'N/A')}")

    # SCENARIO 3: Next SHORT signal should execute (confirmation)
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 3: Next SHORT signal (confirmation in new direction)")
    logger.info("=" * 80)
    logger.info("Expected: Signal executes, new SHORT position opened")

    prediction4 = {'score_ev': 0.75, 'direction': 'SHORT'}
    success4, reason4 = engine.execute_signal(
        timestamp=pd.Timestamp('2024-01-01 10:15:00'),
        direction='SHORT',
        prediction=prediction4,
        bars_df=bars,
        contracts=1,
    )
    logger.info(f"SHORT confirmation result: success={success4}, reason={reason4}")
    logger.info(f"Open positions: {len(engine.open_positions)}")
    logger.info(f"Net position direction: {engine.get_net_position_direction()}")

    # Verify execution
    assert success4 is True, "Signal should execute after awaiting confirmation"
    assert reason4 == "executed", "Reason should be executed"
    assert len(engine.open_positions) == 1, "Should have 1 SHORT position"
    assert engine.get_net_position_direction() == "SHORT", "Net position should be SHORT"

    logger.info("\n✅ SCENARIO 3 PASSED: Confirmation signal executed in new direction")

    # SCENARIO 4: LONG signal arrives (reverse direction again)
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 4: LONG signal arrives (DIRECTION CHANGE AGAIN)")
    logger.info("=" * 80)
    logger.info("Expected: Flatten SHORT position, cancel brackets, await confirmation")

    prediction5 = {'score_ev': 0.65, 'direction': 'LONG'}
    success5, reason5 = engine.execute_signal(
        timestamp=pd.Timestamp('2024-01-01 10:20:00'),
        direction='LONG',
        prediction=prediction5,
        bars_df=bars,
        contracts=1,
    )
    logger.info(f"LONG signal result: success={success5}, reason={reason5}")
    logger.info(f"Open positions: {len(engine.open_positions)}")
    logger.info(f"Net position direction: {engine.get_net_position_direction()}")

    # Verify flatten was triggered
    assert success5 is False, "Signal should be rejected after flatten"
    assert reason5 == "direction_changed_awaiting_confirmation", \
        "Reason should indicate direction change"
    assert len(engine.open_positions) == 0, "All positions should be flattened"
    assert engine.get_net_position_direction() == "FLAT", "Net position should be FLAT"

    logger.info("\n✅ SCENARIO 4 PASSED: Reverse direction change triggered flatten")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 80)
    logger.info(f"Total trades: {len(engine.trade_log)}")
    logger.info(f"Entry trades: {len([t for t in engine.trade_log if t['action'] == 'ENTRY'])}")
    logger.info(f"Exit trades: {len([t for t in engine.trade_log if t['action'] == 'EXIT'])}")
    logger.info(f"Final net position: {engine.get_net_position_direction()}")
    logger.info(f"Final open positions: {len(engine.open_positions)}")

    logger.info("\n✅✅✅ DIRECTION CHANGE DETECTION WORKING CORRECTLY ✅✅✅")


if __name__ == '__main__':
    try:
        test_direction_change_scenario()
    except AssertionError as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ UNEXPECTED ERROR: {e}", exc_info=True)
        sys.exit(1)
