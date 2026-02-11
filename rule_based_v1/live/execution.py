"""Order execution bridge to TopstepX infrastructure.

Reuses ml_intraday_v3/live_trading/execution_engine.py for actual order execution.
This module provides a simplified interface for rule-based signals.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def create_execution_engine(
    risk_cfg: dict,
    execution_spec: dict,
    dry_run: bool = True,
    contract_id: str | None = None,
    account_id: str | None = None,
):
    """Create an execution engine from ml_intraday_v3.

    Returns:
        LiveExecutionEngine instance, or None if import fails.
    """
    try:
        ml_v3_path = Path(__file__).parent.parent.parent / "ml_intraday_v3"
        sys.path.insert(0, str(ml_v3_path))
        from live_trading.execution_engine import LiveExecutionEngine

        # Build label_schema-like config for stop/target calc
        label_schema = {
            "vertical_barrier_hours": 2,
            "profit_taking_stop_multiple": execution_spec.get("exit_strategy", {}).get("profit_target_atr", 2.0),
            "stop_loss_stop_multiple": execution_spec.get("exit_strategy", {}).get("stop_loss_atr", 1.5),
        }

        engine = LiveExecutionEngine(
            risk_cfg=risk_cfg,
            execution_spec=execution_spec,
            label_schema=label_schema,
            dry_run=dry_run,
            contract_id=contract_id,
            account_id=account_id,
        )
        logger.info("Execution engine created successfully")
        return engine

    except ImportError as e:
        logger.warning(f"Could not import execution engine: {e}")
        logger.info("Running in standalone mode (no order execution)")
        return None
    except Exception as e:
        logger.error(f"Error creating execution engine: {e}", exc_info=True)
        return None
