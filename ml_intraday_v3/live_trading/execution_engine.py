"""
Live execution engine for ml_intraday_v3 strategy.

Executes trades via Topstep ProjectX API with full risk enforcement.
"""

import importlib.util
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add parent directory to path for legacy imports
parent_dir = Path(__file__).resolve().parents[2]
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Add paths for imports
ml_v3_dir = Path(__file__).resolve().parents[1]  # ml_intraday_v3/
project_root = ml_v3_dir.parent  # algos 3 topstep/

if str(ml_v3_dir) not in sys.path:
    sys.path.insert(0, str(ml_v3_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from core.projectx_client import ProjectXClient, OrderState
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    core_path = project_root / "core" / "projectx_client.py"
    spec = importlib.util.spec_from_file_location("projectx_client", core_path)
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    sys.modules["projectx_client"] = module
    spec.loader.exec_module(module)
    ProjectXClient = module.ProjectXClient
    OrderState = module.OrderState
from backtesting_v3.risk import RiskManager

logger = logging.getLogger(__name__)


class LiveExecutionEngine:
    """
    Executes trades on Topstep account via ProjectX API.

    Enforces all risk rules from risk.yaml and maintains position tracking.
    """

    def __init__(
        self,
        risk_cfg: dict,
        execution_spec: dict,
        label_schema: dict,
        dry_run: bool = False,
        contract_id: str | None = None,
        account_id: str | None = None,
        order_type: str = "MARKET",
    ):
        """
        Initialize execution engine.

        Args:
            risk_cfg: Risk configuration (from risk.yaml)
            execution_spec: Execution specification (from execution_spec.yaml)
            label_schema: Label schema (for cost calculations)
            dry_run: If True, simulate trades without executing
        """
        self.risk_cfg = risk_cfg
        self.execution_spec = execution_spec
        self.label_schema = label_schema
        self.dry_run = dry_run
        self.contract_id = contract_id
        self.account_id = account_id
        self.order_type = order_type.upper()

        # Initialize risk manager
        self.risk_manager = RiskManager(risk_cfg)

        # Get API credentials from environment
        base_url = os.getenv("TOPSTEPX_PROJECTX_BASE_URL", "https://api.topstepx.com")
        username = os.getenv("TOPSTEPX_USERNAME")
        api_key = os.getenv("TOPSTEPX_PROJECTX_API_KEY")
        env_account_id = os.getenv("TOPSTEPX_ACCOUNT_ID")
        env_contract_id = os.getenv("TOPSTEPX_CONTRACT_ID", "CON.F.US.EP.Z25")

        # Resolve with explicit overrides first, then fall back to environment.
        resolved_account_id = account_id or env_account_id
        resolved_contract_id = contract_id or env_contract_id

        # Skip credential validation in dry run mode (replay/testing)
        if not dry_run and not all([username, api_key, resolved_account_id]):
            raise ValueError("Missing Topstep credentials in .env")

        self.account_id = str(resolved_account_id) if resolved_account_id else "MOCK"
        self.contract_id = resolved_contract_id or "MOCK"

        # Initialize ProjectX client (only if not dry run)
        if not dry_run:
            logger.info(f"Connecting to Topstep: account={self.account_id}, contract={self.contract_id}")
            # ProjectXClient reads credentials from environment; allow override
            self.client = ProjectXClient(contract_id=self.contract_id, account_id=self.account_id)
        else:
            logger.info("DRY RUN MODE - No trades will be executed")
            self.client = None

        # Track open positions locally
        self.open_positions: List[Dict] = []

        # Trade log
        self.trade_log: List[Dict] = []

        logger.info(f"LiveExecutionEngine initialized (dry_run={dry_run})")

    def execute_signal(
        self,
        timestamp: pd.Timestamp,
        direction: str,  # "LONG" or "SHORT"
        prediction: Dict[str, float],
        bars_df: pd.DataFrame,
        contracts: int = 1,
        kelly_sizer: Optional['KellySizer'] = None,  # NEW: Kelly sizing support
        trade_history: Optional[List[Dict]] = None,  # NEW: For Kelly calculation
    ) -> Tuple[bool, str]:
        """
        Execute a trading signal.

        Args:
            timestamp: Signal timestamp
            direction: Trade direction ("LONG" or "SHORT")
            prediction: Model prediction dictionary
            bars_df: Recent bars (for setting stop/target)
            contracts: Number of contracts to trade (fallback if Kelly disabled)
            kelly_sizer: Optional KellySizer instance for dynamic sizing
            trade_history: Optional trade history for Kelly calculation

        Returns:
            (success, reason) tuple
        """
        # Check risk gates
        can_trade, reason = self.risk_manager.can_trade(timestamp)
        if not can_trade:
            logger.warning(f"Trade rejected by risk manager: {reason}")
            return False, f"risk_{reason}"

        # Override contracts with Kelly sizing if enabled
        if kelly_sizer is not None and kelly_sizer.config.get('enabled', False):
            try:
                score_ev = prediction.get('score_ev', 0.0)
                contracts, sizing_reason = kelly_sizer.get_position_size(
                    trade_history=trade_history or [],
                    score_ev=score_ev,
                    max_contracts_limit=self.risk_cfg['position_limits']['max_contracts_per_position'],
                    current_equity=self.get_equity(),
                    contract_margin=self.risk_cfg['margin']['initial_margin_per_contract'],
                )
                logger.info(f"Kelly sizing: {contracts} contracts (reason: {sizing_reason})")
            except Exception as e:
                logger.error(f"Kelly sizing error: {e}, falling back to default contracts")
                # Keep the original contracts value on error

        # Check position limits
        max_concurrent = self.risk_cfg.get('position_limits', {}).get('max_concurrent_positions', 5)
        if len(self.open_positions) >= max_concurrent:
            logger.warning(f"Max concurrent positions reached: {len(self.open_positions)}")
            return False, "max_concurrent_positions"

        # Get current price from latest bar
        if bars_df.empty:
            logger.error("No bars available for execution")
            return False, "no_bars"

        current_bar = bars_df.iloc[-1]
        entry_price = current_bar['close']  # Use close as estimated entry

        # Calculate stop and target prices
        stop_price, target_price = self._calculate_stops_and_targets(
            entry_price=entry_price,
            direction=direction,
            bars_df=bars_df,
        )

        # Log trade intent
        logger.info(
            f"Signal: {direction} {contracts} @ {entry_price:.2f}, "
            f"stop={stop_price:.2f}, target={target_price:.2f}, "
            f"score={prediction.get('score_ev', 0.0):.3f}"
        )

        # Execute trade (or simulate if dry_run)
        stop_order_id = None
        target_order_id = None

        if self.dry_run:
            # Simulate execution
            order_id = f"DRY_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            stop_order_id = f"DRY_STOP_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            target_order_id = f"DRY_TARGET_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"DRY RUN: Simulated order {order_id}, stop={stop_order_id}, target={target_order_id}")
            success = True
        else:
            # Execute via API
            try:
                side = "BUY" if direction == "LONG" else "SELL"
                # Entry order
                order = self.client.place_order(
                    symbol="MES",
                    side=side,
                    quantity=contracts,
                    order_type=self.order_type,
                    contract_id=self.contract_id,
                    stop_loss=stop_price,
                    take_profit=target_price,
                )
                order_id = order.order_id
                logger.info(f"Order placed: {order_id}")
                success = True

                # Submit OCO children (stop + target) linked to entry
                if not self.dry_run:
                    oco_side = "SELL" if direction == "LONG" else "BUY"
                    # Stop child
                    stop_order = self.client.place_order(
                        symbol="MES",
                        side=oco_side,
                        quantity=contracts,
                        order_type="STOP",
                        contract_id=self.contract_id,
                        stop_loss=stop_price,
                        take_profit=None,
                        linked_order_id=int(order_id),
                    )
                    stop_order_id = stop_order.order_id
                    logger.info(f"OCO stop placed: {stop_order_id}")
                    # Target child
                    target_order = self.client.place_order(
                        symbol="MES",
                        side=oco_side,
                        quantity=contracts,
                        order_type="LIMIT",
                        contract_id=self.contract_id,
                        stop_loss=None,
                        take_profit=target_price,
                        linked_order_id=int(order_id),
                    )
                    target_order_id = target_order.order_id
                    logger.info(f"OCO target placed: {target_order_id}")
            except Exception as e:
                logger.error(f"Order execution failed: {e}")
                return False, f"execution_error: {str(e)}"

        # Record position
        position = {
            'order_id': order_id,
            'stop_order_id': stop_order_id,
            'target_order_id': target_order_id,
            'entry_ts': timestamp,
            'direction': direction,
            'contracts': contracts,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'target_price': target_price,
            'prediction': prediction.copy(),
        }
        self.open_positions.append(position)

        # Record in trade log
        self.trade_log.append({
            'timestamp': timestamp,
            'action': 'ENTRY',
            'direction': direction,
            'contracts': contracts,
            'price': entry_price,
            'order_id': order_id,
            'prediction_score': prediction.get('score_ev', 0.0),
        })

        logger.info(f"Position opened: {order_id}, open_positions={len(self.open_positions)}")

        return True, "executed"

    def update_positions(
        self,
        current_time: pd.Timestamp,
        current_bar: pd.Series,
    ) -> List[Dict]:
        """
        Update open positions and check for exits.

        Args:
            current_time: Current timestamp
            current_bar: Current bar data

        Returns:
            List of closed positions
        """
        if not self.open_positions:
            return []

        current_price = current_bar['close']
        high = current_bar['high']
        low = current_bar['low']

        # Query broker for current open orders
        open_order_ids = set()
        if not self.dry_run:
            try:
                open_orders = self.client.search_open_orders()
                open_order_ids = {str(order.order_id) for order in open_orders}
                logger.debug(f"Found {len(open_order_ids)} open orders at broker")
            except Exception as e:
                logger.error(f"Failed to query open orders: {e}")
                # Continue with empty set - will rely on price-based detection

        closed_positions = []

        for position in self.open_positions[:]:  # Iterate over copy
            # Skip if order IDs not tracked (backward compatibility)
            if 'stop_order_id' not in position or 'target_order_id' not in position:
                logger.warning(f"Position {position['order_id']} missing bracket order IDs, skipping")
                continue

            # Check if bracket orders are still open at broker
            stop_still_open = str(position['stop_order_id']) in open_order_ids
            target_still_open = str(position['target_order_id']) in open_order_ids

            # Determine if position exited and which order filled
            exit_price = None
            exit_reason = None
            order_to_cancel = None

            if not stop_still_open and not target_still_open:
                # Both orders closed - position exited but we missed it
                logger.warning(f"Both bracket orders closed for position {position['order_id']}")
                # Use price-based heuristic to determine which filled first
                if position['direction'] == "LONG":
                    if current_price <= position['stop_price']:
                        exit_price = position['stop_price']
                        exit_reason = "stop"
                    else:
                        exit_price = position['target_price']
                        exit_reason = "target"
                else:  # SHORT
                    if current_price >= position['stop_price']:
                        exit_price = position['stop_price']
                        exit_reason = "stop"
                    else:
                        exit_price = position['target_price']
                        exit_reason = "target"

            elif not stop_still_open:
                # Stop order filled, cancel target
                exit_price = position['stop_price']
                exit_reason = "stop"
                order_to_cancel = position['target_order_id']
                logger.info(f"Stop order filled for {position['order_id']}, will cancel target {order_to_cancel}")

            elif not target_still_open:
                # Target order filled, cancel stop
                exit_price = position['target_price']
                exit_reason = "target"
                order_to_cancel = position['stop_order_id']
                logger.info(f"Target order filled for {position['order_id']}, will cancel stop {order_to_cancel}")

            # If position exited, cancel remaining order and close position
            if exit_price is not None:
                # Cancel the remaining bracket order
                if order_to_cancel and not self.dry_run:
                    try:
                        self.client.cancel_order(order_id=order_to_cancel)
                        logger.info(f"Successfully cancelled remaining bracket order {order_to_cancel}")
                    except Exception as e:
                        logger.error(f"Failed to cancel order {order_to_cancel}: {e}")
                        # Continue anyway - position is closed, we'll log the orphaned order

                # Calculate PnL
                pnl = self._calculate_pnl(
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    direction=position['direction'],
                    contracts=position['contracts'],
                )

                logger.info(
                    f"Position closed: {position['order_id']}, "
                    f"reason={exit_reason}, pnl=${pnl:.2f}"
                )

                # Record trade
                self.risk_manager.record_trade(
                    entry_ts=position['entry_ts'],
                    exit_ts=current_time,
                    pnl_usd=pnl,
                )

                # Record in trade log
                self.trade_log.append({
                    'timestamp': current_time,
                    'action': 'EXIT',
                    'reason': exit_reason,
                    'contracts': position['contracts'],
                    'price': exit_price,
                    'pnl_usd': pnl,
                    'order_id': position['order_id'],
                    'stop_order_id': position['stop_order_id'],
                    'target_order_id': position['target_order_id'],
                    'cancelled_order_id': order_to_cancel,
                })

                # Remove from open positions
                self.open_positions.remove(position)
                closed_positions.append({
                    **position,
                    'exit_ts': current_time,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl_usd': pnl,
                })

        return closed_positions

    def flatten_all_positions(
        self,
        current_time: pd.Timestamp,
        current_price: float,
        reason: str = "flatten",
    ):
        """
        Flatten all open positions.

        Args:
            current_time: Current timestamp
            current_price: Current price
            reason: Reason for flattening
        """
        if not self.open_positions:
            return

        logger.warning(f"Flattening all {len(self.open_positions)} positions: {reason}")

        for position in self.open_positions[:]:
            pnl = self._calculate_pnl(
                entry_price=position['entry_price'],
                exit_price=current_price,
                direction=position['direction'],
                contracts=position['contracts'],
            )

            logger.info(f"Closed: {position['order_id']}, pnl=${pnl:.2f}")

            # Record trade
            self.risk_manager.record_trade(
                entry_ts=position['entry_ts'],
                exit_ts=current_time,
                pnl_usd=pnl,
            )

            # Record in trade log
            self.trade_log.append({
                'timestamp': current_time,
                'action': 'EXIT',
                'reason': reason,
                'contracts': position['contracts'],
                'price': current_price,
                'pnl_usd': pnl,
                'order_id': position['order_id'],
            })

        self.open_positions.clear()

    def _calculate_stops_and_targets(
        self,
        entry_price: float,
        direction: str,
        bars_df: pd.DataFrame,
    ) -> Tuple[float, float]:
        """
        Calculate stop and target prices.

        Uses label schema configuration for stop/target multiples.

        Args:
            entry_price: Entry price
            direction: Trade direction
            bars_df: Recent bars for ATR calculation

        Returns:
            (stop_price, target_price) tuple
        """
        # Calculate ATR for stop/target sizing
        if len(bars_df) >= 14:
            high = bars_df['high'].values
            low = bars_df['low'].values
            close = bars_df['close'].values

            tr = np.maximum(high - low,
                           np.maximum(abs(high - np.roll(close, 1)),
                                    abs(low - np.roll(close, 1))))
            tr[0] = high[0] - low[0]
            atr = np.mean(tr[-14:])
        else:
            # Fallback: use fixed points
            atr = 5.0  # 5 points for MES

        # Get stop/target multiples from label schema
        stop_multiple = self.label_schema.get('stop_multiple', 1.0)
        target_multiple = self.label_schema.get('target_multiple', 2.0)

        # Calculate prices
        if direction == "LONG":
            stop_price = entry_price - (stop_multiple * atr)
            target_price = entry_price + (target_multiple * atr)
        else:  # SHORT
            stop_price = entry_price + (stop_multiple * atr)
            target_price = entry_price - (target_multiple * atr)

        # Round to tick size
        tick_size = self.execution_spec['instrument']['tick_size_points']
        stop_price = round(stop_price / tick_size) * tick_size
        target_price = round(target_price / tick_size) * tick_size

        return stop_price, target_price

    def _calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        direction: str,
        contracts: int,
    ) -> float:
        """
        Calculate P&L in USD.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            direction: Trade direction
            contracts: Number of contracts

        Returns:
            P&L in USD
        """
        # Get instrument multiplier
        multiplier = self.execution_spec['instrument']['contract_multiplier_usd_per_point']

        # Calculate point difference
        if direction == "LONG":
            points = exit_price - entry_price
        else:  # SHORT
            points = entry_price - exit_price

        # Calculate gross P&L
        gross_pnl = points * multiplier * contracts

        # Subtract costs
        commission = self.execution_spec['costs']['commission_per_contract']
        slippage_ticks = self.execution_spec['costs']['slippage_ticks'].get('1m', 1.0)
        tick_size = self.execution_spec['instrument']['tick_size_points']
        slippage_usd = slippage_ticks * tick_size * multiplier * contracts

        # Total costs: commission + slippage (round-trip)
        total_costs = (commission * 2 * contracts) + (slippage_usd * 2)

        net_pnl = gross_pnl - total_costs

        return net_pnl

    def get_equity(self) -> float:
        """Get current equity from risk manager."""
        return self.risk_manager.equity

    def get_daily_pnl(self) -> float:
        """Get current daily P&L from risk manager."""
        return self.risk_manager.daily_pnl

    def get_drawdown(self) -> float:
        """Get current drawdown from HWM."""
        return self.risk_manager.hwm - self.risk_manager.equity

    def get_status(self) -> Dict:
        """
        Get current execution engine status.

        Returns:
            Dictionary with status information
        """
        broker_open_positions = len(self.open_positions)
        broker_daily_pnl = self.get_daily_pnl()
        broker_equity = self.get_equity()

        if not self.dry_run and self.client is not None:
            try:
                acct = self.client.get_account_state()
                broker_equity = acct.equity
                broker_daily_pnl = acct.daily_pnl
                broker_open_positions = acct.open_positions

                # Keep risk manager in sync with broker equity to reflect real drawdown.
                self.risk_manager.equity = broker_equity
                self.risk_manager.hwm = max(self.risk_manager.hwm, broker_equity)
                self.risk_manager.daily_pnl = broker_daily_pnl
            except Exception as exc:
                logger.warning("Failed to refresh broker account state; using internal metrics: %s", exc)

        return {
            'open_positions': broker_open_positions,
            'equity': broker_equity,
            'daily_pnl': broker_daily_pnl,
            'drawdown': self.get_drawdown(),
            'trades_today': self.risk_manager.trades_today,
            'consecutive_losses': self.risk_manager.consecutive_losses,
            'halted': self.risk_manager.halted_today,
        }

    def check_api_connection(self) -> bool:
        """
        Test connection to Topstep API.

        Returns:
            True if connection is healthy, False otherwise
        """
        if self.dry_run:
            logger.info("DRY RUN: API check skipped")
            return True

        try:
            # Try to fetch account info
            account = self.client.get_account_state()
            logger.info(f"API connection healthy: account={account.account_id}, equity=${account.equity:,.2f}")
            return True
        except Exception as e:
            logger.error(f"API connection failed: {e}")
            return False


# Import numpy for calculations
import numpy as np
