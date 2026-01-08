"""
Live execution engine for ml_intraday_v3 strategy.

Executes trades via Topstep ProjectX API with full risk enforcement.
"""

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

# Add ml_intraday_v3 to path for backtesting_v3 imports
ml_v3_dir = Path(__file__).resolve().parents[1]
if str(ml_v3_dir) not in sys.path:
    sys.path.insert(0, str(ml_v3_dir))

from ml_intraday_v3.core.projectx_client import ProjectXClient, OrderState
from ml_intraday_v3.backtesting_v3.risk import RiskManager

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

        # Initialize risk manager
        self.risk_manager = RiskManager(risk_cfg)

        # Get API credentials from environment
        base_url = os.getenv("TOPSTEPX_PROJECTX_BASE_URL", "https://api.topstepx.com")
        username = os.getenv("TOPSTEPX_USERNAME")
        api_key = os.getenv("TOPSTEPX_PROJECTX_API_KEY")
        account_id = os.getenv("TOPSTEPX_ACCOUNT_ID")
        contract_id = os.getenv("TOPSTEPX_CONTRACT_ID", "CON.F.US.EP.Z25")

        if not all([username, api_key, account_id]):
            raise ValueError("Missing Topstep credentials in .env")

        self.account_id = account_id
        self.contract_id = contract_id

        # Initialize ProjectX client (only if not dry run)
        if not dry_run:
            logger.info(f"Connecting to Topstep: account={account_id}, contract={contract_id}")
            # ProjectXClient reads credentials from environment, no args needed
            self.client = ProjectXClient()
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
    ) -> Tuple[bool, str]:
        """
        Execute a trading signal.

        Args:
            timestamp: Signal timestamp
            direction: Trade direction ("LONG" or "SHORT")
            prediction: Model prediction dictionary
            bars_df: Recent bars (for setting stop/target)
            contracts: Number of contracts to trade

        Returns:
            (success, reason) tuple
        """
        # Check risk gates
        can_trade, reason = self.risk_manager.can_trade(timestamp)
        if not can_trade:
            logger.warning(f"Trade rejected by risk manager: {reason}")
            return False, f"risk_{reason}"

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
        if self.dry_run:
            # Simulate execution
            order_id = f"DRY_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"DRY RUN: Simulated order {order_id}")
            success = True
        else:
            # Execute via API
            try:
                side = "BUY" if direction == "LONG" else "SELL"
                order = self.client.place_order(
                    symbol="MES",
                    side=side,
                    quantity=contracts,
                    order_type="MARKET",
                    contract_id=self.contract_id,
                )
                order_id = order.order_id
                logger.info(f"Order placed: {order_id}")
                success = True
            except Exception as e:
                logger.error(f"Order execution failed: {e}")
                return False, f"execution_error: {str(e)}"

        # Record position
        position = {
            'order_id': order_id,
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

        closed_positions = []

        for position in self.open_positions[:]:  # Iterate over copy
            # Check for stop/target hit
            hit_stop = False
            hit_target = False

            if position['direction'] == "LONG":
                hit_stop = low <= position['stop_price']
                hit_target = high >= position['target_price']
            else:  # SHORT
                hit_stop = high >= position['stop_price']
                hit_target = low <= position['target_price']

            # Determine exit
            exit_price = None
            exit_reason = None

            if hit_stop:
                exit_price = position['stop_price']
                exit_reason = "stop"
            elif hit_target:
                exit_price = position['target_price']
                exit_reason = "target"

            # Close position if exit triggered
            if exit_price is not None:
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
        return {
            'open_positions': len(self.open_positions),
            'equity': self.get_equity(),
            'daily_pnl': self.get_daily_pnl(),
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
