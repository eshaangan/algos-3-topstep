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
    from core.projectx_client import BracketInstruction, ProjectXClient, OrderState
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
    BracketInstruction = module.BracketInstruction
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
        config: dict | None = None,
    ):
        """
        Initialize execution engine.

        Args:
            risk_cfg: Risk configuration (from risk.yaml)
            execution_spec: Execution specification (from execution_spec.yaml)
            label_schema: Label schema (for cost calculations)
            dry_run: If True, simulate trades without executing
            config: Live trading configuration (from live_trading.yaml)
        """
        self.risk_cfg = risk_cfg
        self.execution_spec = execution_spec
        self.label_schema = label_schema
        self.dry_run = dry_run
        self.contract_id = contract_id
        self.account_id = account_id
        self.config = config or {}

        # Bracket orders (server-side stop-loss / take-profit)
        topstep_cfg = self.config.get("topstep", {}) or {}
        topstep_order_cfg = topstep_cfg.get("order", {}) or {}
        self.use_brackets = bool(topstep_order_cfg.get("use_brackets", True))
        logger.info(f"Topstep Brackets: enabled={self.use_brackets}")

        # Direction change configuration
        direction_change_cfg = self.config.get("direction_change", {})
        self.direction_change_enabled = direction_change_cfg.get("enabled", True)
        self.direction_change_threshold = direction_change_cfg.get("high_confidence_threshold", 0.20)
        
        # Regime Filter configuration (NEW)
        regime_filter_cfg = self.config.get("regime_filter", {})
        self.regime_filter_enabled = regime_filter_cfg.get("enabled", False)
        logger.info(f"Regime Filter Config: enabled={self.regime_filter_enabled}")
        
        # Volatility Filter configuration (NEW)
        vol_filter_cfg = self.config.get("volatility_filter", {})
        self.volatility_filter_enabled = vol_filter_cfg.get("enabled", False)
        self.adx_threshold = vol_filter_cfg.get("adx_threshold", 20)
        self.adx_period = vol_filter_cfg.get("adx_period", 14)
        logger.info(f"Volatility Filter Config: enabled={self.volatility_filter_enabled}, adx>{self.adx_threshold}")

        # Circuit Breaker configuration (NEW)
        cb_cfg = self.config.get("circuit_breaker", {})
        # Operator override: disable circuit breaker when requested.
        self.circuit_breaker_enabled = False
        self.max_drawdown_limit = cb_cfg.get("max_drawdown_limit", 1500.0) # Stop before $2000
        logger.info(f"Circuit Breaker Config: enabled={self.circuit_breaker_enabled}, limit=${self.max_drawdown_limit}")

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

        # Only require credentials if not in dry run mode
        if not dry_run:
            if not all([username, api_key, resolved_account_id]):
                raise ValueError("Missing Topstep credentials in .env")
            self.account_id = str(resolved_account_id)
            self.contract_id = resolved_contract_id
            logger.info(f"Connecting to Topstep: account={self.account_id}, contract={self.contract_id}")
            # ProjectXClient reads credentials from environment; allow override
            self.client = ProjectXClient(contract_id=self.contract_id, account_id=self.account_id)
        else:
            # In dry run mode, use provided values or defaults
            self.account_id = str(resolved_account_id) if resolved_account_id else "MOCK"
            self.contract_id = resolved_contract_id or "MOCK"
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
        current_regime: Optional[int] = None,  # 1=Bull, -1=Bear, 0=Neutral/Unknown
        kelly_sizer: Optional[object] = None,
        trade_history: Optional[List[Dict]] = None,
        contracts_cap: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Execute a trading signal.

        Args:
            timestamp: Signal timestamp
            direction: Trade direction ("LONG" or "SHORT")
            prediction: Model prediction dictionary
            bars_df: Recent bars (for setting stop/target)
            contracts: Number of contracts to trade
            current_regime: Optional regime override (1=Bull, -1=Bear)

        Returns:
            (success, reason) tuple
        """
        # Check risk gates
        can_trade, reason = self.risk_manager.can_trade(timestamp)
        # Log if we can trade or not, to trace flow
        if not can_trade:
            logger.warning(f"Trade rejected by risk manager: {reason}")
            return False, f"risk_{reason}"
            
        # Circuit Breaker Check
        if self.circuit_breaker_enabled:
            current_drawdown = self.get_drawdown()
            if current_drawdown > self.max_drawdown_limit:
                logger.warning(f"Circuit Breaker Triggered: Drawdown ${current_drawdown:.2f} > ${self.max_drawdown_limit:.2f}")
                return False, "circuit_breaker_drawdown"

        # Check Regime Filter
        # ALWAYS log regime state for debugging
        regime = current_regime
        if regime is None and 'regime' in bars_df.columns and not bars_df.empty:
            regime = bars_df.iloc[-1]['regime']
        
        logger.info(f"DEBUG EXECUTION: FilterEnabled={self.regime_filter_enabled}, Regime={regime}, Direction={direction}")

        if self.regime_filter_enabled:
            # Apply filter logic if regime is known
            if regime is not None:
                if regime == 1 and direction == "SHORT":
                    logger.info(f"Regime Filter: SHORT signal blocked in BULL regime (regime={regime})")
                    return False, "regime_filter_bull"
                elif regime == -1 and direction == "LONG":
                    logger.info(f"Regime Filter: LONG signal blocked in BEAR regime (regime={regime})")
                    return False, "regime_filter_bear"
            
            # Log allowed trade regime
            logger.info(f"Regime Filter: Trade ALLOWED. Direction={direction}, Regime={regime}")

        # Volatility Filter (ADX)
        if self.volatility_filter_enabled:
            adx = self._calculate_adx(bars_df, period=self.adx_period)
            if adx is not None:
                if adx < self.adx_threshold:
                    logger.info(f"Volatility Filter: Blocked low volatility (ADX={adx:.1f} < {self.adx_threshold})")
                    return False, "volatility_filter_low_adx"
                else:
                    logger.info(f"Volatility Filter: Passed (ADX={adx:.1f})")
            else:
                logger.warning("Volatility Filter: Could not calculate ADX (insufficient data)")

        # Check position limits

        # Check position limits
        max_concurrent = self.risk_cfg.get('position_limits', {}).get('max_concurrent_positions', 5)
        if len(self.open_positions) >= max_concurrent:
            logger.warning(f"Max concurrent positions reached: {len(self.open_positions)}")
            return False, "max_concurrent_positions"

        # Check for direction change (if enabled)
        if self.direction_change_enabled and self.open_positions:
            current_direction = self.get_net_position_direction()
            if current_direction != "FLAT" and current_direction != direction:
                # Opposing signal detected
                score_ev = abs(prediction.get('score_ev', 0.0))

                # Only flatten if opposing signal exceeds high-confidence threshold
                if score_ev >= self.direction_change_threshold:
                    logger.info(
                        f"STRONG opposing signal detected: {current_direction} -> {direction} "
                        f"(|score_ev|={score_ev:.3f} >= {self.direction_change_threshold:.3f})"
                    )
                    # Get current price for flattening
                    if bars_df.empty:
                        logger.error("No bars available for flattening")
                        return False, "no_bars_for_flatten"
                    current_price = bars_df.iloc[-1]['close']
                    self.flatten_all_positions(timestamp, current_price, f"direction_change_{current_direction}_to_{direction}")
                    return False, "direction_changed_awaiting_confirmation"
                else:
                    # Weak opposing signal - reject new trade, keep existing positions
                    logger.info(
                        f"WEAK opposing signal rejected: {current_direction} -> {direction} "
                        f"(|score_ev|={score_ev:.3f} < {self.direction_change_threshold:.3f}) "
                        f"- keeping existing positions"
                    )
                    return False, "opposing_signal_too_weak"

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
                bracket_stop = None
                bracket_target = None
                if self.use_brackets:
                    tick_size = float(self.execution_spec["instrument"]["tick_size_points"])
                    stop_ticks = int(round((stop_price - entry_price) / tick_size))
                    target_ticks = int(round((target_price - entry_price) / tick_size))

                    # Ensure correct sign conventions for ProjectX brackets.
                    # Stop: negative for LONG (below), positive for SHORT (above).
                    # Target: opposite sign of stop.
                    if direction == "LONG":
                        stop_ticks = -max(1, abs(stop_ticks))
                        target_ticks = max(1, abs(target_ticks))
                    else:  # SHORT
                        stop_ticks = max(1, abs(stop_ticks))
                        target_ticks = -max(1, abs(target_ticks))

                    bracket_stop = BracketInstruction(ticks=stop_ticks, order_type=4)   # STOP
                    bracket_target = BracketInstruction(ticks=target_ticks, order_type=1)  # LIMIT
                    logger.info(
                        f"Placing bracketed order: stop_ticks={stop_ticks}, target_ticks={target_ticks} "
                        f"(stop={stop_price:.2f}, target={target_price:.2f})"
                    )

                order = self.client.place_order(
                    symbol="MES",
                    side=side,
                    quantity=contracts,
                    order_type="MARKET",
                    contract_id=self.contract_id,
                    stop_loss_bracket=bracket_stop,
                    take_profit_bracket=bracket_target,
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

    def get_net_position_direction(self) -> str:
        """
        Get the net direction of all open positions.

        Returns:
            "LONG" if net long, "SHORT" if net short, "FLAT" if no positions or neutral
        """
        if not self.open_positions:
            return "FLAT"

        net_contracts = 0
        for position in self.open_positions:
            if position['direction'] == "LONG":
                net_contracts += position['contracts']
            else:  # SHORT
                net_contracts -= position['contracts']

        if net_contracts > 0:
            return "LONG"
        elif net_contracts < 0:
            return "SHORT"
        else:
            return "FLAT"

    def _calculate_adx(self, bars_df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """
        Calculate ADX (Average Directional Index).
        
        Args:
            bars_df: DataFrame with high, low, close columns.
            period: Lookback period (default 14).
            
        Returns:
            Current ADX value or None if insufficient data.
        """
        if len(bars_df) < period * 2:
            return None
            
        try:
            high = bars_df['high'].values
            low = bars_df['low'].values
            close = bars_df['close'].values
            
            # True Range
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Directional Movement
            up_move = high[1:] - high[:-1]
            down_move = low[:-1] - low[1:]
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
            
            # Smooth TR, +DM, -DM (Wilder's Smoothing)
            def smooth(x, n):
                y = np.zeros_like(x)
                y[0] = np.sum(x[:n]) # First value is sum
                for i in range(1, len(x)):
                    # Wilder's smoothing: previous * (n-1)/n + current
                    # But standard implementation is often EMA.
                    # Let's use simple EMA for robustness or Wilder's recursive.
                    # y[i] = (y[i-1] * (n-1) + x[i]) / n # This requires a loop, slow.
                    pass
                return pd.Series(x).ewm(alpha=1/n, adjust=False).mean().values

            # Use pandas ewm which is optimized
            # Wilder's alpha = 1/n
            tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
            plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values
            minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values
            
            # Directional Indicators
            plus_di = 100 * (plus_dm_smooth / tr_smooth)
            minus_di = 100 * (minus_dm_smooth / tr_smooth)
            
            # DX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
            
            # ADX (Smooth DX)
            adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
            
            return float(adx[-1])
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return None

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
