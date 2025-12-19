"""
Execution layer that bridges signals, risk management, and market fills.

Assumptions for backtesting:
- Entries are filled at the next bar's open plus configured slippage.
- Stops are prioritised over targets when both levels print inside the same bar.
- Commission is assessed as a round-turn on trade close.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional, Tuple

from config import BacktestConfig, RiskConfig, StrategyConfig
from models import Bar, Order, Position, Signal, SignalAction, Trade
from risk_management import RiskManager


@dataclass
class ExecutionResult:
    """Result of attempting to route a signal through the execution layer."""

    position: Optional[Position]
    order: Optional[Order]
    rejection_reason: Optional[str] = None


@dataclass
class ExitResult:
    """Outcome of evaluating exits for an open position on a given bar."""

    trade: Optional[Trade]
    position: Optional[Position]


class ExecutionEngine:
    """Simple execution abstraction used by the backtester (can be extended for live)."""

    def __init__(
        self,
        risk_manager: RiskManager,
        backtest_cfg: BacktestConfig,
        risk_cfg: RiskConfig,
        strategy_cfg: StrategyConfig,
    ) -> None:
        self.risk_manager = risk_manager
        self.backtest_cfg = backtest_cfg
        self.risk_cfg = risk_cfg
        self.strategy_cfg = strategy_cfg
        # Runtime state for the single open position (Topstep allows only one).
        self._active_remaining_contracts: Optional[int] = None
        self._active_partial_taken: bool = False

    def _apply_slippage(self, action: SignalAction, price: float) -> float:
        """Shift fill price by configured slippage in ticks."""

        adjustment = self.backtest_cfg.slippage_ticks * self.risk_cfg.tick_size
        if action == SignalAction.ENTER_LONG:
            return price + adjustment
        if action == SignalAction.ENTER_SHORT:
            return price - adjustment
        return price

    def process_entry(self, signal: Signal, entry_bar: Bar) -> ExecutionResult:
        """
        Evaluate risk, size the trade, and produce a filled position if allowed.

        Parameters
        ----------
        signal:
            Strategy intent emitted on the previous bar close.
        entry_bar:
            The bar at which the trade will be executed (next bar open).
        """

        assessment = self.risk_manager.assess_signal(signal)
        if not assessment.allowed:
            return ExecutionResult(position=None, order=None, rejection_reason=assessment.reason)

        fill_price = self._apply_slippage(signal.action, entry_bar.open)
        stop_price, target_price = self._aligned_levels(signal, fill_price)
        position = Position(
            symbol=signal.symbol,
            contracts=assessment.contracts,
            direction=signal.action,
            entry_price=fill_price,
            stop_price=stop_price,
            target_price=target_price,
            entry_time=entry_bar.timestamp,
        )
        order = Order(
            id=str(uuid.uuid4()),
            signal=signal,
            side=signal.action,
            contracts=assessment.contracts,
            entry_price=fill_price,
            stop_price=stop_price,
            target_price=target_price,
            status="FILLED",
        )
        self.risk_manager.register_position_open(assessment.contracts)
        # Initialise runtime position state.
        self._active_remaining_contracts = assessment.contracts
        self._active_partial_taken = False
        return ExecutionResult(position=position, order=order)

    def process_exit(self, position: Position, bar: Bar) -> ExitResult:
        """
        Check for stop / target hits on the provided bar and close or scale out.

        Orders of precedence within a single bar:
        1) Effective stop (including any trailing/breakeven adjustment)
        2) Partial profit at configured R-multiple (if enabled and size > 1)
        3) Final target
        """

        # No runtime state -> fall back to simple legacy behaviour.
        if self._active_remaining_contracts is None:
            trade = self._legacy_exit(position, bar)
            return ExitResult(trade=trade, position=None if trade else position)

        direction = position.direction
        tick = self.risk_cfg.tick_size
        entry_risk_ticks = abs(position.entry_price - position.stop_price) / tick if tick else 0.0

        # If risk is not well-defined, use legacy stop/target handling.
        if entry_risk_ticks <= 0:
            trade = self._legacy_exit(position, bar)
            return ExitResult(trade=trade, position=None if trade else position)

        # Compute dynamic effective stop (trailing to breakeven once partial is taken).
        effective_stop = position.stop_price
        if (
            self.strategy_cfg.enable_trailing_stop
            and self._active_partial_taken
        ):
            if direction == SignalAction.ENTER_LONG:
                effective_stop = max(effective_stop, position.entry_price)
            else:
                effective_stop = min(effective_stop, position.entry_price)

        # 1) Stop checks (always take precedence).
        if direction == SignalAction.ENTER_LONG:
            if bar.low <= effective_stop:
                trade = self._close_position(position, effective_stop, bar, "STOP")
                self._reset_active_state()
                return ExitResult(trade=trade, position=None)
        else:
            if bar.high >= effective_stop:
                trade = self._close_position(position, effective_stop, bar, "STOP")
                self._reset_active_state()
                return ExitResult(trade=trade, position=None)

        # 2) Partial profit at configured R multiple.
        if (
            self.strategy_cfg.enable_partial_profit
            and not self._active_partial_taken
            and self._active_remaining_contracts
            and self._active_remaining_contracts > 1
        ):
            rr = self.strategy_cfg.partial_profit_rr
            risk_distance = entry_risk_ticks * tick
            if direction == SignalAction.ENTER_LONG:
                partial_price = position.entry_price + rr * risk_distance
                hit_partial = bar.high >= partial_price
            else:
                partial_price = position.entry_price - rr * risk_distance
                hit_partial = bar.low <= partial_price

            if hit_partial:
                initial_contracts = self._active_remaining_contracts
                # Contracts to take off (at least 1, but leave at least 1 running).
                raw_partial = int(initial_contracts * self.strategy_cfg.partial_profit_fraction)
                partial_contracts = max(1, raw_partial)
                if partial_contracts >= initial_contracts:
                    partial_contracts = initial_contracts - 1

                if partial_contracts > 0:
                    trade = self._close_partial(position, partial_price, bar, partial_contracts, "PARTIAL_TP")
                    self._active_remaining_contracts = initial_contracts - partial_contracts
                    self._active_partial_taken = True
                    position.contracts = self._active_remaining_contracts
                    return ExitResult(trade=trade, position=position)

        # 3) Final target.
        if direction == SignalAction.ENTER_LONG:
            if bar.high >= position.target_price:
                trade = self._close_position(position, position.target_price, bar, "TARGET")
                self._reset_active_state()
                return ExitResult(trade=trade, position=None)
        else:
            if bar.low <= position.target_price:
                trade = self._close_position(position, position.target_price, bar, "TARGET")
                self._reset_active_state()
                return ExitResult(trade=trade, position=None)

        return ExitResult(trade=None, position=position)

    def force_exit(self, position: Position, bar: Bar, reason: str = "SESSION_FLAT") -> Trade:
        """Exit at the bar close regardless of stop/target."""

        trade = self._close_position(position, bar.close, bar, reason)
        self._reset_active_state()
        return trade

    # ------------------------------------------------------------------ internals

    def _reset_active_state(self) -> None:
        """Clear runtime tracking for the current open position."""

        self._active_remaining_contracts = None
        self._active_partial_taken = False

    def _legacy_exit(self, position: Position, bar: Bar) -> Optional[Trade]:
        """Previous stop/target behaviour used as a fallback."""

        direction = position.direction
        exit_price: Optional[float] = None
        exit_reason = ""

        if direction == SignalAction.ENTER_LONG:
            if bar.low <= position.stop_price:
                exit_price = position.stop_price
                exit_reason = "STOP"
            elif bar.high >= position.target_price:
                exit_price = position.target_price
                exit_reason = "TARGET"
        else:
            if bar.high >= position.stop_price:
                exit_price = position.stop_price
                exit_reason = "STOP"
            elif bar.low <= position.target_price:
                exit_price = position.target_price
                exit_reason = "TARGET"

        if exit_price is None:
            return None

        return self._close_position(position, exit_price, bar, exit_reason)

    def _build_trade(
        self,
        position: Position,
        exit_price: float,
        bar: Bar,
        contracts: int,
        reason: str,
    ) -> Trade:
        """Create a Trade object for the given number of contracts."""

        ticks = (exit_price - position.entry_price) / self.risk_cfg.tick_size
        if position.direction == SignalAction.ENTER_SHORT:
            ticks *= -1
        pnl_per_contract = ticks * self.risk_cfg.tick_value
        pnl = pnl_per_contract * contracts
        round_turn_commission = self.backtest_cfg.commission_per_contract * contracts * 2
        pnl -= round_turn_commission
        entry_risk = abs(position.entry_price - position.stop_price) / self.risk_cfg.tick_size
        risk_multiple = (ticks / entry_risk) if entry_risk else 0.0

        return Trade(
            symbol=position.symbol,
            entry_time=position.entry_time,
            exit_time=bar.timestamp,
            entry_price=position.entry_price,
            exit_price=exit_price,
            direction=position.direction,
            contracts=contracts,
            pnl=pnl,
            risk_multiple=risk_multiple,
            reason=reason,
        )

    def _close_position(self, position: Position, exit_price: float, bar: Bar, reason: str) -> Trade:
        """Helper to compute trade stats and notify risk manager for a full close."""

        trade = self._build_trade(position, exit_price, bar, position.contracts, reason)
        self.risk_manager.register_position_close(position.contracts)
        self.risk_manager.update_after_trade(trade)
        # Record trade result for adaptive sizing
        self.risk_manager.record_trade_result(trade)
        return trade

    def _close_partial(
        self,
        position: Position,
        exit_price: float,
        bar: Bar,
        contracts: int,
        reason: str,
    ) -> Trade:
        """Scale out of part of the position while keeping it open."""

        trade = self._build_trade(position, exit_price, bar, contracts, reason)
        self.risk_manager.update_after_trade(trade)
        # Record trade result for adaptive sizing
        self.risk_manager.record_trade_result(trade)
        return trade

    def _aligned_levels(self, signal: Signal, fill_price: float) -> Tuple[float, float]:
        """Shift stop/target around the actual fill price to preserve configured risk."""

        tick = self.risk_cfg.tick_size
        ideal_entry = signal.metadata.get("ideal_entry", fill_price)
        risk_ticks = signal.metadata.get("risk_ticks")
        if risk_ticks is None or risk_ticks <= 0:
            risk_ticks = abs(signal.stop_price - ideal_entry) / tick if tick else 0.0
        if risk_ticks <= 0:
            risk_ticks = 1.0
        stop_distance = risk_ticks * tick
        target_multiple = signal.metadata.get("target_rr_multiple", 2.0)
        if signal.action == SignalAction.ENTER_LONG:
            stop = fill_price - stop_distance
            target = fill_price + (stop_distance * target_multiple)
        else:
            stop = fill_price + stop_distance
            target = fill_price - (stop_distance * target_multiple)
        return stop, target

