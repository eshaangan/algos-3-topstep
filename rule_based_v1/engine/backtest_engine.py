"""Backtest Engine - Bar-by-bar simulation with realistic fills.

Simulates trading on historical bars with:
- Slippage and commission modeling
- ATR-based exits (profit target, stop loss, trailing stop, time stop)
- Force-flatten at session close
- Full integration with SignalAggregator and RiskManager
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd

from engine.signal_aggregator import SignalAggregator, TradeDecision
from engine.risk_manager import RiskManager, TradeRecord
from utils.indicators import atr

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """An open position."""
    direction: int
    entry_price: float
    entry_bar_idx: int
    stop_loss: float
    profit_target: float
    time_stop_bar: int
    trailing_active: bool = False
    trailing_stop: float = 0.0
    peak_favorable: float = 0.0
    atr_at_entry: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: list[TradeRecord]
    equity_curve: pd.Series
    daily_pnl: pd.Series

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def max_drawdown(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        peak = self.equity_curve.cummax()
        dd = self.equity_curve - peak
        return dd.min()

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio from daily returns."""
        if self.daily_pnl.empty or self.daily_pnl.std() == 0:
            return 0.0
        return self.daily_pnl.mean() / self.daily_pnl.std() * np.sqrt(252)

    @property
    def max_consecutive_losses(self) -> int:
        max_streak = 0
        current = 0
        for t in self.trades:
            if t.pnl <= 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    def summary(self) -> dict:
        return {
            "total_pnl": self.total_pnl,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "max_consecutive_losses": self.max_consecutive_losses,
            "avg_trade_pnl": self.total_pnl / max(1, self.num_trades),
        }


class BacktestEngine:
    def __init__(
        self,
        aggregator: SignalAggregator,
        risk_manager: RiskManager,
        commission_per_side: float = 0.62,
        slippage_ticks: int = 1,
        # Exit parameters (ATR multiples)
        profit_target_atr: float = 2.0,
        stop_loss_atr: float = 1.5,
        time_stop_bars: int = 24,
        trailing_activation_atr: float = 1.0,
        trailing_distance_atr: float = 0.75,
        atr_period: int = 14,
    ):
        self.aggregator = aggregator
        self.risk_manager = risk_manager
        self.commission_per_side = commission_per_side
        self.slippage_ticks = slippage_ticks
        self.profit_target_atr = profit_target_atr
        self.stop_loss_atr = stop_loss_atr
        self.time_stop_bars = time_stop_bars
        self.trailing_activation_atr = trailing_activation_atr
        self.trailing_distance_atr = trailing_distance_atr
        self.atr_period = atr_period

    def _apply_slippage(self, price: float, direction: int, is_entry: bool) -> float:
        """Apply slippage to fill price. Adverse direction for both entry and exit."""
        tick = self.risk_manager.tick_size
        slippage = self.slippage_ticks * tick
        if is_entry:
            # Entry: pay more for LONG, receive less for SHORT
            return price + slippage * direction
        else:
            # Exit: receive less for LONG, pay more for SHORT
            return price - slippage * direction

    def _round_commission(self) -> float:
        """Total commission for one round-trip trade."""
        return 2 * self.commission_per_side * self.risk_manager.contracts

    def _compute_trade_pnl(self, entry_price: float, exit_price: float, direction: int) -> float:
        """Compute P&L for a trade including commission."""
        raw_pnl = (exit_price - entry_price) * direction * self.risk_manager.contracts * self.risk_manager.point_value
        return raw_pnl - self._round_commission()

    def _check_exit(self, position: Position, bar: pd.Series, bar_idx: int,
                    is_session_close: bool) -> tuple[bool, float, str]:
        """Check if position should be exited.

        Returns:
            (should_exit, exit_price, reason)
        """
        high = bar["high"]
        low = bar["low"]
        close = bar["close"]

        # Force flatten at session close
        if is_session_close:
            exit_price = self._apply_slippage(close, position.direction, is_entry=False)
            return True, exit_price, "session_close"

        # Time stop
        if bar_idx >= position.time_stop_bar:
            exit_price = self._apply_slippage(close, position.direction, is_entry=False)
            return True, exit_price, "time_stop"

        if position.direction == 1:  # LONG
            # Stop loss hit (check low)
            if low <= position.stop_loss:
                exit_price = self._apply_slippage(position.stop_loss, position.direction, is_entry=False)
                return True, exit_price, "stop_loss"

            # Trailing stop
            if position.trailing_active and low <= position.trailing_stop:
                exit_price = self._apply_slippage(position.trailing_stop, position.direction, is_entry=False)
                return True, exit_price, "trailing_stop"

            # Profit target hit (check high)
            if high >= position.profit_target:
                exit_price = self._apply_slippage(position.profit_target, position.direction, is_entry=False)
                return True, exit_price, "profit_target"

            # Update trailing stop
            if not position.trailing_active:
                unrealized = high - position.entry_price
                if unrealized >= self.trailing_activation_atr * position.atr_at_entry:
                    position.trailing_active = True
                    position.peak_favorable = high
                    position.trailing_stop = high - self.trailing_distance_atr * position.atr_at_entry
            else:
                if high > position.peak_favorable:
                    position.peak_favorable = high
                    position.trailing_stop = high - self.trailing_distance_atr * position.atr_at_entry

        else:  # SHORT
            # Stop loss hit (check high)
            if high >= position.stop_loss:
                exit_price = self._apply_slippage(position.stop_loss, position.direction, is_entry=False)
                return True, exit_price, "stop_loss"

            # Trailing stop
            if position.trailing_active and high >= position.trailing_stop:
                exit_price = self._apply_slippage(position.trailing_stop, position.direction, is_entry=False)
                return True, exit_price, "trailing_stop"

            # Profit target hit (check low)
            if low <= position.profit_target:
                exit_price = self._apply_slippage(position.profit_target, position.direction, is_entry=False)
                return True, exit_price, "profit_target"

            # Update trailing stop
            if not position.trailing_active:
                unrealized = position.entry_price - low
                if unrealized >= self.trailing_activation_atr * position.atr_at_entry:
                    position.trailing_active = True
                    position.peak_favorable = low
                    position.trailing_stop = low + self.trailing_distance_atr * position.atr_at_entry
            else:
                if low < position.peak_favorable:
                    position.peak_favorable = low
                    position.trailing_stop = low + self.trailing_distance_atr * position.atr_at_entry

        return False, 0.0, ""

    def run(self, bars: pd.DataFrame, starting_equity: float = 50000.0) -> BacktestResult:
        """Run backtest on historical bars.

        Args:
            bars: DataFrame with OHLCV data, DatetimeIndex.
            starting_equity: Starting account equity.

        Returns:
            BacktestResult with trades, equity curve, daily P&L.
        """
        self.risk_manager.reset_all(starting_equity)

        # Pre-compute ATR for the entire series
        atr_series = atr(bars["high"], bars["low"], bars["close"], self.atr_period)

        min_bars = self.aggregator.required_bars()
        position: Position | None = None
        trades: list[TradeRecord] = []
        equity_values = []
        equity_times = []
        current_equity = starting_equity
        current_date = None

        daily_pnl_records = {}

        for i in range(min_bars, len(bars)):
            bar = bars.iloc[i]
            bar_time = bars.index[i]

            # Convert to Eastern if needed
            if bar_time.tzinfo is not None:
                bar_time_et = bar_time.tz_convert("US/Eastern")
            else:
                bar_time_et = bar_time

            bar_date = bar_time_et.date()

            # New trading day
            if current_date is not None and bar_date != current_date:
                daily_pnl_records[current_date] = self.risk_manager.daily_pnl
                self.risk_manager.reset_daily()
            current_date = bar_date

            self.risk_manager.tick_bar()

            # Check for session close (last bar of the day or near 16:00 ET)
            is_last_bar_of_day = False
            if i + 1 < len(bars):
                next_time = bars.index[i + 1]
                if next_time.tzinfo is not None:
                    next_time = next_time.tz_convert("US/Eastern")
                is_last_bar_of_day = next_time.date() != bar_date
            else:
                is_last_bar_of_day = True

            is_near_close = bar_time_et.hour == 15 and bar_time_et.minute >= 55
            is_session_close = is_last_bar_of_day or is_near_close

            # Check exit if we have a position
            if position is not None:
                should_exit, exit_price, reason = self._check_exit(
                    position, bar, i, is_session_close
                )
                if should_exit:
                    pnl = self._compute_trade_pnl(position.entry_price, exit_price, position.direction)
                    trade = TradeRecord(
                        entry_bar=position.entry_bar_idx,
                        exit_bar=i,
                        direction=position.direction,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                        exit_reason=reason,
                    )
                    trades.append(trade)
                    self.risk_manager.record_trade(trade)
                    current_equity += pnl
                    position = None

            # Check for new entry if no position
            if position is None and not is_session_close:
                can_trade, reason = self.risk_manager.can_trade()
                if can_trade:
                    lookback = bars.iloc[max(0, i - min_bars + 1):i + 1]
                    decision = self.aggregator.evaluate(lookback)

                    if decision.should_trade:
                        current_atr = atr_series.iloc[i]
                        if np.isnan(current_atr) or current_atr <= 0:
                            continue

                        entry_price = self._apply_slippage(
                            bar["close"], decision.direction, is_entry=True
                        )
                        stop_loss = self.risk_manager.compute_stop_price(
                            entry_price, decision.direction, current_atr, self.stop_loss_atr
                        )
                        profit_target = self.risk_manager.compute_target_price(
                            entry_price, decision.direction, current_atr, self.profit_target_atr
                        )

                        position = Position(
                            direction=decision.direction,
                            entry_price=entry_price,
                            entry_bar_idx=i,
                            stop_loss=stop_loss,
                            profit_target=profit_target,
                            time_stop_bar=i + self.time_stop_bars,
                            atr_at_entry=current_atr,
                        )

            equity_values.append(current_equity)
            equity_times.append(bar_time)

        # Record last day
        if current_date is not None:
            daily_pnl_records[current_date] = self.risk_manager.daily_pnl

        equity_curve = pd.Series(equity_values, index=equity_times, name="equity")
        daily_pnl = pd.Series(daily_pnl_records, name="daily_pnl")

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            daily_pnl=daily_pnl,
        )
