"""Risk Manager - Position sizing, daily P&L limits, drawdown tracking.

Enforces Topstep 50k Combine constraints:
- Fixed 1 MES contract
- Daily loss hard stop
- Per-trade max loss
- Circuit breaker on consecutive losses
- Cooldown after losses
- Force flatten before session close
- Trailing drawdown buffer
"""

from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str


class RiskManager:
    def __init__(
        self,
        contracts: int = 1,
        point_value: float = 5.0,
        tick_size: float = 0.25,
        tick_value: float = 1.25,
        max_daily_loss: float = -400.0,
        per_trade_max_loss: float = 100.0,
        max_consecutive_losses: int = 3,
        cooldown_bars: int = 5,
        flatten_minutes_before_close: int = 5,
        drawdown_buffer: float = 500.0,
    ):
        self.contracts = contracts
        self.point_value = point_value
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.max_daily_loss = max_daily_loss
        self.per_trade_max_loss = per_trade_max_loss
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_bars = cooldown_bars
        self.flatten_minutes_before_close = flatten_minutes_before_close
        self.drawdown_buffer = drawdown_buffer

        # State
        self.daily_pnl: float = 0.0
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.consecutive_losses: int = 0
        self.bars_since_last_trade: int = 999  # Start with no cooldown
        self.trades_today: list[TradeRecord] = []
        self._halted: bool = False
        self._halt_reason: str = ""

    def reset_daily(self):
        """Reset daily state at start of new session."""
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.bars_since_last_trade = 999
        self.trades_today = []
        self._halted = False
        self._halt_reason = ""

    def reset_all(self, starting_equity: float = 0.0):
        """Full reset for new backtest/session."""
        self.reset_daily()
        self.peak_equity = starting_equity
        self.current_equity = starting_equity

    def can_trade(self) -> tuple[bool, str]:
        """Check if we're allowed to take a new trade.

        Returns:
            (allowed, reason) tuple.
        """
        if self._halted:
            return False, f"Halted: {self._halt_reason}"

        if self.daily_pnl <= self.max_daily_loss:
            self._halted = True
            self._halt_reason = f"Daily loss limit hit: ${self.daily_pnl:.2f}"
            return False, self._halt_reason

        if self.consecutive_losses >= self.max_consecutive_losses:
            self._halted = True
            self._halt_reason = f"Circuit breaker: {self.consecutive_losses} consecutive losses"
            return False, self._halt_reason

        if self.bars_since_last_trade < self.cooldown_bars:
            return False, f"Cooldown: {self.bars_since_last_trade}/{self.cooldown_bars} bars"

        # Check trailing drawdown buffer
        drawdown = self.peak_equity - self.current_equity
        if drawdown >= self.drawdown_buffer:
            self._halted = True
            self._halt_reason = f"Drawdown buffer: ${drawdown:.2f} >= ${self.drawdown_buffer:.2f}"
            return False, self._halt_reason

        return True, "OK"

    def compute_stop_price(self, entry_price: float, direction: int, atr: float,
                           stop_loss_atr: float = 1.5) -> float:
        """Compute stop-loss price, capping at per-trade max loss."""
        raw_stop_distance = stop_loss_atr * atr
        max_stop_points = self.per_trade_max_loss / (self.contracts * self.point_value)
        stop_distance = min(raw_stop_distance, max_stop_points)

        if direction == 1:  # LONG
            return entry_price - stop_distance
        else:  # SHORT
            return entry_price + stop_distance

    def compute_target_price(self, entry_price: float, direction: int, atr: float,
                              profit_target_atr: float = 2.0) -> float:
        """Compute profit target price."""
        target_distance = profit_target_atr * atr
        if direction == 1:  # LONG
            return entry_price + target_distance
        else:  # SHORT
            return entry_price - target_distance

    def record_trade(self, trade: TradeRecord):
        """Record a completed trade and update state."""
        self.trades_today.append(trade)
        self.daily_pnl += trade.pnl
        self.current_equity += trade.pnl
        self.peak_equity = max(self.peak_equity, self.current_equity)
        self.bars_since_last_trade = 0

        if trade.pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        logger.info(
            f"Trade recorded: {trade.exit_reason}, PnL=${trade.pnl:.2f}, "
            f"Daily=${self.daily_pnl:.2f}, Consecutive losses={self.consecutive_losses}"
        )

    def tick_bar(self):
        """Called each bar to update cooldown counter."""
        self.bars_since_last_trade += 1

    def should_flatten_for_close(self, current_time, session_close_time) -> bool:
        """Check if we should force-flatten due to approaching session close."""
        from datetime import timedelta
        cutoff = (
            session_close_time - timedelta(minutes=self.flatten_minutes_before_close)
        )
        return current_time.time() >= cutoff.time() if hasattr(cutoff, 'time') else False

    @property
    def session_stats(self) -> dict:
        """Summary statistics for the current session."""
        wins = [t for t in self.trades_today if t.pnl > 0]
        losses = [t for t in self.trades_today if t.pnl <= 0]
        return {
            "total_trades": len(self.trades_today),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / max(1, len(self.trades_today)),
            "daily_pnl": self.daily_pnl,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
        }
