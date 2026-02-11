"""
Circuit Breaker System - Quick Win #2

Auto-stops trading when model performance degrades to prevent catastrophic losses.

Research Context:
- Jan 2026 live trading lost -$884.73 over 18 days
- System kept trading despite 35.5% win rate (below profitable threshold)
- No mechanism to stop trading during regime shifts or bad performance

Expected Impact:
- Would have stopped Jan 2026 trading at -$500 instead of -$884
- Prevents drawdowns beyond acceptable limits
- Preserves capital for favorable trading conditions

Usage:
    circuit_breaker = CircuitBreaker()

    # After each trade
    safe_to_continue = circuit_breaker.check(trade_result, daily_pnl)
    if not safe_to_continue:
        logger.critical(f"🚨 CIRCUIT BREAKER TRIPPED: {circuit_breaker.trip_reason}")
        # Stop trading for the day
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Monitors trading performance and stops trading when risk thresholds are breached.

    Triggers on:
    1. Consecutive losses (default: 3 in a row)
    2. Daily loss limit (default: -$500, well before Topstep's -$1,000 limit)
    3. Low win rate after minimum trades (default: <30% after 10 trades)
    """

    def __init__(
        self,
        max_consecutive_losses: int = 3,
        daily_loss_limit: float = -500.0,
        min_win_rate_after_n_trades: Tuple[int, float] = (10, 0.30),
        lookback_trades: int = 20,
        reset_on_new_day: bool = True
    ):
        """
        Initialize circuit breaker with safety thresholds.

        Args:
            max_consecutive_losses: Stop after this many consecutive losses (default: 3)
            daily_loss_limit: Stop if daily P&L drops below this (default: -$500)
            min_win_rate_after_n_trades: (min_trades, min_win_rate) tuple
                - Stop if win rate drops below threshold after min_trades
                - Default: (10, 0.30) = stop if <30% after 10 trades
            lookback_trades: Number of recent trades to track (default: 20)
            reset_on_new_day: Whether to reset state at start of new trading day
        """
        self.max_consecutive_losses = max_consecutive_losses
        self.daily_loss_limit = daily_loss_limit
        self.min_trades, self.min_win_rate = min_win_rate_after_n_trades
        self.lookback_trades = lookback_trades
        self.reset_on_new_day = reset_on_new_day

        # State tracking
        self.recent_trades: List[Dict] = []
        self.is_tripped = False
        self.trip_reason: Optional[str] = None
        self.trip_timestamp: Optional[datetime] = None
        self.last_trade_date: Optional[str] = None

        # Statistics
        self.total_checks = 0
        self.total_trips = 0

    def check(
        self,
        trade_result: Dict,
        daily_pnl: float,
        current_date: Optional[str] = None
    ) -> bool:
        """
        Check if it's safe to continue trading.

        Args:
            trade_result: Dict with keys:
                - 'pnl': Trade profit/loss (float)
                - 'timestamp': Trade timestamp (datetime or str, optional)
                - 'symbol': Symbol traded (str, optional)
            daily_pnl: Cumulative P&L for current trading day
            current_date: Current date in YYYY-MM-DD format (for daily reset)

        Returns:
            True if safe to continue trading, False if circuit breaker tripped
        """
        self.total_checks += 1

        # Reset state on new trading day
        if self.reset_on_new_day and current_date:
            if self.last_trade_date and current_date != self.last_trade_date:
                logger.info(f"Circuit breaker: New trading day ({current_date}), resetting state")
                self.reset()
            self.last_trade_date = current_date

        # Add trade to recent history
        self.recent_trades.append(trade_result)

        # Maintain lookback window
        if len(self.recent_trades) > self.lookback_trades:
            self.recent_trades.pop(0)

        # Check 1: Daily loss limit
        if daily_pnl <= self.daily_loss_limit:
            self._trip(
                f"Daily loss limit breached: ${daily_pnl:.2f} <= ${self.daily_loss_limit:.2f}"
            )
            return False

        # Check 2: Consecutive losses
        consecutive_losses = self._count_consecutive_losses()
        if consecutive_losses >= self.max_consecutive_losses:
            self._trip(
                f"{consecutive_losses} consecutive losses (limit: {self.max_consecutive_losses})"
            )
            return False

        # Check 3: Win rate too low (only after minimum trades)
        if len(self.recent_trades) >= self.min_trades:
            win_rate = self._calculate_win_rate()

            if win_rate < self.min_win_rate:
                self._trip(
                    f"Win rate too low: {win_rate:.1%} < {self.min_win_rate:.1%} "
                    f"(after {len(self.recent_trades)} trades)"
                )
                return False

        # All checks passed
        return True

    def _count_consecutive_losses(self) -> int:
        """Count consecutive losing trades from most recent."""
        consecutive = 0
        for trade in reversed(self.recent_trades):
            if trade['pnl'] < 0:
                consecutive += 1
            else:
                break
        return consecutive

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from recent trades."""
        if not self.recent_trades:
            return 0.0

        wins = sum(1 for t in self.recent_trades if t['pnl'] > 0)
        return wins / len(self.recent_trades)

    def _trip(self, reason: str) -> None:
        """Mark circuit breaker as tripped."""
        self.is_tripped = True
        self.trip_reason = reason
        self.trip_timestamp = datetime.now()
        self.total_trips += 1

        logger.critical(f"🚨 CIRCUIT BREAKER TRIPPED: {reason}")
        logger.critical(f"   Total trips today: {self.total_trips}")
        logger.critical(f"   Recent trades: {self._format_recent_trades()}")

    def _format_recent_trades(self) -> str:
        """Format recent trades for logging."""
        if not self.recent_trades:
            return "None"

        # Show last 5 trades
        last_5 = self.recent_trades[-5:]
        formatted = []
        for i, trade in enumerate(last_5, 1):
            pnl = trade['pnl']
            symbol = trade.get('symbol', '???')
            formatted.append(f"{symbol} ${pnl:+.2f}")

        return ", ".join(formatted)

    def reset(self) -> None:
        """Reset circuit breaker state (for new trading day or manual reset)."""
        logger.info("Circuit breaker: Resetting state")
        self.recent_trades = []
        self.is_tripped = False
        self.trip_reason = None
        self.trip_timestamp = None
        # Note: Don't reset total_trips (cumulative across resets)

    def get_status(self) -> Dict:
        """
        Get current circuit breaker status.

        Returns:
            Dict with status information
        """
        return {
            'is_tripped': self.is_tripped,
            'trip_reason': self.trip_reason,
            'trip_timestamp': self.trip_timestamp.isoformat() if self.trip_timestamp else None,
            'recent_trades_count': len(self.recent_trades),
            'consecutive_losses': self._count_consecutive_losses(),
            'win_rate': self._calculate_win_rate() if self.recent_trades else None,
            'total_checks': self.total_checks,
            'total_trips': self.total_trips,
        }

    def __repr__(self) -> str:
        status = "TRIPPED" if self.is_tripped else "ACTIVE"
        return (
            f"CircuitBreaker(status={status}, "
            f"trades={len(self.recent_trades)}, "
            f"consecutive_losses={self._count_consecutive_losses()})"
        )


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    # Test circuit breaker
    print("=" * 60)
    print("Circuit Breaker Test")
    print("=" * 60)

    cb = CircuitBreaker()

    # Simulate trades
    test_trades = [
        {'pnl': 50, 'symbol': 'MES'},   # Win
        {'pnl': -30, 'symbol': 'MES'},  # Loss 1
        {'pnl': -30, 'symbol': 'MES'},  # Loss 2
        {'pnl': -30, 'symbol': 'MES'},  # Loss 3 (should trip)
    ]

    daily_pnl = 0
    for i, trade in enumerate(test_trades, 1):
        daily_pnl += trade['pnl']
        print(f"\nTrade {i}: {trade['symbol']} ${trade['pnl']:+.2f} (Daily P&L: ${daily_pnl:+.2f})")

        safe = cb.check(trade, daily_pnl)
        print(f"Safe to continue: {safe}")

        if not safe:
            print(f"⛔ Trading stopped: {cb.trip_reason}")
            break

    print("\n" + "=" * 60)
    print("Final Status:")
    print(cb.get_status())
