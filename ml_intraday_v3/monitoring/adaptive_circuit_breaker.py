"""
Adaptive Circuit Breaker - REVISED for Combine Speed

Instead of STOPPING trading after losses, this circuit breaker ADAPTS:
- After consecutive losses: Pause 30 min, raise threshold, reduce size
- After daily loss limit: STOP for the day (firm rule for Topstep protection)
- After low win rate: Raise threshold temporarily (be more selective)

Key Principle: **Don't stop trading - adapt and improve**

Why This Approach:
- Topstep combine requires $3,000 in 15-20 days
- Need 6-8 trades/day to hit target
- Stopping after 3 losses = can't recover, can't pass combine
- Better: Pause, adapt, resume with better parameters

Usage:
    acb = AdaptiveCircuitBreaker()

    # After each trade
    action = acb.check_and_adapt(trade_result, daily_pnl)

    if action == 'stop_today':
        # Hit daily loss limit - stop completely
        stop_trading()
    elif action == 'cooling_off':
        # Paused for 30 min - skip this signal
        continue
    elif action == 'adapted':
        # Adapted settings (higher threshold, smaller size)
        # Continue trading with adjusted parameters
        confidence_threshold = acb.get_current_threshold()
        position_multiplier = acb.get_current_position_multiplier()
"""

import logging
from typing import Dict, List, Optional, Literal
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

ActionType = Literal['continue', 'cooling_off', 'adapted', 'stop_today']


class AdaptiveCircuitBreaker:
    """
    Monitors trading performance and adapts behavior instead of stopping completely.

    Adaptive Actions:
    1. After 3 consecutive losses → Pause 30 min, then trade with higher threshold
    2. After daily loss of -$500 → STOP for the day (Topstep protection)
    3. After low win rate → Raise threshold for rest of day (be more selective)
    """

    def __init__(
        self,
        # Consecutive losses
        consecutive_losses_limit: int = 3,
        cooling_off_minutes: int = 30,
        temp_confidence_boost: float = 0.10,  # Raise threshold by this amount
        temp_position_reduction: float = 0.5,  # Reduce size by this factor

        # Daily loss limit (FIRM STOP)
        daily_loss_limit: float = -500.0,

        # Low win rate
        min_win_rate_after_n_trades: tuple = (10, 0.40),
        low_win_rate_confidence_boost: float = 0.05,

        # State management
        lookback_trades: int = 20,
        reset_on_new_day: bool = True,

        # Base thresholds (from config)
        base_confidence_threshold: float = 0.55
    ):
        """
        Initialize adaptive circuit breaker.

        Args:
            consecutive_losses_limit: Pause after this many consecutive losses
            cooling_off_minutes: Minutes to pause after consecutive losses
            temp_confidence_boost: Temporarily raise threshold by this amount
            temp_position_reduction: Reduce position size by this factor during cooldown
            daily_loss_limit: Stop trading if daily P&L drops below this
            min_win_rate_after_n_trades: (min_trades, min_win_rate) tuple
            low_win_rate_confidence_boost: Raise threshold when win rate low
            lookback_trades: Number of recent trades to track
            reset_on_new_day: Reset state on new trading day
            base_confidence_threshold: Base threshold from config (default: 0.55)
        """
        # Consecutive losses settings
        self.consecutive_losses_limit = consecutive_losses_limit
        self.cooling_off_minutes = cooling_off_minutes
        self.temp_confidence_boost = temp_confidence_boost
        self.temp_position_reduction = temp_position_reduction

        # Daily loss limit
        self.daily_loss_limit = daily_loss_limit

        # Win rate settings
        self.min_trades, self.min_win_rate = min_win_rate_after_n_trades
        self.low_win_rate_confidence_boost = low_win_rate_confidence_boost

        # State management
        self.lookback_trades = lookback_trades
        self.reset_on_new_day = reset_on_new_day
        self.base_confidence_threshold = base_confidence_threshold

        # Current state
        self.recent_trades: List[Dict] = []
        self.cooling_off_until: Optional[datetime] = None
        self.stop_trading_today = False
        self.win_rate_adapted = False
        self.last_trade_date: Optional[str] = None

        # Statistics
        self.total_checks = 0
        self.total_cooling_offs = 0
        self.total_adaptations = 0
        self.total_daily_stops = 0

    def check_and_adapt(
        self,
        trade_result: Dict,
        daily_pnl: float,
        current_time: Optional[datetime] = None
    ) -> ActionType:
        """
        Check trade result and adapt behavior accordingly.

        Args:
            trade_result: Dict with keys:
                - 'pnl': Trade profit/loss (float)
                - 'timestamp': Trade timestamp (datetime or str, optional)
                - 'symbol': Symbol traded (str, optional)
            daily_pnl: Cumulative P&L for current trading day
            current_time: Current time (for testing, otherwise uses datetime.now())

        Returns:
            Action type:
                - 'continue': All good, continue trading normally
                - 'cooling_off': In cooling-off period, skip trades
                - 'adapted': Settings adapted (higher threshold, smaller size), continue trading
                - 'stop_today': Daily loss limit hit, stop trading for the day
        """
        if current_time is None:
            current_time = datetime.now()

        self.total_checks += 1

        # Reset state on new trading day
        current_date = current_time.strftime('%Y-%m-%d')
        if self.reset_on_new_day and self.last_trade_date and current_date != self.last_trade_date:
            logger.info(f"Adaptive CB: New trading day ({current_date}), resetting state")
            self.reset()
        self.last_trade_date = current_date

        # Add trade to recent history
        self.recent_trades.append(trade_result)
        if len(self.recent_trades) > self.lookback_trades:
            self.recent_trades.pop(0)

        # Check 1: Daily loss limit (FIRM STOP)
        if daily_pnl <= self.daily_loss_limit:
            self.stop_trading_today = True
            self.total_daily_stops += 1
            logger.critical(f"🚨 DAILY LOSS LIMIT HIT: ${daily_pnl:.2f} <= ${self.daily_loss_limit:.2f}")
            logger.critical("   STOPPING ALL TRADING FOR TODAY")
            logger.critical(f"   Daily stops this session: {self.total_daily_stops}")
            return 'stop_today'

        # Check 2: Cooling-off period (after consecutive losses)
        if self.cooling_off_until and current_time < self.cooling_off_until:
            remaining = (self.cooling_off_until - current_time).seconds // 60
            logger.info(f"⏸️  Still in cooling-off period ({remaining} min remaining)")
            return 'cooling_off'

        # Check 3: Consecutive losses
        consecutive_losses = self._count_consecutive_losses()
        if consecutive_losses >= self.consecutive_losses_limit:
            self._enter_cooling_off(current_time)
            return 'cooling_off'

        # Check 4: Low win rate (after minimum trades)
        if len(self.recent_trades) >= self.min_trades:
            win_rate = self._calculate_win_rate()
            if win_rate < self.min_win_rate and not self.win_rate_adapted:
                self._adapt_for_low_win_rate(win_rate)
                return 'adapted'

        # All checks passed - check if currently adapted
        if self.win_rate_adapted:
            return 'adapted'
        else:
            return 'continue'

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

    def _enter_cooling_off(self, current_time: datetime) -> None:
        """Enter cooling-off period after consecutive losses."""
        self.cooling_off_until = current_time + timedelta(minutes=self.cooling_off_minutes)
        self.total_cooling_offs += 1

        logger.warning("⚠️  CONSECUTIVE LOSSES DETECTED")
        logger.warning(f"   Entering cooling-off mode for {self.cooling_off_minutes} minutes")
        logger.warning(f"   Will resume at {self.cooling_off_until.strftime('%H:%M:%S')}")
        logger.warning(f"   After cooldown: Higher threshold (+{self.temp_confidence_boost:.2f}), "
                      f"smaller size ({self.temp_position_reduction:.0%})")
        logger.warning(f"   Recent trades: {self._format_recent_trades()}")

    def _adapt_for_low_win_rate(self, current_win_rate: float) -> None:
        """Adapt settings due to low win rate."""
        self.win_rate_adapted = True
        self.total_adaptations += 1

        logger.warning("⚠️  LOW WIN RATE DETECTED")
        logger.warning(f"   Current win rate: {current_win_rate:.1%} (threshold: {self.min_win_rate:.1%})")
        logger.warning(f"   Raising confidence threshold by +{self.low_win_rate_confidence_boost:.2f}")
        logger.warning("   Will reset tomorrow or after win rate improves")

    def _format_recent_trades(self) -> str:
        """Format recent trades for logging."""
        if not self.recent_trades:
            return "None"
        last_5 = self.recent_trades[-5:]
        formatted = []
        for trade in last_5:
            pnl = trade['pnl']
            symbol = trade.get('symbol', '???')
            formatted.append(f"{symbol} ${pnl:+.2f}")
        return ", ".join(formatted)

    def get_current_threshold(self) -> float:
        """
        Get current confidence threshold (may be boosted).

        Returns:
            Current threshold to use for signal filtering
        """
        boost = 0.0

        # Boost if recently exited cooling-off
        if self.cooling_off_until and datetime.now() < self.cooling_off_until + timedelta(hours=1):
            boost += self.temp_confidence_boost

        # Boost if low win rate
        if self.win_rate_adapted:
            boost += self.low_win_rate_confidence_boost

        return self.base_confidence_threshold + boost

    def get_current_position_multiplier(self) -> float:
        """
        Get current position size multiplier (may be reduced).

        Returns:
            Multiplier for position size (1.0 = normal, 0.5 = half size)
        """
        # Reduce size if recently exited cooling-off
        if self.cooling_off_until and datetime.now() < self.cooling_off_until + timedelta(hours=1):
            return self.temp_position_reduction

        return 1.0

    def is_in_cooling_off(self) -> bool:
        """Check if currently in cooling-off period."""
        if self.cooling_off_until is None:
            return False
        return datetime.now() < self.cooling_off_until

    def should_stop_today(self) -> bool:
        """Check if trading should stop for the day."""
        return self.stop_trading_today

    def reset(self) -> None:
        """Reset circuit breaker state (for new trading day)."""
        logger.info("Adaptive CB: Resetting state for new day")
        self.recent_trades = []
        self.cooling_off_until = None
        self.stop_trading_today = False
        self.win_rate_adapted = False
        # Don't reset total counters (cumulative across days)

    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        return {
            'is_stopped': self.stop_trading_today,
            'is_cooling_off': self.is_in_cooling_off(),
            'cooling_off_until': self.cooling_off_until.isoformat() if self.cooling_off_until else None,
            'is_adapted': self.win_rate_adapted,
            'current_threshold': self.get_current_threshold(),
            'current_position_multiplier': self.get_current_position_multiplier(),
            'recent_trades_count': len(self.recent_trades),
            'consecutive_losses': self._count_consecutive_losses(),
            'win_rate': self._calculate_win_rate() if self.recent_trades else None,
            'total_checks': self.total_checks,
            'total_cooling_offs': self.total_cooling_offs,
            'total_adaptations': self.total_adaptations,
            'total_daily_stops': self.total_daily_stops,
        }

    def __repr__(self) -> str:
        if self.stop_trading_today:
            status = "STOPPED"
        elif self.is_in_cooling_off():
            status = "COOLING_OFF"
        elif self.win_rate_adapted:
            status = "ADAPTED"
        else:
            status = "ACTIVE"

        return (
            f"AdaptiveCircuitBreaker(status={status}, "
            f"threshold={self.get_current_threshold():.2f}, "
            f"trades={len(self.recent_trades)})"
        )


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    print("=" * 70)
    print("Adaptive Circuit Breaker Test")
    print("=" * 70)

    acb = AdaptiveCircuitBreaker(base_confidence_threshold=0.55)

    # Simulate trades
    test_trades = [
        {'pnl': 50, 'symbol': 'MES'},   # Win
        {'pnl': -30, 'symbol': 'MES'},  # Loss 1
        {'pnl': -30, 'symbol': 'MES'},  # Loss 2
        {'pnl': -30, 'symbol': 'MES'},  # Loss 3 (should enter cooling-off)
        {'pnl': 40, 'symbol': 'MES'},   # Should be in cooling-off (skip)
        {'pnl': 40, 'symbol': 'MES'},   # Should be in cooling-off (skip)
    ]

    daily_pnl = 0
    for i, trade in enumerate(test_trades, 1):
        daily_pnl += trade['pnl']

        print(f"\nTrade {i}: {trade['symbol']} ${trade['pnl']:+.2f} (Daily P&L: ${daily_pnl:+.2f})")

        action = acb.check_and_adapt(trade, daily_pnl)
        print(f"Action: {action}")

        if action == 'cooling_off':
            print(f"⏸️  In cooling-off period - would skip this signal")
        elif action == 'adapted':
            print(f"⚙️  Adapted - using threshold {acb.get_current_threshold():.2f}, "
                  f"size {acb.get_current_position_multiplier():.0%}")
        elif action == 'stop_today':
            print(f"⛔ Stopped for the day")
            break

    print("\n" + "=" * 70)
    print("Final Status:")
    import json
    print(json.dumps(acb.get_status(), indent=2))
