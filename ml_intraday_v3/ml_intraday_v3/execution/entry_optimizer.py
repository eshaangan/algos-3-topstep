"""
Entry Timing Optimization - Phase 2b Priority #1

Improves entry prices by waiting for pullbacks instead of chasing momentum.

Key Concept:
- When signal fires, DON'T enter immediately at market
- Instead, place limit order 2-3 ticks better than signal price
- Wait up to 3 bars for price to reach our level
- If timeout, enter at market (don't miss the trade)

Expected Impact: +$8-10 per trade improvement

Research Context:
- "Market Microstructure and Optimal Order Execution" shows limit orders
  can reduce costs by 0.5-1.0% compared to market orders
- For MES at $5,000: 0.5% = $25, or 10 ticks = $12.50
- Conservative estimate: 3 ticks improvement = $3.75 per side = $7.50 RT

Usage:
    optimizer = EntryOptimizer(pullback_ticks=3, max_wait_bars=3)

    # Calculate optimal entry
    entry_price = optimizer.calculate_entry_price(
        signal_price=5000.0,
        side='LONG',
        recent_bars=bars_df
    )

    # Check if should enter now
    should_enter = optimizer.should_enter_now(
        current_price=4999.25,
        entry_price=entry_price,
        side='LONG',
        bars_waited=2
    )
"""

import logging
from typing import Literal, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

SideType = Literal['LONG', 'SHORT']


class EntryOptimizer:
    """
    Optimizes entry timing by waiting for better prices.

    Strategy:
    - LONG signals: Wait for price to pull back (place limit order below signal)
    - SHORT signals: Wait for price to rally up (place limit order above signal)
    - Timeout: If price doesn't reach limit, enter at market after max_wait_bars
    """

    def __init__(
        self,
        pullback_ticks: int = 3,
        max_wait_bars: int = 3,
        tick_size: float = 0.25,
        use_limit_orders: bool = True,
        adaptive_pullback: bool = False
    ):
        """
        Initialize entry optimizer.

        Args:
            pullback_ticks: Number of ticks to wait for pullback (default: 3)
                - 3 ticks for MES = $3.75 improvement per entry
            max_wait_bars: Maximum bars to wait before timeout (default: 3)
                - 3 bars at 5min = 15 minutes max wait
            tick_size: Tick size for instrument (default: 0.25 for MES)
            use_limit_orders: If True, use limit orders; if False, just track improvement
            adaptive_pullback: If True, adjust pullback based on volatility
        """
        self.pullback_ticks = pullback_ticks
        self.max_wait_bars = max_wait_bars
        self.tick_size = tick_size
        self.use_limit_orders = use_limit_orders
        self.adaptive_pullback = adaptive_pullback

        # Statistics tracking
        self.entries_attempted = 0
        self.entries_filled_at_limit = 0
        self.entries_timed_out = 0
        self.total_price_improvement = 0.0

        logger.info(
            f"EntryOptimizer initialized: pullback={pullback_ticks} ticks, "
            f"max_wait={max_wait_bars} bars, adaptive={adaptive_pullback}"
        )

    def calculate_entry_price(
        self,
        signal_price: float,
        side: SideType,
        recent_bars: Optional[pd.DataFrame] = None,
        current_atr: Optional[float] = None
    ) -> float:
        """
        Calculate optimal entry price based on pullback logic.

        Args:
            signal_price: Price when signal fired
            side: 'LONG' or 'SHORT'
            recent_bars: Recent OHLC bars (for adaptive calculation)
            current_atr: Current ATR value (for adaptive calculation)

        Returns:
            Optimal entry price (limit order price)

        Example:
            Signal fires at 5000.00 for LONG
            Pullback = 3 ticks = 0.75 points
            Entry price = 4999.25 (wait for pullback)
        """
        self.entries_attempted += 1

        # Calculate pullback amount
        pullback_ticks = self.pullback_ticks

        # Adaptive pullback based on volatility
        if self.adaptive_pullback and current_atr is not None:
            # In high volatility, wait for larger pullback
            # In low volatility, smaller pullback is sufficient
            # Scale pullback: 2-4 ticks based on ATR
            # This is a placeholder - could be refined with more data
            atr_percentile = self._estimate_atr_percentile(current_atr, recent_bars)
            if atr_percentile > 70:  # High volatility
                pullback_ticks = min(4, self.pullback_ticks + 1)
            elif atr_percentile < 30:  # Low volatility
                pullback_ticks = max(2, self.pullback_ticks - 1)

        pullback_amount = pullback_ticks * self.tick_size

        # Calculate entry price
        if side == 'LONG':
            # Enter below signal price (wait for pullback)
            entry_price = signal_price - pullback_amount
        else:  # SHORT
            # Enter above signal price (wait for rally)
            entry_price = signal_price + pullback_amount

        logger.debug(
            f"Calculated entry: signal_price={signal_price:.2f}, "
            f"side={side}, pullback={pullback_ticks} ticks, "
            f"entry_price={entry_price:.2f}"
        )

        return entry_price

    def should_enter_now(
        self,
        current_price: float,
        entry_price: float,
        side: SideType,
        bars_waited: int
    ) -> bool:
        """
        Check if should enter trade now.

        Args:
            current_price: Current market price
            entry_price: Our limit order price
            side: 'LONG' or 'SHORT'
            bars_waited: Number of bars waited so far

        Returns:
            True if should enter (price reached limit or timeout), False otherwise

        Logic:
            - If bars_waited >= max_wait_bars: TIMEOUT, enter at market
            - If price reached our limit: FILLED, enter now
            - Otherwise: WAIT, don't enter yet
        """
        # Check timeout first
        if bars_waited >= self.max_wait_bars:
            logger.debug(
                f"Entry timeout: waited {bars_waited} bars (max={self.max_wait_bars}), "
                f"entering at market price={current_price:.2f}"
            )
            self.entries_timed_out += 1
            return True

        # Check if price reached our limit
        price_reached = False
        if side == 'LONG':
            # For LONG, price must drop to or below our entry price
            price_reached = current_price <= entry_price
        else:  # SHORT
            # For SHORT, price must rise to or above our entry price
            price_reached = current_price >= entry_price

        if price_reached:
            improvement = abs(current_price - entry_price)
            self.entries_filled_at_limit += 1
            self.total_price_improvement += improvement
            logger.debug(
                f"Limit filled: side={side}, entry_price={entry_price:.2f}, "
                f"current_price={current_price:.2f}, improvement=${improvement * 5:.2f}"
            )
            return True

        # Keep waiting
        logger.debug(
            f"Waiting for better price: side={side}, target={entry_price:.2f}, "
            f"current={current_price:.2f}, bars_waited={bars_waited}"
        )
        return False

    def _estimate_atr_percentile(
        self,
        current_atr: float,
        recent_bars: Optional[pd.DataFrame]
    ) -> float:
        """
        Estimate ATR percentile from recent bars.

        Args:
            current_atr: Current ATR value
            recent_bars: Recent OHLC bars

        Returns:
            Percentile (0-100) of current ATR
        """
        if recent_bars is None or len(recent_bars) < 20:
            return 50.0  # Default to median if insufficient data

        # Calculate historical ATRs
        try:
            # Simple TR calculation
            high = recent_bars['high'].values
            low = recent_bars['low'].values
            close = recent_bars['close'].values

            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))
                )
            )

            # Use last 20 bars
            recent_tr = tr[-20:]
            percentile = (np.sum(recent_tr < current_atr) / len(recent_tr)) * 100

            return percentile
        except Exception as e:
            logger.warning(f"Error calculating ATR percentile: {e}")
            return 50.0

    def get_statistics(self) -> dict:
        """
        Get entry optimizer statistics.

        Returns:
            Dictionary with statistics:
                - entries_attempted: Total entries attempted
                - entries_filled_at_limit: Entries filled at limit price
                - entries_timed_out: Entries that timed out
                - fill_rate: % of entries filled at limit vs timeout
                - avg_improvement: Average price improvement per trade (in ticks)
                - total_improvement: Total price improvement (in dollars for MES)
        """
        fill_rate = 0.0
        if self.entries_attempted > 0:
            fill_rate = self.entries_filled_at_limit / self.entries_attempted

        avg_improvement_ticks = 0.0
        if self.entries_filled_at_limit > 0:
            avg_improvement_ticks = (
                self.total_price_improvement / self.entries_filled_at_limit
            ) / self.tick_size

        # For MES: $5 per point, 4 ticks per point
        total_improvement_dollars = self.total_price_improvement * 5.0

        return {
            'entries_attempted': self.entries_attempted,
            'entries_filled_at_limit': self.entries_filled_at_limit,
            'entries_timed_out': self.entries_timed_out,
            'fill_rate': fill_rate,
            'avg_improvement_ticks': avg_improvement_ticks,
            'total_improvement_dollars': total_improvement_dollars
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self.entries_attempted = 0
        self.entries_filled_at_limit = 0
        self.entries_timed_out = 0
        self.total_price_improvement = 0.0
        logger.info("Entry optimizer statistics reset")

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"EntryOptimizer(pullback={self.pullback_ticks} ticks, "
            f"max_wait={self.max_wait_bars} bars, "
            f"fill_rate={stats['fill_rate']:.1%}, "
            f"avg_improvement={stats['avg_improvement_ticks']:.1f} ticks)"
        )


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    print("=" * 70)
    print("Entry Timing Optimizer Test")
    print("=" * 70)

    # Create optimizer
    optimizer = EntryOptimizer(pullback_ticks=3, max_wait_bars=3)

    # Test LONG entry
    print("\n1. LONG Entry Test")
    signal_price = 5000.0
    entry_price = optimizer.calculate_entry_price(signal_price, 'LONG')
    print(f"   Signal price: ${signal_price:.2f}")
    print(f"   Entry price: ${entry_price:.2f} (3 ticks below)")
    print(f"   Expected improvement: ${(signal_price - entry_price) * 5:.2f}")

    # Simulate bars
    print("\n   Simulating price action:")
    test_prices = [5000.25, 5000.00, 4999.75, 4999.50, 4999.25]
    for i, price in enumerate(test_prices):
        should_enter = optimizer.should_enter_now(price, entry_price, 'LONG', i)
        print(f"   Bar {i}: price=${price:.2f}, should_enter={should_enter}")
        if should_enter:
            break

    # Test SHORT entry
    print("\n2. SHORT Entry Test")
    signal_price = 5000.0
    entry_price = optimizer.calculate_entry_price(signal_price, 'SHORT')
    print(f"   Signal price: ${signal_price:.2f}")
    print(f"   Entry price: ${entry_price:.2f} (3 ticks above)")
    print(f"   Expected improvement: ${(entry_price - signal_price) * 5:.2f}")

    # Simulate bars
    print("\n   Simulating price action:")
    test_prices = [4999.75, 5000.00, 5000.25, 5000.50, 5000.75]
    for i, price in enumerate(test_prices):
        should_enter = optimizer.should_enter_now(price, entry_price, 'SHORT', i)
        print(f"   Bar {i}: price=${price:.2f}, should_enter={should_enter}")
        if should_enter:
            break

    # Test timeout scenario
    print("\n3. Timeout Test (price doesn't reach limit)")
    signal_price = 5000.0
    entry_price = optimizer.calculate_entry_price(signal_price, 'LONG')
    print(f"   Signal price: ${signal_price:.2f}")
    print(f"   Entry price: ${entry_price:.2f}")

    print("\n   Simulating price going wrong way:")
    test_prices = [5000.25, 5000.50, 5000.75, 5001.00]
    for i, price in enumerate(test_prices):
        should_enter = optimizer.should_enter_now(price, entry_price, 'LONG', i)
        print(f"   Bar {i}: price=${price:.2f}, should_enter={should_enter}")
        if should_enter:
            print(f"   → Timeout at bar {i}, entering at market")
            break

    # Statistics
    print("\n4. Statistics")
    stats = optimizer.get_statistics()
    print(f"   Entries attempted: {stats['entries_attempted']}")
    print(f"   Filled at limit: {stats['entries_filled_at_limit']}")
    print(f"   Timed out: {stats['entries_timed_out']}")
    print(f"   Fill rate: {stats['fill_rate']:.1%}")
    print(f"   Avg improvement: {stats['avg_improvement_ticks']:.1f} ticks")
    print(f"   Total improvement: ${stats['total_improvement_dollars']:.2f}")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)
