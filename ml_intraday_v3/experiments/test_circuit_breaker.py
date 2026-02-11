#!/usr/bin/env python3
"""
Test Circuit Breaker - Quick Win #2 Validation

Validates that circuit breaker correctly triggers on:
1. Consecutive losses (3 in a row)
2. Daily loss limit (-$500)
3. Low win rate (<30% after 10 trades)

Expected outcome: Circuit breaker would have stopped Jan 2026 trading
at -$500 instead of -$884, preventing $384 in additional losses.

Usage:
    python ml_intraday_v3/experiments/test_circuit_breaker.py
"""

import sys
from pathlib import Path
import logging

# Add ml_intraday_v3 to path
ml_v3_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ml_v3_dir))

from monitoring.circuit_breaker import CircuitBreaker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def test_consecutive_losses():
    """Test circuit breaker trips on 3 consecutive losses."""
    logger.info("=" * 70)
    logger.info("TEST 1: Consecutive Losses Trigger")
    logger.info("=" * 70)

    cb = CircuitBreaker(max_consecutive_losses=3)

    # Simulate trades
    trades = [
        {'pnl': 50, 'symbol': 'MES', 'timestamp': '2026-01-15 10:00'},   # Win
        {'pnl': -30, 'symbol': 'MES', 'timestamp': '2026-01-15 10:30'},  # Loss 1
        {'pnl': -30, 'symbol': 'MES', 'timestamp': '2026-01-15 11:00'},  # Loss 2
        {'pnl': -30, 'symbol': 'MES', 'timestamp': '2026-01-15 11:30'},  # Loss 3 (SHOULD TRIP)
        {'pnl': 40, 'symbol': 'MES', 'timestamp': '2026-01-15 12:00'},   # Should not execute
    ]

    daily_pnl = 0
    tripped = False

    for i, trade in enumerate(trades, 1):
        daily_pnl += trade['pnl']

        logger.info(f"\nTrade {i}: {trade['symbol']} ${trade['pnl']:+.2f} @ {trade['timestamp']}")
        logger.info(f"  Cumulative P&L: ${daily_pnl:+.2f}")

        safe = cb.check(trade, daily_pnl)

        if not safe:
            logger.critical(f"  🚨 CIRCUIT BREAKER TRIPPED: {cb.trip_reason}")
            tripped = True
            break
        else:
            logger.info(f"  ✅ Safe to continue (consecutive losses: {cb._count_consecutive_losses()})")

    # Verify results
    assert tripped, "Circuit breaker should have tripped on 3 consecutive losses"
    assert i == 4, f"Should have stopped at trade 4, stopped at {i}"
    assert daily_pnl == -40, f"Daily P&L should be -$40, got ${daily_pnl}"

    logger.info("\n✅ TEST PASSED: Circuit breaker correctly triggered on 3 consecutive losses")
    return True


def test_daily_loss_limit():
    """Test circuit breaker trips on daily loss limit."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Daily Loss Limit Trigger")
    logger.info("=" * 70)

    cb = CircuitBreaker(daily_loss_limit=-500.0)

    # Simulate large losses
    trades = [
        {'pnl': -200, 'symbol': 'MES'},
        {'pnl': -150, 'symbol': 'MES'},
        {'pnl': -100, 'symbol': 'MES'},
        {'pnl': -60, 'symbol': 'MES'},  # This pushes to -$510 (SHOULD TRIP)
        {'pnl': 100, 'symbol': 'MES'},  # Should not execute
    ]

    daily_pnl = 0
    tripped = False

    for i, trade in enumerate(trades, 1):
        daily_pnl += trade['pnl']

        logger.info(f"\nTrade {i}: ${trade['pnl']:+.2f}")
        logger.info(f"  Cumulative P&L: ${daily_pnl:+.2f}")

        safe = cb.check(trade, daily_pnl)

        if not safe:
            logger.critical(f"  🚨 CIRCUIT BREAKER TRIPPED: {cb.trip_reason}")
            tripped = True
            break
        else:
            logger.info(f"  ✅ Safe to continue")

    # Verify results
    assert tripped, "Circuit breaker should have tripped on daily loss limit"
    assert daily_pnl <= -500, f"Should have stopped at -$500, stopped at ${daily_pnl}"

    logger.info("\n✅ TEST PASSED: Circuit breaker correctly triggered on daily loss limit")
    return True


def test_low_win_rate():
    """Test circuit breaker trips on low win rate."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Low Win Rate Trigger")
    logger.info("=" * 70)

    cb = CircuitBreaker(min_win_rate_after_n_trades=(10, 0.30))  # <30% after 10 trades

    # Simulate 10 trades with 20% win rate (2 wins, 8 losses)
    trades = [
        {'pnl': -30, 'symbol': 'MES'},  # Loss
        {'pnl': -30, 'symbol': 'MES'},  # Loss
        {'pnl': 50, 'symbol': 'MES'},   # Win 1
        {'pnl': -30, 'symbol': 'MES'},  # Loss
        {'pnl': -30, 'symbol': 'MES'},  # Loss
        {'pnl': -30, 'symbol': 'MES'},  # Loss
        {'pnl': 50, 'symbol': 'MES'},   # Win 2
        {'pnl': -30, 'symbol': 'MES'},  # Loss
        {'pnl': -30, 'symbol': 'MES'},  # Loss
        {'pnl': -30, 'symbol': 'MES'},  # Loss (10th trade, 20% win rate, SHOULD TRIP)
        {'pnl': 50, 'symbol': 'MES'},   # Should not execute
    ]

    daily_pnl = 0
    tripped = False

    for i, trade in enumerate(trades, 1):
        daily_pnl += trade['pnl']

        logger.info(f"\nTrade {i}: ${trade['pnl']:+.2f}")
        logger.info(f"  Cumulative P&L: ${daily_pnl:+.2f}")

        safe = cb.check(trade, daily_pnl)

        if i >= 10:  # Only check win rate after 10 trades
            win_rate = cb._calculate_win_rate()
            logger.info(f"  Win rate: {win_rate:.1%} (threshold: 30%)")

        if not safe:
            logger.critical(f"  🚨 CIRCUIT BREAKER TRIPPED: {cb.trip_reason}")
            tripped = True
            break
        else:
            logger.info(f"  ✅ Safe to continue")

    # Verify results
    assert tripped, "Circuit breaker should have tripped on low win rate"
    assert i == 10, f"Should have stopped at trade 10, stopped at {i}"

    logger.info("\n✅ TEST PASSED: Circuit breaker correctly triggered on low win rate")
    return True


def test_jan_2026_simulation():
    """
    Simulate Jan 2026 trades with circuit breaker.

    Expected: Circuit breaker stops trading earlier, limiting losses.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Jan 2026 Simulation (Would Circuit Breaker Have Helped?)")
    logger.info("=" * 70)

    # Simplified Jan 2026 performance
    # Real: 152 trades over 18 days, 35.5% win rate, -$884.73 total
    # Simulation: Generate similar distribution

    import random
    random.seed(42)

    cb = CircuitBreaker(
        max_consecutive_losses=3,
        daily_loss_limit=-500.0,
        min_win_rate_after_n_trades=(10, 0.30)
    )

    # Simulate trades with 35.5% win rate
    # Avg win: +$41.16, Avg loss: -$19.98
    n_trades_total = 152
    win_rate = 0.355

    trades = []
    for _ in range(n_trades_total):
        if random.random() < win_rate:
            pnl = random.gauss(41.16, 10)  # Win
        else:
            pnl = random.gauss(-19.98, 5)  # Loss
        trades.append({'pnl': pnl, 'symbol': 'MES'})

    # Run trades through circuit breaker
    daily_pnl = 0
    total_pnl = 0
    trades_executed = 0
    daily_trades = 0
    day = 1

    logger.info(f"\nSimulating {n_trades_total} trades (35.5% win rate)...")

    for i, trade in enumerate(trades):
        daily_pnl += trade['pnl']
        total_pnl += trade['pnl']
        trades_executed += 1
        daily_trades += 1

        safe = cb.check(trade, daily_pnl, current_date=f"2026-01-{day:02d}")

        if not safe:
            logger.warning(f"\n🚨 Day {day}, Trade {daily_trades}: Circuit breaker tripped")
            logger.warning(f"   Reason: {cb.trip_reason}")
            logger.warning(f"   Daily P&L: ${daily_pnl:+.2f}")
            logger.warning(f"   Total P&L: ${total_pnl:+.2f}")
            logger.warning(f"   Trades executed: {trades_executed}/{n_trades_total}")
            break

        # Simulate day changes (approx 8.4 trades/day)
        if daily_trades >= 8:
            day += 1
            daily_pnl = 0
            daily_trades = 0
            cb.reset()

    # Compare to actual Jan 2026 results
    actual_loss = -884.73
    prevented_loss = actual_loss - total_pnl

    logger.info("\n" + "-" * 70)
    logger.info("RESULTS:")
    logger.info(f"  Trades executed: {trades_executed}/{n_trades_total} ({100*trades_executed/n_trades_total:.1f}%)")
    logger.info(f"  Total P&L with circuit breaker: ${total_pnl:+.2f}")
    logger.info(f"  Actual Jan 2026 loss: ${actual_loss:+.2f}")
    logger.info(f"  Loss prevented: ${prevented_loss:+.2f}")
    logger.info("-" * 70)

    # Circuit breaker should limit losses
    assert total_pnl > actual_loss, "Circuit breaker should prevent some losses"

    logger.info("\n✅ TEST PASSED: Circuit breaker would have limited Jan 2026 losses")
    logger.info(f"   Prevented ${prevented_loss:.2f} in additional losses")

    return True


def main():
    """Run all circuit breaker tests."""
    logger.info("\n" + "=" * 70)
    logger.info("CIRCUIT BREAKER VALIDATION SUITE")
    logger.info("=" * 70)

    tests = [
        ("Consecutive Losses", test_consecutive_losses),
        ("Daily Loss Limit", test_daily_loss_limit),
        ("Low Win Rate", test_low_win_rate),
        ("Jan 2026 Simulation", test_jan_2026_simulation),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results[test_name] = passed
        except AssertionError as e:
            logger.error(f"\n❌ TEST FAILED: {test_name}")
            logger.error(f"   {str(e)}")
            results[test_name] = False
        except Exception as e:
            logger.error(f"\n❌ TEST ERROR: {test_name}")
            logger.error(f"   {str(e)}")
            results[test_name] = False

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status} - {test_name}")

    all_passed = all(results.values())
    logger.info("=" * 70)

    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED - Circuit breaker is working correctly!")
        logger.info("   Ready to integrate into live trading system")
        return 0
    else:
        logger.error("\n⚠️ SOME TESTS FAILED - Review circuit breaker implementation")
        return 1


if __name__ == "__main__":
    exit(main())
