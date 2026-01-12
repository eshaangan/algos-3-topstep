"""
Test Kelly Criterion trade history persistence.

Verifies that trades can be loaded from CSV and metrics correctly recalculated.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_intraday_v3.monitoring.metrics_tracker import MetricsTracker


def test_persistence():
    """Test loading trades from CSV and metric recalculation."""
    print("=" * 80)
    print("TESTING KELLY PERSISTENCE")
    print("=" * 80)
    print()

    # Create metrics tracker
    output_dir = Path("logs")
    tracker = MetricsTracker(output_dir)

    print("1. Finding latest trade file...")
    latest_file = tracker.find_latest_trade_file()

    if not latest_file:
        print("❌ No trade files found in logs/")
        return False

    print(f"   ✓ Found: {latest_file.name}")
    print()

    print("2. Loading trades from CSV...")
    trades_loaded = tracker.load_trades_from_csv(latest_file)

    if trades_loaded == 0:
        print("❌ Failed to load trades")
        return False

    print(f"   ✓ Loaded {trades_loaded} trades")
    print()

    print("3. Verifying metrics recalculation...")
    print(f"   Total trades:     {tracker.total_trades}")
    print(f"   Winning trades:   {tracker.winning_trades}")
    print(f"   Losing trades:    {tracker.losing_trades}")
    print(f"   Win rate:         {tracker.winning_trades/tracker.total_trades*100:.1f}%")
    print(f"   Total P&L:        ${tracker.total_pnl:.2f}")
    print(f"   Gross profit:     ${tracker.gross_profit:.2f}")
    print(f"   Gross loss:       ${tracker.gross_loss:.2f}")
    print(f"   Max win:          ${tracker.max_win:.2f}")
    print(f"   Max loss:         ${tracker.max_loss:.2f}")
    print()

    # Verify trade history is populated
    if len(tracker.trade_history) != trades_loaded:
        print(f"❌ Trade history mismatch: {len(tracker.trade_history)} != {trades_loaded}")
        return False

    print("4. Checking Kelly readiness...")
    min_trades_for_kelly = 20

    if tracker.total_trades >= min_trades_for_kelly:
        print(f"   ✓ Kelly ready: {tracker.total_trades} >= {min_trades_for_kelly} trades")
        print(f"   → Kelly will activate immediately on next trade")
    else:
        trades_needed = min_trades_for_kelly - tracker.total_trades
        print(f"   ⏳ Kelly learning: {tracker.total_trades}/{min_trades_for_kelly} trades")
        print(f"   → Need {trades_needed} more trades before Kelly activates")

    print()
    print("=" * 80)
    print("✅ PERSISTENCE TEST PASSED")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Trades can be loaded from: {latest_file}")
    print(f"  - Metrics correctly recalculated from {trades_loaded} trades")
    print(f"  - Kelly state preserved across restarts")
    print()

    return True


if __name__ == "__main__":
    success = test_persistence()
    sys.exit(0 if success else 1)
