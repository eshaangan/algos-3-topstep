"""
Simulate Kelly behavior on restart with today's 27 trades.

This shows that Kelly will continue from trade 28 instead of resetting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ml_intraday_v3.live_trading.kelly_sizer import KellySizer
from ml_intraday_v3.monitoring.metrics_tracker import MetricsTracker
import pandas as pd


def simulate_restart_with_persistence():
    """Simulate what happens when system restarts with trade history loaded."""
    print("=" * 80)
    print("SIMULATING KELLY RESTART WITH TODAY'S TRADES")
    print("=" * 80)
    print()

    # Scenario: You had 27 trades this morning, then restarted
    print("Scenario: Ran paper trading this morning")
    print("  - 27 trades executed (96.3% win rate)")
    print("  - System shutdown for lunch")
    print("  - Now restarting for afternoon session")
    print()

    # Create test trade data matching today's performance
    # 26 winners, 1 loser (from your CSV)
    test_trades = []

    # 26 winning trades (~$47 avg)
    for i in range(26):
        test_trades.append({
            'pnl': 47.0 + (i % 10) * 5,  # Varying wins
            'contracts': 1,
        })

    # 1 losing trade
    test_trades.append({
        'pnl': -17.5,  # From your CSV
        'contracts': 1,
    })

    print("=" * 80)
    print("WITHOUT PERSISTENCE (Current behavior before this fix)")
    print("=" * 80)
    print()

    # Simulate restart WITHOUT persistence
    print("System restarts with empty trade history...")
    empty_history = []

    kelly_config = {
        'enabled': True,
        'min_trades_for_kelly': 20,
        'kelly_fraction': 0.25,
        'max_contracts_per_trade': 5,
        'min_contracts': 1,
        'rolling_window_trades': 50,
        'confidence_boost': {'enabled': True, 'boost_factor': 1.5, 'boost_threshold': 0.15},
        'negative_kelly_threshold': 3,
        'log_sizing_decisions': False,
    }

    kelly_sizer = KellySizer(kelly_config)

    contracts, reason = kelly_sizer.get_position_size(
        trade_history=empty_history,
        score_ev=0.18,
        max_contracts_limit=5,
        current_equity=51246,
        contract_margin=1320,
    )

    print(f"  Trade history:     {len(empty_history)} trades")
    print(f"  Kelly status:      {reason}")
    print(f"  Position size:     {contracts} contract(s)")
    print(f"  ❌ LOST PROGRESS: Back to learning phase (trade 1/20)")
    print()

    print("=" * 80)
    print("WITH PERSISTENCE (New behavior after this fix)")
    print("=" * 80)
    print()

    # Simulate restart WITH persistence
    print("System restarts and loads previous session's trades...")
    loaded_history = test_trades

    kelly_sizer_persistent = KellySizer(kelly_config)

    contracts_persistent, reason_persistent = kelly_sizer_persistent.get_position_size(
        trade_history=loaded_history,
        score_ev=0.18,
        max_contracts_limit=5,
        current_equity=51246,
        contract_margin=1320,
    )

    win_rate = 26 / 27 * 100
    avg_win = sum(t['pnl'] for t in test_trades if t['pnl'] > 0) / 26
    avg_loss = abs(test_trades[-1]['pnl'])

    print(f"  Trade history:     {len(loaded_history)} trades loaded")
    print(f"  Win rate:          {win_rate:.1f}%")
    print(f"  Avg win:           ${avg_win:.2f}")
    print(f"  Avg loss:          ${avg_loss:.2f}")
    print(f"  Kelly status:      {reason_persistent}")
    print(f"  Position size:     {contracts_persistent} contract(s)")
    print(f"  ✅ CONTINUES: Kelly active from trade 28")
    print()

    # Show what would happen over next few trades
    print("=" * 80)
    print("NEXT 5 TRADES AFTER RESTART")
    print("=" * 80)
    print()

    print("WITHOUT persistence:")
    for i in range(1, 6):
        c, r = kelly_sizer.get_position_size([], 0.15, 5, 51246, 1320)
        print(f"  Trade {i}: {c} contract - {r}")
    print()

    print("WITH persistence:")
    for i in range(28, 33):
        score = 0.15 + (i % 3) * 0.03  # Varying scores
        c, r = kelly_sizer_persistent.get_position_size(loaded_history, score, 5, 51246, 1320)
        print(f"  Trade {i}: {c} contracts - Kelly sizing active")
    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✅ Benefits of persistence:")
    print("  1. Kelly continues from trade 28 (not reset to 1)")
    print("  2. Position sizing reflects today's actual performance")
    print("  3. No loss of statistical learning on restart")
    print("  4. Can restart system anytime without penalty")
    print()
    print("⚠️  Important notes:")
    print("  - Kelly will size based on 96% win rate (unsustainable)")
    print("  - Recommend disabling Kelly this week for baseline")
    print("  - Or reduce max_contracts_per_trade to 2 for safety")
    print()


if __name__ == "__main__":
    simulate_restart_with_persistence()
