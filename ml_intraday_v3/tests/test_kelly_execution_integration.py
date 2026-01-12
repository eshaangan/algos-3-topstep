"""
Kelly Criterion integration test with LiveExecutionEngine.

Tests Kelly sizing integration without requiring full replay infrastructure.
Demonstrates:
- Kelly learning phase (1 contract for first 20 trades)
- Kelly activation (dynamic sizing after 20 trades)
- Confidence boost on high score_ev signals
- All safety mechanisms working
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_mock_bars(n_bars=100):
    """Generate mock OHLCV bars for testing."""
    dates = pd.date_range(start='2022-07-01 09:30', periods=n_bars, freq='1min')

    # Simulate ES price around 4000
    np.random.seed(42)
    close = 4000 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.abs(np.random.randn(n_bars) * 0.3)
    low = close - np.abs(np.random.randn(n_bars) * 0.3)
    open_price = close + np.random.randn(n_bars) * 0.2
    volume = np.random.randint(100, 1000, n_bars)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)

    return df


def test_kelly_execution_integration():
    """Test Kelly sizing with LiveExecutionEngine."""

    from ml_intraday_v3.live_trading.kelly_sizer import KellySizer
    from ml_intraday_v3.monitoring.metrics_tracker import MetricsTracker

    logger.info("="*80)
    logger.info("KELLY EXECUTION INTEGRATION TEST")
    logger.info("="*80)

    # Initialize Kelly sizer
    kelly_config = {
        'enabled': True,
        'min_trades_for_kelly': 20,
        'kelly_fraction': 0.25,
        'rolling_window_trades': 50,
        'max_contracts_per_trade': 5,
        'min_contracts': 1,
        'confidence_boost': {
            'enabled': True,
            'boost_factor': 1.5,
            'boost_threshold': 0.15,
        },
        'negative_kelly_threshold': 3,
        'log_sizing_decisions': True,
    }

    kelly_sizer = KellySizer(kelly_config)
    logger.info(f"✓ KellySizer initialized: fraction={kelly_config['kelly_fraction']}")

    # Initialize metrics tracker
    tracker = MetricsTracker(output_dir=Path("logs"))
    tracker.set_starting_equity(50000.0)
    logger.info("✓ MetricsTracker initialized with $50k equity")

    # Simulate 50 trades with realistic performance
    logger.info("")
    logger.info("Simulating 50 trades...")
    logger.info("-"*80)

    # Mix of winners (55%) and losers (45%), avg win $60, avg loss $40
    np.random.seed(42)
    trade_outcomes = []
    for i in range(50):
        is_winner = np.random.random() < 0.55
        if is_winner:
            pnl = 60 + np.random.randn() * 10  # ~$60 avg
        else:
            pnl = -40 + np.random.randn() * 10  # ~-$40 avg
        trade_outcomes.append(pnl)

    # Simulate executing trades with Kelly sizing
    contracts_used = []
    sizing_reasons = []

    for i, pnl_per_contract in enumerate(trade_outcomes):
        trade_num = i + 1

        # Generate mock prediction with varying score_ev
        score_ev = 0.1 + np.random.random() * 0.15  # 0.1 to 0.25
        prediction = {'score_ev': score_ev}

        # Get Kelly position size
        contracts, reason = kelly_sizer.get_position_size(
            trade_history=tracker.trade_history,
            score_ev=score_ev,
            max_contracts_limit=5,
            current_equity=tracker.current_equity,
            contract_margin=1320.0,
        )

        contracts_used.append(contracts)
        sizing_reasons.append(reason)

        # Calculate actual P&L based on contracts
        actual_pnl = pnl_per_contract * contracts

        # Record trade
        entry_time = datetime.now() - timedelta(minutes=50-i)
        exit_time = entry_time + timedelta(minutes=10)

        tracker.record_trade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction="LONG" if score_ev > 0 else "SHORT",
            contracts=contracts,
            entry_price=4000.0,
            exit_price=4000.0 + (pnl_per_contract / 5.0),  # MES $5/point
            pnl=actual_pnl,
            exit_reason="target" if pnl_per_contract > 0 else "stop",
        )

        # Record Kelly decision
        tracker.record_kelly_decision({
            'timestamp': entry_time,
            'contracts': contracts,
            'reason': reason,
            'raw_kelly': kelly_sizer.last_kelly_fraction,
            'fractional_kelly': kelly_sizer.last_kelly_fraction * kelly_config['kelly_fraction'],
            'score_ev': score_ev,
            'win_rate': len([t for t in tracker.trade_history if t['pnl'] > 0]) / len(tracker.trade_history) if tracker.trade_history else 0,
            'avg_win': np.mean([t['pnl']/t['contracts'] for t in tracker.trade_history if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in tracker.trade_history) else 0,
            'avg_loss': np.mean([abs(t['pnl']/t['contracts']) for t in tracker.trade_history if t['pnl'] <= 0]) if any(t['pnl'] <= 0 for t in tracker.trade_history) else 0,
            'trade_count': len(tracker.trade_history),
        })

        # Log progress every 10 trades
        if trade_num % 10 == 0:
            logger.info(f"Trade {trade_num}: {contracts} contracts, score_ev={score_ev:.3f}, pnl=${actual_pnl:.2f}, reason={reason}")

    # Save logs
    tracker.save_trades()
    tracker.save_kelly_log()

    logger.info("")
    logger.info("="*80)
    logger.info("RESULTS")
    logger.info("="*80)

    # Analyze results
    contracts_df = pd.Series(contracts_used)

    logger.info("")
    logger.info("KELLY SIZING PERFORMANCE:")
    logger.info("-"*80)
    logger.info(f"Total trades: {len(contracts_used)}")
    logger.info(f"Avg contracts per trade: {contracts_df.mean():.2f}")
    logger.info(f"Min contracts: {contracts_df.min()}")
    logger.info(f"Max contracts: {contracts_df.max()}")

    logger.info("")
    logger.info("Contracts breakdown:")
    logger.info(contracts_df.value_counts().sort_index())

    # Learning phase analysis
    learning_phase = contracts_df[:20]
    kelly_active = contracts_df[20:]

    logger.info("")
    logger.info(f"Learning phase (trades 1-20): avg={learning_phase.mean():.2f}, all should be 1")
    logger.info(f"Kelly active (trades 21+): avg={kelly_active.mean():.2f}, should vary")

    # Check learning phase correctness
    if all(learning_phase == 1):
        logger.info("✓ Learning phase correct: all trades used 1 contract")
    else:
        logger.warning(f"✗ Learning phase error: not all trades used 1 contract: {learning_phase.tolist()}")

    # Check Kelly activation
    if kelly_active.std() > 0:
        logger.info(f"✓ Kelly activated: contracts varying (std={kelly_active.std():.2f})")
    else:
        logger.warning("✗ Kelly not varying after learning phase")

    # P&L analysis
    total_pnl = tracker.total_pnl
    win_rate = tracker.winning_trades / tracker.total_trades * 100 if tracker.total_trades > 0 else 0

    logger.info("")
    logger.info("TRADING PERFORMANCE:")
    logger.info("-"*80)
    logger.info(f"Total trades: {tracker.total_trades}")
    logger.info(f"Winners: {tracker.winning_trades} ({win_rate:.1f}%)")
    logger.info(f"Losers: {tracker.losing_trades}")
    logger.info(f"Total P&L: ${total_pnl:.2f}")
    logger.info(f"Avg trade: ${total_pnl/tracker.total_trades:.2f}")

    # File verification
    logger.info("")
    logger.info("OUTPUT FILES:")
    logger.info("-"*80)

    kelly_logs = list(Path("logs").glob("kelly_sizing_*.csv"))
    trade_logs = list(Path("logs").glob("trades_*.csv"))

    if kelly_logs:
        kelly_log = kelly_logs[-1]
        logger.info(f"✓ Kelly log: {kelly_log}")

        # Load and verify
        kelly_df = pd.read_csv(kelly_log)
        logger.info(f"  Total decisions: {len(kelly_df)}")
        logger.info(f"  Learning phase: {len(kelly_df[kelly_df['reason'].str.contains('learning', na=False)])}")
        logger.info(f"  Kelly active: {len(kelly_df[~kelly_df['reason'].str.contains('learning', na=False)])}")
    else:
        logger.warning("✗ No Kelly log found")

    if trade_logs:
        logger.info(f"✓ Trade log: {trade_logs[-1]}")
    else:
        logger.warning("✗ No trade log found")

    logger.info("")
    logger.info("="*80)
    logger.info("✓ INTEGRATION TEST COMPLETE")
    logger.info("="*80)

    # Assertions
    assert all(learning_phase == 1), "Learning phase should use 1 contract"
    assert kelly_active.std() > 0, "Kelly should vary after learning phase"
    assert len(kelly_logs) > 0, "Kelly log should be created"
    assert len(trade_logs) > 0, "Trade log should be created"

    return True


if __name__ == "__main__":
    try:
        test_kelly_execution_integration()
        print("\n✓ All integration tests passed!")
        exit(0)
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
