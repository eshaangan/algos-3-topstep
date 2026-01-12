"""
Replay test with Kelly Criterion enabled.

Tests Kelly sizing with historical data to verify:
- Learning phase (first 20 trades use 1 contract)
- Kelly activation (trades 21+ use dynamic sizing)
- Sizing decisions are logged
- Performance comparison
"""

import sys
from pathlib import Path

# Add project root to path (not ml_intraday_v3)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
import pandas as pd
from ml_intraday_v3.live_trading.replay import replay_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_kelly_replay_test():
    """Run replay test with Kelly enabled."""

    logger.info("="*80)
    logger.info("KELLY CRITERION REPLAY TEST")
    logger.info("="*80)

    # Configuration
    run_dir = Path("/Users/eshaanganguly/Documents/projects/algos 3 topstep/runs/mid2022_20251227_043831_90e86589")
    config_dir = Path("/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/configs")
    bar_size = "1m"

    # Test period: 3 days of data (July 2022, when data starts)
    start = "2022-07-01"
    end = "2022-07-04"

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config directory: {config_dir}")
    logger.info(f"Bar size: {bar_size}")
    logger.info(f"Test period: {start} to {end}")
    logger.info("")

    # Run replay
    logger.info("Starting replay with Kelly enabled...")
    logger.info("-" * 80)

    artifacts = replay_session(
        run_dir=run_dir,
        config_dir=config_dir,
        bar_size=bar_size,
        start=start,
        end=end,
        max_bars=None,
        model_bundle_path=None,  # Auto-detect latest
        output_dir=Path("logs"),
    )

    logger.info("")
    logger.info("="*80)
    logger.info("REPLAY COMPLETE")
    logger.info("="*80)

    # Analyze results
    logger.info("")
    logger.info("RESULTS SUMMARY:")
    logger.info("-" * 80)

    # Trade statistics
    closed = artifacts.closed_positions
    if len(closed) > 0:
        total_trades = len(closed)
        winners = len(closed[closed['pnl_usd'] > 0])
        losers = len(closed[closed['pnl_usd'] <= 0])
        win_rate = winners / total_trades * 100

        total_pnl = closed['pnl_usd'].sum()
        avg_win = closed[closed['pnl_usd'] > 0]['pnl_usd'].mean() if winners > 0 else 0
        avg_loss = closed[closed['pnl_usd'] <= 0]['pnl_usd'].mean() if losers > 0 else 0

        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Winners: {winners} ({win_rate:.1f}%)")
        logger.info(f"Losers: {losers}")
        logger.info(f"Total P&L: ${total_pnl:.2f}")
        logger.info(f"Avg win: ${avg_win:.2f}")
        logger.info(f"Avg loss: ${avg_loss:.2f}")

        # Contract sizing analysis
        if 'contracts' in closed.columns:
            logger.info("")
            logger.info("KELLY SIZING ANALYSIS:")
            logger.info("-" * 80)
            logger.info(f"Avg contracts per trade: {closed['contracts'].mean():.2f}")
            logger.info(f"Min contracts: {closed['contracts'].min()}")
            logger.info(f"Max contracts: {closed['contracts'].max()}")
            logger.info("")
            logger.info("Contracts breakdown:")
            logger.info(closed['contracts'].value_counts().sort_index())

            # Learning phase check
            first_20 = closed.head(20)
            after_20 = closed.iloc[20:] if len(closed) > 20 else pd.DataFrame()

            logger.info("")
            logger.info(f"First 20 trades (learning phase): avg contracts = {first_20['contracts'].mean():.2f}")
            if len(after_20) > 0:
                logger.info(f"After trade 20 (Kelly active): avg contracts = {after_20['contracts'].mean():.2f}")
    else:
        logger.info("No closed trades found")

    # Check if Kelly log exists
    logger.info("")
    logger.info("OUTPUT FILES:")
    logger.info("-" * 80)
    log_dir = Path("logs")

    kelly_logs = list(log_dir.glob("kelly_sizing_*.csv"))
    if kelly_logs:
        kelly_log_path = kelly_logs[-1]  # Latest
        logger.info(f"✓ Kelly log: {kelly_log_path}")

        # Load and analyze Kelly log
        kelly_df = pd.read_csv(kelly_log_path)
        logger.info(f"  Total sizing decisions: {len(kelly_df)}")

        if len(kelly_df) > 0:
            # Check learning phase
            learning = kelly_df[kelly_df['reason'].str.contains('learning_phase', na=False)]
            active = kelly_df[~kelly_df['reason'].str.contains('learning_phase', na=False)]

            logger.info(f"  Learning phase decisions: {len(learning)}")
            logger.info(f"  Kelly active decisions: {len(active)}")

            if len(active) > 0:
                logger.info(f"  Avg Kelly fraction (active): {active['fractional_kelly'].mean():.4f}")
                logger.info(f"  Avg contracts (active): {active['contracts'].mean():.2f}")
    else:
        logger.info("✗ No Kelly log found")

    trade_logs = list(log_dir.glob("trades_*.csv"))
    if trade_logs:
        logger.info(f"✓ Trade log: {trade_logs[-1]}")

    metrics_logs = list(log_dir.glob("metrics_*.csv"))
    if metrics_logs:
        logger.info(f"✓ Metrics log: {metrics_logs[-1]}")

    signal_logs = list(log_dir.glob("signals_*.csv"))
    if signal_logs:
        logger.info(f"✓ Signal log: {signal_logs[-1]}")

    logger.info("")
    logger.info("="*80)
    logger.info("✓ KELLY REPLAY TEST COMPLETE")
    logger.info("="*80)

    return artifacts


if __name__ == "__main__":
    try:
        artifacts = run_kelly_replay_test()
        print("\n✓ Test completed successfully")
        exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
