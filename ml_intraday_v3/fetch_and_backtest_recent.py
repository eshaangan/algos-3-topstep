#!/usr/bin/env python3
"""
Fetch recent MES data from Databento and run direction change backtests.

This script:
1. Fetches last 2-3 weeks of MES 1-minute bars from Databento
2. Prepares data in format expected by replay_session()
3. Runs three backtests with different direction_change thresholds:
   - Baseline: disabled (natural stops/targets only)
   - High-confidence: threshold=0.20 (recommended fix)
   - Aggressive: threshold=0.10 (old problematic behavior)
4. Compares results and analyzes exit reason distribution
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logging.info(f"Loaded environment from {env_path}")

from ml_intraday_v3.live_trading.data_fetcher import LiveDataFetcher
from ml_intraday_v3.live_trading.replay import replay_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch_recent_data(
    symbol: str,
    days_back: int,
    output_dir: Path,
    bar_size: str = "1m",
) -> Path:
    """
    Fetch recent historical data from Databento.

    Args:
        symbol: Symbol to fetch (e.g., "MES")
        days_back: Number of days to fetch
        output_dir: Where to save the data
        bar_size: Bar size (e.g., "1m", "5m")

    Returns:
        Path to the run directory with fetched data
    """
    logger.info("=" * 80)
    logger.info(f"Fetching {days_back} days of {symbol} {bar_size} data from Databento")
    logger.info("=" * 80)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    logger.info(f"Date range: {start_str} to {end_str}")

    # Convert symbol to Databento continuous format if needed
    # For continuous futures: ROOT.ROLL_RULE.RANK
    # e.g., MES -> MES.c.0 (front month, calendar roll)
    databento_symbol = symbol
    if not '.' in symbol:
        # Assume it's a futures root, convert to continuous front month
        databento_symbol = f"{symbol}.c.0"
        logger.info(f"Converting symbol '{symbol}' to Databento format: '{databento_symbol}'")

    # Create data fetcher
    fetcher = LiveDataFetcher(
        symbol=databento_symbol,
        bar_size=bar_size,
        lookback_bars=100,
    )

    # Fetch historical bars
    bars_df = fetcher.fetch_historical(start_date=start_str, end_date=end_str)

    logger.info(f"Fetched {len(bars_df):,} bars")
    logger.info(f"Date range: {bars_df.index[0]} to {bars_df.index[-1]}")

    # Create run directory structure
    run_name = f"recent_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_name
    bar_dir = run_dir / f"bar_size={bar_size}"
    bar_dir.mkdir(parents=True, exist_ok=True)

    # Save bars
    bars_path = bar_dir / "bars.parquet"
    bars_df.to_parquet(bars_path)
    logger.info(f"Saved bars to: {bars_path}")

    # Create label_schema.json (needed by replay)
    # Use default values matching the baseline model
    label_schema = {
        "stop_multiple": 1.0,
        "target_multiple": 2.0,
        "atr_period": 14,
        "atr_bar_size": "1m",
    }
    schema_path = bar_dir / "label_schema.json"
    with open(schema_path, "w") as f:
        json.dump(label_schema, f, indent=2)
    logger.info(f"Saved label schema to: {schema_path}")

    # Create walkforward directory with model bundle
    # Copy from models/saved/model_bundle.pkl
    wf_dir = run_dir / "walkforward" / f"bar_size={bar_size}" / "window_0"
    wf_dir.mkdir(parents=True, exist_ok=True)

    # Check if model bundle exists
    model_bundle_src = Path("ml_intraday_v3/models/saved/model_bundle.pkl")
    if model_bundle_src.exists():
        import shutil
        model_bundle_dst = wf_dir / "model_bundle.pkl"
        shutil.copy(model_bundle_src, model_bundle_dst)
        logger.info(f"Copied model bundle to: {model_bundle_dst}")
    else:
        logger.warning(f"Model bundle not found at {model_bundle_src}")
        logger.warning("You'll need to specify model_bundle_path manually")

    logger.info("=" * 80)
    logger.info(f"Data preparation complete: {run_dir}")
    logger.info("=" * 80)

    return run_dir


def modify_live_config(
    config_path: Path,
    enabled: bool,
    threshold: float,
    output_path: Path,
) -> None:
    """Modify live_trading.yaml with specific direction_change settings."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure direction_change section exists
    if 'direction_change' not in config:
        config['direction_change'] = {}

    config['direction_change']['enabled'] = enabled
    config['direction_change']['high_confidence_threshold'] = threshold

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_single_backtest(
    name: str,
    run_dir: Path,
    config_dir: Path,
    bar_size: str,
    start_date: str,
    end_date: str,
    output_base: Path,
    dc_enabled: bool,
    dc_threshold: float,
) -> dict:
    """
    Run a single backtest with specific direction_change settings.

    Returns:
        Dict with summary statistics
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"BACKTEST: {name}")
    logger.info(f"  Direction change enabled: {dc_enabled}")
    logger.info(f"  High-confidence threshold: {dc_threshold:.2f}")
    logger.info("=" * 80)

    # Create temporary config directory
    import shutil
    temp_config_dir = output_base / name / "configs"
    temp_config_dir.mkdir(parents=True, exist_ok=True)

    # Copy configs
    for config_file in ["risk.yaml", "execution_spec.yaml", "features.yaml"]:
        src = config_dir / config_file
        dst = temp_config_dir / config_file
        if src.exists():
            shutil.copy(src, dst)

    # Modify live_trading.yaml
    modify_live_config(
        config_dir / "live_trading.yaml",
        dc_enabled,
        dc_threshold,
        temp_config_dir / "live_trading.yaml",
    )

    # Create output directory
    output_dir = output_base / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run replay
    try:
        artifacts = replay_session(
            run_dir=run_dir,
            config_dir=temp_config_dir,
            bar_size=bar_size,
            start=start_date,
            end=end_date,
            output_dir=output_dir,
        )

        # Save results
        artifacts.metrics.to_csv(output_dir / "metrics.csv", index=False)
        artifacts.trade_log.to_csv(output_dir / "trade_log.csv", index=False)
        artifacts.closed_positions.to_csv(output_dir / "closed_positions.csv", index=False)

        # Calculate summary statistics
        total_trades = len(artifacts.closed_positions)
        if total_trades == 0:
            logger.warning("No trades executed in backtest")
            return {
                'name': name,
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'profit_factor': 0,
                'exit_reasons': {},
            }

        wins = (artifacts.closed_positions['pnl_usd'] > 0).sum()
        win_rate = wins / total_trades * 100
        total_pnl = artifacts.closed_positions['pnl_usd'].sum()

        # Calculate profit factor
        gross_profit = artifacts.closed_positions[
            artifacts.closed_positions['pnl_usd'] > 0
        ]['pnl_usd'].sum()
        gross_loss = abs(artifacts.closed_positions[
            artifacts.closed_positions['pnl_usd'] < 0
        ]['pnl_usd'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Exit reason distribution
        exit_reasons = artifacts.closed_positions['exit_reason'].value_counts().to_dict()
        exit_pcts = {
            reason: count / total_trades * 100
            for reason, count in exit_reasons.items()
        }

        # Print summary
        logger.info("")
        logger.info(f"RESULTS: {name}")
        logger.info(f"  Total trades: {total_trades}")
        logger.info(f"  Win rate: {win_rate:.1f}%")
        logger.info(f"  Total P&L: ${total_pnl:,.2f}")
        logger.info(f"  Profit factor: {profit_factor:.2f}")
        logger.info("  Exit reasons:")
        for reason, pct in sorted(exit_pcts.items(), key=lambda x: -x[1]):
            count = exit_reasons[reason]
            logger.info(f"    {reason}: {count} ({pct:.1f}%)")
        logger.info("=" * 80)

        return {
            'name': name,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'exit_reasons': exit_reasons,
            'exit_pcts': exit_pcts,
        }

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'name': name,
            'error': str(e),
        }


def compare_results(results: list[dict], output_dir: Path) -> None:
    """Compare backtest results and print analysis."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARATIVE ANALYSIS")
    logger.info("=" * 80)

    # Create comparison table
    comparison = []
    for r in results:
        if 'error' in r:
            continue
        comparison.append({
            'Backtest': r['name'],
            'Total Trades': r['total_trades'],
            'Win Rate': f"{r['win_rate']:.1f}%",
            'Total P&L': f"${r['total_pnl']:,.2f}",
            'Profit Factor': f"{r['profit_factor']:.2f}",
            'Target %': f"{r['exit_pcts'].get('target_hit', 0):.1f}%",
            'Stop %': f"{r['exit_pcts'].get('stop_hit', 0):.1f}%",
            'Direction Change %': f"{sum(r['exit_pcts'].get(k, 0) for k in r['exit_pcts'] if 'direction_change' in k):.1f}%",
        })

    df = pd.DataFrame(comparison)
    logger.info("\n" + df.to_string(index=False))

    # Save comparison
    df.to_csv(output_dir / "comparison.csv", index=False)
    logger.info(f"\nComparison saved to: {output_dir / 'comparison.csv'}")

    # Analysis
    logger.info("")
    logger.info("KEY INSIGHTS:")

    baseline = next((r for r in results if r['name'] == 'baseline'), None)
    high_conf = next((r for r in results if r['name'] == 'high_confidence'), None)
    aggressive = next((r for r in results if r['name'] == 'aggressive'), None)

    if baseline and high_conf:
        pnl_diff = high_conf['total_pnl'] - baseline['total_pnl']
        pnl_pct = (pnl_diff / abs(baseline['total_pnl']) * 100) if baseline['total_pnl'] != 0 else 0

        logger.info(f"\n1. High-Confidence vs Baseline:")
        logger.info(f"   P&L difference: ${pnl_diff:,.2f} ({pnl_pct:+.1f}%)")

        baseline_dc = sum(baseline['exit_pcts'].get(k, 0) for k in baseline['exit_pcts'] if 'direction_change' in k)
        highconf_dc = sum(high_conf['exit_pcts'].get(k, 0) for k in high_conf['exit_pcts'] if 'direction_change' in k)
        logger.info(f"   Direction change exits: {baseline_dc:.1f}% → {highconf_dc:.1f}%")

    if aggressive and high_conf:
        pnl_diff = high_conf['total_pnl'] - aggressive['total_pnl']
        pnl_pct = (pnl_diff / abs(aggressive['total_pnl']) * 100) if aggressive['total_pnl'] != 0 else 0

        logger.info(f"\n2. High-Confidence vs Aggressive (old behavior):")
        logger.info(f"   P&L difference: ${pnl_diff:,.2f} ({pnl_pct:+.1f}%)")

        aggressive_dc = sum(aggressive['exit_pcts'].get(k, 0) for k in aggressive['exit_pcts'] if 'direction_change' in k)
        highconf_dc = sum(high_conf['exit_pcts'].get(k, 0) for k in high_conf['exit_pcts'] if 'direction_change' in k)
        logger.info(f"   Direction change exits: {aggressive_dc:.1f}% → {highconf_dc:.1f}%")

    logger.info("")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Fetch recent data and run direction change backtests")
    parser.add_argument("--symbol", default="MES", help="Symbol to fetch (default: MES)")
    parser.add_argument("--days", type=int, default=21, help="Days to fetch (default: 21)")
    parser.add_argument("--bar-size", default="1m", help="Bar size (default: 1m)")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetch, use existing run_dir")
    parser.add_argument("--run-dir", type=Path, help="Existing run directory (if --skip-fetch)")
    parser.add_argument("--start-date", help="Backtest start date (YYYY-MM-DD), default: earliest in data")
    parser.add_argument("--end-date", help="Backtest end date (YYYY-MM-DD), default: latest in data")
    parser.add_argument("--output-dir", type=Path, default=Path("ml_intraday_v3/backtest_results"),
                        help="Output directory for results")

    args = parser.parse_args()

    # Set working directory
    os.chdir(project_root)

    # Fetch or use existing data
    if args.skip_fetch:
        if not args.run_dir:
            logger.error("--run-dir required when --skip-fetch is used")
            return 1
        run_dir = args.run_dir
        logger.info(f"Using existing data: {run_dir}")
    else:
        run_dir = fetch_recent_data(
            symbol=args.symbol,
            days_back=args.days,
            output_dir=Path("ml_intraday_v3/runs"),
            bar_size=args.bar_size,
        )

    # Determine date range
    bars_path = run_dir / f"bar_size={args.bar_size}" / "bars.parquet"
    bars_df = pd.read_parquet(bars_path)
    start_date = args.start_date or bars_df.index[0].strftime("%Y-%m-%d")
    end_date = args.end_date or bars_df.index[-1].strftime("%Y-%m-%d")

    logger.info(f"Backtest date range: {start_date} to {end_date}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = args.output_dir / f"direction_change_validation_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)

    config_dir = Path("ml_intraday_v3/configs")

    # Run three backtests
    results = []

    # 1. Baseline: No direction change
    results.append(run_single_backtest(
        name="baseline",
        run_dir=run_dir,
        config_dir=config_dir,
        bar_size=args.bar_size,
        start_date=start_date,
        end_date=end_date,
        output_base=output_base,
        dc_enabled=False,
        dc_threshold=0.20,  # Doesn't matter when disabled
    ))

    # 2. High-confidence: threshold=0.20 (RECOMMENDED)
    results.append(run_single_backtest(
        name="high_confidence",
        run_dir=run_dir,
        config_dir=config_dir,
        bar_size=args.bar_size,
        start_date=start_date,
        end_date=end_date,
        output_base=output_base,
        dc_enabled=True,
        dc_threshold=0.20,
    ))

    # 3. Aggressive: threshold=0.10 (old problematic behavior)
    results.append(run_single_backtest(
        name="aggressive",
        run_dir=run_dir,
        config_dir=config_dir,
        bar_size=args.bar_size,
        start_date=start_date,
        end_date=end_date,
        output_base=output_base,
        dc_enabled=True,
        dc_threshold=0.10,
    ))

    # Compare results
    compare_results(results, output_base)

    logger.info("")
    logger.info(f"All results saved to: {output_base}")
    logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
