#!/usr/bin/env python3
"""
Run three backtests to validate direction change fix:
1. Baseline: No direction change logic
2. High-confidence: direction_change.high_confidence_threshold = 0.20 (RECOMMENDED)
3. Aggressive: direction_change.high_confidence_threshold = 0.10 (old behavior)

This uses the live trading stack (replay.py) which includes the direction change logic.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml
import shutil
import os

# Add project root to path (must be done before any ml_intraday_v3 imports)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Change working directory to project root for imports to work
os.chdir(project_root)

from ml_intraday_v3.live_trading.replay import replay_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def modify_config(config_path: Path, enabled: bool, threshold: float, output_path: Path):
    """
    Modify live_trading.yaml with specific direction_change settings.

    Args:
        config_path: Original config path
        enabled: Enable/disable direction change
        threshold: High-confidence threshold value
        output_path: Where to write modified config
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure direction_change section exists
    if 'direction_change' not in config:
        config['direction_change'] = {}

    config['direction_change']['enabled'] = enabled
    config['direction_change']['high_confidence_threshold'] = threshold

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Modified config: enabled={enabled}, threshold={threshold:.2f} -> {output_path}")


def run_backtest(
    name: str,
    run_dir: Path,
    config_dir: Path,
    bar_size: str,
    start_date: str,
    end_date: str,
    output_base: Path,
    dc_enabled: bool,
    dc_threshold: float,
):
    """
    Run a single backtest with specific direction change settings.

    Args:
        name: Backtest name (e.g., "baseline", "high_confidence", "aggressive")
        run_dir: Run directory with trained models
        config_dir: Config directory
        bar_size: Bar size (e.g., "1m", "5m")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_base: Base output directory
        dc_enabled: Enable direction change logic
        dc_threshold: High-confidence threshold
    """
    logger.info("=" * 80)
    logger.info(f"Running backtest: {name}")
    logger.info(f"  Direction change enabled: {dc_enabled}")
    logger.info(f"  High-confidence threshold: {dc_threshold:.2f}")
    logger.info("=" * 80)

    # Create temporary config directory
    temp_config_dir = output_base / name / "configs"
    temp_config_dir.mkdir(parents=True, exist_ok=True)

    # Copy configs to temp directory
    for config_file in ["risk.yaml", "execution_spec.yaml", "features.yaml"]:
        src = config_dir / config_file
        dst = temp_config_dir / config_file
        if src.exists():
            shutil.copy(src, dst)

    # Modify live_trading.yaml with direction change settings
    original_live_config = config_dir / "live_trading.yaml"
    temp_live_config = temp_config_dir / "live_trading.yaml"
    modify_config(original_live_config, dc_enabled, dc_threshold, temp_live_config)

    # Create output directory
    output_dir = output_base / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run replay
    logger.info(f"Starting replay: {start_date} to {end_date}")
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

        # Print summary
        total_trades = len(artifacts.closed_positions)
        if total_trades > 0:
            wins = (artifacts.closed_positions['pnl_usd'] > 0).sum()
            win_rate = wins / total_trades * 100
            total_pnl = artifacts.closed_positions['pnl_usd'].sum()

            # Exit reason distribution
            exit_reasons = artifacts.closed_positions['exit_reason'].value_counts()

            logger.info("=" * 80)
            logger.info(f"RESULTS: {name}")
            logger.info(f"  Total trades: {total_trades}")
            logger.info(f"  Win rate: {win_rate:.1f}%")
            logger.info(f"  Total P&L: ${total_pnl:,.2f}")
            logger.info("  Exit reasons:")
            for reason, count in exit_reasons.items():
                pct = count / total_trades * 100
                logger.info(f"    {reason}: {count} ({pct:.1f}%)")
            logger.info("=" * 80)
        else:
            logger.warning(f"No trades generated for {name}")

        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Backtest {name} failed: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run direction change validation backtests"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/run_20251224_123456"),
        help="Run directory with trained models",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("ml_intraday_v3/configs"),
        help="Config directory",
    )
    parser.add_argument(
        "--bar-size",
        type=str,
        default="1m",
        help="Bar size (e.g., 1m, 5m)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-12-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-01-20",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/direction_change_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")),
        help="Base output directory for all backtests",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline backtest (no direction change)",
    )
    parser.add_argument(
        "--skip-high-confidence",
        action="store_true",
        help="Skip high-confidence backtest",
    )
    parser.add_argument(
        "--skip-aggressive",
        action="store_true",
        help="Skip aggressive backtest",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.run_dir.exists():
        logger.error(f"Run directory not found: {args.run_dir}")
        sys.exit(1)

    if not args.config_dir.exists():
        logger.error(f"Config directory not found: {args.config_dir}")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("DIRECTION CHANGE VALIDATION BACKTESTS")
    logger.info("=" * 80)
    logger.info(f"Run directory: {args.run_dir}")
    logger.info(f"Config directory: {args.config_dir}")
    logger.info(f"Bar size: {args.bar_size}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    # Run backtests
    backtests = []

    if not args.skip_baseline:
        backtests.append(("baseline_no_dc", False, 0.20))

    if not args.skip_high_confidence:
        backtests.append(("high_confidence_dc_0.20", True, 0.20))

    if not args.skip_aggressive:
        backtests.append(("aggressive_dc_0.10", True, 0.10))

    for name, enabled, threshold in backtests:
        try:
            run_backtest(
                name=name,
                run_dir=args.run_dir,
                config_dir=args.config_dir,
                bar_size=args.bar_size,
                start_date=args.start_date,
                end_date=args.end_date,
                output_base=args.output_dir,
                dc_enabled=enabled,
                dc_threshold=threshold,
            )
        except Exception as e:
            logger.error(f"Failed to run {name}: {e}")
            continue

    logger.info("=" * 80)
    logger.info("ALL BACKTESTS COMPLETE")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"1. Review results: ls -la {args.output_dir}/*/")
    logger.info(f"2. Compare metrics: cat {args.output_dir}/*/metrics.csv | grep -A5 'Summary'")
    logger.info(f"3. Check exit reasons: for d in {args.output_dir}/*/; do echo $d; cut -d, -f7 $d/closed_positions.csv | tail -n+2 | sort | uniq -c; done")
    logger.info("")


if __name__ == "__main__":
    main()
