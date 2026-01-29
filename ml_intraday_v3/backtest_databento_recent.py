#!/usr/bin/env python3
"""
Backtest on recent Databento data with direction_change threshold validation.

This script:
1. Loads the fetched 1m Databento bars
2. Resamples to 5m (to match model training)
3. Runs three backtests with different direction_change thresholds
4. Compares results

Focus: Out-of-sample validation on Jan 2026 data (model trained on Dec 2024 and earlier)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from ml_intraday_v3.live_trading.replay import replay_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def resample_bars(bars_1m: pd.DataFrame, target_freq: str = "5T") -> pd.DataFrame:
    """
    Resample 1-minute bars to target frequency.

    Args:
        bars_1m: DataFrame with 1-minute OHLCV bars
        target_freq: Target frequency (e.g., "5T" for 5 minutes)

    Returns:
        Resampled DataFrame
    """
    logger.info(f"Resampling {len(bars_1m):,} bars from 1m to {target_freq}...")

    # Ensure index is datetime
    if not isinstance(bars_1m.index, pd.DatetimeIndex):
        bars_1m.index = pd.to_datetime(bars_1m.index)

    # Resample using OHLCV logic
    resampled = bars_1m.resample(target_freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()

    logger.info(f"Resampled to {len(resampled):,} bars")
    return resampled


def prepare_run_directory(
    bars_df: pd.DataFrame,
    bar_size: str,
    output_dir: Path,
) -> Path:
    """
    Prepare run directory with bars and metadata.

    Args:
        bars_df: OHLCV bars DataFrame
        bar_size: Bar size string (e.g., "5m")
        output_dir: Base output directory

    Returns:
        Path to run directory
    """
    # Create run directory
    run_name = f"databento_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_name
    bar_dir = run_dir / f"bar_size={bar_size}"
    bar_dir.mkdir(parents=True, exist_ok=True)

    # Save bars
    bars_path = bar_dir / "bars.parquet"
    bars_df.to_parquet(bars_path)
    logger.info(f"Saved {len(bars_df):,} bars to: {bars_path}")

    # Create label_schema.json (using FIXED config values)
    label_schema = {
        "stop_multiple": 1.5,    # DIAGNOSTIC FIX: Widened from 1.0
        "target_multiple": 2.5,  # DIAGNOSTIC FIX: Widened from 2.0
        "atr_period": 14,
        "atr_bar_size": bar_size,
    }
    schema_path = bar_dir / "label_schema.json"
    with open(schema_path, "w") as f:
        json.dump(label_schema, f, indent=2)
    logger.info(f"Saved label schema to: {schema_path}")

    # Copy model bundle to walkforward directory
    import shutil
    wf_dir = run_dir / "walkforward" / f"bar_size={bar_size}" / "window_0"
    wf_dir.mkdir(parents=True, exist_ok=True)

    model_src = Path("ml_intraday_v3/models/saved/model_bundle.pkl")
    if model_src.exists():
        model_dst = wf_dir / "model_bundle.pkl"
        shutil.copy(model_src, model_dst)
        logger.info(f"Copied model bundle to: {model_dst}")
    else:
        raise FileNotFoundError(f"Model bundle not found: {model_src}")

    return run_dir


def modify_live_config(
    config_path: Path,
    bar_size: str,
    dc_enabled: bool,
    dc_threshold: float,
    output_path: Path,
) -> None:
    """Modify live_trading.yaml with specific settings."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update bar_size
    config['trading']['bar_size'] = bar_size

    # Update direction_change settings
    if 'direction_change' not in config:
        config['direction_change'] = {}
    config['direction_change']['enabled'] = dc_enabled
    config['direction_change']['high_confidence_threshold'] = dc_threshold

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


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
) -> dict:
    """
    Run a single backtest with specific direction_change settings.
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
        bar_size,
        dc_enabled,
        dc_threshold,
        temp_config_dir / "live_trading.yaml",
    )

    # Create output directory
    output_dir = output_base / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run replay
    try:
        logger.info(f"Starting replay: {start_date} to {end_date}")
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
            return {'name': name, 'total_trades': 0, 'error': 'No trades'}

        wins = (artifacts.closed_positions['pnl_usd'] > 0).sum()
        win_rate = wins / total_trades * 100
        total_pnl = artifacts.closed_positions['pnl_usd'].sum()

        # Profit factor
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
        return {'name': name, 'error': str(e)}


def compare_results(results: list[dict], output_dir: Path) -> None:
    """Compare backtest results and print analysis."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARATIVE ANALYSIS")
    logger.info("=" * 80)

    # Filter successful results
    valid_results = [r for r in results if 'error' not in r and r['total_trades'] > 0]

    if not valid_results:
        logger.error("No valid results to compare")
        return

    # Create comparison table
    comparison = []
    for r in valid_results:
        dc_pct = sum(r['exit_pcts'].get(k, 0) for k in r['exit_pcts'] if 'direction_change' in k)
        comparison.append({
            'Backtest': r['name'],
            'Trades': r['total_trades'],
            'Win Rate': f"{r['win_rate']:.1f}%",
            'Total P&L': f"${r['total_pnl']:,.2f}",
            'Profit Factor': f"{r['profit_factor']:.2f}",
            'Target %': f"{r['exit_pcts'].get('target_hit', 0):.1f}%",
            'Stop %': f"{r['exit_pcts'].get('stop_hit', 0):.1f}%",
            'Dir Change %': f"{dc_pct:.1f}%",
        })

    df = pd.DataFrame(comparison)
    logger.info("\n" + df.to_string(index=False))

    # Save comparison
    df.to_csv(output_dir / "comparison.csv", index=False)
    logger.info(f"\nComparison saved to: {output_dir / 'comparison.csv'}")

    # Key insights
    logger.info("")
    logger.info("KEY INSIGHTS:")

    baseline = next((r for r in valid_results if r['name'] == 'baseline'), None)
    high_conf = next((r for r in valid_results if r['name'] == 'high_confidence'), None)
    aggressive = next((r for r in valid_results if r['name'] == 'aggressive'), None)

    if baseline and high_conf:
        pnl_diff = high_conf['total_pnl'] - baseline['total_pnl']
        pnl_pct = (pnl_diff / abs(baseline['total_pnl']) * 100) if baseline['total_pnl'] != 0 else 0

        logger.info(f"\n1. High-Confidence (0.20) vs Baseline (disabled):")
        logger.info(f"   P&L difference: ${pnl_diff:,.2f} ({pnl_pct:+.1f}%)")

        baseline_dc = sum(baseline['exit_pcts'].get(k, 0) for k in baseline['exit_pcts'] if 'direction_change' in k)
        highconf_dc = sum(high_conf['exit_pcts'].get(k, 0) for k in high_conf['exit_pcts'] if 'direction_change' in k)
        logger.info(f"   Direction change exits: {baseline_dc:.1f}% → {highconf_dc:.1f}%")

    if aggressive and high_conf:
        pnl_diff = high_conf['total_pnl'] - aggressive['total_pnl']
        pnl_pct = (pnl_diff / abs(aggressive['total_pnl']) * 100) if aggressive['total_pnl'] != 0 else 0

        logger.info(f"\n2. High-Confidence (0.20) vs Aggressive (0.10):")
        logger.info(f"   P&L difference: ${pnl_diff:,.2f} ({pnl_pct:+.1f}%)")

        aggressive_dc = sum(aggressive['exit_pcts'].get(k, 0) for k in aggressive['exit_pcts'] if 'direction_change' in k)
        highconf_dc = sum(high_conf['exit_pcts'].get(k, 0) for k in high_conf['exit_pcts'] if 'direction_change' in k)
        logger.info(f"   Direction change exits: {aggressive_dc:.1f}% → {highconf_dc:.1f}%")

    logger.info("")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Backtest on recent Databento data")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("ml_intraday_v3/runs/recent_data_20260124_181401"),
                        help="Directory with fetched 1m bars")
    parser.add_argument("--bar-size", default="5m", help="Target bar size for backtest")
    parser.add_argument("--start-date", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("ml_intraday_v3/backtest_results"),
                        help="Output directory")

    args = parser.parse_args()

    os.chdir(project_root)

    logger.info("=" * 80)
    logger.info("DATABENTO BACKTEST - Direction Change Threshold Validation")
    logger.info("=" * 80)
    logger.info(f"Data source: {args.data_dir}")
    logger.info(f"Target bar size: {args.bar_size}")
    logger.info("=" * 80)

    # Load 1m bars
    bars_1m_path = args.data_dir / "bar_size=1m" / "bars.parquet"
    if not bars_1m_path.exists():
        logger.error(f"Bars not found: {bars_1m_path}")
        return 1

    bars_1m = pd.read_parquet(bars_1m_path)
    logger.info(f"Loaded {len(bars_1m):,} 1-minute bars")
    logger.info(f"Date range: {bars_1m.index[0]} to {bars_1m.index[-1]}")

    # Resample to target frequency
    freq_map = {"1m": "1T", "5m": "5T", "15m": "15T", "30m": "30T", "1h": "1H"}
    target_freq = freq_map.get(args.bar_size, "5T")
    bars_resampled = resample_bars(bars_1m, target_freq)

    logger.info(f"Resampled bars: {len(bars_resampled):,}")
    logger.info(f"Date range: {bars_resampled.index[0]} to {bars_resampled.index[-1]}")

    # Prepare run directory
    run_dir = prepare_run_directory(
        bars_resampled,
        args.bar_size,
        Path("ml_intraday_v3/runs"),
    )

    # Determine date range
    start_date = args.start_date or bars_resampled.index[0].strftime("%Y-%m-%d")
    end_date = args.end_date or bars_resampled.index[-1].strftime("%Y-%m-%d")

    logger.info(f"Backtest date range: {start_date} to {end_date}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = args.output_dir / f"databento_validation_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)

    config_dir = Path("ml_intraday_v3/configs")

    # Run three backtests
    results = []

    # 1. Baseline: No direction change
    results.append(run_backtest(
        name="baseline",
        run_dir=run_dir,
        config_dir=config_dir,
        bar_size=args.bar_size,
        start_date=start_date,
        end_date=end_date,
        output_base=output_base,
        dc_enabled=False,
        dc_threshold=0.20,
    ))

    # 2. High-confidence: threshold=0.20 (RECOMMENDED)
    results.append(run_backtest(
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
    results.append(run_backtest(
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
