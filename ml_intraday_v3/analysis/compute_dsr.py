"""
Compute Deflated Sharpe Ratio (DSR) from Backtest Results

This script computes DSR across CPCV paths from backtest results, accounting for:
- Selection bias from testing multiple configurations
- Non-normality of returns (skewness, kurtosis)
- Statistical uncertainty in Sharpe ratio estimation

Usage:
    python -m ml_intraday_v3.analysis.compute_dsr \
        --run-dir runs/run_20251224_123456 \
        --bar-size 1m \
        [--n-trials 27]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_intraday_v3.experiments.diagnostics import (
    compute_dsr,
    generate_dsr_report,
    plot_dsr_distribution,
    plot_dsr_with_confidence
)
from ml_intraday_v3.experiments.trial_tracker import TrialTracker

logger = logging.getLogger(__name__)


def load_backtest_trades(backtest_dir: Path) -> pd.DataFrame:
    """
    Load trades from backtest results.

    Parameters
    ----------
    backtest_dir : Path
        Directory containing backtest results (trades.parquet or trades.csv)

    Returns
    -------
    pd.DataFrame
        DataFrame with trades including 'pnl' column
    """
    # Try parquet first
    trades_parquet = backtest_dir / "trades.parquet"
    if trades_parquet.exists():
        return pd.read_parquet(trades_parquet)

    # Try CSV
    trades_csv = backtest_dir / "trades.csv"
    if trades_csv.exists():
        return pd.read_csv(trades_csv)

    raise FileNotFoundError(f"No trades file found in {backtest_dir}")


def compute_dsr_for_path(
    trades_df: pd.DataFrame,
    n_trials: int,
    target_sharpe: float = 0.0,
    annualization_factor: float = None,
    path_id: str = None
) -> Dict:
    """
    Compute DSR for a single CPCV path.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Trades DataFrame with 'pnl' column
    n_trials : int
        Number of configurations tested
    target_sharpe : float
        Benchmark Sharpe ratio
    annualization_factor : float, optional
        Annualization factor for Sharpe
    path_id : str, optional
        Path identifier for logging

    Returns
    -------
    dict
        DSR result
    """
    if trades_df.empty:
        logger.warning(f"Path {path_id}: No trades")
        return {'dsr': None, 'reason': 'no_trades', 'path_id': path_id}

    # Extract returns
    returns = trades_df['pnl'].values

    # Compute DSR
    dsr_result = compute_dsr(
        returns=returns,
        n_trials=n_trials,
        target_sharpe=target_sharpe,
        annualization_factor=annualization_factor
    )

    dsr_result['path_id'] = path_id
    dsr_result['n_trades'] = len(trades_df)

    return dsr_result


def compute_dsr_across_paths(
    run_dir: Path,
    bar_size: str,
    n_trials: int = None,
    target_sharpe: float = 0.0,
    trades_per_day: float = None,
    output_dir: Path = None
) -> Dict:
    """
    Compute DSR across all CPCV paths from backtest results.

    Parameters
    ----------
    run_dir : Path
        Run directory
    bar_size : str
        Bar size ('1m' or '5m')
    n_trials : int, optional
        Number of trials tested. If None, tries to load from TrialTracker
    target_sharpe : float
        Benchmark Sharpe ratio
    trades_per_day : float, optional
        Average trades per day for annualization
    output_dir : Path, optional
        Output directory for reports/plots

    Returns
    -------
    dict
        Results including DSR values per path and aggregate statistics
    """
    run_dir = Path(run_dir)
    bar_dir = run_dir / f"bar_size={bar_size}"

    if output_dir is None:
        output_dir = bar_dir / "dsr_analysis"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*80}")
    logger.info("DSR Analysis Across CPCV Paths")
    logger.info(f"{'='*80}\n")

    # Try to get n_trials from TrialTracker if not provided
    if n_trials is None:
        try:
            tracker = TrialTracker(str(run_dir))
            n_trials = len(tracker.trials)
            logger.info(f"Loaded n_trials from TrialTracker: {n_trials}")
        except Exception as e:
            logger.warning(f"Could not load TrialTracker: {e}")
            n_trials = 10  # Conservative default
            logger.warning(f"Using conservative default n_trials = {n_trials}")

    # Compute annualization factor if trades_per_day provided
    annualization_factor = None
    if trades_per_day is not None and trades_per_day > 0:
        annualization_factor = np.sqrt(252 * trades_per_day)
        logger.info(f"Annualization factor: {annualization_factor:.2f} (for {trades_per_day} trades/day)")

    # Load CPCV paths metadata
    cpcv_file = bar_dir / "cpcv_paths.json"
    if not cpcv_file.exists():
        raise FileNotFoundError(f"CPCV paths file not found: {cpcv_file}")

    with open(cpcv_file, 'r') as f:
        cpcv_data = json.load(f)

    paths = cpcv_data.get('paths', [])
    logger.info(f"Found {len(paths)} CPCV paths")

    # Compute DSR for each path
    dsr_results = []

    for i, path_info in enumerate(paths, 1):
        path_id = path_info.get('path_id')
        logger.info(f"\nPath {i}/{len(paths)}: {path_id}")

        # Look for backtest results for this path
        backtest_dir = bar_dir / "backtest" / f"path_{path_id}"

        if not backtest_dir.exists():
            logger.warning(f"  Backtest directory not found: {backtest_dir}")
            continue

        try:
            # Load trades
            trades_df = load_backtest_trades(backtest_dir)
            logger.info(f"  Loaded {len(trades_df)} trades")

            # Compute DSR
            dsr_result = compute_dsr_for_path(
                trades_df=trades_df,
                n_trials=n_trials,
                target_sharpe=target_sharpe,
                annualization_factor=annualization_factor,
                path_id=path_id
            )

            if dsr_result.get('dsr') is not None:
                logger.info(f"  DSR: {dsr_result['dsr']:.4f}")
                logger.info(f"  Sharpe: {dsr_result['sharpe']:.4f}")
                logger.info(f"  SR*: {dsr_result['sr_star']:.4f}")
                logger.info(f"  Skewness: {dsr_result['skewness']:.2f}, Kurtosis: {dsr_result['kurtosis']:.2f}")
            else:
                logger.warning(f"  DSR computation failed: {dsr_result.get('reason')}")

            dsr_results.append(dsr_result)

        except Exception as e:
            logger.error(f"  Error processing path {path_id}: {e}", exc_info=True)
            continue

    if not dsr_results:
        raise ValueError("No valid DSR results computed")

    # Filter valid results
    valid_results = [r for r in dsr_results if r.get('dsr') is not None]

    if not valid_results:
        raise ValueError("All DSR computations failed")

    logger.info(f"\n{'='*80}")
    logger.info(f"Computed DSR for {len(valid_results)} / {len(paths)} paths")
    logger.info(f"{'='*80}\n")

    # Aggregate statistics
    dsr_values = [r['dsr'] for r in valid_results]
    sharpe_values = [r['sharpe'] for r in valid_results]

    aggregate = {
        'n_paths_total': len(paths),
        'n_paths_valid': len(valid_results),
        'dsr_mean': float(np.mean(dsr_values)),
        'dsr_median': float(np.median(dsr_values)),
        'dsr_std': float(np.std(dsr_values)),
        'dsr_min': float(np.min(dsr_values)),
        'dsr_max': float(np.max(dsr_values)),
        'dsr_q25': float(np.percentile(dsr_values, 25)),
        'dsr_q75': float(np.percentile(dsr_values, 75)),
        'sharpe_mean': float(np.mean(sharpe_values)),
        'sharpe_median': float(np.median(sharpe_values)),
        'pct_dsr_above_0.5': float(100 * np.mean([d > 0.5 for d in dsr_values])),
        'pct_dsr_above_0.95': float(100 * np.mean([d > 0.95 for d in dsr_values])),
    }

    logger.info("Aggregate Statistics:")
    logger.info(f"  DSR Mean: {aggregate['dsr_mean']:.4f}")
    logger.info(f"  DSR Median: {aggregate['dsr_median']:.4f}")
    logger.info(f"  DSR Std: {aggregate['dsr_std']:.4f}")
    logger.info(f"  DSR Range: [{aggregate['dsr_min']:.4f}, {aggregate['dsr_max']:.4f}]")
    logger.info(f"  % DSR > 0.5: {aggregate['pct_dsr_above_0.5']:.1f}%")
    logger.info(f"  % DSR > 0.95: {aggregate['pct_dsr_above_0.95']:.1f}%")

    # Risk assessment
    median_dsr = aggregate['dsr_median']

    if median_dsr > 0.95:
        risk = "🟢 STRONG EVIDENCE"
        interpretation = "Very likely skill, not luck (p < 0.05 equivalent)"
        action = "High confidence for deployment"
    elif median_dsr > 0.5:
        risk = "🟠 MODERATE EVIDENCE"
        interpretation = "More likely skill than luck, but not conclusive"
        action = "Proceed with caution, monitor closely"
    else:
        risk = "🔴 WEAK EVIDENCE"
        interpretation = "Likely overfitting or luck, not genuine skill"
        action = "DO NOT deploy - likely overfitting"

    logger.info(f"\nRisk Assessment:")
    logger.info(f"  Risk Level: {risk}")
    logger.info(f"  Interpretation: {interpretation}")
    logger.info(f"  Action: {action}")

    # Generate report for median path
    logger.info(f"\nGenerating DSR report for median path...")

    median_idx = np.argsort(dsr_values)[len(dsr_values) // 2]
    median_result = valid_results[median_idx]

    report_path = output_dir / 'dsr_report_median.md'
    generate_dsr_report(median_result, save_path=str(report_path))
    logger.info(f"✓ Report saved to: {report_path}")

    # Generate visualizations
    logger.info(f"\nGenerating visualizations...")

    # DSR distribution across paths
    fig1 = plot_dsr_distribution(valid_results)
    plot1_path = output_dir / 'dsr_distribution_across_paths.png'
    fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    logger.info(f"✓ DSR distribution saved to: {plot1_path}")

    # DSR with confidence intervals
    fig2 = plot_dsr_with_confidence(valid_results)
    plot2_path = output_dir / 'dsr_with_confidence.png'
    fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f"✓ DSR with CI saved to: {plot2_path}")

    # Save all results to JSON
    results_path = output_dir / 'dsr_results_all_paths.json'
    with open(results_path, 'w') as f:
        json.dump({
            'aggregate': aggregate,
            'paths': dsr_results,
            'parameters': {
                'n_trials': n_trials,
                'target_sharpe': target_sharpe,
                'annualization_factor': annualization_factor,
                'trades_per_day': trades_per_day
            }
        }, f, indent=2)
    logger.info(f"✓ All results saved to: {results_path}")

    logger.info(f"\n{'='*80}")
    logger.info("DSR Analysis Complete")
    logger.info(f"{'='*80}\n")

    logger.info("Output files:")
    logger.info(f"  Report (median): {report_path}")
    logger.info(f"  Distribution plot: {plot1_path}")
    logger.info(f"  Confidence plot: {plot2_path}")
    logger.info(f"  All results (JSON): {results_path}")

    return {
        'aggregate': aggregate,
        'paths': dsr_results,
        'risk_assessment': {
            'level': risk,
            'interpretation': interpretation,
            'action': action
        }
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute DSR from backtest results across CPCV paths"
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Run directory path'
    )
    parser.add_argument(
        '--bar-size',
        type=str,
        required=True,
        choices=['1m', '5m'],
        help='Bar size'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=None,
        help='Number of trials tested (default: load from TrialTracker or use 10)'
    )
    parser.add_argument(
        '--target-sharpe',
        type=float,
        default=0.0,
        help='Benchmark Sharpe ratio (default: 0.0)'
    )
    parser.add_argument(
        '--trades-per-day',
        type=float,
        default=None,
        help='Average trades per day for annualization (default: None)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: run_dir/bar_size=X/dsr_analysis)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run DSR computation
    compute_dsr_across_paths(
        run_dir=Path(args.run_dir),
        bar_size=args.bar_size,
        n_trials=args.n_trials,
        target_sharpe=args.target_sharpe,
        trades_per_day=args.trades_per_day,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )


if __name__ == '__main__':
    main()
