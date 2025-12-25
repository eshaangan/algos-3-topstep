"""
PBO Analysis for Tracked Trials

Loads trials tracked during hyperparameter search and computes:
- Probability of Backtest Overfitting (PBO)
- Bootstrap confidence intervals
- Visualizations
- Risk assessment report

Usage:
    python -m ml_intraday_v3.training.analyze_pbo \
        --run-dir runs/run_20251224_123456 \
        [--metric roc_auc] \
        [--output-dir runs/run_20251224_123456/pbo_analysis]
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_intraday_v3.experiments.trial_tracker import TrialTracker
from ml_intraday_v3.experiments.diagnostics import (
    compute_pbo_with_confidence,
    generate_pbo_report,
    plot_pbo_distribution,
    plot_pbo_with_confidence
)

logger = logging.getLogger(__name__)


def analyze_trials(
    run_dir: Path,
    metric_name: str = 'roc_auc',
    higher_is_better: bool = True,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    output_dir: Path = None,
    random_state: int = 42
):
    """
    Analyze tracked trials and compute PBO.

    Parameters:
        run_dir: Run directory containing trials.json
        metric_name: Metric to use for PBO ('roc_auc', 'sharpe', etc.)
        higher_is_better: Whether higher metric values are better
        n_bootstrap: Number of bootstrap samples for CI
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        output_dir: Directory to save outputs (default: run_dir/pbo_analysis)
        random_state: Random seed for reproducibility
    """
    run_dir = Path(run_dir)

    if output_dir is None:
        output_dir = run_dir / 'pbo_analysis'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*80}")
    logger.info("PBO Analysis - Tracked Trials")
    logger.info(f"{'='*80}\n")

    # Load trials
    logger.info(f"Loading trials from: {run_dir}")
    tracker = TrialTracker(str(run_dir))

    if not tracker.trials:
        logger.error("No trials found! Run hyperparameter search first.")
        return

    # Convert to DataFrame
    trials_df = tracker.to_dataframe()
    summary = tracker.summary()

    logger.info(f"Loaded {summary['total_trials']} trials")
    logger.info(f"Model types: {summary['model_types']}")
    logger.info(f"CPCV paths per trial: {summary['paths_per_trial']}")

    # Check if we have enough data
    n_trials = len(trials_df)
    n_paths = len([c for c in trials_df.columns if c.endswith('_oos')])

    if n_trials < 2:
        logger.error(f"Need at least 2 trials for PBO, found {n_trials}")
        return

    if n_paths < 2:
        logger.error(f"Need at least 2 CPCV paths for PBO, found {n_paths}")
        return

    logger.info(f"\nComputing PBO:")
    logger.info(f"  Metric: {metric_name}")
    logger.info(f"  Higher is better: {higher_is_better}")
    logger.info(f"  Bootstrap samples: {n_bootstrap}")
    logger.info(f"  Confidence level: {confidence_level}")

    # Compute PBO with confidence intervals
    pbo_result = compute_pbo_with_confidence(
        trials_df=trials_df,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state
    )

    # Display results
    logger.info(f"\n{'='*80}")
    logger.info("PBO Results")
    logger.info(f"{'='*80}\n")

    pbo = pbo_result['pbo']
    pbo_lower = pbo_result['pbo_lower']
    pbo_upper = pbo_result['pbo_upper']

    logger.info(f"PBO: {pbo:.3f} ({pbo*100:.1f}%)")
    logger.info(f"{int(confidence_level*100)}% CI: [{pbo_lower:.3f}, {pbo_upper:.3f}]")
    logger.info(f"\nLambda Statistics:")
    logger.info(f"  Mean: {pbo_result['lambda_mean']:.3f}")
    logger.info(f"  Median: {pbo_result['lambda_median']:.3f}")
    logger.info(f"  Std: {pbo_result['lambda_std']:.3f}")

    # Risk assessment
    if pbo < 0.3:
        risk = "🟢 LOW RISK"
        interpretation = "Configuration appears robust"
        action = "Proceed with selected configuration"
    elif pbo < 0.5:
        risk = "🟠 MODERATE RISK"
        interpretation = "Some overfitting risk detected"
        action = "Monitor carefully, consider additional validation"
    else:
        risk = "🔴 HIGH RISK"
        interpretation = "Likely overfitting detected"
        action = "Do NOT deploy - reduce search space or increase regularization"

    logger.info(f"\nRisk Level: {risk}")
    logger.info(f"Interpretation: {interpretation}")
    logger.info(f"Recommended Action: {action}")

    # Generate report
    logger.info(f"\nGenerating report...")
    report_path = output_dir / 'pbo_report.md'
    generate_pbo_report(pbo_result, save_path=str(report_path))
    logger.info(f"✓ Report saved to: {report_path}")

    # Generate visualizations
    logger.info(f"\nGenerating visualizations...")

    # Lambda distribution
    fig1 = plot_pbo_distribution(
        pbo_result['lambda_values'],
        pbo_result['pbo']
    )
    plot1_path = output_dir / 'pbo_lambda_distribution.png'
    fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    logger.info(f"✓ Lambda distribution saved to: {plot1_path}")

    # PBO with confidence intervals
    fig2 = plot_pbo_with_confidence(pbo_result)
    plot2_path = output_dir / 'pbo_with_confidence.png'
    fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f"✓ PBO with CI saved to: {plot2_path}")

    logger.info(f"\n{'='*80}")
    logger.info("Analysis Complete")
    logger.info(f"{'='*80}\n")

    logger.info("Output files:")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  Lambda distribution: {plot1_path}")
    logger.info(f"  PBO with CI: {plot2_path}")
    logger.info(f"  Trials data: {tracker.trials_file}")

    return pbo_result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze tracked trials and compute PBO"
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Run directory containing trials.json'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='roc_auc',
        help='Metric name for PBO (default: roc_auc)'
    )
    parser.add_argument(
        '--higher-is-better',
        action='store_true',
        default=True,
        help='Whether higher metric values are better (default: True)'
    )
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=1000,
        help='Number of bootstrap samples (default: 1000)'
    )
    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.95,
        help='Confidence level for CI (default: 0.95)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: run_dir/pbo_analysis)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
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

    # Run analysis
    analyze_trials(
        run_dir=Path(args.run_dir),
        metric_name=args.metric,
        higher_is_better=args.higher_is_better,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        random_state=args.random_state
    )


if __name__ == '__main__':
    main()
