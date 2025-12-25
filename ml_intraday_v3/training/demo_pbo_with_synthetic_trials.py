"""
Quick demo: Generate synthetic trials and compute PBO

This creates realistic synthetic trials to demonstrate PBO functionality.
Run this to test the PBO notebook sections without waiting for real training.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.trial_tracker import TrialTracker
from experiments.diagnostics import compute_pbo_with_confidence, generate_pbo_report, plot_pbo_distribution, plot_pbo_with_confidence


def create_demo_trials(run_dir: str, scenario: str = 'moderate_overfitting'):
    """
    Create synthetic trials demonstrating different overfitting scenarios.

    Parameters
    ----------
    run_dir : str
        Path to run directory
    scenario : str
        'no_overfitting': IS ≈ OOS (PBO should be low)
        'moderate_overfitting': IS slightly > OOS (PBO ~0.3-0.5)
        'severe_overfitting': IS >> OOS (PBO should be high)
    """
    print(f"\n{'='*80}")
    print(f"Creating Demo Trials - Scenario: {scenario}")
    print(f"{'='*80}\n")

    tracker = TrialTracker(run_dir)

    # Simulate hyperparameter search over C values and penalties
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    penalties = ['l1', 'l2']
    n_paths = 5

    np.random.seed(42)

    trial_configs = []
    for C in C_values:
        for penalty in penalties:
            trial_configs.append({'C': C, 'penalty': penalty})

    print(f"Simulating {len(trial_configs)} trials across {n_paths} CPCV paths...")

    for trial_idx, params in enumerate(trial_configs):
        config = {
            'model': {
                'kind': 'logreg',
                'params': params
            },
            'seed': 42,
            'features': {'use_columns': 'all'}
        }

        trial_id = tracker.log_trial(
            config=config,
            model_type='logreg',
            hyperparameters=params,
            metadata={'scenario': scenario}
        )

        # Simulate CPCV results based on scenario
        for path_idx in range(n_paths):
            if scenario == 'no_overfitting':
                # No overfitting: IS ≈ OOS
                base_perf = 0.60 + trial_idx * 0.015
                is_metric = base_perf + np.random.rand() * 0.03
                oos_metric = base_perf + np.random.rand() * 0.03

            elif scenario == 'moderate_overfitting':
                # Moderate overfitting: IS slightly > OOS
                is_metric = 0.60 + trial_idx * 0.02 + np.random.rand() * 0.05
                oos_metric = is_metric - 0.04 - np.random.rand() * 0.03

            elif scenario == 'severe_overfitting':
                # Severe overfitting: IS >> OOS
                is_metric = 0.70 + trial_idx * 0.01 + np.random.rand() * 0.05
                oos_metric = 0.55 + np.random.rand() * 0.10

            else:
                raise ValueError(f"Unknown scenario: {scenario}")

            tracker.update_path_metrics(
                trial_id,
                f'path_{path_idx}',
                is_metric=is_metric,
                oos_metric=oos_metric
            )

    tracker.save()
    print(f"\n✓ Saved {len(trial_configs)} trials to: {tracker.trials_file}")

    return tracker


def analyze_demo_trials(run_dir: str):
    """Analyze demo trials and display results."""
    print(f"\n{'='*80}")
    print("Analyzing Trials")
    print(f"{'='*80}\n")

    # Load trials
    tracker = TrialTracker(run_dir)
    trials_df = tracker.to_dataframe()

    print(f"Trials loaded: {len(trials_df)}")
    print(f"CPCV paths: {len([c for c in trials_df.columns if c.endswith('_is')])}")

    # Compute PBO with confidence intervals
    print("\nComputing PBO with bootstrap confidence intervals...")
    pbo_result = compute_pbo_with_confidence(
        trials_df,
        metric_name='roc_auc',
        higher_is_better=True,
        n_bootstrap=1000,
        confidence_level=0.95,
        random_state=42
    )

    # Display results
    print(f"\n{'='*80}")
    print("PBO Results")
    print(f"{'='*80}\n")

    pbo = pbo_result['pbo']
    pbo_lower = pbo_result['pbo_lower']
    pbo_upper = pbo_result['pbo_upper']

    print(f"PBO: {pbo:.3f} ({pbo*100:.1f}%)")
    print(f"95% CI: [{pbo_lower:.3f}, {pbo_upper:.3f}]")
    print(f"\nLambda Statistics:")
    print(f"  Mean: {pbo_result['lambda_mean']:.3f}")
    print(f"  Median: {pbo_result['lambda_median']:.3f}")
    print(f"  Std: {pbo_result['lambda_std']:.3f}")

    # Interpret
    if pbo < 0.3:
        risk = "🟢 LOW RISK"
        interpretation = "Configuration appears robust"
    elif pbo < 0.5:
        risk = "🟠 MODERATE RISK"
        interpretation = "Some overfitting risk, monitor carefully"
    else:
        risk = "🔴 HIGH RISK"
        interpretation = "Likely overfitting, do not deploy"

    print(f"\nRisk Level: {risk}")
    print(f"Interpretation: {interpretation}")

    # Generate report
    report_path = Path(run_dir) / 'pbo_demo_report.md'
    generate_pbo_report(pbo_result, save_path=str(report_path))
    print(f"\n✓ Report saved to: {report_path}")

    # Create visualizations
    print("\nGenerating visualizations...")

    fig1 = plot_pbo_distribution(
        pbo_result['lambda_values'],
        pbo_result['pbo']
    )
    plot1_path = Path(run_dir) / 'pbo_demo_lambda_distribution.png'
    fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
    print(f"✓ Lambda distribution saved to: {plot1_path}")

    fig2 = plot_pbo_with_confidence(pbo_result)
    plot2_path = Path(run_dir) / 'pbo_demo_with_ci.png'
    fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
    print(f"✓ PBO with CI saved to: {plot2_path}")

    print(f"\n{'='*80}")
    print("Demo Complete!")
    print(f"{'='*80}\n")
    print("Next steps:")
    print("1. Open the notebook and run Section 4.6.4 (Enhanced PBO Analysis)")
    print("2. Run Section 5.3 (PBO Validation)")
    print("3. Review the generated plots and report")
    print("\nThe notebook sections will now find and analyze these demo trials.")


def main():
    """Run demo."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python demo_pbo_with_synthetic_trials.py <run_dir> [scenario]")
        print("\nScenarios:")
        print("  no_overfitting       - IS ≈ OOS (PBO should be low)")
        print("  moderate_overfitting - IS slightly > OOS (PBO ~0.3-0.5)")
        print("  severe_overfitting   - IS >> OOS (PBO should be high)")
        print("\nExample:")
        print("  python demo_pbo_with_synthetic_trials.py runs/run_20251224_201700_1e88d003 moderate_overfitting")
        sys.exit(1)

    run_dir = sys.argv[1]
    scenario = sys.argv[2] if len(sys.argv) > 2 else 'moderate_overfitting'

    # Create demo trials
    tracker = create_demo_trials(run_dir, scenario)

    # Analyze
    analyze_demo_trials(run_dir)


if __name__ == '__main__':
    main()
