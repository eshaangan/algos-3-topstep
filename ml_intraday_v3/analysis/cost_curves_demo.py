"""
Cost Curves Demonstration

This script demonstrates how to use cost curves for classifier performance
visualization and model comparison in the ML Pipeline V3.

Usage:
    python ml_intraday_v3/analysis/cost_curves_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ml_intraday_v3.analysis.cost_curves import (
    compute_cost_curve,
    bootstrap_cost_curve,
    plot_cost_curve,
    plot_cost_difference,
    compute_trading_cost_curve,
    compare_models_cost_curves,
    compute_area_under_cost_curve
)

# Note: diagnostics integration available via:
# from ml_intraday_v3.experiments.diagnostics import (
#     compute_model_cost_curves,
#     compute_trading_cost_diagnostics,
#     compare_models_cost_diagnostics
# )


def generate_synthetic_predictions(n=1000, quality='good', random_state=42):
    """Generate synthetic predictions for demonstration."""
    rng = np.random.RandomState(random_state)
    y_true = rng.binomial(1, 0.4, n)  # 40% positive class (profitable trades)

    if quality == 'good':
        # Good classifier: high separation between classes
        y_prob = y_true * 0.75 + (1 - y_true) * 0.25 + rng.normal(0, 0.1, n)
    elif quality == 'medium':
        # Medium classifier: moderate separation
        y_prob = y_true * 0.65 + (1 - y_true) * 0.35 + rng.normal(0, 0.15, n)
    elif quality == 'poor':
        # Poor classifier: weak separation
        y_prob = y_true * 0.55 + (1 - y_true) * 0.45 + rng.normal(0, 0.2, n)
    else:  # random
        # Random classifier
        y_prob = rng.uniform(0, 1, n)

    y_prob = np.clip(y_prob, 0, 1)
    return y_true, y_prob


def demo_basic_cost_curve():
    """Demo 1: Basic cost curve computation and plotting."""
    print("\n" + "="*80)
    print("DEMO 1: Basic Cost Curve Computation")
    print("="*80)

    # Generate synthetic data
    y_true, y_prob = generate_synthetic_predictions(n=500, quality='good')

    # Compute cost curve
    curve_df = compute_cost_curve(y_true, y_prob)

    print(f"\nCost curve computed with {len(curve_df)} points")
    print(f"PC range: [{curve_df['pc'].min():.3f}, {curve_df['pc'].max():.3f}]")
    print(f"NC range: [{curve_df['nc'].min():.3f}, {curve_df['nc'].max():.3f}]")

    # Compute AUCC
    aucc = compute_area_under_cost_curve(curve_df)
    print(f"Area Under Cost Curve (AUCC): {aucc:.4f} (lower is better)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_cost_curve(curve_df, ax=ax, label="Good Classifier")
    plt.savefig("demo_cost_curve_basic.png", dpi=150, bbox_inches='tight')
    print("\nSaved: demo_cost_curve_basic.png")
    plt.close()


def demo_bootstrap_confidence():
    """Demo 2: Cost curve with bootstrap confidence intervals."""
    print("\n" + "="*80)
    print("DEMO 2: Cost Curve with Bootstrap Confidence Intervals")
    print("="*80)

    y_true, y_prob = generate_synthetic_predictions(n=300, quality='medium')

    # Compute cost curve with bootstrap
    mean_curve, lower, upper = bootstrap_cost_curve(
        y_true, y_prob,
        n_bootstrap=500,  # Use 500 for faster demo (1000 recommended)
        confidence_level=0.95,
        random_state=42
    )

    print(f"\nBootstrap confidence intervals computed (95% CI)")
    print(f"Mean AUCC: {compute_area_under_cost_curve(mean_curve):.4f}")

    # Compute CI width at PC=0.5
    mid_idx = len(mean_curve) // 2
    ci_width = upper.iloc[mid_idx]['nc'] - lower.iloc[mid_idx]['nc']
    print(f"CI width at PC≈0.5: {ci_width:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_cost_curve(
        mean_curve,
        ax=ax,
        label="Medium Classifier",
        show_confidence=True,
        confidence_bounds=(lower, upper)
    )
    plt.savefig("demo_cost_curve_bootstrap.png", dpi=150, bbox_inches='tight')
    print("\nSaved: demo_cost_curve_bootstrap.png")
    plt.close()


def demo_model_comparison():
    """Demo 3: Comparing multiple models."""
    print("\n" + "="*80)
    print("DEMO 3: Comparing Multiple Models with Cost Curves")
    print("="*80)

    # Generate predictions for 3 models
    y_true_good, y_prob_good = generate_synthetic_predictions(n=500, quality='good', random_state=42)
    y_true_med, y_prob_med = generate_synthetic_predictions(n=500, quality='medium', random_state=42)
    y_true_poor, y_prob_poor = generate_synthetic_predictions(n=500, quality='poor', random_state=42)

    # All models use same y_true (same test set)
    y_true = y_true_good

    models_data = {
        'Good Model': (y_true, y_prob_good),
        'Medium Model': (y_true, y_prob_med),
        'Poor Model': (y_true, y_prob_poor)
    }

    # Compute comparison
    fig, curves = compare_models_cost_curves(
        models_data,
        show_confidence=False
    )

    plt.savefig("demo_cost_curve_comparison.png", dpi=150, bbox_inches='tight')
    print("\nSaved: demo_cost_curve_comparison.png")
    plt.close()

    # Print AUCC values
    print("\nAUCC Comparison (lower is better):")
    for name, curve in curves.items():
        aucc = compute_area_under_cost_curve(curve)
        print(f"  {name:15s}: {aucc:.4f}")


def demo_cost_difference():
    """Demo 4: Cost difference visualization."""
    print("\n" + "="*80)
    print("DEMO 4: Cost Curve Difference (Model A vs Baseline)")
    print("="*80)

    y_true, _ = generate_synthetic_predictions(n=500, random_state=42)
    _, y_prob_a = generate_synthetic_predictions(n=500, quality='good', random_state=42)
    _, y_prob_b = generate_synthetic_predictions(n=500, quality='medium', random_state=43)

    curve_a = compute_cost_curve(y_true, y_prob_a)
    curve_b = compute_cost_curve(y_true, y_prob_b)

    # Plot difference
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_cost_difference(
        curve_a, curve_b,
        ax=ax,
        label1="Model A",
        label2="Baseline Model"
    )
    plt.savefig("demo_cost_curve_difference.png", dpi=150, bbox_inches='tight')
    print("\nSaved: demo_cost_curve_difference.png")
    plt.close()

    # Compute dominance
    nc_diff = curve_a['nc'].values - curve_b['nc'].values
    pct_better = (nc_diff < 0).mean() * 100
    print(f"\nModel A is better (lower NC) for {pct_better:.1f}% of cost ratios")


def demo_trading_cost_curve():
    """Demo 5: Trading-specific cost curves with risk/reward ratios."""
    print("\n" + "="*80)
    print("DEMO 5: Trading-Specific Cost Curves")
    print("="*80)

    y_true, y_prob = generate_synthetic_predictions(n=500, quality='good', random_state=42)

    # Use trading-specific RR ratios
    rr_ratios = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    curve_df = compute_trading_cost_curve(
        y_true, y_prob,
        risk_reward_ratios=rr_ratios
    )

    print(f"\nTrading cost curve computed for {len(rr_ratios)} risk/reward ratios")
    print("\nResults by Risk/Reward Ratio:")
    print(curve_df[['risk_reward_ratio', 'cost_ratio', 'nc', 'threshold', 'tpr', 'fpr']].to_string(index=False))

    # Find optimal RR
    optimal_idx = curve_df['nc'].idxmin()
    optimal_rr = curve_df.loc[optimal_idx, 'risk_reward_ratio']
    optimal_nc = curve_df.loc[optimal_idx, 'nc']
    print(f"\nOptimal Risk/Reward Ratio: {optimal_rr:.1f} (NC={optimal_nc:.4f})")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_cost_curve(curve_df, ax=ax, label=f"Trading Strategy")
    plt.savefig("demo_trading_cost_curve.png", dpi=150, bbox_inches='tight')
    print("\nSaved: demo_trading_cost_curve.png")
    plt.close()


def demo_diagnostics_integration():
    """Demo 6: Summary statistics and diagnostics."""
    print("\n" + "="*80)
    print("DEMO 6: Summary Statistics and Diagnostics")
    print("="*80)

    y_true, y_prob = generate_synthetic_predictions(n=500, quality='good', random_state=42)

    # Compute cost curve
    curve_df = compute_cost_curve(y_true, y_prob)
    aucc = compute_area_under_cost_curve(curve_df)

    print(f"\nModel Statistics:")
    print(f"Samples: {len(y_true)} (positive: {np.sum(y_true)}, negative: {len(y_true) - np.sum(y_true)})")
    print(f"AUCC: {aucc:.4f}")

    # Trading diagnostics
    trading_curve = compute_trading_cost_curve(
        y_true, y_prob,
        risk_reward_ratios=[1.0, 1.5, 2.0, 2.5, 3.0]
    )

    optimal_idx = trading_curve['nc'].idxmin()
    optimal_row = trading_curve.loc[optimal_idx]

    print(f"\nTrading Cost Diagnostics:")
    print(f"  Optimal RR: {optimal_row['risk_reward_ratio']:.2f}")
    print(f"  Optimal NC: {optimal_row['nc']:.4f}")
    print(f"  Optimal Threshold: {optimal_row['threshold']:.3f}")
    print(f"  Optimal TPR: {optimal_row['tpr']:.3f}")
    print(f"  Optimal FPR: {optimal_row['fpr']:.3f}")

    print("\nNote: For full diagnostics integration, use:")
    print("  from ml_intraday_v3.experiments.diagnostics import compute_model_cost_curves")


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("COST CURVES DEMONSTRATION - ML Pipeline V3")
    print("="*80)

    # Create output directory
    output_dir = Path("cost_curves_demo_output")
    output_dir.mkdir(exist_ok=True)

    # Change to output directory for plots
    import os
    original_dir = os.getcwd()
    os.chdir(output_dir)

    try:
        # Run demos
        demo_basic_cost_curve()
        demo_bootstrap_confidence()
        demo_model_comparison()
        demo_cost_difference()
        demo_trading_cost_curve()
        demo_diagnostics_integration()

        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        print(f"\nAll plots saved to: {output_dir.absolute()}/")

    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
