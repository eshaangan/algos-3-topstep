"""
Rare Events Corrections - Demonstration and Validation

This script demonstrates the King & Zeng (2001) rare events corrections
and validates that they improve calibration on imbalanced datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rare_events import (
    correct_rare_events_probabilities,
    compute_rare_event_weights,
    RelogitClassifier,
    apply_prior_correction_to_intercept
)


def demo_probability_correction():
    """Demonstrate probability correction."""
    print("\n" + "="*80)
    print("DEMO 1: Probability Correction")
    print("="*80)

    # Create highly imbalanced dataset (5% positive, like profitable trades)
    print("\nCreating imbalanced dataset (5% positive class)...")
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        weights=[0.95, 0.05],
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Training set: {len(y_train)} samples, {y_train.mean()*100:.1f}% positive")
    print(f"Test set: {len(y_test)} samples, {y_test.mean()*100:.1f}% positive")

    # Fit standard logistic regression
    print("\nFitting standard logistic regression...")
    lr = LogisticRegression(max_iter=200, random_state=42)
    lr.fit(X_train, y_train)
    prob_uncorrected = lr.predict_proba(X_test)[:, 1]

    # Apply rare events correction
    print("Applying King & Zeng rare events correction...")
    tau = np.mean(y_train)  # Estimate from training set
    prob_corrected = correct_rare_events_probabilities(
        prob_uncorrected,
        y_test,  # Test labels (same length as prob_uncorrected)
        tau=tau  # Use training set tau
    )

    # Compare statistics
    print(f"\nProbability Statistics:")
    print(f"  True positive rate: {y_test.mean():.3f}")
    print(f"  Uncorrected mean: {prob_uncorrected.mean():.3f}")
    print(f"  Corrected mean: {prob_corrected.mean():.3f}")
    print(f"  Difference: {prob_uncorrected.mean() - prob_corrected.mean():.3f}")

    # Calibration metrics
    brier_uncorrected = brier_score_loss(y_test, prob_uncorrected)
    brier_corrected = brier_score_loss(y_test, prob_corrected)

    print(f"\nCalibration (Brier Score - lower is better):")
    print(f"  Uncorrected: {brier_uncorrected:.4f}")
    print(f"  Corrected: {brier_corrected:.4f}")
    print(f"  Improvement: {(brier_uncorrected - brier_corrected):.4f}")

    return prob_uncorrected, prob_corrected, y_test


def demo_sample_weighting():
    """Demonstrate sample weighting."""
    print("\n" + "="*80)
    print("DEMO 2: Sample Weighting")
    print("="*80)

    y_train = np.array([0]*9 + [1])  # 10% positive
    print(f"\nSample: {len(y_train)} observations, {np.mean(y_train)*100:.0f}% positive")

    # King & Zeng weights
    weights_kz = compute_rare_event_weights(y_train, method="king_zeng", tau=0.1)
    print(f"\nKing & Zeng weights:")
    print(f"  Weight for y=0: {weights_kz[y_train==0][0]:.3f}")
    print(f"  Weight for y=1: {weights_kz[y_train==1][0]:.3f}")
    print(f"  Total weight y=0: {np.sum(weights_kz[y_train==0]):.3f}")
    print(f"  Total weight y=1: {np.sum(weights_kz[y_train==1]):.3f}")

    # Inverse frequency weights
    weights_if = compute_rare_event_weights(y_train, method="inverse_freq")
    print(f"\nInverse frequency weights:")
    print(f"  Weight for y=0: {weights_if[y_train==0][0]:.3f}")
    print(f"  Weight for y=1: {weights_if[y_train==1][0]:.3f}")
    print(f"  Total weight y=0: {np.sum(weights_if[y_train==0]):.3f}")
    print(f"  Total weight y=1: {np.sum(weights_if[y_train==1]):.3f}")


def demo_relogit_classifier():
    """Demonstrate RelogitClassifier."""
    print("\n" + "="*80)
    print("DEMO 3: RelogitClassifier vs Standard Logistic Regression")
    print("="*80)

    # Create imbalanced dataset
    print("\nCreating dataset (10% positive)...")
    X, y = make_classification(
        n_samples=1500,
        n_features=20,
        n_informative=15,
        weights=[0.9, 0.1],
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Standard logistic regression
    print("\nFitting standard logistic regression...")
    lr = LogisticRegression(max_iter=200, random_state=42)
    lr.fit(X_train, y_train)
    prob_lr = lr.predict_proba(X_test)[:, 1]

    # RelogitClassifier
    print("Fitting RelogitClassifier...")
    relogit = RelogitClassifier(max_iter=200, random_state=42)
    relogit.fit(X_train, y_train)
    prob_relogit = relogit.predict_proba(X_test)[:, 1]

    # Compare
    print(f"\nPerformance Comparison:")
    print(f"  True positive rate: {y_test.mean():.3f}")
    print(f"\nMean predicted probability:")
    print(f"  Standard LR: {prob_lr.mean():.3f} (error: {abs(prob_lr.mean() - y_test.mean()):.3f})")
    print(f"  RelogitClassifier: {prob_relogit.mean():.3f} (error: {abs(prob_relogit.mean() - y_test.mean()):.3f})")

    # ROC AUC (should be similar)
    auc_lr = roc_auc_score(y_test, prob_lr)
    auc_relogit = roc_auc_score(y_test, prob_relogit)
    print(f"\nROC AUC (discrimination - higher is better):")
    print(f"  Standard LR: {auc_lr:.4f}")
    print(f"  RelogitClassifier: {auc_relogit:.4f}")

    # Brier score (calibration)
    brier_lr = brier_score_loss(y_test, prob_lr)
    brier_relogit = brier_score_loss(y_test, prob_relogit)
    print(f"\nBrier Score (calibration - lower is better):")
    print(f"  Standard LR: {brier_lr:.4f}")
    print(f"  RelogitClassifier: {brier_relogit:.4f}")
    print(f"  Improvement: {(brier_lr - brier_relogit):.4f}")

    return prob_lr, prob_relogit, y_test


def plot_calibration_curves(prob_uncorrected, prob_corrected, y_true, title=""):
    """Plot calibration curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Calibration curve
    fraction_uncorr, mean_pred_uncorr = calibration_curve(
        y_true, prob_uncorrected, n_bins=10, strategy='uniform'
    )
    fraction_corr, mean_pred_corr = calibration_curve(
        y_true, prob_corrected, n_bins=10, strategy='uniform'
    )

    # Plot 1: Calibration curves
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.plot(mean_pred_uncorr, fraction_uncorr, 'o-', label='Uncorrected', linewidth=2)
    ax1.plot(mean_pred_corr, fraction_corr, 's-', label='Corrected (King & Zeng)', linewidth=2)
    ax1.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax1.set_ylabel('Fraction of Positives', fontsize=11)
    ax1.set_title('Calibration Curve', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Probability distributions
    ax2.hist(prob_uncorrected[y_true==0], bins=20, alpha=0.5, label='Negative (uncorr)', color='blue')
    ax2.hist(prob_corrected[y_true==0], bins=20, alpha=0.5, label='Negative (corr)', color='lightblue')
    ax2.hist(prob_uncorrected[y_true==1], bins=20, alpha=0.5, label='Positive (uncorr)', color='red')
    ax2.hist(prob_corrected[y_true==1], bins=20, alpha=0.5, label='Positive (corr)', color='pink')
    ax2.set_xlabel('Predicted Probability', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Probability Distributions', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def demo_intercept_correction():
    """Demonstrate intercept correction."""
    print("\n" + "="*80)
    print("DEMO 4: Intercept Correction")
    print("="*80)

    # Scenario: Population has 5% positive, but we trained on balanced sample (50%)
    intercept_orig = 0.0  # Intercept from balanced training
    tau_population = 0.05  # True population rate
    y_sample_mean = 0.50   # Balanced sample

    print(f"\nScenario:")
    print(f"  Population positive rate: {tau_population*100:.0f}%")
    print(f"  Training sample positive rate: {y_sample_mean*100:.0f}%")
    print(f"  Original intercept: {intercept_orig:.3f}")

    intercept_corrected = apply_prior_correction_to_intercept(
        intercept_orig,
        tau_population,
        y_sample_mean
    )

    print(f"\nCorrected intercept: {intercept_corrected:.3f}")
    print(f"Correction amount: {intercept_corrected - intercept_orig:.3f}")

    print(f"\nInterpretation:")
    print(f"  The intercept is shifted down by {abs(intercept_corrected - intercept_orig):.2f}")
    print(f"  This accounts for the oversampling of positive class in training")
    print(f"  Predictions will now match the true {tau_population*100:.0f}% population rate")


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("RARE EVENTS CORRECTIONS - DEMONSTRATION")
    print("King & Zeng (2001) for Imbalanced Classification")
    print("="*80)

    # Demo 1: Probability correction
    prob_uncorr1, prob_corr1, y_test1 = demo_probability_correction()

    # Demo 2: Sample weighting
    demo_sample_weighting()

    # Demo 3: RelogitClassifier
    prob_lr, prob_relogit, y_test3 = demo_relogit_classifier()

    # Demo 4: Intercept correction
    demo_intercept_correction()

    # Visualization
    print("\n" + "="*80)
    print("VISUALIZATIONS")
    print("="*80)

    print("\nGenerating calibration plots...")

    # Plot 1: Probability correction
    fig1 = plot_calibration_curves(
        prob_uncorr1, prob_corr1, y_test1,
        title="Demo 1: Probability Correction (5% Positive Class)"
    )
    fig1.savefig('rare_events_demo_calibration.png', dpi=150, bbox_inches='tight')
    print("  Saved: rare_events_demo_calibration.png")

    # Plot 2: Relogit vs Standard LR
    fig2 = plot_calibration_curves(
        prob_lr, prob_relogit, y_test3,
        title="Demo 3: RelogitClassifier vs Standard LR (10% Positive Class)"
    )
    fig2.savefig('rare_events_demo_relogit.png', dpi=150, bbox_inches='tight')
    print("  Saved: rare_events_demo_relogit.png")

    plt.show()

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. Standard LR overestimates probabilities on rare events data")
    print("  2. King & Zeng correction improves calibration (lower Brier score)")
    print("  3. RelogitClassifier applies corrections automatically")
    print("  4. Corrections preserve discrimination (ROC AUC)")
    print("  5. Most important when classes are highly imbalanced (<10% positive)")


if __name__ == "__main__":
    main()
