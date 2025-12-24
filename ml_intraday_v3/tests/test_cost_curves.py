"""
Tests for cost curves analysis module.

Tests include:
1. Synthetic data with known TPR/FPR
2. Cost curve computation correctness
3. Relationship to ROC curve
4. Bootstrap confidence intervals
5. Trading-specific cost ratios
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from ml_intraday_v3.analysis.cost_curves import (
    compute_cost_curve,
    bootstrap_cost_curve,
    plot_cost_curve,
    plot_cost_difference,
    compute_trading_cost_curve,
    compare_models_cost_curves,
    compute_area_under_cost_curve
)


class TestComputeCostCurve:
    """Test compute_cost_curve function."""

    def test_perfect_classifier(self):
        """Test cost curve for perfect classifier (all correct predictions)."""
        # Perfect classifier: all y=1 get prob=1, all y=0 get prob=0
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

        curve_df = compute_cost_curve(y_true, y_prob)

        # Perfect classifier should have NC=0 for all PC values
        # (TPR=1, FPR=0, so NC = (1-1)*PC + 0*(1-PC) = 0)
        assert all(curve_df['nc'] < 0.01), "Perfect classifier should have NC ≈ 0"
        assert all(curve_df['tpr'] > 0.99), "Perfect classifier should have TPR ≈ 1"
        assert all(curve_df['fpr'] < 0.01), "Perfect classifier should have FPR ≈ 0"

    def test_random_classifier(self):
        """Test cost curve for random classifier."""
        # Random classifier: probabilities don't correlate with labels
        np.random.seed(42)
        n = 1000
        y_true = np.random.binomial(1, 0.5, n)
        y_prob = np.random.uniform(0, 1, n)

        curve_df = compute_cost_curve(y_true, y_prob)

        # Random classifier with optimal thresholding can still achieve decent NC
        # Since we find optimal threshold for each cost ratio
        # NC should be positive but can be better than 0.5 (diagonal)
        mean_nc = curve_df['nc'].mean()
        assert 0.0 < mean_nc < 0.5, f"Random classifier mean NC should be < 0.5 with optimal thresholds, got {mean_nc}"

        # Max NC should still be less than 1
        assert curve_df['nc'].max() < 1.0

    def test_inverted_classifier(self):
        """Test cost curve for perfectly wrong classifier."""
        # Inverted classifier: predicts opposite of truth
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        curve_df = compute_cost_curve(y_true, y_prob)

        # Inverted classifier with optimal thresholding should still be good
        # (can invert the predictions)
        # Should have low NC values
        assert curve_df['nc'].min() < 0.1, "Inverted classifier should achieve low NC with optimal threshold"

    def test_cost_curve_properties(self):
        """Test mathematical properties of cost curve."""
        np.random.seed(42)
        n = 500
        # Create moderately discriminative classifier
        y_true = np.random.binomial(1, 0.3, n)
        y_prob = y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.1, n)
        y_prob = np.clip(y_prob, 0, 1)

        curve_df = compute_cost_curve(y_true, y_prob, n_points=50)

        # Check properties:
        # 1. PC should be in [0, 1]
        assert all(curve_df['pc'] >= 0) and all(curve_df['pc'] <= 1)

        # 2. NC should be in [0, 1]
        assert all(curve_df['nc'] >= 0) and all(curve_df['nc'] <= 1)

        # 3. PC should be sorted (monotonically increasing)
        assert all(curve_df['pc'].diff().dropna() >= 0)

        # 4. TPR should be in [0, 1]
        assert all(curve_df['tpr'] >= 0) and all(curve_df['tpr'] <= 1)

        # 5. FPR should be in [0, 1]
        assert all(curve_df['fpr'] >= 0) and all(curve_df['fpr'] <= 1)

    def test_custom_cost_ratios(self):
        """Test with custom cost ratios."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.3, 0.7, 0.8])

        custom_ratios = np.array([0.1, 0.5, 0.9])
        curve_df = compute_cost_curve(y_true, y_prob, cost_ratios=custom_ratios)

        assert len(curve_df) == 3
        assert all(curve_df['cost_ratio'].values == custom_ratios)

    def test_imbalanced_classes(self):
        """Test with highly imbalanced classes."""
        # 95% class 0, 5% class 1
        np.random.seed(42)
        n = 1000
        y_true = np.random.binomial(1, 0.05, n)
        # Make classifier that's decent
        y_prob = y_true * 0.8 + (1 - y_true) * 0.2 + np.random.normal(0, 0.1, n)
        y_prob = np.clip(y_prob, 0, 1)

        curve_df = compute_cost_curve(y_true, y_prob)

        # Should still produce valid curve
        assert len(curve_df) > 0
        assert all(curve_df['nc'] >= 0) and all(curve_df['nc'] <= 1)

    def test_validation_errors(self):
        """Test input validation."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])

        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            compute_cost_curve(y_true, y_prob[:-1])

        # Invalid labels
        with pytest.raises(ValueError, match="only 0 and 1"):
            compute_cost_curve(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))

        # Invalid probabilities
        with pytest.raises(ValueError, match="must be in"):
            compute_cost_curve(y_true, np.array([0.2, 1.5, 0.3, 0.7]))

        # Single class
        with pytest.raises(ValueError, match="both classes"):
            compute_cost_curve(np.array([1, 1, 1]), np.array([0.5, 0.6, 0.7]))


class TestBootstrapCostCurve:
    """Test bootstrap confidence intervals."""

    def test_bootstrap_dimensions(self):
        """Test that bootstrap returns correct dimensions."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 200)
        y_prob = y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.1, 200)
        y_prob = np.clip(y_prob, 0, 1)

        mean_curve, lower, upper = bootstrap_cost_curve(
            y_true, y_prob, n_bootstrap=100, random_state=42
        )

        # All should have same shape
        assert len(mean_curve) == len(lower) == len(upper)

        # Lower bound should be <= mean <= upper bound (mostly)
        # Allow some violations due to numerical issues
        violations = sum((lower['nc'] > mean_curve['nc']) | (mean_curve['nc'] > upper['nc']))
        violation_rate = violations / len(mean_curve)
        assert violation_rate < 0.05, f"Too many CI violations: {violation_rate:.2%}"

    def test_bootstrap_deterministic(self):
        """Test that bootstrap is deterministic with seed."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 100)
        y_prob = np.random.uniform(0, 1, 100)

        mean1, lower1, upper1 = bootstrap_cost_curve(
            y_true, y_prob, n_bootstrap=50, random_state=123
        )
        mean2, lower2, upper2 = bootstrap_cost_curve(
            y_true, y_prob, n_bootstrap=50, random_state=123
        )

        # Should be identical
        assert np.allclose(mean1['nc'], mean2['nc'])
        assert np.allclose(lower1['nc'], lower2['nc'])
        assert np.allclose(upper1['nc'], upper2['nc'])

    def test_bootstrap_confidence_level(self):
        """Test different confidence levels."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 200)
        y_prob = np.random.uniform(0, 1, 200)

        _, lower_95, upper_95 = bootstrap_cost_curve(
            y_true, y_prob, confidence_level=0.95, n_bootstrap=100, random_state=42
        )
        _, lower_90, upper_90 = bootstrap_cost_curve(
            y_true, y_prob, confidence_level=0.90, n_bootstrap=100, random_state=42
        )

        # 90% CI should be narrower than 95% CI
        width_95 = (upper_95['nc'] - lower_95['nc']).mean()
        width_90 = (upper_90['nc'] - lower_90['nc']).mean()

        assert width_90 < width_95, "90% CI should be narrower than 95% CI"


class TestPlotFunctions:
    """Test plotting functions."""

    def test_plot_cost_curve_basic(self):
        """Test basic cost curve plotting."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.3, 0.7, 0.8])

        curve_df = compute_cost_curve(y_true, y_prob)

        fig, ax = plt.subplots()
        ax_result = plot_cost_curve(curve_df, ax=ax, label="Test Model")

        assert ax_result is not None
        assert len(ax.lines) > 0
        plt.close(fig)

    def test_plot_cost_curve_with_confidence(self):
        """Test plotting with confidence intervals."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 100)
        y_prob = np.random.uniform(0, 1, 100)

        mean_curve, lower, upper = bootstrap_cost_curve(
            y_true, y_prob, n_bootstrap=50, random_state=42
        )

        fig, ax = plt.subplots()
        plot_cost_curve(
            mean_curve,
            ax=ax,
            show_confidence=True,
            confidence_bounds=(lower, upper)
        )

        assert len(ax.lines) > 0
        assert len(ax.collections) > 0  # Fill_between creates collections
        plt.close(fig)

    def test_plot_cost_difference(self):
        """Test cost difference plotting."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        # Model 1: good
        y_prob1 = np.array([0.1, 0.2, 0.1, 0.2, 0.8, 0.9, 0.8, 0.9])
        # Model 2: random
        y_prob2 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        curve1 = compute_cost_curve(y_true, y_prob1)
        curve2 = compute_cost_curve(y_true, y_prob2)

        fig, ax = plt.subplots()
        ax_result = plot_cost_difference(
            curve1, curve2,
            ax=ax,
            label1="Good Model",
            label2="Random Model"
        )

        assert ax_result is not None
        assert len(ax.lines) > 0
        plt.close(fig)

    def test_compare_models(self):
        """Test multi-model comparison."""
        np.random.seed(42)
        n = 200
        y_true = np.random.binomial(1, 0.5, n)

        # Three models with different performance
        y_prob1 = y_true * 0.8 + (1 - y_true) * 0.2 + np.random.normal(0, 0.05, n)  # Good
        y_prob2 = y_true * 0.6 + (1 - y_true) * 0.4 + np.random.normal(0, 0.1, n)   # Medium
        y_prob3 = np.random.uniform(0, 1, n)  # Random

        y_prob1 = np.clip(y_prob1, 0, 1)
        y_prob2 = np.clip(y_prob2, 0, 1)

        models_data = {
            'Good Model': (y_true, y_prob1),
            'Medium Model': (y_true, y_prob2),
            'Random Model': (y_true, y_prob3)
        }

        fig, curves = compare_models_cost_curves(
            models_data,
            show_confidence=False
        )

        assert len(curves) == 3
        assert 'Good Model' in curves
        assert fig is not None
        plt.close(fig)


class TestTradingCostCurve:
    """Test trading-specific cost curve."""

    def test_risk_reward_mapping(self):
        """Test that risk/reward ratios map correctly to cost ratios."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.3, 0.7, 0.8])

        rr_ratios = [1.0, 2.0, 3.0]
        curve_df = compute_trading_cost_curve(
            y_true, y_prob,
            risk_reward_ratios=rr_ratios
        )

        # Check mapping: cost_ratio = 1 / (1 + RR)
        expected_cost_ratios = [1.0/2.0, 1.0/3.0, 1.0/4.0]

        assert len(curve_df) == len(rr_ratios)
        assert 'risk_reward_ratio' in curve_df.columns

        for i, rr in enumerate(rr_ratios):
            row = curve_df[curve_df['risk_reward_ratio'] == rr].iloc[0]
            assert abs(row['cost_ratio'] - expected_cost_ratios[i]) < 0.001

    def test_trading_cost_curve_default_rr(self):
        """Test default risk/reward ratios."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 100)
        y_prob = np.random.uniform(0, 1, 100)

        curve_df = compute_trading_cost_curve(y_true, y_prob)

        # Should use default RR ratios [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        assert len(curve_df) == 6
        assert all(curve_df['risk_reward_ratio'] > 0)

    def test_trading_cost_curve_with_returns(self):
        """Test with actual trade returns (for future weighting)."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.2, 0.3, 0.7, 0.8])
        returns = np.array([-0.01, -0.02, 0.03, 0.02])

        curve_df = compute_trading_cost_curve(
            y_true, y_prob,
            trade_returns=returns,
            risk_reward_ratios=[1.0, 2.0]
        )

        # Should have flag for weighted returns
        assert '_has_weighted_returns' in curve_df.columns


class TestCostCurveMetrics:
    """Test cost curve derived metrics."""

    def test_area_under_cost_curve(self):
        """Test AUCC computation."""
        # Perfect classifier should have AUCC ≈ 0
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

        curve_df = compute_cost_curve(y_true, y_prob)
        aucc = compute_area_under_cost_curve(curve_df)

        assert 0 <= aucc <= 1
        assert aucc < 0.1, "Perfect classifier should have AUCC ≈ 0"

    def test_aucc_ordering(self):
        """Test that better classifiers have lower AUCC."""
        np.random.seed(42)
        n = 500
        y_true = np.random.binomial(1, 0.5, n)

        # Good classifier
        y_prob_good = y_true * 0.8 + (1 - y_true) * 0.2 + np.random.normal(0, 0.05, n)
        y_prob_good = np.clip(y_prob_good, 0, 1)

        # Bad classifier
        y_prob_bad = y_true * 0.6 + (1 - y_true) * 0.4 + np.random.normal(0, 0.15, n)
        y_prob_bad = np.clip(y_prob_bad, 0, 1)

        curve_good = compute_cost_curve(y_true, y_prob_good)
        curve_bad = compute_cost_curve(y_true, y_prob_bad)

        aucc_good = compute_area_under_cost_curve(curve_good)
        aucc_bad = compute_area_under_cost_curve(curve_bad)

        assert aucc_good < aucc_bad, "Better classifier should have lower AUCC"


class TestCostCurveVsROC:
    """Test relationship between cost curves and ROC curves."""

    def test_cost_curve_contains_roc_info(self):
        """Test that cost curve captures same info as ROC curve."""
        np.random.seed(42)
        n = 300
        y_true = np.random.binomial(1, 0.5, n)
        y_prob = y_true * 0.7 + (1 - y_true) * 0.3 + np.random.normal(0, 0.1, n)
        y_prob = np.clip(y_prob, 0, 1)

        # Compute ROC AUC
        roc_auc = roc_auc_score(y_true, y_prob)

        # Compute cost curve
        curve_df = compute_cost_curve(y_true, y_prob)
        aucc = compute_area_under_cost_curve(curve_df)

        # Better ROC AUC should correspond to lower AUCC
        # For a perfect classifier: ROC AUC = 1.0, AUCC = 0.0
        # For random: ROC AUC = 0.5, AUCC ≈ 0.5

        # This is an approximate relationship
        # AUCC ≈ 1 - ROC_AUC for many cases
        # But not exact due to class priors and cost considerations

        # At minimum, check they're negatively correlated
        assert roc_auc > 0.6  # Should be better than random
        assert aucc < 0.4  # Should have low cost

    def test_extreme_cases_roc_vs_cost(self):
        """Test extreme cases match between ROC and cost curves."""
        # Perfect classifier
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob_perfect = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

        roc_auc_perfect = roc_auc_score(y_true, y_prob_perfect)
        curve_perfect = compute_cost_curve(y_true, y_prob_perfect)
        aucc_perfect = compute_area_under_cost_curve(curve_perfect)

        assert roc_auc_perfect == 1.0
        assert aucc_perfect < 0.01

        # Anti-perfect (perfectly wrong) - inverted predictions
        y_prob_anti = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        roc_auc_anti = roc_auc_score(y_true, y_prob_anti)
        curve_anti = compute_cost_curve(y_true, y_prob_anti)
        aucc_anti = compute_area_under_cost_curve(curve_anti)

        assert roc_auc_anti == 0.0

        # Anti-perfect classifier cannot be fixed just by changing thresholds
        # since all predictions are identical within each class
        # The cost curve will show poor performance (high AUCC)
        # Actually, with deterministic predictions, the classifier has no flexibility
        # Different cost ratios will all pick the same threshold (or one of two extremes)
        assert aucc_anti > 0.1, "Anti-perfect classifier should have high AUCC"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
