"""
Tests for rare events corrections module.

Tests the implementation of King & Zeng (2001) rare events corrections
for logistic regression.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

from ml_intraday_v3.training.rare_events import (
    correct_rare_events_probabilities,
    compute_rare_event_weights,
    RelogitClassifier,
    apply_prior_correction_to_intercept,
    estimate_population_prior
)


class TestCorrectRareEventsProbabilities:
    """Test probability correction function."""

    def test_basic_correction(self):
        """Test basic probability correction."""
        # Rare event: 20% positive
        y_true = np.array([0, 0, 0, 0, 1])
        y_prob = np.array([0.3, 0.4, 0.2, 0.5, 0.8])
        tau = 0.2

        corrected = correct_rare_events_probabilities(y_prob, y_true, tau=tau)

        # Corrected probabilities should be lower (rare event)
        assert all(corrected <= y_prob)

        # Should still be valid probabilities
        assert all(corrected >= 0) and all(corrected <= 1)

    def test_auto_tau_estimation(self):
        """Test automatic tau estimation."""
        y_true = np.array([0, 0, 0, 1, 0, 0, 1, 0])  # 25% positive
        y_prob = np.array([0.5] * len(y_true))

        corrected = correct_rare_events_probabilities(y_prob, y_true, tau=None)

        # Should use tau = 0.25 (mean of y_true)
        expected_tau = 0.25
        expected = correct_rare_events_probabilities(y_prob, y_true, tau=expected_tau)

        assert np.allclose(corrected, expected)

    def test_correction_formula(self):
        """Test King & Zeng correction formula."""
        # Known example
        p = 0.5
        tau = 0.1

        corrected = correct_rare_events_probabilities(
            np.array([p]),
            np.array([1]),  # Dummy
            tau=tau
        )[0]

        # Manual calculation: (P * τ) / (P * τ + (1-P) * (1-τ))
        expected = (p * tau) / (p * tau + (1 - p) * (1 - tau))

        assert np.isclose(corrected, expected)

    def test_edge_cases(self):
        """Test edge cases."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.0, 1.0, 0.5, 0.5])
        tau = 0.5

        corrected = correct_rare_events_probabilities(y_prob, y_true, tau=tau)

        # When tau=0.5 (balanced), correction should be identity
        assert np.allclose(corrected, y_prob)

    def test_validation_errors(self):
        """Test input validation."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.7, 0.2, 0.8])

        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            correct_rare_events_probabilities(y_prob, y_true[:-1])

        # Invalid probabilities
        with pytest.raises(ValueError, match="must be in"):
            correct_rare_events_probabilities(
                np.array([0.3, 1.5, 0.2, 0.8]),
                y_true
            )

        # Invalid labels
        with pytest.raises(ValueError, match="only 0 and 1"):
            correct_rare_events_probabilities(
                y_prob,
                np.array([0, 1, 2, 1])
            )

        # Invalid tau
        with pytest.raises(ValueError, match="must be in"):
            correct_rare_events_probabilities(y_prob, y_true, tau=0.0)

        with pytest.raises(ValueError, match="must be in"):
            correct_rare_events_probabilities(y_prob, y_true, tau=1.0)

    def test_simple_method(self):
        """Test simple correction method."""
        y_true = np.array([0, 0, 0, 1])  # 25% positive
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])  # Mean = 0.5
        tau = 0.25

        corrected = correct_rare_events_probabilities(
            y_prob, y_true, tau=tau, method="simple"
        )

        # Simple method rescales to match tau
        # Mean should be tau
        assert np.isclose(np.mean(corrected), tau)


class TestComputeRareEventWeights:
    """Test sample weighting function."""

    def test_king_zeng_weights(self):
        """Test King & Zeng weighting."""
        y_true = np.array([0, 0, 0, 1, 0, 0, 1, 0])
        tau = 0.25  # True prior

        weights = compute_rare_event_weights(y_true, method="king_zeng", tau=tau)

        # w_i = 1 for y=1
        # w_i = τ/(1-τ) for y=0
        expected_w0 = tau / (1 - tau)  # 0.25 / 0.75 = 1/3
        expected_w1 = 1.0

        assert np.allclose(weights[y_true == 0], expected_w0)
        assert np.allclose(weights[y_true == 1], expected_w1)

    def test_inverse_freq_weights(self):
        """Test inverse frequency weighting."""
        y_true = np.array([0, 0, 0, 1])  # 3 neg, 1 pos

        weights = compute_rare_event_weights(y_true, method="inverse_freq")

        # Should give equal total weight to each class
        total_weight_0 = np.sum(weights[y_true == 0])
        total_weight_1 = np.sum(weights[y_true == 1])

        assert np.isclose(total_weight_0, total_weight_1)

    def test_balanced_weights(self):
        """Test balanced class weighting."""
        y_true = np.array([0, 0, 0, 1])

        weights = compute_rare_event_weights(y_true, method="balanced")

        # Balanced should give equal total weight
        total_weight_0 = np.sum(weights[y_true == 0])
        total_weight_1 = np.sum(weights[y_true == 1])

        assert np.isclose(total_weight_0, total_weight_1)

    def test_validation_errors(self):
        """Test input validation."""
        # Invalid labels
        with pytest.raises(ValueError, match="only 0 and 1"):
            compute_rare_event_weights(np.array([0, 1, 2]))

        # Single class
        with pytest.raises(ValueError, match="both classes"):
            compute_rare_event_weights(np.array([1, 1, 1]))


class TestRelogitClassifier:
    """Test RelogitClassifier."""

    def test_basic_fit_predict(self):
        """Test basic fitting and prediction."""
        # Create imbalanced dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            weights=[0.9, 0.1],  # 10% positive
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit relogit
        clf = RelogitClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        # Check outputs
        assert len(y_pred) == len(X_test)
        assert y_prob.shape == (len(X_test), 2)
        assert np.all((y_prob >= 0) & (y_prob <= 1))
        assert np.allclose(y_prob.sum(axis=1), 1.0)

    def test_tau_estimation(self):
        """Test automatic tau estimation."""
        X, y = make_classification(
            n_samples=500,
            n_features=5,
            weights=[0.8, 0.2],
            random_state=42
        )

        clf = RelogitClassifier(tau=None, random_state=42)
        clf.fit(X, y)

        # Should estimate tau from training data
        expected_tau = np.mean(y)
        assert np.isclose(clf.tau_, expected_tau)

    def test_probability_correction(self):
        """Test that probabilities are corrected."""
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            weights=[0.95, 0.05],  # Very imbalanced
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Standard logistic regression
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train, y_train)
        prob_standard = lr.predict_proba(X_test)[:, 1]

        # Relogit
        relogit = RelogitClassifier(random_state=42)
        relogit.fit(X_train, y_train)
        prob_corrected = relogit.predict_proba(X_test)[:, 1]

        # Corrected probabilities should generally be lower for rare events
        # (since standard LR tends to overestimate rare event probabilities)
        assert np.mean(prob_corrected) <= np.mean(prob_standard)

    def test_sklearn_compatibility(self):
        """Test sklearn compatibility."""
        from sklearn.model_selection import cross_val_score

        X, y = make_classification(
            n_samples=500,
            n_features=10,
            weights=[0.9, 0.1],
            random_state=42
        )

        clf = RelogitClassifier(random_state=42)

        # Should work with cross_val_score
        scores = cross_val_score(clf, X, y, cv=3, scoring='roc_auc')

        assert len(scores) == 3
        assert all(scores >= 0) and all(scores <= 1)

    def test_without_sample_weights(self):
        """Test relogit without sample weighting."""
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            weights=[0.9, 0.1],
            random_state=42
        )

        clf = RelogitClassifier(
            use_sample_weights=False,
            random_state=42
        )
        clf.fit(X, y)

        # Should still fit and predict
        y_prob = clf.predict_proba(X)[:, 1]

        assert len(y_prob) == len(X)
        assert all(y_prob >= 0) and all(y_prob <= 1)

    def test_decision_function(self):
        """Test decision function."""
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            weights=[0.9, 0.1],
            random_state=42
        )

        clf = RelogitClassifier(random_state=42)
        clf.fit(X, y)

        scores = clf.decision_function(X)

        # Decision function should be log-odds
        assert len(scores) == len(X)
        # Higher scores should correspond to higher probabilities
        proba = clf.predict_proba(X)[:, 1]
        assert np.corrcoef(scores, proba)[0, 1] > 0.9


class TestPriorCorrectionToIntercept:
    """Test intercept correction function."""

    def test_basic_correction(self):
        """Test basic intercept correction."""
        # Population has 5% positive, sample has 50% (balanced)
        intercept = 0.0
        tau_population = 0.05
        y_sample_mean = 0.50

        corrected = apply_prior_correction_to_intercept(
            intercept, tau_population, y_sample_mean
        )

        # Should shift intercept down (to account for oversampling positives)
        assert corrected < intercept

    def test_no_correction_needed(self):
        """Test when sample matches population."""
        intercept = 1.5
        tau = 0.3

        corrected = apply_prior_correction_to_intercept(
            intercept, tau, tau  # Same prior
        )

        # No correction needed
        assert np.isclose(corrected, intercept)

    def test_validation_errors(self):
        """Test input validation."""
        # Invalid tau
        with pytest.raises(ValueError, match="must be in"):
            apply_prior_correction_to_intercept(0.0, 0.0, 0.5)

        with pytest.raises(ValueError, match="must be in"):
            apply_prior_correction_to_intercept(0.0, 1.0, 0.5)

        # Invalid sample mean
        with pytest.raises(ValueError, match="must be in"):
            apply_prior_correction_to_intercept(0.0, 0.5, 0.0)


class TestEstimatePopulationPrior:
    """Test prior estimation function."""

    def test_train_method(self):
        """Test estimation from training set."""
        y_train = np.array([0, 0, 0, 1, 0])  # 20% positive

        tau = estimate_population_prior(y_train, method="train")

        assert np.isclose(tau, 0.2)

    def test_val_method(self):
        """Test estimation from validation set."""
        y_train = np.array([0, 0, 1, 1])  # 50%
        y_val = np.array([0, 0, 0, 1])    # 25%

        tau = estimate_population_prior(y_train, y_val, method="val")

        assert np.isclose(tau, 0.25)

    def test_pooled_method(self):
        """Test pooled estimation."""
        y_train = np.array([0, 0, 0, 1])  # 25%
        y_val = np.array([0, 1, 1, 1])    # 75%

        tau = estimate_population_prior(y_train, y_val, method="pooled")

        # Should be 50% (4 pos out of 8 total)
        assert np.isclose(tau, 0.5)

    def test_pooled_without_val(self):
        """Test pooled when val is None."""
        y_train = np.array([0, 0, 0, 1])

        tau = estimate_population_prior(y_train, None, method="pooled")

        # Should fall back to train
        assert np.isclose(tau, 0.25)


class TestRareEventsCalibration:
    """Test that rare events corrections improve calibration."""

    def test_calibration_improvement(self):
        """Test that relogit improves calibration on imbalanced data."""
        # Create highly imbalanced dataset
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            weights=[0.95, 0.05],  # Very imbalanced
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Standard logistic regression
        lr = LogisticRegression(max_iter=200, random_state=42)
        lr.fit(X_train, y_train)
        prob_standard = lr.predict_proba(X_test)[:, 1]

        # Relogit
        relogit = RelogitClassifier(max_iter=200, random_state=42)
        relogit.fit(X_train, y_train)
        prob_relogit = relogit.predict_proba(X_test)[:, 1]

        # Compute calibration curves
        # For rare events, relogit should have better calibration
        # (predicted probabilities closer to observed frequencies)

        # Simple check: mean predicted prob should be closer to observed rate
        observed_rate = np.mean(y_test)
        error_standard = abs(np.mean(prob_standard) - observed_rate)
        error_relogit = abs(np.mean(prob_relogit) - observed_rate)

        # Relogit should have lower error (better calibration)
        assert error_relogit <= error_standard

    def test_on_balanced_data(self):
        """Test that relogit doesn't hurt on balanced data."""
        # Balanced dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            weights=[0.5, 0.5],
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Standard LR
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train, y_train)
        score_lr = lr.score(X_test, y_test)

        # Relogit
        relogit = RelogitClassifier(random_state=42)
        relogit.fit(X_train, y_train)
        score_relogit = relogit.score(X_test, y_test)

        # Scores should be similar on balanced data
        assert abs(score_lr - score_relogit) < 0.05


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
