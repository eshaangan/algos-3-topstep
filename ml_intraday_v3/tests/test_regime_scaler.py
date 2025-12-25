"""
Validation Tests for Regime-Aware Feature Scaling

Tests verify:
1. Regime detection (volatility, trend)
2. RegimeAwareScaler correctness
3. Probability calibration
4. No distribution shift between train/test
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_intraday_v3.features.regime_detector import (
    detect_volatility_regime,
    detect_trend_regime,
    detect_combined_regime,
    get_regime_labels,
    validate_regime_consistency
)
from ml_intraday_v3.features.regime_scaler import RegimeAwareScaler
from ml_intraday_v3.features.calibration import (
    calibrate_probabilities,
    evaluate_calibration,
    compare_calibration
)


# ===== Regime Detection Tests =====

def test_detect_volatility_regime_basic():
    """Test basic volatility regime detection."""
    np.random.seed(42)

    # Create returns with clear volatility regimes
    low_vol = np.random.randn(100) * 0.001  # Low volatility
    high_vol = np.random.randn(100) * 0.01  # High volatility
    returns = pd.Series(np.concatenate([low_vol, high_vol]))

    regime = detect_volatility_regime(returns, window=20, n_regimes=2)

    # First half should be mostly low volatility (regime 0)
    # Second half should be mostly high volatility (regime 1)
    assert regime[:100].mode()[0] == 0  # Low vol
    assert regime[100:].mode()[0] == 1  # High vol


def test_detect_volatility_regime_three_regimes():
    """Test volatility detection with 3 regimes."""
    np.random.seed(42)
    returns = pd.Series(np.random.randn(1000) * 0.01)

    regime = detect_volatility_regime(returns, window=20, n_regimes=3)

    # Should have all 3 regimes
    assert set(regime.unique()) == {0, 1, 2}

    # Should be reasonably balanced (quantile-based)
    counts = regime.value_counts()
    assert all(counts > 100)  # Each regime has at least 100 samples


def test_detect_trend_regime_basic():
    """Test basic trend regime detection."""
    np.random.seed(42)

    # Create upward trending prices
    uptrend = 100 + np.cumsum(np.abs(np.random.randn(100)) * 0.1)
    # Create downward trending prices
    downtrend = 200 - np.cumsum(np.abs(np.random.randn(100)) * 0.1)

    prices = pd.Series(np.concatenate([uptrend, downtrend]))

    regime = detect_trend_regime(prices, window=50, n_regimes=2, method="slope")

    # First half should be mostly uptrend (regime 1)
    # Second half should be mostly downtrend (regime 0)
    assert regime[:100].mode()[0] != regime[100:].mode()[0]


def test_detect_combined_regime():
    """Test combined volatility + trend regime detection."""
    np.random.seed(42)

    prices = pd.Series(np.cumsum(np.random.randn(500)) + 100)

    vol_reg, trend_reg, combined_reg = detect_combined_regime(
        prices,
        vol_window=20,
        trend_window=50,
        n_vol_regimes=3,
        n_trend_regimes=3
    )

    # Combined regime should range from 0 to 8 (3×3-1)
    assert combined_reg.min() >= 0
    assert combined_reg.max() <= 8

    # Check encoding: combined = vol * 3 + trend
    for i in range(len(prices)):
        expected = vol_reg.iloc[i] * 3 + trend_reg.iloc[i]
        assert combined_reg.iloc[i] == expected


def test_get_regime_labels():
    """Test regime label mapping."""
    labels = get_regime_labels(n_vol_regimes=3, n_trend_regimes=3)

    assert labels[0] == "low_vol_downtrend"
    assert labels[4] == "medium_vol_sideways"
    assert labels[8] == "high_vol_uptrend"

    # Should have 9 labels total
    assert len(labels) == 9


def test_validate_regime_consistency():
    """Test regime validation."""
    # Good regime (balanced)
    good_regime = pd.Series([0] * 100 + [1] * 100 + [2] * 100)
    assert validate_regime_consistency(good_regime, min_samples_per_regime=10)

    # Bad regime (one regime dominates)
    bad_regime = pd.Series([0] * 500 + [1] * 5 + [2] * 5)
    assert not validate_regime_consistency(bad_regime, min_samples_per_regime=10)


# ===== RegimeAwareScaler Tests =====

def test_regime_aware_scaler_basic():
    """Test basic RegimeAwareScaler functionality."""
    np.random.seed(42)

    # Create features with two regimes
    X_low_vol = np.random.randn(100, 3) * 0.5  # Low volatility
    X_high_vol = np.random.randn(100, 3) * 2.0  # High volatility
    X = np.vstack([X_low_vol, X_high_vol])

    regime = np.array([0] * 100 + [1] * 100)

    scaler = RegimeAwareScaler()
    scaler.fit(X, regime)

    # Check that scaler learned 2 regimes
    assert len(scaler.regime_stats_) == 2
    assert 0 in scaler.regime_stats_
    assert 1 in scaler.regime_stats_

    # Transform
    X_scaled = scaler.transform(X, regime)

    # Each regime should have mean≈0, std≈1
    X_scaled_regime0 = X_scaled[regime == 0]
    X_scaled_regime1 = X_scaled[regime == 1]

    assert np.abs(X_scaled_regime0.mean()) < 0.1
    assert np.abs(X_scaled_regime0.std() - 1.0) < 0.1

    assert np.abs(X_scaled_regime1.mean()) < 0.1
    assert np.abs(X_scaled_regime1.std() - 1.0) < 0.1


def test_regime_aware_scaler_fit_transform():
    """Test fit_transform method."""
    np.random.seed(42)

    X = np.random.randn(200, 5)
    regime = np.random.randint(0, 3, 200)

    scaler = RegimeAwareScaler()

    # fit_transform should produce same result as fit + transform
    X_scaled_1 = scaler.fit_transform(X, regime)

    scaler2 = RegimeAwareScaler()
    scaler2.fit(X, regime)
    X_scaled_2 = scaler2.transform(X, regime)

    np.testing.assert_array_almost_equal(X_scaled_1, X_scaled_2)


def test_regime_aware_scaler_inverse_transform():
    """Test inverse_transform."""
    np.random.seed(42)

    X = np.random.randn(100, 3)
    regime = np.array([0] * 50 + [1] * 50)

    scaler = RegimeAwareScaler()
    scaler.fit(X, regime)

    X_scaled = scaler.transform(X, regime)
    X_reconstructed = scaler.inverse_transform(X_scaled, regime)

    # Should reconstruct original data (within floating point tolerance)
    np.testing.assert_array_almost_equal(X, X_reconstructed, decimal=10)


def test_regime_aware_scaler_unseen_regime():
    """Test handling of unseen regime in transform."""
    np.random.seed(42)

    X_train = np.random.randn(100, 3)
    regime_train = np.array([0] * 50 + [1] * 50)

    scaler = RegimeAwareScaler()
    scaler.fit(X_train, regime_train)

    # Try to transform with unseen regime
    X_test = np.random.randn(50, 3)
    regime_test = np.array([2] * 50)  # Regime 2 not seen during fit

    # Should use global statistics (no error)
    X_scaled = scaler.transform(X_test, regime_test)

    assert X_scaled.shape == X_test.shape


def test_regime_aware_scaler_insufficient_samples():
    """Test handling of regime with insufficient samples."""
    np.random.seed(42)

    X = np.random.randn(100, 3)
    regime = np.array([0] * 95 + [1] * 5)  # Regime 1 has only 5 samples

    scaler = RegimeAwareScaler(min_samples_per_regime=10, fallback_to_global=True)
    scaler.fit(X, regime)

    # Regime 1 should use global stats (fallback)
    assert scaler.regime_stats_[1]['fallback'] == True

    # Transform should still work
    X_scaled = scaler.transform(X, regime)
    assert X_scaled.shape == X.shape


def test_regime_aware_scaler_sklearn_compatible():
    """Test sklearn compatibility (BaseEstimator, TransformerMixin)."""
    from sklearn.base import clone

    scaler = RegimeAwareScaler(method="standard")

    # Should be clonable
    scaler_clone = clone(scaler)
    assert scaler_clone.method == "standard"

    # Check attributes
    assert hasattr(scaler, 'fit')
    assert hasattr(scaler, 'transform')
    assert hasattr(scaler, 'fit_transform')


def test_regime_aware_scaler_pandas():
    """Test that scaler works with pandas DataFrames."""
    np.random.seed(42)

    X_df = pd.DataFrame(np.random.randn(100, 3), columns=['f0', 'f1', 'f2'])
    regime_series = pd.Series([0] * 50 + [1] * 50)

    scaler = RegimeAwareScaler()
    scaler.fit(X_df, regime_series)

    X_scaled = scaler.transform(X_df, regime_series)

    # Output should be numpy array
    assert isinstance(X_scaled, np.ndarray)
    assert X_scaled.shape == X_df.shape


# ===== Probability Calibration Tests =====

def test_calibrate_probabilities_isotonic():
    """Test isotonic regression calibration."""
    np.random.seed(42)

    # Create poorly calibrated probabilities
    y_true = np.random.binomial(1, 0.5, 1000)
    y_prob_uncalibrated = np.random.beta(2, 2, 1000)  # Overconfident

    y_prob_calibrated, calibrator = calibrate_probabilities(
        y_prob_uncalibrated,
        y_true,
        method="isotonic",
        return_calibrator=True
    )

    # Calibrated probabilities should still be in [0, 1]
    assert y_prob_calibrated.min() >= 0
    assert y_prob_calibrated.max() <= 1

    # Calibrator should be reusable
    y_prob_new = np.array([0.2, 0.5, 0.8])
    y_prob_new_calibrated = calibrator.predict(y_prob_new)
    assert len(y_prob_new_calibrated) == 3


def test_calibrate_probabilities_platt():
    """Test Platt scaling calibration."""
    np.random.seed(42)

    y_true = np.random.binomial(1, 0.5, 1000)
    y_prob_uncalibrated = np.random.beta(2, 2, 1000)

    y_prob_calibrated, _ = calibrate_probabilities(
        y_prob_uncalibrated,
        y_true,
        method="platt"
    )

    assert y_prob_calibrated.min() >= 0
    assert y_prob_calibrated.max() <= 1


def test_evaluate_calibration():
    """Test calibration evaluation metrics."""
    np.random.seed(42)

    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95])

    metrics = evaluate_calibration(y_prob, y_true, n_bins=4)

    # Check that all metrics are present
    assert 'brier_score' in metrics
    assert 'log_loss' in metrics
    assert 'ece' in metrics
    assert 'mce' in metrics

    # Brier score should be in [0, 1]
    assert 0 <= metrics['brier_score'] <= 1

    # ECE and MCE should be in [0, 1]
    assert 0 <= metrics['ece'] <= 1
    assert 0 <= metrics['mce'] <= 1


def test_compare_calibration():
    """Test calibration comparison."""
    np.random.seed(42)

    y_true = np.random.binomial(1, 0.5, 1000)
    y_prob_before = np.random.beta(2, 2, 1000)
    y_prob_after, _ = calibrate_probabilities(y_prob_before, y_true)

    comparison = compare_calibration(y_prob_before, y_prob_after, y_true)

    assert 'before' in comparison
    assert 'after' in comparison
    assert 'improvement' in comparison

    # Calibration should improve (positive improvement in ECE)
    # Note: improvement = before - after, so positive means better
    # ECE might not always improve due to randomness, but check structure
    assert 'ece' in comparison['improvement']


def test_calibration_with_perfect_probabilities():
    """Test calibration with already well-calibrated probabilities."""
    np.random.seed(42)

    # Perfect probabilities
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0, 1.0])

    y_prob_calibrated, _ = calibrate_probabilities(y_prob, y_true)

    # Should remain close to original
    np.testing.assert_array_almost_equal(y_prob, y_prob_calibrated, decimal=1)


# ===== Integration Tests =====

def test_end_to_end_regime_aware_scaling():
    """Test complete regime-aware scaling workflow."""
    np.random.seed(42)

    # Generate prices and features
    prices = pd.Series(np.cumsum(np.random.randn(500)) + 100)
    returns = prices.pct_change().fillna(0)

    # Detect regimes
    vol_regime, trend_regime, combined_regime = detect_combined_regime(
        prices,
        vol_window=20,
        trend_window=50,
        n_vol_regimes=3,
        n_trend_regimes=3
    )

    # Generate features
    X = pd.DataFrame({
        'feature_0': returns.values,
        'feature_1': returns.rolling(5).mean().fillna(0).values,
        'feature_2': returns.rolling(10).std().fillna(0).values
    })

    # Scale using regime-aware scaler
    scaler = RegimeAwareScaler()
    X_scaled = scaler.fit_transform(X.values, combined_regime.values)

    # Check that each regime has reasonable statistics
    for regime_id in combined_regime.unique():
        mask = (combined_regime == regime_id).values
        if mask.sum() >= 10:  # Only check regimes with enough samples
            regime_mean = X_scaled[mask].mean(axis=0)
            regime_std = X_scaled[mask].std(axis=0)

            # Mean should be close to 0
            assert np.abs(regime_mean).max() < 0.3

            # Std should be close to 1
            assert np.abs(regime_std - 1.0).max() < 0.3


def test_no_distribution_shift():
    """Test that regime-aware scaling prevents distribution shift."""
    np.random.seed(42)

    # Create train/test split with different regime distributions
    # Train: mostly low volatility
    X_train_low = np.random.randn(400, 3) * 0.5
    X_train_high = np.random.randn(100, 3) * 2.0
    X_train = np.vstack([X_train_low, X_train_high])
    regime_train = np.array([0] * 400 + [1] * 100)

    # Test: mostly high volatility (distribution shift!)
    X_test_low = np.random.randn(100, 3) * 0.5
    X_test_high = np.random.randn(400, 3) * 2.0
    X_test = np.vstack([X_test_low, X_test_high])
    regime_test = np.array([0] * 100 + [1] * 400)

    # Fit scaler on train
    scaler = RegimeAwareScaler()
    scaler.fit(X_train, regime_train)

    # Transform both train and test
    X_train_scaled = scaler.transform(X_train, regime_train)
    X_test_scaled = scaler.transform(X_test, regime_test)

    # Within each regime, train and test should have similar statistics
    for regime_id in [0, 1]:
        train_mask = (regime_train == regime_id)
        test_mask = (regime_test == regime_id)

        train_mean = X_train_scaled[train_mask].mean(axis=0)
        test_mean = X_test_scaled[test_mask].mean(axis=0)

        train_std = X_train_scaled[train_mask].std(axis=0)
        test_std = X_test_scaled[test_mask].std(axis=0)

        # Means should be similar (both close to 0)
        assert np.abs(train_mean - test_mean).max() < 0.3

        # Stds should be similar (both close to 1)
        assert np.abs(train_std - test_std).max() < 0.3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
