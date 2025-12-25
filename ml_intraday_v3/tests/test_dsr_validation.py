"""
Validation Tests for Deflated Sharpe Ratio (DSR) Implementation

Tests verify:
1. DSR formula correctness
2. Selection bias adjustment
3. Non-normality adjustments (skewness, kurtosis)
4. Edge cases
"""

import numpy as np
import pytest
from statistics import NormalDist

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_intraday_v3.experiments.diagnostics import compute_dsr


def test_dsr_single_trial_no_bias():
    """
    Test that with n_trials=1, DSR reduces to standard significance test.

    With only 1 trial, SR* = target_sharpe (no selection bias adjustment).
    """
    np.random.seed(42)

    # Generate returns with known Sharpe ~ 1.0
    returns = np.random.normal(0.01, 0.01, size=1000)

    result = compute_dsr(returns, n_trials=1, target_sharpe=0.0)

    assert result['dsr'] is not None
    assert result['n_trials'] == 1
    assert result['sr_star'] == 0.0  # No selection bias with single trial

    # DSR should be close to Φ(SR / σ_SR)
    expected_z = result['sharpe'] / result['sr_std']
    expected_dsr = NormalDist().cdf(expected_z)

    assert abs(result['dsr'] - expected_dsr) < 0.01


def test_dsr_decreases_with_more_trials():
    """
    Test that DSR decreases as n_trials increases (selection bias penalty).

    The same returns should yield lower DSR with more trials tested.
    """
    np.random.seed(42)
    returns = np.random.normal(0.01, 0.01, size=500)

    dsr_1_trial = compute_dsr(returns, n_trials=1)['dsr']
    dsr_10_trials = compute_dsr(returns, n_trials=10)['dsr']
    dsr_100_trials = compute_dsr(returns, n_trials=100)['dsr']

    # DSR should decrease with more trials (higher selection bias penalty)
    assert dsr_1_trial > dsr_10_trials
    assert dsr_10_trials > dsr_100_trials


def test_dsr_sr_star_increases_with_trials():
    """
    Test that SR* (expected max Sharpe from luck) increases with n_trials.
    """
    np.random.seed(42)
    returns = np.random.normal(0.01, 0.01, size=500)

    sr_star_1 = compute_dsr(returns, n_trials=1)['sr_star']
    sr_star_10 = compute_dsr(returns, n_trials=10)['sr_star']
    sr_star_100 = compute_dsr(returns, n_trials=100)['sr_star']

    # SR* should increase with more trials
    assert sr_star_1 < sr_star_10 < sr_star_100


def test_dsr_with_negative_sharpe():
    """
    Test DSR with negative Sharpe ratio.

    Losing strategy should have very low DSR.
    """
    np.random.seed(42)
    # Generate losing returns (negative mean)
    returns = np.random.normal(-0.01, 0.01, size=500)

    result = compute_dsr(returns, n_trials=10)

    assert result['dsr'] is not None
    assert result['sharpe'] < 0
    assert result['dsr'] < 0.5  # Low DSR for losing strategy


def test_dsr_with_high_skewness():
    """
    Test DSR adjusts for non-normal distributions with high skewness.
    """
    np.random.seed(42)

    # Generate right-skewed returns (occasional large wins)
    returns = np.random.lognormal(mean=-0.5, sigma=1.0, size=500) - 1.0

    result = compute_dsr(returns, n_trials=10)

    assert result['dsr'] is not None
    assert abs(result['skewness']) > 0.1  # Should detect skewness

    # σ_SR formula accounts for skewness
    # Manually compute to verify
    sr = result['sharpe']
    gamma = result['skewness']
    kappa = result['kurtosis']
    n = result['n_obs']

    sr_var_expected = (1 - gamma * sr + (kappa - 1) / 4.0 * sr ** 2) / (n - 1)
    sr_std_expected = np.sqrt(sr_var_expected)

    assert abs(result['sr_std'] - sr_std_expected) < 0.001


def test_dsr_with_high_kurtosis():
    """
    Test DSR adjusts for non-normal distributions with high kurtosis (fat tails).
    """
    np.random.seed(42)

    # Generate returns with fat tails (t-distribution)
    from scipy.stats import t
    returns = t.rvs(df=3, size=500) * 0.01

    result = compute_dsr(returns, n_trials=10)

    assert result['dsr'] is not None
    assert result['kurtosis'] > 3.5  # Should detect heavy tails


def test_dsr_annualization():
    """
    Test that annualization factor correctly scales Sharpe ratio.
    """
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, size=500)

    # Without annualization
    result_no_annualization = compute_dsr(returns, n_trials=10, annualization_factor=None)

    # With annualization (e.g., sqrt(252) for daily to annual)
    annualization_factor = np.sqrt(252)
    result_with_annualization = compute_dsr(returns, n_trials=10, annualization_factor=annualization_factor)

    # Annualized Sharpe should be scaled version
    expected_annualized_sharpe = result_no_annualization['sharpe_raw'] * annualization_factor

    assert abs(result_with_annualization['sharpe'] - expected_annualized_sharpe) < 0.01
    assert result_with_annualization['annualization_factor'] == annualization_factor


def test_dsr_target_sharpe():
    """
    Test that target_sharpe parameter correctly adjusts the benchmark.
    """
    np.random.seed(42)
    returns = np.random.normal(0.01, 0.01, size=500)

    # Test against zero (no skill)
    result_target_0 = compute_dsr(returns, n_trials=10, target_sharpe=0.0)

    # Test against higher bar
    result_target_1 = compute_dsr(returns, n_trials=10, target_sharpe=1.0)

    # DSR should be lower when testing against higher bar
    assert result_target_0['dsr'] > result_target_1['dsr']
    assert result_target_0['sr_star'] < result_target_1['sr_star']


def test_dsr_edge_case_insufficient_data():
    """
    Test DSR handles edge case of insufficient data.
    """
    returns = np.array([0.01])  # Single observation

    result = compute_dsr(returns, n_trials=10)

    assert result['dsr'] is None
    assert result['reason'] == 'insufficient_returns'


def test_dsr_edge_case_zero_variance():
    """
    Test DSR handles edge case of zero variance.
    """
    returns = np.ones(100) * 0.01  # All same value

    result = compute_dsr(returns, n_trials=10)

    assert result['dsr'] is None
    assert result['reason'] == 'zero_variance'


def test_dsr_reproducibility():
    """
    Test that DSR computation is reproducible with same inputs.
    """
    np.random.seed(42)
    returns = np.random.normal(0.01, 0.01, size=500)

    result1 = compute_dsr(returns, n_trials=10, target_sharpe=0.0)
    result2 = compute_dsr(returns, n_trials=10, target_sharpe=0.0)

    assert result1['dsr'] == result2['dsr']
    assert result1['sharpe'] == result2['sharpe']
    assert result1['sr_star'] == result2['sr_star']


def test_dsr_known_values():
    """
    Test DSR against known values from Bailey & López de Prado (2014) paper.

    This is a simplified test using synthetic data designed to approximate
    examples from the paper.
    """
    np.random.seed(42)

    # Generate returns with Sharpe ~ 1.5, n=1000
    sharpe_target = 1.5
    returns = np.random.normal(0.015, 0.01, size=1000)

    # With 10 trials, DSR should be lower than naive p-value
    result = compute_dsr(returns, n_trials=10, target_sharpe=0.0)

    # Observed Sharpe should be close to target
    assert abs(result['sharpe'] - sharpe_target) < 0.3

    # DSR should be > 0.5 but < 1.0 for moderate positive Sharpe with 10 trials
    assert 0.5 < result['dsr'] < 1.0

    # With more trials, DSR should decrease
    result_100_trials = compute_dsr(returns, n_trials=100, target_sharpe=0.0)
    assert result_100_trials['dsr'] < result['dsr']


def test_dsr_formula_components():
    """
    Test that all DSR formula components are correctly computed.
    """
    np.random.seed(42)
    returns = np.random.normal(0.01, 0.01, size=500)

    result = compute_dsr(returns, n_trials=10, target_sharpe=0.0)

    # Verify all components exist
    assert 'sharpe' in result
    assert 'sr_star' in result
    assert 'sr_std' in result
    assert 'skewness' in result
    assert 'kurtosis' in result
    assert 'n_obs' in result
    assert 'n_trials' in result
    assert 'z_score' in result
    assert 'dsr' in result

    # Manually verify z-score
    z_score_expected = (result['sharpe'] - result['sr_star']) / result['sr_std']
    assert abs(result['z_score'] - z_score_expected) < 0.0001

    # Manually verify DSR
    dsr_expected = NormalDist().cdf(z_score_expected)
    assert abs(result['dsr'] - dsr_expected) < 0.0001


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
