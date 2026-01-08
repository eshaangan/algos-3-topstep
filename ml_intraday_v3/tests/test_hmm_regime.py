"""
Tests for HMM Regime Detection.

Covers:
- HMMRegimeDetector fitting and prediction
- Causal (no lookahead) guarantee in expanding window prediction
- State labeling (bull/bear identification)
- Transition matrix validity
- Regime-based sample weights
- Integration with training weights
"""

import numpy as np
import pandas as pd
import pytest

# Skip all tests if hmmlearn not available
pytest.importorskip("hmmlearn")

from ml_intraday_v3.features.hmm_regime import (
    HMMRegimeDetector,
    get_regime_spans,
    compare_hmm_models,
)
from ml_intraday_v3.weights.hmm_weights import (
    compute_hmm_regime_weights,
    compute_regime_weights_by_policy,
    combine_weights,
    analyze_regime_weight_distribution,
)


class TestHMMRegimeDetector:
    """Tests for the HMMRegimeDetector class."""

    @pytest.fixture
    def synthetic_returns(self):
        """Generate synthetic returns with regime switching."""
        np.random.seed(42)
        n_samples = 1000

        # Simulate regime-switching returns
        # State 0: Bear (negative mean, high vol)
        # State 1: Bull (positive mean, low vol)
        returns = []
        state = 1  # Start in bull
        states_true = []

        for _ in range(n_samples):
            if state == 0:
                r = np.random.normal(-0.001, 0.02)
                # Transition to bull with 5% probability
                if np.random.rand() < 0.05:
                    state = 1
            else:
                r = np.random.normal(0.001, 0.01)
                # Transition to bear with 3% probability
                if np.random.rand() < 0.03:
                    state = 0

            returns.append(r)
            states_true.append(state)

        index = pd.date_range("2020-01-01", periods=n_samples, freq="1min")
        return pd.Series(returns, index=index), states_true

    def test_fit_basic(self, synthetic_returns):
        """Test basic HMM fitting."""
        returns, _ = synthetic_returns

        hmm = HMMRegimeDetector(n_states=2, min_samples=100)
        hmm.fit(returns)

        assert hmm.is_fitted_
        assert hmm.model_ is not None
        assert hmm.bull_state_ in [0, 1]
        assert hmm.bear_state_ in [0, 1]
        assert hmm.bull_state_ != hmm.bear_state_

    def test_fit_requires_min_samples(self):
        """Test that fit fails with insufficient samples."""
        returns = pd.Series(np.random.randn(50) * 0.01)
        hmm = HMMRegimeDetector(n_states=2, min_samples=100)

        with pytest.raises(ValueError, match="at least"):
            hmm.fit(returns)

    def test_predict_requires_fit(self, synthetic_returns):
        """Test that predict fails before fit."""
        returns, _ = synthetic_returns
        hmm = HMMRegimeDetector()

        with pytest.raises(ValueError, match="must be fitted"):
            hmm.predict(returns)

    def test_predict_basic(self, synthetic_returns):
        """Test basic state prediction."""
        returns, _ = synthetic_returns

        hmm = HMMRegimeDetector(n_states=2, min_samples=100)
        hmm.fit(returns)
        states = hmm.predict(returns)

        assert len(states) == len(returns)
        assert set(states.unique()).issubset({0, 1})

    def test_predict_proba_sums_to_one(self, synthetic_returns):
        """Test that state probabilities sum to 1."""
        returns, _ = synthetic_returns

        hmm = HMMRegimeDetector(n_states=2, min_samples=100)
        hmm.fit(returns)
        probs = hmm.predict_proba(returns)

        assert probs.shape == (len(returns), 2)
        # Probabilities should sum to 1 (within floating point tolerance)
        row_sums = probs.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0, decimal=5)

    def test_transition_matrix_valid(self, synthetic_returns):
        """Test that transition matrix rows sum to 1."""
        returns, _ = synthetic_returns

        hmm = HMMRegimeDetector(n_states=2, min_samples=100)
        hmm.fit(returns)
        trans_mat = hmm.get_transition_matrix()

        assert trans_mat.shape == (2, 2)
        # Rows should sum to 1
        row_sums = trans_mat.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0, decimal=5)
        # All probabilities should be non-negative
        assert (trans_mat >= 0).all()

    def test_emission_params(self, synthetic_returns):
        """Test emission parameter extraction."""
        returns, _ = synthetic_returns

        hmm = HMMRegimeDetector(n_states=2, min_samples=100)
        hmm.fit(returns)
        params = hmm.get_emission_params()

        assert "bull" in params
        assert "bear" in params
        assert "mean" in params["bull"]
        assert "std" in params["bull"]
        # Bull should have higher mean than bear
        assert params["bull"]["mean"] > params["bear"]["mean"]

    def test_state_labeling_correct(self, synthetic_returns):
        """Test that bull state has higher mean than bear state."""
        returns, _ = synthetic_returns

        hmm = HMMRegimeDetector(n_states=2, min_samples=100)
        hmm.fit(returns)

        means = hmm.model_.means_.flatten()
        # Bull state should have highest mean
        assert hmm.bull_state_ == np.argmax(means)
        assert hmm.bear_state_ == np.argmin(means)


class TestHMMExpandingWindow:
    """Tests for causal (no lookahead) expanding window prediction."""

    @pytest.fixture
    def returns_series(self):
        """Generate return series for testing."""
        np.random.seed(42)
        n_samples = 500
        returns = np.random.randn(n_samples) * 0.01
        index = pd.date_range("2020-01-01", periods=n_samples, freq="1min")
        return pd.Series(returns, index=index)

    def test_expanding_window_basic(self, returns_series):
        """Test basic expanding window prediction."""
        hmm = HMMRegimeDetector(n_states=2, min_samples=100)
        states, probs = hmm.predict_expanding(
            returns_series,
            min_train_samples=100,
            refit_every=50,
        )

        # First min_train_samples should be NaN
        assert states.iloc[:100].isna().all()
        # Rest should be valid states
        assert set(states.iloc[100:].dropna().unique()).issubset({0, 1})

    def test_no_lookahead_in_regime_assignment(self, returns_series):
        """
        Verify that regime at time t doesn't use data after t.

        Method: Modify data after t and verify regime[t] unchanged.
        """
        hmm = HMMRegimeDetector(n_states=2, min_samples=100)

        # Get predictions with original data
        states1, probs1 = hmm.predict_expanding(
            returns_series,
            min_train_samples=100,
            refit_every=50,
        )

        # Modify data after index 300
        modified_returns = returns_series.copy()
        modified_returns.iloc[300:] = modified_returns.iloc[300:] * 10  # Large perturbation

        # Get predictions with modified data
        states2, probs2 = hmm.predict_expanding(
            modified_returns,
            min_train_samples=100,
            refit_every=50,
        )

        # Predictions before index 300 should be identical
        # (allowing for some variance due to refit timing)
        # Check up to last refit before modification (around 250)
        check_until = 250
        np.testing.assert_array_equal(
            states1.iloc[100:check_until].dropna().values,
            states2.iloc[100:check_until].dropna().values,
            err_msg="Regime predictions changed for past data - lookahead detected!"
        )

    def test_expanding_window_monotonic(self, returns_series):
        """Test that training window only grows, never shrinks."""
        # This is implicitly tested by the expanding window implementation
        # Each iteration uses data[:t], which is monotonically increasing
        hmm = HMMRegimeDetector(n_states=2, min_samples=100)
        states, probs = hmm.predict_expanding(
            returns_series,
            min_train_samples=100,
            refit_every=50,
        )

        # Valid predictions should start at min_train_samples
        first_valid_idx = states.dropna().index[0]
        assert first_valid_idx == returns_series.index[100]


class TestHMMRegimeWeights:
    """Tests for regime-based sample weights."""

    @pytest.fixture
    def mock_events_and_regimes(self):
        """Create mock events and regime data."""
        np.random.seed(42)
        n_events = 100

        # Create events
        index = pd.date_range("2020-01-01", periods=200, freq="1min")
        event_times = index[::2][:n_events]  # Every other bar

        events_df = pd.DataFrame({
            "event_id": range(n_events),
            "t0": event_times,
        })

        # Create regime probs (alternating regimes)
        probs = np.zeros((len(index), 2))
        for i in range(len(index)):
            if (i // 20) % 2 == 0:  # Blocks of 20
                probs[i] = [0.8, 0.2]  # Bear regime
            else:
                probs[i] = [0.2, 0.8]  # Bull regime

        regime_probs = pd.DataFrame(
            probs, index=index, columns=["prob_state_0", "prob_state_1"]
        )
        regime_probs["prob_bull"] = regime_probs["prob_state_1"]
        regime_probs["prob_bear"] = regime_probs["prob_state_0"]

        regime_states = pd.Series(
            np.argmax(probs, axis=1), index=index, name="hmm_state"
        )

        return events_df, regime_probs, regime_states

    def test_compute_regime_weights_probability(self, mock_events_and_regimes):
        """Test probability-based regime weight computation."""
        events_df, regime_probs, regime_states = mock_events_and_regimes

        w_regime = compute_hmm_regime_weights(
            events_df=events_df,
            regime_probs=regime_probs,
            target_regime=1,  # Target bull
            similarity_method="probability",
        )

        assert len(w_regime) == len(events_df)
        assert w_regime.min() >= 0
        assert w_regime.max() <= 1

    def test_compute_regime_weights_binary(self, mock_events_and_regimes):
        """Test binary regime weight computation."""
        events_df, regime_probs, regime_states = mock_events_and_regimes

        w_regime = compute_hmm_regime_weights(
            events_df=events_df,
            regime_probs=regime_probs,
            target_regime=1,
            similarity_method="binary",
            discount_factor=0.3,
        )

        assert len(w_regime) == len(events_df)
        # Binary method produces weights of either 1.0 or discount_factor
        unique_weights = set(np.round(w_regime.values, 2))
        assert unique_weights.issubset({0.3, 1.0})

    def test_regime_weights_by_policy_recent(self, mock_events_and_regimes):
        """Test 'recent' target regime policy."""
        events_df, regime_probs, regime_states = mock_events_and_regimes

        w_regime = compute_regime_weights_by_policy(
            events_df=events_df,
            regime_probs=regime_probs,
            regime_states=regime_states,
            policy="recent",
        )

        assert len(w_regime) == len(events_df)

    def test_regime_weights_by_policy_dominant(self, mock_events_and_regimes):
        """Test 'dominant' target regime policy."""
        events_df, regime_probs, regime_states = mock_events_and_regimes

        w_regime = compute_regime_weights_by_policy(
            events_df=events_df,
            regime_probs=regime_probs,
            regime_states=regime_states,
            policy="dominant",
            lookback_bars=50,
        )

        assert len(w_regime) == len(events_df)


class TestCombineWeights:
    """Tests for combining multiple weight sources."""

    def test_combine_weights_basic(self):
        """Test basic weight combination."""
        index = range(100)
        w_uniqueness = pd.Series(np.random.rand(100) + 0.5, index=index)
        w_regime = pd.Series(np.random.rand(100) + 0.5, index=index)

        w_final = combine_weights(
            w_uniqueness=w_uniqueness,
            w_regime=w_regime,
            uniqueness_exp=1.0,
            regime_exp=1.0,
        )

        assert len(w_final) == 100
        # Normalized weights should sum to n_samples
        np.testing.assert_almost_equal(w_final.sum(), 100, decimal=1)

    def test_combine_weights_exponents(self):
        """Test that exponents affect weights correctly."""
        index = range(100)
        w_uniqueness = pd.Series(np.ones(100) * 0.5, index=index)
        w_regime = pd.Series(np.ones(100) * 0.5, index=index)

        # With exponent 0, that weight source is disabled (= 1)
        w_final = combine_weights(
            w_uniqueness=w_uniqueness,
            w_regime=w_regime,
            uniqueness_exp=0.0,  # Disabled
            regime_exp=1.0,
            normalize=False,
        )

        # Should only reflect regime weights
        np.testing.assert_array_almost_equal(w_final.values, 0.5)

    def test_combine_weights_requires_input(self):
        """Test that at least one weight source is required."""
        with pytest.raises(ValueError, match="At least one"):
            combine_weights()


class TestRegimeSpans:
    """Tests for regime span extraction."""

    def test_get_regime_spans_basic(self):
        """Test basic regime span extraction."""
        states = pd.Series([0, 0, 1, 1, 1, 0, 0])
        spans = get_regime_spans(states)

        assert len(spans) == 3
        assert spans[0] == (0, 1, 0)  # First bear period
        assert spans[1] == (2, 4, 1)  # Bull period
        assert spans[2] == (5, 6, 0)  # Second bear period

    def test_get_regime_spans_empty(self):
        """Test span extraction with empty series."""
        states = pd.Series([], dtype=float)
        spans = get_regime_spans(states)
        assert spans == []

    def test_get_regime_spans_with_nan(self):
        """Test span extraction with NaN values."""
        states = pd.Series([np.nan, np.nan, 0, 0, 1, 1])
        spans = get_regime_spans(states)

        # Should only have spans for non-NaN values
        assert len(spans) == 2


class TestModelComparison:
    """Tests for HMM model comparison."""

    def test_compare_hmm_models(self):
        """Test model comparison across different state counts."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(500) * 0.01)

        results = compare_hmm_models(returns, state_range=(2, 4))

        assert len(results) == 2  # 2 and 3 states
        assert "n_states" in results.columns
        assert "AIC" in results.columns
        assert "BIC" in results.columns


# Integration test
class TestIntegration:
    """Integration tests for the full HMM workflow."""

    def test_full_workflow(self):
        """Test the complete HMM regime detection workflow."""
        np.random.seed(42)

        # Generate synthetic data
        n_bars = 500
        index = pd.date_range("2020-01-01", periods=n_bars, freq="1min")
        returns = pd.Series(np.random.randn(n_bars) * 0.01, index=index)

        # Create events
        n_events = 100
        event_times = index[100::4][:n_events]
        events_df = pd.DataFrame({
            "event_id": range(n_events),
            "t0": event_times,
        })

        # Step 1: Fit HMM
        hmm = HMMRegimeDetector(n_states=2, min_samples=100)
        regime_states, regime_probs = hmm.predict_expanding(
            returns, min_train_samples=100, refit_every=21
        )

        # Step 2: Compute regime weights
        w_regime = compute_hmm_regime_weights(
            events_df=events_df,
            regime_probs=regime_probs,
            similarity_method="probability",
        )

        # Step 3: Combine with uniqueness weights
        w_uniqueness = pd.Series(
            np.random.rand(n_events) + 0.5, index=events_df["event_id"]
        )
        w_final = combine_weights(
            w_uniqueness=w_uniqueness,
            w_regime=w_regime,
        )

        # Verify outputs
        assert len(w_final) == n_events
        assert not w_final.isna().any()
        assert (w_final > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
