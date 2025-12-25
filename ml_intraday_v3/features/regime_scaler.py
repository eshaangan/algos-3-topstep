"""
Regime-Aware Feature Scaler

Implements sklearn-compatible scaler that normalizes features per market regime
to prevent distribution shifts between train/test when market conditions change.

Key class:
- RegimeAwareScaler: Fit and transform features using regime-specific statistics

References:
- López de Prado (2018), "Advances in Financial Machine Learning", Chapter 19
- sklearn.preprocessing API for compatibility
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)


class RegimeAwareScaler(BaseEstimator, TransformerMixin):
    """
    Regime-aware feature scaler (sklearn-compatible).

    Fits separate normalization statistics (mean, std) for each market regime,
    then scales features using regime-specific parameters. This prevents
    distribution shift when market conditions change between train/test.

    Parameters
    ----------
    method : str
        Scaling method:
        - "standard": (X - mean) / std per regime
        - "robust": (X - median) / IQR per regime (future)
    min_samples_per_regime : int
        Minimum samples required to fit a regime (default: 10)
        If a regime has fewer samples, fall back to global statistics
    fallback_to_global : bool
        If True, use global statistics for regimes with insufficient samples
        If False, raise error for insufficient samples

    Attributes
    ----------
    regime_stats_ : dict
        Mapping from regime label to dict of statistics:
        {regime_id: {'mean': array, 'std': array, 'n_samples': int}}
    global_stats_ : dict
        Global statistics used as fallback: {'mean': array, 'std': array}
    n_features_in_ : int
        Number of features seen during fit
    regimes_seen_ : set
        Set of regime labels seen during fit

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from ml_intraday_v3.features.regime_detector import detect_volatility_regime
    >>>
    >>> # Generate data with regime shifts
    >>> np.random.seed(42)
    >>> X = pd.DataFrame({
    ...     'feature_0': np.concatenate([
    ...         np.random.randn(100) * 0.5,  # Low vol
    ...         np.random.randn(100) * 2.0   # High vol
    ...     ]),
    ...     'feature_1': np.random.randn(200)
    ... })
    >>>
    >>> # Detect regimes
    >>> prices = pd.Series(np.cumsum(X['feature_0'].values) + 100)
    >>> returns = prices.pct_change().fillna(0)
    >>> regime = detect_volatility_regime(returns, window=20)
    >>>
    >>> # Fit scaler
    >>> scaler = RegimeAwareScaler()
    >>> scaler.fit(X, regime_labels=regime)
    >>>
    >>> # Transform
    >>> X_scaled = scaler.transform(X, regime_labels=regime)
    >>>
    >>> # Check that each regime has mean≈0, std≈1
    >>> for r in regime.unique():
    ...     mask = (regime == r)
    ...     print(f"Regime {r}: mean={X_scaled[mask].mean():.3f}, std={X_scaled[mask].std():.3f}")
    """

    def __init__(
        self,
        method: str = "standard",
        min_samples_per_regime: int = 10,
        fallback_to_global: bool = True
    ):
        self.method = method
        self.min_samples_per_regime = min_samples_per_regime
        self.fallback_to_global = fallback_to_global

        # Fitted attributes (set during fit())
        self.regime_stats_ = {}
        self.global_stats_ = {}
        self.n_features_in_ = None
        self.regimes_seen_ = set()

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        regime_labels: Union[pd.Series, np.ndarray],
        y=None  # Ignored, for sklearn compatibility
    ):
        """
        Fit scaler by computing statistics per regime.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix (n_samples, n_features)
        regime_labels : pd.Series or np.ndarray
            Regime label for each sample (n_samples,)
        y : Ignored
            Not used, present for sklearn compatibility

        Returns
        -------
        self : RegimeAwareScaler
            Fitted scaler
        """
        # Convert to numpy for consistency
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(regime_labels, pd.Series):
            regime_labels = regime_labels.values

        if len(X) != len(regime_labels):
            raise ValueError(
                f"X and regime_labels must have same length: "
                f"{len(X)} != {len(regime_labels)}"
            )

        self.n_features_in_ = X.shape[1]
        unique_regimes = np.unique(regime_labels)
        self.regimes_seen_ = set(unique_regimes)

        # Compute global statistics (fallback)
        if self.method == "standard":
            self.global_stats_ = {
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0) + 1e-8  # Add epsilon to avoid division by zero
            }
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        # Compute per-regime statistics
        self.regime_stats_ = {}

        for regime in unique_regimes:
            mask = (regime_labels == regime)
            X_regime = X[mask]

            n_samples = len(X_regime)

            if n_samples < self.min_samples_per_regime:
                logger.warning(
                    f"Regime {regime} has only {n_samples} samples "
                    f"(< {self.min_samples_per_regime}). "
                    f"{'Using global statistics.' if self.fallback_to_global else 'Error!'}"
                )

                if not self.fallback_to_global:
                    raise ValueError(
                        f"Regime {regime} has insufficient samples ({n_samples})"
                    )

                # Use global stats as fallback
                self.regime_stats_[regime] = {
                    'mean': self.global_stats_['mean'].copy(),
                    'std': self.global_stats_['std'].copy(),
                    'n_samples': n_samples,
                    'fallback': True
                }
            else:
                # Compute regime-specific statistics
                if self.method == "standard":
                    self.regime_stats_[regime] = {
                        'mean': np.mean(X_regime, axis=0),
                        'std': np.std(X_regime, axis=0) + 1e-8,
                        'n_samples': n_samples,
                        'fallback': False
                    }

        logger.info(
            f"Fitted RegimeAwareScaler on {len(unique_regimes)} regimes "
            f"with {self.n_features_in_} features"
        )

        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        regime_labels: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Transform features using regime-specific statistics.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix to transform (n_samples, n_features)
        regime_labels : pd.Series or np.ndarray
            Regime label for each sample (n_samples,)

        Returns
        -------
        X_scaled : np.ndarray
            Scaled features (n_samples, n_features)
        """
        # Check if fitted
        if self.n_features_in_ is None:
            raise ValueError("Scaler not fitted. Call fit() before transform().")

        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(regime_labels, pd.Series):
            regime_labels = regime_labels.values

        if len(X) != len(regime_labels):
            raise ValueError(
                f"X and regime_labels must have same length: "
                f"{len(X)} != {len(regime_labels)}"
            )

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but scaler was fitted on "
                f"{self.n_features_in_} features"
            )

        # Initialize scaled array
        X_scaled = np.zeros_like(X, dtype=np.float64)

        unique_regimes = np.unique(regime_labels)

        for regime in unique_regimes:
            mask = (regime_labels == regime)

            if regime not in self.regime_stats_:
                # Unseen regime - use global statistics
                logger.warning(
                    f"Regime {regime} not seen during fit. Using global statistics."
                )
                mean = self.global_stats_['mean']
                std = self.global_stats_['std']
            else:
                # Use regime-specific statistics
                mean = self.regime_stats_[regime]['mean']
                std = self.regime_stats_[regime]['std']

            # Scale: (X - mean) / std
            X_scaled[mask] = (X[mask] - mean) / std

        return X_scaled

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        regime_labels: Union[pd.Series, np.ndarray],
        y=None
    ) -> np.ndarray:
        """
        Fit scaler and transform data in one step.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix (n_samples, n_features)
        regime_labels : pd.Series or np.ndarray
            Regime label for each sample (n_samples,)
        y : Ignored
            Not used, present for sklearn compatibility

        Returns
        -------
        X_scaled : np.ndarray
            Scaled features (n_samples, n_features)
        """
        return self.fit(X, regime_labels, y).transform(X, regime_labels)

    def inverse_transform(
        self,
        X_scaled: Union[pd.DataFrame, np.ndarray],
        regime_labels: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Inverse transform scaled features back to original scale.

        Parameters
        ----------
        X_scaled : pd.DataFrame or np.ndarray
            Scaled feature matrix (n_samples, n_features)
        regime_labels : pd.Series or np.ndarray
            Regime label for each sample (n_samples,)

        Returns
        -------
        X : np.ndarray
            Original-scale features (n_samples, n_features)
        """
        if self.n_features_in_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        # Convert to numpy
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
        if isinstance(regime_labels, pd.Series):
            regime_labels = regime_labels.values

        # Initialize unscaled array
        X = np.zeros_like(X_scaled, dtype=np.float64)

        unique_regimes = np.unique(regime_labels)

        for regime in unique_regimes:
            mask = (regime_labels == regime)

            if regime not in self.regime_stats_:
                mean = self.global_stats_['mean']
                std = self.global_stats_['std']
            else:
                mean = self.regime_stats_[regime]['mean']
                std = self.regime_stats_[regime]['std']

            # Unscale: X = X_scaled * std + mean
            X[mask] = X_scaled[mask] * std + mean

        return X

    def get_regime_stats(self) -> Dict:
        """
        Get fitted regime statistics.

        Returns
        -------
        dict
            Regime statistics including mean, std, n_samples per regime
        """
        if self.n_features_in_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        return {
            'regime_stats': self.regime_stats_,
            'global_stats': self.global_stats_,
            'n_features': self.n_features_in_,
            'regimes_seen': list(self.regimes_seen_)
        }
