"""
Regime Detection for Feature Scaling

Detects market regimes (volatility, trend) to enable regime-aware feature normalization
that prevents distribution shifts between train/test.

Key functions:
- detect_volatility_regime(): Classify volatility as low/medium/high
- detect_trend_regime(): Classify trend as downtrend/sideways/uptrend
- detect_combined_regime(): Combined volatility + trend regime

References:
- López de Prado (2018), "Advances in Financial Machine Learning", Chapter 19
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def detect_volatility_regime(
    returns: Union[pd.Series, np.ndarray],
    window: int = 20,
    n_regimes: int = 3,
    method: str = "quantile"
) -> pd.Series:
    """
    Detect volatility regime (low/medium/high) based on rolling volatility.

    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Return series (not prices!)
    window : int
        Rolling window for volatility computation (default: 20)
    n_regimes : int
        Number of regimes (default: 3 for low/medium/high)
    method : str
        Classification method:
        - "quantile": Use quantile-based thresholds
        - "kmeans": Use k-means clustering (future)

    Returns
    -------
    pd.Series
        Regime labels: 0 (low), 1 (medium), 2 (high)

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(1000) * 0.01)
    >>> regime = detect_volatility_regime(returns, window=20)
    >>> regime.value_counts()
    0    334
    1    333
    2    333
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    if len(returns) < window:
        raise ValueError(f"Returns length ({len(returns)}) must be >= window ({window})")

    # Compute rolling volatility (standard deviation)
    rolling_vol = returns.rolling(window=window, min_periods=window).std()

    # Handle NaN at beginning (use first valid volatility)
    first_valid_idx = rolling_vol.first_valid_index()
    if first_valid_idx is not None:
        first_valid_vol = rolling_vol.loc[first_valid_idx]
        rolling_vol = rolling_vol.fillna(first_valid_vol)

    if method == "quantile":
        # Use quantiles to define regime boundaries
        if n_regimes == 3:
            # Low: [0, 33%], Medium: [33%, 67%], High: [67%, 100%]
            q33 = rolling_vol.quantile(0.33)
            q67 = rolling_vol.quantile(0.67)

            regime = pd.Series(index=returns.index, dtype=int)
            regime.loc[rolling_vol <= q33] = 0  # Low volatility
            regime.loc[(rolling_vol > q33) & (rolling_vol <= q67)] = 1  # Medium
            regime.loc[rolling_vol > q67] = 2  # High volatility

        elif n_regimes == 2:
            # Low: [0, 50%], High: [50%, 100%]
            median = rolling_vol.median()
            regime = pd.Series(index=returns.index, dtype=int)
            regime.loc[rolling_vol <= median] = 0  # Low
            regime.loc[rolling_vol > median] = 1  # High

        else:
            # Generic n-regime quantile-based classification
            quantiles = np.linspace(0, 1, n_regimes + 1)
            bins = [rolling_vol.quantile(q) for q in quantiles]
            regime = pd.cut(rolling_vol, bins=bins, labels=False, include_lowest=True)
            regime = pd.Series(regime, index=returns.index, dtype=int)

    else:
        raise ValueError(f"Unsupported method: {method}. Use 'quantile'.")

    return regime


def detect_trend_regime(
    prices: Union[pd.Series, np.ndarray],
    window: int = 50,
    n_regimes: int = 3,
    method: str = "slope"
) -> pd.Series:
    """
    Detect trend regime (downtrend/sideways/uptrend) based on price trend.

    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Price series (not returns!)
    window : int
        Rolling window for trend computation (default: 50)
    n_regimes : int
        Number of regimes (default: 3 for down/sideways/up)
    method : str
        Trend detection method:
        - "slope": Linear regression slope of price
        - "ma_diff": Difference between price and moving average

    Returns
    -------
    pd.Series
        Regime labels: 0 (downtrend), 1 (sideways), 2 (uptrend)

    Examples
    --------
    >>> prices = pd.Series(np.cumsum(np.random.randn(1000)) + 100)
    >>> regime = detect_trend_regime(prices, window=50)
    >>> regime.value_counts()
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)

    if len(prices) < window:
        raise ValueError(f"Prices length ({len(prices)}) must be >= window ({window})")

    if method == "slope":
        # Compute rolling linear regression slope
        def rolling_slope(series):
            """Compute slope of linear regression over window."""
            if len(series) < 2:
                return np.nan
            x = np.arange(len(series))
            y = series.values
            # Simple linear regression: slope = cov(x,y) / var(x)
            slope = np.cov(x, y)[0, 1] / np.var(x)
            return slope

        trend_strength = prices.rolling(window=window, min_periods=window).apply(
            rolling_slope, raw=False
        )

    elif method == "ma_diff":
        # Compute difference between price and moving average
        ma = prices.rolling(window=window, min_periods=window).mean()
        trend_strength = (prices - ma) / ma  # Normalized difference

    else:
        raise ValueError(f"Unsupported method: {method}. Use 'slope' or 'ma_diff'.")

    # Handle NaN at beginning
    first_valid_idx = trend_strength.first_valid_index()
    if first_valid_idx is not None:
        first_valid_val = trend_strength.loc[first_valid_idx]
        trend_strength = trend_strength.fillna(first_valid_val)

    # Classify into regimes based on quantiles
    if n_regimes == 3:
        # Downtrend: [0, 33%], Sideways: [33%, 67%], Uptrend: [67%, 100%]
        q33 = trend_strength.quantile(0.33)
        q67 = trend_strength.quantile(0.67)

        regime = pd.Series(index=prices.index, dtype=int)
        regime.loc[trend_strength <= q33] = 0  # Downtrend
        regime.loc[(trend_strength > q33) & (trend_strength <= q67)] = 1  # Sideways
        regime.loc[trend_strength > q67] = 2  # Uptrend

    elif n_regimes == 2:
        # Down: [0, 50%], Up: [50%, 100%]
        median = trend_strength.median()
        regime = pd.Series(index=prices.index, dtype=int)
        regime.loc[trend_strength <= median] = 0  # Down
        regime.loc[trend_strength > median] = 1  # Up

    else:
        # Generic n-regime quantile-based classification
        quantiles = np.linspace(0, 1, n_regimes + 1)
        bins = [trend_strength.quantile(q) for q in quantiles]
        regime = pd.cut(trend_strength, bins=bins, labels=False, include_lowest=True)
        regime = pd.Series(regime, index=prices.index, dtype=int)

    return regime


def detect_combined_regime(
    prices: Union[pd.Series, np.ndarray],
    returns: Optional[Union[pd.Series, np.ndarray]] = None,
    vol_window: int = 20,
    trend_window: int = 50,
    n_vol_regimes: int = 3,
    n_trend_regimes: int = 3
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Detect combined volatility + trend regime.

    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Price series
    returns : pd.Series or np.ndarray, optional
        Return series. If None, computed from prices
    vol_window : int
        Window for volatility regime detection (default: 20)
    trend_window : int
        Window for trend regime detection (default: 50)
    n_vol_regimes : int
        Number of volatility regimes (default: 3)
    n_trend_regimes : int
        Number of trend regimes (default: 3)

    Returns
    -------
    vol_regime : pd.Series
        Volatility regime (0=low, 1=medium, 2=high)
    trend_regime : pd.Series
        Trend regime (0=down, 1=sideways, 2=up)
    combined_regime : pd.Series
        Combined regime (encoded as vol * n_trend + trend)

    Examples
    --------
    >>> prices = pd.Series(np.cumsum(np.random.randn(1000)) + 100)
    >>> vol_reg, trend_reg, combined_reg = detect_combined_regime(prices)
    >>> combined_reg.value_counts()
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)

    # Compute returns if not provided
    if returns is None:
        returns = prices.pct_change().fillna(0)
    elif isinstance(returns, np.ndarray):
        returns = pd.Series(returns, index=prices.index)

    # Detect volatility regime
    vol_regime = detect_volatility_regime(
        returns=returns,
        window=vol_window,
        n_regimes=n_vol_regimes
    )

    # Detect trend regime
    trend_regime = detect_trend_regime(
        prices=prices,
        window=trend_window,
        n_regimes=n_trend_regimes
    )

    # Combine regimes into single label
    # Encoding: combined = vol * n_trend_regimes + trend
    # E.g., for 3x3 regimes: combined ranges from 0 to 8
    combined_regime = vol_regime * n_trend_regimes + trend_regime

    return vol_regime, trend_regime, combined_regime


def get_regime_labels(n_vol_regimes: int = 3, n_trend_regimes: int = 3) -> dict:
    """
    Get human-readable labels for regime combinations.

    Parameters
    ----------
    n_vol_regimes : int
        Number of volatility regimes
    n_trend_regimes : int
        Number of trend regimes

    Returns
    -------
    dict
        Mapping from combined regime integer to string label

    Examples
    --------
    >>> labels = get_regime_labels(3, 3)
    >>> labels[0]
    'low_vol_downtrend'
    >>> labels[8]
    'high_vol_uptrend'
    """
    # Define all possible labels
    all_vol_labels = {0: "low_vol", 1: "medium_vol", 2: "high_vol"}
    all_trend_labels = {0: "downtrend", 1: "sideways", 2: "uptrend"}

    # Select only the labels we need
    vol_labels = {i: all_vol_labels[i] for i in range(n_vol_regimes)}
    trend_labels = {i: all_trend_labels[i] for i in range(n_trend_regimes)}

    labels = {}
    for vol_idx in range(n_vol_regimes):
        for trend_idx in range(n_trend_regimes):
            combined_idx = vol_idx * n_trend_regimes + trend_idx
            labels[combined_idx] = f"{vol_labels[vol_idx]}_{trend_labels[trend_idx]}"

    return labels


def validate_regime_consistency(
    regime: pd.Series,
    min_samples_per_regime: int = 10
) -> bool:
    """
    Validate that regime detection produces reasonable results.

    Parameters
    ----------
    regime : pd.Series
        Detected regime labels
    min_samples_per_regime : int
        Minimum samples required per regime

    Returns
    -------
    bool
        True if validation passes

    Raises
    ------
    ValueError
        If validation fails
    """
    # Check for empty regimes
    regime_counts = regime.value_counts()
    empty_regimes = [r for r in regime.unique() if regime_counts[r] < min_samples_per_regime]

    if empty_regimes:
        logger.warning(
            f"Regimes {empty_regimes} have < {min_samples_per_regime} samples. "
            f"Counts: {regime_counts.to_dict()}"
        )
        return False

    # Check for reasonable distribution
    regime_pcts = regime.value_counts(normalize=True)
    if regime_pcts.max() > 0.8:
        logger.warning(
            f"Regime distribution is highly skewed: {regime_pcts.to_dict()}. "
            f"One regime dominates (>{regime_pcts.max()*100:.1f}%)"
        )
        return False

    return True
