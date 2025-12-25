"""
Probability Calibration for ML Models

Calibrates predicted probabilities to ensure P(actual=1 | predicted=p) ≈ p.
Essential for reliable probability estimates in production.

Key functions:
- calibrate_probabilities(): Calibrate using isotonic or Platt scaling
- evaluate_calibration(): Compute calibration metrics
- plot_calibration_curve(): Visualize calibration quality

References:
- Niculescu-Mizil & Caruana (2005), "Predicting Good Probabilities with Supervised Learning"
- sklearn.calibration.CalibratedClassifierCV
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Literal
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def calibrate_probabilities(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    method: Literal["isotonic", "platt"] = "isotonic",
    return_calibrator: bool = False
) -> Tuple[np.ndarray, Optional[object]]:
    """
    Calibrate predicted probabilities.

    Parameters
    ----------
    y_prob : np.ndarray
        Uncalibrated probabilities (n_samples,)
    y_true : np.ndarray
        True binary labels (n_samples,)
    method : {"isotonic", "platt"}
        Calibration method:
        - "isotonic": Isotonic regression (non-parametric, flexible)
        - "platt": Platt scaling (logistic regression, parametric)
    return_calibrator : bool
        If True, return fitted calibrator object for reuse

    Returns
    -------
    y_prob_calibrated : np.ndarray
        Calibrated probabilities (n_samples,)
    calibrator : object or None
        Fitted calibrator if return_calibrator=True, else None

    Examples
    --------
    >>> import numpy as np
    >>> # Simulate poorly calibrated probabilities
    >>> np.random.seed(42)
    >>> y_true = np.random.binomial(1, 0.5, 1000)
    >>> y_prob_uncalibrated = np.random.beta(2, 2, 1000)  # Overconfident
    >>>
    >>> # Calibrate
    >>> y_prob_calibrated, calibrator = calibrate_probabilities(
    ...     y_prob_uncalibrated, y_true, method="isotonic", return_calibrator=True
    ... )
    >>>
    >>> # Apply to new data
    >>> y_prob_new = np.array([0.2, 0.5, 0.8])
    >>> y_prob_new_calibrated = calibrator.predict(y_prob_new)
    """
    if len(y_prob) != len(y_true):
        raise ValueError(
            f"y_prob and y_true must have same length: {len(y_prob)} != {len(y_true)}"
        )

    # Validate binary labels
    unique_labels = np.unique(y_true)
    if not np.array_equal(unique_labels, [0, 1]) and not np.array_equal(unique_labels, [0]) and not np.array_equal(unique_labels, [1]):
        raise ValueError(f"y_true must be binary (0/1), got: {unique_labels}")

    if method == "isotonic":
        # Isotonic regression calibration
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_prob, y_true)
        y_prob_calibrated = calibrator.predict(y_prob)

    elif method == "platt":
        # Platt scaling (logistic regression)
        # Convert probabilities to log-odds, then fit logistic regression
        calibrator = LogisticRegression()
        calibrator.fit(y_prob.reshape(-1, 1), y_true)
        y_prob_calibrated = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]

    else:
        raise ValueError(f"Unsupported method: {method}. Use 'isotonic' or 'platt'.")

    logger.info(
        f"Calibrated probabilities using {method}. "
        f"Uncalibrated range: [{y_prob.min():.3f}, {y_prob.max():.3f}], "
        f"Calibrated range: [{y_prob_calibrated.min():.3f}, {y_prob_calibrated.max():.3f}]"
    )

    if return_calibrator:
        return y_prob_calibrated, calibrator
    else:
        return y_prob_calibrated, None


def evaluate_calibration(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10
) -> dict:
    """
    Evaluate calibration quality.

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities (n_samples,)
    y_true : np.ndarray
        True binary labels (n_samples,)
    n_bins : int
        Number of bins for calibration curve (default: 10)

    Returns
    -------
    dict
        Calibration metrics:
        - brier_score: Brier score (lower is better, 0 is perfect)
        - log_loss: Log loss (lower is better)
        - ece: Expected Calibration Error
        - mce: Maximum Calibration Error
        - bin_edges: Probability bin edges
        - bin_true_freq: True frequency in each bin
        - bin_pred_freq: Predicted frequency in each bin
        - bin_counts: Number of samples in each bin

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1, 1])
    >>> y_prob = np.array([0.2, 0.3, 0.6, 0.7, 0.9])
    >>> metrics = evaluate_calibration(y_prob, y_true, n_bins=5)
    >>> print(f"Brier score: {metrics['brier_score']:.4f}")
    >>> print(f"ECE: {metrics['ece']:.4f}")
    """
    if len(y_prob) != len(y_true):
        raise ValueError(
            f"y_prob and y_true must have same length: {len(y_prob)} != {len(y_true)}"
        )

    # Brier score
    brier = brier_score_loss(y_true, y_prob)

    # Log loss
    # Clip probabilities to avoid log(0)
    y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
    logloss = log_loss(y_true, y_prob_clipped)

    # Calibration curve (binned)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])  # Bin index for each sample

    bin_true_freq = np.zeros(n_bins)
    bin_pred_freq = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for bin_idx in range(n_bins):
        mask = (bin_indices == bin_idx)
        bin_counts[bin_idx] = mask.sum()

        if bin_counts[bin_idx] > 0:
            bin_true_freq[bin_idx] = y_true[mask].mean()
            bin_pred_freq[bin_idx] = y_prob[mask].mean()

    # Expected Calibration Error (ECE)
    # ECE = sum(|true_freq - pred_freq| * bin_count / total_count)
    ece = np.sum(
        np.abs(bin_true_freq - bin_pred_freq) * bin_counts
    ) / len(y_prob)

    # Maximum Calibration Error (MCE)
    # MCE = max(|true_freq - pred_freq|) over bins with samples
    valid_bins = bin_counts > 0
    if valid_bins.any():
        mce = np.max(np.abs(bin_true_freq[valid_bins] - bin_pred_freq[valid_bins]))
    else:
        mce = 0.0

    return {
        'brier_score': float(brier),
        'log_loss': float(logloss),
        'ece': float(ece),
        'mce': float(mce),
        'bin_edges': bin_edges,
        'bin_true_freq': bin_true_freq,
        'bin_pred_freq': bin_pred_freq,
        'bin_counts': bin_counts
    }


def plot_calibration_curve(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
    ax: Optional[plt.Axes] = None,
    label: str = "Model"
) -> plt.Figure:
    """
    Plot calibration curve (reliability diagram).

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities (n_samples,)
    y_true : np.ndarray
        True binary labels (n_samples,)
    n_bins : int
        Number of bins for calibration curve (default: 10)
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure
    label : str
        Label for the calibration curve

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> y_true = np.random.binomial(1, 0.5, 1000)
    >>> y_prob_uncalibrated = np.random.beta(2, 2, 1000)
    >>> y_prob_calibrated, _ = calibrate_probabilities(y_prob_uncalibrated, y_true)
    >>>
    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    >>> plot_calibration_curve(y_prob_uncalibrated, y_true, ax=axes[0], label="Uncalibrated")
    >>> plot_calibration_curve(y_prob_calibrated, y_true, ax=axes[1], label="Calibrated")
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    # Compute calibration metrics
    metrics = evaluate_calibration(y_prob, y_true, n_bins=n_bins)

    # Plot calibration curve
    bin_centers = (metrics['bin_edges'][:-1] + metrics['bin_edges'][1:]) / 2
    valid_bins = metrics['bin_counts'] > 0

    ax.plot(
        bin_centers[valid_bins],
        metrics['bin_true_freq'][valid_bins],
        marker='o',
        linestyle='-',
        label=label,
        linewidth=2
    )

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')

    # Add histogram of predicted probabilities
    ax.hist(
        y_prob,
        bins=metrics['bin_edges'],
        alpha=0.3,
        density=True,
        label='Predicted probability distribution'
    )

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('True Frequency', fontsize=12)
    ax.set_title(
        f'Calibration Curve\n'
        f'Brier={metrics["brier_score"]:.4f}, '
        f'ECE={metrics["ece"]:.4f}, '
        f'MCE={metrics["mce"]:.4f}',
        fontsize=13,
        fontweight='bold'
    )
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return fig


def compare_calibration(
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10
) -> dict:
    """
    Compare calibration before and after calibration.

    Parameters
    ----------
    y_prob_before : np.ndarray
        Uncalibrated probabilities
    y_prob_after : np.ndarray
        Calibrated probabilities
    y_true : np.ndarray
        True binary labels
    n_bins : int
        Number of bins for calibration curve

    Returns
    -------
    dict
        Comparison metrics:
        - before: Calibration metrics before
        - after: Calibration metrics after
        - improvement: Improvement in each metric

    Examples
    --------
    >>> y_true = np.random.binomial(1, 0.5, 1000)
    >>> y_prob_before = np.random.beta(2, 2, 1000)
    >>> y_prob_after, _ = calibrate_probabilities(y_prob_before, y_true)
    >>>
    >>> comparison = compare_calibration(y_prob_before, y_prob_after, y_true)
    >>> print(f"ECE improvement: {comparison['improvement']['ece']:.4f}")
    """
    metrics_before = evaluate_calibration(y_prob_before, y_true, n_bins=n_bins)
    metrics_after = evaluate_calibration(y_prob_after, y_true, n_bins=n_bins)

    improvement = {
        'brier_score': metrics_before['brier_score'] - metrics_after['brier_score'],
        'log_loss': metrics_before['log_loss'] - metrics_after['log_loss'],
        'ece': metrics_before['ece'] - metrics_after['ece'],
        'mce': metrics_before['mce'] - metrics_after['mce']
    }

    return {
        'before': metrics_before,
        'after': metrics_after,
        'improvement': improvement
    }
