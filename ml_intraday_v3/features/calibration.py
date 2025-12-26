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


class MulticlassIsotonicCalibrator:
    """
    Multiclass isotonic calibration using one-vs-rest approach.

    For each class k, fits an isotonic regressor that maps
    predicted P(class=k) to calibrated P(class=k|features).

    After calibration, probabilities are renormalized to sum to 1.

    Attributes
    ----------
    calibrators_ : dict
        Mapping from class label to fitted IsotonicRegression
    classes_ : list
        List of class labels in order

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> # 3-class problem: -1 (stop), 0 (vertical), 1 (target)
    >>> y_true = np.random.choice([-1, 0, 1], size=1000, p=[0.2, 0.6, 0.2])
    >>> proba_uncal = np.random.dirichlet([2, 3, 2], size=1000)
    >>>
    >>> calibrator = MulticlassIsotonicCalibrator(classes=[-1, 0, 1])
    >>> calibrator.fit(proba_uncal, y_true)
    >>> proba_cal = calibrator.transform(proba_uncal)
    >>> print(f"Calibrated probs sum to 1: {np.allclose(proba_cal.sum(axis=1), 1)}")
    """

    def __init__(self, classes: list):
        """
        Initialize multiclass calibrator.

        Parameters
        ----------
        classes : list
            List of class labels in the same order as probability columns.
            E.g., [-1, 0, 1] for stop/vertical/target.
        """
        self.classes_ = list(classes)
        self.calibrators_ = {}
        self._fitted = False

    def fit(self, y_proba: np.ndarray, y_true: np.ndarray) -> "MulticlassIsotonicCalibrator":
        """
        Fit isotonic calibrators for each class.

        Parameters
        ----------
        y_proba : np.ndarray, shape (n_samples, n_classes)
            Uncalibrated probability estimates from model
        y_true : np.ndarray, shape (n_samples,)
            True class labels (must be values from self.classes_)

        Returns
        -------
        self
        """
        n_samples, n_classes = y_proba.shape
        if n_classes != len(self.classes_):
            raise ValueError(
                f"y_proba has {n_classes} columns but expected {len(self.classes_)} classes"
            )

        y_true = np.asarray(y_true)

        for i, cls in enumerate(self.classes_):
            # One-vs-rest: binary indicator for this class
            y_binary = (y_true == cls).astype(int)
            p_cls = y_proba[:, i]

            # Fit isotonic regression
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(p_cls, y_binary)
            self.calibrators_[cls] = iso

            logger.info(
                f"Fitted isotonic calibrator for class {cls}: "
                f"p_uncal range [{p_cls.min():.3f}, {p_cls.max():.3f}], "
                f"actual rate {y_binary.mean():.3f}"
            )

        self._fitted = True
        return self

    def transform(self, y_proba: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Apply calibration to probability estimates.

        Parameters
        ----------
        y_proba : np.ndarray, shape (n_samples, n_classes)
            Uncalibrated probability estimates
        normalize : bool
            If True (default), renormalize calibrated probabilities to sum to 1

        Returns
        -------
        y_proba_cal : np.ndarray, shape (n_samples, n_classes)
            Calibrated probability estimates
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        n_samples, n_classes = y_proba.shape
        if n_classes != len(self.classes_):
            raise ValueError(
                f"y_proba has {n_classes} columns but expected {len(self.classes_)} classes"
            )

        y_proba_cal = np.zeros_like(y_proba)

        for i, cls in enumerate(self.classes_):
            p_cls = y_proba[:, i]
            y_proba_cal[:, i] = self.calibrators_[cls].predict(p_cls)

        # Renormalize so probabilities sum to 1
        if normalize:
            row_sums = y_proba_cal.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            y_proba_cal = y_proba_cal / row_sums

        return y_proba_cal

    def fit_transform(self, y_proba: np.ndarray, y_true: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(y_proba, y_true)
        return self.transform(y_proba, normalize=normalize)

    def get_calibration_report(self, y_proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> dict:
        """
        Generate calibration metrics before and after calibration.

        Returns dict with per-class ECE, Brier scores, and improvements.
        """
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        y_proba_cal = self.transform(y_proba)
        y_true = np.asarray(y_true)

        report = {"classes": {}, "summary": {}}

        total_ece_before = 0.0
        total_ece_after = 0.0
        total_brier_before = 0.0
        total_brier_after = 0.0

        for i, cls in enumerate(self.classes_):
            y_binary = (y_true == cls).astype(int)

            metrics_before = evaluate_calibration(y_proba[:, i], y_binary, n_bins=n_bins)
            metrics_after = evaluate_calibration(y_proba_cal[:, i], y_binary, n_bins=n_bins)

            report["classes"][cls] = {
                "ece_before": metrics_before["ece"],
                "ece_after": metrics_after["ece"],
                "ece_improvement": metrics_before["ece"] - metrics_after["ece"],
                "brier_before": metrics_before["brier_score"],
                "brier_after": metrics_after["brier_score"],
                "brier_improvement": metrics_before["brier_score"] - metrics_after["brier_score"],
                "actual_rate": float(y_binary.mean()),
                "predicted_rate_before": float(y_proba[:, i].mean()),
                "predicted_rate_after": float(y_proba_cal[:, i].mean()),
            }

            total_ece_before += metrics_before["ece"]
            total_ece_after += metrics_after["ece"]
            total_brier_before += metrics_before["brier_score"]
            total_brier_after += metrics_after["brier_score"]

        n_classes = len(self.classes_)
        report["summary"] = {
            "mean_ece_before": total_ece_before / n_classes,
            "mean_ece_after": total_ece_after / n_classes,
            "mean_ece_improvement": (total_ece_before - total_ece_after) / n_classes,
            "mean_brier_before": total_brier_before / n_classes,
            "mean_brier_after": total_brier_after / n_classes,
            "mean_brier_improvement": (total_brier_before - total_brier_after) / n_classes,
        }

        return report


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
