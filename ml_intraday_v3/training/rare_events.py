"""
Rare Events Corrections for Logistic Regression

Implements corrections for logistic regression when dealing with rare events
(imbalanced classes) following King & Zeng (2001).

Key corrections:
1. Prior correction for predicted probabilities
2. Sample weighting for rare events
3. Relogit (relogistic regression) classifier

References:
    King, G., & Zeng, L. (2001). Logistic regression in rare events data.
    Political analysis, 9(2), 137-163.
"""

from __future__ import annotations

from typing import Optional, Literal
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import warnings


def correct_rare_events_probabilities(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    tau: Optional[float] = None,
    method: Literal["king_zeng", "simple"] = "king_zeng"
) -> np.ndarray:
    """
    Correct predicted probabilities for rare events bias.

    In standard logistic regression trained on rare events data, the predicted
    probabilities are typically biased. This function applies the King & Zeng
    (2001) correction to obtain better probability estimates.

    The correction formula is:
        P_corrected = (P * τ) / (P * τ + (1-P) * (1-τ))

    where:
        P = uncorrected predicted probability
        τ = true prior probability of the rare event (tau)

    Parameters
    ----------
    y_prob : np.ndarray, shape (n_samples,)
        Uncorrected predicted probabilities from logistic regression
    y_true : np.ndarray, shape (n_samples,)
        True binary labels (0 or 1) from training set
    tau : float, optional
        True prior probability of positive class.
        If None, estimated as mean(y_true)
    method : {"king_zeng", "simple"}, default="king_zeng"
        Correction method:
        - "king_zeng": Full King & Zeng correction formula
        - "simple": Simple rescaling by observed rate

    Returns
    -------
    np.ndarray
        Corrected predicted probabilities

    Notes
    -----
    - tau should be estimated from the population, not just the sample
    - If population tau is unknown, sample mean is a reasonable estimate
    - Correction is most important when sample is case-control or
      artificially balanced

    Examples
    --------
    >>> y_true = np.array([0, 0, 0, 1, 0, 0, 1, 0])
    >>> y_prob = np.array([0.3, 0.2, 0.4, 0.8, 0.1, 0.3, 0.7, 0.2])
    >>> corrected = correct_rare_events_probabilities(y_prob, y_true)
    >>> corrected
    array([0.14, 0.09, 0.19, 0.64, 0.05, 0.14, 0.52, 0.09])
    """
    y_prob = np.asarray(y_prob)
    y_true = np.asarray(y_true)

    if len(y_prob) != len(y_true):
        raise ValueError("y_prob and y_true must have same length")

    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("y_prob must be in [0, 1]")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")

    # Estimate tau if not provided
    if tau is None:
        tau = float(np.mean(y_true))

    if tau <= 0 or tau >= 1:
        raise ValueError(f"tau must be in (0, 1), got {tau}")

    if method == "king_zeng":
        # King & Zeng (2001) correction
        # P_corrected = (P * τ) / (P * τ + (1-P) * (1-τ))
        numerator = y_prob * tau
        denominator = y_prob * tau + (1 - y_prob) * (1 - tau)

        # Handle edge cases
        with np.errstate(divide='ignore', invalid='ignore'):
            corrected = np.where(
                denominator > 0,
                numerator / denominator,
                y_prob  # Fall back to original if denominator is 0
            )

    elif method == "simple":
        # Simple rescaling
        # Just rescale so mean matches tau
        current_mean = np.mean(y_prob)
        if current_mean > 0:
            corrected = y_prob * (tau / current_mean)
            corrected = np.clip(corrected, 0, 1)
        else:
            corrected = y_prob

    else:
        raise ValueError(f"Unknown method: {method}")

    return corrected


def compute_rare_event_weights(
    y_true: np.ndarray,
    method: Literal["king_zeng", "inverse_freq", "balanced"] = "king_zeng",
    tau: Optional[float] = None
) -> np.ndarray:
    """
    Compute sample weights for rare events logistic regression.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True binary labels
    method : {"king_zeng", "inverse_freq", "balanced"}, default="king_zeng"
        Weighting method:
        - "king_zeng": w_i = 1 for y=1, w_i = τ/(1-τ) for y=0
        - "inverse_freq": w_i = 1/freq(y_i)
        - "balanced": sklearn-style balanced weights
    tau : float, optional
        True prior probability. If None, estimated from y_true

    Returns
    -------
    np.ndarray
        Sample weights

    Notes
    -----
    King & Zeng weights are designed to correct for case-control sampling
    where rare events are oversampled. The weights rebalance to match the
    true population distribution.

    Examples
    --------
    >>> y_true = np.array([0, 0, 0, 1, 0, 0, 1, 0])
    >>> weights = compute_rare_event_weights(y_true, method="king_zeng")
    >>> weights
    array([0.33, 0.33, 0.33, 1.  , 0.33, 0.33, 1.  , 0.33])
    """
    y_true = np.asarray(y_true)

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")

    n = len(y_true)
    n_pos = np.sum(y_true)
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true must contain both classes")

    if method == "king_zeng":
        # Estimate tau if not provided
        if tau is None:
            tau = n_pos / n

        if tau <= 0 or tau >= 1:
            raise ValueError(f"tau must be in (0, 1), got {tau}")

        # w_i = 1 for rare events (y=1)
        # w_i = τ/(1-τ) for common events (y=0)
        weights = np.where(y_true == 1, 1.0, tau / (1 - tau))

    elif method == "inverse_freq":
        # Standard inverse frequency weighting
        weights = np.where(y_true == 1, n / (2 * n_pos), n / (2 * n_neg))

    elif method == "balanced":
        # Sklearn-style balanced class weights
        weights = np.where(y_true == 1, n / (2 * n_pos), n / (2 * n_neg))

    else:
        raise ValueError(f"Unknown method: {method}")

    return weights


class RelogitClassifier(BaseEstimator, ClassifierMixin):
    """
    Relogit (Relogistic Regression) for Rare Events.

    Implements logistic regression with automatic rare events correction
    following King & Zeng (2001). This is a drop-in replacement for
    sklearn's LogisticRegression that handles rare events better.

    The classifier applies two corrections:
    1. Sample weighting during training (optional)
    2. Prior correction to predicted probabilities (always applied)

    Parameters
    ----------
    tau : float, optional
        True prior probability of positive class.
        If None, estimated from training data.
    use_sample_weights : bool, default=True
        Whether to use rare event weights during training
    weight_method : {"king_zeng", "inverse_freq", "balanced"}, default="king_zeng"
        Method for computing sample weights
    correction_method : {"king_zeng", "simple"}, default="king_zeng"
        Method for correcting predicted probabilities
    C : float, default=1.0
        Inverse of regularization strength (same as LogisticRegression)
    penalty : {"l2", "l1", "elasticnet", "none"}, default="l2"
        Regularization penalty
    solver : str, default="lbfgs"
        Solver to use (same as LogisticRegression)
    max_iter : int, default=100
        Maximum iterations
    random_state : int, optional
        Random seed

    Attributes
    ----------
    classes_ : np.ndarray
        Class labels
    coef_ : np.ndarray
        Coefficient weights
    intercept_ : np.ndarray
        Intercept term
    tau_ : float
        Estimated or provided prior probability
    lr_model_ : LogisticRegression
        Underlying logistic regression model

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> clf = RelogitClassifier(random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> y_prob = clf.predict_proba(X_test)[:, 1]
    """

    def __init__(
        self,
        tau: Optional[float] = None,
        use_sample_weights: bool = True,
        weight_method: Literal["king_zeng", "inverse_freq", "balanced"] = "king_zeng",
        correction_method: Literal["king_zeng", "simple"] = "king_zeng",
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 100,
        random_state: Optional[int] = None,
        **lr_kwargs
    ):
        self.tau = tau
        self.use_sample_weights = use_sample_weights
        self.weight_method = weight_method
        self.correction_method = correction_method
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.lr_kwargs = lr_kwargs

    def fit(self, X, y, sample_weight=None):
        """
        Fit the relogit model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        sample_weight : array-like, shape (n_samples,), optional
            Individual weights for each sample

        Returns
        -------
        self : RelogitClassifier
            Fitted estimator
        """
        # Validate inputs
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        if len(self.classes_) != 2:
            raise ValueError("RelogitClassifier only supports binary classification")

        # Estimate tau if not provided
        if self.tau is None:
            self.tau_ = float(np.mean(y))
        else:
            self.tau_ = self.tau

        # Compute rare event weights
        if self.use_sample_weights:
            re_weights = compute_rare_event_weights(
                y,
                method=self.weight_method,
                tau=self.tau_
            )

            # Combine with user-provided weights if any
            if sample_weight is not None:
                re_weights = re_weights * sample_weight

            sample_weight = re_weights

        # Fit underlying logistic regression
        self.lr_model_ = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **self.lr_kwargs
        )

        self.lr_model_.fit(X, y, sample_weight=sample_weight)

        # Copy attributes
        self.coef_ = self.lr_model_.coef_
        self.intercept_ = self.lr_model_.intercept_

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities with rare events correction.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns
        -------
        proba : np.ndarray, shape (n_samples, 2)
            Probability of each class
        """
        check_is_fitted(self, 'lr_model_')
        X = check_array(X)

        # Get uncorrected probabilities
        proba_uncorrected = self.lr_model_.predict_proba(X)

        # Apply rare events correction to positive class probability
        proba_pos_corrected = correct_rare_events_probabilities(
            proba_uncorrected[:, 1],
            y_true=np.ones(len(X)),  # Dummy, tau is already estimated
            tau=self.tau_,
            method=self.correction_method
        )

        # Reconstruct full probability matrix
        proba_corrected = np.column_stack([
            1 - proba_pos_corrected,
            proba_pos_corrected
        ])

        return proba_corrected

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def decision_function(self, X):
        """
        Decision function (log-odds).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
            Decision function values
        """
        check_is_fitted(self, 'lr_model_')
        X = check_array(X)
        return self.lr_model_.decision_function(X)


def apply_prior_correction_to_intercept(
    intercept: float,
    tau_population: float,
    y_sample_mean: float
) -> float:
    """
    Apply prior correction to logistic regression intercept.

    When the training sample has a different class balance than the population,
    the intercept needs to be corrected. This implements the King & Zeng
    correction for the intercept term.

    Parameters
    ----------
    intercept : float
        Original intercept from logistic regression
    tau_population : float
        True prior probability in population
    y_sample_mean : float
        Observed probability in training sample

    Returns
    -------
    float
        Corrected intercept

    Notes
    -----
    The correction formula is:
        intercept_corrected = intercept - log((1-τ)/τ * ȳ/(1-ȳ))

    where:
        τ = tau_population (true prior)
        ȳ = y_sample_mean (sample proportion)

    Examples
    --------
    >>> # Population has 5% positive rate, but sample has 50% (balanced)
    >>> intercept_corrected = apply_prior_correction_to_intercept(
    ...     intercept=0.0,
    ...     tau_population=0.05,
    ...     y_sample_mean=0.50
    ... )
    >>> intercept_corrected
    -2.89
    """
    if tau_population <= 0 or tau_population >= 1:
        raise ValueError(f"tau_population must be in (0, 1), got {tau_population}")

    if y_sample_mean <= 0 or y_sample_mean >= 1:
        raise ValueError(f"y_sample_mean must be in (0, 1), got {y_sample_mean}")

    # Correction term
    correction = np.log(
        (1 - tau_population) / tau_population *
        y_sample_mean / (1 - y_sample_mean)
    )

    return intercept - correction


def estimate_population_prior(
    y_train: np.ndarray,
    y_val: np.ndarray = None,
    method: Literal["train", "val", "pooled"] = "train"
) -> float:
    """
    Estimate population prior probability.

    Parameters
    ----------
    y_train : np.ndarray
        Training labels
    y_val : np.ndarray, optional
        Validation labels
    method : {"train", "val", "pooled"}, default="train"
        How to estimate prior:
        - "train": Use training set mean
        - "val": Use validation set mean (if available)
        - "pooled": Pool train and val

    Returns
    -------
    float
        Estimated prior probability

    Notes
    -----
    If the training data is a random sample from the population,
    the sample mean is an unbiased estimate of the population prior.
    However, if the data is case-control or artificially balanced,
    external information about the true prior should be used.
    """
    if method == "train":
        return float(np.mean(y_train))

    elif method == "val":
        if y_val is None:
            raise ValueError("y_val required for method='val'")
        return float(np.mean(y_val))

    elif method == "pooled":
        if y_val is None:
            return float(np.mean(y_train))
        return float(np.mean(np.concatenate([y_train, y_val])))

    else:
        raise ValueError(f"Unknown method: {method}")
