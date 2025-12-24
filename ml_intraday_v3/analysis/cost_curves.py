"""
Cost Curves for Classifier Performance Visualization

Implements cost curves as described in Drummond & Holte (2006):
"Cost curves: An improved method for visualizing classifier performance"

Cost curves visualize classifier performance across all class distributions
and misclassification cost ratios by plotting Normalized Expected Cost (NC)
against Probability Cost (PC).
"""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from dataclasses import dataclass


@dataclass
class CostCurvePoint:
    """Single point on a cost curve."""
    pc: float  # Probability cost
    nc: float  # Normalized expected cost
    threshold: float  # Decision threshold
    tpr: float  # True positive rate
    fpr: float  # False positive rate
    cost_ratio: float  # Original cost ratio c = cost(FP) / (cost(FP) + cost(FN))


def compute_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_ratios: Optional[np.ndarray] = None,
    n_points: int = 100
) -> pd.DataFrame:
    """
    Compute cost curve for binary classifier.

    For each cost ratio c = cost(FP) / (cost(FP) + cost(FN)):
    1. Compute Probability Cost: PC = c * π₁ / (c * π₁ + (1-c) * π₀)
    2. Find optimal threshold that minimizes expected cost
    3. Compute Normalized Expected Cost: NC = (1 - TPR) * PC + FPR * (1 - PC)

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True binary labels (0 or 1)
    y_prob : array-like, shape (n_samples,)
        Predicted probabilities for positive class
    cost_ratios : array-like, optional
        Cost ratios c = cost(FP) / (cost(FP) + cost(FN))
        If None, uses 100 evenly spaced values from 0.001 to 0.999
    n_points : int, default=100
        Number of cost ratio points if cost_ratios not provided

    Returns
    -------
    pd.DataFrame
        Cost curve data with columns:
        - pc: Probability cost
        - nc: Normalized expected cost
        - threshold: Optimal decision threshold
        - tpr: True positive rate at threshold
        - fpr: False positive rate at threshold
        - cost_ratio: Cost ratio c

    Notes
    -----
    The cost curve dominance relationship:
    - Lower NC is better across all PC values
    - A curve strictly below another indicates strict dominance
    - Curves crossing indicate context-dependent performance
    """
    # Validate inputs
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have same length")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")

    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("y_prob must be in [0, 1]")

    # Compute class priors
    pi_1 = np.mean(y_true)  # P(y=1)
    pi_0 = 1 - pi_1  # P(y=0)

    if pi_1 == 0 or pi_0 == 0:
        raise ValueError("y_true must contain both classes")

    # Default cost ratios: from 0.001 to 0.999
    if cost_ratios is None:
        # Use logspace to get more points near 0 and 1
        # Map from [0.001, 0.999] with denser sampling at extremes
        cost_ratios = np.concatenate([
            np.linspace(0.001, 0.1, n_points // 4),
            np.linspace(0.1, 0.9, n_points // 2),
            np.linspace(0.9, 0.999, n_points // 4)
        ])
    else:
        cost_ratios = np.asarray(cost_ratios)

    # Compute ROC curve (all possible thresholds)
    fpr_all, tpr_all, thresholds = roc_curve(y_true, y_prob)

    # Store results
    results = []

    for c in cost_ratios:
        # Compute Probability Cost (PC)
        # PC = c * π₁ / (c * π₁ + (1-c) * π₀)
        pc = (c * pi_1) / (c * pi_1 + (1 - c) * pi_0)

        # Find optimal threshold by minimizing expected cost
        # Expected cost = c * FP + (1-c) * FN
        #               = c * FPR * N₀ + (1-c) * FNR * N₁
        #               = c * FPR * (1-π₁) + (1-c) * (1-TPR) * π₁
        # Normalized: NC = (1 - TPR) * PC + FPR * (1 - PC)

        nc_all = (1 - tpr_all) * pc + fpr_all * (1 - pc)

        # Find threshold with minimum normalized cost
        min_idx = np.argmin(nc_all)
        nc_opt = nc_all[min_idx]
        tpr_opt = tpr_all[min_idx]
        fpr_opt = fpr_all[min_idx]

        # Threshold at this index
        # Note: sklearn roc_curve returns thresholds in descending order
        # thresholds[0] is the max prediction + 1, thresholds[-1] is the min
        if min_idx < len(thresholds):
            threshold_opt = thresholds[min_idx]
        else:
            # Edge case: min_idx might be at the end
            threshold_opt = thresholds[-1]

        results.append({
            'pc': pc,
            'nc': nc_opt,
            'threshold': threshold_opt,
            'tpr': tpr_opt,
            'fpr': fpr_opt,
            'cost_ratio': c
        })

    df = pd.DataFrame(results)

    # Sort by PC for plotting
    df = df.sort_values('pc').reset_index(drop=True)

    return df


def bootstrap_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_ratios: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute cost curve with bootstrap confidence intervals.

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    cost_ratios : array-like, optional
        Cost ratios to evaluate
    n_bootstrap : int, default=1000
        Number of bootstrap samples
    confidence_level : float, default=0.95
        Confidence level for intervals (e.g., 0.95 for 95% CI)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    mean_curve : pd.DataFrame
        Mean cost curve across bootstrap samples
    lower_bound : pd.DataFrame
        Lower confidence bound
    upper_bound : pd.DataFrame
        Upper confidence bound
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)

    # Compute reference curve for PC values
    reference_curve = compute_cost_curve(y_true, y_prob, cost_ratios)
    pc_values = reference_curve['pc'].values
    cost_ratios_used = reference_curve['cost_ratio'].values

    # Store NC values across bootstrap samples
    nc_bootstrap = np.zeros((n_bootstrap, len(pc_values)))

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]

        try:
            # Compute cost curve for this sample
            curve_boot = compute_cost_curve(
                y_true_boot,
                y_prob_boot,
                cost_ratios_used
            )
            nc_bootstrap[i, :] = curve_boot['nc'].values
        except ValueError:
            # If bootstrap sample has only one class, use reference NC
            nc_bootstrap[i, :] = reference_curve['nc'].values

    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    nc_mean = np.mean(nc_bootstrap, axis=0)
    nc_lower = np.percentile(nc_bootstrap, lower_percentile, axis=0)
    nc_upper = np.percentile(nc_bootstrap, upper_percentile, axis=0)

    # Create DataFrames
    base_df = reference_curve[['pc', 'cost_ratio']].copy()

    mean_curve = base_df.copy()
    mean_curve['nc'] = nc_mean

    lower_bound = base_df.copy()
    lower_bound['nc'] = nc_lower

    upper_bound = base_df.copy()
    upper_bound['nc'] = nc_upper

    return mean_curve, lower_bound, upper_bound


def plot_cost_curve(
    cost_curve_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    show_confidence: bool = False,
    confidence_bounds: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
    color: Optional[str] = None,
    alpha: float = 0.7,
    **kwargs
) -> plt.Axes:
    """
    Plot cost curve: Normalized Expected Cost vs Probability Cost.

    Parameters
    ----------
    cost_curve_df : pd.DataFrame
        Cost curve data from compute_cost_curve()
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    label : str, optional
        Label for the curve (for legend)
    show_confidence : bool, default=False
        Whether to show confidence intervals
    confidence_bounds : tuple of (lower_df, upper_df), optional
        Confidence bound DataFrames from bootstrap_cost_curve()
    color : str, optional
        Color for the curve
    alpha : float, default=0.7
        Alpha transparency for the curve
    **kwargs
        Additional arguments passed to plot()

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Plot main curve
    ax.plot(
        cost_curve_df['pc'],
        cost_curve_df['nc'],
        label=label,
        color=color,
        alpha=alpha,
        **kwargs
    )

    # Plot confidence intervals if provided
    if show_confidence and confidence_bounds is not None:
        lower_df, upper_df = confidence_bounds
        ax.fill_between(
            lower_df['pc'],
            lower_df['nc'],
            upper_df['nc'],
            alpha=0.2,
            color=color
        )

    # Formatting
    ax.set_xlabel('Probability Cost (PC)', fontsize=11)
    ax.set_ylabel('Normalized Expected Cost (NC)', fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    if label:
        ax.legend()

    ax.set_title('Cost Curve', fontsize=12, fontweight='bold')

    return ax


def plot_cost_difference(
    curve1_df: pd.DataFrame,
    curve2_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    label1: str = "Model 1",
    label2: str = "Model 2",
    **kwargs
) -> plt.Axes:
    """
    Plot difference between two cost curves.

    Shows NC_curve1 - NC_curve2 vs PC.
    Positive values indicate curve1 has higher cost (worse).
    Negative values indicate curve1 has lower cost (better).

    Parameters
    ----------
    curve1_df : pd.DataFrame
        First cost curve
    curve2_df : pd.DataFrame
        Second cost curve (baseline for comparison)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    label1 : str
        Label for first curve
    label2 : str
        Label for second curve
    **kwargs
        Additional arguments passed to plot()

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Align curves by PC (in case they have different PC values)
    # Interpolate curve2 to match curve1's PC values
    pc_common = curve1_df['pc'].values

    nc1 = curve1_df['nc'].values
    nc2 = np.interp(
        pc_common,
        curve2_df['pc'].values,
        curve2_df['nc'].values
    )

    nc_diff = nc1 - nc2

    # Plot difference
    ax.plot(pc_common, nc_diff, **kwargs)

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Color regions
    ax.fill_between(
        pc_common,
        0,
        nc_diff,
        where=(nc_diff >= 0),
        alpha=0.2,
        color='red',
        label=f'{label1} worse'
    )
    ax.fill_between(
        pc_common,
        0,
        nc_diff,
        where=(nc_diff < 0),
        alpha=0.2,
        color='green',
        label=f'{label1} better'
    )

    # Formatting
    ax.set_xlabel('Probability Cost (PC)', fontsize=11)
    ax.set_ylabel(f'NC Difference ({label1} - {label2})', fontsize=11)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Cost Curve Difference: {label1} vs {label2}',
                 fontsize=12, fontweight='bold')

    return ax


def compute_trading_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    trade_returns: Optional[np.ndarray] = None,
    risk_reward_ratios: List[float] = None
) -> pd.DataFrame:
    """
    Compute cost curve with trading-specific cost ratios.

    Maps risk/reward ratios to cost ratios:
    cost_ratio = 1 / (1 + risk_reward_ratio)

    For example:
    - RR = 1.0 (1:1) -> cost_ratio = 0.5 (equal weight to FP and FN)
    - RR = 2.0 (2:1) -> cost_ratio = 0.33 (FN costs 2x more than FP)
    - RR = 3.0 (3:1) -> cost_ratio = 0.25 (FN costs 3x more than FP)

    Parameters
    ----------
    y_true : array-like
        True binary labels (1 for profitable, 0 for unprofitable)
    y_prob : array-like
        Predicted probabilities
    trade_returns : array-like, optional
        Actual trade returns (for weighting costs)
        If provided, costs are weighted by magnitude
    risk_reward_ratios : list of float, optional
        Risk/reward ratios to evaluate
        Default: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    Returns
    -------
    pd.DataFrame
        Cost curve with additional column 'risk_reward_ratio'
    """
    if risk_reward_ratios is None:
        risk_reward_ratios = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Convert RR ratios to cost ratios
    # cost_ratio = cost(FP) / (cost(FP) + cost(FN))
    # If RR = reward/risk, then cost(FN)/cost(FP) = RR
    # So cost_ratio = 1 / (1 + RR)
    cost_ratios = [1.0 / (1.0 + rr) for rr in risk_reward_ratios]

    # Create mapping for later merge
    rr_mapping = pd.DataFrame({
        'cost_ratio': cost_ratios,
        'risk_reward_ratio': risk_reward_ratios
    })

    # Compute cost curve
    curve_df = compute_cost_curve(y_true, y_prob, cost_ratios=np.array(cost_ratios))

    # Merge to add RR ratio column (handles sorting correctly)
    curve_df = curve_df.merge(rr_mapping, on='cost_ratio', how='left')

    # If trade returns provided, could weight by magnitude (future enhancement)
    if trade_returns is not None:
        # Store for reference, but don't modify NC calculation yet
        # This would require a modified cost curve computation
        curve_df['_has_weighted_returns'] = True

    return curve_df


def compare_models_cost_curves(
    models_data: dict,
    cost_ratios: Optional[np.ndarray] = None,
    show_confidence: bool = False,
    n_bootstrap: int = 1000,
    figsize: Tuple[int, int] = (12, 5)
) -> Tuple[plt.Figure, dict]:
    """
    Compare multiple models using cost curves.

    Parameters
    ----------
    models_data : dict
        Dictionary mapping model names to (y_true, y_prob) tuples
        Example: {'Model A': (y_true_a, y_prob_a), 'Model B': (y_true_b, y_prob_b)}
    cost_ratios : array-like, optional
        Cost ratios to evaluate
    show_confidence : bool
        Whether to show bootstrap confidence intervals
    n_bootstrap : int
        Number of bootstrap samples
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with cost curve comparison
    curves : dict
        Dictionary mapping model names to cost curve DataFrames
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))
    curves = {}

    # Plot cost curves
    for (model_name, (y_true, y_prob)), color in zip(models_data.items(), colors):
        if show_confidence:
            mean_curve, lower, upper = bootstrap_cost_curve(
                y_true, y_prob, cost_ratios, n_bootstrap
            )
            plot_cost_curve(
                mean_curve,
                ax=ax1,
                label=model_name,
                show_confidence=True,
                confidence_bounds=(lower, upper),
                color=color
            )
            curves[model_name] = mean_curve
        else:
            curve = compute_cost_curve(y_true, y_prob, cost_ratios)
            plot_cost_curve(curve, ax=ax1, label=model_name, color=color)
            curves[model_name] = curve

    ax1.legend()
    ax1.set_title('Cost Curves Comparison', fontsize=12, fontweight='bold')

    # Plot differences relative to first model
    if len(models_data) >= 2:
        model_names = list(models_data.keys())
        baseline_name = model_names[0]
        baseline_curve = curves[baseline_name]

        for model_name in model_names[1:]:
            plot_cost_difference(
                curves[model_name],
                baseline_curve,
                ax=ax2,
                label1=model_name,
                label2=baseline_name,
                label=f'{model_name} vs {baseline_name}'
            )

        if len(model_names) > 2:
            ax2.legend()

    plt.tight_layout()

    return fig, curves


# Utility function for area under cost curve
def compute_area_under_cost_curve(cost_curve_df: pd.DataFrame) -> float:
    """
    Compute area under the cost curve (AUCC).

    Lower AUCC indicates better overall performance across all cost ratios.

    Parameters
    ----------
    cost_curve_df : pd.DataFrame
        Cost curve data

    Returns
    -------
    float
        Area under the cost curve (0 to 1, lower is better)
    """
    return auc(cost_curve_df['pc'].values, cost_curve_df['nc'].values)
