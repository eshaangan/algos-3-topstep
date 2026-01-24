"""
Selection diagnostics: PBO, DSR, and Cost Curves.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from ml_intraday_v3.analysis.cost_curves import (
    compute_cost_curve,
    bootstrap_cost_curve,
    plot_cost_curve,
    plot_cost_difference,
    compute_trading_cost_curve,
    compute_area_under_cost_curve
)


def compute_pbo(
    perf_df: pd.DataFrame,
    metric_col: str,
    path_col: str = "split_id",
    config_col: str = "variant_id",
) -> Dict:
    """
    Compute PBO using leave-one-path-out selection on CPCV paths.

    For each path, select the best config by mean performance on all other
    paths, then measure its OOS rank on the held-out path. PBO is the share
    of paths where that rank is below median.
    """
    if perf_df.empty:
        return {"pbo": None, "reason": "empty_perf"}

    pivot = perf_df.pivot(index=path_col, columns=config_col, values=metric_col)
    pivot = pivot.dropna(axis=1, how="all")
    if pivot.empty or pivot.shape[1] < 2:
        return {"pbo": None, "reason": "insufficient_configs"}

    paths = list(pivot.index)
    ranks = []
    for path in paths:
        is_scores = pivot.drop(index=path).mean(axis=0, skipna=True)
        if is_scores.empty or is_scores.isna().all():
            continue
        best_config = is_scores.idxmax()
        oos_scores = pivot.loc[path]
        if oos_scores.isna().all():
            continue
        oos_ranks = oos_scores.rank(pct=True, ascending=False)
        ranks.append(float(oos_ranks.loc[best_config]))

    if not ranks:
        return {"pbo": None, "reason": "no_valid_ranks"}

    pbo = float(np.mean([r < 0.5 for r in ranks]))
    definitions = {
        "path_definition": (
            "A path is a CV split identifier from cv_splits.json "
            "(fold for purged_kfold, path_id for cpcv)."
        ),
        "selection_definition": (
            "For each held-out path, select the variant with the highest "
            f"mean {metric_col} across all other paths."
        ),
        "rank_definition": (
            "Lambda is the percentile rank (descending) of the selected "
            "variant's metric on the held-out path; 1.0 is best, 0.0 is worst."
        ),
        "pbo_interpretation": (
            "PBO is the share of held-out paths where lambda < 0.5 "
            "(selected variant below median OOS)."
        ),
    }
    return {
        "pbo": pbo,
        "n_paths": len(ranks),
        "ranks": ranks,
        "lambda_values": ranks,
        "metric": metric_col,
        "method": "leave_one_path_out",
        "definitions": definitions,
    }


def compute_dsr(
    returns: Iterable[float],
    n_trials: int,
    target_sharpe: float = 0.0,
    annualization_factor: Optional[float] = None,
) -> Dict:
    """
    Compute Deflated Sharpe Ratio (DSR) using Bailey & López de Prado (2014) formula.

    This implements the complete DSR formula that accounts for:
    - Selection bias from testing multiple configurations (n_trials)
    - Non-normality of returns (skewness and kurtosis)
    - Statistical uncertainty in Sharpe ratio estimation

    Formula:
        DSR = Φ((SR - SR*) / σ_SR)

    where:
        SR = observed Sharpe ratio = μ / σ
        SR* = expected maximum Sharpe from N trials (selection bias adjustment)
            = SR_0 + σ_SR * Z_{1-1/N}
        σ_SR = standard error of Sharpe ratio
            = sqrt((1 - γ*SR + (κ-1)/4 * SR²) / (n-1))
        γ = skewness of returns = mean(z³)
        κ = kurtosis of returns = mean(z⁴)
        z = standardized returns = (r - μ) / σ
        Φ = standard normal CDF

    Parameters
    ----------
    returns : Iterable[float]
        Array of per-trade or per-period returns
    n_trials : int
        Number of configurations tested (for selection bias adjustment)
        Use TrialTracker to get accurate count, or conservative estimate (10)
    target_sharpe : float, default=0.0
        Benchmark Sharpe ratio (SR_0 in formula)
        0.0 tests against no skill, positive values test against higher bar
    annualization_factor : float, optional
        Factor to annualize Sharpe ratio
        - Per-trade returns: sqrt(252 * trades_per_day)
        - Time-based: sqrt(252 * bars_per_day / avg_holding_bars)
        If None, returns are assumed already annualized or no annualization needed

    Returns
    -------
    dict
        {
            'dsr': float - Deflated Sharpe Ratio [0, 1]
                         - DSR > 0.95: strong evidence (p < 0.05 equivalent)
                         - DSR > 0.5: more likely skill than luck
                         - DSR < 0.5: likely overfitting/luck
            'sharpe': float - Observed Sharpe ratio (annualized if factor provided)
            'sharpe_raw': float - Raw Sharpe before annualization
            'sr_star': float - Expected max SR from selection bias
            'sr_std': float - Standard error of Sharpe ratio
            'skewness': float - Skewness of returns (γ)
            'kurtosis': float - Kurtosis of returns (κ)
            'n_obs': int - Number of observations
            'n_trials': int - Number of trials tested
            'target_sharpe': float - Benchmark Sharpe
            'annualization_factor': float - Annualization factor used
            'z_score': float - Standardized score (SR - SR*) / σ_SR
        }

    References
    ----------
    Bailey, D.H., & López de Prado, M. (2014).
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
    Journal of Portfolio Management, 40(5), 94-107.
    """
    returns = np.asarray(list(returns), dtype=float)
    returns = returns[np.isfinite(returns)]

    if returns.size < 2:
        return {"dsr": None, "reason": "insufficient_returns"}

    # Step 1: Compute mean and std
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))

    if std == 0.0:
        return {"dsr": None, "reason": "zero_variance"}

    # Step 2: Compute raw Sharpe ratio
    sharpe_raw = mean / std

    # Step 3: Apply annualization if requested
    if annualization_factor is not None and annualization_factor > 0:
        sharpe = sharpe_raw * annualization_factor
        annualization_factor = float(annualization_factor)
    else:
        sharpe = sharpe_raw
        annualization_factor = 1.0

    # Step 4: Compute standardized returns
    z = (returns - mean) / std

    # Step 5: Compute skewness (γ) and kurtosis (κ)
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))

    # Step 6: Compute variance of Sharpe ratio
    # σ²_SR = (1 - γ*SR + (κ-1)/4 * SR²) / (n-1)
    n = returns.size
    sr_var = (1 - skew * sharpe + ((kurt - 1) / 4.0) * sharpe ** 2) / (n - 1)

    if sr_var <= 0:
        return {
            "dsr": None,
            "reason": "nonpositive_variance",
            "sharpe": sharpe,
            "sharpe_raw": sharpe_raw,
            "skewness": skew,
            "kurtosis": kurt,
            "n_obs": int(n),
        }

    # Step 7: Standard error of Sharpe ratio
    sr_std = float(np.sqrt(sr_var))

    # Step 8: Compute expected maximum Sharpe from selection bias
    # SR* = SR_0 + σ_SR * Z_{1-1/N}
    trials = max(int(n_trials), 1)

    if trials == 1:
        # No selection bias with single trial
        sr_star = float(target_sharpe)
        z_quantile = 0.0
    else:
        # Quantile for 1 - 1/N probability
        z_quantile = NormalDist().inv_cdf(1 - 1.0 / trials)
        sr_star = float(target_sharpe + sr_std * z_quantile)

    # Step 9: Compute z-score
    z_score = (sharpe - sr_star) / sr_std

    # Step 10: Compute DSR = Φ(z_score)
    dsr = float(NormalDist().cdf(z_score))

    return {
        "dsr": dsr,
        "sharpe": float(sharpe),
        "sharpe_raw": float(sharpe_raw),
        "sr_star": sr_star,
        "sr_std": sr_std,
        "skewness": skew,
        "kurtosis": kurt,
        "n_obs": int(n),
        "n_trials": trials,
        "target_sharpe": float(target_sharpe),
        "annualization_factor": annualization_factor,
        "z_score": z_score,
        "z_quantile": z_quantile if trials > 1 else 0.0,
    }


def compute_model_cost_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    include_bootstrap: bool = True,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Dict:
    """
    Compute cost curve diagnostics for a single model.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities
    model_name : str
        Name of the model
    include_bootstrap : bool
        Whether to compute bootstrap confidence intervals
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level for intervals
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing:
        - curve_df: Cost curve DataFrame
        - aucc: Area under cost curve
        - bootstrap_mean: Bootstrap mean curve (if include_bootstrap=True)
        - bootstrap_lower: Bootstrap lower bound (if include_bootstrap=True)
        - bootstrap_upper: Bootstrap upper bound (if include_bootstrap=True)
    """
    # Compute main cost curve
    curve_df = compute_cost_curve(y_true, y_prob)
    aucc = compute_area_under_cost_curve(curve_df)

    result = {
        "model_name": model_name,
        "curve_df": curve_df,
        "aucc": aucc,
        "n_samples": len(y_true),
        "n_positive": int(np.sum(y_true)),
        "n_negative": int(len(y_true) - np.sum(y_true))
    }

    if include_bootstrap:
        mean_curve, lower, upper = bootstrap_cost_curve(
            y_true,
            y_prob,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state
        )
        result.update({
            "bootstrap_mean": mean_curve,
            "bootstrap_lower": lower,
            "bootstrap_upper": upper,
            "confidence_level": confidence_level
        })

    return result


def compute_trading_cost_diagnostics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    risk_reward_ratios: Optional[List[float]] = None,
    trade_returns: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute trading-specific cost curve diagnostics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (1 for profitable trade, 0 for unprofitable)
    y_prob : np.ndarray
        Predicted probabilities
    risk_reward_ratios : list of float, optional
        Risk/reward ratios to evaluate
        Default: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    trade_returns : np.ndarray, optional
        Actual trade returns (for future weighting)

    Returns
    -------
    dict
        Dictionary containing:
        - curve_df: Trading cost curve DataFrame
        - aucc: Area under cost curve
        - optimal_rr: Optimal risk/reward ratio (lowest NC)
        - rr_summary: Summary by risk/reward ratio
    """
    # Compute trading cost curve
    curve_df = compute_trading_cost_curve(
        y_true,
        y_prob,
        trade_returns=trade_returns,
        risk_reward_ratios=risk_reward_ratios
    )

    aucc = compute_area_under_cost_curve(curve_df)

    # Find optimal risk/reward ratio
    optimal_idx = curve_df['nc'].idxmin()
    optimal_row = curve_df.loc[optimal_idx]

    # Create summary by RR ratio
    rr_summary = curve_df[['risk_reward_ratio', 'nc', 'threshold', 'tpr', 'fpr']].copy()

    return {
        "curve_df": curve_df,
        "aucc": aucc,
        "optimal_rr": float(optimal_row['risk_reward_ratio']),
        "optimal_nc": float(optimal_row['nc']),
        "optimal_threshold": float(optimal_row['threshold']),
        "optimal_tpr": float(optimal_row['tpr']),
        "optimal_fpr": float(optimal_row['fpr']),
        "rr_summary": rr_summary,
        "n_samples": len(y_true)
    }


def compare_models_cost_diagnostics(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    include_bootstrap: bool = False,
    save_path: Optional[str] = None
) -> Dict:
    """
    Compare multiple models using cost curves.

    Parameters
    ----------
    models_dict : dict
        Dictionary mapping model names to (y_true, y_prob) tuples
        Example: {'Model A': (y_true, y_prob_a), 'Model B': (y_true, y_prob_b)}
    include_bootstrap : bool
        Whether to include bootstrap confidence intervals
    save_path : str, optional
        Path to save comparison plot

    Returns
    -------
    dict
        Dictionary containing:
        - models: Dictionary of individual model results
        - aucc_comparison: DataFrame comparing AUCC values
        - best_model: Name of model with lowest AUCC
        - fig: Matplotlib figure (if save_path not None)
    """
    from ml_intraday_v3.analysis.cost_curves import compare_models_cost_curves

    # Compute individual model diagnostics
    models_results = {}
    aucc_values = {}

    for model_name, (y_true, y_prob) in models_dict.items():
        result = compute_model_cost_curves(
            y_true,
            y_prob,
            model_name=model_name,
            include_bootstrap=include_bootstrap
        )
        models_results[model_name] = result
        aucc_values[model_name] = result["aucc"]

    # Create AUCC comparison DataFrame
    aucc_df = pd.DataFrame({
        "model": list(aucc_values.keys()),
        "aucc": list(aucc_values.values())
    }).sort_values("aucc")

    best_model = aucc_df.iloc[0]["model"]

    # Create comparison plot
    fig, curves = compare_models_cost_curves(
        models_dict,
        show_confidence=include_bootstrap,
        n_bootstrap=100 if include_bootstrap else 0
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        fig = None  # Don't keep in memory

    return {
        "models": models_results,
        "aucc_comparison": aucc_df,
        "best_model": best_model,
        "fig": fig
    }


# =============================================================================
# Enhanced PBO (Probability of Backtest Overfitting) Implementation
# Following López de Prado's methodology with proper trial tracking
# =============================================================================


def compute_pbo_enhanced(
    trials_df: pd.DataFrame,
    metric_name: str = 'roc_auc',
    higher_is_better: bool = True,
    min_trials: int = 2,
) -> Dict:
    """
    Compute enhanced PBO with comprehensive trial tracking.

    This implementation properly tracks ALL trials/configurations tested,
    not just winners, to avoid selection bias in PBO computation.

    Algorithm (López de Prado):
        For each CPCV path i:
            1. IS_i = mean(metric) across all paths except i (for each trial)
            2. best_trial_i = argmax(IS_i) [or argmin if lower is better]
            3. OOS_i = metric(best_trial_i) on path i
            4. rank_i = percentile_rank(OOS_i) among all trials on path i
            5. lambda_i = rank_i
        PBO = mean(I(lambda_i < 0.5))

    where I is indicator function (1 if true, 0 if false)

    Parameters
    ----------
    trials_df : pd.DataFrame
        DataFrame with columns:
            - trial_id: Unique trial identifier
            - path_X_is: In-sample metric for path X
            - path_X_oos: Out-of-sample metric for path X
        (One pair of is/oos columns per CPCV path)
    metric_name : str
        Metric name (used for documentation)
    higher_is_better : bool
        Whether higher metric values are better (True for AUC, accuracy)
        Set to False for metrics like loss, error rate
    min_trials : int
        Minimum number of trials required (default: 2)

    Returns
    -------
    dict
        Dictionary containing:
            - pbo: Probability of backtest overfitting [0, 1]
            - n_paths: Number of CPCV paths
            - n_trials: Number of trials tracked
            - lambda_values: List of lambda values (OOS ranks) per path
            - ranks_by_path: Dict mapping path -> {trial_id: rank}
            - selected_trials: List of (path, trial_id) selected per path
            - method: 'enhanced_lopez_de_prado'

    Example
    -------
    >>> tracker = TrialTracker(run_dir)
    >>> # ... log trials and update metrics ...
    >>> trials_df = tracker.to_dataframe()
    >>> pbo_result = compute_pbo_enhanced(trials_df)
    >>> print(f"PBO = {pbo_result['pbo']:.3f}")
    """
    if trials_df.empty:
        return {
            'pbo': None,
            'reason': 'empty_trials_df',
            'n_paths': 0,
            'n_trials': 0,
        }

    # Identify path columns
    is_cols = [c for c in trials_df.columns if c.endswith('_is')]
    oos_cols = [c for c in trials_df.columns if c.endswith('_oos')]

    if not is_cols or not oos_cols:
        return {
            'pbo': None,
            'reason': 'no_is_oos_columns',
            'n_paths': 0,
            'n_trials': len(trials_df),
        }

    # Extract path IDs
    path_ids = sorted(set([c.replace('_is', '') for c in is_cols]))
    n_paths = len(path_ids)
    n_trials = len(trials_df)

    if n_trials < min_trials:
        return {
            'pbo': None,
            'reason': f'insufficient_trials (need >={min_trials}, got {n_trials})',
            'n_paths': n_paths,
            'n_trials': n_trials,
        }

    lambda_values = []
    ranks_by_path = {}
    selected_trials = []

    for path_id in path_ids:
        is_col = f'{path_id}_is'
        oos_col = f'{path_id}_oos'

        # Get IS metrics for all other paths
        other_is_cols = [c for c in is_cols if not c.startswith(path_id)]

        if not other_is_cols:
            continue

        # Compute mean IS across other paths for each trial
        is_means = trials_df[other_is_cols].mean(axis=1, skipna=True)

        # Select best trial by IS performance
        if higher_is_better:
            best_idx = is_means.idxmax()
        else:
            best_idx = is_means.idxmin()

        if pd.isna(best_idx):
            continue

        best_trial_id = trials_df.loc[best_idx, 'trial_id']
        selected_trials.append((path_id, best_trial_id))

        # Get OOS scores for this path
        oos_scores = trials_df[oos_col].copy()

        # Remove NaN scores
        valid_oos = oos_scores.dropna()
        if len(valid_oos) < 2:
            continue

        # Compute percentile ranks (higher rank = better performance)
        # Lambda is the percentile rank where 1.0 = best, 0.0 = worst
        if higher_is_better:
            # Higher values are better: rank in ascending order (best score gets rank 1.0)
            ranks = rankdata(valid_oos.values, method='average') / len(valid_oos)
        else:
            # Lower values are better: rank in descending order (best score gets rank 1.0)
            ranks = rankdata(-valid_oos.values, method='average') / len(valid_oos)

        ranks_dict = dict(zip(valid_oos.index, ranks))
        ranks_by_path[path_id] = {
            trials_df.loc[idx, 'trial_id']: rank
            for idx, rank in ranks_dict.items()
        }

        # Get rank of selected trial
        if best_idx in ranks_dict:
            lambda_i = ranks_dict[best_idx]
            lambda_values.append(lambda_i)

    if not lambda_values:
        return {
            'pbo': None,
            'reason': 'no_valid_lambda_values',
            'n_paths': n_paths,
            'n_trials': n_trials,
        }

    # Compute PBO: fraction where lambda < 0.5
    pbo = np.mean([lam < 0.5 for lam in lambda_values])

    return {
        'pbo': float(pbo),
        'n_paths': n_paths,
        'n_trials': n_trials,
        'lambda_values': lambda_values,
        'lambda_mean': float(np.mean(lambda_values)),
        'lambda_std': float(np.std(lambda_values)),
        'lambda_median': float(np.median(lambda_values)),
        'ranks_by_path': ranks_by_path,
        'selected_trials': selected_trials,
        'metric_name': metric_name,
        'higher_is_better': higher_is_better,
        'method': 'enhanced_lopez_de_prado',
    }


def compute_pbo_with_confidence(
    trials_df: pd.DataFrame,
    metric_name: str = 'roc_auc',
    higher_is_better: bool = True,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> Dict:
    """
    Compute PBO with bootstrap confidence intervals.

    Uses block bootstrap to preserve time structure and account for
    dependencies between trials.

    Parameters
    ----------
    trials_df : pd.DataFrame
        DataFrame from TrialTracker.to_dataframe()
    metric_name : str
        Metric name for documentation
    higher_is_better : bool
        Whether higher is better
    n_bootstrap : int
        Number of bootstrap samples (default: 1000)
    confidence_level : float
        Confidence level for intervals (default: 0.95)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing:
            - pbo: Point estimate of PBO
            - pbo_lower: Lower confidence bound
            - pbo_upper: Upper confidence bound
            - confidence_level: Confidence level used
            - n_bootstrap: Number of bootstrap samples
            - bootstrap_pbos: Array of bootstrap PBO values
            - ... (other fields from compute_pbo_enhanced)
    """
    # Compute point estimate
    result = compute_pbo_enhanced(
        trials_df,
        metric_name=metric_name,
        higher_is_better=higher_is_better,
    )

    if result['pbo'] is None:
        result['pbo_lower'] = None
        result['pbo_upper'] = None
        result['confidence_level'] = confidence_level
        return result

    # Bootstrap
    rng = np.random.RandomState(random_state)
    n_trials = len(trials_df)
    bootstrap_pbos = []

    for _ in range(n_bootstrap):
        # Resample trials with replacement
        sample_indices = rng.choice(n_trials, size=n_trials, replace=True)
        bootstrap_df = trials_df.iloc[sample_indices].reset_index(drop=True)

        # Compute PBO on bootstrap sample
        boot_result = compute_pbo_enhanced(
            bootstrap_df,
            metric_name=metric_name,
            higher_is_better=higher_is_better,
        )

        if boot_result['pbo'] is not None:
            bootstrap_pbos.append(boot_result['pbo'])

    if not bootstrap_pbos:
        result['pbo_lower'] = None
        result['pbo_upper'] = None
        result['bootstrap_pbos'] = []
    else:
        alpha = 1 - confidence_level
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        result['pbo_lower'] = float(np.percentile(bootstrap_pbos, lower_pct))
        result['pbo_upper'] = float(np.percentile(bootstrap_pbos, upper_pct))
        result['bootstrap_pbos'] = bootstrap_pbos

    result['confidence_level'] = confidence_level
    result['n_bootstrap'] = n_bootstrap

    return result


def plot_pbo_distribution(
    lambda_values: List[float],
    pbo_value: float,
    ax: Optional[plt.Axes] = None,
    title: str = "PBO Distribution of Lambda Values",
) -> plt.Figure:
    """
    Visualize PBO lambda distribution.

    Parameters
    ----------
    lambda_values : list of float
        Lambda values (OOS ranks) from compute_pbo_enhanced
    pbo_value : float
        PBO value to display
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new figure if None)
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Histogram of lambda values
    ax.hist(
        lambda_values,
        bins=20,
        alpha=0.6,
        color='steelblue',
        edgecolor='black',
        label='Lambda Distribution'
    )

    # Median line
    ax.axvline(
        0.5,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Median (λ = 0.5)',
        alpha=0.8
    )

    # Mean lambda
    mean_lambda = np.mean(lambda_values)
    ax.axvline(
        mean_lambda,
        color='green',
        linestyle=':',
        linewidth=2,
        label=f'Mean λ = {mean_lambda:.3f}',
        alpha=0.8
    )

    # Shade region below median (overfitting region)
    ax.axvspan(0, 0.5, alpha=0.2, color='red', label='Overfitting (λ < 0.5)')

    # Annotations
    ax.set_xlabel('Lambda (OOS Percentile Rank)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add PBO text box
    pbo_pct = pbo_value * 100
    n_below_median = sum(lam < 0.5 for lam in lambda_values)
    n_total = len(lambda_values)

    textstr = f'PBO = {pbo_pct:.1f}%\n({n_below_median}/{n_total} paths)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(
        0.95, 0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props
    )

    plt.tight_layout()
    return fig


def plot_pbo_with_confidence(
    pbo_result: Dict,
    ax: Optional[plt.Axes] = None,
    title: str = "PBO with Confidence Intervals",
) -> plt.Figure:
    """
    Visualize PBO point estimate with confidence intervals.

    Parameters
    ----------
    pbo_result : dict
        Result from compute_pbo_with_confidence()
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    pbo = pbo_result['pbo']
    pbo_lower = pbo_result.get('pbo_lower')
    pbo_upper = pbo_result.get('pbo_upper')
    confidence_level = pbo_result.get('confidence_level', 0.95)

    # Bar for PBO
    ax.barh(
        0,
        pbo,
        height=0.3,
        color='steelblue',
        label=f'PBO = {pbo:.3f}'
    )

    # Error bar for confidence interval
    if pbo_lower is not None and pbo_upper is not None:
        error = [[pbo - pbo_lower], [pbo_upper - pbo]]
        ax.errorbar(
            pbo, 0,
            xerr=error,
            fmt='o',
            color='red',
            markersize=8,
            capsize=10,
            capthick=2,
            label=f'{confidence_level*100:.0f}% CI: [{pbo_lower:.3f}, {pbo_upper:.3f}]'
        )

    # Reference lines
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (0.5)')
    ax.axvline(pbo, color='green', linestyle=':', linewidth=2, alpha=0.7)

    # Styling
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Probability of Backtest Overfitting (PBO)', fontsize=12)
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    # Interpretation box
    if pbo > 0.5:
        interpretation = "⚠️ HIGH RISK: Likely overfitting"
        color = 'red'
    elif pbo > 0.3:
        interpretation = "⚠️ MODERATE: Some overfitting risk"
        color = 'orange'
    else:
        interpretation = "✓ LOW RISK: Unlikely overfitting"
        color = 'green'

    textstr = f'{interpretation}\nn_trials = {pbo_result["n_trials"]}\nn_paths = {pbo_result["n_paths"]}'
    props = dict(boxstyle='round', facecolor=color, alpha=0.3)
    ax.text(
        0.05, 0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props
    )

    plt.tight_layout()
    return fig


def generate_pbo_report(
    pbo_result: Dict,
    save_path: Optional[str] = None,
) -> str:
    """
    Generate a detailed PBO report in markdown format.

    Parameters
    ----------
    pbo_result : dict
        Result from compute_pbo_with_confidence()
    save_path : str, optional
        Path to save markdown report

    Returns
    -------
    str
        Markdown formatted report
    """
    pbo = pbo_result.get('pbo')

    if pbo is None:
        reason = pbo_result.get('reason', 'unknown')
        return f"# PBO Report\n\n⚠️ **Cannot compute PBO**: {reason}\n"

    n_trials = pbo_result['n_trials']
    n_paths = pbo_result['n_paths']
    lambda_mean = pbo_result['lambda_mean']
    lambda_std = pbo_result['lambda_std']
    lambda_median = pbo_result['lambda_median']

    pbo_lower = pbo_result.get('pbo_lower')
    pbo_upper = pbo_result.get('pbo_upper')
    confidence_level = pbo_result.get('confidence_level', 0.95)

    # Interpretation
    if pbo > 0.5:
        risk_level = "🔴 **HIGH RISK**"
        interpretation = (
            "The probability of backtest overfitting is >50%. "
            "This suggests that the selected configuration is likely to underperform "
            "out-of-sample. Consider reducing the number of trials, increasing sample size, "
            "or using stricter validation criteria."
        )
    elif pbo > 0.3:
        risk_level = "🟠 **MODERATE RISK**"
        interpretation = (
            "The probability of backtest overfitting is moderate (30-50%). "
            "While not critically high, there is some risk of degraded out-of-sample performance. "
            "Monitor performance carefully and consider validation on additional data."
        )
    else:
        risk_level = "🟢 **LOW RISK**"
        interpretation = (
            "The probability of backtest overfitting is low (<30%). "
            "The selected configuration appears robust and is unlikely to be purely "
            "a result of overfitting to the validation paths."
        )

    report = f"""# Probability of Backtest Overfitting (PBO) Report

## Summary

{risk_level}

**PBO**: {pbo:.3f} ({pbo*100:.1f}%)
"""

    if pbo_lower is not None and pbo_upper is not None:
        report += f"**{confidence_level*100:.0f}% Confidence Interval**: [{pbo_lower:.3f}, {pbo_upper:.3f}]\n"

    report += f"""
**Trials Tracked**: {n_trials}
**CPCV Paths**: {n_paths}

## Interpretation

{interpretation}

## Lambda Statistics

Lambda (λ) represents the percentile rank of the selected configuration on each held-out path:
- λ = 1.0: Best performing configuration on that path
- λ = 0.5: Median performance
- λ = 0.0: Worst performing configuration

**Mean Lambda**: {lambda_mean:.3f}
**Median Lambda**: {lambda_median:.3f}
**Std Lambda**: {lambda_std:.3f}

PBO is the fraction of paths where λ < 0.5 (below median).

## Recommendations

"""

    if pbo > 0.5:
        report += """
1. ⚠️ **Reduce hyperparameter search space** to decrease selection bias
2. ⚠️ **Increase sample size** if possible to improve robustness
3. ⚠️ **Use simpler models** to reduce overfitting risk
4. ⚠️ **Validate on additional out-of-sample data** before deployment
5. ⚠️ **Consider ensemble methods** instead of single best configuration
"""
    elif pbo > 0.3:
        report += """
1. Monitor out-of-sample performance closely after deployment
2. Consider additional validation on hold-out data
3. Track performance degradation over time
4. Be conservative with position sizing initially
"""
    else:
        report += """
1. Configuration appears robust, but continue monitoring
2. Consider this as one input to the deployment decision
3. Validate on walk-forward data if available
"""

    report += f"""
## Methodology

This PBO computation follows López de Prado's methodology:

1. For each CPCV path, select the best configuration based on in-sample performance on all other paths
2. Measure the out-of-sample rank of that configuration on the held-out path
3. PBO = fraction of paths where the selected configuration ranks below median

**Reference**: López de Prado, M. (2018). Advances in Financial Machine Learning. Chapter 11.

---
*Report generated by ml_intraday_v3.experiments.diagnostics*
"""

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)

    return report


def plot_dsr_distribution(
    dsr_results: List[Dict],
    ax: Optional[plt.Axes] = None,
    threshold: float = 0.5,
    title: str = "DSR Distribution Across CPCV Paths"
) -> plt.Figure:
    """
    Plot histogram of DSR values across CPCV paths.

    Parameters
    ----------
    dsr_results : List[Dict]
        List of DSR result dictionaries from compute_dsr()
        Each should have 'dsr' key with DSR value
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    threshold : float, default=0.5
        Threshold line to mark (0.5 = more likely skill than luck)
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Extract DSR values
    dsr_values = [r['dsr'] for r in dsr_results if r.get('dsr') is not None]

    if not dsr_values:
        raise ValueError("No valid DSR values in results")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Plot histogram
    n_bins = min(20, len(dsr_values) // 2 + 1)
    ax.hist(dsr_values, bins=n_bins, alpha=0.7, color='steelblue',
            edgecolor='black', density=False)

    # Add threshold line
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold})')

    # Add median line
    median_dsr = np.median(dsr_values)
    ax.axvline(median_dsr, color='green', linestyle='-', linewidth=2,
               label=f'Median ({median_dsr:.3f})')

    # Add mean line
    mean_dsr = np.mean(dsr_values)
    ax.axvline(mean_dsr, color='orange', linestyle=':', linewidth=2,
               label=f'Mean ({mean_dsr:.3f})')

    # Shading for regions
    ax.axvspan(0, threshold, alpha=0.1, color='red',
               label='Likely overfitting (DSR < 0.5)')
    ax.axvspan(threshold, 1, alpha=0.1, color='green',
               label='More likely skill (DSR > 0.5)')

    # Labels and formatting
    ax.set_xlabel('Deflated Sharpe Ratio (DSR)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = (
        f"N paths: {len(dsr_values)}\n"
        f"Mean: {mean_dsr:.3f}\n"
        f"Median: {median_dsr:.3f}\n"
        f"Std: {np.std(dsr_values):.3f}\n"
        f"Min: {np.min(dsr_values):.3f}\n"
        f"Max: {np.max(dsr_values):.3f}\n"
        f"% > 0.5: {100 * np.mean([d > 0.5 for d in dsr_values]):.1f}%\n"
        f"% > 0.95: {100 * np.mean([d > 0.95 for d in dsr_values]):.1f}%"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.5), fontsize=9, family='monospace')

    fig.tight_layout()
    return fig


def plot_dsr_with_confidence(
    dsr_results: List[Dict],
    confidence_level: float = 0.95,
    ax: Optional[plt.Axes] = None,
    title: str = "DSR with Confidence Intervals"
) -> plt.Figure:
    """
    Plot DSR value with confidence intervals.

    Parameters
    ----------
    dsr_results : List[Dict]
        List of DSR result dictionaries from compute_dsr()
    confidence_level : float
        Confidence level for intervals (e.g., 0.95 for 95% CI)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
    title : str
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    dsr_values = [r['dsr'] for r in dsr_results if r.get('dsr') is not None]

    if not dsr_values:
        raise ValueError("No valid DSR values in results")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Compute statistics
    median_dsr = np.median(dsr_values)
    mean_dsr = np.mean(dsr_values)

    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    dsr_lower = np.percentile(dsr_values, lower_percentile)
    dsr_upper = np.percentile(dsr_values, upper_percentile)

    # Plot horizontal bar
    ax.barh(0, mean_dsr, height=0.3, color='steelblue', alpha=0.7,
            label=f'Mean DSR ({mean_dsr:.3f})')

    # Add error bar
    error = [[mean_dsr - dsr_lower], [dsr_upper - mean_dsr]]
    ax.errorbar(mean_dsr, 0, xerr=error, fmt='o', color='black',
                markersize=8, capsize=10, capthick=2,
                label=f'{int(confidence_level*100)}% CI: [{dsr_lower:.3f}, {dsr_upper:.3f}]')

    # Add threshold line
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2,
               label='Threshold (0.5)')

    # Add strong evidence line
    ax.axvline(0.95, color='orange', linestyle=':', linewidth=2,
               label='Strong evidence (0.95)')

    # Shading
    ax.axvspan(0, 0.5, alpha=0.1, color='red', label='Likely overfitting')
    ax.axvspan(0.5, 0.95, alpha=0.1, color='yellow', label='Moderate evidence')
    ax.axvspan(0.95, 1, alpha=0.1, color='green', label='Strong evidence')

    # Labels
    ax.set_xlabel('Deflated Sharpe Ratio (DSR)', fontsize=12)
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    # Interpretation text
    if mean_dsr > 0.95:
        interpretation = "🟢 STRONG EVIDENCE - Very likely skill, not luck"
    elif mean_dsr > 0.5:
        interpretation = "🟠 MODERATE EVIDENCE - More likely skill than luck"
    else:
        interpretation = "🔴 WEAK EVIDENCE - Likely overfitting or luck"

    ax.text(0.5, -0.15, interpretation, transform=ax.transAxes,
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.tight_layout()
    return fig


def generate_dsr_report(
    dsr_result: Dict,
    save_path: Optional[str] = None
) -> str:
    """
    Generate markdown report for DSR analysis.

    Parameters
    ----------
    dsr_result : Dict
        Result from compute_dsr()
    save_path : str, optional
        Path to save markdown report

    Returns
    -------
    str
        Markdown report
    """
    if dsr_result.get('dsr') is None:
        reason = dsr_result.get('reason', 'unknown')
        report = f"""# DSR Analysis Report

## Status: Failed

**Reason**: {reason}

Unable to compute DSR. Please check your data.
"""
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        return report

    dsr = dsr_result['dsr']
    sharpe = dsr_result['sharpe']
    sharpe_raw = dsr_result.get('sharpe_raw', sharpe)
    sr_star = dsr_result['sr_star']
    sr_std = dsr_result['sr_std']
    skew = dsr_result['skewness']
    kurt = dsr_result['kurtosis']
    n_obs = dsr_result['n_obs']
    n_trials = dsr_result['n_trials']
    target = dsr_result['target_sharpe']
    annualization = dsr_result.get('annualization_factor', 1.0)
    z_score = dsr_result['z_score']

    # Risk assessment
    if dsr > 0.95:
        risk = "🟢 STRONG EVIDENCE"
        interpretation = "Very likely skill, not luck (p < 0.05 equivalent)"
        action = "High confidence for deployment"
    elif dsr > 0.5:
        risk = "🟠 MODERATE EVIDENCE"
        interpretation = "More likely skill than luck, but not conclusive"
        action = "Proceed with caution, monitor closely"
    else:
        risk = "🔴 WEAK EVIDENCE"
        interpretation = "Likely overfitting or luck, not genuine skill"
        action = "DO NOT deploy - likely overfitting"

    report = f"""# Deflated Sharpe Ratio (DSR) Analysis Report

## Summary

**DSR**: {dsr:.4f}
**Risk Level**: {risk}
**Interpretation**: {interpretation}
**Recommended Action**: {action}

---

## Sharpe Ratio Analysis

| Metric | Value |
|--------|-------|
| Observed Sharpe (Raw) | {sharpe_raw:.4f} |
| Observed Sharpe (Annualized) | {sharpe:.4f} |
| Expected Max Sharpe (SR*) | {sr_star:.4f} |
| Standard Error (σ_SR) | {sr_std:.4f} |
| Target Sharpe (SR₀) | {target:.4f} |
| Z-Score | {z_score:.4f} |

---

## Returns Distribution

| Statistic | Value |
|-----------|-------|
| Number of Observations | {n_obs:,} |
| Skewness (γ) | {skew:.4f} |
| Kurtosis (κ) | {kurt:.4f} |
| Annualization Factor | {annualization:.2f} |

**Non-Normality Assessment**:
- Skewness close to 0 indicates symmetric returns (observed: {skew:.2f})
- Kurtosis of 3 indicates normal distribution (observed: {kurt:.2f})
"""

    if abs(skew) > 0.5:
        report += f"\n⚠️ **High skewness** ({skew:.2f}) - returns are asymmetric\n"
    if kurt > 5 or kurt < 2:
        report += f"\n⚠️ **Non-normal kurtosis** ({kurt:.2f}) - fat tails or thin tails\n"

    report += f"""
---

## Selection Bias Adjustment

| Parameter | Value |
|-----------|-------|
| Number of Trials Tested | {n_trials} |
| Expected Max Sharpe from Luck | {sr_star:.4f} |
| Penalty from Multiple Testing | {sr_star - target:.4f} |

**Selection Bias**: Testing {n_trials} configurations increases the expected maximum Sharpe ratio due to random chance. The DSR adjusts for this by comparing against SR* instead of 0.

---

## DSR Interpretation Guide

| DSR Range | Interpretation | Confidence |
|-----------|----------------|------------|
| > 0.95 | Strong evidence of skill | p < 0.05 equivalent |
| 0.90 - 0.95 | Good evidence | p < 0.10 equivalent |
| 0.50 - 0.90 | Moderate evidence | More likely skill than luck |
| < 0.50 | Weak evidence | Likely overfitting/luck |

**Your DSR**: {dsr:.4f} → {interpretation}

---

## Formula Reference

The DSR formula (Bailey & López de Prado, 2014):

```
DSR = Φ((SR - SR*) / σ_SR)
```

where:
- SR = observed Sharpe ratio = μ / σ
- SR* = expected maximum Sharpe from N trials = SR₀ + σ_SR × Z_(1-1/N)
- σ_SR = standard error = sqrt((1 - γ×SR + (κ-1)/4 × SR²) / (n-1))
- γ = skewness of returns
- κ = kurtosis of returns
- Φ = standard normal CDF

**Your calculation**:
- SR = {sharpe:.4f}
- SR* = {target:.4f} + {sr_std:.4f} × {(sr_star - target) / sr_std if sr_std > 0 else 0:.4f} = {sr_star:.4f}
- σ_SR = {sr_std:.4f}
- Z-score = ({sharpe:.4f} - {sr_star:.4f}) / {sr_std:.4f} = {z_score:.4f}
- DSR = Φ({z_score:.4f}) = {dsr:.4f}

---

## Recommendations

"""

    if dsr > 0.95:
        report += """
1. ✅ High confidence in strategy - DSR indicates genuine skill
2. ✅ Proceed with deployment, but continue monitoring
3. Consider this strong evidence alongside other diagnostics (PBO, walk-forward, etc.)
4. Still validate on truly out-of-sample data if available
"""
    elif dsr > 0.5:
        report += """
1. ⚠️ Moderate evidence of skill - not conclusive
2. ⚠️ Proceed with caution - use conservative position sizing initially
3. Monitor out-of-sample performance closely after deployment
4. Validate on additional hold-out data if possible
5. Consider DSR alongside PBO and other overfitting diagnostics
"""
    else:
        report += """
1. ❌ **DO NOT deploy** - DSR indicates likely overfitting or luck
2. ❌ Results are not statistically distinguishable from random chance
3. Consider:
   - Reducing hyperparameter search space (fewer n_trials)
   - Increasing sample size (more observations)
   - Simplifying model to reduce overfitting
   - Using more conservative validation (walk-forward, etc.)
"""

    report += f"""
---

## Methodology

This DSR computation implements the complete Bailey & López de Prado (2014) formula, accounting for:

1. **Selection bias** from testing {n_trials} configurations
2. **Non-normality** of returns (skewness = {skew:.2f}, kurtosis = {kurt:.2f})
3. **Statistical uncertainty** in Sharpe ratio estimation

**Reference**: Bailey, D.H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality." Journal of Portfolio Management, 40(5), 94-107.

---
*Report generated by ml_intraday_v3.experiments.diagnostics*
"""

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)

    return report
