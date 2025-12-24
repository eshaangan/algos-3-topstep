"""
Selection diagnostics: PBO, DSR, and Cost Curves.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
) -> Dict:
    """
    Compute Deflated Sharpe Ratio (DSR) using per-trade returns.
    """
    returns = np.asarray(list(returns), dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size < 2:
        return {"dsr": None, "reason": "insufficient_returns"}

    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return {"dsr": None, "reason": "zero_variance"}

    sharpe = mean / std
    z = (returns - mean) / std
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    n = returns.size
    sr_var = (1 - skew * sharpe + ((kurt - 1) / 4.0) * sharpe ** 2) / (
        n - 1
    )
    if sr_var <= 0:
        return {"dsr": None, "reason": "nonpositive_variance"}

    sr_std = float(np.sqrt(sr_var))
    trials = max(int(n_trials), 1)
    if trials == 1:
        sr_star = float(target_sharpe)
    else:
        z_trials = NormalDist().inv_cdf(1 - 1.0 / trials)
        sr_star = float(target_sharpe + sr_std * z_trials)

    z_score = (sharpe - sr_star) / sr_std
    dsr = float(NormalDist().cdf(z_score))
    return {
        "dsr": dsr,
        "sharpe": float(sharpe),
        "sr_star": sr_star,
        "sr_std": sr_std,
        "n_obs": int(n),
        "n_trials": trials,
        "target_sharpe": float(target_sharpe),
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
