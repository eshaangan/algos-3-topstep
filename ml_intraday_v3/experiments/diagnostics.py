"""
Selection diagnostics: PBO and DSR.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


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
