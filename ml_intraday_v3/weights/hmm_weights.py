"""
HMM Regime-based Sample Weights.

Computes sample weights based on regime similarity to target market conditions.
Allows training on more historical data while downweighting samples from
dissimilar market regimes.

Key functions:
- compute_hmm_regime_weights(): Weight samples by regime probability
- combine_weights(): Merge regime weights with uniqueness/magnitude weights
"""

import logging
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_hmm_regime_weights(
    events_df: pd.DataFrame,
    regime_probs: pd.DataFrame,
    target_regime: Optional[int] = None,
    similarity_method: Literal["probability", "binary"] = "probability",
    discount_factor: float = 0.5,
    t0_column: str = "t0",
) -> pd.Series:
    """
    Compute sample weights based on regime similarity.

    Assigns higher weights to training samples from regimes similar to
    the target regime (typically the current/recent market regime).

    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe with t0 timestamps.
    regime_probs : pd.DataFrame
        State probabilities at each timestep from HMM.
        Must contain columns like 'prob_state_0', 'prob_state_1', etc.
        Optionally 'prob_bull', 'prob_bear' for convenience.
    target_regime : int, optional
        Target regime to weight towards (0 or 1 for 2-state HMM).
        If None, uses the regime at the most recent timestep.
    similarity_method : str, default="probability"
        How to compute weights:
        - "probability": weight = P(target_regime) at event t0 (smooth)
        - "binary": weight = 1.0 if regime matches, else discount_factor
    discount_factor : float, default=0.5
        For binary method: weight for non-matching regime samples.
        Higher = less aggressive downweighting.
    t0_column : str, default="t0"
        Name of timestamp column in events_df.

    Returns
    -------
    w_regime : pd.Series
        Regime-based sample weights, indexed by event_id or events_df index.
        Values in [discount_factor, 1.0] for binary method,
        or [0, 1] for probability method.

    Examples
    --------
    >>> events_df = pd.DataFrame({'t0': pd.date_range('2022-01-01', periods=100)})
    >>> regime_probs = pd.DataFrame({
    ...     'prob_state_0': np.random.rand(100),
    ...     'prob_state_1': np.random.rand(100)
    ... }, index=events_df['t0'])
    >>> w_regime = compute_hmm_regime_weights(events_df, regime_probs, target_regime=1)
    """
    # Validate inputs
    if t0_column not in events_df.columns:
        raise ValueError(f"Column '{t0_column}' not found in events_df")

    # Get event timestamps
    event_times = pd.to_datetime(events_df[t0_column])

    # Detect number of states from column names
    state_cols = [c for c in regime_probs.columns if c.startswith("prob_state_")]
    n_states = len(state_cols)

    if n_states == 0:
        raise ValueError(
            "regime_probs must contain columns like 'prob_state_0', 'prob_state_1', ..."
        )

    # Determine target regime if not specified
    if target_regime is None:
        # Use regime at most recent timestep
        last_probs = regime_probs.iloc[-1]
        target_regime = int(np.argmax([last_probs[f"prob_state_{i}"] for i in range(n_states)]))
        logger.info(f"Using most recent regime as target: state_{target_regime}")

    target_col = f"prob_state_{target_regime}"
    if target_col not in regime_probs.columns:
        raise ValueError(f"Target regime {target_regime} not found in regime_probs")

    # Align regime probs with event times using merge_asof for proper handling
    # of duplicates and forward-fill semantics
    # Ensure timezone consistency by converting both to UTC
    event_times_series = pd.to_datetime(event_times)
    if event_times_series.dt.tz is None:
        event_times_utc = event_times_series.dt.tz_localize('UTC')
    else:
        event_times_utc = event_times_series.dt.tz_convert('UTC')

    if regime_probs.index.tz is None:
        regime_times_utc = regime_probs.index.tz_localize('UTC')
    else:
        regime_times_utc = regime_probs.index.tz_convert('UTC')

    events_for_merge = pd.DataFrame({
        'event_time': event_times_utc,
        'event_idx': range(len(event_times))
    }).sort_values('event_time')

    # Get regime probs with time column
    regime_for_merge = regime_probs[[target_col]].copy()
    regime_for_merge['regime_time'] = regime_times_utc

    # Use merge_asof to align (handles duplicates correctly)
    merged = pd.merge_asof(
        events_for_merge,
        regime_for_merge.reset_index(drop=True),
        left_on='event_time',
        right_on='regime_time',
        direction='backward'  # Use most recent regime at or before event time
    )

    # Sort back to original order and extract probs
    merged = merged.sort_values('event_idx')
    target_probs = merged[target_col].values

    if similarity_method == "probability":
        # Weight = P(target_regime) at event time
        # This provides smooth weighting based on how confident the HMM is
        # about the regime at each event
        weights = target_probs

    elif similarity_method == "binary":
        # Weight = 1.0 if most likely regime matches target, else discount_factor
        # Determine most likely regime at each event time using merge_asof
        regime_all_cols = regime_probs[state_cols].copy()
        regime_all_cols['regime_time'] = regime_times_utc

        merged_all = pd.merge_asof(
            events_for_merge,
            regime_all_cols.reset_index(drop=True),
            left_on='event_time',
            right_on='regime_time',
            direction='backward'
        )
        merged_all = merged_all.sort_values('event_idx')

        event_probs = merged_all[state_cols].values
        most_likely = np.argmax(event_probs, axis=1)

        weights = np.where(most_likely == target_regime, 1.0, discount_factor)

    else:
        raise ValueError(
            f"Unknown similarity_method: {similarity_method}. "
            f"Use 'probability' or 'binary'."
        )

    # Handle any NaN (events before HMM had enough data)
    weights = np.nan_to_num(weights, nan=1.0)

    # Ensure weights are positive
    weights = np.clip(weights, 0.01, 1.0)

    # Create output series
    if "event_id" in events_df.columns:
        index = events_df["event_id"]
    else:
        index = events_df.index

    w_regime = pd.Series(weights, index=index, name="w_regime")

    logger.info(
        f"Computed regime weights: mean={weights.mean():.3f}, "
        f"std={weights.std():.3f}, min={weights.min():.3f}, max={weights.max():.3f}"
    )

    return w_regime


def compute_regime_weights_by_policy(
    events_df: pd.DataFrame,
    regime_probs: pd.DataFrame,
    regime_states: pd.Series,
    policy: Literal["recent", "dominant", "high_vol", "low_vol"] = "recent",
    lookback_bars: int = 100,
    **kwargs,
) -> pd.Series:
    """
    Compute regime weights using different target selection policies.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe with t0 timestamps.
    regime_probs : pd.DataFrame
        State probabilities from HMM.
    regime_states : pd.Series
        Most likely state assignments from HMM.
    policy : str, default="recent"
        Target regime selection policy:
        - "recent": Use regime at most recent bar
        - "dominant": Use most frequent regime in recent lookback window
        - "high_vol": Always target high-volatility state (if labeled)
        - "low_vol": Always target low-volatility state (if labeled)
    lookback_bars : int, default=100
        Lookback window for "dominant" policy.
    **kwargs
        Additional arguments passed to compute_hmm_regime_weights().

    Returns
    -------
    w_regime : pd.Series
        Regime-based sample weights.
    """
    if policy == "recent":
        # Use most likely regime at latest timestep
        target_regime = int(regime_states.dropna().iloc[-1])

    elif policy == "dominant":
        # Use most frequent regime in recent window
        recent_states = regime_states.dropna().iloc[-lookback_bars:]
        target_regime = int(recent_states.mode().iloc[0])

    elif policy == "high_vol":
        # Target the high-volatility state
        # Assumes state with higher variance in emission is high-vol
        # For simplicity, assume state_1 is high-vol (can be refined)
        if "prob_bear" in regime_probs.columns:
            # If we have bull/bear labels, bear is typically high-vol
            target_regime = 0  # Assuming bear = state_0 = high-vol
        else:
            target_regime = 1  # Default assumption

    elif policy == "low_vol":
        # Target the low-volatility state
        if "prob_bull" in regime_probs.columns:
            # Bull is typically low-vol
            target_regime = 1  # Assuming bull = state_1 = low-vol
        else:
            target_regime = 0

    else:
        raise ValueError(f"Unknown policy: {policy}")

    logger.info(f"Using target regime {target_regime} based on '{policy}' policy")

    return compute_hmm_regime_weights(
        events_df=events_df,
        regime_probs=regime_probs,
        target_regime=target_regime,
        **kwargs,
    )


def combine_weights(
    w_uniqueness: Optional[pd.Series] = None,
    w_magnitude: Optional[pd.Series] = None,
    w_regime: Optional[pd.Series] = None,
    uniqueness_exp: float = 1.0,
    magnitude_exp: float = 0.0,
    regime_exp: float = 1.0,
    normalize: bool = True,
) -> pd.Series:
    """
    Combine multiple weight sources into final sample weights.

    w_final = w_uniqueness^a * w_magnitude^b * w_regime^c

    Parameters
    ----------
    w_uniqueness : pd.Series, optional
        Uniqueness-based weights (from event concurrency).
    w_magnitude : pd.Series, optional
        Return magnitude-based weights.
    w_regime : pd.Series, optional
        HMM regime-based weights.
    uniqueness_exp : float, default=1.0
        Exponent for uniqueness weights.
    magnitude_exp : float, default=0.0
        Exponent for magnitude weights (0 = disabled).
    regime_exp : float, default=1.0
        Exponent for regime weights.
    normalize : bool, default=True
        Whether to normalize weights to sum to number of samples.

    Returns
    -------
    w_final : pd.Series
        Combined sample weights.

    Examples
    --------
    >>> w_uniqueness = pd.Series([0.8, 0.9, 1.0, 0.7])
    >>> w_regime = pd.Series([1.0, 0.5, 0.8, 0.3])
    >>> w_final = combine_weights(w_uniqueness, w_regime=w_regime)
    """
    # Collect all weight series
    weights_list = []
    exponents = []

    if w_uniqueness is not None and uniqueness_exp != 0:
        weights_list.append(w_uniqueness)
        exponents.append(uniqueness_exp)

    if w_magnitude is not None and magnitude_exp != 0:
        weights_list.append(w_magnitude)
        exponents.append(magnitude_exp)

    if w_regime is not None and regime_exp != 0:
        weights_list.append(w_regime)
        exponents.append(regime_exp)

    if len(weights_list) == 0:
        raise ValueError("At least one weight source must be provided")

    # Align indices
    common_index = weights_list[0].index
    for w in weights_list[1:]:
        common_index = common_index.intersection(w.index)

    if len(common_index) == 0:
        raise ValueError("No common indices between weight sources")

    # Compute combined weight
    w_final = pd.Series(np.ones(len(common_index)), index=common_index)

    for w, exp in zip(weights_list, exponents):
        aligned_w = w.loc[common_index].fillna(1.0)
        # Clip to avoid issues with 0^exp
        aligned_w = np.clip(aligned_w, 0.01, np.inf)
        w_final = w_final * (aligned_w ** exp)

    # Normalize so weights sum to n_samples (sklearn convention)
    if normalize:
        w_final = w_final * len(w_final) / w_final.sum()

    w_final.name = "w_final"

    logger.info(
        f"Combined weights: mean={w_final.mean():.3f}, "
        f"std={w_final.std():.3f}, sum={w_final.sum():.1f}"
    )

    return w_final


def analyze_regime_weight_distribution(
    w_regime: pd.Series,
    regime_states: pd.Series,
    events_df: pd.DataFrame,
    t0_column: str = "t0",
) -> pd.DataFrame:
    """
    Analyze how regime weights are distributed across time and regimes.

    Parameters
    ----------
    w_regime : pd.Series
        Regime-based sample weights.
    regime_states : pd.Series
        HMM regime assignments.
    events_df : pd.DataFrame
        Events dataframe.
    t0_column : str
        Timestamp column name.

    Returns
    -------
    analysis : pd.DataFrame
        Statistics per regime: count, mean_weight, std_weight, total_weight.
    """
    event_times = pd.to_datetime(events_df[t0_column])

    # Get regime at each event
    regime_at_event = regime_states.reindex(
        regime_states.index.union(event_times)
    ).sort_index().ffill().loc[event_times]

    # Build analysis dataframe
    analysis_df = pd.DataFrame({
        "w_regime": w_regime.values,
        "regime": regime_at_event.values,
    })

    # Group by regime
    analysis = analysis_df.groupby("regime").agg(
        count=("w_regime", "count"),
        mean_weight=("w_regime", "mean"),
        std_weight=("w_regime", "std"),
        total_weight=("w_regime", "sum"),
    ).round(4)

    analysis["pct_samples"] = (analysis["count"] / analysis["count"].sum() * 100).round(2)
    analysis["pct_weight"] = (analysis["total_weight"] / analysis["total_weight"].sum() * 100).round(2)

    return analysis
