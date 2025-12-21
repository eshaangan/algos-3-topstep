"""
Aligned fixed-horizon labels that match next-bar entry and time-exit execution.

Entry assumption: next bar open (t+1).
Exit assumption: horizon bar close after entry (t+1+horizon).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def make_aligned_fixed_horizon_labels(
    df: pd.DataFrame,
    horizon_bars: int,
    threshold_ticks: int,
    tick_size: float,
    *,
    entry_price_col: str = "open",
    exit_price_col: str = "close",
) -> pd.DataFrame:
    """
    Create fixed-horizon labels aligned to next-bar entry and time-exit execution.

    Args:
        df: Bars DataFrame with price columns.
        horizon_bars: Number of bars to hold after entry.
        threshold_ticks: Movement threshold in ticks for directional labels.
        tick_size: Tick size for the instrument.
        entry_price_col: Column name for entry price (default: next bar open).
        exit_price_col: Column name for exit price (default: horizon bar close).

    Returns:
        DataFrame with idx, ret_ticks, y_long, y_short.
    """
    if horizon_bars < 1:
        raise ValueError(f"horizon_bars must be >= 1, got {horizon_bars}")
    if threshold_ticks < 1:
        raise ValueError(f"threshold_ticks must be >= 1, got {threshold_ticks}")
    if tick_size <= 0:
        raise ValueError(f"tick_size must be > 0, got {tick_size}")
    if entry_price_col not in df.columns:
        raise ValueError(f"df missing entry_price_col={entry_price_col!r}")
    if exit_price_col not in df.columns:
        raise ValueError(f"df missing exit_price_col={exit_price_col!r}")

    min_len = horizon_bars + 2  # needs t+1 (entry) and t+1+horizon (exit)
    if len(df) < min_len:
        raise ValueError(
            f"Not enough rows for horizon_bars={horizon_bars}: need >= {min_len}, got {len(df)}"
        )

    entry_price = df[entry_price_col].astype(float).shift(-1)
    exit_price = df[exit_price_col].astype(float).shift(-1 - horizon_bars)
    ret_ticks = (exit_price - entry_price) / tick_size

    y_long = (ret_ticks >= float(threshold_ticks)).astype(int)
    y_short = (ret_ticks <= -float(threshold_ticks)).astype(int)

    labels = pd.DataFrame(
        {
            "idx": np.arange(len(df)),
            "ret_ticks": ret_ticks,
            "y_long": y_long,
            "y_short": y_short,
        }
    )

    labels = labels.dropna(subset=["ret_ticks"]).copy()
    if labels.empty:
        raise ValueError("Aligned labels are empty after dropping NaNs.")
    if labels["ret_ticks"].isna().any():
        raise ValueError("Aligned labels contain NaNs; check input data.")

    return labels
