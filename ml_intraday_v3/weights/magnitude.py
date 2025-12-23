"""
Magnitude-based sample weights (optional).
"""

import numpy as np
import pandas as pd


def compute_magnitude_weights(
    events_df: pd.DataFrame,
    method: str = "abs_ret_net",
    clip_quantiles: None = None,
) -> pd.Series:
    """
    Compute magnitude weights from realized returns.

    Prefers abs(ret_net) if available; else abs(ret_gross); else abs(ret_points).
    Returns weights >= 0, with optional quantile clipping.
    """
    if method != "abs_ret_net":
        raise ValueError(f"Unsupported method: {method}")

    if "ret_net" in events_df.columns:
        base = events_df["ret_net"].abs()
    elif "ret_gross" in events_df.columns:
        base = events_df["ret_gross"].abs()
    elif "ret_points" in events_df.columns:
        base = events_df["ret_points"].abs()
    else:
        raise ValueError(
            "No return column found for magnitude weights "
            "(expected ret_net, ret_gross, or ret_points)."
        )

    base = base.fillna(0.0).astype(float)

    if clip_quantiles:
        q_low, q_high = clip_quantiles
        low = base.quantile(q_low)
        high = base.quantile(q_high)
        base = base.clip(lower=low, upper=high)

    return base
