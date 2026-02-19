"""Meta-label generation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_meta_labels(
    realized_returns: pd.Series,
    primary_side: pd.Series,
    transaction_cost: float = 0.0,
    min_edge: float = 0.0,
) -> pd.Series:
    """Meta-label is 1 when the primary signal would have been profitable net of costs."""
    ret = pd.to_numeric(realized_returns, errors="coerce").fillna(0.0)
    side = pd.to_numeric(primary_side, errors="coerce").fillna(0.0)
    pnl = side * ret - float(transaction_cost)
    return pd.Series((pnl > float(min_edge)).astype(int), index=ret.index, name="meta_y")


def attach_meta_labels(labels_df: pd.DataFrame, primary_side_col: str = "side") -> pd.DataFrame:
    """Attach default `meta_y` label to labels/events dataframe."""
    required = {"ret_net", primary_side_col}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"Missing columns for meta labels: {sorted(missing)}")
    out = labels_df.copy()
    out["meta_y"] = create_meta_labels(out["ret_net"], out[primary_side_col])
    return out
