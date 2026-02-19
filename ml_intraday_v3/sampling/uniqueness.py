"""Sample uniqueness + optional time-decay weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_intraday_v3.weights.uniqueness import (
    compute_concurrency,
    compute_uniqueness_weights,
    map_event_intervals_to_index,
)


def compute_uniqueness_decay_weights(
    events_df: pd.DataFrame,
    bars_index: pd.Index,
    decay_lambda: float = 0.0,
    reference_time: pd.Timestamp | None = None,
) -> pd.Series:
    """Compute event weights from average uniqueness and optional exponential time decay."""
    required = {"event_id", "t0", "t1"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Missing columns for uniqueness weighting: {sorted(missing)}")

    idx = pd.DatetimeIndex(bars_index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    events = events_df.copy()
    events["t0"] = pd.to_datetime(events["t0"], utc=True)
    events["t1"] = pd.to_datetime(events["t1"], utc=True)

    start_idx, end_idx = map_event_intervals_to_index(events, idx)
    valid = (start_idx >= 0) & (end_idx >= 0) & (end_idx >= start_idx)
    if not valid.any():
        return pd.Series(np.ones(len(events)), index=events["event_id"].to_numpy(), dtype=float)

    c = compute_concurrency(len(idx), start_idx[valid], end_idx[valid]).astype(float)
    inv_c = np.zeros_like(c)
    positive = c > 0
    inv_c[positive] = 1.0 / c[positive]

    prefix = np.zeros(len(inv_c) + 1, dtype=float)
    prefix[1:] = np.cumsum(inv_c)

    uniq = np.ones(len(events), dtype=float)
    uniq_vals = compute_uniqueness_weights(start_idx[valid], end_idx[valid], prefix)
    uniq[valid] = uniq_vals

    if float(decay_lambda) > 0:
        ref = reference_time
        if ref is None:
            ref = pd.to_datetime(events["t0"], utc=True).max()
        age_days = (ref - pd.to_datetime(events["t0"], utc=True)).dt.total_seconds() / 86400.0
        decay = np.exp(-float(decay_lambda) * age_days.to_numpy(dtype=float))
        uniq = uniq * decay

    uniq = np.clip(uniq, 1e-8, None)
    uniq = uniq * (len(uniq) / uniq.sum())

    return pd.Series(uniq, index=events["event_id"].to_numpy(), dtype=float)
