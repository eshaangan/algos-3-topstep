"""
Purged K-Fold CV with embargo for event-based labeling.
"""

import logging
from typing import List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _shift_by_bars(
    bars_index: pd.Index, timestamp: pd.Timestamp, embargo_bars: int
) -> pd.Timestamp | None:
    """
    Shift timestamp forward by embargo_bars on bars_index (anchored to next bar).

    Uses the first bar at or after timestamp as the anchor.
    Returns None if timestamp is beyond the last bar.
    """
    end_pos = bars_index.searchsorted(timestamp, side="left")
    if end_pos >= len(bars_index):
        return None
    embargo_end_pos = min(end_pos + int(embargo_bars), len(bars_index) - 1)
    return bars_index[embargo_end_pos]


def build_purged_kfold_splits(
    events_df: pd.DataFrame,
    bars_index: pd.Index,
    n_splits: int,
    embargo_bars: int,
) -> List[Dict]:
    """
    Build time-contiguous purged K-Fold splits over events.

    Splits are contiguous based on sorted event t0.
    Purge removes any training event whose interval overlaps the test interval.
    Embargo removes events with t0 in (test_end, embargo_end].
    """
    required_cols = ["event_id", "t0", "t1"]
    missing = [c for c in required_cols if c not in events_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if n_splits <= 1:
        raise ValueError("n_splits must be >= 2")

    df = events_df.copy()
    df["t0"] = pd.to_datetime(df["t0"])
    df["t1"] = pd.to_datetime(df["t1"])
    df = df.sort_values("t0").reset_index(drop=True)

    n_events = len(df)
    fold_sizes = np.full(n_splits, n_events // n_splits, dtype=int)
    fold_sizes[: n_events % n_splits] += 1
    fold_starts = np.concatenate([[0], np.cumsum(fold_sizes)[:-1]])
    fold_ends = fold_starts + fold_sizes

    all_ids = df["event_id"].to_numpy()
    t0 = df["t0"].to_numpy()
    t1 = df["t1"].to_numpy()

    splits = []
    for i in range(n_splits):
        start = fold_starts[i]
        end = fold_ends[i]
        test_ids = all_ids[start:end]
        test_t0 = t0[start:end]
        test_t1 = t1[start:end]

        test_start = pd.Timestamp(test_t0.min())
        test_end = pd.Timestamp(test_t1.max())

        overlap_mask = (t0 <= test_end.to_datetime64()) & (
            t1 >= test_start.to_datetime64()
        )

        test_mask = np.zeros(n_events, dtype=bool)
        test_mask[start:end] = True

        train_mask = ~test_mask
        n_purged = int((train_mask & overlap_mask).sum())
        train_mask = train_mask & (~overlap_mask)

        n_embargoed = 0
        if embargo_bars > 0:
            embargo_end = _shift_by_bars(bars_index, test_end, embargo_bars)
            if embargo_end is not None:
                embargo_mask = (t0 > test_end.to_datetime64()) & (
                    t0 <= embargo_end.to_datetime64()
                )
                n_embargoed = int((train_mask & embargo_mask).sum())
                train_mask = train_mask & (~embargo_mask)

        train_ids = np.sort(all_ids[train_mask]).tolist()
        test_ids = np.sort(test_ids).tolist()

        splits.append(
            {
                "fold": i,
                "test_event_ids": test_ids,
                "train_event_ids": train_ids,
                "test_interval": {
                    "start": test_start.isoformat(),
                    "end": test_end.isoformat(),
                },
                "purge": {"n_purged": n_purged, "n_embargoed": n_embargoed},
                "params": {"n_splits": n_splits, "embargo_bars": embargo_bars},
            }
        )

    return splits
