"""
Uniqueness-based sample weights for non-IID events.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def map_event_intervals_to_index(
    events_df: pd.DataFrame, index: pd.Index
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map event t0/t1 timestamps to integer index positions.

    Inclusive convention: event spans [start_idx, end_idx] inclusive.
    Returns -1 for events whose t0/t1 cannot be aligned to the index.
    """
    t0 = pd.to_datetime(events_df["t0"])
    t1 = pd.to_datetime(events_df["t1"])

    start_idx = index.get_indexer(t0)
    end_idx = index.get_indexer(t1)
    return start_idx.astype(int), end_idx.astype(int)


def compute_concurrency(
    index_len: int, start_idx: np.ndarray, end_idx: np.ndarray
) -> np.ndarray:
    """
    Compute concurrency c(t) using diff-array + prefix sum.

    Args:
        index_len: Length of time index
        start_idx: Event start indices (inclusive)
        end_idx: Event end indices (inclusive)

    Returns:
        c: concurrency array of length index_len
    """
    diff = np.zeros(index_len + 1, dtype=np.int64)
    np.add.at(diff, start_idx, 1)
    np.add.at(diff, end_idx + 1, -1)
    return np.cumsum(diff)[:-1]


def compute_uniqueness_weights(
    start_idx: np.ndarray,
    end_idx: np.ndarray,
    inv_conc_prefix: np.ndarray,
) -> np.ndarray:
    """
    Compute uniqueness weights using prefix sums of 1 / c(t).

    Args:
        start_idx: Event start indices (inclusive)
        end_idx: Event end indices (inclusive)
        inv_conc_prefix: Prefix sum array of inv_conc with leading zero

    Returns:
        w_uniqueness: Uniqueness weights for each event
    """
    span_len = (end_idx - start_idx + 1).astype(float)
    sum_inv = inv_conc_prefix[end_idx + 1] - inv_conc_prefix[start_idx]
    return sum_inv / span_len
