"""
Combinatorial Purged Cross-Validation (CPCV) path generation.
"""

import logging
from typing import List, Dict
from itertools import combinations

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


def build_cpcv_paths(
    base_folds: List[Dict],
    events_df: pd.DataFrame,
    bars_index: pd.Index,
    K: int,
    test_groups: int,
    embargo_bars: int,
    max_paths: int | None,
    selection: str = "lexicographic",
) -> List[Dict]:
    """
    Build CPCV paths from base folds.

    Uses combinations of fold indices of size test_groups.
    Applies purge+embargo relative to combined test interval.
    """
    required_cols = ["event_id", "t0", "t1"]
    missing = [c for c in required_cols if c not in events_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(base_folds) != K:
        raise ValueError("base_folds length must equal K")
    if test_groups < 1 or test_groups > K:
        raise ValueError("test_groups must be in [1, K]")
    if selection != "lexicographic":
        raise ValueError(f"Unsupported selection: {selection}")

    df = events_df.copy()
    df["t0"] = pd.to_datetime(df["t0"])
    df["t1"] = pd.to_datetime(df["t1"])
    df = df.sort_values("t0").reset_index(drop=True)

    all_ids = df["event_id"].to_numpy()
    t0 = df["t0"].to_numpy()
    t1 = df["t1"].to_numpy()

    combos = list(combinations(range(K), test_groups))
    if max_paths is not None:
        combos = combos[: int(max_paths)]

    id_to_pos = {eid: i for i, eid in enumerate(all_ids)}

    paths = []
    for path_id, combo in enumerate(combos):
        test_ids = []
        for fold_idx in combo:
            test_ids.extend(base_folds[fold_idx]["test_event_ids"])
        test_ids = sorted(set(test_ids))

        test_pos = [id_to_pos[eid] for eid in test_ids if eid in id_to_pos]
        if not test_pos:
            continue

        test_start = pd.Timestamp(t0[min(test_pos)])
        test_end = pd.Timestamp(t1[max(test_pos)])

        overlap_mask = (t0 <= test_end.to_datetime64()) & (
            t1 >= test_start.to_datetime64()
        )

        test_mask = np.zeros(len(df), dtype=bool)
        test_mask[test_pos] = True

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

        paths.append(
            {
                "path_id": path_id,
                "test_folds": list(combo),
                "test_event_ids": test_ids,
                "train_event_ids": train_ids,
                "test_interval": {
                    "start": test_start.isoformat(),
                    "end": test_end.isoformat(),
                },
                "purge": {"n_purged": n_purged, "n_embargoed": n_embargoed},
                "params": {
                    "K": K,
                    "test_groups": test_groups,
                    "embargo_bars": embargo_bars,
                    "selection": selection,
                },
            }
        )

    return paths
