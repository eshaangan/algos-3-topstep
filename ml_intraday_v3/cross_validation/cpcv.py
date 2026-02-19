"""CPCV wrapper module with purge/embargo convenience configuration."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ml_intraday_v3.validation.cpcv import build_cpcv_paths
from ml_intraday_v3.validation.purged_cv import build_purged_kfold_splits


def build_cpcv_splits(
    events_df: pd.DataFrame,
    bars_index: pd.Index,
    n_splits: int = 5,
    n_test_splits: int = 2,
    purge_pct: float = 0.02,
    embargo_pct: float = 0.01,
    max_paths: Optional[int] = None,
    selection: str = "balanced",
    random_state: Optional[int] = 42,
):
    """Build CPCV paths using the existing purged-kfold and CPCV implementations."""
    _ = purge_pct  # Purging is interval-based in underlying implementation.
    embargo_bars = max(1, int(round(len(bars_index) * float(embargo_pct))))

    base_folds = build_purged_kfold_splits(
        events_df=events_df,
        bars_index=bars_index,
        n_splits=int(n_splits),
        embargo_bars=embargo_bars,
    )

    paths = build_cpcv_paths(
        base_folds=base_folds,
        events_df=events_df,
        bars_index=bars_index,
        K=int(n_splits),
        test_groups=int(n_test_splits),
        embargo_bars=embargo_bars,
        max_paths=max_paths,
        selection=selection,
        random_state=random_state,
    )
    return paths
