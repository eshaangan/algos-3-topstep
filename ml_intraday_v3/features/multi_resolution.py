"""
Multi-resolution feature utilities.

Builds higher-timeframe OHLCV bars from a base timeframe and aligns
their derived features back to the base index without lookahead.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Iterable

import pandas as pd


def aggregate_bars(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    Aggregate OHLCV bars to a larger bar duration in minutes.
    """
    rule = f"{int(minutes)}min"
    agg = df.resample(rule, label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"])
    return agg


def build_multi_resolution_features(
    bars_df: pd.DataFrame,
    base_index: pd.DatetimeIndex,
    config: dict,
    feature_builder: Callable[[pd.DataFrame, str, dict], pd.DataFrame],
) -> pd.DataFrame:
    """
    Build and align higher-timeframe features to the base bar index.

    The `feature_builder` callback is expected to be `build_features`.
    This function disables nested multi-resolution recursion in each call.
    """
    mr_cfg = config.get("multi_resolution", {})
    if not mr_cfg.get("enabled", False):
        return pd.DataFrame(index=base_index)

    resolutions: Iterable[int] = mr_cfg.get("resolutions_min", [15, 30, 60])
    feature_prefix = mr_cfg.get("prefix", "mr")
    align_method = mr_cfg.get("align_method", "ffill")

    result = pd.DataFrame(index=base_index)
    for minutes in resolutions:
        minutes = int(minutes)
        if minutes <= 5:
            continue

        agg_bars = aggregate_bars(bars_df, minutes)
        if agg_bars.empty:
            continue

        nested_cfg = deepcopy(config)
        nested_cfg.setdefault("multi_resolution", {})
        nested_cfg["multi_resolution"]["enabled"] = False

        agg_features = feature_builder(agg_bars, "5m", nested_cfg)

        aligned = agg_features.reindex(base_index, method=align_method)
        aligned.columns = [
            f"{feature_prefix}_{col}_{minutes}m" for col in aligned.columns
        ]
        result = result.join(aligned, how="left")

    return result
