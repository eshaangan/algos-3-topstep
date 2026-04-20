"""Causal CRT/TBS structure features.

CRT is modeled as candle/range context. TBS is modeled as a Turtle
Soup-style sweep of a prior range with same-bar reclaim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_crt_tbs_features(
    bars_df: pd.DataFrame,
    *,
    atr: pd.Series | None = None,
    range_lookback: int = 20,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Build causal CRT/TBS features from OHLC bars."""
    if range_lookback < 2:
        raise ValueError("range_lookback must be >= 2")

    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(bars_df.columns))
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")

    df = bars_df.sort_index()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    close = df["close"].astype(float)

    prior_high = high.shift(1).rolling(range_lookback, min_periods=range_lookback).max()
    prior_low = low.shift(1).rolling(range_lookback, min_periods=range_lookback).min()
    prior_mid = (prior_high + prior_low) / 2.0
    prior_width = prior_high - prior_low

    candle_range = high - low
    body_abs = (close - open_).abs()

    if atr is None:
        atr_ref = candle_range.ewm(span=14, adjust=False, min_periods=1).mean()
    else:
        atr_ref = atr.reindex(df.index).astype(float)

    sweep_low = low < prior_low
    sweep_high = high > prior_high
    long_reclaim = sweep_low & (close > prior_low)
    short_reclaim = sweep_high & (close < prior_high)

    long_sweep_distance = (prior_low - low).where(sweep_low, 0.0)
    short_sweep_distance = (high - prior_high).where(sweep_high, 0.0)

    out = pd.DataFrame(index=df.index)
    out["crt_range_pos"] = (close - prior_low) / (prior_width + eps)
    out["crt_close_vs_mid"] = (close - prior_mid) / (prior_width + eps)
    out["crt_body_quality"] = body_abs / (candle_range + eps)
    out["crt_close_location"] = (close - low) / (candle_range + eps)
    out["crt_displacement_atr"] = body_abs / (atr_ref + eps)
    out["tbs_long_setup"] = long_reclaim.astype(int)
    out["tbs_short_setup"] = short_reclaim.astype(int)
    out["tbs_sweep_distance_atr"] = (
        long_sweep_distance - short_sweep_distance
    ) / (atr_ref + eps)
    out["tbs_reclaim_confirmed"] = long_reclaim.astype(int) - short_reclaim.astype(int)

    numeric_cols = [
        "crt_range_pos",
        "crt_close_vs_mid",
        "crt_body_quality",
        "crt_close_location",
        "crt_displacement_atr",
        "tbs_sweep_distance_atr",
    ]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out
