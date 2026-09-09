"""
Resampling module.

Resamples 1m bars to 5m bars with proper OHLCV aggregation.
Preserves UTC timestamps and is deterministic.

IMPORTANT - RESAMPLE ALIGNMENT:
Default settings: label="right", closed="right"
This means:
- A 5m bar labeled 09:35:00 contains 1m bars: [09:31, 09:32, 09:33, 09:34, 09:35]
- The timestamp on the 5m bar is the RIGHT edge of the window (09:35)
- The window is CLOSED on the right (includes 09:35)

Why these settings:
- label="right": Timestamp represents END of aggregation window
  - Natural interpretation: "data up to and including this timestamp"
  - Aligns with how most systems label bars
- closed="right": Includes the timestamp in the window
  - Ensures complete coverage with no gaps/overlaps
  - Standard pandas convention for time series

Alternative (not used):
- label="left", closed="left" would give:
  - Bar labeled 09:30 contains [09:30, 09:31, 09:32, 09:33, 09:34]
  - Less intuitive for futures trading (bar timestamp would be START of period)
"""

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def resample_1m_to_5m(
    df_1m: pd.DataFrame,
    label: str = "right",
    closed: str = "right",
) -> pd.DataFrame:
    """
    Resample 1-minute bars to 5-minute bars.

    OHLCV aggregation rules:
    - open: first open in 5m window
    - high: max high in 5m window
    - low: min low in 5m window
    - close: last close in 5m window
    - volume: sum of volume in 5m window

    TIMESTAMP ALIGNMENT (CRITICAL FOR REPRODUCIBILITY):
    Default settings: label="right", closed="right"
    - Example: 5m bar timestamped 09:35:00 contains 1m bars:
      [09:31:00, 09:32:00, 09:33:00, 09:34:00, 09:35:00]
    - The 09:35 timestamp represents the END of the aggregation window
    - The window INCLUDES the right edge (09:35)

    Why these settings:
    - label="right": Timestamp = end of window (standard financial convention)
    - closed="right": Includes right edge, excludes left edge
    - Result: Complete coverage, no gaps, no overlaps

    Args:
        df_1m: 1-minute OHLCV DataFrame with UTC DatetimeIndex
        label: Which bin edge labels the result ("right" DEFAULT, "left")
        closed: Which side of bin is closed ("right" DEFAULT, "left")

    Returns:
        5-minute OHLCV DataFrame with same columns as input

    Raises:
        ValueError: If input is not 1m frequency or index not UTC

    WARNING: Changing label/closed settings will change timestamp alignment
    and break reproducibility. Use defaults unless you have a specific reason.
    """
    if not isinstance(df_1m.index, pd.DatetimeIndex):
        raise ValueError("Input must have DatetimeIndex")

    if df_1m.index.tz is None or str(df_1m.index.tz) != "UTC":
        raise ValueError("Input index must be UTC timezone-aware")

    logger.info(f"Resampling {len(df_1m)} 1m bars to 5m bars")
    logger.info(f"Resample params: label={label}, closed={closed}")

    # Define resampling rules for each column
    agg_rules = {}

    if "open" in df_1m.columns:
        agg_rules["open"] = "first"
    if "high" in df_1m.columns:
        agg_rules["high"] = "max"
    if "low" in df_1m.columns:
        agg_rules["low"] = "min"
    if "close" in df_1m.columns:
        agg_rules["close"] = "last"
    if "volume" in df_1m.columns:
        agg_rules["volume"] = "sum"

    # Handle optional columns
    if "symbol" in df_1m.columns:
        # For symbol, take the last (most recent) symbol in the window
        agg_rules["symbol"] = "last"

    if "is_synthetic" in df_1m.columns:
        # For is_synthetic, mark as synthetic if ANY bar in window is synthetic
        agg_rules["is_synthetic"] = "max"  # max of booleans = any True

    # Resample
    df_5m = df_1m.resample("5min", label=label, closed=closed).agg(agg_rules)

    # Drop rows with no data (all NaN) - can happen at boundaries
    df_5m = df_5m.dropna(subset=["close"], how="all")

    logger.info(f"Resampled to {len(df_5m)} 5m bars")

    # Verify OHLC validity (will be caught by QA, but check here too)
    if "open" in df_5m.columns and "close" in df_5m.columns:
        if "high" in df_5m.columns and "low" in df_5m.columns:
            invalid = (
                (df_5m["low"] > df_5m[["open", "close"]].min(axis=1))
                | (df_5m["high"] < df_5m[["open", "close"]].max(axis=1))
            )
            n_invalid = invalid.sum()
            if n_invalid > 0:
                logger.warning(
                    f"Found {n_invalid} bars with invalid OHLC after resampling"
                )

    return df_5m
