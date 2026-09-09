"""
Data ingestion module.

Loads raw data from HDF5, Parquet, or CSV and standardizes to canonical OHLCV format.
All timestamps are converted to UTC and enforced to be strictly monotonic.
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
import re

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_raw_data(
    input_path: Path,
    input_format: str = "hdf5",
    timestamp_column: Optional[str] = None,
    required_columns: Optional[List[str]] = None,
    hdf_key: Optional[str] = None,
    filter_cfg: Optional[dict] = None,
    symbol_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load raw data from file with optional filtering.

    Args:
        input_path: Path to input file (HDF5, Parquet, or CSV)
        input_format: File format ("hdf5", "parquet", or "csv")
        timestamp_column: Name of timestamp column (None if index is timestamp, or None for auto-detection)
        required_columns: List of required column names
        hdf_key: HDF5 key (required unless file has a single key)
        filter_cfg: Filtering configuration dict (optional)
        symbol_column: Name of symbol column for filtering (optional)

    Returns:
        DataFrame with raw data (filtered if filter_cfg provided)

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required columns are missing
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading raw data from {input_path} (format: {input_format})")

    # Load based on format
    if input_format == "hdf5":
        if hdf_key is None:
            with pd.HDFStore(input_path, mode="r") as store:
                keys = store.keys()
            if len(keys) == 1:
                hdf_key = keys[0]
                logger.info(f"Auto-selected HDF5 key: {hdf_key}")
            else:
                raise ValueError(
                    "HDF5 key not specified and multiple keys found: "
                    f"{keys}. Set ingestion.hdf_key in config."
                )
        df = pd.read_hdf(input_path, key=hdf_key)
    elif input_format == "parquet":
        df = pd.read_parquet(input_path)
    elif input_format == "csv":
        df = pd.read_csv(input_path)
        logger.info(f"Loaded CSV file with {len(df)} rows and columns: {list(df.columns)}")
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = timestamp_column
        if ts_col is None:
            # Smart timestamp column detection (case-insensitive)
            candidates = ["ts_event", "timestamp", "datetime", "date", "time", "ts"]
            lower_map = {c.lower(): c for c in df.columns}
            for name in candidates:
                if name in lower_map:
                    ts_col = lower_map[name]
                    logger.info(f"Auto-detected timestamp column: {ts_col}")
                    break
        if ts_col is None or ts_col not in df.columns:
            raise ValueError(
                "Timestamp column not found. Set ingestion.timestamp_col in config."
            )
        ts = pd.to_datetime(df[ts_col], utc=True, errors="raise")
        df = df.drop(columns=[ts_col]).set_index(ts)
        df = df.sort_index()

        n_dups = df.index.duplicated().sum()
        if n_dups > 0:
            logger.warning(f"Dropping {n_dups} duplicate timestamps (keep last)")
            df = df[~df.index.duplicated(keep="last")]
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif str(df.index.tz) != "UTC":
            df.index = df.index.tz_convert("UTC")

    # Verify required columns exist
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    # Apply filtering if configured
    if filter_cfg and filter_cfg.get("enabled", False):
        df, filter_stats = filter_data(df, filter_cfg, symbol_column)
        logger.info(
            f"Filtering removed {filter_stats['rows_removed']:,} rows, "
            f"{filter_stats['rows_remaining']:,} remaining"
        )

    return df


def filter_data(
    df: pd.DataFrame,
    filter_cfg: dict,
    symbol_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Filter corrupted/invalid data based on configuration.

    Args:
        df: Input DataFrame with raw data
        filter_cfg: Filtering configuration dict
        symbol_column: Name of symbol column (optional)

    Returns:
        Tuple of (filtered_df, stats_dict)
            - filtered_df: DataFrame with corrupted rows removed
            - stats_dict: Statistics about filtering (rows_removed, etc.)

    Raises:
        ValueError: If configuration is invalid
    """
    if not filter_cfg or not filter_cfg.get("enabled", False):
        logger.info("Filtering disabled - returning data unchanged")
        return df, {"rows_removed": 0, "rows_total": len(df)}

    logger.info("Filtering data based on configuration")
    original_len = len(df)
    df_filtered = df.copy()
    stats = {
        "rows_total": original_len,
        "rows_removed": 0,
        "date_filter_removed": 0,
        "symbol_filter_removed": 0,
        "price_validation_removed": 0,
    }

    # Date filtering (train on recent data only)
    date_filter_cfg = filter_cfg.get("date_filter", {})
    if date_filter_cfg.get("enabled", False):
        min_date = date_filter_cfg.get("min_date")
        max_date = date_filter_cfg.get("max_date")

        before_count = len(df_filtered)
        if min_date is not None:
            min_dt = pd.to_datetime(min_date, utc=True)
            df_filtered = df_filtered[df_filtered.index >= min_dt]
        if max_date is not None:
            max_dt = pd.to_datetime(max_date, utc=True)
            df_filtered = df_filtered[df_filtered.index <= max_dt]

        n_removed = before_count - len(df_filtered)
        if n_removed > 0:
            logger.info(
                f"Date filter: {before_count:,} -> {len(df_filtered):,} rows "
                f"(min_date={min_date}, max_date={max_date})"
            )
            stats["date_filter_removed"] = int(n_removed)

    # Symbol filtering (for spread contracts, etc.)
    symbol_filter_cfg = filter_cfg.get("symbol_filter", {})
    if symbol_filter_cfg.get("enabled", False) and symbol_column and symbol_column in df_filtered.columns:
        mode = symbol_filter_cfg.get("mode", "exclude_patterns")
        patterns = symbol_filter_cfg.get("exclude_patterns" if mode == "exclude_patterns" else "include_patterns", [])

        if patterns:
            logger.info(f"Applying symbol filter (mode={mode}, {len(patterns)} patterns)")

            # Combine patterns into single regex
            combined_pattern = "|".join(f"({p})" for p in patterns)
            regex = re.compile(combined_pattern)

            # Apply filter
            symbols = df_filtered[symbol_column].astype(str)
            if mode == "exclude_patterns":
                # Keep rows that DON'T match the patterns
                mask_keep = ~symbols.str.match(regex, na=False)
            else:  # include_patterns
                # Keep rows that DO match the patterns
                mask_keep = symbols.str.match(regex, na=False)

            n_removed = (~mask_keep).sum()
            if n_removed > 0:
                logger.info(f"Symbol filter removing {n_removed:,} rows ({n_removed/original_len*100:.1f}%)")
                df_filtered = df_filtered[mask_keep]
                stats["symbol_filter_removed"] = int(n_removed)

    # Price validation
    price_val_cfg = filter_cfg.get("price_validation", {})
    if price_val_cfg.get("enabled", False):
        min_price = price_val_cfg.get("min_price")
        max_price = price_val_cfg.get("max_price")
        action = price_val_cfg.get("violation_action", "drop_bar")

        price_cols = [c for c in ["open", "high", "low", "close"] if c in df_filtered.columns]
        if price_cols and (min_price is not None or max_price is not None):
            logger.info(f"Validating prices: min={min_price}, max={max_price}, action={action}")

            # Find violations
            violations = pd.Series(False, index=df_filtered.index)
            for col in price_cols:
                if min_price is not None:
                    violations |= df_filtered[col] < min_price
                if max_price is not None:
                    violations |= df_filtered[col] > max_price

            n_violations = violations.sum()
            if n_violations > 0:
                if action == "drop_bar":
                    logger.warning(
                        f"Price validation removing {n_violations:,} rows "
                        f"({n_violations/len(df_filtered)*100:.1f}% of remaining)"
                    )
                    df_filtered = df_filtered[~violations]
                    stats["price_validation_removed"] = int(n_violations)
                elif action == "warn":
                    logger.warning(
                        f"Price validation found {n_violations:,} violations "
                        f"({n_violations/len(df_filtered)*100:.1f}%) - keeping data (action=warn)"
                    )
                elif action == "raise":
                    raise ValueError(
                        f"Price validation failed: {n_violations:,} rows outside "
                        f"[{min_price}, {max_price}]"
                    )

    # Compute final stats
    stats["rows_removed"] = original_len - len(df_filtered)
    stats["rows_remaining"] = len(df_filtered)
    stats["pct_removed"] = (stats["rows_removed"] / original_len * 100) if original_len > 0 else 0.0

    # Log summary
    if filter_cfg.get("log_filtered_stats", True):
        logger.info(
            f"Filtering complete: {stats['rows_removed']:,} rows removed "
            f"({stats['pct_removed']:.1f}%), {stats['rows_remaining']:,} remaining"
        )
        if stats.get("date_filter_removed", 0) > 0:
            logger.info(f"  - Date filter: {stats['date_filter_removed']:,} rows")
        if stats["symbol_filter_removed"] > 0:
            logger.info(f"  - Symbol filter: {stats['symbol_filter_removed']:,} rows")
        if stats["price_validation_removed"] > 0:
            logger.info(f"  - Price validation: {stats['price_validation_removed']:,} rows")

    return df_filtered, stats


def standardize_ohlcv(
    df: pd.DataFrame,
    symbol_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Standardize raw data to canonical OHLCV format.

    Enforces:
    - Timezone-aware UTC timestamps as index
    - Strict monotonic increasing index
    - Standard column names: open, high, low, close, volume, symbol (optional)
    - No duplicate timestamps

    Args:
        df: Raw DataFrame with OHLCV data
        symbol_column: Optional name of symbol column to preserve

    Returns:
        Standardized DataFrame with UTC timestamps and canonical columns

    Raises:
        ValueError: If index is not DatetimeIndex or required columns missing
    """
    logger.info("Standardizing OHLCV data")

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")

    # Make timezone-aware UTC if not already
    if df.index.tz is None:
        logger.info("Converting naive timestamps to UTC")
        df.index = df.index.tz_localize("UTC")
    elif df.index.tz != pd.Timestamp.now(tz="UTC").tz:
        logger.info(f"Converting timezone from {df.index.tz} to UTC")
        df.index = df.index.tz_convert("UTC")

    # Rename index to 'ts' for clarity
    df.index.name = "ts"

    # Verify required OHLCV columns exist
    required = ["open", "high", "low", "close", "volume"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    # Select and order columns
    columns = ["open", "high", "low", "close", "volume"]
    if symbol_column and symbol_column in df.columns:
        columns.append("symbol")
        df = df.rename(columns={symbol_column: "symbol"})

    # Keep only necessary columns
    df = df[columns].copy()

    # Check for duplicates
    n_duplicates = df.index.duplicated().sum()
    if n_duplicates > 0:
        logger.warning(f"Found {n_duplicates} duplicate timestamps - dropping")
        df = df[~df.index.duplicated(keep="first")]

    # Check monotonic increasing
    if not df.index.is_monotonic_increasing:
        logger.warning("Index not monotonic increasing - sorting")
        df = df.sort_index()

    # Final monotonic check after sort
    if not df.index.is_monotonic_increasing:
        raise ValueError("Failed to create monotonic increasing index")

    logger.info(f"Standardized to {len(df)} rows with {len(df.columns)} columns")

    return df
