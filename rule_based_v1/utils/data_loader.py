"""Data loader for historical MES bars from HDF5 files."""

import pandas as pd
from pathlib import Path


def load_bars(
    path: str | Path = "../ml_intraday_v3/data/processed/mes_bars_databento_rth.h5",
    key: str = "bars",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load historical bars from HDF5 file.

    Args:
        path: Path to HDF5 file.
        key: HDF5 key for the bars dataset.
        start_date: Optional start date filter (inclusive).
        end_date: Optional end date filter (inclusive).

    Returns:
        DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex in US/Eastern timezone.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_hdf(path, key=key)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        elif "datetime" in df.columns:
            df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index)

    # Ensure timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    # Date filtering
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date, tz="US/Eastern")]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date, tz="US/Eastern") + pd.Timedelta(days=1)]

    # Validate required columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df.sort_index()
