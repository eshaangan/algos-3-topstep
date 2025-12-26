"""
Example: Loading CSV data using the ingestion module.

This script demonstrates how to load your own CSV files with OHLCV data.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_intraday_v3.data.ingest import load_raw_data, standardize_ohlcv


def load_csv_example():
    """Example: Load CSV with auto-detected timestamp column."""

    # Replace this with your actual CSV file path
    csv_path = Path("data/your_data.csv")

    print("=" * 70)
    print("Example 1: Auto-detect timestamp column")
    print("=" * 70)

    # Load CSV with auto-detection (recommended)
    df = load_raw_data(
        input_path=csv_path,
        input_format="csv",
        timestamp_column=None,  # Auto-detect timestamp column
        required_columns=["open", "high", "low", "close", "volume"]
    )

    print(f"\n✓ Loaded {len(df):,} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\nFirst 5 rows:")
    print(df.head())

    return df


def load_csv_explicit_column():
    """Example: Load CSV with explicit timestamp column name."""

    csv_path = Path("data/your_data.csv")

    print("\n" + "=" * 70)
    print("Example 2: Specify timestamp column explicitly")
    print("=" * 70)

    # Load CSV with explicit timestamp column
    df = load_raw_data(
        input_path=csv_path,
        input_format="csv",
        timestamp_column="ts_event",  # Specify column name
        required_columns=["open", "high", "low", "close", "volume"]
    )

    print(f"\n✓ Loaded {len(df):,} rows")
    print(f"Columns: {list(df.columns)}")

    return df


def load_and_standardize():
    """Example: Load CSV and standardize to canonical OHLCV format."""

    csv_path = Path("data/your_data.csv")

    print("\n" + "=" * 70)
    print("Example 3: Load and standardize OHLCV data")
    print("=" * 70)

    # Load raw data
    df = load_raw_data(
        input_path=csv_path,
        input_format="csv",
        timestamp_column=None,  # Auto-detect
        required_columns=["open", "high", "low", "close", "volume"]
    )

    # Standardize to canonical format
    df_std = standardize_ohlcv(df)

    print(f"\n✓ Standardized {len(df_std):,} rows")
    print(f"Index name: {df_std.index.name}")
    print(f"Index timezone: {df_std.index.tz}")
    print(f"Monotonic increasing: {df_std.index.is_monotonic_increasing}")
    print(f"\nData summary:")
    print(df_std.describe())

    return df_std


def quick_csv_load(csv_path: str):
    """
    Quick helper function to load any CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        Standardized DataFrame with OHLCV data
    """
    csv_path = Path(csv_path)

    # First peek at the file
    print(f"Loading CSV: {csv_path}")
    print(f"File size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")

    df_peek = pd.read_csv(csv_path, nrows=3)
    print(f"\nColumns found: {list(df_peek.columns)}")
    print(f"First 3 rows:\n{df_peek}\n")

    # Load with auto-detection
    df = load_raw_data(
        input_path=csv_path,
        input_format="csv",
        timestamp_column=None,
        required_columns=["open", "high", "low", "close", "volume"]
    )

    # Standardize
    df_std = standardize_ohlcv(df)

    print(f"✓ Loaded and standardized {len(df_std):,} rows")
    print(f"Date range: {df_std.index[0]} to {df_std.index[-1]}")

    return df_std


if __name__ == "__main__":
    # If CSV path provided as argument, load it
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"Loading user CSV: {csv_file}\n")
        df = quick_csv_load(csv_file)

        # Show some statistics
        print("\n" + "=" * 70)
        print("Statistics:")
        print("=" * 70)
        print(f"Total bars: {len(df):,}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Duration: {df.index[-1] - df.index[0]}")
        print(f"\nPrice range:")
        print(f"  Low:  ${df['low'].min():,.2f}")
        print(f"  High: ${df['high'].max():,.2f}")
        print(f"\nVolume:")
        print(f"  Mean: {df['volume'].mean():,.0f}")
        print(f"  Total: {df['volume'].sum():,.0f}")

    else:
        print("Usage:")
        print("  python load_csv_data.py /path/to/your/data.csv")
        print("\nOr modify this script to load your specific CSV file.")
