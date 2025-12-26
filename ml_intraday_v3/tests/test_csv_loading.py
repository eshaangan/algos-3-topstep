"""
Test CSV loading functionality in data ingestion module.

Tests both auto-detection and explicit timestamp column specification.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_intraday_v3.data.ingest import load_raw_data, standardize_ohlcv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_csv(output_path: Path, timestamp_col_name: str = "ts_event"):
    """
    Create a sample CSV file for testing.

    Args:
        output_path: Path to save the CSV file
        timestamp_col_name: Name of the timestamp column
    """
    logger.info(f"Creating sample CSV with timestamp column: {timestamp_col_name}")

    # Generate sample OHLCV data
    n_rows = 1000
    dates = pd.date_range('2024-01-01', periods=n_rows, freq='5min')

    # Generate realistic OHLCV data
    np.random.seed(42)
    base_price = 5000.0
    returns = np.random.randn(n_rows) * 0.001
    close_prices = base_price * (1 + returns).cumprod()

    # Generate OHLC from close
    high = close_prices * (1 + np.abs(np.random.randn(n_rows) * 0.0005))
    low = close_prices * (1 - np.abs(np.random.randn(n_rows) * 0.0005))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    # Volume
    volume = np.random.randint(100, 10000, n_rows)

    df = pd.DataFrame({
        timestamp_col_name: dates,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close_prices,
        'volume': volume
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Created sample CSV at: {output_path}")
    logger.info(f"  Rows: {len(df)}")
    logger.info(f"  Columns: {list(df.columns)}")

    return output_path


def test_csv_auto_detection():
    """Test CSV loading with automatic timestamp column detection."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: CSV Loading with Auto-Detection")
    logger.info("="*70)

    # Create sample CSV with different timestamp column names
    test_cases = [
        ("ts_event", "ts_event (primary)"),
        ("timestamp", "timestamp"),
        ("DateTime", "DateTime (mixed case)"),
        ("Date", "Date"),
        ("Time", "Time"),
    ]

    test_dir = Path("/tmp/ml_intraday_v3_csv_tests")
    test_dir.mkdir(exist_ok=True)

    for col_name, description in test_cases:
        logger.info(f"\nTesting auto-detection with: {description}")
        csv_path = test_dir / f"test_data_{col_name.lower()}.csv"

        # Create sample CSV
        create_sample_csv(csv_path, timestamp_col_name=col_name)

        # Load with auto-detection (timestamp_column=None)
        try:
            df = load_raw_data(
                input_path=csv_path,
                input_format="csv",
                timestamp_column=None,  # Auto-detect
                required_columns=["open", "high", "low", "close", "volume"]
            )

            logger.info(f"✓ Successfully loaded CSV with auto-detected column")
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  Index type: {type(df.index)}")
            logger.info(f"  Index timezone: {df.index.tz}")
            logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")

        except Exception as e:
            logger.error(f"✗ Failed to load CSV: {e}")
            raise


def test_csv_explicit_column():
    """Test CSV loading with explicit timestamp column specification."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: CSV Loading with Explicit Column Name")
    logger.info("="*70)

    test_dir = Path("/tmp/ml_intraday_v3_csv_tests")
    csv_path = test_dir / "test_data_explicit.csv"

    # Create sample CSV
    create_sample_csv(csv_path, timestamp_col_name="my_custom_timestamp")

    # Load with explicit column name
    try:
        df = load_raw_data(
            input_path=csv_path,
            input_format="csv",
            timestamp_column="my_custom_timestamp",
            required_columns=["open", "high", "low", "close", "volume"]
        )

        logger.info(f"✓ Successfully loaded CSV with explicit column name")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")

    except Exception as e:
        logger.error(f"✗ Failed to load CSV: {e}")
        raise


def test_csv_standardization():
    """Test full pipeline: CSV load + standardization."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: CSV Loading + OHLCV Standardization")
    logger.info("="*70)

    test_dir = Path("/tmp/ml_intraday_v3_csv_tests")
    csv_path = test_dir / "test_data_standardize.csv"

    # Create sample CSV
    create_sample_csv(csv_path, timestamp_col_name="timestamp")

    # Load and standardize
    try:
        df = load_raw_data(
            input_path=csv_path,
            input_format="csv",
            timestamp_column=None,
            required_columns=["open", "high", "low", "close", "volume"]
        )

        df_std = standardize_ohlcv(df)

        logger.info(f"✓ Successfully standardized OHLCV data")
        logger.info(f"  Shape: {df_std.shape}")
        logger.info(f"  Columns: {list(df_std.columns)}")
        logger.info(f"  Index name: {df_std.index.name}")
        logger.info(f"  Monotonic increasing: {df_std.index.is_monotonic_increasing}")
        logger.info(f"\nFirst 3 rows:")
        logger.info(f"\n{df_std.head(3)}")
        logger.info(f"\nLast 3 rows:")
        logger.info(f"\n{df_std.tail(3)}")

    except Exception as e:
        logger.error(f"✗ Failed to standardize: {e}")
        raise


def test_with_user_csv(csv_path: str):
    """
    Test with user's actual CSV file.

    Args:
        csv_path: Path to user's CSV file
    """
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Loading User's CSV File")
    logger.info("="*70)

    csv_path = Path(csv_path)

    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        logger.info("Skipping user CSV test")
        return

    logger.info(f"Loading CSV: {csv_path}")
    logger.info(f"File size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")

    try:
        # First, peek at the CSV to see columns
        df_peek = pd.read_csv(csv_path, nrows=5)
        logger.info(f"CSV columns: {list(df_peek.columns)}")
        logger.info(f"\nFirst 3 rows:")
        logger.info(f"\n{df_peek.head(3)}")

        # Try to load with auto-detection
        df = load_raw_data(
            input_path=csv_path,
            input_format="csv",
            timestamp_column=None,  # Auto-detect
            required_columns=["open", "high", "low", "close", "volume"]
        )

        logger.info(f"\n✓ Successfully loaded user's CSV")
        logger.info(f"  Total rows: {len(df):,}")
        logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"  Columns: {list(df.columns)}")

        # Show some statistics
        logger.info(f"\nData summary:")
        logger.info(f"\n{df.describe()}")

    except Exception as e:
        logger.error(f"✗ Failed to load user CSV: {e}")
        logger.info(f"\nYou may need to specify the timestamp column explicitly:")
        logger.info(f"  timestamp_column='your_column_name'")
        raise


if __name__ == "__main__":
    logger.info("Starting CSV loading tests...")

    # Run automated tests
    try:
        test_csv_auto_detection()
        test_csv_explicit_column()
        test_csv_standardization()

        logger.info("\n" + "="*70)
        logger.info("All automated tests passed! ✓")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"\n✗ Tests failed: {e}")
        sys.exit(1)

    # Try to test with user's actual CSV if path provided
    if len(sys.argv) > 1:
        user_csv_path = sys.argv[1]
        test_with_user_csv(user_csv_path)
    else:
        logger.info("\n" + "="*70)
        logger.info("To test with your own CSV file, run:")
        logger.info(f"  python {Path(__file__).name} /path/to/your/data.csv")
        logger.info("="*70)
