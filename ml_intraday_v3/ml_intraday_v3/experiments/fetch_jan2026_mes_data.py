#!/usr/bin/env python3
"""
Fetch Full January 2026 MES Data from Data Bento

Downloads 1-minute bars for the complete month of January 2026
and saves both 1m and 5m versions for backtesting and analysis.

This will give us REAL market data to validate our model's performance
instead of using simulated data.

Usage:
    python ml_intraday_v3/experiments/fetch_jan2026_mes_data.py
"""

import sys
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[3]  # Go up to algos 3 topstep
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from {env_path}")
else:
    print(f"⚠️ No .env file found at {env_path}")
    print("   Make sure DATABENTO_API_KEY is set in your environment")

from ml_intraday_v3.live_trading.data_fetcher import LiveDataFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def resample_to_5min(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute bars to 5-minute bars.

    Args:
        bars_1m: DataFrame with 1-minute OHLCV bars

    Returns:
        DataFrame with 5-minute OHLCV bars
    """
    logger.info(f"Resampling {len(bars_1m):,} 1-minute bars to 5-minute bars...")

    # Ensure index is datetime
    if not isinstance(bars_1m.index, pd.DatetimeIndex):
        bars_1m.index = pd.to_datetime(bars_1m.index)

    # Resample using OHLCV logic
    bars_5m = bars_1m.resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()

    logger.info(f"✅ Resampled to {len(bars_5m):,} 5-minute bars")
    return bars_5m


def main():
    print("="*80)
    print("FETCHING JANUARY 2026 MES DATA FROM DATA BENTO")
    print("="*80)
    print()

    # Configuration
    symbol = "MES.c.0"  # Continuous front month MES
    start_date = "2026-01-01"
    end_date = "2026-01-31"

    print(f"Symbol:     {symbol}")
    print(f"Start Date: {start_date}")
    print(f"End Date:   {end_date}")
    print(f"Bar Size:   1-minute (will also create 5-minute)")
    print()

    # Create output directory
    output_dir = Path("ml_intraday_v3/data/jan2026_mes")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output Directory: {output_dir}")
    print()

    # Initialize data fetcher
    print("="*80)
    print("STEP 1: INITIALIZE DATA FETCHER")
    print("="*80)

    try:
        fetcher = LiveDataFetcher(
            symbol=symbol,
            bar_size="1m",
            lookback_bars=100,
        )
        print("✅ Data fetcher initialized successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize data fetcher")
        print(f"   {e}")
        print()
        print("TROUBLESHOOTING:")
        print("1. Check that DATABENTO_API_KEY is set in .env file")
        print("2. Verify your Data Bento API key is valid")
        print("3. Ensure you have an active Data Bento subscription")
        return 1

    print()

    # Fetch 1-minute bars
    print("="*80)
    print("STEP 2: FETCH 1-MINUTE BARS")
    print("="*80)
    print(f"Downloading from {start_date} to {end_date}...")
    print("(This may take 1-2 minutes depending on data size)")
    print()

    try:
        bars_1m = fetcher.fetch_historical(
            start_date=start_date,
            end_date=end_date
        )

        print(f"✅ Downloaded {len(bars_1m):,} 1-minute bars")
        print(f"   Date range: {bars_1m.index[0]} to {bars_1m.index[-1]}")
        print(f"   Columns: {list(bars_1m.columns)}")

        # Calculate trading days
        trading_days = bars_1m.index.normalize().nunique()
        bars_per_day = len(bars_1m) / trading_days
        print(f"   Trading days: {trading_days}")
        print(f"   Avg bars/day: {bars_per_day:.1f}")

    except Exception as e:
        print(f"❌ ERROR: Failed to download data")
        print(f"   {e}")
        print()
        print("POSSIBLE CAUSES:")
        print("1. No data available for this date range (Jan 2026 in future?)")
        print("2. API key doesn't have access to GLBX.MDP3 dataset")
        print("3. Network connection issue")
        return 1

    print()

    # Save 1-minute bars
    print("="*80)
    print("STEP 3: SAVE 1-MINUTE BARS")
    print("="*80)

    bars_1m_path = output_dir / "mes_jan2026_1m.parquet"
    bars_1m.to_parquet(bars_1m_path)
    print(f"✅ Saved to: {bars_1m_path}")

    # Also save as CSV for easy inspection
    bars_1m_csv = output_dir / "mes_jan2026_1m.csv"
    bars_1m.to_csv(bars_1m_csv)
    print(f"✅ Saved to: {bars_1m_csv} (for inspection)")

    print()

    # Resample to 5-minute bars
    print("="*80)
    print("STEP 4: RESAMPLE TO 5-MINUTE BARS")
    print("="*80)

    bars_5m = resample_to_5min(bars_1m)

    bars_per_day_5m = len(bars_5m) / trading_days
    print(f"   Trading days: {trading_days}")
    print(f"   Avg bars/day: {bars_per_day_5m:.1f}")
    print()

    # Save 5-minute bars
    print("="*80)
    print("STEP 5: SAVE 5-MINUTE BARS")
    print("="*80)

    bars_5m_path = output_dir / "mes_jan2026_5m.parquet"
    bars_5m.to_parquet(bars_5m_path)
    print(f"✅ Saved to: {bars_5m_path}")

    # Also save as CSV
    bars_5m_csv = output_dir / "mes_jan2026_5m.csv"
    bars_5m.to_csv(bars_5m_csv)
    print(f"✅ Saved to: {bars_5m_csv} (for inspection)")

    print()

    # Summary statistics
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print()

    print("1-MINUTE BARS:")
    print(f"   Total bars: {len(bars_1m):,}")
    print(f"   Start: {bars_1m.index[0]}")
    print(f"   End: {bars_1m.index[-1]}")
    print(f"   Price range: ${bars_1m['low'].min():.2f} - ${bars_1m['high'].max():.2f}")
    print(f"   Total volume: {bars_1m['volume'].sum():,}")
    print()

    print("5-MINUTE BARS:")
    print(f"   Total bars: {len(bars_5m):,}")
    print(f"   Start: {bars_5m.index[0]}")
    print(f"   End: {bars_5m.index[-1]}")
    print(f"   Price range: ${bars_5m['low'].min():.2f} - ${bars_5m['high'].max():.2f}")
    print(f"   Total volume: {bars_5m['volume'].sum():,}")
    print()

    # Create metadata file
    print("="*80)
    print("STEP 6: CREATE METADATA")
    print("="*80)

    metadata = {
        'symbol': symbol,
        'start_date': start_date,
        'end_date': end_date,
        'downloaded_at': datetime.now().isoformat(),
        'bars_1m': {
            'count': len(bars_1m),
            'file': str(bars_1m_path),
            'start': str(bars_1m.index[0]),
            'end': str(bars_1m.index[-1]),
        },
        'bars_5m': {
            'count': len(bars_5m),
            'file': str(bars_5m_path),
            'start': str(bars_5m.index[0]),
            'end': str(bars_5m.index[-1]),
        },
        'trading_days': trading_days,
        'bars_per_day_1m': bars_per_day,
        'bars_per_day_5m': bars_per_day_5m,
    }

    import json
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to: {metadata_path}")
    print()

    # Next steps
    print("="*80)
    print("✅ DOWNLOAD COMPLETE")
    print("="*80)
    print()
    print("NEXT STEPS:")
    print()
    print("1. Inspect the data:")
    print(f"   head {bars_5m_csv}")
    print()
    print("2. Run backtest with real Jan 2026 data:")
    print("   python ml_intraday_v3/backtesting_v3/backtest_runner.py \\")
    print(f"       --data-file {bars_5m_path} \\")
    print("       --start-date 2026-01-01 \\")
    print("       --end-date 2026-01-31")
    print()
    print("3. Test 2-contract sizing on REAL Jan 2026 data:")
    print("   (Will create this script next)")
    print()
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
