#!/usr/bin/env python3
"""
Fetch Jan 1 - Feb 10, 2026 MES data from Databento.

This creates a clearly marked out-of-sample test dataset for
validating the top 100 exhaustive search models.

Output:
- data/processed/jan_feb_2026_oos_test.h5 (5-minute bars)
- data/processed/jan_feb_2026_oos_test_1m.h5 (1-minute bars, optional)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from ml_intraday_v3.live_trading.data_fetcher import LiveDataFetcher


def main():
    logger.info("="*80)
    logger.info("FETCHING JAN 1 - FEB 10, 2026 OUT-OF-SAMPLE TEST DATA")
    logger.info("="*80)

    # Create fetcher
    fetcher = LiveDataFetcher(
        symbol="MES.c.0",  # MES front month continuous
        bar_size="1m",
        lookback_bars=100,
    )

    # Fetch extended period
    start_date = "2026-01-01"
    end_date = "2026-02-10"

    logger.info(f"\nFetching: {start_date} to {end_date}")
    logger.info("  Symbol: MES.c.0 (front month continuous)")
    logger.info("  Bar size: 1-minute")

    try:
        bars_1m = fetcher.fetch_historical(start_date, end_date)
        logger.info(f"\n✓ Fetched {len(bars_1m):,} 1-minute bars")
        logger.info(f"  Date range: {bars_1m.index[0]} to {bars_1m.index[-1]}")

        if len(bars_1m) == 0:
            logger.error("ERROR: No data fetched!")
            logger.error("Check:")
            logger.error("  1. DATABENTO_API_KEY is set in .env")
            logger.error("  2. Databento subscription includes MES")
            logger.error("  3. Date range is valid")
            sys.exit(1)

        # Resample to 5-minute bars
        logger.info("\nResampling to 5-minute bars...")
        bars_5m = bars_1m.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        logger.info(f"✓ Resampled to {len(bars_5m):,} 5-minute bars")

        # Save datasets
        output_dir = project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save 5-minute bars (primary test data)
        output_5m = output_dir / "jan_feb_2026_oos_test.h5"
        bars_5m.to_hdf(output_5m, key='bars_5min', mode='w')
        logger.info(f"\n✓ Saved 5-minute bars: {output_5m}")
        logger.info(f"  Size: {output_5m.stat().st_size / 1024:.1f} KB")

        # Save 1-minute bars (for reference)
        output_1m = output_dir / "jan_feb_2026_oos_test_1m.h5"
        bars_1m.to_hdf(output_1m, key='bars_1min', mode='w')
        logger.info(f"✓ Saved 1-minute bars: {output_1m}")
        logger.info(f"  Size: {output_1m.stat().st_size / 1024:.1f} KB")

        # Save metadata
        metadata = {
            'fetch_date': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'symbol': 'MES.c.0',
            'bars_1m_count': len(bars_1m),
            'bars_5m_count': len(bars_5m),
            'actual_start': str(bars_1m.index[0]),
            'actual_end': str(bars_1m.index[-1]),
            'purpose': 'Out-of-sample test for top 100 exhaustive search models',
        }

        import json
        metadata_path = output_dir / "jan_feb_2026_oos_test_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Saved metadata: {metadata_path}")

        # Summary statistics
        logger.info("\n" + "="*80)
        logger.info("DATASET SUMMARY")
        logger.info("="*80)
        logger.info(f"Period:        {start_date} to {end_date}")
        logger.info(f"Actual range:  {bars_1m.index[0].date()} to {bars_1m.index[-1].date()}")
        logger.info(f"Trading days:  ~{len(bars_1m) / (6.5 * 60):.0f} days (assuming 6.5h RTH)")
        logger.info(f"1-min bars:    {len(bars_1m):,}")
        logger.info(f"5-min bars:    {len(bars_5m):,}")
        logger.info(f"\nFiles created:")
        logger.info(f"  {output_5m.name}")
        logger.info(f"  {output_1m.name}")
        logger.info(f"  {metadata_path.name}")

        logger.info("\n✅ Data fetch complete!")
        logger.info("\nNext step:")
        logger.info("  python ml_intraday_v3/experiments/batch_validate_top100_jan_feb_2026.py")

    except Exception as e:
        logger.error(f"\n❌ Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
