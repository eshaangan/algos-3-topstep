#!/usr/bin/env python3
"""
Download REAL December 2025 MES Data from Data Bento

This script downloads actual market data to validate model performance
on the first out-of-sample month after training ended (Nov 2025).

December 2025 is expected to show BETTER performance than January 2026
because it's a "normal" month without regime shifts.

Output:
- mes_dec2025_1m.parquet: 1-minute bars
- mes_dec2025_5m.parquet: 5-minute bars (resampled)
- CSV versions for inspection
"""

import sys
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime

# Add paths - go up to project root where live_trading/ is
project_root = Path(__file__).resolve().parents[2]  # Up to ml_intraday_v3/
ml_v3_dir = Path(__file__).resolve().parents[1]     # ml_intraday_v3/ml_intraday_v3/
sys.path.insert(0, str(project_root))

from live_trading.data_fetcher import LiveDataFetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    print("="*80)
    print("DOWNLOADING DECEMBER 2025 MES DATA FROM DATA BENTO")
    print("="*80)
    print()

    # Initialize data fetcher
    logger.info("Initializing Data Bento fetcher...")
    fetcher = LiveDataFetcher(
        symbol="MES.c.0",  # Continuous front month E-mini S&P 500
        bar_size="1m",
        lookback_bars=100,  # Not used for historical fetch
    )

    # Download December 2025 (1-minute bars)
    logger.info("Downloading December 2025 1-minute bars...")
    logger.info("  Start: 2025-12-01")
    logger.info("  End: 2025-12-31")

    try:
        bars_1m = fetcher.fetch_historical(
            start_date="2025-12-01",
            end_date="2025-12-31"
        )

        logger.info(f"✅ Downloaded {len(bars_1m):,} 1-minute bars")
        logger.info(f"   Date range: {bars_1m.index[0]} to {bars_1m.index[-1]}")

    except Exception as e:
        logger.error(f"❌ Error downloading data: {e}")
        return 1

    # Resample to 5-minute bars
    logger.info("\nResampling to 5-minute bars...")

    bars_5m = bars_1m.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()

    logger.info(f"✅ Resampled to {len(bars_5m):,} 5-minute bars")

    # Create output directory
    output_dir = ml_v3_dir / "data" / "dec2025_mes"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving to {output_dir}...")

    # Save 1-minute data
    bars_1m.to_parquet(output_dir / "mes_dec2025_1m.parquet")
    bars_1m.to_csv(output_dir / "mes_dec2025_1m.csv")
    logger.info(f"  ✅ Saved 1-minute data ({len(bars_1m):,} bars)")

    # Save 5-minute data
    bars_5m.to_parquet(output_dir / "mes_dec2025_5m.parquet")
    bars_5m.to_csv(output_dir / "mes_dec2025_5m.csv")
    logger.info(f"  ✅ Saved 5-minute data ({len(bars_5m):,} bars)")

    # Save metadata
    metadata = {
        'downloaded_at': datetime.now().isoformat(),
        'symbol': 'MES.c.0',
        'source': 'Data Bento (GLBX.MDP3)',
        'period': 'December 2025',
        'start_date': '2025-12-01',
        'end_date': '2025-12-31',
        'bars_1m': len(bars_1m),
        'bars_5m': len(bars_5m),
        'first_timestamp': str(bars_1m.index[0]),
        'last_timestamp': str(bars_1m.index[-1]),
    }

    import json
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  ✅ Saved metadata")

    # Print summary
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print(f"\nDecember 2025 MES Data:")
    print(f"  1-minute bars: {len(bars_1m):,}")
    print(f"  5-minute bars: {len(bars_5m):,}")
    print(f"  Date range: {bars_1m.index[0].date()} to {bars_1m.index[-1].date()}")
    print(f"  Trading days: {len(bars_5m.index.date.unique())}")
    print(f"\nFiles saved to: {output_dir}")
    print()

    # Basic statistics
    print("Price Statistics (5-minute bars):")
    print(f"  High: ${bars_5m['high'].max():.2f}")
    print(f"  Low: ${bars_5m['low'].min():.2f}")
    print(f"  Range: ${bars_5m['high'].max() - bars_5m['low'].min():.2f}")
    print(f"  Avg Volume: {bars_5m['volume'].mean():,.0f}")
    print()

    print("="*80)
    print("READY FOR MODEL TESTING")
    print("="*80)
    print("\nNext step:")
    print("  python ml_intraday_v3/experiments/test_model_on_real_dec2025.py")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
