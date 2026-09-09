#!/usr/bin/env python3
"""
Download FREE historical futures data from Yahoo Finance for multi-market strategy.

Markets:
- MES (Micro E-mini S&P 500) - US session
- NKD (Nikkei Dollar futures) - Asian session

Usage:
    python download_free_futures.py --start 2020-01-01 --end 2025-12-31
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse


def download_futures_data(symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    """
    Download historical futures data from Yahoo Finance (FREE).

    Args:
        symbol: Yahoo Finance symbol (e.g., "MES=F", "NKD=F")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval (1d, 1h, 5m, etc.)

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Downloading {symbol} from {start_date} to {end_date}...")

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    if df.empty:
        print(f"⚠️  WARNING: No data returned for {symbol}")
        return None

    print(f"✓ Downloaded {len(df)} bars for {symbol}")
    return df


def save_to_parquet(df: pd.DataFrame, output_path: Path, symbol: str):
    """Save DataFrame to parquet format"""
    df.to_parquet(output_path)
    print(f"✓ Saved to {output_path}")

    # Print summary stats
    print(f"\nSummary for {symbol}:")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Total bars: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Download free futures data from Yahoo Finance")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval (1d, 1h, 5m)")
    parser.add_argument("--output-dir", type=str, default="data/raw_futures", help="Output directory")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FREE FUTURES DATA DOWNLOADER")
    print("Source: Yahoo Finance (yfinance)")
    print("=" * 60)
    print()

    # Markets to download
    markets = {
        "MES=F": "mes",  # Micro E-mini S&P 500
        "NKD=F": "nkd",  # Nikkei Dollar futures
    }

    # Optional: Add more markets
    # "FDAX": "fdax",  # DAX futures (if available on Yahoo)
    # "MNQ=F": "mnq",  # Micro Nasdaq

    for yahoo_symbol, short_name in markets.items():
        try:
            # Download data
            df = download_futures_data(yahoo_symbol, args.start, args.end, args.interval)

            if df is not None and not df.empty:
                # Save to parquet
                output_path = output_dir / f"{short_name}_daily_{args.start}_{args.end}.parquet"
                save_to_parquet(df, output_path, short_name.upper())

        except Exception as e:
            print(f"❌ Error downloading {yahoo_symbol}: {e}")
            continue

    print("=" * 60)
    print("✓ Download complete!")
    print(f"Files saved to: {output_dir.absolute()}")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Verify data quality (check for gaps)")
    print("2. Calculate correlations between MES and NKD")
    print("3. Adapt ml_intraday_v3 pipeline for multi-market training")
    print("4. Backtest multi-market strategy")


if __name__ == "__main__":
    main()
