#!/usr/bin/env python3
"""
Download NKD 1-minute intraday data from Databento
Matches MES data characteristics for multi-market training
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import databento as db
from dotenv import load_dotenv

# Load environment
load_dotenv()

def download_nkd_data(
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    output_dir: str = "data/raw",
    api_key: str = None
):
    """
    Download NKD 1-minute OHLCV data from Databento

    Args:
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        output_dir: Where to save parquet file
        api_key: Databento API key (or from env)
    """

    # Get API key
    if api_key is None:
        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            raise ValueError("DATABENTO_API_KEY not found in environment")

    print(f"Downloading NKD data from {start_date} to {end_date}")
    print(f"API Key: {api_key[:10]}...")

    # Initialize client
    client = db.Historical(api_key)

    # NKD is traded on CME (Nikkei 225 Micro futures)
    # Symbol format in Databento: NKD.n.0
    # For continuous contract, we use the front month

    print("\nFetching NKD data...")
    try:
        # Download OHLCV-1m data
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",  # CME Globex
            symbols=["NKD.n.0"],  # Front month continuous
            schema="ohlcv-1m",
            start=start_date,
            end=end_date,
            stype_in="continuous",  # Use continuous contract
        )

        # Convert to DataFrame
        df = data.to_df()

        print(f"✓ Downloaded {len(df)} bars")
        print(f"\nDate range: {df.index.min()} to {df.index.max()}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())

        # Save to parquet
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f"nkd_1m_{start_date}_{end_date}.parquet"
        df.to_parquet(output_file)

        print(f"\n✓ Saved to: {output_file}")
        print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        # Basic stats
        print(f"\nBasic Statistics:")
        print(f"  Total bars: {len(df)}")
        print(f"  Trading days: ~{len(df) / 390:.0f}")  # Approx 390 bars/day
        print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"  Avg volume: {df['volume'].mean():.0f}")

        return df

    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        print(f"\nTrying alternative symbol format...")

        # Try alternative symbol format
        try:
            data = client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=["NKD"],  # Without suffix
                schema="ohlcv-1m",
                start=start_date,
                end=end_date,
            )
            df = data.to_df()

            print(f"✓ Downloaded {len(df)} bars with alternative format")

            # Save
            output_file = output_path / f"nkd_1m_{start_date}_{end_date}.parquet"
            df.to_parquet(output_file)
            print(f"✓ Saved to: {output_file}")

            return df

        except Exception as e2:
            print(f"✗ Alternative format also failed: {e2}")
            print("\nAvailable NKD symbols:")

            # List available symbols
            try:
                symbols = client.metadata.list_symbols(
                    dataset="GLBX.MDP3",
                    start=start_date
                )
                nkd_symbols = [s for s in symbols if 'NKD' in s]
                print(nkd_symbols[:20])  # First 20
            except:
                pass

            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download NKD intraday data")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--api-key", default=None, help="Databento API key (or use env)")

    args = parser.parse_args()

    try:
        df = download_nkd_data(
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output_dir,
            api_key=args.api_key
        )
        print("\n✓ NKD data download complete!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        sys.exit(1)
