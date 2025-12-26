"""
Detailed analysis of GLBX CSV data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ml_intraday_v3.data.ingest import load_raw_data

print("="*70)
print("GLBX CSV Data Analysis")
print("="*70)

csv_path = Path("data/raw/GLBX-20251220-LWFB9HCEL5/glbx-mdp3-20100606-20251219.ohlcv-1m.csv")

# First, peek at raw CSV
print("\n1. RAW CSV STRUCTURE")
print("-" * 70)
df_raw = pd.read_csv(csv_path, nrows=10000)
print(f"Total rows sampled: {len(df_raw):,}")
print(f"Columns: {list(df_raw.columns)}")
print(f"\nColumn types:")
print(df_raw.dtypes)

# Check for multiple symbols
print(f"\n2. SYMBOL/CONTRACT ANALYSIS")
print("-" * 70)
if 'symbol' in df_raw.columns:
    unique_symbols = df_raw['symbol'].unique()
    print(f"Number of unique symbols in first 10k rows: {len(unique_symbols)}")
    print(f"Sample symbols: {sorted(unique_symbols)[:10]}")

    symbol_counts = df_raw['symbol'].value_counts()
    print(f"\nTop 10 symbols by frequency:")
    print(symbol_counts.head(10))

if 'instrument_id' in df_raw.columns:
    print(f"\nUnique instrument IDs: {df_raw['instrument_id'].nunique()}")

# Check timestamp format
print(f"\n3. TIMESTAMP ANALYSIS")
print("-" * 70)
print(f"Sample timestamps (first 5):")
print(df_raw['ts_event'].head())
print(f"\nTimestamp format: ISO 8601 with nanoseconds")

# Check for data quality issues
print(f"\n4. DATA QUALITY CHECK")
print("-" * 70)

# Price ranges
print(f"Open price range: ${df_raw['open'].min():.2f} to ${df_raw['open'].max():.2f}")
print(f"High price range: ${df_raw['high'].min():.2f} to ${df_raw['high'].max():.2f}")
print(f"Low price range: ${df_raw['low'].min():.2f} to ${df_raw['low'].max():.2f}")
print(f"Close price range: ${df_raw['close'].min():.2f} to ${df_raw['close'].max():.2f}")

# Check for negative prices
negative_prices = df_raw[
    (df_raw['open'] < 0) | (df_raw['high'] < 0) |
    (df_raw['low'] < 0) | (df_raw['close'] < 0)
]
if len(negative_prices) > 0:
    print(f"\n⚠️  WARNING: Found {len(negative_prices)} rows with negative prices!")
    print(f"Sample negative price rows:")
    print(negative_prices[['ts_event', 'symbol', 'open', 'high', 'low', 'close']].head())

# Volume analysis
print(f"\nVolume range: {df_raw['volume'].min():,} to {df_raw['volume'].max():,}")
print(f"Volume mean: {df_raw['volume'].mean():.2f}")
print(f"Zero volume bars: {(df_raw['volume'] == 0).sum()} ({(df_raw['volume'] == 0).sum() / len(df_raw) * 100:.2f}%)")

# Check for OHLC validity
invalid_ohlc = df_raw[
    (df_raw['low'] > df_raw['high']) |
    (df_raw['open'] > df_raw['high']) | (df_raw['open'] < df_raw['low']) |
    (df_raw['close'] > df_raw['high']) | (df_raw['close'] < df_raw['low'])
]
print(f"\nInvalid OHLC bars (L>H or O/C outside H/L): {len(invalid_ohlc)} ({len(invalid_ohlc) / len(df_raw) * 100:.2f}%)")

# Now test loading with ingestion module
print(f"\n5. LOADING WITH INGESTION MODULE")
print("-" * 70)

try:
    df = load_raw_data(
        input_path=csv_path,
        input_format="csv",
        timestamp_column="ts_event",
        required_columns=["open", "high", "low", "close", "volume"]
    )

    print(f"✓ Successfully loaded with ingestion module")
    print(f"  Total rows: {len(df):,}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Duration: {df.index[-1] - df.index[0]}")
    print(f"  Columns: {list(df.columns)}")

    # Check if symbol column is preserved
    if 'symbol' in df.columns:
        print(f"\n  ✓ Symbol column preserved")
        print(f"  Unique symbols in full dataset: {df['symbol'].nunique()}")
        print(f"\n  Symbol distribution:")
        symbol_dist = df['symbol'].value_counts().head(20)
        for sym, count in symbol_dist.items():
            print(f"    {sym}: {count:,} bars ({count/len(df)*100:.2f}%)")

    # Duplicate analysis
    n_dups = df.index.duplicated().sum()
    if n_dups > 0:
        print(f"\n  ⚠️  WARNING: {n_dups:,} duplicate timestamps found")
        print(f"     This is expected for multi-contract data (same time, different symbols)")
        print(f"     You may need to:")
        print(f"     1. Filter to a specific contract (e.g., symbol.startswith('ES'))")
        print(f"     2. Use continuization to create a continuous contract")

except Exception as e:
    print(f"✗ Failed to load: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print("""
This appears to be RAW multi-contract futures data. You likely need to:

1. FILTER TO SPECIFIC CONTRACTS
   - Data contains multiple ES contracts (ESM0, ESU0, ESZ0, etc.)
   - Each contract trades concurrently, causing duplicate timestamps

2. BUILD CONTINUOUS CONTRACT
   - Use the continuization module to stitch contracts together
   - Define roll schedule (front month, volume-based, etc.)

3. HANDLE DATA QUALITY ISSUES
   - Check for negative prices (may be spread contracts)
   - Validate OHLC relationships
   - Handle zero-volume bars

4. NEXT STEPS
   - Examine unique symbols: df['symbol'].unique()
   - Filter to ES contracts: df[df['symbol'].str.startswith('ES')]
   - Set up continuization configuration
""")
