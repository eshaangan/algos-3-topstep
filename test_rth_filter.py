"""
Quick test to verify RTH filtering works correctly.
"""

import sys
from pathlib import Path
import pandas as pd
import pytz

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_intraday_v3.live_trading.topstepx_rest_data_fetcher import TopstepXRestDataFetcher

def test_rth_filter():
    """Test RTH filtering logic."""
    print("=" * 80)
    print("RTH FILTERING TEST")
    print("=" * 80)
    print()

    # Create a mock fetcher object with just the filter method
    class MockFetcher:
        def __init__(self):
            self.enable_rth_filter = True
            self.chicago_tz = pytz.timezone('America/Chicago')

        def filter_rth_bars(self, df: pd.DataFrame) -> pd.DataFrame:
            """Filter to RTH (copied from TopstepXRestDataFetcher)."""
            if df.empty or not self.enable_rth_filter:
                return df

            df_ct = df.copy()
            if df_ct.index.tz != self.chicago_tz:
                if df_ct.index.tz is not None:
                    df_ct.index = df_ct.index.tz_convert(self.chicago_tz)

            rth_mask = (df_ct.index.hour > 8) | ((df_ct.index.hour == 8) & (df_ct.index.minute >= 30))
            rth_mask &= (df_ct.index.hour < 15)

            df_rth = df[rth_mask]
            return df_rth

    fetcher = MockFetcher()
    print(f"✓ Created mock fetcher with RTH filter: {fetcher.enable_rth_filter}")
    print()

    # Create test data with mixed RTH and pre-market bars
    chicago_tz = pytz.timezone('America/Chicago')

    test_times = [
        "2026-01-10 06:00",  # Pre-market
        "2026-01-10 07:30",  # Pre-market
        "2026-01-10 08:00",  # Pre-market
        "2026-01-10 08:30",  # RTH START
        "2026-01-10 09:00",  # RTH
        "2026-01-10 12:00",  # RTH
        "2026-01-10 14:55",  # RTH
        "2026-01-10 15:00",  # Post-market (RTH END at 3 PM)
        "2026-01-10 16:00",  # Post-market
    ]

    data = []
    for time_str in test_times:
        ts = pd.Timestamp(time_str, tz=chicago_tz)
        data.append({
            'timestamp': ts,
            'open': 6800.0,
            'high': 6810.0,
            'low': 6790.0,
            'close': 6805.0,
            'volume': 1000
        })

    df = pd.DataFrame(data)
    df = df.set_index('timestamp')

    print(f"Test data: {len(df)} bars")
    print()

    # Filter to RTH
    df_rth = fetcher.filter_rth_bars(df)

    print(f"After RTH filter: {len(df_rth)} bars")
    print()

    print("Filtered bars:")
    for i, (ts, row) in enumerate(df_rth.iterrows()):
        print(f"  {i+1}. {ts.strftime('%Y-%m-%d %H:%M')} (RTH)")

    print()
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    # Expected: 8:30, 9:00, 12:00, 14:55 = 4 bars
    expected_count = 4

    if len(df_rth) == expected_count:
        print(f"✓ PASS: Got {len(df_rth)} RTH bars (expected {expected_count})")

        # Verify times
        expected_times = ["08:30", "09:00", "12:00", "14:55"]
        actual_times = [ts.strftime("%H:%M") for ts in df_rth.index]

        if actual_times == expected_times:
            print(f"✓ PASS: RTH times match expected: {expected_times}")
        else:
            print(f"✗ FAIL: RTH times don't match")
            print(f"  Expected: {expected_times}")
            print(f"  Got:      {actual_times}")
    else:
        print(f"✗ FAIL: Got {len(df_rth)} RTH bars (expected {expected_count})")

    print()

if __name__ == "__main__":
    test_rth_filter()
