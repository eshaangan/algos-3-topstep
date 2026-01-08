"""
Test retrieve_bars with different parameters
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from core.projectx_client import ProjectXClient

def main():
    print("=" * 80)
    print("TESTING RETRIEVE_BARS WITH DIFFERENT PARAMETERS")
    print("=" * 80)

    client = ProjectXClient()
    print(f"Account: {client._account_id}")
    print(f"Contract: {client._contract_id}")

    # Try different date ranges and live parameter combinations
    test_cases = [
        {
            "name": "Recent data (last 24h) with live=True",
            "start": datetime.now(timezone.utc) - timedelta(hours=24),
            "end": datetime.now(timezone.utc),
            "live": True,
        },
        {
            "name": "Recent data (last 24h) with live=False",
            "start": datetime.now(timezone.utc) - timedelta(hours=24),
            "end": datetime.now(timezone.utc),
            "live": False,
        },
        {
            "name": "Older data (5-4 days ago) with live=False",
            "start": datetime.now(timezone.utc) - timedelta(days=5),
            "end": datetime.now(timezone.utc) - timedelta(days=4),
            "live": False,
        },
        {
            "name": "Older data (7-6 days ago) with live=False",
            "start": datetime.now(timezone.utc) - timedelta(days=7),
            "end": datetime.now(timezone.utc) - timedelta(days=6),
            "live": False,
        },
        {
            "name": "Yesterday's data with live=False",
            "start": datetime.now(timezone.utc) - timedelta(days=1),
            "end": datetime.now(timezone.utc) - timedelta(hours=12),
            "live": False,
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   Start: {test['start'].isoformat()}")
        print(f"   End: {test['end'].isoformat()}")
        print(f"   Live: {test['live']}")

        try:
            bars = client.retrieve_bars(
                start_time=test['start'],
                end_time=test['end'],
                unit=2,  # minutes
                unit_number=5,
                limit=100,
                include_partial_bar=False,
                live=test['live'],
            )

            print(f"   ✓ Success! Retrieved {len(bars)} bars")
            if bars:
                print(f"     First: {bars[0].timestamp.isoformat()}")
                print(f"     Last: {bars[-1].timestamp.isoformat()}")
                print(f"     Sample bar: O={bars[0].open} H={bars[0].high} L={bars[0].low} C={bars[0].close} V={bars[0].volume}")
                break  # Found working parameters!

        except Exception as e:
            print(f"   ✗ Failed: {e}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
