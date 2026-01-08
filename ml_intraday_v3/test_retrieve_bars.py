"""
Quick test to debug retrieve_bars API call
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from core.projectx_client import ProjectXClient

def main():
    print("=" * 80)
    print("RETRIEVE_BARS API DEBUG TEST")
    print("=" * 80)

    # Check environment variables
    print("\n1. Checking environment variables...")
    contract_id = os.getenv("TOPSTEPX_CONTRACT_ID")
    account_id = os.getenv("TOPSTEPX_ACCOUNT_ID")
    print(f"   TOPSTEPX_CONTRACT_ID: '{contract_id}' (type: {type(contract_id).__name__}, len: {len(contract_id) if contract_id else 0})")
    print(f"   TOPSTEPX_ACCOUNT_ID: '{account_id}'")

    # Check for any non-printable characters
    if contract_id:
        print(f"   Contract ID bytes: {contract_id.encode('utf-8')}")
        print(f"   Contract ID repr: {repr(contract_id)}")

    # Initialize client
    print("\n2. Initializing ProjectXClient...")
    try:
        client = ProjectXClient()
        print(f"   ✓ Client initialized")
        print(f"   Internal _contract_id: '{client._contract_id}'")
        print(f"   Internal _account_id: {client._account_id}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return

    # Test account connection
    print("\n3. Testing account connection...")
    try:
        account = client.get_account_state()
        print(f"   ✓ Account {account.account_id}: ${account.equity:,.2f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test retrieve_bars WITHOUT contract_id parameter
    print("\n4. Testing retrieve_bars WITHOUT explicit contract_id...")
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=24)

        print(f"   Start: {start_time.isoformat()}")
        print(f"   End: {end_time.isoformat()}")
        print(f"   Unit: 2 (minutes)")
        print(f"   Unit Number: 5")
        print(f"   Limit: 100")

        bars = client.retrieve_bars(
            start_time=start_time,
            end_time=end_time,
            unit=2,
            unit_number=5,
            limit=100,
            include_partial_bar=False,
        )

        print(f"   ✓ Success! Retrieved {len(bars)} bars")
        if bars:
            print(f"   First bar: {bars[0].timestamp}")
            print(f"   Last bar: {bars[-1].timestamp}")

    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test retrieve_bars WITH explicit contract_id
    print("\n5. Testing retrieve_bars WITH explicit contract_id...")
    try:
        bars = client.retrieve_bars(
            contract_id=contract_id,
            start_time=start_time,
            end_time=end_time,
            unit=2,
            unit_number=5,
            limit=100,
            include_partial_bar=False,
        )

        print(f"   ✓ Success! Retrieved {len(bars)} bars")
        if bars:
            print(f"   First bar: {bars[0].timestamp}")
            print(f"   Last bar: {bars[-1].timestamp}")

    except Exception as e:
        print(f"   ✗ Failed: {e}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
