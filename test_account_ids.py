#!/usr/bin/env python3
"""
Test script to verify which Topstep account ID is active
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

from core.projectx_client import ProjectXClient


def test_account_id(account_id: str, label: str):
    """Test if an account ID is active"""
    print(f"\n{'='*60}")
    print(f"Testing {label}: {account_id}")
    print(f"{'='*60}")

    # Set the account ID in environment
    os.environ['TOPSTEPX_ACCOUNT_ID'] = account_id

    try:
        # Create client
        client = ProjectXClient()

        # Try to get account state
        account = client.get_account_state()

        # Success!
        print(f"✓ {label} ACTIVE")
        print(f"  Account ID: {account.account_id}")
        print(f"  Balance: ${account.balance:,.2f}")
        print(f"  Equity: ${account.equity:,.2f}")
        print(f"  Open P&L: ${account.open_pnl:,.2f}")
        print(f"  Realized P&L: ${account.realized_pnl:,.2f}")

        return True, account

    except Exception as e:
        # Failed
        print(f"✗ {label} FAILED")
        print(f"  Error: {str(e)}")
        return False, None


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("TOPSTEP ACCOUNT ID VERIFICATION")
    print("="*60)

    # Get credentials from environment
    username = os.getenv('TOPSTEPX_USERNAME')
    api_key = os.getenv('TOPSTEPX_PROJECTX_API_KEY')
    base_url = os.getenv('TOPSTEPX_PROJECTX_BASE_URL')

    print(f"\nCredentials:")
    print(f"  Username: {username}")
    print(f"  API Key: {'*' * 20}{api_key[-10:] if api_key else 'NOT SET'}")
    print(f"  Base URL: {base_url}")

    # Test Account A
    success_a, account_a = test_account_id("15266746", "Account A")

    # Test Account B
    success_b, account_b = test_account_id("15390514", "Account B")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if success_a and success_b:
        print("✓ Both accounts are active!")
        print(f"\nAccount A (15266746):")
        print(f"  Balance: ${account_a.balance:,.2f}")
        print(f"  Equity: ${account_a.equity:,.2f}")
        print(f"\nAccount B (15390514):")
        print(f"  Balance: ${account_b.balance:,.2f}")
        print(f"  Equity: ${account_b.equity:,.2f}")
        print(f"\nRECOMMENDATION: Use the account with higher equity or the one you intend to use for the Topstep combine.")

    elif success_a:
        print("✓ Account A (15266746) is ACTIVE")
        print("✗ Account B (15390514) is INACTIVE")
        print(f"\nRECOMMENDATION: Use Account A (15266746) in .env file")
        print(f"  Update line 6: TOPSTEPX_ACCOUNT_ID=15266746")

    elif success_b:
        print("✗ Account A (15266746) is INACTIVE")
        print("✓ Account B (15390514) is ACTIVE")
        print(f"\nRECOMMENDATION: Use Account B (15390514) in .env file")
        print(f"  Update line 6: TOPSTEPX_ACCOUNT_ID=15390514")

    else:
        print("✗ Both accounts FAILED to connect")
        print("\nPossible issues:")
        print("  1. API credentials are invalid or expired")
        print("  2. Session token is expired (run authentication again)")
        print("  3. Network connectivity issues")
        print("  4. Account IDs are not associated with this API key")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
