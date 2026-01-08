#!/usr/bin/env python3
"""
Check Topstep account status and trading permissions
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import pytz

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from core.projectx_client import ProjectXClient


def check_account_status():
    """Check account status and permissions"""

    print("\n" + "="*60)
    print("TOPSTEP ACCOUNT STATUS CHECK")
    print("="*60)

    account_id = os.getenv('TOPSTEPX_ACCOUNT_ID')
    contract_id = os.getenv('TOPSTEPX_CONTRACT_ID')

    print(f"\nEnvironment Config:")
    print(f"  Account ID: {account_id}")
    print(f"  Contract ID: {contract_id}")

    # Check current time
    chicago_tz = pytz.timezone('America/Chicago')
    now_chicago = datetime.now(chicago_tz)
    print(f"\nCurrent Time (Chicago): {now_chicago.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    hour = now_chicago.hour
    if 8 <= hour < 15:
        print(f"  ✓ RTH SESSION (08:30-15:00 CT)")
    elif 17 <= hour or hour < 8:
        print(f"  ⚠️  EXTENDED HOURS (17:00-08:30 CT)")
    else:
        print(f"  ⚠️  CLOSED SESSION (15:00-17:00 CT)")

    # Create client
    client = ProjectXClient()

    # Get account state
    print("\n" + "-"*60)
    print("Account State:")
    print("-"*60)

    try:
        account = client.get_account_state()
        print(f"  Account ID: {account.account_id}")
        print(f"  Balance: ${account.balance:,.2f}")
        print(f"  Equity: ${account.equity:,.2f}")
        print(f"  Open P&L: ${account.open_pnl:,.2f}")
        print(f"  Realized P&L: ${account.realized_pnl:,.2f}")
        print(f"\n  ✓ Account API access working")
    except Exception as e:
        print(f"  ✗ Failed to get account state: {e}")
        return

    # Try to search for available contracts
    print("\n" + "-"*60)
    print("Searching for MES contracts:")
    print("-"*60)

    try:
        # This is a manual API call since it's not in the client
        import requests

        headers = {
            'accept': 'text/plain',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("TOPSTEPX_SESSION_TOKEN")}'
        }

        response = requests.post(
            f"{os.getenv('TOPSTEPX_PROJECTX_BASE_URL')}/api/Contract/search",
            headers=headers,
            json={
                "searchText": "MES",
                "live": True  # Search live contracts
            },
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                contracts = data.get('contracts', [])
                print(f"  Found {len(contracts)} MES contracts:")
                for contract in contracts[:5]:  # Show first 5
                    active = "✓" if contract.get('activeContract') else "✗"
                    print(f"    {active} {contract['id']}: {contract['description']}")

                # Check if our contract is in the list
                our_contract = next((c for c in contracts if c['id'] == contract_id), None)
                if our_contract:
                    print(f"\n  ✓ Your contract ({contract_id}) is ACTIVE")
                else:
                    print(f"\n  ⚠️  Your contract ({contract_id}) not found in active list")
            else:
                print(f"  ✗ Search failed: {data.get('errorMessage')}")
        else:
            print(f"  ✗ API error: {response.status_code}")

    except Exception as e:
        print(f"  ⚠️  Could not search contracts: {e}")

    # Try to get account details via search
    print("\n" + "-"*60)
    print("Account Trading Permissions:")
    print("-"*60)

    try:
        import requests

        headers = {
            'accept': 'text/plain',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("TOPSTEPX_SESSION_TOKEN")}'
        }

        response = requests.post(
            f"{os.getenv('TOPSTEPX_PROJECTX_BASE_URL')}/api/Account/search",
            headers=headers,
            json={
                "onlyActiveAccounts": True
            },
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                accounts = data.get('accounts', [])
                print(f"  Found {len(accounts)} active account(s):")
                for acc in accounts:
                    active_marker = "✓" if acc['id'] == int(account_id) else " "
                    can_trade = "✓ CAN TRADE" if acc.get('canTrade') else "✗ CANNOT TRADE"
                    visible = "✓ VISIBLE" if acc.get('isVisible') else "✗ HIDDEN"
                    print(f"  {active_marker} Account {acc['id']}: {acc['name']}")
                    print(f"      {can_trade}, {visible}")

                # Check our account
                our_account = next((a for a in accounts if a['id'] == int(account_id)), None)
                if our_account:
                    if our_account.get('canTrade'):
                        print(f"\n  ✓ Your account ({account_id}) CAN TRADE")
                    else:
                        print(f"\n  ✗ Your account ({account_id}) CANNOT TRADE")
                        print(f"      This may be why orders are rejected.")
                else:
                    print(f"\n  ⚠️  Your account ({account_id}) not found in active accounts")
            else:
                print(f"  ✗ Search failed: {data.get('errorMessage')}")
        else:
            print(f"  ✗ API error: {response.status_code}")

    except Exception as e:
        print(f"  ⚠️  Could not check account permissions: {e}")

    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    check_account_status()
