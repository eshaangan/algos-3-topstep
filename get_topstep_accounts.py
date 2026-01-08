#!/usr/bin/env python3
"""
Fetch all Topstep account IDs using the ProjectX API.

This script authenticates with Topstep and retrieves all active accounts
associated with your API credentials.
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

import requests
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

PROJECTX_DEFAULT_BASE_URL = "https://api.topstepx.com"


def authenticate() -> str:
    """
    Authenticate with Topstep API and return a JWT session token.
    
    Returns:
        JWT token string
        
    Raises:
        RuntimeError: If authentication fails
    """
    username = os.getenv("TOPSTEPX_USERNAME")
    api_key = os.getenv("TOPSTEPX_PROJECTX_API_KEY")
    
    if not username or not api_key:
        raise EnvironmentError(
            "TOPSTEPX_USERNAME and TOPSTEPX_PROJECTX_API_KEY must be set in the environment. "
            "Add them to your .env file."
        )
    
    url = f"{PROJECTX_DEFAULT_BASE_URL}/api/Auth/loginKey"
    body = {
        "userName": username,
        "apiKey": api_key,
    }
    
    resp = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/plain",
        },
        json=body,
        timeout=10.0,
    )
    
    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Invalid JSON from loginKey: {exc}") from exc
    
    if not resp.ok or not payload.get("success") or payload.get("errorCode") not in (0, None):
        raise RuntimeError(f"loginKey failed: {payload}")
    
    token = payload.get("token")
    if not token:
        raise RuntimeError("loginKey response missing 'token'.")
    
    return token


def get_all_accounts(token: str, only_active: bool = True) -> List[Dict[str, Any]]:
    """
    Fetch all accounts from Topstep API.
    
    Args:
        token: JWT authentication token
        only_active: If True, only return active accounts
        
    Returns:
        List of account dictionaries with id, name, balance, canTrade, etc.
        
    Raises:
        RuntimeError: If API call fails
    """
    url = f"{PROJECTX_DEFAULT_BASE_URL}/api/Account/search"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    # Try both request formats (the API may accept different formats)
    # Format 1: Direct format (from Context7 docs)
    body1 = {"onlyActiveAccounts": only_active}
    # Format 2: Nested format (from ProjectXClient implementation)
    body2 = {"request": {"onlyActiveAccounts": only_active}}
    
    # Try format 1 first
    resp = requests.post(url, headers=headers, json=body1, timeout=10.0)
    
    # If that fails, try format 2
    if not resp.ok and resp.status_code != 429:
        resp = requests.post(url, headers=headers, json=body2, timeout=10.0)
    
    if resp.status_code == 429:
        raise RuntimeError("ProjectX rate limit exceeded.")
    
    if not resp.ok:
        text = resp.text.strip()
        raise RuntimeError(f"Account search failed ({resp.status_code}): {text[:200]}")
    
    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Invalid JSON from Account/search: {exc}") from exc
    
    if not payload.get("success") or payload.get("errorCode") not in (0, None):
        error_msg = payload.get("errorMessage") or str(payload)
        raise RuntimeError(f"Account search failed: {error_msg}")
    
    accounts = payload.get("accounts", [])
    if not isinstance(accounts, list):
        raise RuntimeError(f"Unexpected accounts payload format: {payload}")
    
    return accounts


def format_account_info(account: Dict[str, Any]) -> str:
    """Format account information for display."""
    account_id = account.get("id", "N/A")
    name = account.get("name", "N/A")
    balance = account.get("balance")
    equity = account.get("equity")
    can_trade = account.get("canTrade", False)
    is_visible = account.get("isVisible", False)
    open_pnl = account.get("openPnl") or account.get("open_pnl")
    realized_pnl = account.get("realizedPnl") or account.get("realized_pnl")
    
    lines = [
        f"  Account ID: {account_id}",
        f"  Name: {name}",
    ]
    
    if balance is not None:
        lines.append(f"  Balance: ${balance:,.2f}")
    if equity is not None:
        lines.append(f"  Equity: ${equity:,.2f}")
    if open_pnl is not None:
        lines.append(f"  Open P&L: ${open_pnl:,.2f}")
    if realized_pnl is not None:
        lines.append(f"  Realized P&L: ${realized_pnl:,.2f}")
    
    lines.append(f"  Can Trade: {'Yes' if can_trade else 'No'}")
    lines.append(f"  Visible: {'Yes' if is_visible else 'No'}")
    
    return "\n".join(lines)


def main():
    """Main function to fetch and display all Topstep accounts."""
    print("\n" + "="*70)
    print("TOPSTEP ACCOUNT ID RETRIEVAL")
    print("="*70)
    
    # Check credentials
    username = os.getenv("TOPSTEPX_USERNAME")
    api_key = os.getenv("TOPSTEPX_PROJECTX_API_KEY")
    
    if not username or not api_key:
        print("\nERROR: Missing credentials")
        print("Please set the following environment variables:")
        print("  - TOPSTEPX_USERNAME")
        print("  - TOPSTEPX_PROJECTX_API_KEY")
        print("\nYou can add them to a .env file in the project root.")
        sys.exit(1)
    
    print(f"\nAuthenticating as: {username}")
    
    try:
        # Authenticate
        token = authenticate()
        print("✓ Authentication successful")
        
        # Fetch active accounts
        print("\nFetching active accounts...")
        accounts = get_all_accounts(token, only_active=True)
        
        if not accounts:
            print("\nNo active accounts found.")
            return
        
        print(f"\n✓ Found {len(accounts)} active account(s):\n")
        print("-"*70)
        
        for i, account in enumerate(accounts, 1):
            print(f"\nAccount {i}:")
            print(format_account_info(account))
            if i < len(accounts):
                print("-"*70)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\nTotal active accounts: {len(accounts)}")
        
        account_ids = [str(acc.get("id", "N/A")) for acc in accounts]
        print(f"\nAccount IDs: {', '.join(account_ids)}")
        
        # Check if any account is configured in environment
        env_account_id = os.getenv("TOPSTEPX_ACCOUNT_ID")
        if env_account_id:
            if env_account_id in account_ids:
                print(f"\n✓ Your configured account ID ({env_account_id}) is in the active accounts list.")
            else:
                print(f"\n⚠️  Your configured account ID ({env_account_id}) is NOT in the active accounts list.")
                print("   You may need to update TOPSTEPX_ACCOUNT_ID in your .env file.")
        
        # Also try to fetch all accounts (including inactive)
        print("\n" + "-"*70)
        print("Fetching all accounts (including inactive)...")
        try:
            all_accounts = get_all_accounts(token, only_active=False)
            if len(all_accounts) > len(accounts):
                inactive_count = len(all_accounts) - len(accounts)
                print(f"✓ Found {len(all_accounts)} total accounts ({inactive_count} inactive)\n")
                
                # Show inactive accounts
                active_ids = {acc.get("id") for acc in accounts}
                inactive_accounts = [acc for acc in all_accounts if acc.get("id") not in active_ids]
                
                if inactive_accounts:
                    print("Inactive Accounts:")
                    print("-"*70)
                    for i, account in enumerate(inactive_accounts, 1):
                        print(f"\nInactive Account {i}:")
                        print(format_account_info(account))
                        if i < len(inactive_accounts):
                            print("-"*70)
                    
                    inactive_ids = [str(acc.get("id", "N/A")) for acc in inactive_accounts]
                    print(f"\nInactive Account IDs: {', '.join(inactive_ids)}")
            else:
                print(f"✓ All accounts are active")
        except Exception as e:
            print(f"⚠️  Could not fetch inactive accounts: {e}")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

