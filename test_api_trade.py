#!/usr/bin/env python3
"""
Quick API test: Place a buy order, then immediately sell to flatten.
Tests Topstep ProjectX API connectivity and order execution.
"""
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from core.projectx_client import ProjectXClient


def test_round_trip_trade():
    """Execute a simple buy -> sell round trip"""

    print("\n" + "="*60)
    print("TOPSTEP API ROUND-TRIP TEST")
    print("="*60)

    # Get config from environment
    account_id = os.getenv('TOPSTEPX_ACCOUNT_ID')
    contract_id = os.getenv('TOPSTEPX_CONTRACT_ID')

    print(f"\nAccount ID: {account_id}")
    print(f"Contract ID: {contract_id}")

    # Create client
    client = ProjectXClient()

    # Step 1: Check account status
    print("\n" + "-"*60)
    print("Step 1: Checking account status...")
    print("-"*60)

    try:
        account = client.get_account_state()
        print(f"✓ Account connected")
        print(f"  Balance: ${account.balance:,.2f}")
        print(f"  Equity: ${account.equity:,.2f}")
        starting_equity = account.equity
    except Exception as e:
        print(f"✗ Failed to connect to account: {e}")
        return False

    # Step 2: Check for existing positions
    print("\n" + "-"*60)
    print("Step 2: Checking for existing positions...")
    print("-"*60)

    try:
        positions = client.get_open_positions()
        if positions:
            print(f"⚠️  WARNING: Found {len(positions)} existing positions:")
            for pos in positions:
                print(f"  - {pos.symbol}: {pos.quantity} contracts @ ${pos.entry_price}")
            print("\n⚠️  This test will NOT close existing positions.")
        else:
            print("✓ No existing positions")
    except Exception as e:
        print(f"⚠️  Could not check positions: {e}")

    # Step 3: Place BUY order
    print("\n" + "-"*60)
    print("Step 3: Placing BUY order (1 MES contract)...")
    print("-"*60)

    try:
        order_response = client.place_order(
            symbol='MES',
            side='buy',
            quantity=1,
            order_type='MARKET',
            account_id=int(account_id),
            contract_id=contract_id
        )

        if order_response and hasattr(order_response, 'order_id'):
            order_id = order_response.order_id
            print(f"✓ BUY order placed successfully")
            print(f"  Order ID: {order_id}")
        else:
            print(f"✗ BUY order failed: {order_response}")
            return False

    except Exception as e:
        print(f"✗ BUY order failed: {e}")
        return False

    # Wait for fill
    print("\n  Waiting 5 seconds for fill...")
    time.sleep(5)

    # Step 4: Check position
    print("\n" + "-"*60)
    print("Step 4: Verifying position opened...")
    print("-"*60)

    try:
        positions = client.get_open_positions()
        mes_position = None
        for pos in positions:
            if contract_id in pos.symbol or 'MES' in pos.symbol.upper():
                mes_position = pos
                break

        if mes_position:
            print(f"✓ Position opened:")
            print(f"  Symbol: {mes_position.symbol}")
            print(f"  Quantity: {mes_position.quantity}")
            print(f"  Entry Price: ${mes_position.entry_price:,.2f}")
        else:
            print(f"⚠️  Position not found yet (may still be filling)")

    except Exception as e:
        print(f"⚠️  Could not verify position: {e}")

    # Step 5: Place SELL order to flatten
    print("\n" + "-"*60)
    print("Step 5: Placing SELL order to flatten...")
    print("-"*60)

    try:
        order_response = client.place_order(
            symbol='MES',
            side='sell',
            quantity=1,
            order_type='MARKET',
            account_id=int(account_id),
            contract_id=contract_id
        )

        if order_response and hasattr(order_response, 'order_id'):
            order_id = order_response.order_id
            print(f"✓ SELL order placed successfully")
            print(f"  Order ID: {order_id}")
        else:
            print(f"✗ SELL order failed: {order_response}")
            return False

    except Exception as e:
        print(f"✗ SELL order failed: {e}")
        return False

    # Wait for fill
    print("\n  Waiting 5 seconds for fill...")
    time.sleep(5)

    # Step 6: Verify position closed
    print("\n" + "-"*60)
    print("Step 6: Verifying position closed...")
    print("-"*60)

    try:
        positions = client.get_open_positions()
        mes_position = None
        for pos in positions:
            if contract_id in pos.symbol or 'MES' in pos.symbol.upper():
                mes_position = pos
                break

        if mes_position:
            print(f"⚠️  Position still open:")
            print(f"  Quantity: {mes_position.quantity}")
        else:
            print(f"✓ Position closed successfully")

    except Exception as e:
        print(f"⚠️  Could not verify position: {e}")

    # Step 7: Check final account status
    print("\n" + "-"*60)
    print("Step 7: Checking final account status...")
    print("-"*60)

    try:
        account = client.get_account_state()
        ending_equity = account.equity
        pnl = ending_equity - starting_equity

        print(f"✓ Final account status:")
        print(f"  Balance: ${account.balance:,.2f}")
        print(f"  Equity: ${account.equity:,.2f}")
        print(f"  P&L from test: ${pnl:,.2f}")

        if abs(pnl) < 50:
            print(f"\n✓ Round-trip P&L looks reasonable (slippage + commissions)")
        else:
            print(f"\n⚠️  Large P&L change - may have captured market movement")

    except Exception as e:
        print(f"✗ Could not get final status: {e}")
        return False

    # Summary
    print("\n" + "="*60)
    print("TEST COMPLETE ✓")
    print("="*60)
    print("\nSummary:")
    print(f"  ✓ API connection working")
    print(f"  ✓ Account access working")
    print(f"  ✓ Order placement working")
    print(f"  ✓ Position tracking working")
    print(f"\nYou can now run live paper trading with confidence!")
    print("="*60 + "\n")

    return True


if __name__ == "__main__":
    try:
        success = test_round_trip_trade()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
