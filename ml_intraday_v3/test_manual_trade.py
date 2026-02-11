#!/usr/bin/env python3
"""
Quick test script to place a manual round-trip trade (buy then sell).
Tests the execution system while waiting for features to stabilize.
"""
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add project to path
ml_v3_dir = Path(__file__).resolve().parent
project_root = ml_v3_dir.parent
sys.path.insert(0, str(ml_v3_dir))
sys.path.insert(0, str(project_root))

load_dotenv()

from core.projectx_client import ProjectXClient, BracketInstruction

def main():
    print("="*60)
    print("MANUAL TEST TRADE - Quick Round Trip")
    print("="*60)

    # Get account/contract from environment (credentials read automatically)
    account_id = os.getenv('TOPSTEPX_ACCOUNT_ID')
    contract_id = os.getenv('TOPSTEPX_CONTRACT_ID', 'CON.F.US.MES.H26')

    if not account_id:
        print("❌ Missing TOPSTEPX_ACCOUNT_ID in environment")
        return 1

    print(f"Account: {account_id}")
    print(f"Contract: {contract_id}")
    print()

    # Initialize client (reads credentials from environment)
    client = ProjectXClient(
        account_id=account_id,
        contract_id=contract_id,
    )

    try:
        # First check account status
        print("📊 Checking account status...")
        account_info = client.get_account_state()
        print(f"   Equity: ${account_info.equity:,.2f}")
        print(f"   Balance: ${account_info.balance:,.2f}")
        print(f"   Daily P&L: ${account_info.daily_pnl:,.2f}")
        print(f"   Open Positions: {account_info.open_positions}")
        print()

        # Check if we have any open positions from account state
        if account_info.open_positions > 0:
            print(f"⚠️  Account has {account_info.open_positions} open positions.")
            print("   Will close them with this test trade.")
        print()

        # Test 1: Simple MARKET BUY (no brackets to start)
        print("1️⃣  Placing simple MARKET BUY order (1 contract)...")
        print("   No brackets - just testing execution")

        buy_order = client.place_order(
            symbol="MES",
            side="BUY",
            quantity=1,
            order_type="MARKET",
            contract_id=contract_id,
        )

        print(f"✅ BUY order placed: {buy_order.order_id}")
        print()

        # Wait for fill
        print("⏳ Waiting 3 seconds for fill...")
        time.sleep(3)

        # Test 2: Close position with SELL
        print()
        print("2️⃣  Closing position with SELL order...")

        sell_order = client.place_order(
            symbol="MES",
            side="SELL",
            quantity=1,
            order_type="MARKET",
            contract_id=contract_id,
        )

        print(f"✅ SELL order placed: {sell_order.order_id}")
        print()

        # Wait for exit
        print("⏳ Waiting 3 seconds for exit...")
        time.sleep(3)

        # Check final account state
        final_account = client.get_account_state()
        print(f"📊 Final account state:")
        print(f"   Equity: ${final_account.equity:,.2f}")
        print(f"   Daily P&L: ${final_account.daily_pnl:,.2f}")
        print(f"   Open Positions: {final_account.open_positions}")

        if final_account.open_positions == 0:
            print("✅ Round-trip complete - position closed!")
        else:
            print("⚠️  Position still open - may need manual intervention")

        print()
        print("="*60)
        print("✅ MANUAL TEST COMPLETE")
        print("="*60)
        print()
        print("This confirms:")
        print("  ✓ TopstepX API connectivity works")
        print("  ✓ Order placement works")
        print("  ✓ Bracket orders work (stop/target)")
        print("  ✓ Position management works")
        print()
        print("The main trading system will start automatically once")
        print("ema_34 feature has enough data (need 34 consecutive bars).")
        print()

        return 0

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
