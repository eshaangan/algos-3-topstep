"""
List all TopstepX accounts and available MNQ contracts.
Run this once to find your practice account ID, then set it in orb_env.txt.

Usage:
    python ml_intraday_v3/live/discover_accounts.py

Requires env vars: TOPSTEPX_USERNAME, TOPSTEPX_PROJECTX_API_KEY
(loads from .env if present)
"""

import os, sys, json, requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

BASE_URL = os.getenv("TOPSTEPX_PROJECTX_BASE_URL", "https://api.topstepx.com")
USERNAME = os.getenv("TOPSTEPX_USERNAME")
API_KEY  = os.getenv("TOPSTEPX_PROJECTX_API_KEY")

if not USERNAME or not API_KEY:
    print("ERROR: Set TOPSTEPX_USERNAME and TOPSTEPX_PROJECTX_API_KEY in your environment or .env")
    sys.exit(1)

def auth() -> str:
    resp = requests.post(
        f"{BASE_URL}/api/Auth/loginKey",
        json={"userName": USERNAME, "apiKey": API_KEY},
        timeout=10,
    )
    resp.raise_for_status()
    payload = resp.json()
    token = payload.get("token") or payload.get("data", {}).get("token")
    if not token:
        raise RuntimeError(f"No token in auth response: {payload}")
    return token

def post(token: str, path: str, body: dict) -> dict:
    resp = requests.post(
        f"{BASE_URL}{path}",
        json=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()

print(f"Authenticating as {USERNAME}...")
token = auth()
print("OK\n")

# --- Accounts ---
payload = post(token, "/api/Account/search", {"request": {"onlyActiveAccounts": True}})
accounts = payload.get("accounts") or payload.get("data") or []
if not accounts:
    # Try alternate payload structure
    accounts = payload if isinstance(payload, list) else []

print("=" * 60)
print("ACCOUNTS")
print("=" * 60)
for a in accounts:
    aid    = a.get("id")
    name   = a.get("name") or a.get("accountName") or ""
    bal    = a.get("balance") or a.get("equity") or 0
    status = a.get("status") or a.get("accountType") or ""
    active = a.get("active") if "active" in a else a.get("isActive", True)
    print(f"  ID={aid:<12}  balance=${bal:>12,.2f}  active={active}  name={name!r}  status={status}")
print()

# --- Contracts ---
try:
    payload2 = post(token, "/api/Contract/available", {"live": True})
    contracts = payload2.get("contracts") or payload2.get("data") or []
    if not contracts and isinstance(payload2, list):
        contracts = payload2
    mnq = [c for c in contracts if "MNQ" in str(c.get("id","")) or "MNQ" in str(c.get("name",""))]

    print("=" * 60)
    print("MNQ CONTRACTS")
    print("=" * 60)
    for c in mnq[:10]:
        cid  = c.get("id") or c.get("contractId")
        name = c.get("name") or c.get("description") or ""
        print(f"  ID={cid!r:<30}  name={name!r}")
    print()
except Exception as e:
    print(f"Contract search failed ({e}) — current front month is likely CON.F.US.MNQ.M26\n")

print("=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("Practice account ID : 15266746  (PRAC-V2, balance $151,291.99)")
print("Set in /home/eshaanganguly/orb_env.txt on the GCP VM:")
print("  TOPSTEPX_ACCOUNT_ID=15266746")
print("  TOPSTEPX_CONTRACT_ID=CON.F.US.MNQ.M26   # roll to M26 or next front month")
print("  ML_N_CONTRACTS=15")
