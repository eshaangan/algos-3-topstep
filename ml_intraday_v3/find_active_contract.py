#!/usr/bin/env python3
"""
Find the active MES contract using TopstepX API.
"""
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# Get credentials from environment
session_token = os.getenv('TOPSTEPX_SESSION_TOKEN')
api_key = os.getenv('TOPSTEPX_PROJECTX_API_KEY')

# Search for MES contracts
url = 'https://api.topstepx.com/api/Contract/search'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {session_token}'
}
body = {
    'searchText': 'MES',
    'live': True
}

print("Searching for active MES contracts...")
response = requests.post(url, headers=headers, json=body)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")

if response.status_code == 200:
    data = response.json()
    if data.get('success'):
        contracts = data.get('contracts', [])
        active_contracts = [c for c in contracts if c.get('activeContract')]

        print(f"\n✅ Found {len(active_contracts)} active MES contracts:\n")
        for contract in active_contracts:
            print(f"  ID: {contract['id']}")
            print(f"  Name: {contract['name']}")
            print(f"  Description: {contract['description']}")
            print(f"  Active: {contract['activeContract']}")
            print()

        # Find the front month (earliest expiration that's active)
        if active_contracts:
            # Sort by name to get the earliest month
            sorted_contracts = sorted(active_contracts, key=lambda x: x['name'])
            front_month = sorted_contracts[0]
            print(f"🎯 FRONT MONTH CONTRACT: {front_month['id']}")
            print(f"   Use this contract: {front_month['id']}")
    else:
        print(f"❌ API Error: {data.get('errorMessage')}")
else:
    print(f"❌ HTTP Error: {response.status_code}")
