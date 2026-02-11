#!/usr/bin/env python3
"""
Refresh TopstepX Session Token

This script generates a new session token from TopstepX API and updates:
1. Local .env file
2. GCP deployment (optional)

Usage:
    python refresh_topstep_token.py [--deploy]

Options:
    --deploy    Automatically redeploy to GCP after updating token
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import requests
import re
import base64

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
def load_env_file(env_path: Path) -> dict:
    """Load .env file and return as dictionary."""
    env_vars = {}
    if not env_path.exists():
        logger.error(f".env file not found at {env_path}")
        return env_vars

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()

    return env_vars

def save_env_file(env_path: Path, env_vars: dict):
    """Save environment variables to .env file."""
    with open(env_path, 'w') as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    logger.info(f"Updated .env file at {env_path}")

def decode_jwt_payload(token: str) -> dict:
    """Decode JWT token payload to check expiration."""
    try:
        payload = token.split('.')[1]
        # Add padding if needed
        payload += '=' * (4 - len(payload) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(payload))
        return decoded
    except Exception as e:
        logger.warning(f"Could not decode JWT: {e}")
        return {}

def print_token_info(token: str, label: str = "Token"):
    """Print token expiration info."""
    payload = decode_jwt_payload(token)
    if 'exp' in payload:
        exp_timestamp = payload['exp']
        exp_datetime = datetime.fromtimestamp(exp_timestamp)
        now = datetime.now()
        if now < exp_datetime:
            time_remaining = exp_datetime - now
            logger.info(f"{label} expires: {exp_datetime} (in {time_remaining})")
        else:
            logger.warning(f"{label} EXPIRED at {exp_datetime}")
    else:
        logger.info(f"{label} info: Could not determine expiration")

def get_new_session_token(username: str, api_key: str, base_url: str = "https://api.topstepx.com") -> str:
    """
    Generate new session token from TopstepX API.

    Args:
        username: TopstepX username (email)
        api_key: TopstepX API key
        base_url: API base URL

    Returns:
        New JWT session token

    Raises:
        Exception if authentication fails
    """
    url = f"{base_url.rstrip('/')}/api/Auth/loginKey"
    body = {
        "userName": username,
        "apiKey": api_key,
    }

    logger.info(f"Requesting new session token from {url}...")
    logger.info(f"Username: {username}")

    resp = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/plain",
        },
        json=body,
        timeout=10,
    )

    try:
        payload = resp.json()
    except ValueError as e:
        logger.error(f"Invalid JSON response: {resp.text}")
        raise Exception(f"Invalid JSON from loginKey: {e}")

    if not resp.ok or not payload.get("success") or payload.get("errorCode") not in (0, None):
        logger.error(f"Authentication failed: {payload}")
        raise Exception(f"loginKey failed: {payload}")

    token = payload.get("token")
    if not token:
        raise Exception("loginKey response missing 'token'.")

    logger.info("✅ Successfully generated new session token!")
    return token

def main():
    parser = argparse.ArgumentParser(description="Refresh TopstepX session token")
    parser.add_argument("--deploy", action="store_true", help="Redeploy to GCP after updating token")
    parser.add_argument("--env-file", type=str, default="../.env", help="Path to .env file (relative to ml_intraday_v3/)")
    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    env_path = (script_dir / args.env_file).resolve()

    logger.info("=" * 60)
    logger.info("TopstepX Session Token Refresh")
    logger.info("=" * 60)
    logger.info(f"Project root: {project_root}")
    logger.info(f".env file: {env_path}")
    logger.info("")

    # Load current .env
    env_vars = load_env_file(env_path)
    if not env_vars:
        logger.error("Failed to load .env file")
        return 1

    # Check for required variables
    username = env_vars.get("TOPSTEPX_USERNAME")
    api_key = env_vars.get("TOPSTEPX_PROJECTX_API_KEY")
    base_url = env_vars.get("TOPSTEPX_PROJECTX_BASE_URL", "https://api.topstepx.com")
    current_token = env_vars.get("TOPSTEPX_SESSION_TOKEN")

    if not username or not api_key:
        logger.error("Missing required environment variables:")
        if not username:
            logger.error("  - TOPSTEPX_USERNAME")
        if not api_key:
            logger.error("  - TOPSTEPX_PROJECTX_API_KEY")
        return 1

    # Show current token status
    if current_token:
        logger.info("Current token status:")
        print_token_info(current_token, "Current token")
        logger.info("")

    # Generate new token
    try:
        new_token = get_new_session_token(username, api_key, base_url)
    except Exception as e:
        logger.error(f"Failed to generate new token: {e}")
        return 1

    # Show new token expiration
    logger.info("")
    logger.info("New token status:")
    print_token_info(new_token, "New token")
    logger.info("")

    # Update .env file
    env_vars["TOPSTEPX_SESSION_TOKEN"] = new_token
    save_env_file(env_path, env_vars)

    logger.info("✅ Token refresh complete!")
    logger.info("")

    # Optionally deploy to GCP
    if args.deploy:
        logger.info("Deploying to GCP...")
        deploy_script = script_dir / "deploy_to_gcp.sh"
        if deploy_script.exists():
            import subprocess
            result = subprocess.run([str(deploy_script)], cwd=str(script_dir))
            if result.returncode == 0:
                logger.info("✅ GCP deployment complete!")
            else:
                logger.error("❌ GCP deployment failed")
                return 1
        else:
            logger.warning(f"Deploy script not found: {deploy_script}")
            logger.info("To deploy manually, run:")
            logger.info(f"  cd {script_dir}")
            logger.info("  ./deploy_to_gcp.sh")
    else:
        logger.info("To deploy to GCP, run:")
        logger.info(f"  cd {script_dir}")
        logger.info("  ./deploy_to_gcp.sh")
        logger.info("")
        logger.info("Or run this script with --deploy flag:")
        logger.info("  python refresh_topstep_token.py --deploy")

    logger.info("")
    logger.info("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
