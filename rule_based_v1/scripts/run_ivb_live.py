"""Live trading entry point for IVB ORB strategy (MES).

Usage:
    python scripts/run_ivb_live.py --dry-run
    python scripts/run_ivb_live.py --live --yes --config-dir rule_based_v1/configs/vm_ivb_mes
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="IVB ORB live trading (MES)")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true", help="Enable real orders (overrides --dry-run)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--contract-id", type=str)
    parser.add_argument("--account-id", type=str)
    parser.add_argument("--config-dir", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    dry_run = not args.live

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    if dry_run:
        logger.info("IVB ORB — DRY RUN mode (paper trading)")
    else:
        logger.warning("IVB ORB — LIVE TRADING MODE — real orders will be placed!")
        if not args.yes:
            confirm = input("Type 'CONFIRM' to proceed with live trading: ")
            if confirm != "CONFIRM":
                logger.info("Live trading cancelled.")
                return

    from live.ivb_runner import IVBLiveRunner

    config_dir = (
        Path(args.config_dir)
        if args.config_dir
        else Path(__file__).parent.parent / "configs" / "vm_ivb_mes"
    )

    runner = IVBLiveRunner(
        config_dir=config_dir,
        dry_run=dry_run,
        contract_id=args.contract_id,
        account_id=args.account_id,
    )
    runner.run()


if __name__ == "__main__":
    main()
