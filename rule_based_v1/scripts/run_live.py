"""Live trading entry point for rule-based system.

Usage:
    python scripts/run_live.py --dry-run
    python scripts/run_live.py --contract-id MESZ5
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Rule-based live trading")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Paper trade mode (default: True)")
    parser.add_argument("--live", action="store_true",
                        help="Enable real trading (overrides --dry-run)")
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt (for cron/non-interactive use)")
    parser.add_argument("--contract-id", type=str, help="TopstepX contract ID")
    parser.add_argument("--account-id", type=str, help="TopstepX account ID")
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
        logger.info("Starting in DRY RUN mode (paper trading)")
    else:
        logger.warning("LIVE TRADING MODE - real orders will be placed!")
        if not args.yes:
            confirm = input("Type 'CONFIRM' to proceed with live trading: ")
            if confirm != "CONFIRM":
                logger.info("Live trading cancelled.")
                return

    try:
        from live.runner import LiveRunner

        config_dir = Path(args.config_dir) if args.config_dir else Path(__file__).parent.parent / "configs"
        runner = LiveRunner(
            config_dir=config_dir,
            dry_run=dry_run,
            contract_id=args.contract_id,
            account_id=args.account_id,
        )
        runner.run()
    except ImportError:
        logger.error("Live trading module not yet available. Use --dry-run for paper trading.")
        logger.info("Implement live/runner.py to enable live trading.")
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
