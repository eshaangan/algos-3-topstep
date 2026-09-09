#!/usr/bin/env python3
"""
Launcher for the weekend_hold_v1 forward observation.

Loads .env via python-dotenv rather than shell sourcing. `. ./.env` fails on this
repo's .env: line 10 holds an unquoted value containing '=' and the shell tries to
execute it ("command not found: data.topstep.live=true"). dotenv parses it fine.

Sets RAW_DIR to the L2 recorder's output so the runner can read a live mid. The
recorder is a SEPARATE process and must already be running:

    python3 data_collection/record_l2.py --symbol MNQ --out-dir data/l2_raw

Usage:
    python3 rule_based_v1/live/launch_weekend_fwd.py --out-dir logs/weekend_fwd/live
    python3 rule_based_v1/live/launch_weekend_fwd.py --out-dir ... --live

Without --live the runner connects, resolves the contract and logs what it would
do, but submits no orders.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--raw-dir", default=str(ROOT / "data" / "l2_raw"))
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    from dotenv import load_dotenv
    env = ROOT / ".env"
    if not env.exists():
        print(f"FATAL: {env} not found", file=sys.stderr)
        return 2
    load_dotenv(env)

    for k in ("RITHMIC_USERNAME", "RITHMIC_PASSWORD"):
        if not os.environ.get(k):
            print(f"FATAL: {k} missing from {env}", file=sys.stderr)
            return 2

    raw = Path(args.raw_dir)
    if not sorted(raw.glob("stream_MNQ_*.csv")):
        print(f"FATAL: no stream_MNQ_*.csv in {raw}. Start the recorder first:\n"
              f"  python3 data_collection/record_l2.py --symbol MNQ "
              f"--out-dir {raw}", file=sys.stderr)
        return 2
    os.environ["RAW_DIR"] = str(raw)

    sys.path.insert(0, str(ROOT / "rule_based_v1" / "live"))
    sys.argv = ["weekend_live_runner", "--out-dir", args.out_dir] + (
        ["--live"] if args.live else [])
    import weekend_live_runner
    weekend_live_runner.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
