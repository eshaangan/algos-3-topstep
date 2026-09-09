#!/usr/bin/env python3
"""
Launcher for the L2 quote recorder.

record_l2.py reads os.environ["RITHMIC_USERNAME"] directly and does not load
.env itself. On a workstation that is masked by having sourced .env in the
shell; on a fresh host it fails immediately with KeyError. Shell sourcing is not
an option anyway -- this repo's .env has an unquoted value containing '=' on
line 10, so `. ./.env` tries to execute it.

So: load .env with python-dotenv, then hand off to record_l2.main().

Usage:
    python3 rule_based_v1/live/launch_recorder.py --symbol MNQ --out-dir data/l2_raw
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="MNQ")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "l2_raw"))
    ap.add_argument("--flush-secs", type=float, default=60.0)
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

    sys.path.insert(0, str(ROOT / "data_collection"))
    sys.argv = ["record_l2", "--symbol", args.symbol,
                "--out-dir", args.out_dir, "--flush-secs", str(args.flush_secs)]
    import record_l2
    record_l2.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
