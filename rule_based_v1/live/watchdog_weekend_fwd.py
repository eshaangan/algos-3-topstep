#!/usr/bin/env python3
"""
Watchdog for the weekend_hold_v1 forward observation.

Checks that the two processes the observation depends on are not just ALIVE but
actually WORKING, and alerts if not. Designed to be run from cron every few
minutes; it does one pass and exits.

    recorder  data_collection/record_l2.py   -- feeds the quote stream
    runner    weekend_live_runner.py         -- places the Sunday 18:00 entry

WHY LIVENESS AND NOT JUST PRESENCE
A process can be alive and useless. The recorder holds a websocket; if Rithmic
drops it in a way the client does not recover from, the process keeps running
while the stream goes stale. `pgrep` would say everything is fine and the
Sunday entry would silently skip on a dead quote. So the recorder check is
"is stream_MNQ_*.csv still growing", not "does the pid exist".

RESTART POLICY -- THE IMPORTANT PART
The recorder MAY be auto-restarted: it places no orders and a restart costs at
most a small gap in recorded depth.

The runner is NEVER auto-restarted, and this is deliberate. weekend_live_runner
flattens any open position on startup:

    if self.st.get("open") and self.live:  await self.flatten("startup_recovery")

That is correct behaviour for a crash-recovery path, but it means restarting the
runner while a weekend hold is OPEN would CLOSE the trade at market. A watchdog
that "helpfully" restarts it on Sunday night would liquidate the very position
it is supposed to be protecting. So: alert a human, never act.

ALERTS
  - macOS notification (local, private)
  - ntfy.sh push if NTFY_TOPIC is set -- note ntfy topics are PUBLIC, so
    messages here name no account, size, price or PnL
  - a line in the watchdog log

Usage:
    python3 rule_based_v1/live/watchdog_weekend_fwd.py            # one pass
    python3 rule_based_v1/live/watchdog_weekend_fwd.py --no-restart
    python3 rule_based_v1/live/watchdog_weekend_fwd.py --dry      # report only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "l2_raw"
OUT_DIR = ROOT / "logs" / "weekend_fwd"
LIVE_DIR = OUT_DIR / "live"
WD_LOG = OUT_DIR / "watchdog.log"
STALE_SECS = 900          # quote older than this on a trading day = stale
# Match either spelling: the recorder may be started directly (record_l2.py, as
# on a workstation with .env sourced) or via its dotenv launcher
# (launch_recorder.py, which is how the watchdog and the mini start it). A
# pattern matching only one of the two would report the recorder permanently
# down and restart-loop it forever.
RECORDER_PAT = "record_l2.py|launch_recorder.py"
RUNNER_PAT = "launch_weekend_fwd.py"


def _load_env() -> None:
    """Pick up NTFY_TOPIC from .env. cron gives us almost no environment."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}"
    print(line, flush=True)
    WD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(WD_LOG, "a") as fh:
        fh.write(line + "\n")


def alert(title: str, body: str) -> None:
    """Local notification + optional public push. Never include account detail."""
    log(f"ALERT: {title} -- {body}")
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification {json.dumps(body)} with title {json.dumps(title)}'],
            check=False, capture_output=True, timeout=10)
    except Exception:
        pass
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if topic:
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"https://ntfy.sh/{topic}",
                data=f"{title}: {body}".encode(),
                headers={"Title": "weekend_fwd watchdog", "Priority": "high"},
                method="POST"), timeout=10)
        except Exception as exc:
            log(f"ntfy push failed: {type(exc).__name__}")


def pids(pattern: str) -> list:
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True,
                             text=True, timeout=10).stdout.split()
        return [int(p) for p in out if p.isdigit()]
    except Exception:
        return []


def quote_age() -> float:
    """Seconds since the newest line in the recorder's stream file."""
    files = sorted(glob.glob(str(RAW_DIR / "stream_MNQ_*.csv")))
    if not files:
        return float("inf")
    newest = files[-1]
    try:
        with open(newest, "rb") as fh:
            try:
                fh.seek(-65536, 2)
            except OSError:
                fh.seek(0)
            lines = fh.read().decode(errors="ignore").strip().splitlines()
        for line in reversed(lines):
            parts = line.split(",")
            if len(parts) == 5:
                return time.time() - float(parts[0]) / 1e9
    except Exception:
        pass
    # Fall back to file mtime if the tail is unparseable.
    return time.time() - os.path.getmtime(newest)


def market_should_be_open() -> bool:
    """
    CME equity index roughly: Sun 18:00 ET -> Fri 17:00 ET, daily halt 17:00-18:00.
    Used only to avoid crying 'stale quote' during a legitimate closure.
    """
    import pandas as pd
    now = pd.Timestamp.now(tz="America/New_York")
    dow, hour = now.weekday(), now.hour
    if dow == 5:                                  # Saturday
        return False
    if dow == 6:                                  # Sunday: opens 18:00
        return hour >= 18
    if dow == 4 and hour >= 17:                   # Friday after 17:00
        return False
    return hour != 17                             # daily maintenance hour


def restart_recorder() -> bool:
    # Go through launch_recorder, not record_l2 directly: record_l2 reads
    # os.environ and does not load .env, so invoking it straight from here
    # (cron/daemon context, no sourced shell) dies on KeyError RITHMIC_USERNAME.
    cmd = ["caffeinate", "-is", sys.executable,
           str(ROOT / "rule_based_v1" / "live" / "launch_recorder.py"),
           "--symbol", "MNQ", "--out-dir", str(RAW_DIR)]
    try:
        with open(OUT_DIR / "recorder.log", "a") as fh:
            subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        return True
    except Exception as exc:
        log(f"recorder restart FAILED: {type(exc).__name__}: {exc}")
        return False


def position_is_open() -> bool:
    try:
        return json.load(open(LIVE_DIR / "state.json")).get("open") is not None
    except Exception:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-restart", action="store_true",
                    help="alert only; never restart the recorder")
    ap.add_argument("--dry", action="store_true", help="report, take no action")
    ap.add_argument("--stale-secs", type=int, default=STALE_SECS)
    ap.add_argument("--loop", type=int, default=0, metavar="SECS",
                    help="run forever, checking every SECS (0 = single pass)")
    args = ap.parse_args(argv)

    _load_env()
    if args.loop:
        # cron is not usable on this machine: it has no Full Disk Access, so it
        # cannot even read a script under ~/Documents ("Operation not
        # permitted"), and it resolves python3 to the CommandLineTools build
        # which lacks pandas. A long-lived user process inherits the launching
        # terminal's rights, which is how the recorder and runner already run.
        log(f"watchdog loop started (every {args.loop}s)")
        while True:
            try:
                _one_pass(args)
            except Exception as exc:
                log(f"watchdog pass error: {type(exc).__name__}: {exc}")
            time.sleep(args.loop)
    return _one_pass(args)


def _one_pass(args) -> int:
    problems = []

    # --- runner: presence only, NEVER restarted ------------------------------
    rp = pids(RUNNER_PAT)
    if rp:
        log(f"runner OK (pid {rp[0]})")
    else:
        state = " -- A POSITION IS OPEN" if position_is_open() else ""
        problems.append("runner DOWN" + state)
        alert("weekend runner DOWN",
              "Sunday entry will not fire. Restart it by hand; do NOT let anything "
              "auto-restart it while a position is open (startup flattens)." + state)

    # --- recorder: liveness, may be restarted --------------------------------
    cp = pids(RECORDER_PAT)
    age = quote_age()
    open_mkt = market_should_be_open()
    stale = open_mkt and age > args.stale_secs

    if cp and not stale:
        log(f"recorder OK (pid {cp[0]}, quote age {age:.0f}s, "
            f"market {'open' if open_mkt else 'closed'})")
    elif not cp:
        problems.append("recorder DOWN")
        if args.dry or args.no_restart:
            alert("L2 recorder DOWN", "Quote feed absent; Sunday entry would skip.")
        elif restart_recorder():
            alert("L2 recorder restarted", "Was down; watchdog restarted it.")
        else:
            alert("L2 recorder DOWN", "Restart FAILED; needs a human.")
    else:
        problems.append(f"recorder STALLED (quote {age:.0f}s old)")
        if args.dry or args.no_restart:
            alert("L2 recorder STALLED",
                  f"Process alive but quote is {age:.0f}s old.")
        else:
            for p in cp:
                try:
                    os.kill(p, 15)
                except Exception:
                    pass
            time.sleep(3)
            if restart_recorder():
                alert("L2 recorder restarted",
                      f"Was stalled ({age:.0f}s old quote); restarted.")
            else:
                alert("L2 recorder STALLED", "Restart FAILED; needs a human.")

    if not problems:
        log("all checks passed")
        return 0
    log("PROBLEMS: " + "; ".join(problems))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
