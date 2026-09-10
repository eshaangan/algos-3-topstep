#!/bin/bash
# Supervisor for the two-leg live book (weekend_hold_v1 + fomc_drift_v1).
#
# Runs two processes and restarts either if it dies:
#   record_l2.py       -- holds the Rithmic TICKER session, writes the quote stream
#   book_live_runner   -- holds the ORDER + PNL sessions, places and protects trades
#
# They are deliberately separate Rithmic sessions on different plants. Rithmic
# permits ONE concurrent ticker session per login, which is why the runner never
# subscribes to market data itself and reads the recorder's file instead.
#
# cron is unusable on this machine (documented in the project ledger), so this is
# a foreground loop intended to be started under caffeinate and left running.
#
# STOP: touch logs/book_live/KILL  -- the runner flattens and exits, and this
#       supervisor sees the file and stops restarting it.
#
#   ./rule_based_v1/live/launch_book.sh          # dry run (no orders)
#   ./rule_based_v1/live/launch_book.sh --live   # REAL ORDERS
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
ROOT="$PWD"

OUT="$ROOT/logs/book_live"
RAW="$ROOT/data/l2_raw"
KILL="$OUT/KILL"
mkdir -p "$OUT" "$RAW"

LIVE_FLAG=""
MODE="DRY"
if [ "${1:-}" = "--live" ]; then
  LIVE_FLAG="--live"
  MODE="LIVE"
fi

export RAW_DIR="$RAW"
export PYTHONPATH="$ROOT"

echo "=== book supervisor starting in $MODE mode $(date -u +%FT%TZ) ===" \
  | tee -a "$OUT/supervisor.log"
echo "    out=$OUT raw=$RAW  (touch $KILL to stop)" | tee -a "$OUT/supervisor.log"

sup() {                       # sup <name> <logfile> <command...>
  local name="$1" logf="$2"; shift 2
  (
    while true; do
      if [ -f "$KILL" ]; then
        echo "[$(date -u +%H:%M:%SZ)] $name: KILL present, not restarting" >> "$logf"
        break
      fi
      echo "[$(date -u +%H:%M:%SZ)] $name: starting" >> "$logf"
      "$@" >> "$logf" 2>&1
      rc=$?
      echo "[$(date -u +%H:%M:%SZ)] $name: exited rc=$rc" >> "$logf"
      # A clean exit from the runner means it hit its KILL file; do not respawn.
      if [ "$name" = "runner" ] && [ $rc -eq 0 ] && [ -f "$KILL" ]; then
        break
      fi
      sleep 10
    done
  ) &
  echo "    $name supervised (pid $!)" | tee -a "$OUT/supervisor.log"
}

sup recorder "$OUT/recorder.log" \
  python3 "$ROOT/rule_based_v1/live/launch_recorder.py" --symbol MNQ --out-dir "$RAW"

# Let the recorder establish a session and lay down quotes before the runner
# starts looking for them, so the first entry window is not skipped for
# "no fresh quote" purely because of startup ordering.
sleep 30

sup runner "$OUT/runner.log" \
  python3 -m rule_based_v1.live.book_live_runner --out-dir "$OUT" $LIVE_FLAG

wait
