#!/bin/bash
# Stop the L2 recorder on the remote Mac (does NOT touch the trader).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

read_remote_session
if [[ -z "${REMOTE_DIR:-}" ]]; then
  echo "No remote session found." >&2
  exit 1
fi

ssh_remote "bash -s" <<EOF
cd '${REMOTE_DIR}' 2>/dev/null || true
if [[ -f .l2pid ]]; then
  WD=\$(cat .l2pid)
  pkill -TERM -P "\${WD}" 2>/dev/null || true
  kill "\${WD}" 2>/dev/null || true
  sleep 1
fi
pkill -f 'record_l2.py' 2>/dev/null || true
pkill -f '.l2wd' 2>/dev/null || true
rm -f .l2pid
echo "L2 recorder stopped."
EOF
