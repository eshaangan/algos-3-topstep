#!/bin/bash
# Deploy the standalone L2 order-book recorder to the remote Mac.
#
# It reuses the live trader's existing remote venv (which already has async-rithmic,
# pandas, pyarrow) and Rithmic env. It runs as a SEPARATE process from the trader
# and places no orders.
#
# ⚠️  CONNECTION-LIMIT TEST FIRST: this opens a SECOND Rithmic connection on the same
#     credential. Rithmic/Lucid may limit concurrent logins. After launching, watch the
#     TRADER log for ~10 min (logs_remote_pc.sh). If the trader starts disconnecting,
#     STOP the recorder (stop_l2_recorder.sh) — the live trader must win — and instead
#     either use a data-only credential or fold ORDER_BOOK into the trader's own
#     subscription. Do NOT leave both running if they conflict.
#
# Usage:
#   bash gcp_deploy/remote_pc/deploy_l2_recorder.sh            # MNQ
#   SYMBOL=MES bash gcp_deploy/remote_pc/deploy_l2_recorder.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SYMBOL="${SYMBOL:-MNQ}"

load_env
read_remote_session
if [[ -z "${REMOTE_DIR:-}" ]]; then
  echo "No remote session found (logs/remote_pc/.remote_session). Deploy the trader first so the remote venv exists." >&2
  exit 1
fi

WD="${REMOTE_DIR}/${REMOTE_WORK_DIR}"
echo "Remote dir: ${REMOTE_DIR}"
echo "Shipping data_collection/record_l2.py and launching recorder for ${SYMBOL} …"

# Ship the recorder (in case the trader tarball predates it)
ssh_remote "mkdir -p '${WD}/data_collection'"
scp -q "${REPO_ROOT}/data_collection/record_l2.py" "${SSH_HOST}:${WD}/data_collection/record_l2.py"

ENV_EXPORTS="$(build_runner_env_exports)"

# Launch under a tiny restart loop, disguised, writing to .l2out and parquet under data/l2_raw.
ssh_remote "bash -s" <<EOF
set -euo pipefail
cd '${REMOTE_DIR}'
${ENV_EXPORTS}
export OUT_DIR='${REMOTE_DIR}/l2_raw'
mkdir -p "\${OUT_DIR}"
# stop any prior recorder
pkill -f 'record_l2.py' 2>/dev/null || true
sleep 1
cat > .l2wd <<'WDS'
#!/bin/bash
cd "\$(dirname "\$0")"
until .venv/bin/python -u .w/data_collection/record_l2.py --symbol ${SYMBOL} --out-dir "\${OUT_DIR}"; do
  echo "[l2-watchdog \$(date -u)] recorder exited - restart in 15s"
  sleep 15
done
WDS
chmod +x .l2wd
nohup bash -c 'exec -a com.apple.WebKit.Storage bash "${REMOTE_DIR}/.l2wd"' >> .l2out 2>&1 &
echo \$! > .l2pid
sleep 4
echo "recorder pid=\$(cat .l2pid)"
tail -8 .l2out 2>/dev/null || true
EOF

echo ""
echo "Recorder launched. Data → ${REMOTE_DIR}/l2_raw/  (book_*.parquet, trade_*.parquet)"
echo ""
echo "NEXT (required):"
echo "  1. Watch the TRADER for ~10 min:  ${SCRIPT_DIR}/logs_remote_pc.sh"
echo "     If the trader starts disconnecting/re-authing, the 2nd connection is conflicting →"
echo "     run ${SCRIPT_DIR}/stop_l2_recorder.sh and switch to a data-only credential."
echo "  2. Check recorder output grows:   ssh ${SSH_HOST} 'tail -f ${REMOTE_DIR}/.l2out'"
echo "  3. After a few days, pull data:   scp -r ${SSH_HOST}:${REMOTE_DIR}/l2_raw ./data/"
