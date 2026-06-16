#!/bin/bash
# Install a launchd LaunchAgent on the remote Mac so the L2 recorder auto-starts on
# boot and restarts if it dies — fixing the "nohup dies on reboot" problem.
#
# Honest, non-disguised label: com.jg.mnq-l2recorder
# Run this YOURSELF (it performs reboot-persistence on your own machine):
#     bash gcp_deploy/remote_pc/install_l2_launchagent.sh
#
# Caveat: a LaunchAgent auto-starts at boot only if the Mac AUTO-LOGS-IN to this
# user account. For a headless box, enable: System Settings → Users & Groups →
# Automatic login. Without it, the recorder starts at next manual login.
#
# Uninstall: launchctl unload ~/Library/LaunchAgents/com.jg.mnq-l2recorder.plist
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SYMBOL="${SYMBOL:-MNQ}"
LABEL="com.jg.mnq-l2recorder"

load_env
read_remote_session
if [[ -z "${REMOTE_DIR:-}" ]]; then
  echo "No remote session (logs/remote_pc/.remote_session). Deploy the recorder first." >&2
  exit 1
fi
RD="${REMOTE_DIR}"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# 1. creds env file (chmod 600 on remote), sourced by the agent runner
{
  echo "export RITHMIC_USERNAME=$(printf %q "${RITHMIC_USERNAME}")"
  echo "export RITHMIC_PASSWORD=$(printf %q "${RITHMIC_PASSWORD}")"
  echo "export RITHMIC_SYSTEM_NAME=$(printf %q "${RITHMIC_SYSTEM_NAME:-LucidTrading}")"
  echo "export RITHMIC_APP_NAME=$(printf %q "${RITHMIC_APP_NAME:-XXXX:mnq_l2_recorder}")"
  [[ -n "${RITHMIC_GATEWAY_URI:-}" ]] && echo "export RITHMIC_GATEWAY_URI=$(printf %q "${RITHMIC_GATEWAY_URI}")"
  echo "export OUT_DIR=${RD}/l2_raw"
} > "${TMP}/l2env"

# 2. agent runner
cat > "${TMP}/l2agent.sh" <<AG
#!/bin/bash
cd "${RD}"
source ./.l2env
mkdir -p "\${OUT_DIR}"
exec .venv/bin/python -u .w/data_collection/record_l2.py --symbol ${SYMBOL} --out-dir "\${OUT_DIR}"
AG

# 3. launchd plist (RunAtLoad + KeepAlive = boot-start + auto-restart)
cat > "${TMP}/agent.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>${LABEL}</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>${RD}/.l2agent.sh</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>20</integer>
  <key>StandardOutPath</key><string>${RD}/.l2out</string>
  <key>StandardErrorPath</key><string>${RD}/.l2out</string>
</dict></plist>
PL

ssh_remote "mkdir -p Library/LaunchAgents '${RD}'"
scp -q "${TMP}/l2env"      "${SSH_HOST}:${RD}/.l2env"
scp -q "${TMP}/l2agent.sh" "${SSH_HOST}:${RD}/.l2agent.sh"
scp -q "${TMP}/agent.plist" "${SSH_HOST}:Library/LaunchAgents/${LABEL}.plist"

ssh_remote "bash -s" <<REMOTE
set -e
RD='${RD}'; LABEL='${LABEL}'
chmod 600 "\${RD}/.l2env"; chmod +x "\${RD}/.l2agent.sh"
# stop the old nohup recorder so we don't run two connections
[ -f "\${RD}/.l2pid" ] && kill "\$(cat \${RD}/.l2pid)" 2>/dev/null || true
pkill -f 'record_l2\.py' 2>/dev/null || true
pkill -f '\.l2wd' 2>/dev/null || true
sleep 2
PLIST="\${HOME}/Library/LaunchAgents/\${LABEL}.plist"
launchctl unload "\${PLIST}" 2>/dev/null || true
launchctl load -w "\${PLIST}"
sleep 7
echo "--- verify ---"
launchctl list | grep "\${LABEL}" && echo "agent registered ✓" || echo "agent NOT registered ✗"
echo "record_l2 procs: \$(pgrep -fc 'record_l2\.py')"
pgrep -fl 'record_l2\.py' >/dev/null && echo "recorder running under launchd ✓" || echo "recorder NOT running ✗"
tail -2 "\${RD}/.l2out"
REMOTE

echo ""
echo "Done. The recorder is now launchd-managed (auto-starts on boot, restarts if it dies)."
echo "If the Mac does NOT auto-login, enable Automatic Login so it starts at boot."
