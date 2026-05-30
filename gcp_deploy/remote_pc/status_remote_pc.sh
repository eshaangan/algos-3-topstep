#!/bin/bash
# Show remote runner status and recent logs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

preflight_remote
read_remote_session

echo "Remote host: ${SSH_HOST}"
echo "Deploy mode: ${DEPLOY_MODE:-unknown}"
echo ""

if [[ "${DEPLOY_MODE:-}" == "native" && -n "${REMOTE_DIR:-}" ]]; then
  remote_log="${REMOTE_LOG:-.out}"
  pid_file="${REMOTE_PID:-.pid}"
  echo "Native session: active"
  runner_pid="$(ssh_remote "cat '${REMOTE_DIR}/${pid_file}' 2>/dev/null || echo ''")"
  if [[ -n "${runner_pid}" ]] && ssh_remote "kill -0 '${runner_pid}' 2>/dev/null"; then
    echo "Process: running (pid ${runner_pid})"
  else
    echo "Process: not running"
  fi
  echo ""
  echo "Last 20 log lines:"
  ssh_remote "tail -20 '${REMOTE_DIR}/${remote_log}' 2>&1" || true
elif remote_has_docker; then
  echo "Docker containers:"
  ssh_remote "docker ps -a --filter name=${CONTAINER_NAME} --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'" || true
  echo ""
  if ssh_remote "docker ps --format '{{.Names}}' | grep -qx ${CONTAINER_NAME}"; then
    echo "Last 20 log lines:"
    ssh_remote "docker logs --tail 20 ${CONTAINER_NAME} 2>&1" || true
  else
    echo "Container ${CONTAINER_NAME} is not running."
  fi
else
  echo "No active docker or native session detected."
fi

echo ""
if [[ -f "${LOG_PID_FILE}" ]]; then
  pid="$(cat "${LOG_PID_FILE}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Mac log tail: running (PID ${pid})"
  else
    echo "Mac log tail: stale PID file (${pid})"
  fi
else
  echo "Mac log tail: not running"
fi

if [[ -d "${LOG_DIR}" ]]; then
  latest_log="$(ls -t "${LOG_DIR}"/session_*.log 2>/dev/null | head -1 || true)"
  if [[ -n "${latest_log}" ]]; then
    echo "Latest Mac log file: ${latest_log}"
  fi
fi
