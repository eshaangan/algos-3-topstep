#!/bin/bash
# Tail remote runner logs and append to Mac logs/remote_pc/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

FOLLOW=1

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Stream remote logs to ${LOG_DIR}/ on this Mac.

Options:
  --no-follow     Print last 100 lines and exit (no background tail).
  -h, --help      Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-follow)
      FOLLOW=0
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "${LOG_DIR}"
preflight_remote
read_remote_session

mode="${DEPLOY_MODE:-docker}"
log_file="${LOG_DIR}/session_$(date +%Y%m%d_%H%M%S).log"

if [[ "${mode}" == "native" && -n "${REMOTE_DIR:-}" ]]; then
  remote_log="${REMOTE_LOG:-.out}"
  if [[ ${FOLLOW} -eq 0 ]]; then
    ssh_remote "tail -100 '${REMOTE_DIR}/${remote_log}' 2>&1" | tee "${log_file}"
    exit 0
  fi
  stop_log_tail
  (
    ssh_remote "tail -F '${REMOTE_DIR}/${remote_log}'" 2>/dev/null | tee -a "${log_file}"
  ) &
  echo $! > "${LOG_PID_FILE}"
  echo "Log tail PID: $(cat "${LOG_PID_FILE}")"
  wait "$(cat "${LOG_PID_FILE}")"
  exit 0
fi

if ! ssh_remote "docker ps --format '{{.Names}}' | grep -qx ${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} is not running on ${SSH_HOST}." >&2
  echo "Deploy first: ${SCRIPT_DIR}/deploy_remote_pc.sh" >&2
  exit 1
fi

if [[ ${FOLLOW} -eq 0 ]]; then
  echo "Fetching last 100 log lines -> ${log_file}"
  ssh_remote "docker logs --tail 100 ${CONTAINER_NAME} 2>&1" | tee "${log_file}"
  exit 0
fi

stop_log_tail
echo "Following remote docker logs -> ${log_file}"
(
  ssh_remote "docker logs -f ${CONTAINER_NAME} 2>&1" | tee -a "${log_file}"
) &
echo $! > "${LOG_PID_FILE}"
echo "Log tail PID: $(cat "${LOG_PID_FILE}")"
wait "$(cat "${LOG_PID_FILE}")"
