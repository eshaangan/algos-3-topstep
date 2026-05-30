#!/bin/bash
# Stop remote ML strategy runner and optionally purge remote artifacts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

PURGE_IMAGE=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Stop remote ML strategy on ${SSH_HOST}.

Options:
  --purge-image   Remove Docker image from remote PC (docker mode only).
  -h, --help      Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --purge-image)
      PURGE_IMAGE=1
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

echo "Stopping log tail on Mac..."
stop_log_tail

read_remote_session
mode="${DEPLOY_MODE:-docker}"

if [[ "${mode}" == "native" && -n "${REMOTE_DIR:-}" ]]; then
  stop_native_remote
elif remote_has_docker; then
  echo "Stopping container on ${SSH_HOST}..."
  ssh_remote "docker rm -f ${CONTAINER_NAME} 2>/dev/null || true"
  clear_remote_session

  if [[ ${PURGE_IMAGE} -eq 1 ]]; then
    echo "Removing image ${IMAGE_NAME} from remote..."
    ssh_remote "docker rmi ${IMAGE_NAME} 2>/dev/null || true"
    echo "Remote PC cleaned (no container, no image)."
  else
    echo "Container stopped. Image ${IMAGE_NAME} may remain on remote."
    echo "Use --purge-image to remove Docker image artifacts."
  fi
else
  echo "No active remote session found."
  stop_native_remote
fi
