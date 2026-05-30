#!/bin/bash
# Build ML strategy on Mac and run on remote PC (Docker if available, else ephemeral native SSH).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

LIVE_MODE=0
START_LOGS=1
RUNNER_ARGS=(--dry-run -v)

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Deploy ml_strategy to remote PC (${SSH_HOST}).

Options:
  --live          Live trading (--live --yes). Default is dry-run.
  --no-logs       Do not start background log tail on Mac after deploy.
  --docker        Force Docker deploy (requires Docker on remote).
  --native        Force native SSH deploy (ephemeral /tmp, cleaned on stop).
  -h, --help      Show this help.

Environment overrides:
  SSH_HOST, ML_N_CONTRACTS, DEPLOY_MODE (auto|docker|native)

Examples:
  $(basename "$0")                          # dry-run, auto mode
  ML_N_CONTRACTS=2 $(basename "$0") --live  # conservative live test
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --live)
      LIVE_MODE=1
      RUNNER_ARGS=(--live --yes -v)
      shift
      ;;
    --no-logs)
      START_LOGS=0
      shift
      ;;
    --docker)
      DEPLOY_MODE=docker
      shift
      ;;
    --native)
      DEPLOY_MODE=native
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

cd "${REPO_ROOT}"

echo "========================================="
echo "Remote PC ML Strategy Deploy"
echo "Host:      ${SSH_HOST}"
echo "Image:     ${IMAGE_NAME}"
echo "Container: ${CONTAINER_NAME}"
echo "Mode:      $([[ ${LIVE_MODE} -eq 1 ]] && echo LIVE || echo dry-run)"
echo "Contracts: ${ML_N_CONTRACTS}"
echo "Account:   ${LUCID_ACCOUNT_ID:-auto-discover}"
echo "========================================="
echo ""

load_env
preflight_remote
ensure_model

MODE="$(resolve_deploy_mode)"
echo "Deploy transport: ${MODE}"
echo ""

RUNNER_CMD="${RUNNER_ARGS[*]}"

if [[ "${MODE}" == "docker" ]]; then
  echo "Building Docker image locally..."
  docker build -f gcp_deploy/Dockerfile.ml_strategy -t "${IMAGE_NAME}" .

  echo "Transferring image to ${SSH_HOST} (streamed, no tarball on remote)..."
  docker save "${IMAGE_NAME}" | gzip | ssh_remote 'gunzip | docker load'

  echo "Stopping old container on remote (if any)..."
  ssh_remote "docker rm -f ${CONTAINER_NAME} 2>/dev/null || true"

  echo "Starting container on remote..."
  REMOTE_RUN="
docker run -d --name ${CONTAINER_NAME} --restart unless-stopped \
  -e TOPSTEPX_USERNAME=$(printf '%q' "${TOPSTEPX_USERNAME}") \
  -e TOPSTEPX_PROJECTX_API_KEY=$(printf '%q' "${TOPSTEPX_PROJECTX_API_KEY}") \
  -e TOPSTEPX_ACCOUNT_ID=$(printf '%q' "${PRACTICE_ACCOUNT_ID}") \
  -e TOPSTEPX_CONTRACT_ID=$(printf '%q' "${PRACTICE_CONTRACT_ID}") \
  -e ML_N_CONTRACTS=$(printf '%q' "${ML_N_CONTRACTS}") \
  -e ML_STDOUT_ONLY=1"

  if [[ -n "${TOPSTEPX_PROJECTX_BASE_URL:-}" ]]; then
    REMOTE_RUN="${REMOTE_RUN} -e TOPSTEPX_PROJECTX_BASE_URL=$(printf '%q' "${TOPSTEPX_PROJECTX_BASE_URL}")"
  fi
  if [[ -n "${TOPSTEPX_SESSION_TOKEN:-}" ]]; then
    REMOTE_RUN="${REMOTE_RUN} -e TOPSTEPX_SESSION_TOKEN=$(printf '%q' "${TOPSTEPX_SESSION_TOKEN}")"
  fi

  REMOTE_RUN="${REMOTE_RUN} ${IMAGE_NAME} ${RUNNER_CMD}"
  ssh_remote "${REMOTE_RUN}"
  write_remote_session "" "" "docker"

  echo ""
  echo "Container started. Checking status..."
  ssh_remote "docker ps --filter name=${CONTAINER_NAME} --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'"

  echo ""
  echo "Recent logs:"
  ssh_remote "docker logs --tail 20 ${CONTAINER_NAME} 2>&1" || true

  if [[ ${START_LOGS} -eq 1 ]]; then
    start_log_tail
  fi
else
  deploy_native "${RUNNER_CMD}"

  echo ""
  read_remote_session
  remote_log="${REMOTE_LOG:-.out}"
  ssh_remote "tail -20 '${REMOTE_DIR}/${remote_log}' 2>&1" || true

  if [[ ${START_LOGS} -eq 1 ]]; then
    start_log_tail_native
  fi
fi

echo ""
if [[ ${START_LOGS} -eq 1 ]]; then
  echo "Log tail running in background. Stop with: ${SCRIPT_DIR}/stop_remote_pc.sh"
fi

echo ""
if [[ ${LIVE_MODE} -eq 1 ]]; then
  echo "LIVE mode — real orders on Lucid account ${LUCID_ACCOUNT_ID:-auto-discover}."
else
  echo "Dry-run mode — no real orders. Re-run with --live for practice live trading."
fi
echo "Monitor: ${SCRIPT_DIR}/logs_remote_pc.sh"
echo "Status:  ${SCRIPT_DIR}/status_remote_pc.sh"
echo "Stop:    ${SCRIPT_DIR}/stop_remote_pc.sh"
