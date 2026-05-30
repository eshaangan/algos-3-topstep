#!/bin/bash
# Shared config for remote PC ML strategy deployment (Mac orchestrates, PC runs Docker only).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SSH_HOST="${SSH_HOST:-jg@100.81.204.115}"
IMAGE_NAME="${IMAGE_NAME:-local-cache:1}"
CONTAINER_NAME="${CONTAINER_NAME:-svc-cache-1}"

# Hidden ephemeral layout on remote (dot-prefixed tmp + generic process title).
REMOTE_TMP_TEMPLATE="/tmp/.svc-XXXXXX"
REMOTE_WORK_DIR=".w"
REMOTE_LOG=".out"
REMOTE_PID=".pid"
REMOTE_PROC_TITLE="${REMOTE_PROC_TITLE:-com.apple.WebKit.Networking}"

LUCID_ACCOUNT_ID="${LUCID_ACCOUNT_ID:-}"
ML_N_CONTRACTS="${ML_N_CONTRACTS:-6}"

LOG_DIR="${REPO_ROOT}/logs/remote_pc"
LOG_PID_FILE="${LOG_DIR}/.log_tail_pid"
REMOTE_SESSION_FILE="${LOG_DIR}/.remote_session"
ENV_FILE="${REPO_ROOT}/.env"

DEPLOY_MODE="${DEPLOY_MODE:-auto}" # auto | docker | native

ssh_remote() {
  ssh -o BatchMode=yes -o ConnectTimeout=15 "${SSH_HOST}" "$@"
}

require_env_vars() {
  local missing=()
  for var in RITHMIC_USERNAME RITHMIC_PASSWORD; do
    if [[ -z "${!var:-}" ]]; then
      missing+=("${var}")
    fi
  done
  if ((${#missing[@]} > 0)); then
    echo "Missing required env vars: ${missing[*]}" >&2
    echo "Source ${ENV_FILE} or export them before running." >&2
    exit 1
  fi
}

load_env() {
  if [[ -f "${ENV_FILE}" ]]; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
      line="${line%%#*}"
      line="${line%"${line##*[![:space:]]}"}"
      [[ -z "${line}" ]] && continue
      if [[ "${line}" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        value="${value#\"}"
        value="${value%\"}"
        value="${value#\'}"
        value="${value%\'}"
        export "${key}=${value}"
      fi
    done < "${ENV_FILE}"
  fi
  require_env_vars
}

preflight_remote() {
  echo "Checking SSH to ${SSH_HOST}..."
  ssh_remote "echo ok" >/dev/null
}

remote_has_docker() {
  ssh_remote "command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1"
}

resolve_deploy_mode() {
  if [[ "${DEPLOY_MODE}" == "docker" ]]; then
    if ! remote_has_docker; then
      echo "Docker requested but unavailable on ${SSH_HOST}." >&2
      exit 1
    fi
    echo "docker"
    return
  fi
  if [[ "${DEPLOY_MODE}" == "native" ]]; then
    echo "native"
    return
  fi
  if remote_has_docker; then
    echo "docker"
  else
    echo "Docker not found on ${SSH_HOST}; using native SSH deploy (ephemeral /tmp, cleaned on stop)."
    echo "native"
  fi
}

write_remote_session() {
  mkdir -p "${LOG_DIR}"
  cat > "${REMOTE_SESSION_FILE}" <<EOF
REMOTE_DIR=${1}
RUNNER_PID=${2}
DEPLOY_MODE=${3}
REMOTE_LOG=${4:-${REMOTE_LOG}}
EOF
}

read_remote_session() {
  if [[ -f "${REMOTE_SESSION_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${REMOTE_SESSION_FILE}"
  fi
}

clear_remote_session() {
  rm -f "${REMOTE_SESSION_FILE}"
}

build_runner_env_exports() {
  local account_id="${1:-${LUCID_ACCOUNT_ID}}"
  local n_contracts="${2:-${ML_N_CONTRACTS}}"

  cat <<EOF
export RITHMIC_USERNAME=$(printf '%q' "${RITHMIC_USERNAME}")
export RITHMIC_PASSWORD=$(printf '%q' "${RITHMIC_PASSWORD}")
export RITHMIC_SYSTEM_NAME=$(printf '%q' "${RITHMIC_SYSTEM_NAME:-Lucid Trading}")
export RITHMIC_APP_NAME=$(printf '%q' "${RITHMIC_APP_NAME:-XXXX:mnq_ml_trader}")
export LUCID_ACCOUNT_ID=$(printf '%q' "${account_id}")
export ML_N_CONTRACTS=$(printf '%q' "${n_contracts}")
export TRADING_BROKER=rithmic
export RISK_CONFIG=risk_lucid_100k.yaml
export ML_STDOUT_ONLY=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=${REMOTE_DIR}/${REMOTE_WORK_DIR}:${REMOTE_DIR}/${REMOTE_WORK_DIR}/rule_based_v1:${REMOTE_DIR}/${REMOTE_WORK_DIR}/ml_intraday_v3
EOF
  if [[ -n "${RITHMIC_GATEWAY_URI:-}" ]]; then
    echo "export RITHMIC_GATEWAY_URI=$(printf '%q' "${RITHMIC_GATEWAY_URI}")"
  fi
}

deploy_native() {
  local runner_cmd="$1"

  stop_native_remote 2>/dev/null || true
  ssh_remote "bash -c 'rm -rf /tmp/ml-strategy-* 2>/dev/null || true'"

  REMOTE_DIR="$(ssh_remote "mktemp -d ${REMOTE_TMP_TEMPLATE}")"

  tar czf - \
    -C "${REPO_ROOT}" \
    core \
    rule_based_v1/live \
    rule_based_v1/diagnostics/ml_strategy_search.py \
    rule_based_v1/models/ml_strategy_mnq_v7.pkl \
    rule_based_v1/configs/risk_lucid_100k.yaml \
    rule_based_v1/engine \
    ml_intraday_v3/live_trading \
    ml_intraday_v3/__init__.py \
    ml_intraday_v3/scripts/__init__.py \
    ml_intraday_v3/scripts/ml_scalper_v7.py \
    gcp_deploy/requirements_ml_strategy.txt \
    | ssh_remote "mkdir -p '${REMOTE_DIR}/${REMOTE_WORK_DIR}' && tar xzf - -C '${REMOTE_DIR}/${REMOTE_WORK_DIR}'"

  ssh_remote "cd '${REMOTE_DIR}' && PY3=\$(command -v python3.11 2>/dev/null || echo /opt/homebrew/bin/python3.11) && \"\$PY3\" -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r ${REMOTE_WORK_DIR}/gcp_deploy/requirements_ml_strategy.txt"

  ssh_remote "cp '${REMOTE_DIR}/${REMOTE_WORK_DIR}/rule_based_v1/live/ml_strategy_runner.py' '${REMOTE_DIR}/${REMOTE_WORK_DIR}/rule_based_v1/live/.run.py'"

  # Write watchdog script via Python to avoid nested-quoting issues.
  # 'until' exits when python returns 0 (clean stop); restarts on crash (non-zero).
  ssh_remote "python3 - '${REMOTE_DIR}' '${runner_cmd}' << 'PYEOF'
import os, sys, stat
remote_dir, runner_cmd = sys.argv[1], sys.argv[2]
wd = os.path.join(remote_dir, '.wd')
script = (
    '#!/bin/bash\n'
    'cd \"' + remote_dir + '\"\n'
    'until .venv/bin/python -u .w/rule_based_v1/live/.run.py --model-path .w/rule_based_v1/models/ml_strategy_mnq_v7.pkl ' + runner_cmd + '; do\n'
    '  echo \"[watchdog \$(date -u)] crash - restarting in 30s\"\n'
    '  sleep 30\n'
    'done\n'
)
open(wd, 'w').write(script)
os.chmod(wd, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
PYEOF"

  local env_script
  env_script="$(build_runner_env_exports "${LUCID_ACCOUNT_ID}" "${ML_N_CONTRACTS}")"

  runner_pid="$(ssh_remote "bash -s" <<EOF
set -euo pipefail
cd '${REMOTE_DIR}'
${env_script}
nohup bash -c 'exec -a "${REMOTE_PROC_TITLE}" bash "${REMOTE_DIR}/.wd"' >> ${REMOTE_LOG} 2>&1 &
echo \$! > ${REMOTE_PID}
cat ${REMOTE_PID}
EOF
)"
  write_remote_session "${REMOTE_DIR}" "${runner_pid}" "native" "${REMOTE_LOG}"
}

start_log_tail_native() {
  read_remote_session
  if [[ -z "${REMOTE_DIR:-}" ]]; then
    echo "No remote session found." >&2
    return 1
  fi
  local remote_log="${REMOTE_LOG:-${REMOTE_LOG}}"
  mkdir -p "${LOG_DIR}"
  stop_log_tail
  local log_file="${LOG_DIR}/session_$(date +%Y%m%d).log"
  (
    ssh_remote "tail -F '${REMOTE_DIR}/${remote_log}'" 2>/dev/null | tee -a "${log_file}"
  ) &
  echo $! > "${LOG_PID_FILE}"
}

stop_native_remote() {
  read_remote_session
  if [[ -z "${REMOTE_DIR:-}" ]]; then
    return 0
  fi
  local remote_log="${REMOTE_LOG:-.out}"
  local pid_file="${REMOTE_PID:-.pid}"
  ssh_remote "
    if [[ -f '${REMOTE_DIR}/${pid_file}' ]]; then
      WD_PID=\$(cat '${REMOTE_DIR}/${pid_file}')
      # Kill the watchdog and all its children (catches the Python runner)
      pkill -TERM -P \"\${WD_PID}\" 2>/dev/null || true
      kill \"\${WD_PID}\" 2>/dev/null || true
      sleep 1
      pkill -KILL -P \"\${WD_PID}\" 2>/dev/null || true
      kill -9 \"\${WD_PID}\" 2>/dev/null || true
    fi
    # Belt-and-suspenders: kill any .run.py process (our unique hidden script name)
    pkill -f '\.run\.py' 2>/dev/null || true
    pkill -f '${REMOTE_DIR}/' 2>/dev/null || true
    rm -rf '${REMOTE_DIR}'
  " || true
  clear_remote_session
}

preflight_remote_docker() {
  preflight_remote
  echo "Checking Docker on remote..."
  if ! remote_has_docker; then
    echo "Docker not available on ${SSH_HOST}." >&2
    exit 1
  fi
}

ensure_model() {
  local model_path="${REPO_ROOT}/rule_based_v1/models/ml_strategy_mnq_v7.pkl"
  if [[ ! -f "${model_path}" ]]; then
    echo "Model bundle missing: ${model_path}" >&2
    exit 1
  fi
  echo "Model: ${model_path} ($(du -sh "${model_path}" | cut -f1))"
}

stop_log_tail() {
  if [[ -f "${LOG_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${LOG_PID_FILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
    rm -f "${LOG_PID_FILE}"
  fi
}

start_log_tail() {
  mkdir -p "${LOG_DIR}"
  stop_log_tail

  local log_file="${LOG_DIR}/session_$(date +%Y%m%d).log"
  echo "Tailing remote logs -> ${log_file}"
  (
    ssh_remote "docker logs -f ${CONTAINER_NAME} 2>&1" | tee -a "${log_file}"
  ) &
  echo $! > "${LOG_PID_FILE}"
  echo "Log tail PID: $(cat "${LOG_PID_FILE}")"
}

docker_run_env_args() {
  local account_id="${1:-${PRACTICE_ACCOUNT_ID}}"
  local contract_id="${2:-${PRACTICE_CONTRACT_ID}}"
  local n_contracts="${3:-${ML_N_CONTRACTS}}"

  printf '%s\n' \
    "-e" "TOPSTEPX_USERNAME=${TOPSTEPX_USERNAME}" \
    "-e" "TOPSTEPX_PROJECTX_API_KEY=${TOPSTEPX_PROJECTX_API_KEY}" \
    "-e" "TOPSTEPX_ACCOUNT_ID=${account_id}" \
    "-e" "TOPSTEPX_CONTRACT_ID=${contract_id}" \
    "-e" "ML_N_CONTRACTS=${n_contracts}" \
    "-e" "ML_STDOUT_ONLY=1"

  if [[ -n "${TOPSTEPX_PROJECTX_BASE_URL:-}" ]]; then
    printf '%s\n' "-e" "TOPSTEPX_PROJECTX_BASE_URL=${TOPSTEPX_PROJECTX_BASE_URL}"
  fi
  if [[ -n "${TOPSTEPX_SESSION_TOKEN:-}" ]]; then
    printf '%s\n' "-e" "TOPSTEPX_SESSION_TOKEN=${TOPSTEPX_SESSION_TOKEN}"
  fi
}
