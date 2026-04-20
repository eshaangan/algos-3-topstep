#!/bin/bash
# Dual-meta MES image -> VM topstep-dual-meta-mes-vm (default). Does not touch ORB VM.
# Second instance: gcp_deploy/provision_orb_vm.sh + deploy_orb.sh (see gcp_deploy/BOTH_VMS.txt).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE_PATH="gcp_deploy/Dockerfile"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-${REGION}-a}"
APP_NAME="${APP_NAME:-topstep-dual-meta-mes}"
VM_NAME="${VM_NAME:-${APP_NAME}-vm}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-medium}"
IMAGE_TAG="${IMAGE_TAG:-gcr.io/${PROJECT_ID}/${APP_NAME}:latest}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env.gcp.live}"

if [[ ! -f "${ENV_FILE}" && -f "${REPO_ROOT}/.env" ]]; then
  ENV_FILE="${REPO_ROOT}/.env"
fi

echo "Deploying ${APP_NAME} to ${PROJECT_ID} (${ZONE})"

echo "Building image with Cloud Build..."
TMP_BUILD_CONFIG="$(mktemp)"
trap 'rm -f "${TMP_BUILD_CONFIG}"' EXIT
cat > "${TMP_BUILD_CONFIG}" <<EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-f'
      - '${DOCKERFILE_PATH}'
      - '-t'
      - '${IMAGE_TAG}'
      - '.'
images:
  - '${IMAGE_TAG}'
EOF
gcloud builds submit "${REPO_ROOT}" --project="${PROJECT_ID}" --config="${TMP_BUILD_CONFIG}"

ENV_FLAGS=()
if [[ -f "${ENV_FILE}" ]]; then
  echo "Reading container env from ${ENV_FILE}"
  while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    line="${raw_line%%#*}"
    line="$(echo "${line}" | xargs)"
    [[ -z "${line}" ]] && continue
    if [[ "${line}" == *"="* ]]; then
      key="${line%%=*}"
      value="${line#*=}"
      key="$(echo "${key}" | xargs)"
      value="$(echo "${value}" | xargs)"
      if [[ -z "${key}" ]]; then
        continue
      fi
      if [[ ! "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        echo "Skipping invalid container env key: ${key}" >&2
        continue
      fi
      ENV_FLAGS+=("--container-env=${key}=${value}")
    fi
  done < "${ENV_FILE}"
else
  echo "WARNING: no env file found; continuing without container env overrides"
fi

if gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" >/dev/null 2>&1; then
  echo "Updating existing VM container..."
  gcloud compute instances update-container "${VM_NAME}" \
    --zone="${ZONE}" \
    --container-image="${IMAGE_TAG}" \
    "${ENV_FLAGS[@]}"
else
  echo "Creating new VM ${VM_NAME}..."
  gcloud compute instances create-with-container "${VM_NAME}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --tags=trading \
    --container-image="${IMAGE_TAG}" \
    "${ENV_FLAGS[@]}"
fi

echo "Deployment complete."
echo "If ${VM_NAME} runs Docker manually (update-container not used), recreate the runner:"
echo "  bash gcp_deploy/recreate_dual_meta_docker_on_vm.sh"
echo "Logs: gcloud compute ssh ${VM_NAME} --zone=${ZONE}"
