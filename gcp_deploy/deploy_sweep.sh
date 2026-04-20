#!/bin/bash
# Cloud Build + run MNQ liquidity sweep sidecar on a dedicated VM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE_PATH="gcp_deploy/Dockerfile.sweep"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-us-central1}"
SWEEP_VM_NAME="${SWEEP_VM_NAME:-topstep-sweep-mnq-vm}"
SWEEP_ZONE="${SWEEP_ZONE:-us-central1-a}"
SWEEP_IMAGE="${SWEEP_IMAGE:-gcr.io/${PROJECT_ID}/topstep-sweep-mnq:latest}"

echo "Cloud Build sweep image -> ${SWEEP_IMAGE}"
echo "Sweep VM target: ${SWEEP_VM_NAME} (${SWEEP_ZONE})"

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
      - '${SWEEP_IMAGE}'
      - '.'
images:
  - '${SWEEP_IMAGE}'
EOF
gcloud builds submit "${REPO_ROOT}" --project="${PROJECT_ID}" --config="${TMP_BUILD_CONFIG}"

echo "Build complete."
cat <<EOF
Provision VM if needed:
  PROJECT_ID=${PROJECT_ID} SWEEP_VM_NAME=${SWEEP_VM_NAME} SWEEP_ZONE=${SWEEP_ZONE} bash gcp_deploy/provision_sweep_vm.sh

Run on VM:
  gcloud compute ssh ${SWEEP_VM_NAME} --zone=${SWEEP_ZONE} --project=${PROJECT_ID}
  TOKEN_JSON=\$(curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token")
  ACCESS_TOKEN=\$(echo "\$TOKEN_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')
  echo "\$ACCESS_TOKEN" | docker login -u oauth2accesstoken --password-stdin https://gcr.io
  docker pull ${SWEEP_IMAGE}
  docker stop sweep-live 2>/dev/null || true
  docker rm sweep-live 2>/dev/null || true
  docker run -d --name sweep-live --restart always --env-file ~/sweep_env.txt ${SWEEP_IMAGE} \\
    --live --yes --config-dir /app/rule_based_v1/configs/vm_sweep_mnq -v
EOF
