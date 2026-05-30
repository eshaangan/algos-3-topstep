#!/bin/bash
# Cloud Build ML strategy image and deploy to topstep-vwap-vm as a container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
ML_STRATEGY_IMAGE="${ML_STRATEGY_IMAGE:-gcr.io/${PROJECT_ID}/topstep-ml-strategy:latest}"
VM_NAME="${VM_NAME:-topstep-vwap-vm}"
VM_ZONE="${VM_ZONE:-us-central1-a}"
CONTAINER_NAME="${CONTAINER_NAME:-ml-strategy-live}"

echo "========================================="
echo "Building ML strategy image -> ${ML_STRATEGY_IMAGE}"
echo "Target VM: ${VM_NAME} (${VM_ZONE})"
echo "========================================="
echo ""

# Export model locally before build (bundle must exist in rule_based_v1/models/)
if [[ ! -f "${REPO_ROOT}/rule_based_v1/models/ml_strategy_mnq_v1.pkl" ]]; then
  echo "Model bundle missing — running export..."
  python "${REPO_ROOT}/rule_based_v1/diagnostics/ml_strategy_search.py" --export
fi

TMP="$(mktemp)"
trap 'rm -f "${TMP}"' EXIT
cat > "${TMP}" <<EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-f'
      - 'gcp_deploy/Dockerfile.ml_strategy'
      - '-t'
      - '${ML_STRATEGY_IMAGE}'
      - '.'
images:
  - '${ML_STRATEGY_IMAGE}'
EOF
gcloud builds submit "${REPO_ROOT}" --project="${PROJECT_ID}" --config="${TMP}"

echo ""
echo "Build complete. To deploy on ${VM_NAME}:"
echo ""
echo "  gcloud compute ssh ${VM_NAME} --zone=${VM_ZONE} --project=${PROJECT_ID}"
echo ""
echo "  # Pull new image"
echo "  TOKEN=\$(curl -sf -H 'Metadata-Flavor: Google' \\"
echo "    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \\"
echo "    | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"access_token\"])')"
echo "  echo \"\$TOKEN\" | docker login -u oauth2accesstoken --password-stdin https://gcr.io"
echo "  docker pull ${ML_STRATEGY_IMAGE}"
echo ""
echo "  # Stop old container if running"
echo "  docker stop ${CONTAINER_NAME} 2>/dev/null || true"
echo "  docker rm   ${CONTAINER_NAME} 2>/dev/null || true"
echo ""
echo "  # Env file: /home/eshaanganguly/orb_env.txt (TOPSTEPX_* vars + ML_N_CONTRACTS=6)"
echo ""
echo "  # Dry-run — logs signals only, zero real orders:"
echo "  docker run -d --name ${CONTAINER_NAME} --restart always \\"
echo "    --env-file /home/eshaanganguly/orb_env.txt \\"
echo "    ${ML_STRATEGY_IMAGE} --dry-run -v"
echo ""
echo "  # 150k combine practice (set TOPSTEPX_ACCOUNT_ID to combine account):"
echo "  docker run -d --name ${CONTAINER_NAME} --restart always \\"
echo "    --env-file /home/eshaanganguly/orb_env.txt \\"
echo "    -e ML_N_CONTRACTS=6 \\"
echo "    ${ML_STRATEGY_IMAGE} --live --yes -v"
echo ""
echo "  # Tail logs"
echo "  docker logs -f ${CONTAINER_NAME}"
