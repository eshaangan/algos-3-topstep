#!/bin/bash
# Cloud Build ML scalper image and deploy to topstep-vwap-vm as second container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
ML_IMAGE="${ML_IMAGE:-gcr.io/${PROJECT_ID}/topstep-ml:latest}"
VM_NAME="${VM_NAME:-topstep-vwap-vm}"
VM_ZONE="${VM_ZONE:-us-central1-a}"

echo "========================================="
echo "Building ML runner image -> ${ML_IMAGE}"
echo "Target VM: ${VM_NAME} (${VM_ZONE})"
echo "========================================="
echo ""

# 1. Build & push via Cloud Build
TMP="$(mktemp)"
trap 'rm -f "${TMP}"' EXIT
cat > "${TMP}" <<EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-f'
      - 'gcp_deploy/Dockerfile.ml'
      - '-t'
      - '${ML_IMAGE}'
      - '.'
images:
  - '${ML_IMAGE}'
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
echo "  docker pull ${ML_IMAGE}"
echo ""
echo "  # Stop old ML container if running"
echo "  docker stop ml-live 2>/dev/null || true"
echo "  docker rm   ml-live 2>/dev/null || true"
echo ""
echo "  # Env file: /home/eshaanganguly/orb_env.txt already contains all needed vars."
echo "  # Optional: ML_N_CONTRACTS=15 can be set there to override default."
echo ""
echo "  # Dry-run — logs signals only, zero real orders:"
echo "  docker run -d --name ml-live --restart always \\"
echo "    --env-file /home/eshaanganguly/orb_env.txt \\"
echo "    ${ML_IMAGE} --dry-run -v"
echo ""
echo "  # Practice account (real API orders on SIM/paper account — set correct account ID first):"
echo "  # 1. Discover your practice account ID:"
echo "  #    python ml_intraday_v3/live/discover_accounts.py"
echo "  # 2. Add to /home/eshaanganguly/orb_env.txt:"
echo "  #    TOPSTEPX_ACCOUNT_ID=<practice_account_id>"
echo "  #    ML_N_CONTRACTS=15"
echo "  docker run -d --name ml-live --restart always \\"
echo "    --env-file /home/eshaanganguly/orb_env.txt \\"
echo "    ${ML_IMAGE} --live --yes --n-contracts 15 -v"
echo ""
echo "  # Live funded account (after passing Combine — use funded account ID):"
echo "  docker run -d --name ml-live --restart always \\"
echo "    --env-file /home/eshaanganguly/orb_env.txt \\"
echo "    ${ML_IMAGE} --live --yes --n-contracts 15 -v"
echo ""
echo "  # Tail logs"
echo "  docker logs -f ml-live"
