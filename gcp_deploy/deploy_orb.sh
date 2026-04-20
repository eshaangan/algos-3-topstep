#!/bin/bash
# Cloud Build ORB image (topstep-orb:latest). ORB runs on a *separate* VM from dual-meta.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE_PATH="gcp_deploy/Dockerfile.orb"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-${REGION}-a}"
MES_VM_NAME="${MES_VM_NAME:-topstep-dual-meta-mes-vm}"
ORB_VM_NAME="${ORB_VM_NAME:-topstep-trading-vm}"
ORB_ZONE="${ORB_ZONE:-us-central1-b}"
ORB_IMAGE="${ORB_IMAGE:-gcr.io/${PROJECT_ID}/topstep-orb:latest}"

echo "Cloud Build ORB image -> ${ORB_IMAGE}"
echo "MES VM (unchanged by this script): ${MES_VM_NAME}"
echo "ORB VM (docker pull/run target):   ${ORB_VM_NAME} (zone: ${ORB_ZONE}; override ORB_ZONE if instance is elsewhere)"
echo ""

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
      - '${ORB_IMAGE}'
      - '.'
images:
  - '${ORB_IMAGE}'
EOF
gcloud builds submit "${REPO_ROOT}" --project="${PROJECT_ID}" --config="${TMP_BUILD_CONFIG}"

echo ""
echo "Build complete."
echo ""
echo "1) Ensure ORB VM exists (one-time):  PROJECT_ID=${PROJECT_ID} bash gcp_deploy/provision_orb_vm.sh"
echo ""
echo "2) Run ORB on ${ORB_VM_NAME} only (stop any orb-live on topstep-orb-vm to avoid duplicate orders):"
echo "   gcloud compute ssh topstep-orb-vm --zone=\${ORB_ZONE} --project=${PROJECT_ID} --command='docker stop orb-live 2>/dev/null; docker rm orb-live 2>/dev/null; true'"
echo ""
echo "3) On ${ORB_VM_NAME}, GCR login, pull, run:"
echo "   # Default ORB_ZONE is us-central1-b; set ORB_ZONE if your VM is in another zone"
echo "   gcloud compute ssh ${ORB_VM_NAME} --zone=\${ORB_ZONE} --project=${PROJECT_ID}"
echo "   TOKEN_JSON=\$(curl -sf -H \"Metadata-Flavor: Google\" \"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token\")"
echo "   ACCESS_TOKEN=\$(echo \"\$TOKEN_JSON\" | python3 -c \"import sys,json; print(json.load(sys.stdin)['access_token'])\")"
echo "   echo \"\$ACCESS_TOKEN\" | docker login -u oauth2accesstoken --password-stdin https://gcr.io"
echo "   docker pull ${ORB_IMAGE}"
echo "   docker stop orb-live 2>/dev/null || true; docker rm orb-live 2>/dev/null || true"
echo "   # /tmp/orb_env.txt — KEY=VALUE per line, no quotes (example):"
echo "   #   TOPSTEPX_USERNAME=..."
echo "   #   TOPSTEPX_PROJECTX_API_KEY=..."
echo "   #   TOPSTEPX_ACCOUNT_ID=12345"
echo "   #   TOPSTEPX_CONTRACT_ID=CON.F.US.MNQ.M25   # roll to current front month"
echo "   # Optional: TOPSTEPX_PROJECTX_BASE_URL=https://api.topstepx.com"
echo ""
echo "   # Paper (default image CMD — still needs API env for bar fetch):"
echo "   docker run -d --name orb-live --restart always --env-file /tmp/orb_env.txt ${ORB_IMAGE}"
echo ""
echo "   # Live ORB — MNQ (real orders — same config dir as Dockerfile):"
echo "   docker run -d --name orb-live --restart always --env-file /tmp/orb_env.txt ${ORB_IMAGE} \\"
echo "     --live --yes --config-dir /app/rule_based_v1/configs/vm_orb_sidecar -v"
echo ""
echo "   # Live ORB — Micro Nikkei (MNK): separate container name + MNK contract id in env"
echo "   #   TOPSTEPX_CONTRACT_ID=CON.F.US.MNK.H26   # example; use TopstepX contract search for current front"
echo "   docker run -d --name orb-mnk-live --restart always --env-file /tmp/orb_mnk_env.txt ${ORB_IMAGE} \\"
echo "     --live --yes --config-dir /app/rule_based_v1/configs/vm_orb_mnk -v"
