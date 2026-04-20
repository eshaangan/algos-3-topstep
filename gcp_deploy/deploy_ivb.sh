#!/bin/bash
# Build and deploy IVB ORB image to a new VM (topstep-ivb-mes-vm).
# Does NOT touch topstep-trading-vm (MNQ ORB — stays running).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-us-central1}"
IVB_ZONE="${IVB_ZONE:-${REGION}-a}"
IVB_VM_NAME="${IVB_VM_NAME:-topstep-ivb-mes-vm}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-medium}"
BOOT_DISK_GB="${BOOT_DISK_GB:-20}"
IVB_IMAGE="${IVB_IMAGE:-gcr.io/${PROJECT_ID}/topstep-ivb:latest}"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env.gcp.live}"

if [[ ! -f "${ENV_FILE}" && -f "${REPO_ROOT}/.env" ]]; then
  ENV_FILE="${REPO_ROOT}/.env"
fi

echo "=== IVB ORB Deployment ==="
echo "Project : ${PROJECT_ID}"
echo "Zone    : ${IVB_ZONE}"
echo "VM      : ${IVB_VM_NAME}"
echo "Image   : ${IVB_IMAGE}"
echo "ORB VM  : topstep-trading-vm (untouched)"
echo ""

# --- Step 1: Build image ---
echo "Building IVB image via Cloud Build..."
TMP_BUILD_CONFIG="$(mktemp)"
trap 'rm -f "${TMP_BUILD_CONFIG}"' EXIT
cat > "${TMP_BUILD_CONFIG}" <<EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-f'
      - 'gcp_deploy/Dockerfile.ivb'
      - '-t'
      - '${IVB_IMAGE}'
      - '.'
images:
  - '${IVB_IMAGE}'
EOF
gcloud builds submit "${REPO_ROOT}" --project="${PROJECT_ID}" --config="${TMP_BUILD_CONFIG}"
echo "Image built: ${IVB_IMAGE}"

# --- Step 2: Provision VM if it doesn't exist ---
if ! gcloud compute instances list \
    --project="${PROJECT_ID}" \
    --filter="name=${IVB_VM_NAME}" \
    --format="value(name)" 2>/dev/null | grep -q "${IVB_VM_NAME}"; then
  echo "Provisioning new VM ${IVB_VM_NAME} in ${IVB_ZONE}..."
  gcloud compute instances create "${IVB_VM_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${IVB_ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --boot-disk-size="${BOOT_DISK_GB}GB" \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --tags=trading \
    --metadata-from-file=startup-script="${SCRIPT_DIR}/orb_vm_startup.sh"
  echo "VM created. Waiting 90s for Docker install via startup-script..."
  sleep 90
else
  echo "VM ${IVB_VM_NAME} already exists — skipping creation."
fi

# --- Step 3: GCR auth + pull + run on VM ---
echo ""
echo "=== Run the following on ${IVB_VM_NAME} ==="
echo ""
echo "# 1. SSH in:"
echo "gcloud compute ssh ${IVB_VM_NAME} --zone=${IVB_ZONE} --project=${PROJECT_ID}"
echo ""
echo "# 2. Authenticate with GCR:"
echo 'TOKEN=$(curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | python3 -c "import sys,json; print(json.load(sys.stdin)['"'"'access_token'"'"'])")'
echo 'echo "$TOKEN" | docker login -u oauth2accesstoken --password-stdin https://gcr.io'
echo ""
echo "# 3. Pull image:"
echo "docker pull ${IVB_IMAGE}"
echo ""
echo "# 4. Create env file (edit values):"
echo "cat > ~/ivb_env.txt <<'ENVEOF'"
echo "TOPSTEPX_USERNAME=your_email@example.com"
echo "TOPSTEPX_PROJECTX_API_KEY=your_api_key"
echo "TOPSTEPX_ACCOUNT_ID=your_combine_account_id"
echo "TOPSTEPX_CONTRACT_ID=CON.F.US.MES.M25"
echo "ENVEOF"
echo ""
echo "# 5. Paper trade first (default CMD is --dry-run):"
echo "docker stop ivb-live 2>/dev/null || true; docker rm ivb-live 2>/dev/null || true"
echo "docker run -d --name ivb-live --restart always --env-file ~/ivb_env.txt ${IVB_IMAGE}"
echo ""
echo "# 6. Verify logs:"
echo "docker logs -f ivb-live"
echo ""
echo "# 7. When ready for live — MES 3 contracts:"
echo "docker stop ivb-live && docker rm ivb-live"
echo "docker run -d --name ivb-live --restart always --env-file ~/ivb_env.txt ${IVB_IMAGE} \\"
echo "  --live --yes --config-dir /app/rule_based_v1/configs/vm_ivb_mes -v"
echo ""
echo "NOTE: topstep-trading-vm (MNQ ORB) is untouched and still running."
