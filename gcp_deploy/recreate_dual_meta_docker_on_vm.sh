#!/bin/bash
# Recreate dual-meta live on topstep-dual-meta-mes-vm using Docker (not update-container).
# Uses ~/topstep_dual_meta.env on the VM (TOPSTEPX_ACCOUNT_ID and other secrets). Refresh it with:
#   docker inspect dual-meta-live --format '{{range .Config.Env}}{{println .}}{{end}}' > ~/topstep_dual_meta.env
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-topstep-dual-meta-mes-vm}"
CONTAINER_NAME="${CONTAINER_NAME:-dual-meta-live}"
IMAGE_TAG="${IMAGE_TAG:-gcr.io/${PROJECT_ID}/topstep-dual-meta-mes:latest}"

echo "SSH ${VM_NAME} (${ZONE}) — recreate ${CONTAINER_NAME} with ${IMAGE_TAG}"

gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --command="bash -s" <<EOF
set -euo pipefail
IMAGE=$(printf '%q' "${IMAGE_TAG}")
CONTAINER=$(printf '%q' "${CONTAINER_NAME}")
ENV_FILE="\${HOME}/topstep_dual_meta.env"
if [[ ! -f "\${ENV_FILE}" ]]; then
  echo "Missing \${ENV_FILE}. Create it from a good container or copy env vars there." >&2
  exit 1
fi
echo "Pulling \${IMAGE}..."
docker pull "\${IMAGE}"
if docker inspect "\${CONTAINER}" >/dev/null 2>&1; then
  docker stop "\${CONTAINER}" || true
  docker rm "\${CONTAINER}" || true
fi
echo "Starting \${CONTAINER}..."
docker run -d \\
  --name "\${CONTAINER}" \\
  --restart always \\
  --env-file "\${ENV_FILE}" \\
  "\${IMAGE}" \\
  --config-dir /app/ml_intraday_v3/configs/live_dual_meta_mes_real \\
  --model-bundle /app/ml_intraday_v3/models/live/dual_meta_mes_live_bundle.pkl \\
  --no-confirm \\
  --log-level INFO
echo "TOPSTEPX_ACCOUNT_ID in container:"
docker exec "\${CONTAINER}" printenv TOPSTEPX_ACCOUNT_ID
EOF
