#!/bin/bash
# Create a dedicated VM for the MNQ liquidity sweep sidecar.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-us-central1}"
SWEEP_ZONE="${SWEEP_ZONE:-${ZONE:-${REGION}-a}}"
SWEEP_VM_NAME="${SWEEP_VM_NAME:-topstep-sweep-mnq-vm}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-medium}"
BOOT_DISK_GB="${BOOT_DISK_GB:-20}"
SWEEP_ZONE_FALLBACKS="${SWEEP_ZONE_FALLBACKS:-${REGION}-b ${REGION}-c ${REGION}-f}"

if gcloud compute instances describe "${SWEEP_VM_NAME}" --zone="${SWEEP_ZONE}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "Sweep VM '${SWEEP_VM_NAME}' already exists in ${SWEEP_ZONE}."
  exit 0
fi

if gcloud compute instances list --project="${PROJECT_ID}" --filter="name=${SWEEP_VM_NAME}" --format="value(zone)" 2>/dev/null | grep -q .; then
  z="$(gcloud compute instances list --project="${PROJECT_ID}" --filter="name=${SWEEP_VM_NAME}" --format="value(zone)" | head -1)"
  echo "Sweep VM '${SWEEP_VM_NAME}' already exists in zone ${z}."
  exit 0
fi

try_create() {
  local z="$1"
  echo "Trying zone ${z}..."
  gcloud compute instances create "${SWEEP_VM_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${z}" \
    --machine-type="${MACHINE_TYPE}" \
    --boot-disk-size="${BOOT_DISK_GB}GB" \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --tags=trading \
    --metadata-from-file=startup-script="${SCRIPT_DIR}/orb_vm_startup.sh"
}

if try_create "${SWEEP_ZONE}"; then
  echo "Created ${SWEEP_VM_NAME} in ${SWEEP_ZONE}."
  exit 0
fi

for fz in ${SWEEP_ZONE_FALLBACKS}; do
  [[ "${fz}" == "${SWEEP_ZONE}" ]] && continue
  if try_create "${fz}"; then
    echo "Created ${SWEEP_VM_NAME} in ${fz}. Set SWEEP_ZONE=${fz}."
    exit 0
  fi
done

echo "Could not create VM in ${SWEEP_ZONE} or fallbacks: ${SWEEP_ZONE_FALLBACKS}" >&2
exit 1
