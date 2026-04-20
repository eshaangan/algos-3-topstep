#!/bin/bash
# Create a second GCE VM for ORB so it runs alongside the dual-meta MES VM (separate machine).
# Safe to re-run: skips creation if the instance already exists.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project)}"
REGION="${REGION:-us-central1}"
# Prefer ORB_ZONE; else same -a as MES VM; if create fails (capacity), set ORB_ZONE=us-central1-b etc.
ORB_ZONE="${ORB_ZONE:-${ZONE:-${REGION}-a}}"
ORB_VM_NAME="${ORB_VM_NAME:-topstep-trading-vm}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-medium}"
BOOT_DISK_GB="${BOOT_DISK_GB:-20}"
# Space-separated fallbacks if primary zone returns RESOURCE_POOL_EXHAUSTED
ORB_ZONE_FALLBACKS="${ORB_ZONE_FALLBACKS:-${REGION}-b ${REGION}-c ${REGION}-f}"

if gcloud compute instances describe "${ORB_VM_NAME}" --zone="${ORB_ZONE}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "ORB VM '${ORB_VM_NAME}' already exists in ${ORB_ZONE} — nothing to create."
  exit 0
fi

# If VM exists in another zone, gcloud describe without zone fails — list by name
if gcloud compute instances list --project="${PROJECT_ID}" --filter="name=${ORB_VM_NAME}" --format="value(zone)" 2>/dev/null | grep -q .; then
  z="$(gcloud compute instances list --project="${PROJECT_ID}" --filter="name=${ORB_VM_NAME}" --format="value(zone)" | head -1)"
  echo "ORB VM '${ORB_VM_NAME}' already exists in zone ${z} — nothing to create."
  exit 0
fi

try_create() {
  local z="$1"
  echo "Trying zone ${z}..."
  gcloud compute instances create "${ORB_VM_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${z}" \
    --machine-type="${MACHINE_TYPE}" \
    --boot-disk-size="${BOOT_DISK_GB}GB" \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --tags=trading \
    --metadata-from-file=startup-script="${SCRIPT_DIR}/orb_vm_startup.sh"
}

if try_create "${ORB_ZONE}"; then
  echo "Done. Wait ~1–2 minutes for startup-script (Docker install), then:"
  echo "  gcloud compute ssh ${ORB_VM_NAME} --zone=${ORB_ZONE} --project=${PROJECT_ID}"
  exit 0
fi

for fz in ${ORB_ZONE_FALLBACKS}; do
  [[ "${fz}" == "${ORB_ZONE}" ]] && continue
  echo "Primary zone failed; retrying fallback ${fz}..."
  if try_create "${fz}"; then
    echo "Created in ${fz} (not ${ORB_ZONE}). Use:"
    echo "  gcloud compute ssh ${ORB_VM_NAME} --zone=${fz} --project=${PROJECT_ID}"
    echo "Set ORB_ZONE=${fz} when following deploy_orb.sh SSH hints."
    exit 0
  fi
done

echo "Could not create VM in ${ORB_ZONE} or fallbacks: ${ORB_ZONE_FALLBACKS}" >&2
exit 1
