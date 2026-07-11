#!/bin/bash
# Pull recorded L2 parquet from the remote Mac to local data/l2_raw/ for analysis.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../gcp_deploy/remote_pc/common.sh
source "${REPO_ROOT}/gcp_deploy/remote_pc/common.sh"

read_remote_session
if [[ -z "${REMOTE_DIR:-}" ]]; then
  echo "No remote session found." >&2; exit 1
fi

DEST="${REPO_ROOT}/data/l2_raw"
mkdir -p "${DEST}"
echo "Pulling ${SSH_HOST}:${REMOTE_DIR}/l2_raw/ → ${DEST}/ …"
# -a archive, -z compress, --ignore-existing so re-pulls are incremental
rsync -az --ignore-existing "${SSH_HOST}:${REMOTE_DIR}/l2_raw/" "${DEST}/" || {
  echo "rsync unavailable, falling back to scp"; scp -q "${SSH_HOST}:${REMOTE_DIR}/l2_raw/*.parquet" "${DEST}/" 2>/dev/null || true
}
echo "Local L2 files: $(ls -1 "${DEST}"/*.parquet 2>/dev/null | wc -l | tr -d ' ')"
du -sh "${DEST}" 2>/dev/null || true
echo "Now run: python rule_based_v1/validation/research_l2.py"
