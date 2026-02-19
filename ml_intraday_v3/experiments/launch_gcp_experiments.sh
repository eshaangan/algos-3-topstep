#!/bin/bash
# Launch GCP experiments for all batches
# Usage: ./launch_gcp_experiments.sh [batch1|batch2|batch3|all]

set -e

# Resolve paths relative to this script so it works from any CWD
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARTUP_SCRIPT="${SCRIPT_DIR}/gcp_startup_v2.sh"
if [ ! -f "$STARTUP_SCRIPT" ]; then
    echo "ERROR: Startup script not found: $STARTUP_SCRIPT"
    exit 1
fi

# Configuration
PROJECT_ID="trading-algo-3"
BUCKET="gs://trading-algo-3"
ZONE="us-central1-a"
MACHINE_TYPE="n1-highmem-4"

BATCH_FILTER="${1:-all}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== GCP Experiment Launcher ===${NC}"
echo "Batch filter: $BATCH_FILTER"
echo "Project: $PROJECT_ID"
echo "Bucket: $BUCKET"
echo "Zone: $ZONE"
echo "Machine: $MACHINE_TYPE"
echo ""

# Verify gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}ERROR: gcloud CLI not found. Install from https://cloud.google.com/sdk/install${NC}"
    exit 1
fi

# Verify gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo -e "${RED}ERROR: gsutil not found. Install gcloud SDK.${NC}"
    exit 1
fi

# Set project
gcloud config set project $PROJECT_ID

# Function to upload configs to GCS
upload_batch_configs() {
    local batch_name=$1
    local config_dir="ml_intraday_v3/experiments/${batch_name}_configs"

    if [ ! -d "$config_dir" ]; then
        echo -e "${RED}ERROR: Config directory not found: $config_dir${NC}"
        return 1
    fi

    local num_configs=$(ls $config_dir/batch*.json 2>/dev/null | wc -l | tr -d ' ')

    if [ "$num_configs" -eq 0 ]; then
        echo -e "${RED}ERROR: No configs found in $config_dir${NC}"
        return 1
    fi

    echo -e "${YELLOW}Uploading $num_configs configs for $batch_name...${NC}"
    gsutil -m cp $config_dir/batch*.json ${BUCKET}/experiments/${batch_name}_configs/

    echo -e "${GREEN}✓ Uploaded $num_configs configs for $batch_name${NC}"
}

# Function to launch VM for batch
launch_batch_vm() {
    local batch_name=$1
    local vm_name="ml-${batch_name}-vm"

    echo -e "${YELLOW}Launching VM: $vm_name${NC}"

    # Create VM instance
    gcloud compute instances create $vm_name \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=50GB \
        --boot-disk-type=pd-standard \
        --preemptible \
        --scopes=cloud-platform \
        --metadata=batch=$batch_name,bucket=$BUCKET \
        --metadata-from-file=startup-script="$STARTUP_SCRIPT" \
        --tags=http-server,https-server

    echo -e "${GREEN}✓ Launched VM: $vm_name${NC}"
    echo "  Monitor: gcloud compute ssh $vm_name --zone=$ZONE --command='tail -f /var/log/syslog'"
    echo "  Stop: gcloud compute instances stop $vm_name --zone=$ZONE"
    echo "  Delete: gcloud compute instances delete $vm_name --zone=$ZONE"
    echo ""
}

# Main execution
echo -e "${YELLOW}Step 1: Uploading experiment configs to GCS${NC}"
echo ""

if [ "$BATCH_FILTER" == "all" ] || [ "$BATCH_FILTER" == "batch1" ]; then
    upload_batch_configs "batch1"
fi

if [ "$BATCH_FILTER" == "all" ] || [ "$BATCH_FILTER" == "batch2" ]; then
    upload_batch_configs "batch2"
fi

if [ "$BATCH_FILTER" == "all" ] || [ "$BATCH_FILTER" == "batch3" ]; then
    upload_batch_configs "batch3"
fi

echo ""
echo -e "${YELLOW}Step 2: Uploading code snapshot to GCS${NC}"
echo ""

# Create code snapshot (lightweight - only Python files)
echo "Creating code snapshot..."
SNAPSHOT_FILE="/tmp/ml_intraday_v3_snapshot.tar.gz"

tar -czf $SNAPSHOT_FILE \
    ml_intraday_v3/*.py \
    ml_intraday_v3/**/*.py \
    ml_intraday_v3/configs/*.yaml \
    --exclude='ml_intraday_v3/data/*' \
    --exclude='ml_intraday_v3/.ipynb_checkpoints/*' \
    --exclude='ml_intraday_v3/__pycache__/*' \
    --exclude='ml_intraday_v3/**/__pycache__/*'

echo "Uploading code snapshot..."
gsutil cp $SNAPSHOT_FILE ${BUCKET}/code-snapshots/ml_intraday_v3_snapshot.tar.gz

echo -e "${GREEN}✓ Code snapshot uploaded${NC}"
echo ""

# Upload data files (if not already uploaded)
echo -e "${YELLOW}Step 3: Checking data files in GCS${NC}"
echo ""

DATA_COUNT=$(gsutil ls ${BUCKET}/experiment-data/*.parquet 2>/dev/null | wc -l | tr -d ' ')

if [ "$DATA_COUNT" -eq 0 ]; then
    echo "No data files found in GCS. Upload your training data:"
    echo "  gsutil cp data/processed/*.parquet ${BUCKET}/experiment-data/"
    echo ""
    read -p "Have you uploaded data files? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborting. Please upload data first.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Found $DATA_COUNT data files in GCS${NC}"
fi

echo ""

# Launch VMs
echo -e "${YELLOW}Step 4: Launching GCP VMs${NC}"
echo ""

read -p "Ready to launch VMs? This will incur GCP costs. (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Aborted by user${NC}"
    exit 0
fi

if [ "$BATCH_FILTER" == "all" ] || [ "$BATCH_FILTER" == "batch1" ]; then
    launch_batch_vm "batch1"
fi

if [ "$BATCH_FILTER" == "all" ] || [ "$BATCH_FILTER" == "batch2" ]; then
    launch_batch_vm "batch2"
fi

if [ "$BATCH_FILTER" == "all" ] || [ "$BATCH_FILTER" == "batch3" ]; then
    launch_batch_vm "batch3"
fi

echo ""
echo -e "${GREEN}=== Launch Complete ===${NC}"
echo ""
echo "Monitor progress:"
echo "  gsutil ls ${BUCKET}/experiment-results/batch*/*.json | wc -l"
echo ""
echo "View logs:"
echo "  gcloud compute ssh ml-batch1-vm --zone=$ZONE --command='tail -f /var/log/syslog'"
echo ""
echo "Stop all VMs:"
echo "  gcloud compute instances stop ml-batch1-vm ml-batch2-vm ml-batch3-vm --zone=$ZONE"
echo ""
echo "Delete all VMs:"
echo "  gcloud compute instances delete ml-batch1-vm ml-batch2-vm ml-batch3-vm --zone=$ZONE --quiet"
echo ""
echo "Download results:"
echo "  gsutil -m cp -r ${BUCKET}/experiment-results/batch* ml_intraday_v3/experiments/results/"
echo ""
echo -e "${YELLOW}Estimated cost: ~\$1.20 for 600 experiments${NC}"
echo ""
