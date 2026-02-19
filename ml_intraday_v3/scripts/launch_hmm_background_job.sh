#!/bin/bash
#
# Launch HMM feature pre-computation as a background GCP job.
#
# This creates a CHEAP, non-preemptible VM that runs HMM computation in the background.
# The VM will run for ~5-10 hours, then auto-shutdown.
#
# Cost: ~$0.50 total (e2-highcpu-4 @ $0.10/hour × 5 hours)
# Runtime: 5-10 hours (one-time)
# Output: MES_5min_Oct2024_Dec2025_with_hmm.parquet (reusable forever)

set -e

PROJECT_ID="trading-algo-3"
ZONE="us-central1-a"
VM_NAME="hmm-precompute-job"
BUCKET="gs://trading-algo-3"

echo "=========================================="
echo "Launching HMM Background Pre-computation"
echo "=========================================="
echo ""
echo "VM: $VM_NAME"
echo "Machine: e2-highcpu-4 (cheap, non-preemptible)"
echo "Estimated cost: \$0.50"
echo "Estimated time: 5-10 hours"
echo ""

# Create the VM
gcloud compute instances create $VM_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --machine-type=e2-highcpu-4 \
  --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
  --maintenance-policy=MIGRATE \
  --provisioning-model=STANDARD \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --create-disk=auto-delete=yes,boot=yes,device-name=$VM_NAME,image=projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20260210,mode=rw,size=30,type=pd-balanced \
  --no-shielded-secure-boot \
  --shielded-vtpm \
  --shielded-integrity-monitoring \
  --labels=job=hmm-precompute \
  --metadata=startup-script='#!/bin/bash
set -e

echo "=========================================="
echo "HMM Feature Pre-computation Job"
echo "=========================================="

# Update and install dependencies
apt-get update
apt-get install -y python3-pip wget

# Download code snapshot
cd /root
gsutil cp gs://trading-algo-3/ml_intraday_v3_snapshot.tar.gz .
tar -xzf ml_intraday_v3_snapshot.tar.gz
cd ml_intraday_v3

# Install requirements (including hmmlearn)
pip3 install lightgbm pandas numpy scikit-learn pyarrow h5py pyyaml hmmlearn

# Download data
echo "Downloading data..."
gsutil cp gs://trading-algo-3/experiment-data/MES_5min_Oct2024_Dec2025.parquet data/processed/

# Run HMM pre-computation
echo "Starting HMM computation..."
echo "This will take 5-10 hours. Progress logged to hmm_precompute.log"

python3 scripts/precompute_hmm_features.py \
  --input data/processed/MES_5min_Oct2024_Dec2025.parquet \
  --output data/processed/MES_5min_Oct2024_Dec2025_with_hmm.parquet \
  --n-states 2 \
  2>&1 | tee hmm_precompute.log

# Upload results
echo "Uploading results to GCS..."
gsutil cp data/processed/MES_5min_Oct2024_Dec2025_with_hmm.parquet gs://trading-algo-3/cached-features/
gsutil cp hmm_precompute.log gs://trading-algo-3/logs/

echo "=========================================="
echo "HMM Pre-computation Complete!"
echo "=========================================="
echo "Output: gs://trading-algo-3/cached-features/MES_5min_Oct2024_Dec2025_with_hmm.parquet"
echo "Log: gs://trading-algo-3/logs/hmm_precompute.log"
echo ""
echo "File can now be used by all future experiments (instant HMM features)"
echo ""
echo "Shutting down..."
shutdown -h now
'

echo ""
echo "✅ HMM background job launched!"
echo ""
echo "To monitor progress:"
echo "  gcloud compute instances get-serial-port-output $VM_NAME --project=$PROJECT_ID --zone=$ZONE | tail -50"
echo ""
echo "To check if it's done:"
echo "  gsutil ls gs://trading-algo-3/cached-features/MES_5min_Oct2024_Dec2025_with_hmm.parquet"
echo ""
echo "The VM will auto-shutdown when complete (~5-10 hours)"
