#!/bin/bash
# GCP VM startup script for grid search experiments
set -e

echo "=== Grid Search Worker VM Startup ==="
date

# Install system dependencies
echo "Installing system packages..."
apt-get update -y
apt-get install -y python3.11 python3-pip git

# Get metadata
BATCH_FILE=$(curl -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/batch-file)
PHASE=$(curl -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/phase)

echo "Batch file: $BATCH_FILE"
echo "Phase: $PHASE"

# Create working directory
mkdir -p /workspace
cd /workspace

# Download code from GCS (assumes code has been uploaded)
echo "Downloading code..."
gsutil -m cp -r gs://trading-algo-3/code/ml_intraday_v3 /workspace/

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install numpy pandas scikit-learn lightgbm joblib pyyaml matplotlib seaborn pyarrow

# Download data from GCS
echo "Downloading data..."
mkdir -p /data
gsutil -m cp gs://trading-algo-3/experiment-data/*.parquet /data/

# Download batch file
echo "Downloading batch file..."
gsutil cp $BATCH_FILE /tmp/batch.json

# Count experiments in batch
NUM_EXPERIMENTS=$(python3 -c "import json; f=open('/tmp/batch.json'); print(len(json.load(f)))")
echo "Running $NUM_EXPERIMENTS experiments..."

# Run experiments
cd /workspace/ml_intraday_v3/experiments

for i in $(seq 0 $((NUM_EXPERIMENTS - 1))); do
    echo "=== Experiment $((i + 1))/$NUM_EXPERIMENTS ==="
    
    # Extract config for this experiment
    python3 -c "
import json
with open('/tmp/batch.json', 'r') as f:
    batch = json.load(f)
with open('/tmp/exp_config.json', 'w') as f:
    json.dump(batch[$i], f, indent=2)
"
    
    # Run experiment
    EXP_ID=$(python3 -c "import json; f=open('/tmp/exp_config.json'); print(json.load(f)['exp_id'])")
    OUTPUT_FILE="/tmp/result_${EXP_ID}.json"

    python3 comprehensive_grid_search_v2.py \
        --config /tmp/exp_config.json \
        --data-dir /data \
        --config-dir /workspace/ml_intraday_v3/configs \
        --output $OUTPUT_FILE
    
    # Upload result to GCS
    gsutil cp $OUTPUT_FILE gs://trading-algo-3/experiment-results/phase${PHASE}/
    
    echo "Completed and uploaded: $EXP_ID"
done

echo "=== All experiments complete ==="
date

# Shutdown VM to save costs
echo "Shutting down..."
shutdown -h now
