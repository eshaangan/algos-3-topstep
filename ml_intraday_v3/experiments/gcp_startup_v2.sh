#!/bin/bash
# GCP VM startup script v2 - runs all configs from GCS bucket
set -e

echo "=== ML Intraday V3 Grid Search Worker ==="
date

# Install system dependencies
echo "[1/8] Installing system packages..."
apt-get update -y
apt-get install -y python3.11 python3-pip git python3.11-venv

# Get metadata
BATCH=$(curl -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/batch 2>/dev/null || echo "batch1")
BUCKET=$(curl -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket 2>/dev/null || echo "gs://trading-algo-3")

echo "Batch: $BATCH"
echo "Bucket: $BUCKET"

# Create working directory
echo "[2/8] Setting up workspace..."
mkdir -p /workspace
cd /workspace

# Download code snapshot from GCS
echo "[3/8] Downloading code snapshot..."
gsutil cp ${BUCKET}/code-snapshots/ml_intraday_v3_snapshot.tar.gz /tmp/snapshot.tar.gz
tar -xzf /tmp/snapshot.tar.gz -C /workspace

# Create virtual environment
echo "[4/8] Creating Python environment..."
python3.11 -m venv /workspace/venv
source /workspace/venv/bin/activate

# Install Python dependencies
echo "[5/8] Installing Python dependencies..."
pip install --upgrade pip
pip install numpy pandas scikit-learn lightgbm joblib pyyaml matplotlib seaborn pyarrow hmmlearn statsmodels scipy

# Download data from GCS
echo "[6/8] Downloading data files..."
mkdir -p /data
gsutil -m cp ${BUCKET}/experiment-data/*.parquet /data/ || {
    echo "WARNING: No data files found in ${BUCKET}/experiment-data/"
    echo "Using sample data instead..."
}

# Download experiment configs
echo "[7/8] Downloading experiment configs..."
mkdir -p /workspace/configs
gsutil -m cp ${BUCKET}/experiments/${BATCH}_configs/batch*.json /workspace/configs/

# Count configs
NUM_CONFIGS=$(ls /workspace/configs/batch*.json 2>/dev/null | wc -l)
echo "Found $NUM_CONFIGS experiment configs to run"

if [ "$NUM_CONFIGS" -eq 0 ]; then
    echo "ERROR: No configs found!"
    shutdown -h now
    exit 1
fi

# Run all experiments
echo "[8/8] Running experiments..."
cd /workspace

SUCCESS_COUNT=0
FAIL_COUNT=0

for config_file in /workspace/configs/batch*.json; do
    EXP_ID=$(python3 -c "import json; print(json.load(open('$config_file'))['exp_id'])")
    OUTPUT_FILE="/tmp/result_${EXP_ID}.json"

    echo ""
    echo "=================================================="
    echo "Running: $EXP_ID ($((SUCCESS_COUNT + FAIL_COUNT + 1))/$NUM_CONFIGS)"
    echo "=================================================="
    echo "Config: $config_file"
    echo "Output: $OUTPUT_FILE"
    echo ""

    # Run experiment with error handling
    if python3 ml_intraday_v3/experiments/comprehensive_grid_search_v2.py \
        --config "$config_file" \
        --data-dir /data \
        --config-dir /workspace/ml_intraday_v3/configs \
        --output "$OUTPUT_FILE" 2>&1 | tee /tmp/exp_${EXP_ID}.log; then

        echo "✓ SUCCESS: $EXP_ID"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))

        # Upload result to GCS
        gsutil cp "$OUTPUT_FILE" ${BUCKET}/experiment-results/${BATCH}/
        gsutil cp /tmp/exp_${EXP_ID}.log ${BUCKET}/experiment-logs/${BATCH}/

    else
        echo "✗ FAILED: $EXP_ID"
        FAIL_COUNT=$((FAIL_COUNT + 1))

        # Upload error log
        echo "{\"exp_id\": \"$EXP_ID\", \"status\": \"error\", \"error\": \"See log file\"}" > "$OUTPUT_FILE"
        gsutil cp "$OUTPUT_FILE" ${BUCKET}/experiment-results/${BATCH}/
        gsutil cp /tmp/exp_${EXP_ID}.log ${BUCKET}/experiment-logs/${BATCH}/
    fi

    # Cleanup
    rm -f "$OUTPUT_FILE" /tmp/exp_${EXP_ID}.log
done

echo ""
echo "=================================================="
echo "BATCH COMPLETE"
echo "=================================================="
echo "Total: $NUM_CONFIGS"
echo "Success: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo ""
date

# Create summary file
cat > /tmp/batch_summary.json <<EOF
{
  "batch": "$BATCH",
  "total": $NUM_CONFIGS,
  "success": $SUCCESS_COUNT,
  "failed": $FAIL_COUNT,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

gsutil cp /tmp/batch_summary.json ${BUCKET}/experiment-results/${BATCH}_summary.json

# Shutdown VM to save costs
echo "Shutting down VM..."
shutdown -h now
