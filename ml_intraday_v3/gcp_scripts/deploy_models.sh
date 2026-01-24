#!/bin/bash
set -e

# Configuration
ZONE="us-central1-a"
INSTANCE_NAME="algotrader"

echo "=== Deploying Models to $INSTANCE_NAME ==="

# 1. Package Models
echo "Packaging models (runs/ directory)..."
if [ ! -d "ml_intraday_v3/runs" ]; then
    echo "Error: ml_intraday_v3/runs directory not found."
    exit 1
fi

tar -czf model_bundle.tar.gz -C ml_intraday_v3 runs models

# 2. Upload Bundle
echo "Uploading bundle..."
gcloud compute scp model_bundle.tar.gz $INSTANCE_NAME:~/ --zone=$ZONE

# 3. Remote Extraction
echo "Extracting on remote..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command "
    mkdir -p ~/algos/ml_intraday_v3
    tar -xzf model_bundle.tar.gz -C ~/algos/ml_intraday_v3/
    echo 'Models updated.'
    ls -F ~/algos/ml_intraday_v3/runs/
"

# Cleanup
rm model_bundle.tar.gz

echo "=== Model Deployment Complete ==="
