#!/bin/bash
set -e

# Configuration
ZONE="us-central1-a"
INSTANCE_NAME="algotrader"
REMOTE_DIR="~/algos/ml_intraday_v3"
LOCAL_DIR="ml_intraday_v3" # Run from project root

echo "=== Deploying Code Update to $INSTANCE_NAME ==="

# 1. Package Code (excluding heavy/sensitive files)
echo "Packaging code..."
tar -czf code_bundle.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='runs' \
    --exclude='data' \
    --exclude='.env' \
    --exclude='logs' \
    -C . ml_intraday_v3 core

# 2. Upload Bundle
echo "Uploading bundle..."
gcloud compute scp code_bundle.tar.gz $INSTANCE_NAME:~/ --zone=$ZONE

# 3. Remote Execution
echo "Executing remote update..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command "
    # Ensure directory exists
    mkdir -p ~/algos

    # Extract
    echo 'Extracting...'
    tar -xzf code_bundle.tar.gz -C ~/algos/

    # Update Dependencies
    echo 'Updating dependencies...'
    cd $REMOTE_DIR
    if [ ! -d 'venv' ]; then
        python3.11 -m venv venv
    fi
    source venv/bin/activate
    pip install -r requirements-mlv3.txt

    # Restart Service (if it exists)
    if systemctl list-units --full -all | grep -Fq 'algotrader.service'; then
        echo 'Restarting algotrader service...'
        sudo systemctl restart algotrader
        sudo systemctl status algotrader --no-pager
    else
        echo 'Service not yet installed. Run setup_service.sh remotely if needed.'
    fi
"

# Cleanup
rm code_bundle.tar.gz

echo "=== Deployment Complete ==="
