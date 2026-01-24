#!/bin/bash
set -e

# Configuration
ZONE="us-central1-a"
INSTANCE_NAME="algotrader"

echo "=== Syncing .env to $INSTANCE_NAME ==="

if [ ! -f ".env" ]; then
    echo "Error: .env not found in project root."
    echo "Please create it first."
    exit 1
fi

gcloud compute scp .env $INSTANCE_NAME:~/algos/ml_intraday_v3/.env --zone=$ZONE
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command "chmod 600 ~/algos/ml_intraday_v3/.env"

echo "=== Env Synced ==="
