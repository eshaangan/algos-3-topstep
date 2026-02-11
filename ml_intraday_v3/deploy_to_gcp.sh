#!/bin/bash
set -e

# Configuration
PROJECT_ID="trading-algo-3"
IMAGE_NAME="topstep-trader"
TAG="latest"
ZONE="us-central1-a"
VM_NAME="topstep-trader-vm"

echo "=== Deploying ml_intraday_v3 to GCP ==="

# 1. Build Docker image
echo "Building Docker image..."
docker build -t gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG} .

# 2. Push to Google Container Registry
echo "Pushing image to GCR..."
docker push gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}

# 3. Stop VM if running
echo "Checking VM status..."
VM_STATUS=$(gcloud compute instances describe ${VM_NAME} --zone=${ZONE} --format='get(status)' 2>/dev/null || echo "NOT_FOUND")

if [ "$VM_STATUS" == "RUNNING" ]; then
    echo "Stopping VM..."
    gcloud compute instances stop ${VM_NAME} --zone=${ZONE}
fi

# 4. Update VM container image (force pull latest)
echo "Updating container on VM..."
gcloud compute instances update-container ${VM_NAME} \
    --zone=${ZONE} \
    --container-image=gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}

# 5. Start VM
echo "Starting VM..."
gcloud compute instances start ${VM_NAME} --zone=${ZONE}

# 6. Wait for VM to be ready
echo "Waiting for VM to start..."
sleep 10

# 7. Check logs
echo "Checking container logs..."
gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command="docker logs \$(docker ps -q --filter ancestor=gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}) 2>&1 | tail -20"

echo "=== Deployment Complete ==="
echo "VM: ${VM_NAME}"
echo "Image: gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"
echo ""
echo "To view logs: gcloud compute ssh ${VM_NAME} --zone=${ZONE}"
