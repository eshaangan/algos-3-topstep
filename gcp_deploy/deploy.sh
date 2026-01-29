#!/bin/bash
set -e

# Configuration
PROJECT_ID=$(gcloud config get-value project)
APP_NAME="topstep-trader"
REGION="us-central1"
IMAGE_TAG="gcr.io/${PROJECT_ID}/${APP_NAME}:latest"
VM_NAME="${APP_NAME}-vm"

echo "Deploying to GCP Project: ${PROJECT_ID}"

# 1. Enable Artifact Registry (if needed - traditionally Container Registry works with gcr.io)
# echo "Enabling Container Registry API..."
# gcloud services enable containerregistry.googleapis.com

# 2. Build Docker Image
echo "Building Docker image..."
# We build from the root directory but use the Dockerfile in gcp_deploy
docker build --platform linux/amd64 -f gcp_deploy/Dockerfile -t "${IMAGE_TAG}" .

# 3. Push to Google Container Registry
echo "Pushing image to GCR..."
docker push "${IMAGE_TAG}"

# 4. Create/Update Compute Engine VM
echo "Deploying to Compute Engine..."

# Check if VM exists
if gcloud compute instances describe "${VM_NAME}" --zone="${REGION}-a" > /dev/null 2>&1; then
    echo "VM ${VM_NAME} already exists. Updating container..."
    gcloud compute instances update-container "${VM_NAME}" \
        --container-image="${IMAGE_TAG}" \
        --zone="${REGION}-a"
else
    echo "Creating new VM ${VM_NAME}..."
    # Note: You need to pass environment variables. 
    # For security, we recommend using Secret Manager, but for simplicity here
    # we'll assume a .env file exists and parse it.
    
    # Construct --container-env flags from .env
    ENV_FLAGS=""
    if [ -f .env ]; then
        echo "Reading .env file..."
        while read -r line || [ -n "$line" ]; do
            # Remove inline comments (everything after #)
            line="${line%%#*}"
            # Trim leading/trailing whitespace
            line="$(echo "${line}" | xargs)"
            
            # Skip empty lines
            if [ -z "$line" ]; then continue; fi
            
            # Parse Key=Value
            if [[ "$line" == *"="* ]]; then
                key="${line%%=*}"
                value="${line#*=}"
                
                # Trim key and value
                key="$(echo "${key}" | xargs)"
                value="$(echo "${value}" | xargs)"
                
                if [ -n "$key" ]; then
                    ENV_FLAGS="${ENV_FLAGS}--container-env=${key}=${value} "
                fi
            fi
        done < .env
    else
        echo "WARNING: No .env file found! Bot may fail to start."
    fi

    gcloud compute instances create-with-container "${VM_NAME}" \
        --container-image="${IMAGE_TAG}" \
        --machine-type=e2-medium \
        --zone="${REGION}-a" \
        --tags=http-server \
        $ENV_FLAGS
fi

echo "Deployment complete!"
echo "View logs: gcloud compute instances get-serial-port-output ${VM_NAME} --zone=${REGION}-a"
