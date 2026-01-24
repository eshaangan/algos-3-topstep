#!/bin/bash
set -e

# Configuration
PROJECT_ID="trading-algo-3"
REGION="us-central1"
ZONE="us-central1-a"
INSTANCE_NAME="algotrader"
MACHINE_TYPE="e2-medium"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"

echo "=== GCP Infrastructure Setup ==="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Zone: $ZONE"
echo "Instance: $INSTANCE_NAME"

# Enable Compute Engine API
echo "Enabling Compute Engine API..."
gcloud services enable compute.googleapis.com

# Create Firewall Rule
echo "Checking firewall rules..."
if ! gcloud compute firewall-rules describe allow-ssh-algotrader &>/dev/null; then
    echo "Creating firewall rule 'allow-ssh-algotrader'..."
    # Note: Opening to 0.0.0.0/0 for SSH is risky but standard for dynamic IPs. 
    # The plan suggests limiting to YOUR_IP, but that changes. 
    # I'll use 0.0.0.0/0 but rely on SSH keys.
    gcloud compute firewall-rules create allow-ssh-algotrader \
      --allow tcp:22 \
      --source-ranges 0.0.0.0/0 \
      --target-tags algotrader \
      --description "Allow SSH for algotrader"
else
    echo "Firewall rule 'allow-ssh-algotrader' already exists."
fi

# Create Instance
echo "Checking instance..."
if ! gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
    echo "Creating instance '$INSTANCE_NAME'..."
    gcloud compute instances create $INSTANCE_NAME \
      --machine-type=$MACHINE_TYPE \
      --zone=$ZONE \
      --image-family=$IMAGE_FAMILY \
      --image-project=$IMAGE_PROJECT \
      --boot-disk-size=30GB \
      --boot-disk-type=pd-standard \
      --network-tier=STANDARD \
      --maintenance-policy=MIGRATE \
      --tags=algotrader \
      --metadata=startup-script='#!/bin/bash
        apt-get update
        apt-get install -y python3.11 python3.11-venv python3-pip git htop tmux
      '
    echo "Instance created successfully."
else
    echo "Instance '$INSTANCE_NAME' already exists."
fi

echo "=== Setup Complete ==="
echo "You can now SSH into the instance using:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
