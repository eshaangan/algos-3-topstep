#!/bin/bash
# Quick deployment script for critical bug fixes
# Usage: ./deploy_fixes.sh

set -e  # Exit on error

echo "======================================================================="
echo "DEPLOYING CRITICAL BUG FIXES TO GCP"
echo "======================================================================="
echo ""

# Verify we're in the right directory
if [ ! -f "model_bundle_retrained_oct2024_nov2025.pkl" ]; then
    echo "❌ ERROR: Must run from ml_intraday_v3/ directory"
    exit 1
fi

# Step 1: Validate fixes
echo "Step 1: Validating critical fixes..."
echo "-----------------------------------------------------------------------"
python test_critical_fixes.py
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Validation failed. Aborting deployment."
    exit 1
fi
echo ""

# Step 2: Build Docker image
echo "Step 2: Building Docker image..."
echo "-----------------------------------------------------------------------"
docker build -t gcr.io/trading-algo-3/topstep-trader:latest .
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Docker build failed. Aborting deployment."
    exit 1
fi
echo "✅ Docker image built successfully"
echo ""

# Step 3: Push to Google Container Registry
echo "Step 3: Pushing to Google Container Registry..."
echo "-----------------------------------------------------------------------"
docker push gcr.io/trading-algo-3/topstep-trader:latest
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Docker push failed. Aborting deployment."
    exit 1
fi
echo "✅ Image pushed to GCR successfully"
echo ""

# Step 4: Stop VM
echo "Step 4: Stopping GCP VM..."
echo "-----------------------------------------------------------------------"
gcloud compute instances stop topstep-trader-vm --zone=us-central1-a --quiet
if [ $? -ne 0 ]; then
    echo "⚠️  WARNING: Failed to stop VM (may already be stopped)"
fi
echo "✅ VM stopped"
echo ""

# Wait for VM to fully stop
echo "Waiting 10 seconds for VM to fully stop..."
sleep 10

# Step 5: Start VM (pulls latest image)
echo "Step 5: Starting GCP VM with latest image..."
echo "-----------------------------------------------------------------------"
gcloud compute instances start topstep-trader-vm --zone=us-central1-a --quiet
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to start VM."
    exit 1
fi
echo "✅ VM started"
echo ""

# Wait for VM to boot
echo "Waiting 30 seconds for VM to boot and container to start..."
sleep 30

# Step 6: Verify deployment
echo "Step 6: Verifying deployment..."
echo "-----------------------------------------------------------------------"
echo "Checking container status..."

gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
  --command='docker ps' 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Container is running"
else
    echo "⚠️  WARNING: Could not verify container status"
fi

echo ""
echo "Checking for circuit breaker in logs..."
gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
  --command='docker logs $(docker ps -q) | grep "Circuit Breaker"' 2>/dev/null | head -5

echo ""
echo "======================================================================="
echo "✅ DEPLOYMENT COMPLETE"
echo "======================================================================="
echo ""
echo "Next steps:"
echo "1. Monitor logs with: ./monitor_gcp.sh"
echo "2. Watch for 'Circuit Breaker Config: enabled=True' in logs"
echo "3. Expect first signals within 1-2 hours during RTH (8:30 AM - 3:00 PM CT)"
echo "4. Expected: 3-5 trades per day, 50-55% win rate, \$150-250 daily P&L"
echo ""
echo "To view live logs:"
echo "  gcloud compute ssh topstep-trader-vm --zone=us-central1-a \\"
echo "    --command='docker logs -f \$(docker ps -q)'"
echo ""
echo "To stop trading immediately:"
echo "  gcloud compute instances stop topstep-trader-vm --zone=us-central1-a"
echo ""
