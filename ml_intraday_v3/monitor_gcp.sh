#!/bin/bash
# Monitor live trading system on GCP

set -e

PROJECT_ID="trading-algo-3"
ZONE="us-central1-a"
VM_NAME="topstep-trader-vm"

echo "=========================================="
echo "   TOPSTEP LIVE TRADING - GCP MONITOR"
echo "=========================================="
echo ""

# 1. Check VM status
echo "1. VM Status:"
gcloud compute instances describe $VM_NAME --zone=$ZONE --format="table(name,status,machineType)"
echo ""

# 2. Check container status
echo "2. Container Status:"
gcloud compute ssh $VM_NAME --zone=$ZONE --command="docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'" 2>/dev/null
echo ""

# 3. Check latest logs (last 30 lines)
echo "3. Recent Logs:"
echo "----------------------------------------"
gcloud compute ssh $VM_NAME --zone=$ZONE --command="docker logs \$(docker ps -q) 2>&1 | tail -30" 2>/dev/null
echo ""

# 4. Check for trades
echo "4. Trade Activity:"
echo "----------------------------------------"
gcloud compute ssh $VM_NAME --zone=$ZONE --command="docker logs \$(docker ps -q) 2>&1 | grep -E 'TRADE|SIGNAL|POSITION' | tail -10" 2>/dev/null || echo "No trades yet"
echo ""

# 5. Check for errors
echo "5. Recent Errors/Warnings:"
echo "----------------------------------------"
gcloud compute ssh $VM_NAME --zone=$ZONE --command="docker logs \$(docker ps -q) 2>&1 | grep -E 'ERROR|WARNING|CRITICAL' | tail -10" 2>/dev/null || echo "No errors"
echo ""

echo "=========================================="
echo "Monitor complete. Run this script anytime to check status."
echo ""
echo "To view live logs:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --command='docker logs -f \$(docker ps -q)'"
echo ""
echo "To stop trading:"
echo "  gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo ""
echo "To restart trading:"
echo "  gcloud compute instances start $VM_NAME --zone=$ZONE"
echo "=========================================="
