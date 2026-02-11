#!/bin/bash
# Comprehensive trading system monitor with timestamps
# Shows all important events in real-time

ZONE="us-central1-a"
INSTANCE="topstep-trader-vm"

# Colors for better visibility
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=================================================="
echo "Live Trading Monitor - With Timestamps"
echo "=================================================="
echo ""
echo "Watching for:"
echo "  - Signal generation (CUSUM events)"
echo "  - Feature quality checks"
echo "  - Trade execution"
echo "  - Prediction scores"
echo ""
echo "Press Ctrl+C to stop"
echo ""
echo "=================================================="
echo ""

# Monitor with color-coded output
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command='
docker logs -f $(docker ps -q) 2>&1 | grep --line-buffered -E "Signal generated|Feature quality|CUSUM|Regime change|Executing trade|Order submitted|Trade executed|nan_count|healthy|score_ev|p_target|direction_change|Rejected signal|New bar|Bar received"
' | while IFS= read -r line; do
    # Add local timestamp
    timestamp=$(date "+%H:%M:%S")

    # Color code by event type
    if [[ $line == *"Signal generated"* ]] || [[ $line == *"CUSUM"* ]]; then
        echo -e "${GREEN}[$timestamp]${NC} $line"
    elif [[ $line == *"Trade executed"* ]] || [[ $line == *"Order submitted"* ]]; then
        echo -e "${BLUE}[$timestamp]${NC} $line"
    elif [[ $line == *"Feature quality"* ]] || [[ $line == *"healthy"* ]]; then
        echo -e "${YELLOW}[$timestamp]${NC} $line"
    elif [[ $line == *"WARNING"* ]] || [[ $line == *"ERROR"* ]]; then
        echo -e "${RED}[$timestamp]${NC} $line"
    else
        echo "[$timestamp] $line"
    fi
done
