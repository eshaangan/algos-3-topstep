#!/bin/bash
# Monitor trading system with timestamps
# Shows signal generation, feature quality, and trade execution

ZONE="us-central1-a"
INSTANCE="topstep-trader-vm"

echo "Monitoring trading system with timestamps..."
echo "Press Ctrl+C to stop"
echo ""

gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command='
docker logs -f $(docker ps -q) 2>&1 | grep --line-buffered -E "Signal generated|Feature quality|CUSUM|Executing trade|Order submitted|nan_count|healthy|score_ev|p_target"
'
