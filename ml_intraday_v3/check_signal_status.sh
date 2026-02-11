#!/bin/bash
# Check current signal status with timestamps
# Shows signal counts and recent activity

ZONE="us-central1-a"
INSTANCE="topstep-trader-vm"

echo "=================================================="
echo "Signal Status Check - $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo ""

# Get recent logs with timestamps
LOGS=$(gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command='docker logs --tail=200 $(docker ps -q) 2>&1')

# Extract signal counts from dashboard updates
echo "=== Recent Signal Activity (Last 10 Updates) ==="
echo "$LOGS" | grep "Signals Generated:" | tail -10 | while read line; do
    # Extract timestamp from surrounding log lines
    echo "$line"
done
echo ""

# Show actual signal generation events with timestamps
echo "=== Signal Generation Events (With Timestamps) ==="
SIGNAL_EVENTS=$(echo "$LOGS" | grep -E "Signal generated|CUSUM|score_ev")
if [ -z "$SIGNAL_EVENTS" ]; then
    echo "No signals generated yet."
    echo ""
    echo "System is waiting for CUSUM events..."
    echo "This is normal - signals only generate when market conditions trigger CUSUM detector."
else
    echo "$SIGNAL_EVENTS"
fi
echo ""

# Show feature quality
echo "=== Feature Quality Status ==="
FEATURE_QUALITY=$(echo "$LOGS" | grep -E "Feature quality|nan_count|healthy" | tail -5)
if [ -z "$FEATURE_QUALITY" ]; then
    echo "No feature quality checks logged yet."
else
    echo "$FEATURE_QUALITY"
fi
echo ""

# Show recent bar updates
echo "=== Recent Bar Updates ==="
echo "$LOGS" | grep -E "New bar|Bar received" | tail -5
echo ""

# Show system status
echo "=== System Status ==="
echo "$LOGS" | grep -E "API connection|Buffer initialized" | tail -3
echo ""

echo "=================================================="
echo "To watch live: ./watch_trading.sh"
echo "=================================================="
