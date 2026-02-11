#!/bin/bash
# Signal Monitoring Script for Topstep Live Trading
# Usage: ./monitor_signals.sh

echo "=================================================================================="
echo "                        TOPSTEP SIGNAL MONITOR"
echo "=================================================================================="
echo ""

# Get GCP VM logs
LOGS=$(gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
  --command='docker logs $(docker ps -q) 2>&1')

echo "📊 SIGNAL SUMMARY:"
echo "--------------------------------------------------------------------------------"
SIGNALS_GENERATED=$(echo "$LOGS" | grep -c "Signal generated")
SIGNALS_REJECTED=$(echo "$LOGS" | grep -c "rejected signal")
SIGNALS_EXECUTED=$(echo "$LOGS" | grep -c "Executing trade")

echo "Signals Generated: $SIGNALS_GENERATED"
echo "Signals Executed:  $SIGNALS_EXECUTED"
echo "Signals Rejected:  $SIGNALS_REJECTED"
echo ""

echo "🎯 RECENT SIGNALS:"
echo "--------------------------------------------------------------------------------"
echo "$LOGS" | grep -E "Signal generated" -A 2 | tail -20
echo ""

echo "❌ REJECTION REASONS:"
echo "--------------------------------------------------------------------------------"
echo "$LOGS" | grep -E "rejected|filter.*reject" -i | tail -10
echo ""

echo "⚡ CUSUM EVENTS:"
echo "--------------------------------------------------------------------------------"
echo "$LOGS" | grep "CUSUM event detected" | tail -10
echo ""

echo "💰 TRADES EXECUTED:"
echo "--------------------------------------------------------------------------------"
echo "$LOGS" | grep "Executing trade" | tail -10
echo ""

echo "=================================================================================="
echo "Live monitoring: ./monitor_signals.sh live"
echo "=================================================================================="
