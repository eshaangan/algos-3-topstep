#!/bin/bash
# Verify Feature NaN Fix - Monitor trading system for varying predictions
# Date: February 4, 2026
# Purpose: Confirm that 300-bar buffer and feature quality check fix identical predictions

set -e

ZONE="us-central1-a"
INSTANCE="topstep-trader-vm"

echo "===================================================="
echo "Feature NaN Fix Verification Script"
echo "===================================================="
echo ""

# Function to run SSH command and get logs
get_logs() {
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" \
        --command="docker logs \$(docker ps -q) 2>&1" 2>/dev/null
}

echo "Step 1: Check Buffer Initialization"
echo "----------------------------------------------------"
echo "Expected: 'Buffer initialized with 300 bars'"
echo ""
get_logs | grep -i "buffer initialized" | tail -5
echo ""

echo "Step 2: Check Feature Quality"
echo "----------------------------------------------------"
echo "Expected: 'healthy: True' (no NaN warnings after warmup)"
echo ""
get_logs | grep -i "feature quality" | tail -10
echo ""

echo "Step 3: Check for NaN Features"
echo "----------------------------------------------------"
echo "Expected: No NaN warnings after first ~50 bars"
echo ""
get_logs | grep -i "nan_count\|nan_columns" | tail -10
echo ""

echo "Step 4: Check Signal Diversity"
echo "----------------------------------------------------"
echo "Expected: VARYING scores and probabilities (not identical)"
echo ""
echo "Recent signals:"
get_logs | grep -i "signal generated\|score_ev\|p_target\|p_stop" | tail -20
echo ""

echo "Step 5: Check Trade Executions"
echo "----------------------------------------------------"
echo "Expected: Trades execute when P > 0.55"
echo ""
get_logs | grep -i "executing trade\|trade executed\|order submitted" | tail -10
echo ""

echo "Step 6: Check for Identical Prediction Pattern"
echo "----------------------------------------------------"
echo "Expected: NO repeated identical values"
echo ""
echo "Analyzing prediction patterns..."
PREDICTIONS=$(get_logs | grep -oE "score_ev=[0-9.]+, p_target=[0-9.]+, p_stop=[0-9.]+" | tail -10)
if [ -z "$PREDICTIONS" ]; then
    echo "⚠️  No predictions found yet. System may still be initializing."
else
    echo "$PREDICTIONS"
    echo ""
    # Count unique predictions
    UNIQUE_COUNT=$(echo "$PREDICTIONS" | sort -u | wc -l | tr -d ' ')
    TOTAL_COUNT=$(echo "$PREDICTIONS" | wc -l | tr -d ' ')
    echo "Unique predictions: $UNIQUE_COUNT out of $TOTAL_COUNT"

    if [ "$UNIQUE_COUNT" -eq "$TOTAL_COUNT" ]; then
        echo "✅ SUCCESS: All predictions are unique (varying with market)"
    elif [ "$UNIQUE_COUNT" -eq 1 ]; then
        echo "❌ FAILURE: All predictions are identical (bug still present)"
    else
        echo "⚠️  PARTIAL: Some variation but potential issues"
    fi
fi
echo ""

echo "Step 7: System Health Summary"
echo "----------------------------------------------------"
get_logs | grep -iE "error|warning|critical|fatal" | tail -10
echo ""

echo "===================================================="
echo "Verification Complete"
echo "===================================================="
echo ""
echo "✅ SUCCESS CRITERIA:"
echo "  1. Buffer initialized with 300 bars"
echo "  2. Feature quality shows 'healthy: True'"
echo "  3. No NaN warnings after warmup"
echo "  4. Signals have VARYING scores/probabilities"
echo "  5. Trades execute when P > 0.55"
echo ""
echo "To monitor in real-time:"
echo "  ./monitor_signals.sh"
echo ""
echo "To check full logs:"
echo "  gcloud compute ssh $INSTANCE --zone=$ZONE \\"
echo "    --command='docker logs -f \$(docker ps -q)'"
echo ""
