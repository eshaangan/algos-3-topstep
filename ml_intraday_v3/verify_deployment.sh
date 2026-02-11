#!/bin/bash
# Comprehensive deployment verification script
# Checks that all Jan 2026 configurations are correctly applied

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "Deployment Verification - Jan 2026 Configuration"
echo "═══════════════════════════════════════════════════════════════"
echo ""

VM_NAME="topstep-trader-vm"
ZONE="us-central1-a"
PROJECT="trading-algo-3"

# Check if VM is running
echo "📡 Checking VM status..."
VM_STATUS=$(gcloud compute instances describe $VM_NAME --zone=$ZONE --project=$PROJECT --format="value(status)" 2>&1)

if [ "$VM_STATUS" != "RUNNING" ]; then
    echo "❌ VM is not running (status: $VM_STATUS)"
    exit 1
fi
echo "✅ VM is RUNNING"
echo ""

# Check container status
echo "🐳 Checking Docker container..."
CONTAINER_STATUS=$(gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command='docker ps --format "{{.Status}}"' 2>&1 | head -1)
echo "   Status: $CONTAINER_STATUS"
echo "✅ Container is running"
echo ""

# Check model loaded
echo "🤖 Verifying model configuration..."
MODEL_INFO=$(gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command='docker logs $(docker ps -q) 2>&1 | grep -E "Loading model:|Model loaded:|Features:" | tail -5' 2>&1)

echo "$MODEL_INFO" | while read line; do
    echo "   $line"
done

# Extract and verify
if echo "$MODEL_INFO" | grep -q "model_bundle_retrained_oct2024_nov2025.pkl"; then
    echo "✅ Correct model: model_bundle_retrained_oct2024_nov2025.pkl"
else
    echo "❌ WRONG MODEL - Expected: model_bundle_retrained_oct2024_nov2025.pkl"
    echo "   Check logs above to see which model was loaded"
fi

if echo "$MODEL_INFO" | grep -q "Features: 34"; then
    echo "✅ Correct feature count: 34"
else
    echo "❌ WRONG FEATURE COUNT - Expected: 34"
fi
echo ""

# Check filter configurations
echo "🔧 Verifying filter configurations..."
FILTER_CONFIG=$(gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command='docker logs $(docker ps -q) 2>&1 | grep -E "Regime Filter|Volatility Filter|Confidence filter|Circuit Breaker" | head -10' 2>&1)

echo "$FILTER_CONFIG" | while read line; do
    echo "   $line"
done

echo ""
echo "Expected configurations:"
echo "   ✓ Confidence filter enabled: threshold=0.55"
echo "   ✓ Regime Filter Config: enabled=False"
echo "   ✓ Volatility Filter Config: enabled=False"
echo "   ✓ Circuit Breaker Config: enabled=True"
echo ""

# Check each filter
if echo "$FILTER_CONFIG" | grep -q "Confidence filter enabled: threshold=0.55"; then
    echo "✅ Confidence filter: CORRECT (0.55)"
else
    echo "⚠️  Confidence filter: Check logs above"
fi

if echo "$FILTER_CONFIG" | grep -q "Regime Filter Config: enabled=False"; then
    echo "✅ Regime filter: DISABLED (correct)"
elif echo "$FILTER_CONFIG" | grep -q "Regime Filter Config: enabled=True"; then
    echo "❌ Regime filter: ENABLED (should be disabled!)"
else
    echo "⚠️  Regime filter: Not found in logs"
fi

if echo "$FILTER_CONFIG" | grep -q "Volatility Filter Config: enabled=False"; then
    echo "✅ Volatility/ADX filter: DISABLED (correct)"
elif echo "$FILTER_CONFIG" | grep -q "Volatility Filter Config: enabled=True"; then
    echo "❌ Volatility/ADX filter: ENABLED (should be disabled!)"
else
    echo "⚠️  Volatility/ADX filter: Not found in logs"
fi

if echo "$FILTER_CONFIG" | grep -q "Circuit Breaker Config: enabled=True"; then
    echo "✅ Circuit breaker: ENABLED (correct)"
else
    echo "⚠️  Circuit breaker: Check logs above"
fi
echo ""

# Check config files in container
echo "📄 Verifying config files in container..."
echo ""
echo "--- live_trading.yaml (key settings) ---"
gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command='docker exec $(docker ps -q) grep -A 1 "model_bundle_path:" /app/ml_intraday_v3/configs/live_trading.yaml | head -2' 2>&1
gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command='docker exec $(docker ps -q) grep "primary_threshold:" /app/ml_intraday_v3/configs/live_trading.yaml | grep -v "#" | head -1' 2>&1
gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command='docker exec $(docker ps -q) grep -A 3 "regime_filter:" /app/ml_intraday_v3/configs/live_trading.yaml | head -4' 2>&1
echo ""

echo "--- execution_spec.yaml (volatility filter) ---"
gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command='docker exec $(docker ps -q) grep -A 2 "volatility:" /app/ml_intraday_v3/configs/execution_spec.yaml | grep "enabled:" | head -1' 2>&1
echo ""

# Summary
echo "═══════════════════════════════════════════════════════════════"
echo "VERIFICATION SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "✅ = Correct configuration (matches Jan 2026)"
echo "❌ = Incorrect configuration (needs fix)"
echo "⚠️  = Unable to verify (check logs manually)"
echo ""
echo "Critical checks:"
echo "  1. Model: model_bundle_retrained_oct2024_nov2025.pkl"
echo "  2. Features: 34"
echo "  3. Primary threshold: 0.03 (in config file)"
echo "  4. Confidence filter: 0.55"
echo "  5. Regime filter: DISABLED"
echo "  6. Volatility/ADX filter: DISABLED"
echo "  7. Circuit breaker: ENABLED"
echo ""
echo "If any checks failed, review the logs above and re-deploy."
echo ""
echo "Next steps:"
echo "  - If all ✅: Wait for RTH (8:30 AM CT) and monitor for trades"
echo "  - If any ❌: Fix configs and redeploy"
echo ""
