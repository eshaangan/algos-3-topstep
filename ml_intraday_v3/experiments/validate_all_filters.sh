#!/bin/bash
# Quick validation script for all Phase 1 filters
# Run this to confirm all filters are working before integration

set -e  # Exit on error

echo "========================================================================"
echo "PHASE 1 FILTER VALIDATION"
echo "========================================================================"
echo ""

# Change to project root
cd "$(dirname "$0")/../.."

echo "📍 Working directory: $(pwd)"
echo ""

# Test 1: Circuit Breaker
echo "========================================================================"
echo "TEST 1: Circuit Breaker"
echo "========================================================================"
python ml_intraday_v3/experiments/test_circuit_breaker.py
CB_EXIT=$?
echo ""

# Test 2: Regime Detector
echo "========================================================================"
echo "TEST 2: Regime Detector"
echo "========================================================================"
python ml_intraday_v3/experiments/test_regime_detector.py
RD_EXIT=$?
echo ""

# Summary
echo "========================================================================"
echo "VALIDATION SUMMARY"
echo "========================================================================"

if [ $CB_EXIT -eq 0 ]; then
    echo "✅ Circuit Breaker: PASS"
else
    echo "❌ Circuit Breaker: FAIL (exit code $CB_EXIT)"
fi

if [ $RD_EXIT -eq 0 ]; then
    echo "✅ Regime Detector: PASS"
else
    echo "❌ Regime Detector: FAIL (exit code $RD_EXIT)"
fi

echo "========================================================================"
echo ""

# Overall result
if [ $CB_EXIT -eq 0 ] && [ $RD_EXIT -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED - Filters are ready for integration!"
    echo ""
    echo "Next steps:"
    echo "1. Read ml_intraday_v3/FILTER_INTEGRATION_GUIDE.md"
    echo "2. Integrate filters into live_runner.py"
    echo "3. Run backtest with filters enabled"
    echo "4. Start paper trading"
    echo ""
    exit 0
else
    echo "⚠️ SOME TESTS FAILED - Review output above"
    echo ""
    exit 1
fi
