#!/bin/bash
# Run REAL strategy comparison using actual model predictions
# This script re-runs the best model and compares signal selection strategies

cd "$(dirname "$0")/.."

echo "=========================================="
echo "REAL STRATEGY COMPARISON"
echo "=========================================="
echo ""
echo "Re-running best model:"
echo "  - Model: Conservative (PT=3.0, SL=3.5)"
echo "  - Features: Top 10"
echo "  - Window: 6 months"
echo ""
echo "Comparing strategies:"
echo "  - Solution 1: Fixed thresholds (0.20, 0.25, 0.30)"
echo "  - Solution 2: Percentile ranking (top 10%, 15%, 20%)"
echo ""
echo "This will take ~30 seconds..."
echo "=========================================="
echo ""

python3 experiments/compare_strategies_real.py

echo ""
echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
