#!/bin/bash
# Quick script to run all three direction change validation backtests

set -e

echo "========================================================================"
echo "Direction Change Validation - Running All Three Backtests"
echo "========================================================================"
echo ""
echo "This will run three backtests:"
echo "1. Baseline (no direction change)"
echo "2. High-confidence threshold (0.20) - RECOMMENDED"
echo "3. Aggressive threshold (0.10) - old behavior"
echo ""
echo "Start date: 2025-12-01"
echo "End date: 2026-01-20"
echo ""

# Change to project root (parent of ml_intraday_v3)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# Run the backtest script from project root
python ml_intraday_v3/run_direction_change_backtests.py \
  --run-dir ml_intraday_v3/runs/run_20251224_123456 \
  --config-dir ml_intraday_v3/configs \
  --bar-size 1m \
  --start-date 2025-12-01 \
  --end-date 2026-01-20

echo ""
echo "========================================================================"
echo "Backtests complete!"
echo "========================================================================"
