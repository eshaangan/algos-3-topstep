#!/bin/bash
# Execute Full V3 Training Pipeline with Balanced Events
# This script runs the complete training workflow with balanced LONG/SHORT data

set -e  # Exit on error

echo "================================================================================"
echo "FULL V3 TRAINING PIPELINE - BALANCED BIDIRECTIONAL MODEL"
echo "================================================================================"
echo "Training Period: 2024-01-01 to 2025-11-30 (includes multiple market regimes)"
echo "Test Period: 2025-12-01 to 2025-12-31"
echo "Target: 50% LONG, 50% SHORT event distribution"
echo "================================================================================"

# Configuration
RUN_DIR="../runs/balanced_v3_q1_2024_q4_2025"
BAR_SIZE="5m"
CONFIG_DIR="ml_intraday_v3/configs"

# Create run directory
mkdir -p "$RUN_DIR"
echo "✓ Created run directory: $RUN_DIR"

# Navigate to ml_intraday_v3 directory
cd ml_intraday_v3 || exit 1

echo ""
echo "Step 1/4: Generating events with trend_scanning..."
echo "--------------------------------------------------------------------------------"
python3 -m cli build-events \
    --run-dir "$RUN_DIR" \
    --bar-size "$BAR_SIZE" \
    --labeling-config configs/labeling.yaml \
    --execution-spec configs/execution_spec.yaml

echo ""
echo "Step 2/4: Building features..."
echo "--------------------------------------------------------------------------------"
python3 -m cli build-features \
    --run-dir "$RUN_DIR" \
    --bar-size "$BAR_SIZE" \
    --feature-config configs/features.yaml

echo ""
echo "Step 3/4: Training model with balanced events..."
echo "--------------------------------------------------------------------------------"
python3 -m cli train \
    --run-dir "$RUN_DIR" \
    --bar-size "$BAR_SIZE" \
    --training-config configs/training.yaml

echo ""
echo "Step 4/4: Evaluating model..."
echo "--------------------------------------------------------------------------------"
python3 -m cli evaluate \
    --run-dir "$RUN_DIR" \
    --bar-size "$BAR_SIZE"

echo ""
echo "================================================================================"
echo "TRAINING COMPLETE!"
echo "================================================================================"
echo "Model saved to: $RUN_DIR/walkforward/bar_size=$BAR_SIZE/window_0/model_bundle.pkl"
echo ""
echo "Next steps:"
echo "  1. Run validation: python3 ml_intraday_v3/validate_model_capabilities.py"
echo "  2. Run backtests: python3 ml_intraday_v3/test_directional_fix.py"
echo "  3. Check compliance: python3 ml_intraday_v3/validate_topstep_compliance.py"
echo "================================================================================"
