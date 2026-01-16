#!/bin/bash
# Run bidirectional 24-hour training using the pipeline notebook

echo "================================================================================"
echo "ML INTRADAY V3 - BIDIRECTIONAL 24-HOUR TRAINING"
echo "================================================================================"
echo "Run ID: bidirectional_24h_$(date +%Y%m%d_%H%M%S)"
echo "Date: $(date)"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  ✓ Data: 2022-2025 (24-hour Globex)"
echo "  ✓ Labeling: trend_scanning (adds 'side' feature)"
echo "  ✓ Model: Bidirectional (LONG + SHORT)"
echo "  ✓ Threshold: 0.03 (aggressive)"
echo "================================================================================"
echo ""
echo "Starting Jupyter notebook for training..."
echo "Follow these steps in the notebook:"
echo ""
echo "1. Open: ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb"
echo "2. Set RUN_ID = 'bidirectional_24h_$(date +%Y%m%d_%H%M%S)'"
echo "3. Run all cells to completion"
echo "4. Model will be saved to ml_intraday_v3/models/saved/"
echo ""
echo "Press Enter to open Jupyter notebook..."
read

# Launch Jupyter
cd "$(dirname "$0")"
jupyter notebook ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb
