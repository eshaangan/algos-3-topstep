#!/bin/bash
# Barrier Optimization Sweep
# Tests different PT/SL combinations to find optimal barriers

cd "$(dirname "$0")/.."
mkdir -p results/barrier_opt

echo "=== Barrier Optimization Sweep ==="
echo "Testing: 3 PT × 3 SL × 2 models = 18 experiments"
echo "Estimated time: 3-4 minutes"
echo ""

exp_num=1
total=18

# PT/SL combinations to test
PTs=(2.0 2.5 3.0)
SLs=(2.5 3.0 3.5)
MODELS=("conservative" "moderate")

for model in "${MODELS[@]}"; do
  for pt in "${PTs[@]}"; do
    for sl in "${SLs[@]}"; do
      
      exp_id="barrier_${model}_pt${pt}_sl${sl}"
      
      echo "[$exp_num/$total] Testing PT=${pt}, SL=${sl}, Model=${model}..."
      
      # Generate config
      python3 << EOFPY
import json
import yaml

# Load grid config for model params
with open('experiments/grid_config.yaml', 'r') as f:
    grid_config = yaml.safe_load(f)

config = {
    "exp_id": "$exp_id",
    "phase": 1,
    "model_name": "$model",
    "model_params": grid_config['phase1']['model_configs']['$model'],
    "feature_set_name": "top10",
    "feature_set": grid_config['phase1']['feature_sets']['top10'],
    "training_window_months": 6,
    "labeling": {"pt": $pt, "sl": $sl, "hz": 12},
    "sample_weight": "uniform",
    "calibration": None
}

with open('/tmp/barrier_exp_$exp_num.json', 'w') as f:
    json.dump(config, f, indent=2)
EOFPY
      
      # Run experiment
      python3 experiments/comprehensive_grid_search_v2.py \
        --config /tmp/barrier_exp_$exp_num.json \
        --data-dir data \
        --config-dir configs \
        --output results/barrier_opt/$exp_id.json \
        2>&1 | grep "✅"
      
      exp_num=$((exp_num + 1))
    done
  done
done

echo ""
echo "=== Barrier Optimization Complete! ==="
echo "Results saved to: results/barrier_opt/"
echo ""
