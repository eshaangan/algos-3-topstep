#!/bin/bash
# Run comprehensive local grid search
# Tests 30 key configurations (5-10 minutes total)

cd "$(dirname "$0")/.."
mkdir -p results

echo "=== Running Local Grid Search (30 experiments) ==="
echo "Estimated time: 5-10 minutes"
echo ""

# Define experiment matrix
MODELS=("minimal" "conservative" "moderate")
FEATURES=("top5" "top10" "top20")
WINDOWS=(3 6 12)
CALIBRATIONS=("null" "isotonic")

exp_num=1
for model in "${MODELS[@]}"; do
  for features in "${FEATURES[@]}"; do
    for window in "${WINDOWS[@]}"; do
      for calib in "${CALIBRATIONS[@]}"; do
        
        exp_id="local_${model}_${features}_${window}mo_${calib}"
        
        # Skip some combinations to keep it to ~30 experiments
        if [[ $exp_num -gt 30 ]]; then
          break 4
        fi
        
        echo "[$exp_num/30] Running $exp_id..."
        
        # Generate config
        cat > /tmp/exp_$exp_num.json << EOFJSON
{
  "exp_id": "$exp_id",
  "phase": 1,
  "model_name": "$model",
  "model_params": $(python3 -c "import yaml; c=yaml.safe_load(open('experiments/grid_config.yaml')); import json; print(json.dumps(c['phase1']['model_configs']['$model']))"),
  "feature_set_name": "$features",
  "feature_set": $(python3 -c "import yaml; c=yaml.safe_load(open('experiments/grid_config.yaml')); import json; print(json.dumps(c['phase1']['feature_sets']['$features']))"),
  "training_window_months": $window,
  "labeling": {"pt": 3.0, "sl": 2.5, "hz": 12},
  "sample_weight": "uniform",
  "calibration": $(if [ "$calib" == "null" ]; then echo "null"; else echo "\"$calib\""; fi)
}
EOFJSON
        
        # Run experiment
        python3 experiments/comprehensive_grid_search_v2.py \
          --config /tmp/exp_$exp_num.json \
          --data-dir data \
          --config-dir configs \
          --output results/$exp_id.json \
          2>&1 | grep "✅"
        
        exp_num=$((exp_num + 1))
      done
    done
  done
done

echo ""
echo "=== All experiments complete! ==="
echo "Results saved to: results/"
echo ""
echo "To see top 5 configurations:"
echo "  python3 experiments/analyze_results_simple.py"
