#!/usr/bin/env python3
"""
Quick launcher for Phase 1 grid search.
Generates configs and runs locally (GCP launch commented out for safety).
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Load grid config
config_path = Path(__file__).parent / 'grid_config.yaml'
with open(config_path, 'r') as f:
    grid_config = yaml.safe_load(f)

phase1 = grid_config['phase1']

# Generate experiment configs (first 10 for testing)
model_names = list(phase1['model_configs'].keys())
feature_set_names = list(phase1['feature_sets'].keys())
training_windows = phase1['training_windows']
labeling_barriers = phase1['labeling_barriers']
sample_weights = phase1['sample_weights']
calibration_methods = phase1['calibration_methods']

np.random.seed(42)

print(f"Generating Phase 1 configurations...")

# For initial test, generate just 10 diverse configs
num_test = 10
configs = []

for i in range(num_test):
    exp_id = f"phase1_exp_{i+1:04d}"
    
    # Sample from each dimension
    model_name = np.random.choice(model_names)
    feature_set_name = np.random.choice(feature_set_names)
    training_window = np.random.choice(training_windows)
    labeling = np.random.choice(labeling_barriers)
    sample_weight = np.random.choice(sample_weights)
    calibration = np.random.choice(calibration_methods)
    
    config = {
        'exp_id': exp_id,
        'phase': 1,
        'model_name': model_name,
        'model_params': phase1['model_configs'][model_name],
        'feature_set_name': feature_set_name,
        'feature_set': phase1['feature_sets'][feature_set_name],
        'training_window_months': training_window,
        'labeling': labeling,
        'sample_weight': sample_weight,
        'calibration': calibration
    }
    
    configs.append(config)

# Save configs
output_dir = Path('experiment_configs_phase1')
output_dir.mkdir(exist_ok=True)

for config in configs:
    config_file = output_dir / f"{config['exp_id']}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

print(f"\n✅ Generated {len(configs)} Phase 1 configurations")
print(f"📁 Saved to: {output_dir}/")
print(f"\n🚀 Next: Run experiments locally or on GCP\n")

# Print sample experiment
print("Sample config (phase1_exp_0001):")
print(json.dumps(configs[0], indent=2))

print("\n" + "="*80)
print("TO RUN LOCALLY (one experiment):")
print("="*80)
print(f"""
cd /Users/eshaanganguly/Documents/projects/algos\\ 3\\ topstep/ml_intraday_v3

python experiments/comprehensive_grid_search_v2.py \\
    --config experiments/experiment_configs_phase1/phase1_exp_0001.json \\
    --data-dir data \\
    --config-dir configs \\
    --output results/phase1_exp_0001.json
""")

print("="*80)
print("TO RUN ALL 10 (parallel on local machine):")
print("="*80)
print("""
cd experiments
for config in experiment_configs_phase1/*.json; do
    exp_id=$(basename $config .json)
    python comprehensive_grid_search_v2.py \\
        --config $config \\
        --data-dir ../data \\
        --config-dir ../configs \\
        --output ../results/$exp_id.json &
done
wait
echo "All experiments complete!"
""")

print("="*80)
print("TO ANALYZE RESULTS:")
print("="*80)
print("""
python analyze_results.py --phase 1 --top-k 5
""")
