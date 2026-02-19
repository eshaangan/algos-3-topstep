"""
Generate Batch 8: Calibration & Regularization
Tests different calibration methods and regularization strategies.
"""
import json
import numpy as np
from pathlib import Path

def generate_batch8_configs():
    """Generate 100 configs testing calibration and regularization."""

    # Calibration methods
    calibration_methods = ['none', 'isotonic', 'sigmoid', 'beta']

    # Regularization variations
    reg_alphas = [0.0, 0.1, 0.5, 1.0]
    reg_lambdas = [0.0, 0.2, 0.5, 1.0]

    # Model variations (less than Batch 6/7)
    num_leaves_opts = [31, 63, 127]
    max_depth_opts = [6, 10, 15]

    configs = []
    config_id = 1

    for calibration in calibration_methods:
        for reg_alpha in reg_alphas:
            for reg_lambda in reg_lambdas:
                for num_leaves in num_leaves_opts:
                    for max_depth in max_depth_opts:
                        config = {
                            'exp_id': f'batch8_calibration_{config_id:03d}',
                            'phase': 'batch8_calibration',
                            'labeling_method': 'triple_barrier',
                            'sample_weight': 'uniqueness',
                            'calibration_method': calibration,
                            'model_kind': 'lightgbm',
                            'model_params': {
                                'n_estimators': 400,
                                'learning_rate': 0.03,
                                'num_leaves': num_leaves,
                                'max_depth': max_depth,
                                'min_child_samples': 80,
                                'subsample': 0.8,
                                'colsample_bytree': 0.8,
                                'reg_alpha': reg_alpha,
                                'reg_lambda': reg_lambda,
                            },
                            'cv_method': 'kfold',
                            'cv_n_splits': 5,
                            'cv_embargo_bars': 12,
                            'features_config': {},
                            'feature_set_name': 'baseline',
                            'labeling_params': {
                                'pt_mult': 2.0,
                                'sl_mult': 1.5,
                                'time_mult': 40,
                            }
                        }
                        configs.append(config)
                        config_id += 1

                        if len(configs) >= 100:
                            break
                    if len(configs) >= 100:
                        break
                if len(configs) >= 100:
                    break
            if len(configs) >= 100:
                break
        if len(configs) >= 100:
            break

    return configs[:100]

if __name__ == '__main__':
    configs = generate_batch8_configs()

    # Save individual config files
    output_dir = Path('batch8_calibration_configs')
    output_dir.mkdir(exist_ok=True)

    for config in configs:
        config_path = output_dir / f"{config['exp_id']}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    # Save summary
    summary = {
        'batch': 'batch8_calibration',
        'n_configs': len(configs),
        'calibration_methods': list(set(c['calibration_method'] for c in configs)),
    }

    with open('batch8_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Generated {len(configs)} configs for Batch 8")
    print(f"   Calibration methods: {summary['calibration_methods']}")
    print(f"   Saved to: {output_dir}/")
