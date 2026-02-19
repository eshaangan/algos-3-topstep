"""
Generate Batch 6: Sample Weighting Methods
Tests different approaches to handling temporal overlap and class imbalance.
"""
import json
import numpy as np
from pathlib import Path
from itertools import product

def generate_batch6_configs():
    """Generate 200 configs testing sample weighting methods."""

    # Parameter grid
    sample_weights = ['uniform', 'uniqueness', 'uniqueness_decay', 'time_decay']
    class_weights = ['balanced', 'balanced_subsample', None]
    decay_lambdas = [0.001, 0.005, 0.01, 0.05]

    # Model variations
    num_leaves_opts = [31, 63, 127]
    max_depth_opts = [6, 10, 15]
    learning_rates = [0.01, 0.03, 0.05]

    configs = []
    config_id = 1

    # Generate combinations
    for sample_weight in sample_weights:
        for class_weight in class_weights:
            for num_leaves in num_leaves_opts:
                for max_depth in max_depth_opts:
                    for lr in learning_rates:
                        # For decay methods, vary lambda
                        if 'decay' in sample_weight:
                            for decay_lambda in decay_lambdas:
                                config = {
                                    'exp_id': f'batch6_sample_weight_{config_id:03d}',
                                    'phase': 'batch6_sample_weight',
                                    'labeling_method': 'triple_barrier',  # Use best from Batch 1
                                    'sample_weight': sample_weight,
                                    'sample_weight_params': {
                                        'lambda': decay_lambda
                                    },
                                    'class_weight': class_weight,
                                    'model_kind': 'lightgbm',
                                    'model_params': {
                                        'n_estimators': 400,
                                        'learning_rate': lr,
                                        'num_leaves': num_leaves,
                                        'max_depth': max_depth,
                                        'min_child_samples': 80,
                                        'subsample': 0.8,
                                        'colsample_bytree': 0.8,
                                        'reg_alpha': 0.1,
                                        'reg_lambda': 0.2,
                                        'class_weight': class_weight,
                                    },
                                    'cv_method': 'kfold',
                                    'cv_n_splits': 5,
                                    'cv_embargo_bars': 12,
                                    'features_config': {},
                                    'feature_set_name': 'baseline',
                                    'labeling_params': {
                                        'pt_mult': 2.0,  # Best from Batch 1
                                        'sl_mult': 1.5,
                                        'time_mult': 40,
                                    }
                                }
                                configs.append(config)
                                config_id += 1

                                if len(configs) >= 200:
                                    break
                            if len(configs) >= 200:
                                break
                        else:
                            # No decay parameter for uniform/uniqueness
                            config = {
                                'exp_id': f'batch6_sample_weight_{config_id:03d}',
                                'phase': 'batch6_sample_weight',
                                'labeling_method': 'triple_barrier',
                                'sample_weight': sample_weight,
                                'sample_weight_params': {},
                                'class_weight': class_weight,
                                'model_kind': 'lightgbm',
                                'model_params': {
                                    'n_estimators': 400,
                                    'learning_rate': lr,
                                    'num_leaves': num_leaves,
                                    'max_depth': max_depth,
                                    'min_child_samples': 80,
                                    'subsample': 0.8,
                                    'colsample_bytree': 0.8,
                                    'reg_alpha': 0.1,
                                    'reg_lambda': 0.2,
                                    'class_weight': class_weight,
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

                            if len(configs) >= 200:
                                break

                    if len(configs) >= 200:
                        break
                if len(configs) >= 200:
                    break
            if len(configs) >= 200:
                break
        if len(configs) >= 200:
            break

    return configs[:200]

if __name__ == '__main__':
    configs = generate_batch6_configs()

    # Save individual config files
    output_dir = Path('batch6_sample_weight_configs')
    output_dir.mkdir(exist_ok=True)

    for config in configs:
        config_path = output_dir / f"{config['exp_id']}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    # Save summary
    summary = {
        'batch': 'batch6_sample_weight',
        'n_configs': len(configs),
        'sample_weights': list(set(c['sample_weight'] for c in configs)),
        'class_weights': list(set(str(c.get('class_weight')) for c in configs)),
    }

    with open('batch6_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Generated {len(configs)} configs for Batch 6")
    print(f"   Sample weights: {summary['sample_weights']}")
    print(f"   Class weights: {summary['class_weights']}")
    print(f"   Saved to: {output_dir}/")
