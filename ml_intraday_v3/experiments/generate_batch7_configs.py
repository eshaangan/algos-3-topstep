"""
Generate Batch 7: Feature Engineering
Tests different feature combinations and transformations.
"""
import json
import numpy as np
from pathlib import Path

def generate_batch7_configs():
    """Generate 200 configs testing feature combinations."""

    # Feature set variations
    feature_sets = [
        {
            'name': 'baseline',
            'config': {}
        },
        {
            'name': 'baseline_momentum',
            'config': {
                'momentum': {
                    'enabled': True,
                    'rsi_period': 14,
                    'include_divergence': True
                }
            }
        },
        {
            'name': 'baseline_volume',
            'config': {
                'volume': {
                    'enabled': True,
                    'vwap': True,
                    'volume_delta': True,
                    'obv': True
                }
            }
        },
        {
            'name': 'baseline_volatility',
            'config': {
                'volatility': {
                    'enabled': True,
                    'bbands': True,
                    'keltner': True,
                    'atr_features': True
                }
            }
        },
        {
            'name': 'kitchen_sink',
            'config': {
                'momentum': {'enabled': True, 'rsi_period': 14, 'include_divergence': True},
                'volume': {'enabled': True, 'vwap': True, 'volume_delta': True, 'obv': True},
                'volatility': {'enabled': True, 'bbands': True, 'keltner': True, 'atr_features': True}
            }
        }
    ]

    # Model variations
    num_leaves_opts = [31, 63, 127]
    max_depth_opts = [6, 10, 15, -1]
    learning_rates = [0.01, 0.03, 0.05]
    min_child_samples_opts = [50, 80, 100]

    configs = []
    config_id = 1

    for feature_set in feature_sets:
        for num_leaves in num_leaves_opts:
            for max_depth in max_depth_opts:
                for lr in learning_rates:
                    for min_child_samples in min_child_samples_opts:
                        config = {
                            'exp_id': f'batch7_features_{config_id:03d}',
                            'phase': 'batch7_features',
                            'labeling_method': 'triple_barrier',
                            'sample_weight': 'uniqueness',  # Best from testing
                            'model_kind': 'lightgbm',
                            'model_params': {
                                'n_estimators': 400,
                                'learning_rate': lr,
                                'num_leaves': num_leaves,
                                'max_depth': max_depth,
                                'min_child_samples': min_child_samples,
                                'subsample': 0.8,
                                'colsample_bytree': 0.8,
                                'reg_alpha': 0.1,
                                'reg_lambda': 0.2,
                            },
                            'cv_method': 'kfold',
                            'cv_n_splits': 5,
                            'cv_embargo_bars': 12,
                            'features_config': feature_set['config'],
                            'feature_set_name': feature_set['name'],
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
    configs = generate_batch7_configs()

    # Save individual config files
    output_dir = Path('batch7_features_configs')
    output_dir.mkdir(exist_ok=True)

    for config in configs:
        config_path = output_dir / f"{config['exp_id']}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    # Save summary
    summary = {
        'batch': 'batch7_features',
        'n_configs': len(configs),
        'feature_sets': list(set(c['feature_set_name'] for c in configs)),
    }

    with open('batch7_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Generated {len(configs)} configs for Batch 7")
    print(f"   Feature sets: {summary['feature_sets']}")
    print(f"   Saved to: {output_dir}/")
