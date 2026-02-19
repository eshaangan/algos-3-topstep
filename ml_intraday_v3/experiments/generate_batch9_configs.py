"""
Batch 9: Meta-Labeling
Two-stage architecture: primary direction model + secondary trade-filter.
"""
import json
from pathlib import Path

def generate_batch9_configs():
    configs = []
    config_id = 1

    # Primary model params (high recall focus)
    primary_leaves   = [31, 63, 127]
    primary_depths   = [6, 10, 15]
    primary_lr       = [0.01, 0.03, 0.05]

    # Secondary model threshold (how confident secondary must be to allow trade)
    secondary_thresholds = [0.45, 0.50, 0.55, 0.60]

    # Labeling params — use Batch 1's best barrier settings
    labeling_options = [
        {'pt_mult': 2.0, 'sl_mult': 1.5, 'time_mult': 40},
        {'pt_mult': 2.5, 'sl_mult': 1.5, 'time_mult': 40},
        {'pt_mult': 2.0, 'sl_mult': 1.0, 'time_mult': 40},
    ]

    for labeling in labeling_options:
        for num_leaves in primary_leaves:
            for max_depth in primary_depths:
                for lr in primary_lr:
                    for sec_thresh in secondary_thresholds:
                        config = {
                            'exp_id': f'batch9_metalabel_{config_id:03d}',
                            'phase': 'batch9_metalabeling',
                            'architecture': 'meta_labeling',
                            'labeling_method': 'triple_barrier',
                            'sample_weight': 'uniqueness',
                            'model_kind': 'lightgbm',
                            'model_params': {
                                'n_estimators': 400,
                                'learning_rate': lr,
                                'num_leaves': num_leaves,
                                'max_depth': max_depth,
                                'min_child_samples': 50,   # lower = higher recall
                                'subsample': 0.8,
                                'colsample_bytree': 0.8,
                                'reg_alpha': 0.1,
                                'reg_lambda': 0.2,
                            },
                            'secondary_threshold': sec_thresh,
                            'cv_method': 'kfold',
                            'cv_n_splits': 5,
                            'cv_embargo_bars': 12,
                            'features_config': {},
                            'feature_set_name': 'baseline',
                            'labeling_params': labeling,
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
    configs = generate_batch9_configs()
    out_dir = Path('batch9_metalabel_configs')
    out_dir.mkdir(exist_ok=True)
    for c in configs:
        with open(out_dir / f"{c['exp_id']}.json", 'w') as f:
            json.dump(c, f, indent=2)
    print(f"✅ Generated {len(configs)} configs for Batch 9 (meta-labeling)")
    print(f"   Secondary thresholds: {sorted(set(c['secondary_threshold'] for c in configs))}")
    print(f"   Labeling params: {len(set(str(c['labeling_params']) for c in configs))} variants")
