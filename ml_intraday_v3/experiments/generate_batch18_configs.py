"""
Batch 18: Leak-Free Barrier Grid (CPCV only)

Replicates Batch 1 barrier search with proper CPCV to eliminate any
temporal leakage. Fixed model, sweeps pt_mult x sl_mult x time_mult.
100 barrier combos x 2 CV variant passes = 200 configs total.
"""
import json
from pathlib import Path


FIXED_MODEL_PARAMS = {
    'n_estimators': 400,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': 10,
    'min_child_samples': 80,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.2,
}

PT_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]
SL_MULTS = [0.5, 1.0, 1.5, 2.0]
HZ_VALUES = [20, 30, 40, 60, 80]

CV_VARIANTS = [
    {
        'cv_method': 'cpcv',
        'cv_n_splits': 5,
        'cv_n_test_splits': 2,
        'cv_purge_pct': 0.02,
        'cv_embargo_pct': 0.01,
        'cv_embargo_bars': 12,
    },
    {
        'cv_method': 'cpcv',
        'cv_n_splits': 6,
        'cv_n_test_splits': 2,
        'cv_purge_pct': 0.02,
        'cv_embargo_pct': 0.01,
        'cv_embargo_bars': 12,
    },
]


def generate_batch18_configs():
    configs = []
    config_id = 1

    for cv_block in CV_VARIANTS:
        for pt in PT_MULTS:
            for sl in SL_MULTS:
                for hz in HZ_VALUES:
                    config = {
                        'exp_id': f'batch18_barriers_{config_id:03d}',
                        'phase': 'batch18_barriers',
                        'architecture': 'single_model',
                        'labeling_method': 'triple_barrier',
                        'labeling_params': {
                            'pt_mult': pt,
                            'sl_mult': sl,
                            'time_mult': hz,
                        },
                        'sample_weight': 'uniqueness',
                        'feature_set_name': 'baseline',
                        'features_config': {},
                        'model_kind': 'lightgbm',
                        'model_params': dict(FIXED_MODEL_PARAMS),
                        **cv_block,
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

    return configs[:200]


if __name__ == '__main__':
    configs = generate_batch18_configs()
    out_dir = Path('batch18_barriers_configs')
    out_dir.mkdir(exist_ok=True)
    for c in configs:
        with open(out_dir / f"{c['exp_id']}.json", 'w') as f:
            json.dump(c, f, indent=2)
    print(f"Generated {len(configs)} configs")
