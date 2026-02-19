"""
Batch 17: Feature Normalization Methods

Sweeps normalization strategies (zscore, robust, minmax, quantile, none) and
fractional differencing against LightGBM hyperparameters using CPCV.
"""
import json
from pathlib import Path


CPCV_BLOCK = {
    'cv_method': 'cpcv',
    'cv_n_splits': 5,
    'cv_n_test_splits': 2,
    'cv_purge_pct': 0.02,
    'cv_embargo_pct': 0.01,
    'cv_embargo_bars': 12,
}

FIXED_LABELING = {
    'pt_mult': 2.0,
    'sl_mult': 1.5,
    'time_mult': 40,
}

NORMALIZATION_METHODS = ['none', 'zscore', 'robust', 'minmax', 'quantile']
NORMALIZATION_WINDOWS = ['expanding', 'rolling_252', 'rolling_63']
FRACDIFF_DS = [None, 0.3, 0.4, 0.5]
NUM_LEAVES_OPTS = [31, 63]
LEARNING_RATES = [0.01, 0.03]


def generate_batch17_configs():
    configs = []
    config_id = 1

    for normalization_method in NORMALIZATION_METHODS:
        for normalization_window in NORMALIZATION_WINDOWS:
            for fracdiff_d in FRACDIFF_DS:
                for num_leaves in NUM_LEAVES_OPTS:
                    for lr in LEARNING_RATES:
                        features_config = {
                            'normalization': {
                                'method': normalization_method if normalization_method != 'none' else None,
                                'window': normalization_window,
                            },
                            'fracdiff_d': fracdiff_d if fracdiff_d is not None else None,
                            'enable_fracdiff': fracdiff_d is not None,
                        }

                        config = {
                            'exp_id': f'batch17_norm_{config_id:03d}',
                            'phase': 'batch17_norm',
                            'architecture': 'single_model',
                            'labeling_method': 'triple_barrier',
                            'labeling_params': dict(FIXED_LABELING),
                            'sample_weight': 'uniqueness',
                            'feature_set_name': 'baseline',
                            'features_config': features_config,
                            'normalization_method': normalization_method,
                            'normalization_window': normalization_window,
                            'fracdiff_d': fracdiff_d,
                            'model_kind': 'lightgbm',
                            'model_params': {
                                'n_estimators': 400,
                                'learning_rate': lr,
                                'num_leaves': num_leaves,
                                'max_depth': 10,
                                'min_child_samples': 80,
                                'subsample': 0.8,
                                'colsample_bytree': 0.8,
                                'reg_alpha': 0.1,
                                'reg_lambda': 0.2,
                            },
                            **CPCV_BLOCK,
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
    configs = generate_batch17_configs()
    out_dir = Path('batch17_norm_configs')
    out_dir.mkdir(exist_ok=True)
    for c in configs:
        with open(out_dir / f"{c['exp_id']}.json", 'w') as f:
            json.dump(c, f, indent=2)
    print(f"Generated {len(configs)} configs")
