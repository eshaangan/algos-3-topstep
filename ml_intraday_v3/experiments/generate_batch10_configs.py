"""
Batch 10: Purged Walk-Forward CV (CPCV)
Tests proper temporal cross-validation with purging and embargo to reduce overfitting.
"""
import json
from pathlib import Path

def generate_batch10_configs():
    configs = []
    config_id = 1

    # CV method variants
    cv_methods = [
        # CPCV with varying embargo/purge
        {'cv_method': 'cpcv', 'cv_n_splits': 5, 'cv_n_test_splits': 2,
         'cv_purge_pct': 0.01, 'cv_embargo_pct': 0.005, 'cv_embargo_bars': 6},
        {'cv_method': 'cpcv', 'cv_n_splits': 5, 'cv_n_test_splits': 2,
         'cv_purge_pct': 0.02, 'cv_embargo_pct': 0.01,  'cv_embargo_bars': 12},
        {'cv_method': 'cpcv', 'cv_n_splits': 6, 'cv_n_test_splits': 2,
         'cv_purge_pct': 0.02, 'cv_embargo_pct': 0.01,  'cv_embargo_bars': 12},
        {'cv_method': 'cpcv', 'cv_n_splits': 5, 'cv_n_test_splits': 2,
         'cv_purge_pct': 0.03, 'cv_embargo_pct': 0.02,  'cv_embargo_bars': 24},
        # Purged kfold (middle ground)
        {'cv_method': 'kfold', 'cv_n_splits': 5, 'cv_embargo_bars': 6},
        {'cv_method': 'kfold', 'cv_n_splits': 5, 'cv_embargo_bars': 24},
        {'cv_method': 'kfold', 'cv_n_splits': 5, 'cv_embargo_bars': 48},
        {'cv_method': 'kfold', 'cv_n_splits': 8, 'cv_embargo_bars': 12},
    ]

    # Model params — good coverage around Batch 1 best
    num_leaves_opts = [31, 63, 127]
    max_depth_opts  = [6, 10, 15, -1]
    lr_opts         = [0.01, 0.03, 0.05]

    # Labeling — use Batch 1's known-good barrier
    labeling = {'pt_mult': 2.0, 'sl_mult': 1.5, 'time_mult': 40}

    for cv in cv_methods:
        for num_leaves in num_leaves_opts:
            for max_depth in max_depth_opts:
                for lr in lr_opts:
                    config = {
                        'exp_id': f'batch10_walkfwd_{config_id:03d}',
                        'phase': 'batch10_walkforward',
                        'architecture': 'single_model',
                        'labeling_method': 'triple_barrier',
                        'sample_weight': 'uniqueness',
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
                        },
                        'cv_method':       cv['cv_method'],
                        'cv_n_splits':     cv['cv_n_splits'],
                        'cv_embargo_bars': cv.get('cv_embargo_bars', 12),
                        'features_config': {},
                        'feature_set_name': 'baseline',
                        'labeling_params': labeling,
                    }
                    # Add CPCV-specific keys if present
                    for k in ('cv_n_test_splits', 'cv_purge_pct', 'cv_embargo_pct'):
                        if k in cv:
                            config[k] = cv[k]

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
    configs = generate_batch10_configs()
    out_dir = Path('batch10_walkfwd_configs')
    out_dir.mkdir(exist_ok=True)
    for c in configs:
        with open(out_dir / f"{c['exp_id']}.json", 'w') as f:
            json.dump(c, f, indent=2)

    cv_counts = {}
    for c in configs:
        k = f"{c['cv_method']}_{c['cv_embargo_bars']}bars"
        cv_counts[k] = cv_counts.get(k, 0) + 1

    print(f"✅ Generated {len(configs)} configs for Batch 10 (walk-forward)")
    print(f"   CV method breakdown:")
    for k, v in sorted(cv_counts.items()):
        print(f"     {k}: {v} configs")
