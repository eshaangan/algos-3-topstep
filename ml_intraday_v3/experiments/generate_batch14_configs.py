"""
Batch 14: Training Window Length
Sweeps training window size and retrain frequency with CPCV.
7×3×3×2×3 = 378 → first 200.
"""
import json
from pathlib import Path
from collections import Counter


CPCV = {
    'cv_method': 'cpcv',
    'cv_n_splits': 5,
    'cv_n_test_splits': 2,
    'cv_purge_pct': 0.02,
    'cv_embargo_pct': 0.01,
    'cv_embargo_bars': 12,
}

LABELING = {'pt_mult': 2.0, 'sl_mult': 1.5, 'time_mult': 40}


def generate_batch14_configs():
    configs = []
    config_id = 1

    training_windows = [2, 3, 4, 6, 9, 12, None]
    retrain_freqs = [1, 2, 3]
    num_leaves_opts = [31, 63, 127]
    lr_opts = [0.01, 0.03]
    max_depth_opts = [6, 10, -1]

    for training_window in training_windows:
        for retrain_freq in retrain_freqs:
            for num_leaves in num_leaves_opts:
                for lr in lr_opts:
                    for max_depth in max_depth_opts:
                        config = {
                            'exp_id': f'batch14_trainwindow_{config_id:03d}',
                            'phase': 'batch14_trainwindow',
                            'architecture': 'single_model',
                            'labeling_method': 'triple_barrier',
                            'sample_weight': 'uniqueness',
                            'feature_set_name': 'baseline',
                            'training_window_months': training_window,
                            'retrain_frequency_months': retrain_freq,
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
                            'cv_method': CPCV['cv_method'],
                            'cv_n_splits': CPCV['cv_n_splits'],
                            'cv_n_test_splits': CPCV['cv_n_test_splits'],
                            'cv_purge_pct': CPCV['cv_purge_pct'],
                            'cv_embargo_pct': CPCV['cv_embargo_pct'],
                            'cv_embargo_bars': CPCV['cv_embargo_bars'],
                            'labeling_params': LABELING,
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
    configs = generate_batch14_configs()
    out_dir = Path('batch14_trainwindow_configs')
    out_dir.mkdir(exist_ok=True)
    for c in configs:
        with open(out_dir / f"{c['exp_id']}.json", 'w') as f:
            json.dump(c, f, indent=2)

    tw_counts = Counter(c['training_window_months'] for c in configs)
    rf_counts = Counter(c['retrain_frequency_months'] for c in configs)
    nl_counts = Counter(c['model_params']['num_leaves'] for c in configs)

    print(f"Generated {len(configs)} configs for Batch 14 (training window length)")
    print("  Training window (months) breakdown:")
    for k, v in sorted(tw_counts.items(), key=lambda x: (x[0] is None, x[0])):
        print(f"    {k}: {v} configs")
    print("  Retrain frequency (months) breakdown:")
    for k, v in sorted(rf_counts.items()):
        print(f"    {k}: {v} configs")
    print("  num_leaves breakdown:")
    for k, v in sorted(nl_counts.items()):
        print(f"    {k}: {v} configs")
