"""
Batch 15: Balance Method + Event Ratios
Sweeps class balance method, long/short ratio, and vertical barrier dropping with CPCV.
4×5×2×2×2×2 = 320 → first 200.
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


def generate_batch15_configs():
    configs = []
    config_id = 1

    balance_methods = ['none', 'undersample', 'oversample', 'class_balanced']
    target_long_ratios = [0.33, 0.40, 0.50, 0.60, 0.67]
    drop_vb_opts = [True, False]
    num_leaves_opts = [31, 63]
    lr_opts = [0.01, 0.03]
    max_depth_opts = [6, 10]

    for balance_method in balance_methods:
        for target_long_ratio in target_long_ratios:
            for drop_vb in drop_vb_opts:
                for num_leaves in num_leaves_opts:
                    for lr in lr_opts:
                        for max_depth in max_depth_opts:
                            config = {
                                'exp_id': f'batch15_balance_{config_id:03d}',
                                'phase': 'batch15_balance',
                                'architecture': 'single_model',
                                'labeling_method': 'triple_barrier',
                                'sample_weight': 'uniqueness',
                                'feature_set_name': 'baseline',
                                'balance_method': balance_method,
                                'target_long_ratio': target_long_ratio,
                                'drop_vertical_barrier': drop_vb,
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
        if len(configs) >= 200:
            break

    return configs[:200]


if __name__ == '__main__':
    configs = generate_batch15_configs()
    out_dir = Path('batch15_balance_configs')
    out_dir.mkdir(exist_ok=True)
    for c in configs:
        with open(out_dir / f"{c['exp_id']}.json", 'w') as f:
            json.dump(c, f, indent=2)

    bm_counts = Counter(c['balance_method'] for c in configs)
    lr_counts = Counter(c['target_long_ratio'] for c in configs)
    dvb_counts = Counter(c['drop_vertical_barrier'] for c in configs)
    nl_counts = Counter(c['model_params']['num_leaves'] for c in configs)

    print(f"Generated {len(configs)} configs for Batch 15 (balance method + event ratios)")
    print("  Balance method breakdown:")
    for k, v in sorted(bm_counts.items()):
        print(f"    {k}: {v} configs")
    print("  Target long ratio breakdown:")
    for k, v in sorted(lr_counts.items()):
        print(f"    {k}: {v} configs")
    print("  Drop vertical barrier breakdown:")
    for k, v in sorted(dvb_counts.items()):
        print(f"    {k}: {v} configs")
    print("  num_leaves breakdown:")
    for k, v in sorted(nl_counts.items()):
        print(f"    {k}: {v} configs")
