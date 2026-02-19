"""
Batch 11: Session Mode Sweep
Tests rth_only, eth_only, rth_eth session modes across CV methods and model hyperparameters.
"""
import json
from pathlib import Path
from collections import Counter


def generate_batch11_configs():
    configs = []
    config_id = 1

    session_modes = ['rth_only', 'eth_only', 'rth_eth']
    num_leaves_opts = [31, 63, 127]
    max_depth_opts = [6, 10, 15, -1]
    lr_opts = [0.01, 0.03, 0.05]

    cv_variants = [
        {
            'cv_method': 'kfold',
            'cv_n_splits': 5,
            'cv_embargo_bars': 12,
        },
        {
            'cv_method': 'cpcv',
            'cv_n_splits': 5,
            'cv_n_test_splits': 2,
            'cv_purge_pct': 0.02,
            'cv_embargo_pct': 0.01,
            'cv_embargo_bars': 12,
        },
    ]

    labeling = {'pt_mult': 2.0, 'sl_mult': 1.5, 'time_mult': 40}

    for cv in cv_variants:
        for session_mode in session_modes:
            for num_leaves in num_leaves_opts:
                for max_depth in max_depth_opts:
                    for lr in lr_opts:
                        config = {
                            'exp_id': f'batch11_session_{config_id:03d}',
                            'phase': 'batch11_session',
                            'architecture': 'single_model',
                            'labeling_method': 'triple_barrier',
                            'sample_weight': 'uniqueness',
                            'feature_set_name': 'baseline',
                            'session_mode': session_mode,
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
                            'cv_method': cv['cv_method'],
                            'cv_n_splits': cv['cv_n_splits'],
                            'cv_embargo_bars': cv.get('cv_embargo_bars', 12),
                            'labeling_params': labeling,
                        }
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
        if len(configs) >= 200:
            break

    return configs[:200]


if __name__ == '__main__':
    configs = generate_batch11_configs()
    out_dir = Path('batch11_session_configs')
    out_dir.mkdir(exist_ok=True)
    for c in configs:
        with open(out_dir / f"{c['exp_id']}.json", 'w') as f:
            json.dump(c, f, indent=2)

    session_counts = Counter(c['session_mode'] for c in configs)
    cv_counts = Counter(c['cv_method'] for c in configs)
    nl_counts = Counter(c['model_params']['num_leaves'] for c in configs)

    print(f"Generated {len(configs)} configs for Batch 11 (session mode sweep)")
    print("  Session mode breakdown:")
    for k, v in sorted(session_counts.items()):
        print(f"    {k}: {v} configs")
    print("  CV method breakdown:")
    for k, v in sorted(cv_counts.items()):
        print(f"    {k}: {v} configs")
    print("  num_leaves breakdown:")
    for k, v in sorted(nl_counts.items()):
        print(f"    {k}: {v} configs")
