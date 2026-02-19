"""
Batch 13: CUSUM Event Filtering
Sweeps CUSUM thresholds and event filter types across model hyperparameters with CPCV.
6×3×2×3×2 = 216 → first 200.
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


def generate_batch13_configs():
    configs = []
    config_id = 1

    cusum_mults = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    event_filters = ['cusum', 'all_bars', 'volatility_breakout']
    num_leaves_opts = [31, 63]
    max_depth_opts = [6, 10, 15]
    lr_opts = [0.01, 0.03]

    for cusum_mult in cusum_mults:
        for event_filter in event_filters:
            for num_leaves in num_leaves_opts:
                for max_depth in max_depth_opts:
                    for lr in lr_opts:
                        config = {
                            'exp_id': f'batch13_cusum_{config_id:03d}',
                            'phase': 'batch13_cusum',
                            'architecture': 'single_model',
                            'labeling_method': 'triple_barrier',
                            'sample_weight': 'uniqueness',
                            'feature_set_name': 'baseline',
                            'cusum_threshold_atr_mult': cusum_mult,
                            'event_filter': event_filter,
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
    configs = generate_batch13_configs()
    out_dir = Path('batch13_cusum_configs')
    out_dir.mkdir(exist_ok=True)
    for c in configs:
        with open(out_dir / f"{c['exp_id']}.json", 'w') as f:
            json.dump(c, f, indent=2)

    ef_counts = Counter(c['event_filter'] for c in configs)
    cm_counts = Counter(c['cusum_threshold_atr_mult'] for c in configs)
    nl_counts = Counter(c['model_params']['num_leaves'] for c in configs)

    print(f"Generated {len(configs)} configs for Batch 13 (CUSUM event filtering)")
    print("  Event filter breakdown:")
    for k, v in sorted(ef_counts.items()):
        print(f"    {k}: {v} configs")
    print("  CUSUM ATR multiplier breakdown:")
    for k, v in sorted(cm_counts.items()):
        print(f"    {k}: {v} configs")
    print("  num_leaves breakdown:")
    for k, v in sorted(nl_counts.items()):
        print(f"    {k}: {v} configs")
