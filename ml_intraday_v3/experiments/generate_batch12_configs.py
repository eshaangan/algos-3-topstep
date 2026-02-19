"""
Batch 12: Alternative Model Kinds
Tests Random Forest, Extra Trees, Logistic Regression, MLP, and LightGBM baseline
across CPCV with fixed labeling params.
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

FIXED = {
    'phase': 'batch12_altmodel',
    'architecture': 'single_model',
    'labeling_method': 'triple_barrier',
    'sample_weight': 'uniqueness',
    'feature_set_name': 'baseline',
    'labeling_params': LABELING,
    **CPCV,
}


def _base(config_id, model_kind, model_params):
    c = dict(FIXED)
    c['exp_id'] = f'batch12_altmodel_{config_id:03d}'
    c['model_kind'] = model_kind
    c['model_params'] = model_params
    return c


def generate_batch12_configs():
    configs = []
    config_id = 1

    # Group 1: Random Forest — 3×4×4 = 48 configs
    for n_estimators in [100, 200, 400]:
        for max_depth in [5, 10, 20, None]:
            for min_samples_leaf in [10, 20, 50, 100]:
                configs.append(_base(config_id, 'random_forest', {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_leaf': min_samples_leaf,
                    'class_weight': 'balanced',
                }))
                config_id += 1

    # Group 2: Extra Trees — 3×4×4 = 48 configs
    for n_estimators in [100, 200, 400]:
        for max_depth in [5, 10, 20, None]:
            for min_samples_leaf in [10, 20, 50, 100]:
                configs.append(_base(config_id, 'extra_trees', {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_leaf': min_samples_leaf,
                    'class_weight': 'balanced',
                }))
                config_id += 1

    # Group 3: Logistic Regression — 14 + 5 = 20 configs (truncated/padded to 20)
    logreg_configs = []
    for c_val in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        for penalty in ['l1', 'l2']:
            logreg_configs.append({
                'C': c_val,
                'penalty': penalty,
                'solver': 'saga',
            })
    for c_val in [0.001, 0.01, 0.1, 1.0, 10.0]:
        logreg_configs.append({
            'C': c_val,
            'penalty': 'l2',
            'solver': 'saga',
            'max_iter': 3000,
        })
    # One additional config to reach 20 total
    logreg_configs.append({
        'C': 0.1,
        'penalty': 'l1',
        'solver': 'saga',
        'max_iter': 3000,
    })
    for mp in logreg_configs[:20]:
        configs.append(_base(config_id, 'logreg', mp))
        config_id += 1

    # Group 4: MLP — ~24 configs
    # 5 hidden_layer_sizes × 2 activations × up to 3 alphas, capped at 24
    hidden_sizes = [[64, 32], [128, 64], [256, 128, 64], [64, 64], [128, 128]]
    activations = ['relu', 'tanh']
    alphas = [0.0001, 0.001, 0.01]
    mlp_count = 0
    for hs in hidden_sizes:
        for act in activations:
            for alpha in alphas:
                if mlp_count >= 24:
                    break
                configs.append(_base(config_id, 'mlp', {
                    'hidden_layer_sizes': hs,
                    'activation': act,
                    'alpha': alpha,
                }))
                config_id += 1
                mlp_count += 1
            if mlp_count >= 24:
                break
        if mlp_count >= 24:
            break

    # Group 5: LightGBM baseline — 36 base + 24 extra n_estimators = 60 configs
    lgb_count = 0
    for num_leaves in [31, 63, 127]:
        for max_depth in [6, 10, 15, -1]:
            for lr in [0.01, 0.03, 0.05]:
                configs.append(_base(config_id, 'lightgbm', {
                    'n_estimators': 400,
                    'learning_rate': lr,
                    'num_leaves': num_leaves,
                    'max_depth': max_depth,
                    'min_child_samples': 80,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.2,
                }))
                config_id += 1
                lgb_count += 1

    # Extra LightGBM with n_estimators variants to reach 60 total (36 already done, need 24 more)
    extra_lgb = 0
    for n_est in [200, 600]:
        for num_leaves in [31, 63, 127]:
            for max_depth in [6, 10, -1, 15]:
                for lr in [0.01, 0.03, 0.05]:
                    if extra_lgb >= 24:
                        break
                    configs.append(_base(config_id, 'lightgbm', {
                        'n_estimators': n_est,
                        'learning_rate': lr,
                        'num_leaves': num_leaves,
                        'max_depth': max_depth,
                        'min_child_samples': 80,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'reg_alpha': 0.1,
                        'reg_lambda': 0.2,
                    }))
                    config_id += 1
                    extra_lgb += 1
                if extra_lgb >= 24:
                    break
            if extra_lgb >= 24:
                break
        if extra_lgb >= 24:
            break

    return configs[:200]


if __name__ == '__main__':
    configs = generate_batch12_configs()
    out_dir = Path('batch12_altmodel_configs')
    out_dir.mkdir(exist_ok=True)
    for c in configs:
        with open(out_dir / f"{c['exp_id']}.json", 'w') as f:
            json.dump(c, f, indent=2)

    kind_counts = Counter(c['model_kind'] for c in configs)

    print(f"Generated {len(configs)} configs for Batch 12 (alternative model kinds)")
    print("  Model kind breakdown:")
    for k, v in sorted(kind_counts.items()):
        print(f"    {k}: {v} configs")
