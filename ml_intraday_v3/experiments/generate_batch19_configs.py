"""
Batch 19: Ensemble Voting

Sweeps ensemble compositions (LightGBM, RandomForest, ExtraTrees combos)
against voting method, weighting strategy, balance method, and session mode.
Two labeling passes bring total to ~192 configs; padded to 200 with extra
labeling variants.
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

MEMBER_PRESETS = {
    'lgb_duo': [
        {"name": "lgb31", "model_kind": "lightgbm", "model_params": {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.03, "max_depth": 6}},
        {"name": "lgb127", "model_kind": "lightgbm", "model_params": {"n_estimators": 200, "num_leaves": 127, "learning_rate": 0.01, "max_depth": 10}},
    ],
    'lgb_rf': [
        {"name": "lgb63", "model_kind": "lightgbm", "model_params": {"n_estimators": 200, "num_leaves": 63, "learning_rate": 0.03, "max_depth": 8}},
        {"name": "rf200", "model_kind": "random_forest", "model_params": {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 20}},
    ],
    'lgb_et': [
        {"name": "lgb63", "model_kind": "lightgbm", "model_params": {"n_estimators": 200, "num_leaves": 63, "learning_rate": 0.03, "max_depth": 8}},
        {"name": "et200", "model_kind": "extra_trees", "model_params": {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 20}},
    ],
    'lgb_rf_et': [
        {"name": "lgb63", "model_kind": "lightgbm", "model_params": {"n_estimators": 200, "num_leaves": 63, "learning_rate": 0.03, "max_depth": 8}},
        {"name": "rf200", "model_kind": "random_forest", "model_params": {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 20}},
        {"name": "et200", "model_kind": "extra_trees", "model_params": {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 20}},
    ],
    'lgb_trio': [
        {"name": "lgb31", "model_kind": "lightgbm", "model_params": {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05, "max_depth": 6}},
        {"name": "lgb63", "model_kind": "lightgbm", "model_params": {"n_estimators": 300, "num_leaves": 63, "learning_rate": 0.03, "max_depth": 8}},
        {"name": "lgb127", "model_kind": "lightgbm", "model_params": {"n_estimators": 400, "num_leaves": 127, "learning_rate": 0.01, "max_depth": 12}},
    ],
    'full_quad': [
        {"name": "lgb31", "model_kind": "lightgbm", "model_params": {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05, "max_depth": 6}},
        {"name": "lgb127", "model_kind": "lightgbm", "model_params": {"n_estimators": 400, "num_leaves": 127, "learning_rate": 0.01, "max_depth": 10}},
        {"name": "rf200", "model_kind": "random_forest", "model_params": {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 20}},
        {"name": "et200", "model_kind": "extra_trees", "model_params": {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 20}},
    ],
}

LABELING_PASSES = [
    {'pt_mult': 2.0, 'sl_mult': 1.5, 'time_mult': 40},
    {'pt_mult': 2.5, 'sl_mult': 2.0, 'time_mult': 60},
]

EXTRA_LABELING_PASSES = [
    {'pt_mult': 1.5, 'sl_mult': 1.0, 'time_mult': 30},
    {'pt_mult': 3.0, 'sl_mult': 2.0, 'time_mult': 80},
]

VOTING_METHODS = ['soft', 'hard']
ENSEMBLE_WEIGHTS = ['equal', 'auc_weighted']
BALANCE_METHODS = ['undersample', 'none']
SESSION_MODES = ['rth_eth', 'rth_only']


def generate_batch19_configs():
    configs = []
    config_id = 1

    def add_config(preset_name, voting_method, ensemble_weight, balance_method, session_mode, labeling_params):
        nonlocal config_id
        if len(configs) >= 200:
            return

        members = [dict(m) for m in MEMBER_PRESETS[preset_name]]

        # auc_weighted uses None for ensemble_weights_list (post-processing computes proper weights)
        ensemble_weights_list = None

        config = {
            'exp_id': f'batch19_ensemble_{config_id:03d}',
            'phase': 'batch19_ensemble',
            'architecture': 'single_model',
            'labeling_method': 'triple_barrier',
            'labeling_params': dict(labeling_params),
            'sample_weight': 'uniqueness',
            'feature_set_name': 'baseline',
            'features_config': {},
            'model_kind': 'ensemble_voting',
            'ensemble_preset': preset_name,
            'ensemble_members': members,
            'voting_method': voting_method,
            'ensemble_weights': ensemble_weight,
            'ensemble_weights_list': ensemble_weights_list,
            'balance_method': balance_method,
            'session_mode': session_mode,
            **CPCV_BLOCK,
        }
        configs.append(config)
        config_id += 1

    # Pass 1 and 2: 6 presets x 2 voting x 2 weights x 2 balance x 2 session x 2 labeling = 192
    for labeling_params in LABELING_PASSES:
        for preset_name in MEMBER_PRESETS:
            for voting_method in VOTING_METHODS:
                for ensemble_weight in ENSEMBLE_WEIGHTS:
                    for balance_method in BALANCE_METHODS:
                        for session_mode in SESSION_MODES:
                            add_config(preset_name, voting_method, ensemble_weight, balance_method, session_mode, labeling_params)

    # Pad to 200 with extra labeling variants (cycle through extra passes)
    extra_idx = 0
    presets = list(MEMBER_PRESETS.keys())
    while len(configs) < 200:
        labeling_params = EXTRA_LABELING_PASSES[extra_idx % len(EXTRA_LABELING_PASSES)]
        preset_name = presets[extra_idx % len(presets)]
        add_config(preset_name, 'soft', 'equal', 'none', 'rth_eth', labeling_params)
        extra_idx += 1

    return configs[:200]


if __name__ == '__main__':
    configs = generate_batch19_configs()
    out_dir = Path('batch19_ensemble_configs')
    out_dir.mkdir(exist_ok=True)
    for c in configs:
        with open(out_dir / f"{c['exp_id']}.json", 'w') as f:
            json.dump(c, f, indent=2)
    print(f"Generated {len(configs)} configs")
