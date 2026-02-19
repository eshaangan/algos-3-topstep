"""
Batch 16: HMM Regime + Momentum Feature Sets

Sweeps feature set combinations (baseline, hmm, momentum, multiresolution)
against LightGBM hyperparameters using CPCV cross-validation.
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

FEATURE_SETS = ['baseline', 'baseline_hmm', 'baseline_momentum', 'baseline_hmm_momentum', 'baseline_multiresolution']
HMM_STATES = [2, 3]
MOMENTUM_PERIODS = [[7, 14], [14, 28]]
NUM_LEAVES_OPTS = [31, 63, 127]
MAX_DEPTH_OPTS = [6, 10, 15]
LEARNING_RATES = [0.01, 0.03]


def build_features_config(feature_set_name, hmm_states, mom_periods):
    if feature_set_name == 'baseline_hmm':
        return {'enable_hmm_regime': True, 'hmm_n_states': hmm_states}
    if feature_set_name == 'baseline_momentum':
        return {'enable_momentum_features': True, 'momentum_periods': mom_periods}
    if feature_set_name == 'baseline_hmm_momentum':
        return {
            'enable_hmm_regime': True,
            'hmm_n_states': hmm_states,
            'enable_momentum_features': True,
            'momentum_periods': mom_periods,
        }
    if feature_set_name == 'baseline_multiresolution':
        return {'enable_multiresolution': True, 'multiresolution_windows': [15, 30, 60]}
    return {}


def generate_batch16_configs():
    configs = []
    config_id = 1

    for feature_set_name in FEATURE_SETS:
        for hmm_states in HMM_STATES:
            for mom_periods in MOMENTUM_PERIODS:
                for num_leaves in NUM_LEAVES_OPTS:
                    for max_depth in MAX_DEPTH_OPTS:
                        for lr in LEARNING_RATES:
                            features_config = build_features_config(feature_set_name, hmm_states, mom_periods)

                            config = {
                                'exp_id': f'batch16_hmmmom_{config_id:03d}',
                                'phase': 'batch16_hmmmom',
                                'architecture': 'single_model',
                                'labeling_method': 'triple_barrier',
                                'labeling_params': dict(FIXED_LABELING),
                                'sample_weight': 'uniqueness',
                                'feature_set_name': feature_set_name,
                                'features_config': features_config,
                                'hmm_n_states': hmm_states,
                                'momentum_periods': mom_periods,
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
        if len(configs) >= 200:
            break

    return configs[:200]


if __name__ == '__main__':
    configs = generate_batch16_configs()
    out_dir = Path('batch16_hmmmom_configs')
    out_dir.mkdir(exist_ok=True)
    for c in configs:
        with open(out_dir / f"{c['exp_id']}.json", 'w') as f:
            json.dump(c, f, indent=2)
    print(f"Generated {len(configs)} configs")
