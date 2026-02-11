#!/usr/bin/env python3
"""
Feature Importance Analysis for Binary Calibrated Model

Analyzes feature importance using:
1. LightGBM gain-based importance
2. Permutation importance on test set
3. Identifies harmful features (negative permutation importance)
4. Optionally retrains without harmful features and compares AUC

Usage:
    python ml_intraday_v3/diagnostics/feature_importance_analysis.py \
        --model ml_intraday_v3/models/saved/model_bundle_binary_calibrated_v1.pkl
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from ml_intraday_v3.features.build import build_features
from ml_intraday_v3.labels.events import generate_events, balance_events
from ml_intraday_v3.labels.triple_barrier import apply_triplebarrier
from ml_intraday_v3.core.instrument import InstrumentSpec
from ml_intraday_v3.train_balanced_model import CalibratedBinaryModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_test_data():
    """Load Dec 2025 test set with binary labels."""
    data_path = Path("../data/processed/mes_bars_databento_rth.h5")
    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()

    test_start = pd.Timestamp('2025-12-01', tz='UTC')
    test_end = pd.Timestamp('2025-12-31 23:59:59', tz='UTC')
    bars_test = bars[(bars.index >= test_start) & (bars.index <= test_end)]

    with open("configs/labeling.yaml") as f:
        labeling_config = yaml.safe_load(f)
    with open("configs/execution_spec.yaml") as f:
        execution_spec = yaml.safe_load(f)
    with open("configs/features.yaml") as f:
        feature_config = yaml.safe_load(f)

    instrument_spec = InstrumentSpec(symbol="MES", tick_size_points=0.25, contract_multiplier_usd_per_point=5.0)

    events_test = generate_events(bars_df=bars_test, bar_size="5m",
                                   labeling_config=labeling_config, execution_spec=execution_spec)
    events_test = apply_triplebarrier(bars_df=bars_test, events_df=events_test, bar_size="5m",
                                       labeling_config=labeling_config, execution_spec=execution_spec,
                                       instrument_spec=instrument_spec)

    # Drop vertical barriers for binary
    events_test = events_test[events_test['y'] != 0].reset_index(drop=True)

    features_test = build_features(bars_test, "5m", feature_config)

    t0_test = events_test['t0'].tolist() if 't0' in events_test.columns else events_test.index.tolist()
    features_at_t0 = features_test.reindex(t0_test)

    dataset = pd.concat([
        events_test[['side', 'y']].reset_index(drop=True),
        features_at_t0.reset_index(drop=True)
    ], axis=1)

    y = (dataset['y'] == 1).astype(int)
    valid = ~dataset.isna().any(axis=1)
    dataset = dataset[valid]
    y = y[valid]
    dataset = dataset.drop(columns=['y'])

    return dataset, y


def analyze_importance(model_path: str):
    """Run full feature importance analysis."""
    logger.info("Loading model bundle...")
    bundle = joblib.load(model_path)
    model = bundle['primary_model']
    feature_columns = bundle['primary_feature_columns']
    is_binary = bundle.get('is_binary', False)

    logger.info(f"Model type: {type(model).__name__}")
    logger.info(f"Binary: {is_binary}")
    logger.info(f"Features: {len(feature_columns)}")

    # Load test data
    logger.info("\nLoading test data...")
    dataset, y_test = load_test_data()
    X_test = dataset[feature_columns]
    logger.info(f"Test set: {len(X_test)} samples")

    # 1. Gain-based importance (from underlying LightGBM)
    logger.info("\n--- Gain-Based Feature Importance ---")
    base_model = model
    # Unwrap CalibratedClassifierCV if needed
    if hasattr(model, 'calibrated_classifiers_'):
        base_model = model.calibrated_classifiers_[0].estimator

    if hasattr(base_model, 'feature_importances_'):
        gain_importance = pd.Series(base_model.feature_importances_, index=feature_columns)
        gain_importance = gain_importance.sort_values(ascending=False)
        logger.info("\nTop 20 features by gain:")
        for feat, imp in gain_importance.head(20).items():
            logger.info(f"  {feat:40s} {imp:.1f}")

        # Save to CSV
        output_dir = Path("diagnostics")
        gain_importance.to_csv(output_dir / "feature_importance_gain.csv", header=['gain_importance'])
        logger.info(f"\nSaved gain importance to {output_dir / 'feature_importance_gain.csv'}")
    else:
        logger.warning("Model does not have feature_importances_ attribute")
        gain_importance = None

    # 2. Permutation importance on test set
    logger.info("\n--- Permutation Importance (test set) ---")

    # Compute baseline AUC
    baseline_proba = model.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, baseline_proba)
    logger.info(f"Baseline AUC: {baseline_auc:.4f}")

    # Manual permutation importance (sklearn version has compatibility issues)
    n_repeats = 10
    importances = []

    for feat_idx, feat_name in enumerate(feature_columns):
        logger.info(f"  Computing permutation importance for {feat_name} ({feat_idx+1}/{len(feature_columns)})...")
        scores = []
        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            X_permuted[feat_name] = np.random.permutation(X_test[feat_name].values)
            permuted_proba = model.predict_proba(X_permuted)[:, 1]
            permuted_auc = roc_auc_score(y_test, permuted_proba)
            scores.append(baseline_auc - permuted_auc)  # Drop in AUC
        importances.append({
            'mean': np.mean(scores),
            'std': np.std(scores)
        })

    perm_importance = pd.DataFrame(importances, index=feature_columns).sort_values('mean', ascending=False)

    logger.info("\nTop 20 features by permutation importance:")
    for feat, row in perm_importance.head(20).iterrows():
        logger.info(f"  {feat:40s} {row['mean']:.4f} +/- {row['std']:.4f}")

    # 3. Identify harmful features
    harmful = perm_importance[perm_importance['mean'] < 0]
    logger.info(f"\n--- Harmful Features (negative permutation importance): {len(harmful)} ---")
    if len(harmful) > 0:
        for feat, row in harmful.iterrows():
            logger.info(f"  {feat:40s} {row['mean']:.4f} +/- {row['std']:.4f}")

        # Features that are clearly harmful (mean - std < 0)
        clearly_harmful = harmful[harmful['mean'] - harmful['std'] < 0]
        logger.info(f"\nClearly harmful (mean - std < 0): {len(clearly_harmful)}")
        for feat in clearly_harmful.index:
            logger.info(f"  PRUNE: {feat}")
    else:
        logger.info("  No harmful features detected.")

    # Save permutation importance
    output_dir = Path("ml_intraday_v3/diagnostics")
    perm_importance.to_csv(output_dir / "feature_importance_permutation.csv")
    logger.info(f"\nSaved permutation importance to {output_dir / 'feature_importance_permutation.csv'}")

    # 4. Baseline AUC
    baseline_proba = model.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, baseline_proba)
    logger.info(f"\nBaseline AUC (all features): {baseline_auc:.4f}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Total features: {len(feature_columns)}")
    logger.info(f"Harmful features: {len(harmful)}")
    if len(harmful) > 0:
        clearly_harmful = harmful[harmful['mean'] - harmful['std'] < 0]
        logger.info(f"Clearly harmful: {len(clearly_harmful)}")
        logger.info(f"Recommendation: Retrain without {len(clearly_harmful)} clearly harmful features")
        logger.info(f"Features to prune: {list(clearly_harmful.index)}")
    else:
        logger.info("No features recommended for pruning.")
    logger.info(f"Baseline test AUC: {baseline_auc:.4f}")


def sequential_forward_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    initial_features: list = None,
    max_features: int = 25,
    model_params: dict = None,
) -> dict:
    """Sequential Forward Selection (SFS) for feature pruning.

    Starts with top-K features by permutation importance, greedily adds
    features that improve OOS AUC.

    Args:
        X_train/X_test: Feature DataFrames
        y_train/y_test: Binary labels
        initial_features: Starting feature set (top-5 by default)
        max_features: Maximum number of features to select
        model_params: LightGBM parameters

    Returns:
        Dict with selected features, AUC trajectory, etc.
    """
    from lightgbm import LGBMClassifier

    if model_params is None:
        model_params = {
            'n_estimators': 150, 'learning_rate': 0.05,
            'num_leaves': 31, 'max_depth': 6,
            'min_child_samples': 100, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        }

    all_features = list(X_train.columns)
    if initial_features is None:
        initial_features = all_features[:5]

    selected = list(initial_features)
    remaining = [f for f in all_features if f not in selected]

    def _eval_features(feat_list):
        model = LGBMClassifier(objective='binary', random_state=42, verbose=-1, **model_params)
        model.fit(X_train[feat_list], y_train)
        proba = model.predict_proba(X_test[feat_list])[:, 1]
        return roc_auc_score(y_test, proba)

    # Baseline with initial features
    best_auc = _eval_features(selected)
    trajectory = [{'n_features': len(selected), 'features': list(selected), 'auc': best_auc}]
    logger.info(f"SFS start: {len(selected)} features, AUC={best_auc:.4f}")

    while remaining and len(selected) < max_features:
        best_new_feature = None
        best_new_auc = best_auc

        for candidate in remaining:
            trial_features = selected + [candidate]
            try:
                auc = _eval_features(trial_features)
                if auc > best_new_auc:
                    best_new_auc = auc
                    best_new_feature = candidate
            except Exception:
                continue

        if best_new_feature is None or best_new_auc <= best_auc:
            logger.info(f"SFS stopped: no improvement at {len(selected)} features")
            break

        selected.append(best_new_feature)
        remaining.remove(best_new_feature)
        best_auc = best_new_auc
        trajectory.append({
            'n_features': len(selected),
            'features': list(selected),
            'auc': best_auc,
            'added': best_new_feature,
        })
        logger.info(f"  +{best_new_feature:40s} -> {len(selected)} features, AUC={best_auc:.4f}")

    return {
        'selected_features': selected,
        'final_auc': best_auc,
        'n_features': len(selected),
        'trajectory': trajectory,
    }


def run_feature_selection(model_path: str, output_path: str = None):
    """Run full feature selection: permutation importance + SFS.

    Produces a feature_selection.json file that can be consumed by
    train_balanced_model.py to restrict the feature set.
    """
    from lightgbm import LGBMClassifier

    logger.info("="*60)
    logger.info("FEATURE SELECTION PIPELINE")
    logger.info("="*60)

    # Load model and data
    bundle = joblib.load(model_path)
    model = bundle['primary_model']
    feature_columns = bundle['primary_feature_columns']

    dataset, y_test = load_test_data()
    X_test = dataset[feature_columns]

    # Step 1: Permutation importance
    logger.info("\nStep 1: Permutation importance...")

    def auc_scorer(model, X, y):
        proba = model.predict_proba(X)[:, 1]
        return roc_auc_score(y, proba)

    perm_result = permutation_importance(
        model, X_test, y_test,
        scoring=auc_scorer,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    perm_df = pd.DataFrame({
        'mean': perm_result.importances_mean,
        'std': perm_result.importances_std,
    }, index=feature_columns).sort_values('mean', ascending=False)

    # Remove clearly harmful features (mean - std < 0)
    clearly_harmful = perm_df[perm_df['mean'] - perm_df['std'] < 0].index.tolist()
    surviving = [f for f in feature_columns if f not in clearly_harmful]
    logger.info(f"  Pruned {len(clearly_harmful)} harmful features: {clearly_harmful}")
    logger.info(f"  Surviving: {len(surviving)} features")

    # Step 2: Need train data for SFS
    logger.info("\nStep 2: Loading training data for SFS...")
    data_path = Path("data/processed/mes_bars_databento_rth.h5")
    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()

    with open("configs/labeling.yaml") as f:
        labeling_config = yaml.safe_load(f)
    with open("configs/execution_spec.yaml") as f:
        execution_spec = yaml.safe_load(f)
    with open("configs/features.yaml") as f:
        feature_config = yaml.safe_load(f)
    with open("ml_intraday_v3/configs/training.yaml") as f:
        training_config = yaml.safe_load(f)

    instrument_spec = InstrumentSpec(symbol="MES", tick_size_points=0.25, contract_multiplier_usd_per_point=5.0)

    data_sel = training_config.get('data_selection', {})
    train_start = pd.Timestamp(data_sel.get('train_start', '2025-06-01'), tz='UTC')
    train_end = pd.Timestamp(data_sel.get('train_end', '2025-11-30') + ' 23:59:59', tz='UTC')

    bars_train = bars[(bars.index >= train_start) & (bars.index <= train_end)]

    events_train = generate_events(bars_df=bars_train, bar_size="5m",
                                    labeling_config=labeling_config, execution_spec=execution_spec)
    events_train = apply_triplebarrier(bars_df=bars_train, events_df=events_train, bar_size="5m",
                                        labeling_config=labeling_config, execution_spec=execution_spec,
                                        instrument_spec=instrument_spec)
    events_train = balance_events(events_train, target_long_ratio=0.50, method='undersample')
    events_train = events_train[events_train['y'] != 0].reset_index(drop=True)

    features_train = build_features(bars_train, "5m", feature_config)
    t0_train = events_train['t0'].tolist()
    feat_at_t0 = features_train.reindex(t0_train).reset_index(drop=True)

    ds_train = pd.concat([events_train[['side', 'y']].reset_index(drop=True), feat_at_t0], axis=1)
    y_train = (ds_train['y'] == 1).astype(int)
    valid = ~ds_train.isna().any(axis=1)
    ds_train = ds_train[valid]
    y_train = y_train[valid]
    ds_train = ds_train.drop(columns=['y'])

    X_train_sfs = ds_train[surviving]
    X_test_sfs = X_test[surviving]

    # Step 3: SFS with top-5 from permutation importance as initial set
    logger.info("\nStep 3: Sequential Forward Selection...")
    top_5 = perm_df.head(5).index.tolist()
    top_5_surviving = [f for f in top_5 if f in surviving]

    sfs_result = sequential_forward_selection(
        X_train=X_train_sfs,
        y_train=y_train,
        X_test=X_test_sfs,
        y_test=y_test,
        initial_features=top_5_surviving,
        max_features=25,
    )

    # Save results
    if output_path is None:
        output_path = "ml_intraday_v3/diagnostics/feature_selection.json"

    import json
    results = {
        'selected_features': sfs_result['selected_features'],
        'n_features': sfs_result['n_features'],
        'final_auc': sfs_result['final_auc'],
        'pruned_features': clearly_harmful,
        'permutation_importance': {
            feat: {'mean': float(row['mean']), 'std': float(row['std'])}
            for feat, row in perm_df.iterrows()
        },
        'sfs_trajectory': sfs_result['trajectory'],
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")
    logger.info(f"Selected {sfs_result['n_features']} features with AUC={sfs_result['final_auc']:.4f}")
    logger.info(f"Features: {sfs_result['selected_features']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature importance analysis")
    parser.add_argument("--model", type=str,
                        default="ml_intraday_v3/models/saved/model_bundle_binary_calibrated_v1.pkl",
                        help="Path to model bundle")
    parser.add_argument("--mode", type=str, default="analyze",
                        choices=["analyze", "select"],
                        help="'analyze' for importance analysis, 'select' for full selection pipeline")
    args = parser.parse_args()

    if args.mode == "analyze":
        analyze_importance(args.model)
    else:
        run_feature_selection(args.model)
