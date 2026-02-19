#!/usr/bin/env python3
"""
Batch Validation of 10 Finalist Models on Jan-Feb 2026 OOS Data

Tests the 10 selected finalist models from batch1 and batch3 experiments
on the 41-day out-of-sample period (Jan 1 - Feb 10, 2026).

Strategy:
1. Load top 100 model configs from ranked results
2. Load historical training data (Oct 2024 - Nov 2025)
3. For each config:
   - Generate events with config's labeling params
   - Build features with config's feature set
   - Train model with config's params on historical data
   - Test on Jan-Feb 2026
4. Rank by Jan-Feb 2026 performance
5. Report top performers

This script trains models on-the-fly, so no need to pre-promote all 100.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from ml_intraday_v3.core.instrument import InstrumentSpec
from ml_intraday_v3.features.build import build_features
from ml_intraday_v3.labels.events import generate_events, balance_events
from ml_intraday_v3.labels.triple_barrier import apply_triplebarrier


def load_training_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical training data."""
    logger.info("\nLoading historical training data...")

    # Load MES bars from historical dataset
    data_path = project_root / "data" / "processed" / "mes_bars_databento_rth.h5"

    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        sys.exit(1)

    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()

    if bars.index.tz is None:
        bars = bars.tz_localize('UTC')

    # Use Oct 2024 - Nov 2025 as training window (matches baseline)
    train_start = pd.Timestamp('2024-10-01', tz='UTC')
    train_end = pd.Timestamp('2025-11-30 23:59:59', tz='UTC')

    bars_train = bars[(bars.index >= train_start) & (bars.index <= train_end)].copy()

    logger.info(f"  Training period: {train_start.date()} to {train_end.date()}")
    logger.info(f"  Training bars: {len(bars_train):,}")

    return bars, bars_train


def load_test_data() -> pd.DataFrame:
    """Load Jan-Feb 2026 test data."""
    logger.info("\nLoading Jan-Feb 2026 test data...")

    test_path = project_root / "data" / "processed" / "jan_feb_2026_oos_test.h5"

    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        logger.info("Run fetch_jan_feb_2026_data.py first!")
        sys.exit(1)

    bars_test = pd.read_hdf(test_path, key='bars_5min')

    logger.info(f"  Test bars: {len(bars_test):,}")
    logger.info(f"  Date range: {bars_test.index[0]} to {bars_test.index[-1]}")

    return bars_test


def train_model_from_config(
    config: Dict,
    bars_train: pd.DataFrame,
    base_configs: Dict,
    instrument_spec: InstrumentSpec
) -> Optional[Dict]:
    """Train a model from experiment config on historical data."""

    exp_id = config['exp_id']

    try:
        # Generate events with config's labeling params
        labeling_config = base_configs['labeling'].copy()
        if 'labeling' in config:
            labeling_config.update(config['labeling'])

        events = generate_events(
            bars_df=bars_train,
            bar_size="5m",
            labeling_config=labeling_config,
            execution_spec=base_configs['execution'],
        )

        events = apply_triplebarrier(
            bars_df=bars_train,
            events_df=events,
            bar_size="5m",
            labeling_config=labeling_config,
            execution_spec=base_configs['execution'],
            instrument_spec=instrument_spec,
        )

        # Drop vertical barriers
        events = events[events['y'] != 0].reset_index(drop=True)

        # Balance events if specified
        balance_method = config.get('balance_method', 'undersample')
        if balance_method:
            events = balance_events(events, target_long_ratio=0.50, method=balance_method)

        logger.info(f"    Events: {len(events):,}")

        # Build features
        features_config = base_configs['features'].copy()
        if config.get('feature_set_name') == 'momentum_on':
            features_config['momentum'] = {'enabled': True}

        features = build_features(bars_train, "5m", features_config)

        # Merge events with features
        t0_list = events['t0'].tolist()
        feat_aligned = features.reindex(t0_list).reset_index(drop=True)

        dataset = pd.concat([
            events[['side', 'y']].reset_index(drop=True),
            feat_aligned,
        ], axis=1)

        # Binary labels
        y_train = (dataset['y'] == 1).astype(int)

        # Drop NaN rows
        valid = ~dataset.isna().any(axis=1)
        dataset = dataset[valid]
        y_train = y_train[valid]

        if len(dataset) < 100:
            logger.warning(f"    Insufficient samples: {len(dataset)}")
            return None

        # Remove y and side from features
        feature_cols = [c for c in dataset.columns if c not in ['y', 'side']]
        X_train = dataset[feature_cols].values

        # Sample weights
        sample_weight_method = config.get('sample_weight', 'uniform')
        if sample_weight_method == 'class_balanced':
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y_train)
        else:
            sample_weights = np.ones(len(y_train))

        # Train model
        model_name = config.get('model_name', 'lightgbm')
        model_params = config.get('model_params', {})

        if model_name == 'lightgbm':
            base_model = LGBMClassifier(
                objective='binary',
                random_state=42,
                verbose=-1,
                **model_params
            )
        elif model_name == 'random_forest':
            base_model = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                **model_params
            )
        elif model_name == 'logistic_regression':
            base_model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                **model_params
            )
        elif model_name == 'mlp':
            base_model = MLPClassifier(
                random_state=42,
                max_iter=500,
                **model_params
            )
        else:
            logger.warning(f"    Unknown model: {model_name}")
            return None

        # Train base model
        logger.info(f"    Training {model_name}...")
        base_model.fit(X_train, y_train, sample_weight=sample_weights)

        # Apply calibration if specified
        calibration_method = config.get('calibration', None)
        if calibration_method == 'isotonic':
            logger.info(f"    Applying isotonic calibration...")
            # Simple train/cal split
            from sklearn.model_selection import train_test_split
            X_model, X_cal, y_model, y_cal = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            base_model_recal = base_model.__class__(**model_params, random_state=42)
            if model_name == 'lightgbm':
                base_model_recal.set_params(objective='binary', verbose=-1)
            base_model_recal.fit(X_model, y_model)

            cal_probs = base_model_recal.predict_proba(X_cal)[:, 1]
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(cal_probs, y_cal)

            model = CalibratedBinaryModel(base_model_recal, calibrator)
        elif calibration_method == 'platt':
            model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
            model.fit(X_train, y_train)
        else:
            model = base_model

        # Create bundle
        bundle = {
            'primary_model': model,
            'primary_preprocessor': None,
            'primary_feature_columns': feature_cols,
            'has_side_feature': 'side' in dataset.columns,
            'thresholds': {'primary_threshold': 0.1},
        }

        logger.info(f"    ✓ Trained successfully ({len(feature_cols)} features)")

        return bundle

    except Exception as e:
        logger.error(f"    Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


class CalibratedBinaryModel:
    """Wrapper for isotonic calibrated model."""
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator
        self.classes_ = base_model.classes_

    def predict_proba(self, X):
        raw_proba = self.base_model.predict_proba(X)[:, 1]
        calibrated_p1 = self.calibrator.predict(raw_proba)
        calibrated_p0 = 1.0 - calibrated_p1
        return np.column_stack([calibrated_p0, calibrated_p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def evaluate_on_test(
    bundle: Dict,
    events_test: pd.DataFrame,
    features_test: pd.DataFrame,
    threshold: float = 0.40
) -> Dict:
    """Evaluate model on test data."""

    model = bundle['primary_model']
    feature_cols = bundle['primary_feature_columns']

    # Prepare test dataset
    t0_list = events_test['t0'].tolist()
    feat_aligned = features_test.reindex(t0_list).reset_index(drop=True)

    dataset = pd.concat([
        events_test[['side', 'y', 'ret_net']].reset_index(drop=True),
        feat_aligned,
    ], axis=1)

    y_true = (dataset['y'] == 1).astype(int)
    valid = ~dataset.isna().any(axis=1)
    dataset_clean = dataset[valid].copy()
    y_true_clean = y_true[valid]

    # Check features
    missing = set(feature_cols) - set(dataset_clean.columns)
    if missing:
        return {'status': 'missing_features', 'missing': list(missing)[:5]}

    X_test = dataset_clean[feature_cols].values

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except:
        return {'status': 'prediction_failed'}

    # Filter by threshold
    high_conf_mask = y_prob > threshold
    n_signals = high_conf_mask.sum()

    if n_signals == 0:
        return {
            'status': 'success',
            'signals': 0,
            'pnl': np.nan,
            'sharpe': np.nan,
            'win_rate': np.nan,
        }

    ret_net_filtered = dataset_clean['ret_net'].values[high_conf_mask]
    wins = ret_net_filtered > 0

    total_pnl = ret_net_filtered.sum()
    mean_pnl = ret_net_filtered.mean()
    win_rate = wins.sum() / n_signals

    if len(ret_net_filtered) > 1 and ret_net_filtered.std() > 0:
        sharpe = (mean_pnl / ret_net_filtered.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    return {
        'status': 'success',
        'signals': int(n_signals),
        'pnl': float(total_pnl),
        'sharpe': float(sharpe),
        'win_rate': float(win_rate),
        'mean_pnl': float(mean_pnl),
    }


def main():
    logger.info("="*80)
    logger.info("BATCH VALIDATION: 10 FINALIST MODELS ON JAN-FEB 2026")
    logger.info("="*80)

    # Load base configs
    ml_root = project_root / "ml_intraday_v3"
    config_dir = ml_root / "configs"

    with open(config_dir / "labeling.yaml") as f:
        labeling_config = yaml.safe_load(f)
    with open(config_dir / "execution_spec.yaml") as f:
        execution_spec = yaml.safe_load(f)
    with open(config_dir / "features.yaml") as f:
        features_config = yaml.safe_load(f)

    base_configs = {
        'labeling': labeling_config,
        'execution': execution_spec,
        'features': features_config,
    }

    instrument_spec = InstrumentSpec(
        symbol="MES",
        tick_size_points=0.25,
        contract_multiplier_usd_per_point=5.0,
    )

    # Load 10 finalist configs
    finalist_file = Path("/tmp/finalist_configs_for_validation.json")

    # Load finalist configs
    logger.info("\nLoading 10 finalist configs...")
    with open(finalist_file) as f:
        configs = json.load(f)

    logger.info(f"  Loaded {len(configs)} finalist model configs")

    # Rank by CV performance
    configs_df = pd.DataFrame(configs)
    if 'composite_score' in configs_df.columns:
        configs_df = configs_df.sort_values('composite_score', ascending=False)
    elif 'median_test_auc' in configs_df.columns:
        configs_df = configs_df.sort_values('median_test_auc', ascending=False)

    configs_df = configs_df.head(100)
    logger.info(f"  Testing top {len(configs_df)} models by CV performance")

    # Load data
    bars_all, bars_train = load_training_data()
    bars_test = load_test_data()

    # Generate test events and features (same for all models with base config)
    logger.info("\nGenerating test events and features...")
    events_test = generate_events(
        bars_df=bars_test,
        bar_size="5m",
        labeling_config=base_configs['labeling'],
        execution_spec=base_configs['execution'],
    )
    events_test = apply_triplebarrier(
        bars_df=bars_test,
        events_df=events_test,
        bar_size="5m",
        labeling_config=base_configs['labeling'],
        execution_spec=base_configs['execution'],
        instrument_spec=instrument_spec,
    )
    events_test = events_test[events_test['y'] != 0].reset_index(drop=True)
    logger.info(f"  Test events: {len(events_test):,}")

    # Generate test features (base and momentum)
    logger.info("  Generating base test features...")
    features_test_base = build_features(bars_test, "5m", base_configs['features'])

    logger.info("  Generating momentum test features...")
    features_config_momentum = base_configs['features'].copy()
    features_config_momentum['momentum'] = {'enabled': True}
    features_test_momentum = build_features(bars_test, "5m", features_config_momentum)

    # Test each model
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING {len(configs_df)} MODELS")
    logger.info(f"{'='*80}")

    results = []

    for idx, row in configs_df.iterrows():
        exp_id = row['exp_id']
        logger.info(f"\n[{len(results)+1}/{len(configs_df)}] {exp_id}")
        cv_auc = row.get('median_test_auc', np.nan)
        if pd.notna(cv_auc):
            logger.info(f"  CV AUC: {cv_auc:.4f}")
        else:
            logger.info(f"  CV AUC: N/A")

        start_time = time.time()

        # Train model
        bundle = train_model_from_config(
            row.to_dict(),
            bars_train,
            base_configs,
            instrument_spec
        )

        if bundle is None:
            results.append({
                'exp_id': exp_id,
                'status': 'train_failed',
                'cv_auc': row.get('median_test_auc'),
            })
            continue

        # Select appropriate test features
        has_momentum = row.get('feature_set_name') == 'momentum_on'
        features_test = features_test_momentum if has_momentum else features_test_base

        # Evaluate
        eval_result = evaluate_on_test(bundle, events_test, features_test, threshold=0.40)

        # Merge results
        result = {
            'exp_id': exp_id,
            'model_name': row.get('model_name'),
            'cv_auc': row.get('median_test_auc'),
            **eval_result
        }

        results.append(result)

        elapsed = time.time() - start_time

        if eval_result['status'] == 'success':
            logger.info(f"  Test @ 0.40: PnL=${eval_result.get('pnl', 0):.2f}, "
                       f"Signals={eval_result.get('signals', 0)}, "
                       f"Sharpe={eval_result.get('sharpe', 0):.2f}")
        else:
            logger.info(f"  Status: {eval_result['status']}")

        logger.info(f"  Time: {elapsed:.1f}s")

        # Save checkpoint every 10 models
        if (len(results)) % 10 == 0:
            temp_df = pd.DataFrame(results)
            temp_path = ml_root / "diagnostics" / f"batch_temp_{len(results)}.csv"
            temp_df.to_csv(temp_path, index=False)
            logger.info(f"  💾 Checkpoint saved")

    # Final results
    logger.info("\n" + "="*80)
    logger.info("RANKING RESULTS")
    logger.info("="*80)

    results_df = pd.DataFrame(results)

    # Filter successful
    successful = results_df[results_df['status'] == 'success'].copy()
    successful = successful[successful['signals'] > 0]

    if len(successful) == 0:
        logger.error("❌ No models generated signals!")
        sys.exit(1)

    # Rank by PnL
    successful = successful.sort_values('pnl', ascending=False)

    # Display top 20
    logger.info("\nTOP 20 MODELS BY JAN-FEB 2026 PERFORMANCE:")
    logger.info(f"{'Rank':<6} {'Exp ID':<25} {'PnL':<12} {'Sharpe':<10} {'Signals':<10} {'Win%':<8} {'CV AUC':<10}")
    logger.info("-" * 95)

    for rank, (idx, row) in enumerate(successful.head(20).iterrows(), 1):
        logger.info(
            f"{rank:<6} "
            f"{row['exp_id']:<25} "
            f"${row['pnl']:>10.2f} "
            f"{row['sharpe']:>9.2f} "
            f"{row['signals']:>9} "
            f"{100*row['win_rate']:>6.1f}% "
            f"{row['cv_auc']:>9.4f}"
        )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = ml_root / "diagnostics" / f"finalist_validation_jan_feb_2026_{timestamp}.csv"
    results_df.to_csv(output_path, index=False)

    logger.info(f"\n💾 Full results saved: {output_path}")

    # Check for profitable models
    profitable = successful[successful['pnl'] > 0]

    logger.info("\n" + "="*80)
    logger.info("FINAL VERDICT")
    logger.info("="*80)

    if len(profitable) > 0:
        logger.info(f"✅ FOUND {len(profitable)} PROFITABLE MODELS!")

        best = profitable.iloc[0]
        logger.info(f"\nBest model: {best['exp_id']}")
        logger.info(f"  PnL: ${best['pnl']:.2f}")
        logger.info(f"  Sharpe: {best['sharpe']:.2f}")
        logger.info(f"  Win Rate: {100*best['win_rate']:.1f}%")
        logger.info(f"  Signals: {best['signals']}")
        logger.info(f"\n✅ RECOMMENDATION: Promote this model to production")
    else:
        logger.info(f"❌ NO PROFITABLE MODELS FOUND")
        logger.info(f"  Best PnL: ${successful.iloc[0]['pnl']:.2f}")
        logger.info(f"\n  Consider:")
        logger.info(f"    - Retraining with updated data")
        logger.info(f"    - Exploring different parameter spaces")
        logger.info(f"    - Returning to rule-based system")


if __name__ == "__main__":
    main()
