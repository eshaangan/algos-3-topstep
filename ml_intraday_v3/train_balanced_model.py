#!/usr/bin/env python3
"""
Train Balanced Bidirectional Model

Simplified training script that:
1. Loads 2024-2025 data (no 2026 leak)
2. Generates balanced 50/50 LONG/SHORT events
3. Applies triple-barrier labeling
4. Drops vertical barrier events for binary classification
5. Trains LightGBM model (binary or multiclass from config)
6. Applies isotonic probability calibration (if enabled)
7. Saves model bundle for testing
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ml_intraday_v3.labels.events import generate_events, balance_events
from ml_intraday_v3.labels.triple_barrier import apply_triplebarrier
from ml_intraday_v3.core.instrument import InstrumentSpec
from ml_intraday_v3.features.build import build_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class SigmoidCalibratorWrapper:
    """Wrapper for LogisticRegression calibrator to match IsotonicRegression API.

    IsotonicRegression.predict() takes 1D array and returns 1D array.
    LogisticRegression.predict_proba() takes 2D array and returns 2D array.
    This wrapper bridges the gap.
    """
    def __init__(self, logistic_model):
        self.logistic_model = logistic_model

    def predict(self, raw_proba):
        """Takes 1D raw probabilities, returns 1D calibrated probabilities."""
        return self.logistic_model.predict_proba(raw_proba.reshape(-1, 1))[:, 1]


class CalibratedBinaryModel:
    """Wraps a binary classifier with calibration (isotonic or sigmoid).

    Provides the same predict/predict_proba interface as sklearn classifiers
    so it can be used as a drop-in replacement in the live predictor.
    """

    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator
        # Expose classes_ for downstream code that checks it
        self.classes_ = base_model.classes_

    def predict_proba(self, X):
        raw_proba = self.base_model.predict_proba(X)[:, 1]
        calibrated_p1 = self.calibrator.predict(raw_proba)
        calibrated_p0 = 1.0 - calibrated_p1
        return np.column_stack([calibrated_p0, calibrated_p1])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def feature_importances_(self):
        return self.base_model.feature_importances_

    def fit(self, X, y):
        """Dummy fit method for sklearn compatibility (e.g., permutation_importance)."""
        # Model is already trained, this is just for interface compatibility
        return self


def main():
    logger.info("="*80)
    logger.info("BALANCED BIDIRECTIONAL MODEL TRAINING")
    logger.info("="*80)

    # Load data (2024-2025 only, no 2026)
    data_path = Path("data/processed/mes_bars_databento_rth.h5")
    logger.info(f"\nLoading data from: {data_path}")

    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()

    logger.info(f"   Full dataset: {len(bars):,} bars ({bars.index[0].date()} to {bars.index[-1].date()})")

    # Load configs FIRST (Phase 3: need training_config for date ranges)
    logger.info("\nLoading configurations...")

    with open("ml_intraday_v3/configs/labeling.yaml") as f:
        labeling_config = yaml.safe_load(f)

    with open("ml_intraday_v3/configs/execution_spec.yaml") as f:
        execution_spec = yaml.safe_load(f)

    with open("ml_intraday_v3/configs/features.yaml") as f:
        feature_config = yaml.safe_load(f)

    with open("ml_intraday_v3/configs/training.yaml") as f:
        training_config = yaml.safe_load(f)

    # Split train/test (CRITICAL: no 2026 data)
    # Phase 3: Use config-driven date ranges (default: recent 6 months)
    data_sel = training_config.get('data_selection', {})
    train_start = pd.Timestamp(data_sel.get('train_start', '2025-06-01'), tz='UTC')
    train_end = pd.Timestamp(data_sel.get('train_end', '2025-11-30') + ' 23:59:59', tz='UTC')
    test_start = pd.Timestamp(data_sel.get('test_start', '2025-12-01'), tz='UTC')
    test_end = pd.Timestamp(data_sel.get('test_end', '2025-12-31') + ' 23:59:59', tz='UTC')

    bars_train = bars[(bars.index >= train_start) & (bars.index <= train_end)]
    bars_test = bars[(bars.index >= test_start) & (bars.index <= test_end)]

    logger.info(f"\nTrain period: {bars_train.index[0].date()} to {bars_train.index[-1].date()} ({len(bars_train):,} bars)")
    logger.info(f"Test period:  {bars_test.index[0].date()} to {bars_test.index[-1].date()} ({len(bars_test):,} bars)")

    # Create instrument spec for MES
    instrument_spec = InstrumentSpec(
        symbol="MES",
        tick_size_points=0.25,
        contract_multiplier_usd_per_point=5.0
    )

    # Generate events (train)
    event_policy = (labeling_config.get("primary_labeling") or {}).get("event_policy", "cusum")
    logger.info(f"\nGenerating events (event_policy={event_policy})...")
    events_train = generate_events(
        bars_df=bars_train,
        bar_size="5m",
        labeling_config=labeling_config,
        execution_spec=execution_spec,
    )
    logger.info(f"   Train events: {len(events_train):,}")

    # Apply triple-barrier labeling
    logger.info("\nApplying triple-barrier labeling...")
    events_train = apply_triplebarrier(
        bars_df=bars_train,
        events_df=events_train,
        bar_size="5m",
        labeling_config=labeling_config,
        execution_spec=execution_spec,
        instrument_spec=instrument_spec
    )
    logger.info(f"   Labeled events: {len(events_train):,}")
    logger.info(f"   Label distribution: {events_train['y'].value_counts().to_dict()}")

    # Check side distribution before balancing
    if 'side' in events_train.columns:
        long_pct_before = (events_train['side'] == 1).sum() / len(events_train) * 100
        logger.info(f"   Side distribution (before balance): {long_pct_before:.1f}% LONG")

    # Balance events (50/50 LONG/SHORT)
    logger.info("\nBalancing LONG/SHORT events...")
    events_train = balance_events(
        events=events_train,
        target_long_ratio=0.50,
        method='undersample'
    )

    # --- Phase 3: Sample Decay Weighting ---
    decay_cfg = training_config.get('sample_decay', {})
    if decay_cfg.get('enabled', False) and 't0' in events_train.columns:
        decay_lambda = decay_cfg.get('lambda', 0.005)
        ref_date = decay_cfg.get('reference_date')
        if ref_date is None:
            ref_date = events_train['t0'].max()
        else:
            ref_date = pd.Timestamp(ref_date, tz='UTC')

        age_days = (ref_date - events_train['t0']).dt.total_seconds() / 86400.0
        w_decay = np.exp(-decay_lambda * age_days)
        half_life_days = np.log(2) / decay_lambda

        if 'w_final' in events_train.columns:
            events_train['w_final'] = events_train['w_final'] * w_decay
        else:
            events_train['w_final'] = w_decay

        logger.info(f"\nSample decay applied: lambda={decay_lambda}, half-life={half_life_days:.0f} days")
        logger.info(f"   Decay range: {w_decay.min():.4f} to {w_decay.max():.4f}")
        logger.info(f"   Reference date: {ref_date.date()}")

    # Generate test events
    logger.info("\nGenerating test events...")
    events_test = generate_events(
        bars_df=bars_test,
        bar_size="5m",
        labeling_config=labeling_config,
        execution_spec=execution_spec,
    )

    events_test = apply_triplebarrier(
        bars_df=bars_test,
        events_df=events_test,
        bar_size="5m",
        labeling_config=labeling_config,
        execution_spec=execution_spec,
        instrument_spec=instrument_spec
    )
    logger.info(f"   Test events: {len(events_test):,}, y distribution: {events_test['y'].value_counts().to_dict()}")

    # --- Step 1a: Drop vertical barrier events for binary classification ---
    tb_cfg = labeling_config.get("primary_labeling", {}).get("triple_barrier", {})
    target_cfg = training_config.get('target', {})
    is_binary = target_cfg.get('mode') == 'binary'

    if tb_cfg.get("drop_vertical_barrier", False):
        n_before_train = len(events_train)
        n_before_test = len(events_test)
        events_train = events_train[events_train['y'] != 0].reset_index(drop=True)
        events_test = events_test[events_test['y'] != 0].reset_index(drop=True)
        logger.info(f"Dropped vertical events: train {n_before_train} -> {len(events_train)}, "
                     f"test {n_before_test} -> {len(events_test)}")
        logger.info(f"   Train label distribution after drop: {events_train['y'].value_counts().to_dict()}")
        logger.info(f"   Test label distribution after drop: {events_test['y'].value_counts().to_dict()}")

    # Build features
    logger.info("\nBuilding features...")
    features_train = build_features(bars_train, "5m", feature_config)
    features_test = build_features(bars_test, "5m", feature_config)
    logger.info(f"   Train features: {len(features_train):,} bars, {len(features_train.columns)} features")
    logger.info(f"   Test features:  {len(features_test):,} bars")

    # Merge events with features
    logger.info("\nMerging events with features...")

    # Get t0 timestamps
    if 't0' in events_train.columns:
        t0_train = events_train['t0'].tolist()
        t0_test = events_test['t0'].tolist()
    else:
        t0_train = events_train.index.tolist()
        t0_test = events_test.index.tolist()

    # Reindex features to event timestamps
    features_at_t0_train = features_train.reindex(t0_train)
    features_at_t0_test = features_test.reindex(t0_test)

    # Check for missing
    missing_train = features_at_t0_train.isna().all(axis=1)
    missing_test = features_at_t0_test.isna().all(axis=1)

    if missing_train.any():
        logger.warning(f"   {missing_train.sum()} train events have no features (will be dropped)")
    if missing_test.any():
        logger.warning(f"   {missing_test.sum()} test events have no features (will be dropped)")

    # Combine
    dataset_train = pd.concat([
        events_train[['side', 'y']].reset_index(drop=True),
        features_at_t0_train.reset_index(drop=True)
    ], axis=1)

    dataset_test = pd.concat([
        events_test[['side', 'y']].reset_index(drop=True),
        features_at_t0_test.reset_index(drop=True)
    ], axis=1)

    # Extract labels
    y_train = dataset_train['y'].astype(int)
    y_test = dataset_test['y'].astype(int)

    # Drop rows with missing features
    valid_train = ~dataset_train.isna().any(axis=1)
    valid_test = ~dataset_test.isna().any(axis=1)

    dataset_train = dataset_train[valid_train]
    y_train = y_train[valid_train]
    dataset_test = dataset_test[valid_test]
    y_test = y_test[valid_test]

    # Drop 'y' from features
    dataset_train = dataset_train.drop(columns=['y'])
    dataset_test = dataset_test.drop(columns=['y'])

    logger.info(f"   Final train set: {len(dataset_train):,} events")
    logger.info(f"   Final test set:  {len(dataset_test):,} events")

    # --- Step 1b: Remap labels for binary classification ---
    if is_binary:
        logger.info("\nBinary mode: remapping labels {-1, 1} -> {0, 1}")
        logger.info(f"   Before remap - train unique: {sorted(y_train.unique())}, test unique: {sorted(y_test.unique())}")
        y_train = (y_train == 1).astype(int)  # 1 = target hit, 0 = stop hit
        y_test = (y_test == 1).astype(int)
        logger.info(f"   After remap  - train: {y_train.value_counts().to_dict()}, test: {y_test.value_counts().to_dict()}")

    # --- Phase 5: Feature Selection (optional) ---
    # If feature_selection.json exists, restrict to selected features
    feature_selection_path = Path("ml_intraday_v3/diagnostics/feature_selection.json")
    selected_features = None
    if feature_selection_path.exists():
        import json as _json
        with open(feature_selection_path) as _f:
            fs_result = _json.load(_f)
        selected_features = fs_result.get('selected_features', None)
        if selected_features:
            logger.info(f"\nPhase 5: Using {len(selected_features)} selected features from {feature_selection_path}")
            logger.info(f"   Features: {selected_features}")

    # Feature list
    # INCLUDE 'side' feature so model can learn directional context
    feature_columns = [c for c in dataset_train.columns]

    if selected_features is not None:
        # Always keep 'side' if present
        keep = set(selected_features)
        if 'side' in feature_columns:
            keep.add('side')
        feature_columns = [c for c in feature_columns if c in keep]
        logger.info(f"   Restricted to {len(feature_columns)} features (from selection + side)")

    X_train = dataset_train[feature_columns]
    X_test = dataset_test[feature_columns]

    logger.info(f"   Features: {len(feature_columns)}")
    logger.info(f"   Includes 'side' feature: {'side' in feature_columns}")

    # --- Step 1c: Train model (config-driven objective) ---
    logger.info("\nTraining LightGBM model...")

    model_params = training_config.get('model', {}).get('params', {})
    # Remove 'objective' from params to avoid conflict with explicit kwarg
    model_params = {k: v for k, v in model_params.items() if k != 'objective'}
    logger.info(f"   Model parameters: {model_params}")

    if is_binary:
        logger.info("   Mode: BINARY classification (stop=0 vs target=1)")
        model = LGBMClassifier(
            objective='binary',
            random_state=42,
            verbose=-1,
            **model_params
        )
    else:
        logger.info("   Mode: MULTICLASS classification (stop/vertical/target)")
        model = LGBMClassifier(
            objective='multiclass',
            num_class=3,
            random_state=42,
            verbose=-1,
            **model_params
        )

    # Sample weights (if available)
    sample_weight_col = training_config.get('sample_weight', {}).get('column', 'w_final')
    if sample_weight_col in events_train.columns:
        w_train = events_train.loc[valid_train.values if hasattr(valid_train, 'values') else valid_train, sample_weight_col].values
        logger.info(f"   Using sample weights: {sample_weight_col}")
    else:
        w_train = None
        logger.info(f"   No sample weights (column '{sample_weight_col}' not found)")

    # --- Step 2: Isotonic Probability Calibration ---
    cal_cfg = training_config.get('calibration', {})
    calibration_enabled = cal_cfg.get('enabled', False)

    if calibration_enabled:
        cal_fraction = cal_cfg.get('calibration_fraction', 0.20)
        cal_method = cal_cfg.get('method', 'isotonic')
        cal_seed = cal_cfg.get('calibration_seed', 42)

        logger.info(f"\nCalibration enabled: method={cal_method}, holdout={cal_fraction*100:.0f}%")

        # Split: training portion vs calibration holdout
        if w_train is not None:
            X_train_model, X_cal, y_train_model, y_cal, w_train_model, w_cal = train_test_split(
                X_train, y_train, w_train,
                test_size=cal_fraction, random_state=cal_seed, stratify=y_train
            )
        else:
            X_train_model, X_cal, y_train_model, y_cal = train_test_split(
                X_train, y_train,
                test_size=cal_fraction, random_state=cal_seed, stratify=y_train
            )
            w_train_model = None
            w_cal = None

        logger.info(f"   Train split: {len(X_train_model):,} model, {len(X_cal):,} calibration")

        # Train on reduced set
        model.fit(X_train_model, y_train_model, sample_weight=w_train_model)
        logger.info("   Base model trained on reduced set")

        # Fit calibration manually on held-out calibration set
        cal_method = cal_cfg.get('method', 'isotonic')
        cal_proba = model.predict_proba(X_cal)[:, 1]

        if cal_method == 'sigmoid':
            # Platt scaling: fit logistic regression on raw probabilities
            logistic_cal = LogisticRegression(solver='lbfgs', max_iter=1000)
            logistic_cal.fit(cal_proba.reshape(-1, 1), y_cal)
            calibrator = SigmoidCalibratorWrapper(logistic_cal)
            logger.info(f"   Sigmoid calibration (Platt scaling) fitted on {len(X_cal):,} samples")
        else:  # isotonic (default)
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(cal_proba, y_cal)
            logger.info(f"   Isotonic calibration fitted on {len(X_cal):,} samples")

        # Wrap in a CalibratedModel that behaves like sklearn classifier
        calibrated_model = CalibratedBinaryModel(model, calibrator)
    else:
        # No calibration - train on full set
        model.fit(X_train, y_train, sample_weight=w_train)
        calibrated_model = model
        logger.info("   Model trained (no calibration)")

    logger.info("   Model trained successfully")

    # --- Step 1d: Evaluate ---
    logger.info("\nEvaluating model...")

    unique_labels = sorted(y_train.unique())
    logger.info(f"   Unique labels: {unique_labels}")

    if is_binary:
        # Binary AUC
        train_proba = calibrated_model.predict_proba(X_train)[:, 1]
        test_proba = calibrated_model.predict_proba(X_test)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)
        test_auc = roc_auc_score(y_test, test_proba)

        y_pred_train_class = calibrated_model.predict(X_train)
        y_pred_test_class = calibrated_model.predict(X_test)
        train_acc = accuracy_score(y_train, y_pred_train_class)
        test_acc = accuracy_score(y_test, y_pred_test_class)

        logger.info(f"\n   Train AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}")
        logger.info(f"   Test AUC:  {test_auc:.4f}, Accuracy: {test_acc:.4f}")

        logger.info("\n   Test Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred_test_class, target_names=['Stop', 'Target'])}")

        # Probability distribution analysis
        logger.info("\n   Probability Distribution (P(target) on test set):")
        logger.info(f"     Min: {test_proba.min():.4f}")
        logger.info(f"     25%: {np.percentile(test_proba, 25):.4f}")
        logger.info(f"     50%: {np.percentile(test_proba, 50):.4f}")
        logger.info(f"     75%: {np.percentile(test_proba, 75):.4f}")
        logger.info(f"     Max: {test_proba.max():.4f}")
        logger.info(f"     Signals with P(target) > 0.55: {(test_proba > 0.55).sum()} / {len(test_proba)} "
                     f"({(test_proba > 0.55).mean()*100:.1f}%)")

        # Calibration metrics
        if calibration_enabled:
            prob_true, prob_pred = calibration_curve(y_test, test_proba, n_bins=10)
            ece = np.mean(np.abs(prob_true - prob_pred))
            logger.info(f"\n   Calibration Metrics:")
            logger.info(f"     Expected Calibration Error (ECE): {ece:.4f}")
            logger.info(f"     Reliability diagram (bin_pred -> bin_true):")
            for pt, pp in zip(prob_true, prob_pred):
                logger.info(f"       {pp:.3f} -> {pt:.3f} (gap: {abs(pt-pp):.3f})")
    else:
        # Multiclass AUC
        if len(unique_labels) == 3:
            train_auc = roc_auc_score(y_train, calibrated_model.predict_proba(X_train), multi_class='ovr')
            test_auc = roc_auc_score(y_test, calibrated_model.predict_proba(X_test), multi_class='ovr')
        else:
            train_auc = roc_auc_score(y_train, calibrated_model.predict_proba(X_train)[:, 1])
            test_auc = roc_auc_score(y_test, calibrated_model.predict_proba(X_test)[:, 1])

        y_pred_train_class = calibrated_model.predict(X_train)
        y_pred_test_class = calibrated_model.predict(X_test)
        train_acc = accuracy_score(y_train, y_pred_train_class)
        test_acc = accuracy_score(y_test, y_pred_test_class)

        logger.info(f"\n   Train AUC: {train_auc:.4f}, Accuracy: {train_acc:.4f}")
        logger.info(f"   Test AUC:  {test_auc:.4f}, Accuracy: {test_acc:.4f}")

        logger.info("\n   Test Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred_test_class, target_names=['Stop', 'Vertical', 'Target'])}")

    # --- Step 1e: Create fresh preprocessor ---
    logger.info("\nCreating preprocessor from training data...")
    preprocessor = {
        'impute': 'median',
        'scaler': 'standard',
        'medians': X_train.median().values,
        'means': X_train.mean().values,
        'stds': X_train.std().values
    }
    logger.info(f"   Preprocessor: {len(preprocessor['medians'])} features")

    # Save model bundle
    logger.info("\nSaving model bundle...")

    bundle = {
        'primary_model': calibrated_model,
        'primary_feature_columns': feature_columns,
        'primary_preprocessor': preprocessor,
        'has_side_feature': 'side' in feature_columns,
        'has_dual_model': False,
        'is_binary': is_binary,
        'calibration_applied': calibration_enabled,
        'calibration_method': cal_cfg.get('method', 'isotonic') if calibration_enabled else None,
        'thresholds': {
            'primary_threshold': training_config.get('meta', {}).get('threshold_primary', 0.10)
        },
        'meta_model': None,
        'metadata': {
            'created': datetime.now().isoformat(),
            'training_method': 'Binary_Calibrated_V1' if is_binary else 'Multiclass_V3',
            'objective': 'binary' if is_binary else 'multiclass',
            'calibration': cal_cfg.get('method', 'none') if calibration_enabled else 'none',
            'drop_vertical_barrier': tb_cfg.get('drop_vertical_barrier', False),
            'train_period': f"{train_start.date()} to {train_end.date()}",
            'test_period': f"{test_start.date()} to {test_end.date()}",
            'train_events': len(dataset_train),
            'train_long_pct': (dataset_train['side'] == 1).mean() * 100,
            'train_short_pct': (dataset_train['side'] == -1).mean() * 100,
            'test_events': len(dataset_test),
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'feature_count': len(feature_columns),
        }
    }

    output_dir = Path("ml_intraday_v3/models/saved")
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_binary and calibration_enabled:
        output_path = output_dir / "model_bundle_binary_calibrated_v1.pkl"
    else:
        output_path = output_dir / "model_bundle_balanced_v3.pkl"

    joblib.dump(bundle, output_path)

    logger.info(f"   Saved to: {output_path}")
    logger.info(f"\n   Bundle metadata:")
    for key, value in bundle['metadata'].items():
        logger.info(f"     {key}: {value}")

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nModel: {output_path}")
    logger.info(f"Objective: {'binary' if is_binary else 'multiclass'}")
    logger.info(f"Calibration: {cal_cfg.get('method', 'none') if calibration_enabled else 'none'}")
    logger.info(f"Train events: {len(dataset_train):,} (balanced 50/50)")
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Has 'side' feature: {bundle['has_side_feature']}")
    logger.info(f"\nNext: Test on January 2026 data")
    logger.info("="*80)


if __name__ == "__main__":
    main()
