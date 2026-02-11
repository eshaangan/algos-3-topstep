#!/usr/bin/env python3
"""
Walk-Forward Validation - Historical OOS Testing
Tests if AUC 0.54 edge is stable across multiple time periods.

Strategy:
- Train on 6-month rolling windows
- Test on 1-month OOS periods
- Step forward by 1 month
- Test periods: Jan-Jun 2025 (all BEFORE current training window)
"""

import json
import logging
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, classification_report

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


class CalibratedBinaryModel:
    """Simple wrapper for calibrated model (same as train_balanced_model.py)."""
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

    @property
    def feature_importances_(self):
        return self.base_model.feature_importances_


def train_and_evaluate_window(
    bars: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    configs: dict,
    instrument_spec: InstrumentSpec,
) -> dict:
    """Train on one window, test on next period."""

    logger.info(f"\n{'='*80}")
    logger.info(f"Train: {train_start.date()} to {train_end.date()}")
    logger.info(f"Test:  {test_start.date()} to {test_end.date()}")
    logger.info(f"{'='*80}")

    try:
        # Split data
        bars_train = bars[(bars.index >= train_start) & (bars.index <= train_end)]
        bars_test = bars[(bars.index >= test_start) & (bars.index <= test_end)]

        logger.info(f"Train: {len(bars_train):,} bars, Test: {len(bars_test):,} bars")

        # Generate events + labels for train
        events_train = generate_events(
            bars_df=bars_train, bar_size="5m",
            labeling_config=configs['labeling'],
            execution_spec=configs['execution'],
        )
        events_train = apply_triplebarrier(
            bars_df=bars_train, events_df=events_train, bar_size="5m",
            labeling_config=configs['labeling'],
            execution_spec=configs['execution'],
            instrument_spec=instrument_spec,
        )
        events_train = balance_events(events_train, target_long_ratio=0.50, method='undersample')

        # Drop vertical barriers
        events_train = events_train[events_train['y'] != 0].reset_index(drop=True)

        # Generate events + labels for test
        events_test = generate_events(
            bars_df=bars_test, bar_size="5m",
            labeling_config=configs['labeling'],
            execution_spec=configs['execution'],
        )
        events_test = apply_triplebarrier(
            bars_df=bars_test, events_df=events_test, bar_size="5m",
            labeling_config=configs['labeling'],
            execution_spec=configs['execution'],
            instrument_spec=instrument_spec,
        )
        events_test = events_test[events_test['y'] != 0].reset_index(drop=True)

        if len(events_train) < 500 or len(events_test) < 50:
            logger.warning(f"Insufficient events: train={len(events_train)}, test={len(events_test)}")
            return {'status': 'skipped', 'reason': 'insufficient_events'}

        # Build features
        features_train = build_features(bars_train, "5m", configs['features'])
        features_test = build_features(bars_test, "5m", configs['features'])

        # Merge events with features
        t0_train = events_train['t0'].tolist()
        t0_test = events_test['t0'].tolist()
        feat_train = features_train.reindex(t0_train).reset_index(drop=True)
        feat_test = features_test.reindex(t0_test).reset_index(drop=True)

        dataset_train = pd.concat([
            events_train[['side', 'y']].reset_index(drop=True),
            feat_train,
        ], axis=1)
        dataset_test = pd.concat([
            events_test[['side', 'y']].reset_index(drop=True),
            feat_test,
        ], axis=1)

        # Binary labels
        y_train = (dataset_train['y'] == 1).astype(int)
        y_test = (dataset_test['y'] == 1).astype(int)

        # Drop NaN rows
        valid_train = ~dataset_train.isna().any(axis=1)
        valid_test = ~dataset_test.isna().any(axis=1)
        dataset_train = dataset_train[valid_train]
        y_train = y_train[valid_train]
        dataset_test = dataset_test[valid_test]
        y_test = y_test[valid_test]

        X_train = dataset_train.drop(columns=['y'])
        X_test = dataset_test.drop(columns=['y'])

        if len(X_train) < 500 or len(X_test) < 50:
            return {'status': 'skipped', 'reason': 'insufficient_valid_events'}

        # Apply sample decay (same as training script)
        decay_cfg = configs['training'].get('sample_decay', {})
        if decay_cfg.get('enabled', False):
            decay_lambda = decay_cfg.get('lambda', 0.005)
            t0_col = events_train['t0']
            ref_date = t0_col.max()
            age_days = (ref_date - t0_col).dt.total_seconds() / 86400.0
            age_days = age_days[valid_train].values
            w_decay = np.exp(-decay_lambda * age_days)
        else:
            w_decay = np.ones(len(X_train))

        # Train model with calibration
        model_params = configs['training']['model']['params'].copy()
        model_params = {k: v for k, v in model_params.items() if k != 'objective'}

        base_model = LGBMClassifier(
            objective='binary',
            random_state=42,
            verbose=-1,
            **model_params
        )

        # Split for calibration
        from sklearn.model_selection import train_test_split
        X_model, X_cal, y_model, y_cal, w_model, w_cal = train_test_split(
            X_train, y_train, w_decay, test_size=0.2, random_state=42, stratify=y_train
        )

        # Train base model
        base_model.fit(X_model, y_model, sample_weight=w_model)

        # Calibrate
        from sklearn.isotonic import IsotonicRegression
        cal_probs = base_model.predict_proba(X_cal)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(cal_probs, y_cal)

        model = CalibratedBinaryModel(base_model, calibrator)

        # Evaluate
        test_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_proba)

        train_proba = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)

        # Probability distribution stats
        prob_stats = {
            'min': float(test_proba.min()),
            'p25': float(np.percentile(test_proba, 25)),
            'p50': float(np.percentile(test_proba, 50)),
            'p75': float(np.percentile(test_proba, 75)),
            'max': float(test_proba.max()),
            'range': float(test_proba.max() - test_proba.min()),
            'signals_gt_055': int((test_proba > 0.55).sum()),
        }

        # Target rate
        target_rate_train = y_train.mean()
        target_rate_test = y_test.mean()

        logger.info(f"  Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")
        logger.info(f"  Test events: {len(X_test)}, Target rate: {target_rate_test:.1%}")
        logger.info(f"  Prob range: {prob_stats['min']:.3f} - {prob_stats['max']:.3f}")

        return {
            'status': 'success',
            'train_start': train_start.isoformat(),
            'train_end': train_end.isoformat(),
            'test_start': test_start.isoformat(),
            'test_end': test_end.isoformat(),
            'train_auc': round(train_auc, 4),
            'test_auc': round(test_auc, 4),
            'train_events': len(X_train),
            'test_events': len(X_test),
            'target_rate_train': round(target_rate_train, 4),
            'target_rate_test': round(target_rate_test, 4),
            'prob_stats': prob_stats,
            'feature_count': len(X_train.columns),
        }

    except Exception as e:
        logger.error(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


def main():
    logger.info("="*80)
    logger.info("WALK-FORWARD VALIDATION - HISTORICAL OOS")
    logger.info("="*80)

    # Load data
    data_path = Path("data/processed/mes_bars_databento_rth.h5")
    logger.info(f"\nLoading data from: {data_path}")
    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()
    logger.info(f"Data: {len(bars):,} bars ({bars.index[0].date()} to {bars.index[-1].date()})")

    # Load configs
    with open("ml_intraday_v3/configs/labeling.yaml") as f:
        labeling_config = yaml.safe_load(f)
    with open("ml_intraday_v3/configs/execution_spec.yaml") as f:
        execution_spec = yaml.safe_load(f)
    with open("ml_intraday_v3/configs/features.yaml") as f:
        feature_config = yaml.safe_load(f)
    with open("ml_intraday_v3/configs/training.yaml") as f:
        training_config = yaml.safe_load(f)

    configs = {
        'labeling': labeling_config,
        'execution': execution_spec,
        'features': feature_config,
        'training': training_config,
    }

    instrument_spec = InstrumentSpec(
        symbol="MES",
        tick_size_points=0.25,
        contract_multiplier_usd_per_point=5.0,
    )

    # Define walk-forward windows
    # Test periods: Jan-Jun 2025 (all BEFORE current training window of Jun-Nov 2025)
    windows = [
        # Train Jul-Dec 2024 → Test Jan 2025
        ('2024-07-01', '2024-12-31', '2025-01-01', '2025-01-31'),
        # Train Aug 2024-Jan 2025 → Test Feb 2025
        ('2024-08-01', '2025-01-31', '2025-02-01', '2025-02-28'),
        # Train Sep 2024-Feb 2025 → Test Mar 2025
        ('2024-09-01', '2025-02-28', '2025-03-01', '2025-03-31'),
        # Train Oct 2024-Mar 2025 → Test Apr 2025
        ('2024-10-01', '2025-03-31', '2025-04-01', '2025-04-30'),
        # Train Nov 2024-Apr 2025 → Test May 2025
        ('2024-11-01', '2025-04-30', '2025-05-01', '2025-05-31'),
        # Train Dec 2024-May 2025 → Test Jun 2025
        ('2024-12-01', '2025-05-31', '2025-06-01', '2025-06-30'),
    ]

    logger.info(f"\n{len(windows)} walk-forward windows defined")
    logger.info("Test periods: Jan-Jun 2025 (all OOS, before current training window)\n")

    results = []
    for i, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
        logger.info(f"\n{'='*80}")
        logger.info(f"WINDOW {i+1}/{len(windows)}")
        logger.info(f"{'='*80}")

        result = train_and_evaluate_window(
            bars=bars,
            train_start=pd.Timestamp(tr_start, tz='UTC'),
            train_end=pd.Timestamp(f"{tr_end} 23:59:59", tz='UTC'),
            test_start=pd.Timestamp(te_start, tz='UTC'),
            test_end=pd.Timestamp(f"{te_end} 23:59:59", tz='UTC'),
            configs=configs,
            instrument_spec=instrument_spec,
        )
        results.append(result)

    # Aggregate statistics
    successful = [r for r in results if r['status'] == 'success']

    if successful:
        test_aucs = [r['test_auc'] for r in successful]
        train_aucs = [r['train_auc'] for r in successful]

        logger.info("\n" + "="*80)
        logger.info("WALK-FORWARD SUMMARY")
        logger.info("="*80)
        logger.info(f"Successful windows: {len(successful)}/{len(windows)}")
        logger.info(f"\nTest AUC Statistics:")
        logger.info(f"  Mean:   {np.mean(test_aucs):.4f}")
        logger.info(f"  Median: {np.median(test_aucs):.4f}")
        logger.info(f"  Std:    {np.std(test_aucs):.4f}")
        logger.info(f"  Min:    {np.min(test_aucs):.4f}")
        logger.info(f"  Max:    {np.max(test_aucs):.4f}")
        logger.info(f"\nTrain AUC Statistics:")
        logger.info(f"  Mean:   {np.mean(train_aucs):.4f}")
        logger.info(f"  Median: {np.median(train_aucs):.4f}")

        logger.info(f"\nPer-Window Results:")
        for i, r in enumerate(successful):
            test_period = pd.Timestamp(r['test_start']).strftime('%b %Y')
            logger.info(f"  {test_period}: Test AUC = {r['test_auc']:.4f}, Train AUC = {r['train_auc']:.4f}")

        # Decision metrics
        median_auc = np.median(test_aucs)
        logger.info("\n" + "="*80)
        logger.info("DECISION METRICS")
        logger.info("="*80)
        logger.info(f"Median Test AUC: {median_auc:.4f}")

        if median_auc > 0.55:
            logger.info("✅ STRONG EDGE - Median AUC > 0.55")
            logger.info("   Recommendation: Proceed to barrier optimization")
        elif median_auc > 0.52:
            logger.info("⚠️  WEAK EDGE - Median AUC 0.52-0.55")
            logger.info("   Recommendation: Proceed cautiously, optimize barriers")
        else:
            logger.info("❌ NO EDGE - Median AUC < 0.52")
            logger.info("   Recommendation: Current approach not viable, need redesign")

        # Compare to current model
        current_auc = 0.5384
        logger.info(f"\nCurrent model (Dec 2025 test): AUC = {current_auc:.4f}")
        logger.info(f"Walk-forward median:           AUC = {median_auc:.4f}")
        delta = median_auc - current_auc
        logger.info(f"Delta: {delta:+.4f}")

        if abs(delta) < 0.01:
            logger.info("✅ Consistent performance - Dec 2025 result is representative")
        elif delta > 0.02:
            logger.info("⚠️  Dec 2025 underperformed historical average")
        else:
            logger.info("⚠️  Dec 2025 may have been lucky - historical performance weaker")

    # Save results
    output_dir = Path("ml_intraday_v3/diagnostics")
    output_path = output_dir / f"walkforward_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'windows': results,
            'summary': {
                'total_windows': len(windows),
                'successful': len(successful),
                'median_test_auc': float(np.median(test_aucs)) if successful else None,
                'mean_test_auc': float(np.mean(test_aucs)) if successful else None,
                'std_test_auc': float(np.std(test_aucs)) if successful else None,
            }
        }, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
