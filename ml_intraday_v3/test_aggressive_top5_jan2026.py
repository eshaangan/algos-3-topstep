#!/usr/bin/env python3
"""
Quick Test: Aggressive Top5 Model on Jan 2026
Validates AUC and signal quality on out-of-sample data.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.metrics import roc_auc_score

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ml_intraday_v3.labels.events import generate_events
from ml_intraday_v3.labels.triple_barrier import apply_triplebarrier
from ml_intraday_v3.core.instrument import InstrumentSpec
from ml_intraday_v3.features.build import build_features
from ml_intraday_v3.train_balanced_model import CalibratedBinaryModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def test_model_on_month(month_start: str, month_end: str, model_path: str):
    """Test model on a single month."""

    logger.info(f"\nTesting on: {month_start} to {month_end}")
    logger.info(f"Model: {model_path}")

    # Load data
    data_path = Path("data/processed/mes_bars_databento_rth.h5")
    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()

    # Slice to test period
    test_start = pd.Timestamp(month_start, tz='UTC')
    test_end = pd.Timestamp(f"{month_end} 23:59:59", tz='UTC')
    bars_test = bars[(bars.index >= test_start) & (bars.index <= test_end)]

    logger.info(f"Test bars: {len(bars_test):,}")

    # Load configs
    configs = {}
    config_dir = Path("ml_intraday_v3/configs")
    for name, fname in [('labeling', 'labeling.yaml'), ('execution', 'execution_spec.yaml'),
                         ('features', 'features.yaml')]:
        with open(config_dir / fname) as f:
            configs[name] = yaml.safe_load(f)

    instrument_spec = InstrumentSpec(
        symbol="MES",
        tick_size_points=0.25,
        contract_multiplier_usd_per_point=5.0,
    )

    # Generate events + labels
    events = generate_events(
        bars_df=bars_test, bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution'],
    )
    events = apply_triplebarrier(
        bars_df=bars_test, events_df=events, bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution'],
        instrument_spec=instrument_spec,
    )

    # Drop vertical barriers
    events = events[events['y'] != 0].reset_index(drop=True)

    logger.info(f"Events: {len(events)}, LONG: {(events['side']==1).sum()}, SHORT: {(events['side']==-1).sum()}")
    logger.info(f"Labels: target={( events['y']==1).sum()}, stop={( events['y']==-1).sum()}")

    # Build features
    features = build_features(bars_test, "5m", configs['features'])

    # Drop meta columns
    meta_cols = ['is_synthetic', 'usable_for_training']
    features = features.drop(columns=[c for c in meta_cols if c in features.columns])

    # Merge
    feat_aligned = features.reindex(events['t0'].tolist()).reset_index(drop=True)
    dataset = pd.concat([
        events[['side', 'y']].reset_index(drop=True),
        feat_aligned,
    ], axis=1)

    # Binary labels
    y = (dataset['y'] == 1).astype(int)

    # Drop NaN
    valid = ~dataset.isna().any(axis=1)
    dataset = dataset[valid]
    y = y[valid]

    X = dataset.drop(columns=['y'])

    logger.info(f"Valid events: {len(X)} (after NaN drop)")

    # Load model
    model_bundle = joblib.load(model_path)
    model = model_bundle['primary_model']
    feature_cols = model_bundle.get('primary_feature_columns', None)

    if feature_cols:
        logger.info(f"Model expects {len(feature_cols)} features: {feature_cols}")
        # Filter to model's feature set
        available = [f for f in feature_cols if f in X.columns]
        if len(available) < len(feature_cols):
            missing = set(feature_cols) - set(available)
            logger.warning(f"Missing features: {missing}")
        X = X[available]

    # Predict
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)

    # Stats
    prob_stats = {
        'min': float(proba.min()),
        'p25': float(np.percentile(proba, 25)),
        'p50': float(np.percentile(proba, 50)),
        'p75': float(np.percentile(proba, 75)),
        'max': float(proba.max()),
        'range': float(proba.max() - proba.min()),
        'signals_gt_055': int((proba > 0.55).sum()),
        'signals_gt_053': int((proba > 0.53).sum()),
        'signals_gt_052': int((proba > 0.52).sum()),
    }

    logger.info(f"\nRESULTS:")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  Prob range: {prob_stats['min']:.3f} - {prob_stats['max']:.3f}")
    logger.info(f"  Median prob: {prob_stats['p50']:.3f}")
    logger.info(f"  Signals > 0.55: {prob_stats['signals_gt_055']} / {len(y)} ({100*prob_stats['signals_gt_055']/len(y):.1f}%)")
    logger.info(f"  Signals > 0.53: {prob_stats['signals_gt_053']} / {len(y)} ({100*prob_stats['signals_gt_053']/len(y):.1f}%)")
    logger.info(f"  Signals > 0.52: {prob_stats['signals_gt_052']} / {len(y)} ({100*prob_stats['signals_gt_052']/len(y):.1f}%)")
    logger.info(f"  Target rate: {y.mean():.1%}")

    return {
        'auc': auc,
        'events': len(y),
        'prob_stats': prob_stats,
        'target_rate': float(y.mean()),
    }


def main():
    model_path = "ml_intraday_v3/models/saved/model_bundle_aggressive_top5.pkl"

    logger.info("="*80)
    logger.info("AGGRESSIVE TOP5 MODEL VALIDATION")
    logger.info("="*80)

    # Test on Dec 2025 (should match training output: AUC 0.6202)
    logger.info("\n--- December 2025 (held-out test set) ---")
    dec_result = test_model_on_month('2025-12-01', '2025-12-31', model_path)

    # Test on Jan 2026 (truly out-of-sample)
    logger.info("\n--- January 2026 (truly OOS) ---")
    jan_result = test_model_on_month('2026-01-01', '2026-01-31', model_path)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Dec 2025: AUC {dec_result['auc']:.4f}, Signals>0.55: {dec_result['prob_stats']['signals_gt_055']}")
    logger.info(f"Jan 2026: AUC {jan_result['auc']:.4f}, Signals>0.55: {jan_result['prob_stats']['signals_gt_055']}")

    if jan_result['auc'] > 0.53 and jan_result['prob_stats']['signals_gt_055'] > 0:
        logger.info("\n✅ STRONG PERFORMANCE - Model is ready for deployment")
    elif jan_result['auc'] > 0.52:
        logger.info("\n⚠️  MARGINAL EDGE - Consider lowering confidence threshold to 0.52-0.53")
    else:
        logger.info("\n❌ NO EDGE ON JAN 2026 - Model does not generalize")


if __name__ == '__main__':
    main()
