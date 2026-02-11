#!/usr/bin/env python3
"""
Threshold Sweep Test for Conservative Top10 Model

Tests different confidence thresholds to estimate signal rate and quality.
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ml_intraday_v3.labels.events import generate_events
from ml_intraday_v3.labels.triple_barrier import apply_triplebarrier
from ml_intraday_v3.core.instrument import InstrumentSpec
from ml_intraday_v3.features.build import build_features
from ml_intraday_v3.train_balanced_model import CalibratedBinaryModel, SigmoidCalibratorWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def test_thresholds(month_start: str, month_end: str, model_path: str, thresholds: list):
    """Test model at multiple confidence thresholds."""

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
    events_valid = events[valid].reset_index(drop=True)

    X = dataset.drop(columns=['y'])

    logger.info(f"Valid events: {len(X)}")

    # Load model
    model_bundle = joblib.load(model_path)
    model = model_bundle['primary_model']
    feature_cols = model_bundle.get('primary_feature_columns', None)

    if feature_cols:
        logger.info(f"Model features: {feature_cols}")
        available = [f for f in feature_cols if f in X.columns]
        X = X[available]

    # Predict
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)

    logger.info(f"Overall AUC: {auc:.4f}")
    logger.info(f"Prob range: {proba.min():.3f} - {proba.max():.3f}")

    # Test at multiple thresholds
    results = []
    for threshold in thresholds:
        # Filter to signals above threshold
        signals = proba > threshold
        n_signals = signals.sum()

        if n_signals == 0:
            results.append({
                'threshold': threshold,
                'signals': 0,
                'win_rate': None,
                'avg_return': None,
                'sharpe': None,
            })
            continue

        # Get actual outcomes for signals above threshold
        y_signals = y[signals]
        proba_signals = proba[signals]
        events_signals = events_valid[signals]

        win_rate = y_signals.mean()

        # Calculate returns (use ret_net from events)
        returns = events_signals['ret_net'].values
        avg_return = returns.mean()
        sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252 * 78)  # Annualized, assuming 78 trading days

        results.append({
            'threshold': threshold,
            'signals': int(n_signals),
            'win_rate': float(win_rate),
            'avg_return': float(avg_return),
            'sharpe': float(sharpe),
            'median_prob': float(np.median(proba_signals)),
        })

    return results, auc, len(y), proba


def main():
    model_path = "ml_intraday_v3/models/saved/model_bundle_conservative_top10_full.pkl"

    # Test thresholds from 0.40 to 0.55
    thresholds = [0.40, 0.42, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.52, 0.55]

    logger.info("="*80)
    logger.info("CONSERVATIVE TOP10 THRESHOLD SWEEP")
    logger.info("="*80)

    # Test on Dec 2025
    logger.info("\n--- December 2025 (held-out test set) ---")
    dec_results, dec_auc, dec_events, dec_proba = test_thresholds(
        '2025-12-01', '2025-12-31', model_path, thresholds
    )

    # Summary table
    logger.info("\n" + "="*90)
    logger.info(f"THRESHOLD SWEEP RESULTS (Dec 2025, {dec_events} events, AUC {dec_auc:.4f})")
    logger.info("="*90)
    logger.info(f"{'Threshold':>10} {'Signals':>10} {'Signal %':>10} {'Win Rate':>10} {'Avg Ret':>10} {'Sharpe':>10}")
    logger.info("-"*90)

    for r in dec_results:
        if r['signals'] == 0:
            logger.info(f"{r['threshold']:>10.2f} {r['signals']:>10} {0:>9.1f}% {'--':>10} {'--':>10} {'--':>10}")
        else:
            signal_pct = 100 * r['signals'] / dec_events
            logger.info(f"{r['threshold']:>10.2f} {r['signals']:>10} {signal_pct:>9.1f}% "
                        f"{r['win_rate']:>10.1%} {r['avg_return']:>10.2f} {r['sharpe']:>10.2f}")

    # Recommendation
    logger.info("\n" + "="*90)
    logger.info("RECOMMENDATION")
    logger.info("="*90)

    # Find threshold with at least 5% signal rate and best Sharpe
    valid_results = [r for r in dec_results if r['signals'] > 0 and r['signals'] / dec_events >= 0.05]

    if valid_results:
        best = max(valid_results, key=lambda r: r['sharpe'] if r['sharpe'] is not None else -999)
        signal_pct = 100 * best['signals'] / dec_events

        logger.info(f"Best threshold: {best['threshold']:.2f}")
        logger.info(f"  Signals: {best['signals']} ({signal_pct:.1f}% of events)")
        logger.info(f"  Win rate: {best['win_rate']:.1%}")
        logger.info(f"  Avg return: ${best['avg_return']:.2f} per trade")
        logger.info(f"  Sharpe: {best['sharpe']:.2f}")

        if best['sharpe'] > 1.0:
            logger.info("\n>>> DEPLOY with this threshold.")
        elif best['sharpe'] > 0.5:
            logger.info("\n>>> MARGINAL EDGE - Monitor closely if deploying.")
        else:
            logger.info("\n>>> WEAK EDGE - Consider barrier optimization or model improvement.")
    else:
        logger.info("No threshold produces enough signals (need >= 5% signal rate).")
        logger.info("Lowest threshold tested (0.40) produces:")
        if dec_results[0]['signals'] > 0:
            r = dec_results[0]
            signal_pct = 100 * r['signals'] / dec_events
            logger.info(f"  {r['signals']} signals ({signal_pct:.1f}%), win rate {r['win_rate']:.1%}")

    # Probability distribution
    logger.info("\n" + "="*90)
    logger.info("PROBABILITY DISTRIBUTION")
    logger.info("="*90)
    logger.info(f"Min:    {dec_proba.min():.4f}")
    logger.info(f"25%:    {np.percentile(dec_proba, 25):.4f}")
    logger.info(f"Median: {np.median(dec_proba):.4f}")
    logger.info(f"75%:    {np.percentile(dec_proba, 75):.4f}")
    logger.info(f"Max:    {dec_proba.max():.4f}")


if __name__ == '__main__':
    main()
