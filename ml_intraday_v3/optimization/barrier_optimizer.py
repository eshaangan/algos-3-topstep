#!/usr/bin/env python3
"""
Triple Barrier Parameter Optimizer (Phase 4)

Grid search over PT/SL multipliers and horizon bars, optimizing for
backtest Sharpe ratio rather than classification AUC.

Per Fu et al 2024: optimizing classification accuracy gives different results
than optimizing trading profit. A model with 55% accuracy but good R:R beats
60% accuracy with poor R:R.

Usage:
    python -m ml_intraday_v3.optimization.barrier_optimizer
"""

import itertools
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Add project root
project_root = Path(__file__).resolve().parents[2]
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


def _compute_sharpe(pnl_series: pd.Series, annualization: float = 252.0) -> float:
    """Compute annualized Sharpe ratio from a PnL series."""
    if len(pnl_series) < 2 or pnl_series.std() == 0:
        return 0.0
    return float(pnl_series.mean() / pnl_series.std() * np.sqrt(annualization))


def _simulate_backtest_pnl(
    events_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float = 0.55,
    pt_mult: float = 2.5,
    sl_mult: float = 2.0,
) -> pd.Series:
    """Simulate simple backtest PnL from events + probabilities.

    For each event where P(target) > threshold:
    - If y==1 (target hit): PnL = +pt_mult (in ATR units)
    - If y==-1 (stop hit): PnL = -sl_mult (in ATR units)

    Returns: Series of per-trade PnL values.
    """
    signals = y_prob > threshold
    if signals.sum() == 0:
        return pd.Series(dtype=float)

    trade_pnl = []
    for i in range(len(events_df)):
        if not signals[i]:
            continue
        y = events_df.iloc[i]['y']
        if y == 1:
            trade_pnl.append(pt_mult)
        elif y == -1:
            trade_pnl.append(-sl_mult)
        # y==0 (vertical) already dropped

    return pd.Series(trade_pnl, dtype=float)


def run_barrier_optimization(
    bars: pd.DataFrame,
    base_labeling_config: dict,
    execution_spec: dict,
    feature_config: dict,
    training_config: dict,
    instrument_spec: InstrumentSpec,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    pt_grid: list = None,
    sl_grid: list = None,
    horizon_grid: list = None,
    confidence_threshold: float = 0.55,
) -> dict:
    """Run grid search over triple barrier parameters.

    Args:
        bars: Full bar DataFrame
        base_labeling_config: Base labeling config to modify
        execution_spec: Execution specification
        feature_config: Feature configuration
        training_config: Training configuration
        instrument_spec: Instrument specification
        train_start/end: Training period
        test_start/end: Test period
        pt_grid: Profit target multipliers to search
        sl_grid: Stop loss multipliers to search
        horizon_grid: Vertical barrier horizons to search
        confidence_threshold: Minimum P(target) to take a trade

    Returns:
        Dict with results for each parameter combination
    """
    if pt_grid is None:
        pt_grid = [1.5, 2.0, 2.5, 3.0, 3.5]
    if sl_grid is None:
        sl_grid = [1.0, 1.5, 2.0, 2.5]
    if horizon_grid is None:
        horizon_grid = [6, 12, 18, 24]

    bars_train = bars[(bars.index >= train_start) & (bars.index <= train_end)]
    bars_test = bars[(bars.index >= test_start) & (bars.index <= test_end)]

    logger.info(f"Train: {len(bars_train):,} bars, Test: {len(bars_test):,} bars")

    # Pre-compute features (independent of barrier params)
    logger.info("Pre-computing features...")
    features_train = build_features(bars_train, "5m", feature_config)
    features_test = build_features(bars_test, "5m", feature_config)

    # Generate all parameter combinations where PT > SL
    combos = [
        (pt, sl, hz)
        for pt, sl, hz in itertools.product(pt_grid, sl_grid, horizon_grid)
        if pt > sl
    ]
    logger.info(f"Testing {len(combos)} parameter combinations (PT > SL constraint)")

    results = []
    model_params = training_config.get('model', {}).get('params', {})
    model_params = {k: v for k, v in model_params.items() if k != 'objective'}

    for idx, (pt_mult, sl_mult, horizon) in enumerate(combos):
        logger.info(f"\n--- Combo {idx+1}/{len(combos)}: PT={pt_mult}, SL={sl_mult}, Hz={horizon} ---")

        # Modify labeling config for this combo
        lab_cfg = deepcopy(base_labeling_config)
        tb = lab_cfg['primary_labeling']['triple_barrier']
        tb['pt_multipliers'] = [pt_mult]
        tb['sl_multipliers'] = [sl_mult]
        tb['horizon_bars']['5m'] = [horizon]

        try:
            # Generate events + labels for train
            events_train = generate_events(
                bars_df=bars_train, bar_size="5m",
                labeling_config=lab_cfg, execution_spec=execution_spec,
            )
            events_train = apply_triplebarrier(
                bars_df=bars_train, events_df=events_train, bar_size="5m",
                labeling_config=lab_cfg, execution_spec=execution_spec,
                instrument_spec=instrument_spec,
            )
            events_train = balance_events(events_train, target_long_ratio=0.50, method='undersample')

            # Drop vertical barrier events
            events_train = events_train[events_train['y'] != 0].reset_index(drop=True)

            # Generate events + labels for test
            events_test = generate_events(
                bars_df=bars_test, bar_size="5m",
                labeling_config=lab_cfg, execution_spec=execution_spec,
            )
            events_test = apply_triplebarrier(
                bars_df=bars_test, events_df=events_test, bar_size="5m",
                labeling_config=lab_cfg, execution_spec=execution_spec,
                instrument_spec=instrument_spec,
            )
            events_test = events_test[events_test['y'] != 0].reset_index(drop=True)

            if len(events_train) < 200 or len(events_test) < 50:
                logger.warning(f"   Too few events: train={len(events_train)}, test={len(events_test)}")
                results.append({
                    'pt_mult': pt_mult, 'sl_mult': sl_mult, 'horizon': horizon,
                    'status': 'skipped', 'reason': 'insufficient_events',
                })
                continue

            # Merge features with events
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

            # Labels
            y_train = (dataset_train['y'] == 1).astype(int)
            y_test_labels = (dataset_test['y'] == 1).astype(int)

            # Drop NaN rows
            valid_train = ~dataset_train.isna().any(axis=1)
            valid_test = ~dataset_test.isna().any(axis=1)
            dataset_train = dataset_train[valid_train]
            y_train = y_train[valid_train]
            dataset_test = dataset_test[valid_test]
            y_test_labels = y_test_labels[valid_test]

            X_train = dataset_train.drop(columns=['y'])
            X_test = dataset_test.drop(columns=['y'])

            if len(X_train) < 200 or len(X_test) < 50:
                results.append({
                    'pt_mult': pt_mult, 'sl_mult': sl_mult, 'horizon': horizon,
                    'status': 'skipped', 'reason': 'insufficient_valid_events',
                })
                continue

            # Train model
            model = LGBMClassifier(
                objective='binary', random_state=42, verbose=-1,
                **model_params
            )
            model.fit(X_train, y_train)

            # Predict
            test_proba = model.predict_proba(X_test)[:, 1]
            train_proba = model.predict_proba(X_train)[:, 1]

            # Metrics
            test_auc = roc_auc_score(y_test_labels, test_proba)
            train_auc = roc_auc_score(y_train, train_proba)

            # Simulate backtest PnL
            # Use the original events_test (with y values) filtered by validity
            test_events_valid = events_test[events_test['y'] != 0].reset_index(drop=True)
            test_events_valid = test_events_valid[valid_test.values]

            pnl = _simulate_backtest_pnl(
                test_events_valid, test_proba,
                threshold=confidence_threshold,
                pt_mult=pt_mult, sl_mult=sl_mult,
            )

            sharpe = _compute_sharpe(pnl) if len(pnl) > 1 else 0.0
            total_pnl = pnl.sum() if len(pnl) > 0 else 0.0
            n_trades = len(pnl)
            win_rate = (pnl > 0).mean() if len(pnl) > 0 else 0.0

            target_rate_train = y_train.mean()
            target_rate_test = y_test_labels.mean()

            result = {
                'pt_mult': pt_mult,
                'sl_mult': sl_mult,
                'horizon': horizon,
                'status': 'ok',
                'train_auc': round(train_auc, 4),
                'test_auc': round(test_auc, 4),
                'sharpe': round(sharpe, 3),
                'total_pnl_atr': round(total_pnl, 2),
                'n_trades': n_trades,
                'win_rate': round(win_rate, 4),
                'train_events': len(X_train),
                'test_events': len(X_test),
                'target_rate_train': round(target_rate_train, 4),
                'target_rate_test': round(target_rate_test, 4),
                'rr_ratio': round(pt_mult / sl_mult, 2),
            }
            results.append(result)

            logger.info(f"   AUC: train={train_auc:.3f}, test={test_auc:.3f}")
            logger.info(f"   Sharpe: {sharpe:.3f}, PnL: {total_pnl:.1f} ATR, Trades: {n_trades}, Win: {win_rate:.1%}")

        except Exception as e:
            logger.error(f"   Error: {e}")
            results.append({
                'pt_mult': pt_mult, 'sl_mult': sl_mult, 'horizon': horizon,
                'status': 'error', 'error': str(e),
            })

    # Sort by Sharpe ratio
    ok_results = [r for r in results if r.get('status') == 'ok']
    ok_results.sort(key=lambda r: r['sharpe'], reverse=True)

    logger.info("\n" + "="*80)
    logger.info("TOP 5 PARAMETER COMBINATIONS (by Sharpe)")
    logger.info("="*80)
    for i, r in enumerate(ok_results[:5]):
        logger.info(
            f"  #{i+1}: PT={r['pt_mult']}, SL={r['sl_mult']}, Hz={r['horizon']} | "
            f"Sharpe={r['sharpe']:.3f}, AUC={r['test_auc']:.3f}, "
            f"Win={r['win_rate']:.1%}, Trades={r['n_trades']}, R:R={r['rr_ratio']}"
        )

    return {
        'all_results': results,
        'top_results': ok_results[:5],
        'best': ok_results[0] if ok_results else None,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'pt_grid': pt_grid,
            'sl_grid': sl_grid,
            'horizon_grid': horizon_grid,
            'confidence_threshold': confidence_threshold,
        },
    }


def main():
    logger.info("="*80)
    logger.info("TRIPLE BARRIER PARAMETER OPTIMIZATION")
    logger.info("="*80)

    # Load data
    data_path = Path("data/processed/mes_bars_databento_rth.h5")
    logger.info(f"\nLoading data from: {data_path}")
    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()

    # Load configs
    with open("ml_intraday_v3/configs/labeling.yaml") as f:
        labeling_config = yaml.safe_load(f)
    with open("ml_intraday_v3/configs/execution_spec.yaml") as f:
        execution_spec = yaml.safe_load(f)
    with open("ml_intraday_v3/configs/features.yaml") as f:
        feature_config = yaml.safe_load(f)
    with open("ml_intraday_v3/configs/training.yaml") as f:
        training_config = yaml.safe_load(f)

    instrument_spec = InstrumentSpec(
        symbol="MES",
        tick_size_points=0.25,
        contract_multiplier_usd_per_point=5.0,
    )

    # Use 6-month train, 1-month test
    train_start = pd.Timestamp('2025-06-01', tz='UTC')
    train_end = pd.Timestamp('2025-11-30 23:59:59', tz='UTC')
    test_start = pd.Timestamp('2025-12-01', tz='UTC')
    test_end = pd.Timestamp('2025-12-31 23:59:59', tz='UTC')

    results = run_barrier_optimization(
        bars=bars,
        base_labeling_config=labeling_config,
        execution_spec=execution_spec,
        feature_config=feature_config,
        training_config=training_config,
        instrument_spec=instrument_spec,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        pt_grid=[1.5, 2.0, 2.5, 3.0, 3.5],
        sl_grid=[1.0, 1.5, 2.0, 2.5],
        horizon_grid=[6, 12, 18, 24],
    )

    # Save results
    output_dir = Path("ml_intraday_v3/optimization")
    output_path = output_dir / f"barrier_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {output_path}")

    # Recommend labeling.yaml update
    if results['best']:
        best = results['best']
        logger.info("\n" + "="*80)
        logger.info("RECOMMENDED labeling.yaml UPDATE:")
        logger.info("="*80)
        logger.info(f"  pt_multipliers: [{best['pt_mult']}]")
        logger.info(f"  sl_multipliers: [{best['sl_mult']}]")
        logger.info(f"  horizon_bars: 5m: [{best['horizon']}]")
        logger.info(f"  (Sharpe={best['sharpe']:.3f}, AUC={best['test_auc']:.3f}, Win={best['win_rate']:.1%})")


if __name__ == "__main__":
    main()
