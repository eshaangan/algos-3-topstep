#!/usr/bin/env python3
"""
Validate 10 Finalist Models on Jan-Feb 2026 OOS Data

Tests the 10 finalist models from batch1 and batch3 experiments on the
41-day out-of-sample period (Jan 1 - Feb 10, 2026).

Key Differences from Previous Tests:
1. Trend scanning labels (adaptive horizons) vs triple barrier
2. Fractional differentiation features (stationarity while preserving memory)
3. Sample uniqueness weighting (temporal overlap handling)
4. CPCV (combinatorial purged cross-validation)

Expected Outcome:
Models should produce DIFFERENT results (not identical like before), proving
that the new labeling/feature methods actually matter.
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
from sklearn.preprocessing import StandardScaler
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
from ml_intraday_v3.labels.events import generate_events
from ml_intraday_v3.labels.triple_barrier import apply_triplebarrier
from ml_intraday_v3.features.fractional_diff import apply_fractional_diff
from ml_intraday_v3.sampling.uniqueness import compute_uniqueness_decay_weights


def load_training_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical training data."""
    logger.info("\nLoading historical training data...")

    data_path = project_root / "data" / "processed" / "mes_bars_databento_rth.h5"

    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        sys.exit(1)

    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()

    if bars.index.tz is None:
        bars = bars.tz_localize('UTC')

    # Use Oct 2024 - Nov 2025 as training window
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
    instrument_spec: InstrumentSpec
) -> Optional[Tuple[object, pd.DataFrame, List[str]]]:
    """Train a model from finalist config."""

    exp_id = config['exp_id']
    logger.info(f"\n{'='*80}")
    logger.info(f"Training: {exp_id}")
    logger.info(f"{'='*80}")

    try:
        # Extract config parameters
        labeling_method = config.get('labeling_method', 'triple_barrier')
        labeling_params = config.get('labeling_params', {})
        sample_weight_method = config.get('sample_weight', 'uniform')
        feature_set_name = config.get('feature_set_name', 'baseline')
        model_params = config.get('model_params', {})

        logger.info(f"  Labeling: {labeling_method}")
        logger.info(f"  Features: {feature_set_name}")
        logger.info(f"  Weighting: {sample_weight_method}")

        # Generate events with labeling method
        execution_config = {
            'point_value': 5.0,
            'commission': 0.62,
            'slippage': 0.25,
        }

        # Build labeling config based on method
        if labeling_method == 'triple_barrier':
            labeling_config = {
                'drop_vertical_barrier': True,
                'primary_labeling': {
                    'event_policy': 'cusum',
                    'cusum': {
                        'threshold_atr_mult': 3.0,
                    },
                    'triple_barrier': {
                        'pt_multipliers': [labeling_params.get('pt_mult', 2.0)],
                        'sl_multipliers': [labeling_params.get('sl_mult', 1.5)],
                        'horizon_bars': {
                            '5m': [labeling_params.get('time_mult', 20)],
                        },
                    },
                },
            }
        elif labeling_method == 'trend_scanning':
            labeling_config = {
                'drop_vertical_barrier': True,
                'primary_labeling': {
                    'event_policy': 'trend_scanning',
                    'trend_scanning': {
                        'max_lookahead': labeling_params.get('max_lookahead', 25),
                        'min_t_value': labeling_params.get('min_t_value', 2.0),
                        'cusum_threshold_atr_mult': 3.0,
                    },
                    'triple_barrier': {
                        'pt_multipliers': [2.0],
                        'sl_multipliers': [1.5],
                        'horizon_bars': {
                            '5m': list(range(5, labeling_params.get('max_lookahead', 25) + 1, 5)),
                        },
                    },
                },
            }
        else:
            raise ValueError(f"Unknown labeling method: {labeling_method}")

        events = generate_events(
            bars_df=bars_train,
            bar_size="5m",
            labeling_config=labeling_config,
            execution_spec=execution_config,
        )

        logger.info(f"  Initial events: {len(events):,}")

        # Apply barriers to get labels
        events = apply_triplebarrier(
            bars_df=bars_train,
            events_df=events,
            bar_size="5m",
            labeling_config=labeling_config,
            execution_spec=execution_config,
            instrument_spec=instrument_spec,
        )

        # Drop vertical barriers if specified
        if labeling_config.get('drop_vertical_barrier', False):
            events = events[events['y'] != 0].reset_index(drop=True)

        logger.info(f"  Labeled events: {len(events):,}")

        if len(events) < 100:
            logger.warning(f"  Too few events ({len(events)}), skipping")
            return None

        # Build features
        features_config = {
            'structure': {
                'ema_fast_period': 12,
                'ema_slow_period': 26,
            },
            'time': {
                'include_time_features': True,
            },
            'enable_advanced_features': False,
        }

        X_train = build_features(
            bars_df=bars_train,
            events_df=events,
            bar_size="5m",
            features_config=features_config,
        )

        # Add fractional differentiation if specified
        if feature_set_name == 'baseline_fracdiff':
            logger.info("  Adding fractional differentiation features...")
            X_train = apply_fractional_diff(
                X_train,
                columns=['close', 'ema_fast', 'ema_slow'],
                d=0.4,  # Standard differentiation order
            )

        # Handle NaNs
        initial_count = len(X_train)
        X_train = X_train.dropna()
        events = events.loc[X_train.index].copy()

        if len(X_train) < initial_count * 0.5:
            logger.warning(f"  Lost {initial_count - len(X_train)} rows to NaNs, skipping")
            return None

        logger.info(f"  Features: {X_train.shape[1]} columns, {len(X_train):,} rows")

        # Get labels
        y_train = events['y'].values

        # Remap labels to {0, 1} for binary classification
        # -1 (stop) -> 0, +1 (target) -> 1
        y_train_binary = (y_train == 1).astype(int)

        # Compute sample weights
        if sample_weight_method == 'uniform':
            sample_weights = np.ones(len(y_train_binary))
        elif sample_weight_method == 'uniqueness':
            # Uniqueness weighting (no time decay)
            sample_weights = compute_uniqueness_decay_weights(
                events,
                bars_train.index,
                decay_lambda=0.0,
            ).values
        elif sample_weight_method == 'uniqueness_decay':
            # Uniqueness + time decay weighting
            sample_weights = compute_uniqueness_decay_weights(
                events,
                bars_train.index,
                decay_lambda=0.005,
            ).values
        else:
            logger.warning(f"  Unknown weighting method: {sample_weight_method}, using uniform")
            sample_weights = np.ones(len(y_train_binary))

        logger.info(f"  Sample weights: min={sample_weights.min():.3f}, "
                   f"mean={sample_weights.mean():.3f}, max={sample_weights.max():.3f}")

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train model
        logger.info("  Training LightGBM model...")
        model = LGBMClassifier(
            objective='binary',
            random_state=42,
            verbosity=-1,
            **model_params
        )

        model.fit(
            X_train_scaled,
            y_train_binary,
            sample_weight=sample_weights,
        )

        logger.info(f"  ✓ Model trained successfully")

        # Store feature columns for later
        feature_columns = list(X_train.columns)

        return (model, scaler), events, feature_columns

    except Exception as e:
        logger.error(f"  ✗ Error training {exp_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model(
    model_bundle: Tuple[object, object],
    bars_test: pd.DataFrame,
    config: Dict,
    feature_columns: List[str],
    instrument_spec: InstrumentSpec,
    threshold: float = 0.40
) -> Optional[Dict]:
    """Test model on Jan-Feb 2026 data."""

    exp_id = config['exp_id']
    model, scaler = model_bundle

    try:
        # Extract config
        labeling_method = config.get('labeling_method', 'triple_barrier')
        labeling_params = config.get('labeling_params', {})
        feature_set_name = config.get('feature_set_name', 'baseline')

        # Generate test events with labeling method
        execution_config = {
            'point_value': 5.0,
            'commission': 0.62,
            'slippage': 0.25,
        }

        # Build labeling config based on method
        if labeling_method == 'triple_barrier':
            labeling_config = {
                'drop_vertical_barrier': True,
                'primary_labeling': {
                    'event_policy': 'cusum',
                    'cusum': {
                        'threshold_atr_mult': 3.0,
                    },
                    'triple_barrier': {
                        'pt_multipliers': [labeling_params.get('pt_mult', 2.0)],
                        'sl_multipliers': [labeling_params.get('sl_mult', 1.5)],
                        'horizon_bars': {
                            '5m': [labeling_params.get('time_mult', 20)],
                        },
                    },
                },
            }
        elif labeling_method == 'trend_scanning':
            labeling_config = {
                'drop_vertical_barrier': True,
                'primary_labeling': {
                    'event_policy': 'trend_scanning',
                    'trend_scanning': {
                        'max_lookahead': labeling_params.get('max_lookahead', 25),
                        'min_t_value': labeling_params.get('min_t_value', 2.0),
                        'cusum_threshold_atr_mult': 3.0,
                    },
                    'triple_barrier': {
                        'pt_multipliers': [2.0],
                        'sl_multipliers': [1.5],
                        'horizon_bars': {
                            '5m': list(range(5, labeling_params.get('max_lookahead', 25) + 1, 5)),
                        },
                    },
                },
            }
        else:
            raise ValueError(f"Unknown labeling method: {labeling_method}")

        events_test = generate_events(
            bars_df=bars_test,
            bar_size="5m",
            labeling_config=labeling_config,
            execution_spec=execution_config,
        )

        # Apply barriers to get labels
        events_test = apply_triplebarrier(
            bars_df=bars_test,
            events_df=events_test,
            bar_size="5m",
            labeling_config=labeling_config,
            execution_spec=execution_config,
            instrument_spec=instrument_spec,
        )

        if labeling_config.get('drop_vertical_barrier', False):
            events_test = events_test[events_test['y'] != 0].reset_index(drop=True)

        # Build features
        features_config = {
            'structure': {
                'ema_fast_period': 12,
                'ema_slow_period': 26,
            },
            'time': {
                'include_time_features': True,
            },
            'enable_advanced_features': False,
        }

        X_test = build_features(
            bars_df=bars_test,
            events_df=events_test,
            bar_size="5m",
            features_config=features_config,
        )

        # Add fractional diff if needed
        if feature_set_name == 'baseline_fracdiff':
            X_test = add_fractional_diff_features(
                X_test,
                bars_test,
                d=0.4,
                columns=['close', 'ema_fast', 'ema_slow'],
            )

        # Handle NaNs
        X_test = X_test.dropna()
        events_test = events_test.loc[X_test.index].copy()

        # Ensure feature alignment
        missing_cols = set(feature_columns) - set(X_test.columns)
        extra_cols = set(X_test.columns) - set(feature_columns)

        if missing_cols:
            logger.warning(f"  Missing features in test: {missing_cols}")
            for col in missing_cols:
                X_test[col] = 0.0

        if extra_cols:
            X_test = X_test.drop(columns=list(extra_cols))

        X_test = X_test[feature_columns]  # Ensure same order

        # Get labels
        y_test = events_test['y'].values
        y_test_binary = (y_test == 1).astype(int)

        # Normalize
        X_test_scaled = scaler.transform(X_test)

        # Predict probabilities
        probs = model.predict_proba(X_test_scaled)[:, 1]  # P(target)

        # Apply threshold
        signals = probs >= threshold

        # Calculate performance
        n_signals = signals.sum()
        n_correct = (signals & (y_test_binary == 1)).sum()
        win_rate = n_correct / n_signals if n_signals > 0 else 0.0

        # Calculate PnL
        # For simplicity: win = +$100, loss = -$100, commission = -$0.87
        n_wins = n_correct
        n_losses = n_signals - n_correct
        pnl = (n_wins * 100.0) - (n_losses * 100.0) - (n_signals * 0.87)

        # Probability stats
        prob_mean = probs.mean()
        prob_std = probs.std()
        prob_min = probs.min()
        prob_max = probs.max()

        result = {
            'exp_id': exp_id,
            'labeling_method': labeling_method,
            'feature_set': feature_set_name,
            'n_test_events': len(events_test),
            'n_signals': int(n_signals),
            'n_wins': int(n_wins),
            'n_losses': int(n_losses),
            'win_rate': float(win_rate),
            'pnl': float(pnl),
            'threshold': threshold,
            'prob_mean': float(prob_mean),
            'prob_std': float(prob_std),
            'prob_min': float(prob_min),
            'prob_max': float(prob_max),
        }

        logger.info(f"  Test Results:")
        logger.info(f"    Events: {len(events_test):,}")
        logger.info(f"    Signals: {n_signals}")
        logger.info(f"    Win Rate: {win_rate:.1%}")
        logger.info(f"    PnL: ${pnl:,.2f}")
        logger.info(f"    Prob range: [{prob_min:.3f}, {prob_max:.3f}]")

        return result

    except Exception as e:
        logger.error(f"  ✗ Error testing {exp_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    logger.info("="*80)
    logger.info("VALIDATE 10 FINALISTS ON JAN-FEB 2026 OOS DATA")
    logger.info("="*80)

    # Load finalist configs
    finalists_path = Path("/tmp/finalist_shortlist_batch1_batch3.json")
    if not finalists_path.exists():
        logger.error(f"Finalists file not found: {finalists_path}")
        sys.exit(1)

    with open(finalists_path) as f:
        finalists_data = json.load(f)

    finalists = finalists_data['finalists']
    logger.info(f"\nLoaded {len(finalists)} finalist configs")

    # Load data
    bars_full, bars_train = load_training_data()
    bars_test = load_test_data()

    # Initialize instrument spec
    instrument_spec = InstrumentSpec(
        symbol="MES",
        tick_size_points=0.25,
        contract_multiplier_usd_per_point=5.0,
    )

    # Test each finalist
    results = []
    start_time = time.time()

    for i, config in enumerate(finalists):
        exp_id = config['exp_id']
        logger.info(f"\n[{i+1}/{len(finalists)}] Processing {exp_id}...")

        # Train model
        model_output = train_model_from_config(config, bars_train, instrument_spec)

        if model_output is None:
            logger.warning(f"  Skipping {exp_id} (training failed)")
            continue

        model_bundle, events_train, feature_columns = model_output

        # Test model
        test_result = test_model(
            model_bundle,
            bars_test,
            config,
            feature_columns,
            instrument_spec,
            threshold=0.40,
        )

        if test_result:
            results.append(test_result)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"VALIDATION COMPLETE ({elapsed/60:.1f} minutes)")
    logger.info(f"{'='*80}")

    # Sort by PnL
    results.sort(key=lambda x: x['pnl'], reverse=True)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY (sorted by PnL)")
    logger.info("="*80)
    logger.info(f"{'Rank':<6} {'Exp ID':<20} {'Label Method':<18} {'Features':<18} "
               f"{'Signals':<8} {'Win%':<8} {'PnL':<12}")
    logger.info("-"*80)

    for i, r in enumerate(results):
        logger.info(f"{i+1:<6} {r['exp_id']:<20} {r['labeling_method']:<18} "
                   f"{r['feature_set']:<18} {r['n_signals']:<8} "
                   f"{r['win_rate']:>6.1%}  ${r['pnl']:>10,.2f}")

    # Check for diversity
    unique_pnls = len(set(r['pnl'] for r in results))
    unique_signals = len(set(r['n_signals'] for r in results))

    logger.info("\n" + "="*80)
    logger.info("DIVERSITY CHECK")
    logger.info("="*80)
    logger.info(f"Unique PnL values: {unique_pnls} (out of {len(results)} models)")
    logger.info(f"Unique signal counts: {unique_signals}")

    if unique_pnls == 1:
        logger.warning("⚠️  ALL MODELS PRODUCED IDENTICAL PnL (like before)")
    else:
        logger.info("✓ Models produced DIFFERENT PnL values (labels matter!)")

    # Save results
    output_path = project_root / "data" / "results" / "finalist_validation_jan_feb_2026.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'validation_date': datetime.now().isoformat(),
            'test_period': 'Jan 1 - Feb 10, 2026',
            'n_finalists': len(finalists),
            'n_successful': len(results),
            'threshold': 0.40,
            'diversity': {
                'unique_pnls': unique_pnls,
                'unique_signals': unique_signals,
            },
            'results': results,
        }, f, indent=2)

    logger.info(f"\n✓ Results saved: {output_path}")

    # Print top 3
    logger.info("\n" + "="*80)
    logger.info("TOP 3 MODELS")
    logger.info("="*80)

    for i, r in enumerate(results[:3]):
        logger.info(f"\n#{i+1}: {r['exp_id']}")
        logger.info(f"  Labeling: {r['labeling_method']}")
        logger.info(f"  Features: {r['feature_set']}")
        logger.info(f"  Signals: {r['n_signals']}")
        logger.info(f"  Win Rate: {r['win_rate']:.1%}")
        logger.info(f"  PnL: ${r['pnl']:,.2f}")
        logger.info(f"  Prob Range: [{r['prob_min']:.3f}, {r['prob_max']:.3f}]")


if __name__ == "__main__":
    main()
