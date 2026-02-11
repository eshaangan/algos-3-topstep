"""
Comprehensive grid search - compatible with ml_intraday_v3 architecture.

Uses existing config structure (labeling.yaml, execution_spec.yaml, training.yaml).
"""

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from ml_intraday_v3.labels.events import generate_events, balance_events
from ml_intraday_v3.labels.triple_barrier import apply_triplebarrier
from ml_intraday_v3.features.build import build_features
from ml_intraday_v3.core.instrument import InstrumentSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_base_configs(config_dir: Path) -> Dict:
    """Load base configuration files."""
    configs = {}

    # Load labeling config
    with open(config_dir / 'labeling.yaml', 'r') as f:
        configs['labeling'] = yaml.safe_load(f)

    # Load execution spec
    with open(config_dir / 'execution_spec.yaml', 'r') as f:
        configs['execution'] = yaml.safe_load(f)

    # Load features config
    with open(config_dir / 'features.yaml', 'r') as f:
        configs['features'] = yaml.safe_load(f)

    return configs


def create_experiment_configs(base_configs: Dict, exp_config: Dict) -> Dict:
    """
    Create experiment-specific configs by modifying base configs.
    
    Args:
        base_configs: Base labeling/execution configs
        exp_config: Experiment parameters (pt, sl, hz, features, etc.)
    
    Returns:
        Modified configs for this experiment
    """
    configs = deepcopy(base_configs)
    
    # Modify labeling barriers
    tb = configs['labeling']['primary_labeling']['triple_barrier']
    tb['pt_multipliers'] = [float(exp_config['labeling']['pt'])]
    tb['sl_multipliers'] = [float(exp_config['labeling']['sl'])]
    tb['horizon_bars']['5m'] = [int(exp_config['labeling']['hz'])]
    
    # Modify max_holding_bars to match
    configs['execution']['holding_constraints']['max_holding_bars']['5m'] = int(exp_config['labeling']['hz'])
    
    return configs


def run_single_fold(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    exp_config: Dict,
    configs: Dict,
    fold_num: int,
    return_predictions: bool = False
) -> Dict:
    """
    Run a single fold of walk-forward validation.

    Args:
        train_data: Training bars (OHLCV)
        test_data: Test bars (OHLCV)
        exp_config: Experiment configuration
        configs: Labeling/execution configs
        fold_num: Fold number
        return_predictions: If True, include predictions and outcomes in results

    Returns:
        Dictionary with fold results (includes 'predictions' and 'outcomes' if return_predictions=True)
    """
    logger.info(f"  Fold {fold_num}: Train={len(train_data)}, Test={len(test_data)}")
    
    # 1. Generate events
    train_events = generate_events(
        bars_df=train_data,
        bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution']
    )
    
    test_events = generate_events(
        bars_df=test_data,
        bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution']
    )
    
    logger.info(f"    Events: Train={len(train_events)}, Test={len(test_events)}")
    
    if len(train_events) == 0 or len(test_events) == 0:
        return {
            'fold': fold_num,
            'error': 'No events generated'
        }
    
    # 2. Apply triple-barrier labeling
    instrument_spec = InstrumentSpec(
        symbol="MES",
        tick_size_points=0.25,
        contract_multiplier_usd_per_point=5.0
    )

    train_labels = apply_triplebarrier(
        bars_df=train_data,
        events_df=train_events,
        bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution'],
        instrument_spec=instrument_spec
    )

    test_labels = apply_triplebarrier(
        bars_df=test_data,
        events_df=test_events,
        bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution'],
        instrument_spec=instrument_spec
    )
    
    # 3. Build features (full bars, not events!)
    train_features = build_features(train_data, "5m", configs['features'])
    test_features = build_features(test_data, "5m", configs['features'])

    # 3b. Join events with features (events are indexed by t0)
    # Features are indexed by bar timestamp, join on t0
    train_event_features = train_features.loc[train_events['t0']].copy()
    train_event_features.index = train_events.index  # Reset index to event IDs
    # Add 'side' from events
    train_event_features['side'] = train_events['side']

    test_event_features = test_features.loc[test_events['t0']].copy()
    test_event_features.index = test_events.index
    # Add 'side' from events
    test_event_features['side'] = test_events['side']
    
    # 4. Balance training events FIRST (50/50 LONG/SHORT)
    # Balance operates on events DataFrame with 'side' column
    train_events_bal = balance_events(train_events, target_long_ratio=0.5, method='undersample')

    logger.info(f"    Balanced events: {len(train_events_bal)} (from {len(train_events)})")

    # 5. Now join balanced events with labels and features
    # Re-extract features for balanced events
    train_event_features_bal = train_features.loc[train_events_bal['t0']].copy()
    train_event_features_bal.index = train_events_bal.index
    train_event_features_bal['side'] = train_events_bal['side']

    # Join labels
    train_df = train_event_features_bal.join(train_labels[['y']], how='inner')

    # For test set, use all events (no balancing)
    test_df = test_event_features.join(test_labels[['y']], how='inner')

    logger.info(f"    Final: Train={len(train_df)}, Test={len(test_df)}")

    if len(train_df) < 50 or len(test_df) < 10:
        return {
            'fold': fold_num,
            'error': f'Insufficient samples (train={len(train_df)}, test={len(test_df)})'
        }

    # 6. Select features
    feature_set = exp_config['feature_set']
    if feature_set is None:
        # Use all features except target
        feature_cols = [c for c in train_df.columns if c not in ['y', 'usable_for_training']]
    else:
        feature_cols = feature_set

    # Ensure columns exist
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        return {
            'fold': fold_num,
            'error': f'Missing features: {missing[:5]}'
        }

    X_train = train_df[feature_cols]
    y_train = train_df['y']
    X_test = test_df[feature_cols]
    y_test = test_df['y']


    logger.info(f"    Training: {len(X_train)} samples, {len(feature_cols)} features")

    # 7. Create sample weights
    sample_weights = create_sample_weights(X_train, exp_config['sample_weight'])

    # 8. Remap labels for binary: {-1, 1} -> {0, 1}
    y_train_binary = (y_train == 1).astype(int)
    y_test_binary = (y_test == 1).astype(int)

    # 9. Train model with optional calibration
    if exp_config['calibration'] is not None:
        # Use CalibratedClassifierCV with cross-validation
        base_model = LGBMClassifier(
            objective='binary',
            **exp_config['model_params'],
            verbose=-1,
            force_col_wise=True
        )

        model = CalibratedClassifierCV(
            base_model,
            method=exp_config['calibration'],
            cv=3  # 3-fold cross-validation for calibration
        )
        model.fit(
            X_train.to_numpy(),
            y_train_binary,
            sample_weight=sample_weights
        )
    else:
        # No calibration
        model = LGBMClassifier(
            objective='binary',
            **exp_config['model_params'],
            verbose=-1,
            force_col_wise=True
        )
        model.fit(
            X_train.to_numpy(),
            y_train_binary,
            sample_weight=sample_weights
        )

    # 11. Predict
    y_train_proba = model.predict_proba(X_train.to_numpy())[:, 1]
    y_test_proba = model.predict_proba(X_test.to_numpy())[:, 1]
    
    # 12. Compute metrics
    train_auc = roc_auc_score(y_train_binary, y_train_proba)
    test_auc = roc_auc_score(y_test_binary, y_test_proba)
    
    pct_above_055 = (y_test_proba > 0.55).mean()
    pct_above_060 = (y_test_proba > 0.60).mean()
    mean_prob = y_test_proba.mean()
    std_prob = y_test_proba.std()
    
    # Estimate trades/day (assuming ~80 bars/day for 5m)
    est_signals_per_day = len(test_events) / (len(test_data) / 80)
    est_trades_per_day = est_signals_per_day * pct_above_055
    
    logger.info(f"    AUC: Train={train_auc:.3f}, Test={test_auc:.3f}, Gap={train_auc-test_auc:.3f}")
    logger.info(f"    Signals>0.55: {pct_above_055:.1%}, Est trades/day: {est_trades_per_day:.1f}")

    result = {
        'fold': fold_num,
        'train_auc': float(train_auc),
        'test_auc': float(test_auc),
        'train_test_gap': float(train_auc - test_auc),
        'pct_signals_above_055': float(pct_above_055),
        'pct_signals_above_060': float(pct_above_060),
        'mean_test_prob': float(mean_prob),
        'std_test_prob': float(std_prob),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_train_events': len(train_events),
        'n_test_events': len(test_events),
        'est_signals_per_day': float(est_signals_per_day),
        'est_trades_per_day': float(est_trades_per_day)
    }

    # Optionally include predictions and outcomes
    if return_predictions:
        result['predictions'] = y_test_proba
        result['outcomes'] = y_test_binary.values

    return result


def create_sample_weights(events_df: pd.DataFrame, method: str) -> np.ndarray:
    """Create sample weights based on method."""
    n_samples = len(events_df)
    
    if method == 'uniform':
        return np.ones(n_samples)
    
    elif method.startswith('time_decay_'):
        lambda_val = float(method.split('_')[-1])
        
        # Time-based exponential decay
        timestamps = pd.to_datetime(events_df.index).astype(int).values
        t_min, t_max = timestamps.min(), timestamps.max()
        t_norm = (timestamps - t_min) / (t_max - t_min + 1e-10)
        
        weights = np.exp(lambda_val * t_norm)
        weights = weights * n_samples / weights.sum()
        
        return weights
    
    else:
        raise ValueError(f"Unknown sample weight method: {method}")


def run_experiment(exp_config: Dict, data_path: Path, config_dir: Path) -> Dict:
    """
    Run a single experiment configuration through walk-forward validation.
    
    Args:
        exp_config: Experiment configuration
        data_path: Path to data directory
        config_dir: Path to configs directory
    
    Returns:
        Experiment results
    """
    exp_id = exp_config['exp_id']
    logger.info(f"=== Experiment {exp_id} ===")
    start_time = time.time()
    
    # Load base configs
    base_configs = load_base_configs(config_dir)
    
    # Create experiment-specific configs
    configs = create_experiment_configs(base_configs, exp_config)
    
    # Load data
    data_file = data_path / "MES_5min_Oct2024_Dec2025.parquet"
    df = pd.read_parquet(data_file)
    logger.info(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
    
    # Ensure timezone-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    
    # Define train/test splits
    train_start_base = pd.Timestamp('2024-10-01', tz='UTC')
    train_end = pd.Timestamp('2025-01-01', tz='UTC')
    train_data = df[(df.index >= train_start_base) & (df.index < train_end)]
    
    # Test periods (3 folds: Jan-Feb, Mar-Apr, May-Jun)
    test_periods = [
        (pd.Timestamp('2025-01-01', tz='UTC'), pd.Timestamp('2025-02-28', tz='UTC')),
        (pd.Timestamp('2025-03-01', tz='UTC'), pd.Timestamp('2025-04-30', tz='UTC')),
        (pd.Timestamp('2025-05-01', tz='UTC'), pd.Timestamp('2025-06-30', tz='UTC')),
    ]
    
    # Apply training window restriction
    train_window_months = exp_config['training_window_months']
    if train_window_months < 3:
        train_start = train_end - pd.DateOffset(months=train_window_months)
        train_data = train_data[train_data.index >= train_start]
    
    logger.info(f"Training: {train_data.index.min()} to {train_data.index.max()} ({train_window_months} months)")
    
    # Run folds
    fold_results = []
    for i, (test_start, test_end) in enumerate(test_periods, 1):
        test_data = df[(df.index >= test_start) & (df.index <= test_end)]
        
        try:
            result = run_single_fold(train_data, test_data, exp_config, configs, i)
            fold_results.append(result)
        except Exception as e:
            import traceback
            logger.error(f"  Fold {i} failed: {e}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            fold_results.append({'fold': i, 'error': str(e)})
    
    # Aggregate results
    successful_folds = [f for f in fold_results if 'error' not in f]
    
    if len(successful_folds) == 0:
        return {
            'exp_id': exp_id,
            'config': exp_config,
            'status': 'FAILED',
            'error': 'All folds failed',
            'folds': fold_results,
            'runtime_seconds': time.time() - start_time
        }
    
    summary = {
        'median_test_auc': float(np.median([f['test_auc'] for f in successful_folds])),
        'median_train_auc': float(np.median([f['train_auc'] for f in successful_folds])),
        'mean_train_test_gap': float(np.mean([f['train_test_gap'] for f in successful_folds])),
        'std_test_auc': float(np.std([f['test_auc'] for f in successful_folds])),
        'median_pct_signals_above_055': float(np.median([f['pct_signals_above_055'] for f in successful_folds])),
        'median_pct_signals_above_060': float(np.median([f['pct_signals_above_060'] for f in successful_folds])),
        'mean_test_prob': float(np.mean([f['mean_test_prob'] for f in successful_folds])),
        'mean_std_prob': float(np.mean([f['std_test_prob'] for f in successful_folds])),
        'median_est_trades_per_day': float(np.median([f['est_trades_per_day'] for f in successful_folds])),
        'n_successful_folds': len(successful_folds),
        'n_total_folds': len(fold_results)
    }
    
    elapsed = time.time() - start_time
    
    logger.info(f"✅ {exp_id}: AUC={summary['median_test_auc']:.3f}, "
                f"Gap={summary['mean_train_test_gap']:.3f}, "
                f"Sigs>0.55={summary['median_pct_signals_above_055']:.1%}, "
                f"Trades/day={summary['median_est_trades_per_day']:.1f}")
    
    return {
        'exp_id': exp_id,
        'config': exp_config,
        'status': 'SUCCESS',
        'runtime_seconds': elapsed,
        'folds': fold_results,
        'summary': summary
    }


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive grid search experiment")
    parser.add_argument('--config', type=str, required=True, help='Experiment config JSON')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--config-dir', type=str, required=True, help='Config directory')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    # Load experiment config
    with open(args.config, 'r') as f:
        exp_config = json.load(f)
    
    # Run experiment
    result = run_experiment(
        exp_config,
        Path(args.data_dir),
        Path(args.config_dir)
    )
    
    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
