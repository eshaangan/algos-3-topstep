"""
Single experiment runner for comprehensive grid search.

Runs one configuration through full walk-forward validation.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from ml_intraday_v3.labels.events import generate_events, balance_events
from ml_intraday_v3.labels.triple_barrier import apply_triplebarrier
from ml_intraday_v3.features.build import build_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_weights(events_df: pd.DataFrame, method: str) -> np.ndarray:
    """
    Create sample weights based on specified method.
    
    Args:
        events_df: DataFrame with events (must have timestamp index)
        method: One of 'uniform', 'time_decay_0.005', 'time_decay_0.01'
    
    Returns:
        Array of sample weights
    """
    n_samples = len(events_df)
    
    if method == 'uniform':
        return np.ones(n_samples)
    
    elif method.startswith('time_decay_'):
        # Extract lambda value from method name
        lambda_val = float(method.split('_')[-1])
        
        # Create time-based weights (more recent = higher weight)
        # Normalize timestamps to [0, 1] range
        timestamps = pd.to_datetime(events_df.index).astype(int).values
        t_min, t_max = timestamps.min(), timestamps.max()
        t_norm = (timestamps - t_min) / (t_max - t_min + 1e-10)
        
        # Apply exponential decay: w = exp(lambda * t)
        weights = np.exp(lambda_val * t_norm)
        
        # Normalize to sum to n_samples (for consistency with uniform)
        weights = weights * n_samples / weights.sum()
        
        return weights
    
    else:
        raise ValueError(f"Unknown sample weight method: {method}")


def run_single_fold(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: Dict,
    fold_num: int
) -> Dict:
    """
    Run a single fold of walk-forward validation.
    
    Args:
        train_data: Training data (raw OHLCV)
        test_data: Test data (raw OHLCV)
        config: Experiment configuration
        fold_num: Fold number
    
    Returns:
        Dictionary with fold results
    """
    logger.info(f"  Fold {fold_num}: Train={len(train_data)}, Test={len(test_data)}")
    
    # 1. Generate CUSUM events
    barrier_config = config['labeling']
    
    train_events = generate_events(
        train_data,
        threshold=0.0003,  # Standard CUSUM threshold
        vertical_barrier_hours=barrier_config['hz']
    )
    test_events = generate_events(
        test_data,
        threshold=0.0003,
        vertical_barrier_hours=barrier_config['hz']
    )
    
    # 2. Apply triple-barrier labeling
    train_labels = apply_triplebarrier(
        train_data,
        train_events,
        pt_multiplier=barrier_config['pt'],
        sl_multiplier=barrier_config['sl']
    )
    test_labels = apply_triplebarrier(
        test_data,
        test_events,
        pt_multiplier=barrier_config['pt'],
        sl_multiplier=barrier_config['sl']
    )
    
    # 3. Build features
    train_features = build_features(train_data, train_events)
    test_features = build_features(test_data, test_events)
    
    # 4. Merge with labels
    train_df = train_features.join(train_labels[['y']], how='inner')
    test_df = test_features.join(test_labels[['y']], how='inner')
    
    # 5. Filter to selected feature set
    feature_cols = config['feature_set']
    if feature_cols is None:
        # Use all features except target
        feature_cols = [c for c in train_df.columns if c != 'y']
    
    # Ensure 'side' is in features if needed
    if 'side' not in feature_cols:
        feature_cols = ['side'] + feature_cols
    
    # Filter features
    X_train = train_df[feature_cols]
    y_train = train_df['y']
    X_test = test_df[feature_cols]
    y_test = test_df['y']
    
    # 6. Balance training events (50/50 stop/target)
    X_train_bal, y_train_bal = balance_events(X_train, y_train)
    
    # 7. Create sample weights
    sample_weights = create_sample_weights(
        X_train_bal,
        method=config['sample_weight']
    )
    
    # 8. Remap labels for binary classification: -1 (stop) -> 0, 1 (target) -> 1
    y_train_binary = (y_train_bal == 1).astype(int)
    y_test_binary = (y_test == 1).astype(int)
    
    # 9. Train model
    model = LGBMClassifier(
        objective='binary',
        **config['model_params'],
        verbose=-1,
        force_col_wise=True
    )
    
    model.fit(
        X_train_bal.to_numpy(),
        y_train_binary,
        sample_weight=sample_weights
    )
    
    # 10. Apply calibration if specified
    if config['calibration'] is not None:
        calibrator = CalibratedClassifierCV(
            model,
            method=config['calibration'],
            cv='prefit'
        )
        # Use a validation set for calibration (last 20% of training data)
        cal_size = int(0.2 * len(X_train_bal))
        X_cal = X_train_bal.iloc[-cal_size:]
        y_cal = y_train_binary.iloc[-cal_size:]
        
        calibrator.fit(X_cal.to_numpy(), y_cal)
        model = calibrator
    
    # 11. Predict on train and test
    y_train_proba = model.predict_proba(X_train_bal.to_numpy())[:, 1]
    y_test_proba = model.predict_proba(X_test.to_numpy())[:, 1]
    
    # 12. Compute metrics
    train_auc = roc_auc_score(y_train_binary, y_train_proba)
    test_auc = roc_auc_score(y_test_binary, y_test_proba)
    
    # Probability statistics on test set
    pct_above_055 = (y_test_proba > 0.55).mean()
    pct_above_060 = (y_test_proba > 0.60).mean()
    mean_prob = y_test_proba.mean()
    std_prob = y_test_proba.std()
    
    # Estimate signals per day (assuming 5-min bars, ~80 bars/day)
    est_signals_per_day = len(test_events) / (len(test_data) / 80)
    est_trades_per_day = est_signals_per_day * pct_above_055
    
    return {
        'fold': fold_num,
        'train_auc': float(train_auc),
        'test_auc': float(test_auc),
        'train_test_gap': float(train_auc - test_auc),
        'pct_signals_above_055': float(pct_above_055),
        'pct_signals_above_060': float(pct_above_060),
        'mean_test_prob': float(mean_prob),
        'std_test_prob': float(std_prob),
        'n_train': len(X_train_bal),
        'n_test': len(X_test),
        'est_signals_per_day': float(est_signals_per_day),
        'est_trades_per_day': float(est_trades_per_day)
    }


def run_single_experiment(config: Dict, data_path: Path) -> Dict:
    """
    Run a single experiment configuration through full walk-forward validation.
    
    Args:
        config: Experiment configuration dictionary
        data_path: Path to data directory
    
    Returns:
        Dictionary with experiment results
    """
    exp_id = config['exp_id']
    logger.info(f"Starting experiment {exp_id}")
    start_time = time.time()
    
    # Load data
    data_file = data_path / "MES_5min_Oct2024_Dec2025.parquet"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_parquet(data_file)
    logger.info(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")

    # Ensure index is timezone-aware (match data timezone)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    # Define train/test splits (6-fold walk-forward)
    # Train on all of 2024, test on 2-month windows in 2025
    train_start_base = pd.Timestamp('2024-01-01', tz='UTC')
    train_end = pd.Timestamp('2025-01-01', tz='UTC')
    train_data = df[(df.index >= train_start_base) & (df.index < train_end)]

    test_periods = [
        (pd.Timestamp('2025-01-01', tz='UTC'), pd.Timestamp('2025-02-28', tz='UTC')),
        (pd.Timestamp('2025-03-01', tz='UTC'), pd.Timestamp('2025-04-30', tz='UTC')),
        (pd.Timestamp('2025-05-01', tz='UTC'), pd.Timestamp('2025-06-30', tz='UTC')),
    ]

    # Apply training window restriction
    train_window_months = config['training_window_months']
    if train_window_months < 12:
        # Use only last N months of training data
        train_start = train_end - pd.DateOffset(months=train_window_months)
        train_data = train_data[train_data.index >= train_start]
    
    logger.info(f"Training window: {train_data.index.min()} to {train_data.index.max()} ({train_window_months} months)")
    
    # Run each fold
    fold_results = []
    for i, (test_start, test_end) in enumerate(test_periods, 1):
        test_data = df[(df.index >= test_start) & (df.index <= test_end)]
        
        try:
            fold_result = run_single_fold(train_data, test_data, config, i)
            fold_results.append(fold_result)
        except Exception as e:
            logger.error(f"  Fold {i} failed: {e}")
            fold_results.append({
                'fold': i,
                'error': str(e)
            })
    
    # Aggregate metrics across folds
    successful_folds = [f for f in fold_results if 'error' not in f]
    
    if len(successful_folds) == 0:
        return {
            'exp_id': exp_id,
            'config': config,
            'status': 'FAILED',
            'error': 'All folds failed',
            'folds': fold_results
        }
    
    # Compute summary statistics
    median_test_auc = float(np.median([f['test_auc'] for f in successful_folds]))
    median_train_auc = float(np.median([f['train_auc'] for f in successful_folds]))
    mean_train_test_gap = float(np.mean([f['train_test_gap'] for f in successful_folds]))
    std_test_auc = float(np.std([f['test_auc'] for f in successful_folds]))
    
    median_pct_above_055 = float(np.median([f['pct_signals_above_055'] for f in successful_folds]))
    median_pct_above_060 = float(np.median([f['pct_signals_above_060'] for f in successful_folds]))
    mean_test_prob = float(np.mean([f['mean_test_prob'] for f in successful_folds]))
    mean_std_prob = float(np.mean([f['std_test_prob'] for f in successful_folds]))
    
    median_est_trades_per_day = float(np.median([f['est_trades_per_day'] for f in successful_folds]))
    
    elapsed = time.time() - start_time
    
    result = {
        'exp_id': exp_id,
        'config': config,
        'status': 'SUCCESS',
        'runtime_seconds': elapsed,
        'folds': fold_results,
        'summary': {
            'median_test_auc': median_test_auc,
            'median_train_auc': median_train_auc,
            'mean_train_test_gap': mean_train_test_gap,
            'std_test_auc': std_test_auc,
            'median_pct_signals_above_055': median_pct_above_055,
            'median_pct_signals_above_060': median_pct_above_060,
            'mean_test_prob': mean_test_prob,
            'mean_std_prob': mean_std_prob,
            'median_est_trades_per_day': median_est_trades_per_day,
            'n_successful_folds': len(successful_folds),
            'n_total_folds': len(fold_results)
        }
    }
    
    logger.info(f"Experiment {exp_id} complete: AUC={median_test_auc:.3f}, Gap={mean_train_test_gap:.3f}, "
                f"Signals>0.55={median_pct_above_055:.1%}, Trades/day={median_est_trades_per_day:.1f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Run single grid search experiment")
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config JSON')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Run experiment
    result = run_single_experiment(config, Path(args.data_dir))
    
    # Save result
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
