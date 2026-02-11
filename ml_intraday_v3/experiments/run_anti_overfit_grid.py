#!/usr/bin/env python3
"""
Anti-Overfitting Experiment Grid
=================================
Tests 3 levers to reduce the train-test AUC gap (currently 0.19-0.25):
  1. Fewer features (prune weak features to reduce noise)
  2. Smaller model (reduce complexity to fight memorization)
  3. Longer training data (more samples for stable estimation)

Phase 1: Fix window=6mo, sweep model x features (16 experiments)
Phase 2: Best combo from Phase 1, sweep 4 training windows

Usage:
    python -m ml_intraday_v3.experiments.run_anti_overfit_grid --phase 1
    python -m ml_intraday_v3.experiments.run_anti_overfit_grid --phase 2 --best conservative_top10
    python -m ml_intraday_v3.experiments.run_anti_overfit_grid --phase all
"""

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Add project root to path
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

# ─── Experiment Grid Definitions ────────────────────────────────────────────

# Feature sets: top-N by LightGBM gain importance (from diagnostics/feature_importance_gain.csv)
# 'side' is always included as it's the #1 feature and required for bidirectional model.
# Momentum features (rsi_14, macd_hist, etc.) excluded since features.yaml has momentum.enabled=false.
FEATURE_SETS = {
    'full': None,  # All features from build_features()
    'top15': [
        'side', 'autocorr_5', 'vol_regime', 'vol_20', 'ema_ratio',
        'relative_volume', 'lower_wick', 'vol_forecast', 'parkinson_vol',
        'ema_spread', 'log_return_24', 'atr_14', 'bb_position',
        'minute_of_day_sin', 'minute_of_day_cos',
    ],
    'top10': [
        'side', 'autocorr_5', 'vol_regime', 'vol_20', 'ema_ratio',
        'relative_volume', 'lower_wick', 'vol_forecast', 'parkinson_vol',
        'ema_spread',
    ],
    'top5': [
        'side', 'autocorr_5', 'vol_regime', 'vol_20', 'ema_ratio',
    ],
}

# Model complexity configs
MODEL_CONFIGS = {
    'current': dict(
        n_estimators=150, max_depth=6, num_leaves=31,
        min_child_samples=100, reg_alpha=0.1, reg_lambda=0.1,
    ),
    'conservative': dict(
        n_estimators=100, max_depth=4, num_leaves=15,
        min_child_samples=200, reg_alpha=0.3, reg_lambda=0.3,
    ),
    'aggressive': dict(
        n_estimators=50, max_depth=3, num_leaves=7,
        min_child_samples=300, reg_alpha=0.5, reg_lambda=0.5,
    ),
    'minimal': dict(
        n_estimators=30, max_depth=2, num_leaves=4,
        min_child_samples=500, reg_alpha=1.0, reg_lambda=1.0,
    ),
}

# Training window start dates (all end at test window start)
WINDOWS = {
    '6mo': '2024-07-01',   # Current default (6mo before Jan 2025 test)
    '12mo': '2024-01-01',
    '24mo': '2022-01-01',
    'full': '2019-05-06',  # All available data
}

# Walk-forward test schedule: 6 months, Jan-Jun 2025
# Each tuple: (train_start_offset_from_base, train_end, test_start, test_end)
# For the 6mo base window, train starts 6 months before test.
# For longer windows, train_start shifts backward.
WF_WINDOWS = [
    ('2024-07-01', '2024-12-31', '2025-01-01', '2025-01-31'),
    ('2024-08-01', '2025-01-31', '2025-02-01', '2025-02-28'),
    ('2024-09-01', '2025-02-28', '2025-03-01', '2025-03-31'),
    ('2024-10-01', '2025-03-31', '2025-04-01', '2025-04-30'),
    ('2024-11-01', '2025-04-30', '2025-05-01', '2025-05-31'),
    ('2024-12-01', '2025-05-31', '2025-06-01', '2025-06-30'),
]

# Fixed params that don't change across experiments
FIXED_MODEL_PARAMS = dict(
    learning_rate=0.05,
    subsample=0.8,
    subsample_freq=5,
    colsample_bytree=0.8,
    class_weight='balanced',
)


# ─── Core Experiment Logic ──────────────────────────────────────────────────

class CalibratedBinaryModel:
    """Wrapper for LGBMClassifier + IsotonicRegression calibration."""
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator
        self.classes_ = base_model.classes_

    def predict_proba(self, X):
        raw_p1 = self.base_model.predict_proba(X)[:, 1]
        cal_p1 = self.calibrator.predict(raw_p1)
        return np.column_stack([1.0 - cal_p1, cal_p1])

    @property
    def feature_importances_(self):
        return self.base_model.feature_importances_


def run_single_window(
    bars: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    configs: dict,
    instrument_spec: InstrumentSpec,
    model_params: dict,
    feature_list: list | None,
) -> dict:
    """
    Train and evaluate one walk-forward window with given model params and feature subset.
    Returns dict with metrics or error/skip status.
    """
    ts_train_start = pd.Timestamp(train_start, tz='UTC')
    ts_train_end = pd.Timestamp(f"{train_end} 23:59:59", tz='UTC')
    ts_test_start = pd.Timestamp(test_start, tz='UTC')
    ts_test_end = pd.Timestamp(f"{test_end} 23:59:59", tz='UTC')

    try:
        # Split data
        bars_train = bars[(bars.index >= ts_train_start) & (bars.index <= ts_train_end)]
        bars_test = bars[(bars.index >= ts_test_start) & (bars.index <= ts_test_end)]

        if len(bars_train) < 100 or len(bars_test) < 20:
            return {'status': 'skipped', 'reason': 'insufficient_bars'}

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

        if len(events_train) < 200 or len(events_test) < 30:
            return {'status': 'skipped', 'reason': 'insufficient_events',
                    'train_events': len(events_train), 'test_events': len(events_test)}

        # Build features (full set, then filter)
        features_train = build_features(bars_train, "5m", configs['features'])
        features_test = build_features(bars_test, "5m", configs['features'])

        # Merge events with features
        feat_train = features_train.reindex(events_train['t0'].tolist()).reset_index(drop=True)
        feat_test = features_test.reindex(events_test['t0'].tolist()).reset_index(drop=True)

        # Drop meta columns
        meta_cols = ['is_synthetic', 'usable_for_training']
        feat_train = feat_train.drop(columns=[c for c in meta_cols if c in feat_train.columns])
        feat_test = feat_test.drop(columns=[c for c in meta_cols if c in feat_test.columns])

        # Build dataset with side + y + features
        dataset_train = pd.concat([
            events_train[['side', 'y']].reset_index(drop=True),
            feat_train,
        ], axis=1)
        dataset_test = pd.concat([
            events_test[['side', 'y']].reset_index(drop=True),
            feat_test,
        ], axis=1)

        # Binary labels: {-1, 1} -> {0, 1}
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

        # Filter to selected features (if not 'full')
        if feature_list is not None:
            available = [f for f in feature_list if f in X_train.columns]
            if len(available) < len(feature_list):
                missing = set(feature_list) - set(available)
                logger.warning(f"  Missing features (skipped): {missing}")
            X_train = X_train[available]
            X_test = X_test[available]

        if len(X_train) < 200 or len(X_test) < 30:
            return {'status': 'skipped', 'reason': 'insufficient_valid_events'}

        # Sample decay
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

        # Build LGBMClassifier with experiment-specific params
        all_params = {**FIXED_MODEL_PARAMS, **model_params}
        base_model = LGBMClassifier(
            objective='binary',
            random_state=42,
            verbose=-1,
            **all_params,
        )

        # 80/20 stratified split for calibration
        X_model, X_cal, y_model, y_cal, w_model, _ = train_test_split(
            X_train, y_train, w_decay,
            test_size=0.2, random_state=42, stratify=y_train,
        )

        # Train
        base_model.fit(X_model, y_model, sample_weight=w_model)

        # Calibrate
        cal_probs = base_model.predict_proba(X_cal)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(cal_probs, y_cal)
        model = CalibratedBinaryModel(base_model, calibrator)

        # Evaluate
        test_proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_proba)

        train_proba = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_proba)

        prob_stats = {
            'min': float(test_proba.min()),
            'p25': float(np.percentile(test_proba, 25)),
            'p50': float(np.percentile(test_proba, 50)),
            'p75': float(np.percentile(test_proba, 75)),
            'max': float(test_proba.max()),
            'range': float(test_proba.max() - test_proba.min()),
            'signals_gt_055': int((test_proba > 0.55).sum()),
        }

        return {
            'status': 'success',
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'train_auc': round(train_auc, 4),
            'test_auc': round(test_auc, 4),
            'train_events': len(X_train),
            'test_events': len(X_test),
            'target_rate_train': round(float(y_train.mean()), 4),
            'target_rate_test': round(float(y_test.mean()), 4),
            'prob_stats': prob_stats,
            'feature_count': len(X_train.columns),
        }

    except Exception as e:
        logger.error(f"  Window error: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


def run_experiment(
    experiment_id: str,
    bars: pd.DataFrame,
    configs: dict,
    instrument_spec: InstrumentSpec,
    model_config_name: str,
    feature_set_name: str,
    train_window_name: str = '6mo',
) -> dict:
    """Run all 6 walk-forward windows for one experiment configuration."""
    model_params = MODEL_CONFIGS[model_config_name]
    feature_list = FEATURE_SETS[feature_set_name]

    # Determine training window offset for non-6mo windows
    window_start_override = WINDOWS.get(train_window_name)

    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT: {experiment_id}")
    logger.info(f"  Model: {model_config_name} | Features: {feature_set_name} | Window: {train_window_name}")
    logger.info(f"  Params: {model_params}")
    if feature_list:
        logger.info(f"  Feature count: {len(feature_list)}")
    logger.info(f"{'='*70}")

    t0 = time.time()
    window_results = []

    for i, (tr_start_6mo, tr_end, te_start, te_end) in enumerate(WF_WINDOWS):
        # For non-6mo windows, override the train start
        if train_window_name == '6mo':
            tr_start = tr_start_6mo
        else:
            tr_start = window_start_override

        logger.info(f"  Window {i+1}/6: train {tr_start} to {tr_end}, test {te_start} to {te_end}")

        result = run_single_window(
            bars=bars,
            train_start=tr_start,
            train_end=tr_end,
            test_start=te_start,
            test_end=te_end,
            configs=configs,
            instrument_spec=instrument_spec,
            model_params=model_params,
            feature_list=feature_list,
        )
        window_results.append(result)

        if result['status'] == 'success':
            logger.info(f"    Train AUC: {result['train_auc']:.4f}, "
                        f"Test AUC: {result['test_auc']:.4f}, "
                        f"Signals>0.55: {result['prob_stats']['signals_gt_055']}")
        else:
            logger.info(f"    {result['status']}: {result.get('reason', result.get('error', ''))}")

    elapsed = time.time() - t0

    # Aggregate
    successful = [r for r in window_results if r['status'] == 'success']
    if successful:
        test_aucs = [r['test_auc'] for r in successful]
        train_aucs = [r['train_auc'] for r in successful]
        total_signals = sum(r['prob_stats']['signals_gt_055'] for r in successful)
        prob_ranges = [r['prob_stats']['range'] for r in successful]

        summary = {
            'experiment_id': experiment_id,
            'feature_set': feature_set_name,
            'model_config': model_config_name,
            'train_window': train_window_name,
            'successful_windows': len(successful),
            'median_test_auc': round(float(np.median(test_aucs)), 4),
            'mean_test_auc': round(float(np.mean(test_aucs)), 4),
            'std_test_auc': round(float(np.std(test_aucs)), 4),
            'min_test_auc': round(float(np.min(test_aucs)), 4),
            'max_test_auc': round(float(np.max(test_aucs)), 4),
            'median_train_auc': round(float(np.median(train_aucs)), 4),
            'train_test_gap': round(float(np.median(train_aucs) - np.median(test_aucs)), 4),
            'signals_above_055': total_signals,
            'prob_range_median': round(float(np.median(prob_ranges)), 4),
            'elapsed_seconds': round(elapsed, 1),
            'windows': window_results,
        }
    else:
        summary = {
            'experiment_id': experiment_id,
            'feature_set': feature_set_name,
            'model_config': model_config_name,
            'train_window': train_window_name,
            'successful_windows': 0,
            'median_test_auc': None,
            'train_test_gap': None,
            'signals_above_055': 0,
            'elapsed_seconds': round(elapsed, 1),
            'windows': window_results,
        }

    logger.info(f"\n  RESULT: median_test_auc={summary.get('median_test_auc')}, "
                f"gap={summary.get('train_test_gap')}, "
                f"signals={summary.get('signals_above_055')}, "
                f"time={elapsed:.0f}s")

    return summary


# ─── Phase Runners ──────────────────────────────────────────────────────────

def load_data_and_configs():
    """Load bars and configs once."""
    data_path = Path("data/processed/mes_bars_databento_rth.h5")
    logger.info(f"Loading data from: {data_path}")
    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()
    logger.info(f"Data: {len(bars):,} bars ({bars.index[0].date()} to {bars.index[-1].date()})")

    configs = {}
    config_dir = Path("ml_intraday_v3/configs")
    for name, fname in [('labeling', 'labeling.yaml'), ('execution', 'execution_spec.yaml'),
                         ('features', 'features.yaml'), ('training', 'training.yaml')]:
        with open(config_dir / fname) as f:
            configs[name] = yaml.safe_load(f)

    instrument_spec = InstrumentSpec(
        symbol="MES",
        tick_size_points=0.25,
        contract_multiplier_usd_per_point=5.0,
    )

    return bars, configs, instrument_spec


def run_phase1(bars, configs, instrument_spec):
    """Phase 1: sweep model x features with 6mo window (16 experiments)."""
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: MODEL x FEATURES GRID (6-month window)")
    logger.info("=" * 80)

    feature_names = ['full', 'top15', 'top10', 'top5']
    model_names = ['current', 'conservative', 'aggressive', 'minimal']

    results = []
    exp_num = 0
    total = len(feature_names) * len(model_names)

    for feat_name in feature_names:
        for model_name in model_names:
            exp_num += 1
            exp_id = f"p1_{model_name}_{feat_name}"
            logger.info(f"\n>>> Experiment {exp_num}/{total}: {exp_id}")

            result = run_experiment(
                experiment_id=exp_id,
                bars=bars,
                configs=configs,
                instrument_spec=instrument_spec,
                model_config_name=model_name,
                feature_set_name=feat_name,
                train_window_name='6mo',
            )
            results.append(result)

            # Progress table
            print_progress_table(results, "PHASE 1 PROGRESS")

    return results


def run_phase2(bars, configs, instrument_spec, best_combo: str):
    """Phase 2: sweep training windows with best combo from Phase 1."""
    parts = best_combo.split('_', 1)
    if len(parts) != 2:
        logger.error(f"Invalid --best format '{best_combo}'. Expected 'modelname_featureset' e.g. 'conservative_top10'")
        sys.exit(1)

    model_name, feat_name = parts
    if model_name not in MODEL_CONFIGS:
        logger.error(f"Unknown model config: {model_name}. Options: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    if feat_name not in FEATURE_SETS:
        logger.error(f"Unknown feature set: {feat_name}. Options: {list(FEATURE_SETS.keys())}")
        sys.exit(1)

    logger.info("\n" + "=" * 80)
    logger.info(f"PHASE 2: TRAINING WINDOW SWEEP (model={model_name}, features={feat_name})")
    logger.info("=" * 80)

    window_names = ['6mo', '12mo', '24mo', 'full']
    results = []

    for i, win_name in enumerate(window_names):
        exp_id = f"p2_{model_name}_{feat_name}_{win_name}"
        logger.info(f"\n>>> Experiment {i+1}/{len(window_names)}: {exp_id}")

        result = run_experiment(
            experiment_id=exp_id,
            bars=bars,
            configs=configs,
            instrument_spec=instrument_spec,
            model_config_name=model_name,
            feature_set_name=feat_name,
            train_window_name=win_name,
        )
        results.append(result)

        print_progress_table(results, "PHASE 2 PROGRESS")

    return results


# ─── Output & Reporting ─────────────────────────────────────────────────────

def print_progress_table(results: list, title: str):
    """Print a sorted results table to console."""
    # Filter to successful experiments
    valid = [r for r in results if r.get('median_test_auc') is not None]
    if not valid:
        return

    valid.sort(key=lambda r: r['median_test_auc'], reverse=True)

    print(f"\n{title}")
    print("=" * 90)
    print(f"{'#':>3}  {'Features':<8} {'Model':<14} {'Window':<6} "
          f"{'Med AUC':>8} {'Gap':>6} {'Signals':>8} {'ProbRng':>8} {'Time':>6}")
    print("-" * 90)

    for i, r in enumerate(valid):
        print(f"{i+1:>3}  {r['feature_set']:<8} {r['model_config']:<14} {r['train_window']:<6} "
              f"{r['median_test_auc']:>8.4f} {r['train_test_gap']:>+6.3f} "
              f"{r['signals_above_055']:>8} {r.get('prob_range_median', 0):>8.4f} "
              f"{r['elapsed_seconds']:>5.0f}s")

    # Mark baseline
    baseline = [r for r in valid if r['model_config'] == 'current' and r['feature_set'] == 'full']
    if baseline:
        bl = baseline[0]
        rank = valid.index(bl) + 1
        print(f"\nBaseline (current/full): rank {rank}/{len(valid)}, "
              f"median AUC {bl['median_test_auc']:.4f}, gap {bl['train_test_gap']:+.3f}")

    print()


def save_results(results: list, phase_name: str, output_dir: Path):
    """Save results to JSON and summary CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full JSON
    json_path = output_dir / f"{phase_name}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Full results saved to: {json_path}")

    # Summary CSV
    csv_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != 'windows'}
        csv_rows.append(row)

    csv_path = output_dir / "summary.csv"
    df = pd.DataFrame(csv_rows)

    # Append if exists, else create
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, df], ignore_index=True)
        # Deduplicate by experiment_id, keep latest
        df = df.drop_duplicates(subset='experiment_id', keep='last')

    df.to_csv(csv_path, index=False)
    logger.info(f"Summary CSV saved to: {csv_path}")

    return json_path


def print_final_verdict(results: list):
    """Print go/no-go decision."""
    valid = [r for r in results if r.get('median_test_auc') is not None]
    if not valid:
        logger.info("No successful experiments.")
        return

    best = max(valid, key=lambda r: r['median_test_auc'])

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"Best experiment: {best['experiment_id']}")
    print(f"  Median test AUC: {best['median_test_auc']:.4f}")
    print(f"  Train-test gap:  {best['train_test_gap']:+.4f}")
    print(f"  Signals > 0.55:  {best['signals_above_055']}")

    if best['median_test_auc'] > 0.55:
        print("\n>>> STRONG EDGE detected. Deploy this config.")
    elif best['median_test_auc'] > 0.52 and best['signals_above_055'] > 0:
        print("\n>>> WEAK EDGE detected. Worth optimizing barriers and retesting.")
    elif best['median_test_auc'] > 0.52:
        print("\n>>> MARGINAL EDGE but no actionable signals above 0.55 threshold.")
        print("    Consider lowering confidence threshold from 0.55 to 0.52.")
    else:
        print("\n>>> NO EDGE. ML approach does not have predictive power with this data.")
        print("    Recommend: rule-based system or alternative feature engineering.")
    print()


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anti-overfitting experiment grid")
    parser.add_argument('--phase', type=str, required=True,
                        choices=['1', '2', 'all'],
                        help="Phase to run: 1, 2, or all")
    parser.add_argument('--best', type=str, default=None,
                        help="Best combo from Phase 1 for Phase 2 (e.g. 'conservative_top10')")
    args = parser.parse_args()

    output_dir = Path("ml_intraday_v3/experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    bars, configs, instrument_spec = load_data_and_configs()

    all_results = []

    if args.phase in ('1', 'all'):
        p1_results = run_phase1(bars, configs, instrument_spec)
        save_results(p1_results, "phase1_grid", output_dir)
        all_results.extend(p1_results)

        # Print final Phase 1 table
        print_progress_table(p1_results, "PHASE 1 FINAL RESULTS (sorted by median test AUC)")

        # Auto-select best for Phase 2 if running 'all'
        if args.phase == 'all':
            valid = [r for r in p1_results if r.get('median_test_auc') is not None]
            if valid:
                best = max(valid, key=lambda r: r['median_test_auc'])
                best_combo = f"{best['model_config']}_{best['feature_set']}"
                logger.info(f"Auto-selected best combo for Phase 2: {best_combo}")
                args.best = best_combo

    if args.phase in ('2', 'all'):
        if not args.best:
            logger.error("Phase 2 requires --best argument (e.g. --best conservative_top10)")
            sys.exit(1)

        p2_results = run_phase2(bars, configs, instrument_spec, args.best)
        save_results(p2_results, "phase2_windows", output_dir)
        all_results.extend(p2_results)

        print_progress_table(p2_results, "PHASE 2 FINAL RESULTS (sorted by median test AUC)")

    print_final_verdict(all_results)


if __name__ == '__main__':
    main()
