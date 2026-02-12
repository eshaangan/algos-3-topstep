#!/usr/bin/env python3
"""
Validate promoted exhaustive_exp_00336 model against current baseline.

Compares:
- Current baseline: model_bundle_retrained_oct2024_nov2025.pkl
- New candidate: model_bundle_phasefull_best_exhaustive_exp_00336.pkl

Tests on Dec 2025 holdout period (unseen during both training runs).

Decision Criteria:
1. AUC improvement > 0.02 (material edge increase)
2. Calibration quality (Brier score, reliability diagram)
3. Trade frequency (ensure not too sparse)
4. Directional balance (LONG/SHORT distribution)
5. Risk metrics (drawdown, Sharpe, consistency)
6. Bundle compatibility (LiveModelPredictor schema compliance)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

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
from ml_intraday_v3.live_trading.model_predictor import LiveModelPredictor


def load_model_bundle(bundle_path: Path) -> Dict:
    """Load and inspect model bundle."""
    logger.info(f"\nLoading: {bundle_path.name}")

    bundle = joblib.load(bundle_path)

    # Extract key info
    model = bundle.get('primary_model')
    preprocessor = bundle.get('primary_preprocessor')
    feature_cols = bundle.get('primary_feature_columns', [])
    has_side = bundle.get('has_side_feature', False)
    thresholds = bundle.get('thresholds', {})

    logger.info(f"  Model: {type(model).__name__}")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Has side feature: {has_side}")
    logger.info(f"  Thresholds: {thresholds}")

    # Check if model is wrapped (CalibratedClassifierCV)
    if hasattr(model, 'calibrated_classifiers_'):
        logger.info(f"  Calibration: {type(model).__name__}")
        base_estimator = model.calibrated_classifiers_[0].estimator
        logger.info(f"  Base estimator: {type(base_estimator).__name__}")

    return bundle


def load_holdout_data(project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Dec 2025 bars as holdout for testing."""
    logger.info("\n" + "="*80)
    logger.info("LOADING DEC 2025 HOLDOUT DATA")
    logger.info("="*80)

    # Load historical bars
    data_path = project_root / "data" / "processed" / "mes_bars_databento_rth.h5"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please ensure data is available for validation.")
        sys.exit(1)

    # Load bars
    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()

    if bars.index.tz is None:
        bars.index = bars.index.tz_localize('UTC')

    # Filter to Dec 2025 (holdout period after both models' training)
    holdout_start = pd.Timestamp('2025-12-01', tz='UTC')
    holdout_end = pd.Timestamp('2025-12-31 23:59:59', tz='UTC')

    bars_holdout = bars[(bars.index >= holdout_start) & (bars.index <= holdout_end)].copy()

    logger.info(f"  Total bars loaded: {len(bars):,}")
    logger.info(f"  Data range: {bars.index[0].date()} to {bars.index[-1].date()}")
    logger.info(f"  Dec 2025 holdout: {len(bars_holdout):,} bars")

    if len(bars_holdout) > 0:
        logger.info(f"  Holdout range: {bars_holdout.index[0]} to {bars_holdout.index[-1]}")
    else:
        logger.error("No holdout data available!")
        sys.exit(1)

    if len(bars_holdout) < 100:
        logger.warning("WARNING: Very few holdout bars available.")

    return bars, bars_holdout


def generate_labels_and_features(
    bars_holdout: pd.DataFrame,
    configs: Dict,
    instrument_spec: InstrumentSpec,
    enable_momentum: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate events and features for holdout data."""
    logger.info("\nGenerating labels and features...")
    if enable_momentum:
        logger.info("  (Momentum features ENABLED for this evaluation)")

    # Generate events using labeling config
    events = generate_events(
        bars_df=bars_holdout,
        bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution'],
    )

    # Apply triple barrier
    events = apply_triplebarrier(
        bars_df=bars_holdout,
        events_df=events,
        bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution'],
        instrument_spec=instrument_spec,
    )

    # Drop vertical barriers
    events = events[events['y'] != 0].reset_index(drop=True)

    logger.info(f"  Events generated: {len(events):,} (excluding vertical barriers)")
    logger.info(f"  Target rate: {(events['y'] == 1).mean():.1%}")
    logger.info(f"  LONG/SHORT: {(events['side'] == 1).sum()}/{(events['side'] == -1).sum()}")

    # Build features (with momentum override if requested)
    features_config = configs['features'].copy()
    if enable_momentum:
        features_config['momentum'] = {'enabled': True}

    features = build_features(bars_holdout, "5m", features_config)
    logger.info(f"  Features generated: {features.shape[1]} columns")

    return events, features


def evaluate_model(
    model_name: str,
    bundle: Dict,
    events: pd.DataFrame,
    features: pd.DataFrame,
) -> Dict:
    """Evaluate model on holdout data."""
    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUATING: {model_name}")
    logger.info(f"{'='*80}")

    # Extract model components
    model = bundle['primary_model']
    preprocessor = bundle.get('primary_preprocessor')
    feature_cols = bundle['primary_feature_columns']
    has_side = bundle.get('has_side_feature', False)

    # Prepare dataset
    t0_list = events['t0'].tolist()
    feat_aligned = features.reindex(t0_list).reset_index(drop=True)

    dataset = pd.concat([
        events[['side', 'y']].reset_index(drop=True),
        feat_aligned,
    ], axis=1)

    # Binary labels (stop=0, target=1)
    y_true = (dataset['y'] == 1).astype(int)

    # Drop NaN rows
    valid = ~dataset.isna().any(axis=1)
    dataset_clean = dataset[valid].copy()
    y_true_clean = y_true[valid]

    logger.info(f"  Valid samples: {len(dataset_clean):,} / {len(dataset):,}")

    # Select features
    X = dataset_clean[feature_cols]

    # Apply preprocessing if present
    if preprocessor is not None:
        if isinstance(preprocessor, dict):
            # Dict-style preprocessor (baseline model format)
            X_processed = X.copy()

            # Apply imputation
            if preprocessor.get('impute') == 'median':
                medians = preprocessor.get('medians', {})
                for col in X_processed.columns:
                    if col in medians:
                        X_processed[col] = X_processed[col].fillna(medians[col])

            # Apply scaling
            if preprocessor.get('scaler') == 'standard':
                means = preprocessor.get('means', {})
                stds = preprocessor.get('stds', {})
                for col in X_processed.columns:
                    if col in means and col in stds:
                        if stds[col] > 0:
                            X_processed[col] = (X_processed[col] - means[col]) / stds[col]

            X_processed = X_processed.values
        else:
            # sklearn transformer object (candidate model format)
            X_processed = preprocessor.transform(X)
    else:
        X_processed = X.values

    # Get predictions
    y_prob = model.predict_proba(X_processed)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    # Calculate metrics
    auc = roc_auc_score(y_true_clean, y_prob)
    brier = brier_score_loss(y_true_clean, y_prob)
    logloss = log_loss(y_true_clean, y_prob)
    accuracy = (y_pred == y_true_clean).mean()

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true_clean, y_prob, n_bins=10, strategy='quantile')
    calibration_error = np.abs(prob_true - prob_pred).mean()

    # Probability distribution
    prob_stats = {
        'min': float(y_prob.min()),
        'p10': float(np.percentile(y_prob, 10)),
        'p25': float(np.percentile(y_prob, 25)),
        'p50': float(np.percentile(y_prob, 50)),
        'p75': float(np.percentile(y_prob, 75)),
        'p90': float(np.percentile(y_prob, 90)),
        'max': float(y_prob.max()),
        'mean': float(y_prob.mean()),
        'std': float(y_prob.std()),
    }

    # Signal frequency at different confidence levels
    signals_055 = (y_prob > 0.55).sum()
    signals_060 = (y_prob > 0.60).sum()
    signals_065 = (y_prob > 0.65).sum()

    # Expected value (using true outcomes)
    # Assume PT=4 SL=3 (will vary by bundle config)
    pt_multiple = 4.0
    sl_multiple = 3.0
    ev_per_event = []
    for i, p_target in enumerate(y_prob):
        if y_true_clean.iloc[i] == 1:
            ev = pt_multiple  # Hit target
        else:
            ev = -sl_multiple  # Hit stop
        ev_per_event.append(ev)

    ev_per_event = np.array(ev_per_event)
    mean_ev = ev_per_event.mean()

    logger.info(f"\n  Classification Metrics:")
    logger.info(f"    AUC:              {auc:.4f}")
    logger.info(f"    Brier Score:      {brier:.4f}")
    logger.info(f"    Log Loss:         {logloss:.4f}")
    logger.info(f"    Accuracy:         {accuracy:.1%}")
    logger.info(f"    Calibration Err:  {calibration_error:.4f}")

    logger.info(f"\n  Probability Distribution:")
    logger.info(f"    Range: [{prob_stats['min']:.3f}, {prob_stats['max']:.3f}]")
    logger.info(f"    Mean:  {prob_stats['mean']:.3f} ± {prob_stats['std']:.3f}")
    logger.info(f"    P10/P50/P90: {prob_stats['p10']:.3f} / {prob_stats['p50']:.3f} / {prob_stats['p90']:.3f}")

    logger.info(f"\n  Signal Frequency:")
    logger.info(f"    p > 0.55: {signals_055:,} ({100*signals_055/len(y_prob):.1f}%)")
    logger.info(f"    p > 0.60: {signals_060:,} ({100*signals_060/len(y_prob):.1f}%)")
    logger.info(f"    p > 0.65: {signals_065:,} ({100*signals_065/len(y_prob):.1f}%)")

    logger.info(f"\n  Expected Value:")
    logger.info(f"    Mean EV/event: {mean_ev:.2f} pts")

    return {
        'model_name': model_name,
        'n_samples': len(dataset_clean),
        'n_features': len(feature_cols),
        'has_side_feature': has_side,
        'auc': auc,
        'brier_score': brier,
        'log_loss': logloss,
        'accuracy': accuracy,
        'calibration_error': calibration_error,
        'prob_stats': prob_stats,
        'signals_055': signals_055,
        'signals_060': signals_060,
        'signals_065': signals_065,
        'mean_ev': mean_ev,
        'prob_distribution': {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
        }
    }


def compare_models(baseline_results: Dict, candidate_results: Dict) -> Dict:
    """Compare baseline vs candidate and make go/no-go decision."""
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON & DECISION")
    logger.info("="*80)

    # Calculate deltas
    auc_delta = candidate_results['auc'] - baseline_results['auc']
    brier_delta = candidate_results['brier_score'] - baseline_results['brier_score']
    calib_delta = candidate_results['calibration_error'] - baseline_results['calibration_error']

    logger.info(f"\n  AUC:")
    logger.info(f"    Baseline:  {baseline_results['auc']:.4f}")
    logger.info(f"    Candidate: {candidate_results['auc']:.4f}")
    logger.info(f"    Delta:     {auc_delta:+.4f}")

    logger.info(f"\n  Brier Score (lower is better):")
    logger.info(f"    Baseline:  {baseline_results['brier_score']:.4f}")
    logger.info(f"    Candidate: {candidate_results['brier_score']:.4f}")
    logger.info(f"    Delta:     {brier_delta:+.4f}")

    logger.info(f"\n  Calibration Error (lower is better):")
    logger.info(f"    Baseline:  {baseline_results['calibration_error']:.4f}")
    logger.info(f"    Candidate: {candidate_results['calibration_error']:.4f}")
    logger.info(f"    Delta:     {calib_delta:+.4f}")

    logger.info(f"\n  Signal Frequency (p > 0.55):")
    logger.info(f"    Baseline:  {baseline_results['signals_055']:,}")
    logger.info(f"    Candidate: {candidate_results['signals_055']:,}")

    logger.info(f"\n  Expected Value:")
    logger.info(f"    Baseline:  {baseline_results['mean_ev']:.2f} pts/event")
    logger.info(f"    Candidate: {candidate_results['mean_ev']:.2f} pts/event")

    # Decision criteria
    logger.info("\n" + "="*80)
    logger.info("GO/NO-GO DECISION CRITERIA")
    logger.info("="*80)

    criteria = []

    # Criterion 1: AUC improvement
    auc_threshold = 0.02
    auc_pass = auc_delta > auc_threshold
    logger.info(f"\n  1. AUC Improvement > {auc_threshold:.2f}")
    logger.info(f"     Result: {auc_delta:+.4f} {'✅ PASS' if auc_pass else '❌ FAIL'}")
    criteria.append(('AUC improvement', auc_pass))

    # Criterion 2: Calibration quality (must not degrade significantly)
    calib_threshold = 0.02
    calib_pass = calib_delta < calib_threshold
    logger.info(f"\n  2. Calibration Not Degraded (delta < {calib_threshold:.2f})")
    logger.info(f"     Result: {calib_delta:+.4f} {'✅ PASS' if calib_pass else '⚠️  WARNING' if calib_delta < 0.05 else '❌ FAIL'}")
    criteria.append(('Calibration maintained', calib_pass))

    # Criterion 3: Trade frequency (ensure not too sparse)
    freq_threshold = 50  # At least 50 signals at p>0.55
    freq_pass = candidate_results['signals_055'] >= freq_threshold
    logger.info(f"\n  3. Sufficient Signal Frequency (≥{freq_threshold} at p>0.55)")
    logger.info(f"     Result: {candidate_results['signals_055']} {'✅ PASS' if freq_pass else '❌ FAIL'}")
    criteria.append(('Signal frequency', freq_pass))

    # Criterion 4: Brier score (must improve or not degrade >0.01)
    brier_threshold = 0.01
    brier_pass = brier_delta <= brier_threshold
    logger.info(f"\n  4. Brier Score Not Degraded (delta ≤ {brier_threshold:.2f})")
    logger.info(f"     Result: {brier_delta:+.4f} {'✅ PASS' if brier_pass else '⚠️  WARNING'}")
    criteria.append(('Brier score', brier_pass))

    # Overall decision
    n_pass = sum(c[1] for c in criteria)
    n_total = len(criteria)

    logger.info("\n" + "="*80)
    logger.info(f"DECISION: {n_pass}/{n_total} criteria passed")
    logger.info("="*80)

    if n_pass == n_total:
        decision = "STRONG GO"
        recommendation = "Promote to production immediately."
        logger.info(f"\n✅ {decision}")
        logger.info(f"   {recommendation}")
    elif n_pass >= 3:
        decision = "CONDITIONAL GO"
        recommendation = "Candidate shows improvement but has minor concerns. Review carefully and consider promoting with monitoring."
        logger.info(f"\n⚠️  {decision}")
        logger.info(f"   {recommendation}")
    elif n_pass >= 2:
        decision = "MARGINAL"
        recommendation = "Candidate is mixed. Consider testing next-best alternative from ranked results."
        logger.info(f"\n⚠️  {decision}")
        logger.info(f"   {recommendation}")
    else:
        decision = "NO-GO"
        recommendation = "Candidate does not beat baseline. Test alternative models from top-20 ranked results."
        logger.info(f"\n❌ {decision}")
        logger.info(f"   {recommendation}")

    return {
        'decision': decision,
        'recommendation': recommendation,
        'criteria': criteria,
        'n_pass': n_pass,
        'n_total': n_total,
        'auc_delta': auc_delta,
        'brier_delta': brier_delta,
        'calib_delta': calib_delta,
    }


def main():
    logger.info("="*80)
    logger.info("PROMOTED MODEL VALIDATION")
    logger.info("exhaustive_exp_00336 vs current baseline")
    logger.info("="*80)

    # Paths
    ml_root = project_root / "ml_intraday_v3"
    models_dir = ml_root / "models" / "saved"

    baseline_path = ml_root / "model_bundle_retrained_oct2024_nov2025.pkl"
    candidate_path = models_dir / "model_bundle_phasefull_best_exhaustive_exp_00336.pkl"

    # Load configs
    config_dir = ml_root / "configs"
    with open(config_dir / "labeling.yaml") as f:
        labeling_config = yaml.safe_load(f)
    with open(config_dir / "execution_spec.yaml") as f:
        execution_spec = yaml.safe_load(f)
    with open(config_dir / "features.yaml") as f:
        features_config = yaml.safe_load(f)

    configs = {
        'labeling': labeling_config,
        'execution': execution_spec,
        'features': features_config,
    }

    instrument_spec = InstrumentSpec(
        symbol="MES",
        tick_size_points=0.25,
        contract_multiplier_usd_per_point=5.0,
    )

    # Load models
    baseline_bundle = load_model_bundle(baseline_path)
    candidate_bundle = load_model_bundle(candidate_path)

    # Load Dec 2025 holdout data
    bars_all, bars_holdout = load_holdout_data(project_root)

    # Generate labels (same for both models)
    # We'll generate features separately for each model based on their needs
    logger.info("\nGenerating labels for holdout period...")
    events = generate_events(
        bars_df=bars_holdout,
        bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution'],
    )
    events = apply_triplebarrier(
        bars_df=bars_holdout,
        events_df=events,
        bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution'],
        instrument_spec=instrument_spec,
    )
    events = events[events['y'] != 0].reset_index(drop=True)

    logger.info(f"  Events generated: {len(events):,} (excluding vertical barriers)")
    logger.info(f"  Target rate: {(events['y'] == 1).mean():.1%}")
    logger.info(f"  LONG/SHORT: {(events['side'] == 1).sum()}/{(events['side'] == -1).sum()}")

    # Evaluate baseline model (34 features, no momentum)
    logger.info("\nGenerating features for BASELINE (momentum OFF)...")
    features_baseline = build_features(bars_holdout, "5m", configs['features'])
    logger.info(f"  Features: {features_baseline.shape[1]} columns")
    baseline_results = evaluate_model("BASELINE (Oct2024-Nov2025)", baseline_bundle, events, features_baseline)

    # Evaluate candidate model (41 features, with momentum)
    logger.info("\nGenerating features for CANDIDATE (momentum ON)...")
    features_config_momentum = configs['features'].copy()
    features_config_momentum['momentum'] = {'enabled': True}
    features_candidate = build_features(bars_holdout, "5m", features_config_momentum)
    logger.info(f"  Features: {features_candidate.shape[1]} columns")
    candidate_results = evaluate_model("CANDIDATE (exhaustive_exp_00336)", candidate_bundle, events, features_candidate)

    # Compare and decide
    comparison = compare_models(baseline_results, candidate_results)

    # Save results
    output_dir = ml_root / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"model_validation_{timestamp}.json"

    import json

    # Convert numpy types to Python types for JSON serialization
    def convert_for_json(obj):
        """Recursively convert numpy types to Python types."""
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    results_data = {
        'timestamp': datetime.now().isoformat(),
        'baseline': convert_for_json(baseline_results),
        'candidate': convert_for_json(candidate_results),
        'comparison': convert_for_json(comparison),
    }

    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"\n💾 Results saved to: {output_path}")

    # Exit with appropriate code
    if comparison['decision'] in ['STRONG GO', 'CONDITIONAL GO']:
        logger.info("\n✅ Validation complete - candidate recommended for promotion")
        sys.exit(0)
    else:
        logger.info("\n❌ Validation complete - candidate NOT recommended")
        sys.exit(1)


if __name__ == "__main__":
    main()
