#!/usr/bin/env python3
"""
Batch Validation of Top 100 Models on Jan 2026

Tests the top 100 models from exhaustive search on true out-of-sample
Jan 2026 data to find which models actually perform well in practice.

Strategy:
1. Load top 100 model configs from GCS results
2. For each model:
   - Download result JSON with model config
   - Retrain model with config on full training set
   - Test on Jan 2026 at key thresholds (0.35, 0.40, 0.45)
3. Rank by Jan 2026 performance metrics:
   - Total PnL
   - Sharpe ratio
   - Profit factor
4. Report top 10 performers

Optimization:
- Cache Jan 2026 features (reuse across models)
- Test only 3 thresholds per model (not full sweep)
- Save results incrementally
- Skip models that fail to train
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
from google.cloud import storage
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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

from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from ml_intraday_v3.core.instrument import InstrumentSpec
from ml_intraday_v3.features.build import build_features
from ml_intraday_v3.labels.events import generate_events, balance_events
from ml_intraday_v3.labels.triple_barrier import apply_triplebarrier
from ml_intraday_v3.live_trading.data_fetcher import LiveDataFetcher


def download_top_results(top_n: int = 100) -> pd.DataFrame:
    """Download and parse top N results from GCS."""
    logger.info(f"\nDownloading top {top_n} model configs from GCS...")

    # Check local cache first
    cache_dir = Path("/tmp/phasefull_results")
    ranked_csv = cache_dir / "phasefull_ranked_top20.csv"

    if not ranked_csv.exists():
        logger.error(f"Ranked results not found: {ranked_csv}")
        logger.info("Please ensure you've run the ranking script first.")
        sys.exit(1)

    # Load ranked results
    ranked_df = pd.read_csv(ranked_csv)
    logger.info(f"  Loaded {len(ranked_df)} ranked results")

    # If we need more than 20, download full results
    if top_n > 20:
        logger.info(f"  Need top {top_n}, downloading full results from GCS...")

        # Download all result JSONs
        storage_client = storage.Client()
        bucket = storage_client.bucket("trading-algo-3")
        blobs = bucket.list_blobs(prefix="experiment-results/phasefull/result_exhaustive_exp_")

        all_results = []
        for blob in blobs:
            if blob.name.endswith('.json'):
                content = blob.download_as_text()
                result = json.loads(content)
                all_results.append(result)

        logger.info(f"  Downloaded {len(all_results)} results")

        # Create full DataFrame
        full_df = pd.DataFrame(all_results)

        # Rank by composite score
        if 'composite_score' in full_df.columns:
            full_df = full_df.sort_values('composite_score', ascending=False)
        elif 'median_test_auc' in full_df.columns:
            full_df = full_df.sort_values('median_test_auc', ascending=False)

        # Save extended cache
        extended_cache = cache_dir / f"phasefull_ranked_top{top_n}.csv"
        full_df.head(top_n).to_csv(extended_cache, index=False)
        logger.info(f"  Saved extended cache: {extended_cache}")

        return full_df.head(top_n)

    return ranked_df.head(top_n)


def fetch_jan_2026_data() -> pd.DataFrame:
    """Fetch Jan 2026 data (cached if already fetched)."""
    cache_file = project_root / "data" / "processed" / "jan_2026_5m_bars.parquet"

    if cache_file.exists():
        logger.info("\nLoading cached Jan 2026 data...")
        bars = pd.read_parquet(cache_file)
        logger.info(f"  Loaded {len(bars):,} bars from cache")
        return bars

    logger.info("\nFetching Jan 2026 data from Databento...")

    fetcher = LiveDataFetcher(
        symbol="MES.c.0",
        bar_size="1m",
        lookback_bars=100,
    )

    bars_1m = fetcher.fetch_historical("2026-01-01", "2026-01-31")
    logger.info(f"  Fetched {len(bars_1m):,} 1-minute bars")

    # Resample to 5m
    bars_5m = bars_1m.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    logger.info(f"  Resampled to {len(bars_5m):,} 5-minute bars")

    # Cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    bars_5m.to_parquet(cache_file)
    logger.info(f"  Cached to {cache_file}")

    return bars_5m


def prepare_jan2026_labels_features(
    bars: pd.DataFrame,
    configs: Dict,
    instrument_spec: InstrumentSpec
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Prepare Jan 2026 labels and feature sets (with and without momentum)."""
    logger.info("\nPreparing Jan 2026 labels and features...")

    # Generate events (same for all models)
    events = generate_events(
        bars_df=bars,
        bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution'],
    )

    events = apply_triplebarrier(
        bars_df=bars,
        events_df=events,
        bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution'],
        instrument_spec=instrument_spec,
    )

    events = events[events['y'] != 0].reset_index(drop=True)
    logger.info(f"  Events: {len(events):,}")

    # Generate feature sets
    feature_sets = {}

    # Base features (no momentum)
    logger.info("  Generating base features (no momentum)...")
    features_base = build_features(bars, "5m", configs['features'])
    feature_sets['base'] = features_base
    logger.info(f"    Base: {features_base.shape[1]} features")

    # With momentum
    logger.info("  Generating momentum features...")
    features_config_momentum = configs['features'].copy()
    features_config_momentum['momentum'] = {'enabled': True}
    features_momentum = build_features(bars, "5m", features_config_momentum)
    feature_sets['momentum'] = features_momentum
    logger.info(f"    Momentum: {features_momentum.shape[1]} features")

    return events, feature_sets


def train_model_from_config(
    config: Dict,
    training_data_path: Path,
    configs: Dict
) -> Optional[Dict]:
    """Train a model from experiment config."""

    # For this batch test, we'll use a simplified training approach
    # Load pre-existing training data and train the model

    # This is a placeholder - in reality you'd need to:
    # 1. Load full training bars
    # 2. Generate events with config's labeling params
    # 3. Build features with config's feature set
    # 4. Train model with config's model params

    # For now, skip retraining and load from saved bundles if available
    logger.warning("  Training from config not implemented - using saved bundles")
    return None


def evaluate_model_on_jan2026(
    model_config: Dict,
    bundle: Optional[Dict],
    events: pd.DataFrame,
    feature_sets: Dict[str, pd.DataFrame],
    thresholds: List[float] = [0.35, 0.40, 0.45]
) -> Dict:
    """Evaluate model on Jan 2026 at specified thresholds."""

    if bundle is None:
        return {
            'exp_id': model_config.get('exp_id'),
            'status': 'failed',
            'reason': 'bundle_not_found'
        }

    results = {
        'exp_id': model_config.get('exp_id'),
        'model_name': model_config.get('model_name'),
        'status': 'success',
    }

    # Determine feature set
    feature_set_name = model_config.get('feature_set_name', 'base')
    has_momentum = 'momentum' in feature_set_name or model_config.get('features_config', {}).get('momentum', {}).get('enabled', False)
    features = feature_sets['momentum'] if has_momentum else feature_sets['base']

    # Extract model components
    model = bundle['primary_model']
    preprocessor = bundle.get('primary_preprocessor')
    feature_cols = bundle['primary_feature_columns']

    # Prepare dataset
    t0_list = events['t0'].tolist()
    feat_aligned = features.reindex(t0_list).reset_index(drop=True)

    dataset = pd.concat([
        events[['side', 'y', 'ret_net']].reset_index(drop=True),
        feat_aligned,
    ], axis=1)

    y_true = (dataset['y'] == 1).astype(int)
    valid = ~dataset.isna().any(axis=1)
    dataset_clean = dataset[valid].copy()
    y_true_clean = y_true[valid]

    # Check if features match
    missing_features = set(feature_cols) - set(dataset_clean.columns)
    if missing_features:
        logger.warning(f"    Missing features: {missing_features}")
        return {
            'exp_id': model_config.get('exp_id'),
            'status': 'failed',
            'reason': f'missing_features: {list(missing_features)[:3]}'
        }

    X = dataset_clean[feature_cols]

    # Apply preprocessing
    if preprocessor is not None:
        if isinstance(preprocessor, dict):
            X_processed = X.copy()
            if preprocessor.get('impute') == 'median':
                medians = preprocessor.get('medians', {})
                for col in X_processed.columns:
                    if col in medians:
                        X_processed[col] = X_processed[col].fillna(medians[col])
            if preprocessor.get('scaler') == 'standard':
                means = preprocessor.get('means', {})
                stds = preprocessor.get('stds', {})
                for col in X_processed.columns:
                    if col in means and col in stds:
                        if stds[col] > 0:
                            X_processed[col] = (X_processed[col] - means[col]) / stds[col]
            X_processed = X_processed.values
        else:
            X_processed = preprocessor.transform(X)
    else:
        X_processed = X.values

    # Get predictions
    try:
        y_prob = model.predict_proba(X_processed)[:, 1]
    except Exception as e:
        logger.error(f"    Prediction failed: {e}")
        return {
            'exp_id': model_config.get('exp_id'),
            'status': 'failed',
            'reason': f'prediction_error: {str(e)[:50]}'
        }

    # Evaluate at each threshold
    for threshold in thresholds:
        high_conf_mask = y_prob > threshold
        n_signals = high_conf_mask.sum()

        if n_signals == 0:
            results[f'pnl_{threshold:.2f}'] = np.nan
            results[f'sharpe_{threshold:.2f}'] = np.nan
            results[f'signals_{threshold:.2f}'] = 0
            results[f'win_rate_{threshold:.2f}'] = np.nan
            continue

        ret_net_filtered = dataset_clean['ret_net'].values[high_conf_mask]

        wins = ret_net_filtered > 0
        n_wins = wins.sum()
        n_losses = (ret_net_filtered < 0).sum()

        total_pnl = ret_net_filtered.sum()
        mean_pnl = ret_net_filtered.mean()

        win_rate = n_wins / n_signals if n_signals > 0 else 0.0

        if len(ret_net_filtered) > 1 and ret_net_filtered.std() > 0:
            sharpe = (mean_pnl / ret_net_filtered.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        results[f'pnl_{threshold:.2f}'] = total_pnl
        results[f'sharpe_{threshold:.2f}'] = sharpe
        results[f'signals_{threshold:.2f}'] = int(n_signals)
        results[f'win_rate_{threshold:.2f}'] = win_rate

    return results


def main():
    logger.info("="*80)
    logger.info("BATCH VALIDATION: TOP 100 MODELS ON JAN 2026")
    logger.info("="*80)

    # Load configs
    ml_root = project_root / "ml_intraday_v3"
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

    # Download top 100 configs
    # For now, use the top 20 we already have
    logger.info("\nNOTE: Using top 20 from cached results (top 100 download requires GCS access)")
    logger.info("To test full top 100, first download all results from GCS")

    cache_dir = Path("/tmp/phasefull_results")
    ranked_csv = cache_dir / "phasefull_ranked_top20.csv"

    if not ranked_csv.exists():
        logger.error(f"Ranked results not found: {ranked_csv}")
        logger.info("Please run the exhaustive search ranking script first.")
        sys.exit(1)

    top_configs = pd.read_csv(ranked_csv)
    logger.info(f"Loaded {len(top_configs)} model configs")

    # Fetch Jan 2026 data
    bars_jan = fetch_jan_2026_data()

    # Prepare labels and features
    events, feature_sets = prepare_jan2026_labels_features(bars_jan, configs, instrument_spec)

    # Test each model
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING {len(top_configs)} MODELS")
    logger.info(f"{'='*80}")

    results = []
    thresholds = [0.35, 0.40, 0.45]

    models_dir = ml_root / "models" / "saved"

    for idx, row in top_configs.iterrows():
        exp_id = row['exp_id']
        logger.info(f"\n[{idx+1}/{len(top_configs)}] Testing {exp_id}...")
        logger.info(f"  CV AUC: {row.get('median_test_auc', 'N/A')}")

        # Check if bundle exists
        bundle_path = models_dir / f"model_bundle_phasefull_best_{exp_id}.pkl"

        if not bundle_path.exists():
            logger.warning(f"  Bundle not found: {bundle_path.name}")
            logger.info(f"  Skipping (need to promote this model first)")
            results.append({
                'exp_id': exp_id,
                'status': 'not_promoted',
                'cv_auc': row.get('median_test_auc'),
            })
            continue

        # Load bundle
        try:
            bundle = joblib.load(bundle_path)
            logger.info(f"  Loaded bundle")
        except Exception as e:
            logger.error(f"  Failed to load bundle: {e}")
            results.append({
                'exp_id': exp_id,
                'status': 'load_failed',
                'cv_auc': row.get('median_test_auc'),
            })
            continue

        # Evaluate
        result = evaluate_model_on_jan2026(
            row.to_dict(),
            bundle,
            events,
            feature_sets,
            thresholds
        )

        # Add CV metrics for comparison
        result['cv_auc'] = row.get('median_test_auc')
        result['cv_composite'] = row.get('composite_score')

        results.append(result)

        # Log summary
        if result['status'] == 'success':
            logger.info(f"  Jan 2026 @ 0.40:")
            logger.info(f"    PnL: ${result.get('pnl_0.40', 'N/A'):.2f}")
            logger.info(f"    Signals: {result.get('signals_0.40', 0)}")
            logger.info(f"    Sharpe: {result.get('sharpe_0.40', 'N/A'):.2f}")
        else:
            logger.info(f"  Status: {result['status']} - {result.get('reason', 'unknown')}")

        # Save incremental results
        if (idx + 1) % 5 == 0:
            temp_results = pd.DataFrame(results)
            temp_path = ml_root / "diagnostics" / f"batch_validation_temp_{idx+1}.csv"
            temp_results.to_csv(temp_path, index=False)
            logger.info(f"  Saved checkpoint: {temp_path.name}")

    # Final results
    results_df = pd.DataFrame(results)

    # Rank by Jan 2026 performance
    logger.info("\n" + "="*80)
    logger.info("RANKING BY JAN 2026 PERFORMANCE")
    logger.info("="*80)

    # Filter successful models
    successful = results_df[results_df['status'] == 'success'].copy()

    if len(successful) == 0:
        logger.error("No models succeeded! All need to be promoted first.")
        logger.info("\nTo promote top models, run:")
        logger.info("  cd ml_intraday_v3/experiments")
        logger.info("  python promote_top_exhaustive_model.py --exp_id exhaustive_exp_XXXXX")
        sys.exit(1)

    # Rank by PnL at threshold 0.40
    successful['rank_pnl'] = successful['pnl_0.40'].rank(ascending=False)
    successful['rank_sharpe'] = successful['sharpe_0.40'].rank(ascending=False)
    successful['composite_rank'] = (successful['rank_pnl'] + successful['rank_sharpe']) / 2

    successful = successful.sort_values('composite_rank')

    # Display top 10
    logger.info("\nTOP 10 MODELS BY JAN 2026 PERFORMANCE:")
    logger.info(f"{'Rank':<6} {'Exp ID':<25} {'PnL@0.40':<12} {'Sharpe':<10} {'Signals':<10} {'CV AUC':<10}")
    logger.info("-" * 80)

    for rank, (idx, row) in enumerate(successful.head(10).iterrows(), 1):
        logger.info(
            f"{rank:<6} "
            f"{row['exp_id']:<25} "
            f"${row['pnl_0.40']:>10.2f} "
            f"{row['sharpe_0.40']:>9.2f} "
            f"{row['signals_0.40']:>9} "
            f"{row['cv_auc']:>9.4f}"
        )

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = ml_root / "diagnostics" / f"batch_validation_top100_jan2026_{timestamp}.csv"
    results_df.to_csv(output_path, index=False)

    logger.info(f"\n💾 Full results saved to: {output_path}")

    # Check if any model is profitable
    profitable = successful[successful['pnl_0.40'] > 0]

    if len(profitable) > 0:
        logger.info(f"\n✅ FOUND {len(profitable)} PROFITABLE MODELS!")
        best_model = profitable.iloc[0]
        logger.info(f"\nBest model: {best_model['exp_id']}")
        logger.info(f"  Jan 2026 PnL: ${best_model['pnl_0.40']:.2f}")
        logger.info(f"  Sharpe: {best_model['sharpe_0.40']:.2f}")
        logger.info(f"  Signals: {best_model['signals_0.40']}")
        logger.info(f"\nRECOMMENDATION: Promote this model to production")
    else:
        logger.info(f"\n❌ NO PROFITABLE MODELS FOUND")
        logger.info(f"  All {len(successful)} tested models lose money on Jan 2026")
        logger.info(f"  Consider:")
        logger.info(f"    1. Testing more models from top 100")
        logger.info(f"    2. Retraining with updated data including Jan 2026")
        logger.info(f"    3. Returning to rule-based system")


if __name__ == "__main__":
    main()
