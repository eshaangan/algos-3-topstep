#!/usr/bin/env python3
"""
Jan 2026 Validation with Threshold Sweep

Fetch true out-of-sample Jan 2026 data and test candidate model
with various confidence thresholds to find optimal operating point.

Tests:
1. Fetch Jan 2026 MES data from Databento
2. Generate labels and features
3. Evaluate candidate vs baseline at multiple thresholds
4. Calculate expected value, Sharpe, win rate at each threshold
5. Recommend optimal threshold for production
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml

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
from ml_intraday_v3.labels.events import generate_events
from ml_intraday_v3.labels.triple_barrier import apply_triplebarrier
from ml_intraday_v3.live_trading.data_fetcher import LiveDataFetcher


def fetch_jan_2026_data() -> pd.DataFrame:
    """Fetch Jan 2026 MES data from Databento."""
    logger.info("="*80)
    logger.info("FETCHING JAN 2026 DATA FROM DATABENTO")
    logger.info("="*80)

    try:
        # Create fetcher for MES front month continuous contract
        fetcher = LiveDataFetcher(
            symbol="MES.c.0",  # MES front month, calendar roll
            bar_size="1m",
            lookback_bars=100,
        )

        # Fetch January 2026 data
        start_date = "2026-01-01"
        end_date = "2026-01-31"

        logger.info(f"Fetching: {start_date} to {end_date}")

        bars_1m = fetcher.fetch_historical(start_date, end_date)
        logger.info(f"Fetched {len(bars_1m):,} 1-minute bars")

        if len(bars_1m) == 0:
            logger.error("No data fetched - check date range or Databento subscription")
            sys.exit(1)

        # Resample to 5-minute bars
        logger.info("Resampling to 5-minute bars...")
        bars_5m = bars_1m.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        logger.info(f"Resampled to {len(bars_5m):,} 5-minute bars")
        logger.info(f"Date range: {bars_5m.index[0]} to {bars_5m.index[-1]}")

        return bars_5m

    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        logger.error("Please check:")
        logger.error("  1. DATABENTO_API_KEY is set in .env")
        logger.error("  2. Your Databento subscription includes MES")
        raise


def generate_labels_features(
    bars: pd.DataFrame,
    configs: Dict,
    instrument_spec: InstrumentSpec,
    enable_momentum: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate events and features for evaluation."""
    logger.info("\nGenerating labels and features...")
    if enable_momentum:
        logger.info("  (Momentum features ENABLED)")

    # Generate events
    events = generate_events(
        bars_df=bars,
        bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution'],
    )

    # Apply triple barrier
    events = apply_triplebarrier(
        bars_df=bars,
        events_df=events,
        bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution'],
        instrument_spec=instrument_spec,
    )

    # Drop vertical barriers
    events = events[events['y'] != 0].reset_index(drop=True)

    logger.info(f"  Events: {len(events):,} (no vertical barriers)")
    logger.info(f"  Target rate: {(events['y'] == 1).mean():.1%}")
    logger.info(f"  LONG/SHORT: {(events['side'] == 1).sum()}/{(events['side'] == -1).sum()}")

    # Build features
    features_config = configs['features'].copy()
    if enable_momentum:
        features_config['momentum'] = {'enabled': True}

    features = build_features(bars, "5m", features_config)
    logger.info(f"  Features: {features.shape[1]} columns")

    return events, features


def evaluate_at_threshold(
    model_name: str,
    bundle: Dict,
    events: pd.DataFrame,
    features: pd.DataFrame,
    threshold: float,
    pt_multiple: float = 4.0,
    sl_multiple: float = 3.0
) -> Dict:
    """Evaluate model at specific confidence threshold."""

    # Extract model components
    model = bundle['primary_model']
    preprocessor = bundle.get('primary_preprocessor')
    feature_cols = bundle['primary_feature_columns']
    has_side = bundle.get('has_side_feature', False)

    # Prepare dataset
    t0_list = events['t0'].tolist()
    feat_aligned = features.reindex(t0_list).reset_index(drop=True)

    dataset = pd.concat([
        events[['side', 'y', 'ret_net']].reset_index(drop=True),
        feat_aligned,
    ], axis=1)

    # Binary labels (stop=0, target=1)
    y_true = (dataset['y'] == 1).astype(int)

    # Drop NaN rows
    valid = ~dataset.isna().any(axis=1)
    dataset_clean = dataset[valid].copy()
    y_true_clean = y_true[valid]

    # Select features
    X = dataset_clean[feature_cols]

    # Apply preprocessing
    if preprocessor is not None:
        if isinstance(preprocessor, dict):
            # Dict-style preprocessor
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
            # sklearn transformer object
            X_processed = preprocessor.transform(X)
    else:
        X_processed = X.values

    # Get predictions
    y_prob = model.predict_proba(X_processed)[:, 1]

    # Filter by threshold
    high_conf_mask = y_prob > threshold
    n_signals = high_conf_mask.sum()

    if n_signals == 0:
        return {
            'model_name': model_name,
            'threshold': threshold,
            'n_signals': 0,
            'n_total': len(y_prob),
            'signal_rate': 0.0,
            'win_rate': np.nan,
            'avg_winner': np.nan,
            'avg_loser': np.nan,
            'mean_pnl': np.nan,
            'total_pnl': np.nan,
            'sharpe': np.nan,
            'max_dd': np.nan,
            'profit_factor': np.nan,
        }

    # Get returns for high-confidence signals
    y_true_filtered = y_true_clean[high_conf_mask]
    ret_net_filtered = dataset_clean['ret_net'].values[high_conf_mask]

    # Calculate trading metrics
    wins = ret_net_filtered > 0
    losses = ret_net_filtered < 0

    n_wins = wins.sum()
    n_losses = losses.sum()

    win_rate = n_wins / n_signals if n_signals > 0 else 0.0
    avg_winner = ret_net_filtered[wins].mean() if n_wins > 0 else 0.0
    avg_loser = ret_net_filtered[losses].mean() if n_losses > 0 else 0.0

    total_pnl = ret_net_filtered.sum()
    mean_pnl = ret_net_filtered.mean()

    # Sharpe ratio (annualized, assuming 252 trading days)
    if len(ret_net_filtered) > 1 and ret_net_filtered.std() > 0:
        sharpe = (mean_pnl / ret_net_filtered.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    cumulative = np.cumsum(ret_net_filtered)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = drawdown.min() if len(drawdown) > 0 else 0.0

    # Profit factor
    total_wins = ret_net_filtered[wins].sum() if n_wins > 0 else 0.0
    total_losses = abs(ret_net_filtered[losses].sum()) if n_losses > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else np.inf

    return {
        'model_name': model_name,
        'threshold': threshold,
        'n_signals': int(n_signals),
        'n_total': len(y_prob),
        'signal_rate': n_signals / len(y_prob),
        'win_rate': win_rate,
        'n_wins': int(n_wins),
        'n_losses': int(n_losses),
        'avg_winner': avg_winner,
        'avg_loser': avg_loser,
        'mean_pnl': mean_pnl,
        'total_pnl': total_pnl,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'profit_factor': profit_factor,
    }


def main():
    logger.info("="*80)
    logger.info("JAN 2026 VALIDATION - THRESHOLD SWEEP")
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

    # Fetch Jan 2026 data
    bars_jan = fetch_jan_2026_data()

    # Load models
    logger.info("\nLoading models...")
    baseline_bundle = joblib.load(baseline_path)
    logger.info(f"  Baseline: {baseline_path.name}")

    candidate_bundle = joblib.load(candidate_path)
    logger.info(f"  Candidate: {candidate_path.name}")

    # Generate labels (same for both)
    logger.info("\nGenerating labels...")
    events = generate_events(
        bars_df=bars_jan,
        bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution'],
    )
    events = apply_triplebarrier(
        bars_df=bars_jan,
        events_df=events,
        bar_size="5m",
        labeling_config=configs['labeling'],
        execution_spec=configs['execution'],
        instrument_spec=instrument_spec,
    )
    events = events[events['y'] != 0].reset_index(drop=True)

    logger.info(f"  Events: {len(events):,}")

    # Generate features for each model
    logger.info("\nGenerating features for BASELINE (no momentum)...")
    features_baseline = build_features(bars_jan, "5m", configs['features'])
    logger.info(f"  Features: {features_baseline.shape[1]}")

    logger.info("\nGenerating features for CANDIDATE (with momentum)...")
    features_config_momentum = configs['features'].copy()
    features_config_momentum['momentum'] = {'enabled': True}
    features_candidate = build_features(bars_jan, "5m", features_config_momentum)
    logger.info(f"  Features: {features_candidate.shape[1]}")

    # Test thresholds
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    logger.info("\n" + "="*80)
    logger.info("THRESHOLD SWEEP")
    logger.info("="*80)

    results = []

    for threshold in thresholds:
        logger.info(f"\n--- Threshold: {threshold:.2f} ---")

        # Baseline
        baseline_result = evaluate_at_threshold(
            "BASELINE",
            baseline_bundle,
            events,
            features_baseline,
            threshold
        )
        logger.info(f"  BASELINE: {baseline_result['n_signals']} signals, "
                   f"Win rate: {baseline_result['win_rate']:.1%}, "
                   f"Mean PnL: ${baseline_result['mean_pnl']:.2f}")
        results.append(baseline_result)

        # Candidate
        candidate_result = evaluate_at_threshold(
            "CANDIDATE",
            candidate_bundle,
            events,
            features_candidate,
            threshold
        )
        logger.info(f"  CANDIDATE: {candidate_result['n_signals']} signals, "
                   f"Win rate: {candidate_result['win_rate']:.1%}, "
                   f"Mean PnL: ${candidate_result['mean_pnl']:.2f}")
        results.append(candidate_result)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Summary table
    logger.info("\n" + "="*80)
    logger.info("SUMMARY TABLE")
    logger.info("="*80)

    for threshold in thresholds:
        baseline_row = results_df[(results_df['model_name'] == 'BASELINE') & (results_df['threshold'] == threshold)].iloc[0]
        candidate_row = results_df[(results_df['model_name'] == 'CANDIDATE') & (results_df['threshold'] == threshold)].iloc[0]

        logger.info(f"\n=== Threshold: {threshold:.2f} ===")
        logger.info(f"  BASELINE:")
        logger.info(f"    Signals: {baseline_row['n_signals']} ({100*baseline_row['signal_rate']:.1f}%)")
        if baseline_row['n_signals'] > 0:
            logger.info(f"    Win Rate: {100*baseline_row['win_rate']:.1f}% ({baseline_row['n_wins']}W / {baseline_row['n_losses']}L)")
            logger.info(f"    Avg Winner: ${baseline_row['avg_winner']:.2f}, Avg Loser: ${baseline_row['avg_loser']:.2f}")
            logger.info(f"    Total PnL: ${baseline_row['total_pnl']:.2f}, Mean: ${baseline_row['mean_pnl']:.2f}")
            logger.info(f"    Sharpe: {baseline_row['sharpe']:.2f}, Max DD: ${baseline_row['max_dd']:.2f}")
            logger.info(f"    Profit Factor: {baseline_row['profit_factor']:.2f}")

        logger.info(f"  CANDIDATE:")
        logger.info(f"    Signals: {candidate_row['n_signals']} ({100*candidate_row['signal_rate']:.1f}%)")
        if candidate_row['n_signals'] > 0:
            logger.info(f"    Win Rate: {100*candidate_row['win_rate']:.1f}% ({candidate_row['n_wins']}W / {candidate_row['n_losses']}L)")
            logger.info(f"    Avg Winner: ${candidate_row['avg_winner']:.2f}, Avg Loser: ${candidate_row['avg_loser']:.2f}")
            logger.info(f"    Total PnL: ${candidate_row['total_pnl']:.2f}, Mean: ${candidate_row['mean_pnl']:.2f}")
            logger.info(f"    Sharpe: {candidate_row['sharpe']:.2f}, Max DD: ${candidate_row['max_dd']:.2f}")
            logger.info(f"    Profit Factor: {candidate_row['profit_factor']:.2f}")

    # Find optimal threshold for candidate
    logger.info("\n" + "="*80)
    logger.info("OPTIMAL THRESHOLD ANALYSIS")
    logger.info("="*80)

    candidate_results = results_df[results_df['model_name'] == 'CANDIDATE'].copy()
    candidate_results = candidate_results[candidate_results['n_signals'] > 0]  # Only consider thresholds with signals

    if len(candidate_results) > 0:
        # Find threshold with best Sharpe ratio
        best_sharpe_idx = candidate_results['sharpe'].idxmax()
        best_sharpe_row = candidate_results.loc[best_sharpe_idx]

        # Find threshold with best total PnL
        best_pnl_idx = candidate_results['total_pnl'].idxmax()
        best_pnl_row = candidate_results.loc[best_pnl_idx]

        # Find threshold with best profit factor
        best_pf_idx = candidate_results['profit_factor'].idxmax()
        best_pf_row = candidate_results.loc[best_pf_idx]

        logger.info(f"\nBest by Sharpe Ratio: Threshold {best_sharpe_row['threshold']:.2f}")
        logger.info(f"  Sharpe: {best_sharpe_row['sharpe']:.2f}")
        logger.info(f"  Total PnL: ${best_sharpe_row['total_pnl']:.2f}")
        logger.info(f"  Signals: {best_sharpe_row['n_signals']}")

        logger.info(f"\nBest by Total PnL: Threshold {best_pnl_row['threshold']:.2f}")
        logger.info(f"  Total PnL: ${best_pnl_row['total_pnl']:.2f}")
        logger.info(f"  Sharpe: {best_pnl_row['sharpe']:.2f}")
        logger.info(f"  Signals: {best_pnl_row['n_signals']}")

        logger.info(f"\nBest by Profit Factor: Threshold {best_pf_row['threshold']:.2f}")
        logger.info(f"  Profit Factor: {best_pf_row['profit_factor']:.2f}")
        logger.info(f"  Total PnL: ${best_pf_row['total_pnl']:.2f}")
        logger.info(f"  Signals: {best_pf_row['n_signals']}")

        # Recommendation
        logger.info("\n" + "="*80)
        logger.info("RECOMMENDATION")
        logger.info("="*80)

        if best_pnl_row['total_pnl'] > 0 and best_pnl_row['sharpe'] > 0.5:
            logger.info(f"✅ CANDIDATE MODEL VIABLE at threshold {best_pnl_row['threshold']:.2f}")
            logger.info(f"   Expected: ${best_pnl_row['total_pnl']:.2f} total PnL on {best_pnl_row['n_signals']} signals")
            logger.info(f"   Sharpe: {best_pnl_row['sharpe']:.2f}, Win Rate: {100*best_pnl_row['win_rate']:.1f}%")
            logger.info(f"\n   Suggested live_trading.yaml config:")
            logger.info(f"   primary_threshold: {best_pnl_row['threshold']:.2f}")
            logger.info(f"   min_probability_distance: {best_pnl_row['threshold']:.2f}  # in execution_spec.yaml")
        else:
            logger.info("❌ CANDIDATE MODEL NOT VIABLE")
            logger.info("   No threshold produces positive expected value with acceptable risk-adjusted returns.")
            logger.info("   Consider testing alternative models from top-20 ranked results.")

    else:
        logger.info("❌ NO VIABLE THRESHOLDS")
        logger.info("   Candidate model produces no signals at any tested threshold.")

    # Save results
    output_dir = ml_root / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"threshold_sweep_jan2026_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)

    logger.info(f"\n💾 Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
