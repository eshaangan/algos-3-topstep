"""
Run holdout test on reserved Oct-Dec 2025 data using the final trained model.
"""

import sys
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from ml_intraday_v3.backtesting_v3.simulator import run_backtest
from ml_intraday_v3.core.instrument import InstrumentSpec

# Import preprocessing utilities
import sys
import os
# Add parent to path for preprocessing imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_holdout_evaluation(run_dir: Path):
    """Run holdout test on Oct 2025+ data."""

    run_dir = Path(run_dir)
    bar_size = "1m"

    logger.info("="*80)
    logger.info("HOLDOUT TEST - Oct 2025+ (Unseen Data)")
    logger.info("="*80)

    # Load configs
    config_dir = Path("configs")
    with open(config_dir / "backtest.yaml") as f:
        backtest_cfg = yaml.safe_load(f)
    with open(config_dir / "risk.yaml") as f:
        risk_cfg = yaml.safe_load(f)
    with open(config_dir / "walkforward.yaml") as f:
        wf_cfg = yaml.safe_load(f)

    holdout_cfg = wf_cfg.get("holdout", {})
    if not holdout_cfg.get("enabled", False):
        logger.error("Holdout not enabled in walkforward.yaml")
        return None

    start_date = holdout_cfg.get("start_date")
    end_date = holdout_cfg.get("end_date")

    logger.info(f"Holdout period: {start_date} to {end_date or 'latest'}")

    # Load the final trained model (from last walk-forward window)
    wf_dir = run_dir / "walkforward" / f"bar_size={bar_size}"
    windows = sorted(wf_dir.glob("window_*"), key=lambda x: int(x.name.split("_")[1]))

    if not windows:
        logger.error("No walk-forward windows found")
        return None

    final_window = windows[-1]
    logger.info(f"Using final model from: {final_window.name}")

    # Load model bundle
    import joblib
    bundle_path = final_window / "model_bundle.pkl"
    if not bundle_path.exists():
        logger.error(f"Bundle not found: {bundle_path}")
        return None

    bundle = joblib.load(bundle_path)

    # Load bars, events, and features from main run
    bar_dir = run_dir / f"bar_size={bar_size}"
    bars_df = pd.read_parquet(bar_dir / "bars.parquet")
    events_df = pd.read_parquet(bar_dir / "events.parquet")
    features_df = pd.read_parquet(bar_dir / "features.parquet")

    # Filter to holdout period
    if start_date:
        start_dt = pd.to_datetime(start_date)
        # Make timezone-aware if needed
        if events_df['t0'].dt.tz is not None and start_dt.tz is None:
            start_dt = start_dt.tz_localize('UTC')
        events_df = events_df[events_df['t0'] >= start_dt].copy()

        if bars_df.index.tz is not None and start_dt.tz is None:
            start_dt = start_dt.tz_localize('UTC')
        bars_df = bars_df[bars_df.index >= start_dt].copy()

        if features_df.index.tz is not None and start_dt.tz is None:
            start_dt = start_dt.tz_localize('UTC')
        features_df = features_df[features_df.index >= start_dt].copy()

    if end_date:
        end_dt = pd.to_datetime(end_date)
        # Make timezone-aware if needed
        if events_df['t0'].dt.tz is not None and end_dt.tz is None:
            end_dt = end_dt.tz_localize('UTC')
        events_df = events_df[events_df['t0'] <= end_dt].copy()

        if bars_df.index.tz is not None and end_dt.tz is None:
            end_dt = end_dt.tz_localize('UTC')
        bars_df = bars_df[bars_df.index <= end_dt].copy()

        if features_df.index.tz is not None and end_dt.tz is None:
            end_dt = end_dt.tz_localize('UTC')
        features_df = features_df[features_df.index <= end_dt].copy()

    logger.info(f"Holdout events: {len(events_df)}")
    logger.info(f"Holdout bars: {len(bars_df)}")
    logger.info(f"Holdout features: {len(features_df)}")

    if len(events_df) == 0:
        logger.warning("No events in holdout period")
        return None

    # Generate predictions using the final model
    logger.info("Generating predictions on holdout data...")

    # Load preprocessor state from bundle
    preprocessor_state = bundle.get("primary_preprocessor")
    if preprocessor_state is None:
        logger.error("No primary_preprocessor in bundle")
        return None

    # Get feature columns from bundle
    feature_columns = bundle.get("primary_feature_columns")
    if feature_columns is None:
        logger.error("No feature columns in bundle")
        return None

    # Merge features with events on t0 timestamp
    # Reset features index to make timestamp a column for merging
    features_with_ts = features_df.reset_index()
    features_with_ts.columns = ['t0'] + list(features_df.columns)

    # Merge with events
    events_with_features = events_df.merge(features_with_ts, on='t0', how='left')

    # Prepare features (X must be in same order as events for predictions to match)
    X = events_with_features[feature_columns].values

    # Apply preprocessing manually from state dict
    impute = preprocessor_state.get('impute', 'median')
    scaler = preprocessor_state.get('scaler', 'standard')
    medians = np.array(preprocessor_state['medians'])
    means = np.array(preprocessor_state['means'])
    stds = np.array(preprocessor_state['stds'])

    # Impute missing values
    if impute == 'median':
        mask = np.isnan(X)
        X[mask] = np.take(medians, np.where(mask)[1])
    elif impute == 'zero':
        X = np.nan_to_num(X, 0.0)

    # Scale
    if scaler == 'standard':
        X_scaled = (X - means) / stds
    elif scaler == 'minmax':
        X_scaled = X  # MinMax not implemented in state
    else:
        X_scaled = X

    # Load model (stored in bundle)
    model = bundle.get("primary_model")
    if model is None:
        logger.error("Primary model not found in bundle")
        return None

    # Generate predictions
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_scaled)

        # For multiclass (3 outcomes: stop, target, vertical)
        if proba.shape[1] == 3:
            primary_preds_df = pd.DataFrame({
                'event_id': events_with_features['event_id'].values,
                'p_stop': proba[:, 0],
                'p_target': proba[:, 1],
                'p_vertical': proba[:, 2],
                'y_prob': proba[:, 1],  # Target probability
                'score_ev': proba[:, 1] - proba[:, 0],  # EV score
            })
        else:
            primary_preds_df = pd.DataFrame({
                'event_id': events_with_features['event_id'].values,
                'y_prob': proba[:, 1] if proba.shape[1] > 1 else proba[:, 0],
                'score_ev': proba[:, 1] if proba.shape[1] > 1 else proba[:, 0],
            })
    else:
        y_pred = model.predict(X_scaled)
        primary_preds_df = pd.DataFrame({
            'event_id': events_with_features['event_id'].values,
            'y_prob': y_pred,
            'score_ev': y_pred,
        })

    logger.info(f"Generated {len(primary_preds_df)} predictions")

    # Load execution spec and instrument
    exec_spec_path = Path("configs/execution_spec.yaml")
    with open(exec_spec_path) as f:
        execution_spec = yaml.safe_load(f)

    instrument_spec = InstrumentSpec.from_execution_spec(execution_spec)

    # Load label schema from file
    import json
    label_schema_path = bar_dir / "label_schema.json"
    with open(label_schema_path) as f:
        label_schema = json.load(f)

    # Run backtest
    logger.info("Running backtest on holdout data...")

    trades_df, equity_df, metrics = run_backtest(
        events_df=events_df,
        bars_df=bars_df,
        primary_preds_df=primary_preds_df,
        meta_preds_df=None,
        execution_spec=execution_spec,
        instrument_spec=instrument_spec,
        label_schema=label_schema,
        risk_cfg=risk_cfg,
        backtest_cfg=backtest_cfg,
        bar_size=bar_size,
    )

    # Display results
    logger.info("")
    logger.info("="*80)
    logger.info("HOLDOUT TEST RESULTS")
    logger.info("="*80)
    logger.info(f"Period: {start_date} to {end_date or 'latest'}")
    logger.info(f"Total PnL: ${metrics.get('total_pnl_usd', 0):,.2f}")
    logger.info(f"Trades: {metrics.get('trades_count', 0)}")
    logger.info(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
    logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    logger.info(f"Avg Trade: ${metrics.get('avg_trade_usd', 0):,.2f}")
    logger.info(f"Max Drawdown: ${metrics.get('max_drawdown_usd', 0):,.2f}")
    logger.info("")

    # Check for position limit rejections
    position_limit_rejections = (trades_df['reason_skipped'] == 'max_concurrent_positions').sum()
    logger.info(f"Trades rejected (max_concurrent_positions): {position_limit_rejections}")

    # Save results
    holdout_results = {
        'period': f"{start_date} to {end_date or 'latest'}",
        'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                   for k, v in metrics.items()},
        'position_limit_rejections': int(position_limit_rejections),
    }

    results_path = run_dir / "holdout_results.json"
    with open(results_path, 'w') as f:
        json.dump(holdout_results, f, indent=2)

    logger.info(f"Results saved to: {results_path}")

    # Save trades
    trades_path = run_dir / "holdout_trades.parquet"
    trades_df.to_parquet(trades_path)
    logger.info(f"Trades saved to: {trades_path}")

    return holdout_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_holdout_test.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    run_holdout_evaluation(run_dir)
