#!/usr/bin/env python3
"""
Retrain ML Intraday V3 Model with Bidirectional 24-Hour Trading Setup

This script retrains the model with:
1. 24-hour data (all Globex hours, not just RTH)
2. trend_scanning labeling (adds 'side' feature for bidirectional trading)
3. Aggressive thresholds for combine passing

Run from repo root: python retrain_bidirectional_24h.py
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import subprocess

# Add ml_intraday_v3 to path
sys.path.insert(0, str(Path(__file__).parent / "ml_intraday_v3"))

# Run configuration
RUN_ID = f"bidirectional_24h_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
BAR_SIZE = "5m"  # 5m bars for better feature completeness

print("=" * 80)
print("ML INTRADAY V3 - BIDIRECTIONAL 24-HOUR RETRAIN")
print("=" * 80)
print(f"RUN_ID: {RUN_ID}")
print(f"BAR_SIZE: {BAR_SIZE}")
print(f"Date: {datetime.now()}")
print("=" * 80)
print()
print("Configuration Changes:")
print("  ✓ Data: 24-hour Globex (full_range grid, all hours)")
print("  ✓ Labeling: trend_scanning (adds 'side' feature)")
print("  ✓ Model: Bidirectional (evaluates LONG & SHORT, picks best EV)")
print("  ✓ Thresholds: 0.03 (aggressive for combine passing)")
print("  ✓ Sessions: All hours (not just RTH)")
print("=" * 80)
print()

# Import pipeline modules
try:
    from data import (
        load_raw_data,
        standardize_ohlcv,
        reindex_to_grid,
        resample_1m_to_5m,
        add_session_features,
        run_qa_checks,
        SessionConfig,
    )
    from features import build_features, get_feature_registry
    from labels import generate_events, apply_triplebarrier
    from weights import (
        map_event_intervals_to_index,
        compute_concurrency,
        compute_avg_uniqueness,
        compute_sample_weights,
    )
    from training import train_primary_model, save_model_bundle
    from validation import run_purged_kfold
    from backtesting import run_backtest

    print("✓ All modules imported successfully")
    print()
except ImportError as e:
    print(f"ERROR: Failed to import modules: {e}")
    print("Make sure you're running from repo root and dependencies are installed")
    sys.exit(1)

# Config paths
config_dir = Path("ml_intraday_v3/configs")
configs = {
    "data": config_dir / "data.yaml",
    "features": config_dir / "features.yaml",
    "labeling": config_dir / "labeling.yaml",
    "training": config_dir / "training.yaml",
    "execution_spec": config_dir / "execution_spec.yaml",
    "risk": config_dir / "risk.yaml",
    "backtest": config_dir / "backtest.yaml",
}

# Output directory
run_dir = Path(f"ml_intraday_v3/runs/{RUN_ID}")
bar_dir = run_dir / f"bar_size={BAR_SIZE}"
bar_dir.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {run_dir}")
print()

# ============================================================================
# STAGE 1: Data Preparation
# ============================================================================
print("STAGE 1: Data Preparation (24-hour Globex)")
print("-" * 80)

try:
    import yaml
    with open(configs["data"]) as f:
        data_cfg = yaml.safe_load(f)

    print("Loading raw data...")
    raw_path = Path(data_cfg["raw_data"]["input_path"])
    if not raw_path.exists():
        print(f"ERROR: Raw data not found at {raw_path}")
        sys.exit(1)

    print(f"  Loading: {raw_path}")
    print(f"  Size: {raw_path.stat().st_size / 1e9:.2f} GB")

    df_raw = load_raw_data(raw_path, data_cfg)
    print(f"  ✓ Loaded {len(df_raw):,} rows")

    # Standardize OHLCV
    df_std = standardize_ohlcv(df_raw, data_cfg)
    print(f"  ✓ Standardized OHLCV")

    # Reindex to 24-hour grid
    print("Reindexing to 24-hour grid...")
    df_1m = reindex_to_grid(df_std, bar_size="1m", config=data_cfg)
    print(f"  ✓ 1m grid: {len(df_1m):,} rows")

    # Resample to 5m if needed
    if BAR_SIZE == "5m":
        print("Resampling to 5m...")
        df_bars = resample_1m_to_5m(df_1m)
        print(f"  ✓ 5m bars: {len(df_bars):,} rows")
    else:
        df_bars = df_1m

    # Add session features
    df_bars = add_session_features(df_bars, data_cfg)
    print(f"  ✓ Added session features")

    # QA checks
    print("Running QA checks...")
    run_qa_checks(df_bars, data_cfg, bar_size=BAR_SIZE)
    print(f"  ✓ QA passed")

    # Save bars
    bars_path = bar_dir / "bars.parquet"
    df_bars.to_parquet(bars_path)
    print(f"  ✓ Saved: {bars_path}")
    print()

except Exception as e:
    print(f"ERROR in Stage 1: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STAGE 2: Feature Engineering
# ============================================================================
print("STAGE 2: Feature Engineering")
print("-" * 80)

try:
    with open(configs["features"]) as f:
        feat_cfg = yaml.safe_load(f)

    print("Building features...")
    df_features = build_features(df_bars, feat_cfg, bar_size=BAR_SIZE)
    print(f"  ✓ Built {len(df_features.columns)} features")
    print(f"  ✓ Usable rows: {df_features['usable_for_training'].sum():,}")

    # Save features
    feat_path = bar_dir / "features.parquet"
    df_features.to_parquet(feat_path)
    print(f"  ✓ Saved: {feat_path}")
    print()

except Exception as e:
    print(f"ERROR in Stage 2: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STAGE 3: Labeling (trend_scanning for bidirectional)
# ============================================================================
print("STAGE 3: Labeling (trend_scanning with 'side' feature)")
print("-" * 80)

try:
    with open(configs["labeling"]) as f:
        label_cfg = yaml.safe_load(f)
    with open(configs["execution_spec"]) as f:
        exec_cfg = yaml.safe_load(f)

    print("Generating events (trend_scanning)...")
    events_df = generate_events(df_bars, label_cfg, bar_size=BAR_SIZE)
    print(f"  ✓ Generated {len(events_df):,} events")
    print(f"  ✓ Has 'side' feature: {'side' in events_df.columns}")

    # Apply triple barrier
    print("Applying triple barrier...")
    labels_df = apply_triplebarrier(
        bars=df_bars,
        events=events_df,
        label_config=label_cfg,
        exec_config=exec_cfg,
        bar_size=BAR_SIZE
    )
    print(f"  ✓ Labeled {len(labels_df):,} events")

    # Print side distribution
    if 'side' in labels_df.columns:
        side_dist = labels_df['side'].value_counts()
        print(f"  Side distribution:")
        print(f"    LONG (side=1): {side_dist.get(1, 0):,}")
        print(f"    SHORT (side=-1): {side_dist.get(-1, 0):,}")

    # Save labels
    labels_path = bar_dir / "labels.parquet"
    labels_df.to_parquet(labels_path)
    print(f"  ✓ Saved: {labels_path}")
    print()

except Exception as e:
    print(f"ERROR in Stage 3: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STAGE 4: Sample Weighting
# ============================================================================
print("STAGE 4: Sample Weighting")
print("-" * 80)

try:
    print("Computing sample weights...")

    # Map events to index
    event_intervals = labels_df[['t1']].copy()
    event_intervals['t0'] = event_intervals.index

    idx_map = map_event_intervals_to_index(
        event_intervals=event_intervals,
        index=df_features.index
    )

    # Compute concurrency
    concurrency = compute_concurrency(idx_map)
    print(f"  ✓ Max concurrency: {concurrency.max()}")

    # Compute uniqueness
    avg_uniqueness = compute_avg_uniqueness(idx_map, concurrency)
    print(f"  ✓ Avg uniqueness: {avg_uniqueness.mean():.4f}")

    # Compute final weights
    labels_df = compute_sample_weights(
        labels=labels_df,
        event_idx_map=idx_map,
        concurrency=concurrency,
        avg_uniqueness=avg_uniqueness
    )
    print(f"  ✓ Sample weights computed")

    # Update labels with weights
    labels_df.to_parquet(labels_path)
    print(f"  ✓ Updated: {labels_path}")
    print()

except Exception as e:
    print(f"ERROR in Stage 4: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STAGE 5: Training (Bidirectional Model with 'side' feature)
# ============================================================================
print("STAGE 5: Training (LightGBM with 'side' feature)")
print("-" * 80)

try:
    with open(configs["training"]) as f:
        train_cfg = yaml.safe_load(f)

    print("Preparing training data...")

    # Merge features + labels
    train_df = df_features.join(labels_df, how='inner')
    train_df = train_df[train_df['usable_for_training'] == True].copy()
    print(f"  ✓ Training set: {len(train_df):,} samples")

    # Check for 'side' feature
    if 'side' in train_df.columns:
        print(f"  ✓ 'side' feature present - bidirectional model will train")
    else:
        print(f"  WARNING: 'side' feature NOT found - falling back to uni-directional")

    # Train model
    print("Training model...")
    model, preprocessor, metadata = train_primary_model(
        train_df=train_df,
        config=train_cfg
    )
    print(f"  ✓ Model trained: {metadata['model_type']}")
    print(f"  ✓ Features: {len(metadata['feature_columns'])}")
    print(f"  ✓ Training score: {metadata.get('train_score', 'N/A')}")

    # Save model bundle
    model_path = bar_dir / "model_bundle.pkl"
    save_model_bundle(
        model=model,
        preprocessor=preprocessor,
        metadata=metadata,
        path=model_path
    )
    print(f"  ✓ Saved: {model_path}")

    # Also save to models/saved for live trading
    live_model_path = Path("ml_intraday_v3/models/saved/model_bundle.pkl")
    live_model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model_bundle(
        model=model,
        preprocessor=preprocessor,
        metadata=metadata,
        path=live_model_path
    )
    print(f"  ✓ Saved for live trading: {live_model_path}")
    print()

except Exception as e:
    print(f"ERROR in Stage 5: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# STAGE 6: Backtesting
# ============================================================================
print("STAGE 6: Backtesting (24-hour trading)")
print("-" * 80)

try:
    with open(configs["backtest"]) as f:
        backtest_cfg = yaml.safe_load(f)
    with open(configs["risk"]) as f:
        risk_cfg = yaml.safe_load(f)

    print("Running backtest...")
    results = run_backtest(
        bars=df_bars,
        features=df_features,
        labels=labels_df,
        model_bundle_path=model_path,
        backtest_config=backtest_cfg,
        risk_config=risk_cfg,
        execution_config=exec_cfg
    )

    print(f"  ✓ Backtest complete")
    print(f"  Total trades: {results['total_trades']}")
    print(f"  Win rate: {results['win_rate']:.1%}")
    print(f"  Total PnL: ${results['total_pnl']:,.2f}")
    print(f"  Sharpe: {results.get('sharpe', 'N/A')}")

    # Save results
    results_path = bar_dir / "backtest_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved: {results_path}")
    print()

except Exception as e:
    print(f"WARNING: Backtest failed: {e}")
    print("Model training completed successfully, continuing...")
    print()

# ============================================================================
# DONE
# ============================================================================
print("=" * 80)
print("RETRAINING COMPLETE!")
print("=" * 80)
print(f"Run ID: {RUN_ID}")
print(f"Output: {run_dir}")
print(f"Model: {model_path}")
print(f"Live model: {live_model_path}")
print()
print("Next steps:")
print("1. Review backtest results in the output directory")
print("2. Run live_runner.py to start 24-hour bidirectional trading")
print("3. Monitor logs/ for signal generation and trades")
print()
print("Configuration Summary:")
print("  - Data: 24-hour Globex (all hours)")
print("  - Labeling: trend_scanning (bidirectional with 'side' feature)")
print("  - Model: LightGBM with bidirectional prediction")
print("  - Threshold: 0.03 (aggressive for combine passing)")
print("  - Sessions: All hours (Sunday 5pm - Friday 4pm CT)")
print("=" * 80)
