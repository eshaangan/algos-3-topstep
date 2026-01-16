#!/usr/bin/env python3
"""
FIXED: Add 'side' feature and retrain bidirectional model.

The key fix: Set events index to 't0' BEFORE merging with features.
"""

import sys
from pathlib import Path
import pandas as pd
import pickle
import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Add ml_intraday_v3 to path
sys.path.insert(0, str(Path.cwd()))

print("=" * 80)
print("RETRAIN MODEL WITH 'side' FEATURE (FIXED)")
print("=" * 80)
print()

# Configuration
RUN_ID = "bidirectional_24h_20260114"
BAR_SIZE = "5m"

run_dir = Path(f"runs/{RUN_ID}/bar_size={BAR_SIZE}")

print(f"Run directory: {run_dir}")
print()

# Load features
print("Loading features...")
features_path = run_dir / "features.parquet"
df_features = pd.read_parquet(features_path)
print(f"  ✓ Features shape: {df_features.shape}")
print(f"  ✓ Features index type: {type(df_features.index)}")
print(f"  ✓ Feature columns: {len(df_features.columns)}")
print()

# Load events
print("Loading events...")
events_path = run_dir / "events.parquet"
df_events = pd.read_parquet(events_path)
print(f"  ✓ Events shape: {df_events.shape}")
print(f"  ✓ Has 'side' column: {'side' in df_events.columns}")

side_dist = df_events['side'].value_counts()
print(f"  Side distribution:")
print(f"    LONG (side=1): {side_dist.get(1, 0):,}")
print(f"    SHORT (side=-1): {side_dist.get(-1, 0):,}")
print()

# KEY FIX: Set events index to 't0' before joining
print("Setting events index to 't0' for proper alignment...")
df_events_indexed = df_events.set_index('t0')
print(f"  ✓ Events indexed by t0")
print(f"  ✓ Events index type: {type(df_events_indexed.index)}")

# Verify alignment
t0_in_features = df_events['t0'].isin(df_features.index).sum()
print(f"  ✓ Event t0 timestamps in features: {t0_in_features:,} / {len(df_events):,}")
print()

# Merge 'side' from events into features
print("Merging 'side' into features...")
df_features_with_side = df_features.join(
    df_events_indexed[['side', 'y', 't1']],
    how='left'
)
print(f"  ✓ Merged shape: {df_features_with_side.shape}")

n_with_side = df_features_with_side['side'].notna().sum()
print(f"  ✓ Rows with 'side': {n_with_side:,} / {len(df_features_with_side):,}")
print()

# Prepare training data
print("Preparing training data...")

# Filter to rows with 'side' (event timestamps only)
df_train = df_features_with_side[df_features_with_side['side'].notna()].copy()
print(f"  ✓ Training samples (with side): {len(df_train):,}")

# Filter usable_for_training
if 'usable_for_training' in df_train.columns:
    df_train = df_train[df_train['usable_for_training'] == True].copy()
    print(f"  ✓ After usable_for_training filter: {len(df_train):,}")

# Load weights if available
weights_path = run_dir / "weights.parquet"
if weights_path.exists():
    df_weights = pd.read_parquet(weights_path)
    # Set weights index to match
    if 't0' in df_weights.columns:
        df_weights = df_weights.set_index('t0')

    df_train = df_train.join(df_weights[['w_final']], how='left')
    sample_weight = df_train['w_final'].fillna(1.0).values
    print(f"  ✓ Using sample weights")
else:
    sample_weight = None
    print(f"  No sample weights found")

print()

# Prepare X and y
print("Preparing feature matrix...")

# Feature columns (exclude metadata)
exclude_cols = ['y', 't1', 'is_synthetic', 'usable_for_training', 'w_final']
feature_cols = [c for c in df_train.columns if c not in exclude_cols]

print(f"  ✓ Total features: {len(feature_cols)}")
print(f"  ✓ Features: {feature_cols}")

# CRITICAL CHECK: 'side' must be in features
assert 'side' in feature_cols, "ERROR: 'side' not in feature columns!"
print()
print("  ✓✓✓ 'side' CONFIRMED in feature set!")
print()

X = df_train[feature_cols].values
y = df_train['y'].values

print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

y_dist = dict(zip(*np.unique(y, return_counts=True)))
print(f"  y distribution: {y_dist}")
print()

# Check we have data
if len(X) == 0:
    print("❌ ERROR: No training samples!")
    print("   This shouldn't happen if alignment check passed.")
    sys.exit(1)

# Train model
print("Training LightGBM model with 'side' feature...")

# Load training config
with open("ml_intraday_v3/configs/training.yaml") as f:
    train_cfg = yaml.safe_load(f)

# Preprocessing
print("  Preprocessing...")
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
print("  ✓ Imputation and scaling complete")

# Train
print("  Training LightGBM...")
from lightgbm import LGBMClassifier

model_params = train_cfg['model']['params']
model = LGBMClassifier(**model_params, random_state=42, verbose=-1)

model.fit(
    X_scaled,
    y,
    sample_weight=sample_weight if sample_weight is not None else None
)
print("  ✓ Training complete")
print()

# Evaluate
train_score = model.score(X_scaled, y)
print(f"  Training accuracy: {train_score:.4f}")
print()

# Save model bundle
print("Saving bidirectional model...")

bundle = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'feature_columns': feature_cols,
    'model_type': 'LGBMClassifier',
    'train_score': train_score,
    'n_features': len(feature_cols),
    'has_side_feature': True,
}

# Save to run directory
model_path = run_dir / "model_bundle_bidirectional.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(bundle, f)
print(f"  ✓ Saved: {model_path}")

# Save to live trading location
live_model_path = Path("ml_intraday_v3/models/saved/model_bundle.pkl")
live_model_path.parent.mkdir(parents=True, exist_ok=True)
with open(live_model_path, 'wb') as f:
    pickle.dump(bundle, f)
print(f"  ✓ Saved for live trading: {live_model_path}")
print()

# Verify
print("Verifying bidirectional model...")
test_bundle = pickle.load(open(live_model_path, 'rb'))
print(f"  ✓ Model type: {test_bundle['model_type']}")
print(f"  ✓ Features: {test_bundle['n_features']}")
print(f"  ✓ Has 'side': {test_bundle['has_side_feature']}")
print(f"  ✓ 'side' in feature_columns: {'side' in test_bundle['feature_columns']}")

if 'side' in test_bundle['feature_columns']:
    side_idx = test_bundle['feature_columns'].index('side')
    print(f"  ✓ 'side' at index: {side_idx}")
print()

print("=" * 80)
print("✓✓✓ BIDIRECTIONAL MODEL TRAINED AND READY ✓✓✓")
print("=" * 80)
print()
print("Next steps:")
print("1. Verify model:")
print("   python check_model.py")
print()
print("2. Start paper trading:")
print("   cd ml_intraday_v3/live_trading")
print("   python live_runner.py --paper --no-confirm")
print()
print("Model saved to:")
print(f"  - {model_path}")
print(f"  - {live_model_path}")
print("=" * 80)
