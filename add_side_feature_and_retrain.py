#!/usr/bin/env python3
"""
Quick fix: Add 'side' from labels to features and retrain model.

This script:
1. Loads existing features and labels
2. Merges 'side' column from labels into features
3. Retrains the model with 'side' as a feature
4. Saves the bidirectional model for live trading
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
print("QUICK FIX: Add 'side' Feature and Retrain Bidirectional Model")
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
print(f"  ✓ Loaded {len(df_features):,} rows, {len(df_features.columns)} columns")

# Load labels/events
print("Loading events/labels...")
events_path = run_dir / "events.parquet"
df_events = pd.read_parquet(events_path)
print(f"  ✓ Loaded {len(df_events):,} events")
print(f"  ✓ Has 'side' column: {'side' in df_events.columns}")

if 'side' not in df_events.columns:
    print("ERROR: Events don't have 'side' column!")
    sys.exit(1)

side_dist = df_events['side'].value_counts()
print(f"  Side distribution:")
print(f"    LONG (side=1): {side_dist.get(1, 0):,}")
print(f"    SHORT (side=-1): {side_dist.get(-1, 0):,}")
print()

# Merge 'side' from events into features at event timestamps
print("Merging 'side' into features...")
# Events are indexed by t0 (event start time)
# We need to add 'side' to features at those timestamps
df_side = df_events[['side']].copy()
df_side.index.name = 'timestamp'

# Merge on index (timestamp)
df_features_with_side = df_features.join(df_side, how='left')

# Check merge
n_with_side = df_features_with_side['side'].notna().sum()
print(f"  ✓ Merged: {n_with_side:,} / {len(df_features):,} bars have 'side' values")

# Save updated features
features_with_side_path = run_dir / "features_with_side.parquet"
df_features_with_side.to_parquet(features_with_side_path)
print(f"  ✓ Saved: {features_with_side_path}")
print()

# Prepare training data
print("Preparing training data...")

# Merge with labels to get y and weights
# Load labels (if different from events)
labels_files = list(run_dir.glob("labels*.parquet"))
if labels_files:
    df_labels = pd.read_parquet(labels_files[0])
    print(f"  ✓ Loaded labels: {len(df_labels):,} rows")
else:
    # Use events as labels
    df_labels = df_events.copy()

# Merge features + labels
df_train = df_features_with_side.join(df_labels[['y', 't1']], how='inner')

# Filter to usable rows
if 'usable_for_training' in df_train.columns:
    df_train = df_train[df_train['usable_for_training'] == True].copy()
    print(f"  ✓ Usable for training: {len(df_train):,} samples")

# Filter to rows with 'side' (event timestamps)
df_train = df_train[df_train['side'].notna()].copy()
print(f"  ✓ With 'side' feature: {len(df_train):,} samples")

# Load weights if available
weights_path = run_dir / "weights.parquet"
if weights_path.exists():
    df_weights = pd.read_parquet(weights_path)
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
print(f"  ✓ Using {len(feature_cols)} features (including 'side')")
print(f"  Features: {feature_cols}")
print()

# Check 'side' is in features
assert 'side' in feature_cols, "ERROR: 'side' not in feature columns!"
print("  ✓✓✓ 'side' confirmed in feature set!")
print()

X = df_train[feature_cols].values
y = df_train['y'].values

print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  y distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print()

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
print(f"  ✓ side in feature_columns: {'side' in test_bundle['feature_columns']}")

side_idx = test_bundle['feature_columns'].index('side')
print(f"  ✓ side at index: {side_idx}")
print()

print("=" * 80)
print("✓✓✓ BIDIRECTIONAL MODEL TRAINED AND READY ✓✓✓")
print("=" * 80)
print()
print("Next steps:")
print("1. Test prediction:")
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
