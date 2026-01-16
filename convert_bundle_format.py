#!/usr/bin/env python3
"""
Convert model bundle to format expected by LiveModelPredictor.
"""

import pickle
from pathlib import Path

print("=" * 80)
print("CONVERT MODEL BUNDLE TO LIVE TRADING FORMAT")
print("=" * 80)
print()

# Load current bundle
model_path = Path("ml_intraday_v3/models/saved/model_bundle.pkl")
print(f"Loading: {model_path}")
bundle = pickle.load(open(model_path, 'rb'))

print(f"Current keys: {list(bundle.keys())}")
print()

# Extract components
model = bundle['model']
imputer = bundle['imputer']
scaler = bundle['scaler']
feature_columns = bundle['feature_columns']

print(f"Model type: {type(model).__name__}")
print(f"Features: {len(feature_columns)}")
print(f"Has 'side': {'side' in feature_columns}")
print()

# Create preprocessor state (compatible with LiveModelPredictor)
preprocessor_state = {
    'impute': 'median',
    'scaler': 'standard',
    'medians': imputer.statistics_,  # medians from SimpleImputer
    'means': scaler.mean_,  # means from StandardScaler
    'stds': scaler.scale_,  # stds from StandardScaler
}

print("Preprocessor state created:")
print(f"  - Imputer strategy: median")
print(f"  - Scaler: standard")
print(f"  - Medians shape: {preprocessor_state['medians'].shape}")
print(f"  - Means shape: {preprocessor_state['means'].shape}")
print(f"  - Stds shape: {preprocessor_state['stds'].shape}")
print()

# Create new bundle in expected format
new_bundle = {
    'primary_model': model,
    'primary_preprocessor': preprocessor_state,
    'primary_feature_columns': feature_columns,
    'thresholds': {
        'primary_threshold': 0.03,  # From live_trading.yaml
    },
    'meta_model': None,
    'meta_preprocessor': None,
    'meta_feature_columns': None,
    # Keep metadata for reference
    'model_type': bundle.get('model_type'),
    'train_score': bundle.get('train_score'),
    'n_features': bundle.get('n_features'),
    'has_side_feature': bundle.get('has_side_feature'),
}

print("New bundle format:")
print(f"  Keys: {list(new_bundle.keys())}")
print()

# Save converted bundle
print("Saving converted bundle...")
with open(model_path, 'wb') as f:
    pickle.dump(new_bundle, f)
print(f"  ✓ Saved: {model_path}")
print()

# Verify
print("Verifying...")
test_bundle = pickle.load(open(model_path, 'rb'))
print(f"  ✓ primary_model: {type(test_bundle['primary_model']).__name__}")
print(f"  ✓ primary_feature_columns: {len(test_bundle['primary_feature_columns'])} features")
print(f"  ✓ Has 'side': {'side' in test_bundle['primary_feature_columns']}")
print(f"  ✓ primary_threshold: {test_bundle['thresholds']['primary_threshold']}")
print()

print("=" * 80)
print("✓ BUNDLE CONVERTED TO LIVE TRADING FORMAT")
print("=" * 80)
print()
print("Next steps:")
print("1. Test prediction:")
print("   cd ml_intraday_v3/live_trading")
print("   python test_predictor.py  # (if exists)")
print()
print("2. Start paper trading:")
print("   python live_runner.py --paper --no-confirm")
print("=" * 80)
