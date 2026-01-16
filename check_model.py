#!/usr/bin/env python3
"""
Check if the trained model is bidirectional (has 'side' feature).
"""

import pickle
from pathlib import Path

model_path = Path("ml_intraday_v3/models/saved/model_bundle.pkl")

print("=" * 80)
print("MODEL VERIFICATION")
print("=" * 80)
print()

if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    print()
    print("Run training first!")
    exit(1)

print(f"Loading model from: {model_path}")
bundle = pickle.load(open(model_path, 'rb'))

print()
print("Model Information:")
print(f"  Model type: {bundle.get('model_type', 'Unknown')}")
# Try both old and new format
features = bundle.get('primary_feature_columns') or bundle.get('feature_columns', [])
print(f"  Total features: {len(features)}")
print(f"  Training score: {bundle.get('train_score', 'N/A')}")
print()

has_side = 'side' in features

print("Bidirectional Check:")
print(f"  Has 'side' feature: {has_side}")

if has_side:
    side_idx = features.index('side')
    print(f"  Side feature index: {side_idx}")
    print()
    print("✓✓✓ MODEL IS BIDIRECTIONAL ✓✓✓")
    print()
    print("The model will:")
    print("  - Evaluate both LONG (side=1) and SHORT (side=-1)")
    print("  - Choose the direction with higher expected value")
    print("  - Skip trades where both sides have negative EV")
else:
    print()
    print("❌ MODEL IS NOT BIDIRECTIONAL")
    print()
    print("The model will only trade in one direction.")
    print("Run 'python add_side_feature_and_retrain.py' to fix this.")

print()
print("Features (first 10):")
for i, feat in enumerate(features[:10]):
    print(f"  {i}: {feat}")

print()
print("Features (last 10):")
for i, feat in enumerate(features[-10:], start=len(features)-10):
    print(f"  {i}: {feat}")

print()
print("=" * 80)
