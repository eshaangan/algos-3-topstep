#!/usr/bin/env python3
"""
Comprehensive setup verification for live trading.
Checks model, code, and config alignment.
"""

import pickle
import yaml
from pathlib import Path

print("="*80)
print("LIVE TRADING SETUP VERIFICATION")
print("="*80)

# 1. Check which model will be auto-loaded
print("\n1. MODEL AUTO-DETECTION")
print("-"*80)

runs_dir = Path("runs")
bar_size = "5m"

# Simulate find_latest_model logic
run_dirs = sorted(runs_dir.glob("*"), key=lambda x: x.name, reverse=True)
model_path = None

for run_dir in run_dirs:
    wf_dir = run_dir / "walkforward" / f"bar_size={bar_size}"
    if not wf_dir.exists():
        continue
    
    windows = sorted(wf_dir.glob("window_*"), key=lambda x: int(x.name.split("_")[1]))
    if not windows:
        continue
    
    final_window = windows[-1]
    bundle_path = final_window / "model_bundle.pkl"
    
    if bundle_path.exists():
        model_path = bundle_path
        print(f"✓ Auto-detected model: {model_path}")
        print(f"  Run: {run_dir.name}")
        print(f"  Window: {final_window.name} (latest of {len(windows)} windows)")
        break

if model_path:
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    
    model = bundle.get('primary_model')
    features = bundle.get('primary_feature_columns', [])
    
    print(f"\n  Model details:")
    print(f"    Type: {type(model).__name__}")
    print(f"    Features: {len(features)}")
    print(f"    Has 'side': {'side' in features}")
    
    if 'side' in features:
        print(f"    'side' position: {features.index('side')}")
    
    if hasattr(model, 'n_classes_'):
        print(f"    Classes: {model.n_classes_}")
    
    is_bidirectional = 'side' in features
    print(f"\n  → This is a {'BIDIRECTIONAL' if is_bidirectional else 'LONG-ONLY'} model")
else:
    print("✗ No model found!")
    is_bidirectional = False

# 2. Check code
print("\n2. CODE COMPATIBILITY")
print("-"*80)

with open('ml_intraday_v3/live_trading/model_predictor.py') as f:
    predictor_code = f.read()

with open('ml_intraday_v3/live_trading/live_runner.py') as f:
    runner_code = f.read()

has_bidir_predictor = 'X_long' in predictor_code and 'X_short' in predictor_code
has_bidir_runner = "'side' in prediction" in runner_code

print(f"model_predictor.py has bidirectional logic: {has_bidir_predictor}")
print(f"live_runner.py has bidirectional logic: {has_bidir_runner}")

code_is_bidirectional = has_bidir_predictor and has_bidir_runner
print(f"\n  → Code is {'BIDIRECTIONAL' if code_is_bidirectional else 'LONG-ONLY'}")

# 3. Check config
print("\n3. CONFIGURATION")
print("-"*80)

with open('ml_intraday_v3/configs/live_trading.yaml') as f:
    live_cfg = yaml.safe_load(f)

with open('ml_intraday_v3/configs/backtest.yaml') as f:
    backtest_cfg = yaml.safe_load(f)

bar_size_cfg = live_cfg['trading']['bar_size']
threshold = live_cfg['signals']['primary_threshold']
use_meta = live_cfg['signals']['use_meta_model']
enable_shorts = backtest_cfg['decision']['regime_filter'].get('enable_shorts', False)

print(f"bar_size: {bar_size_cfg}")
print(f"primary_threshold: {threshold}")
print(f"use_meta_model: {use_meta}")
print(f"regime_filter.enable_shorts: {enable_shorts}")

# 4. Compatibility check
print("\n4. COMPATIBILITY ANALYSIS")
print("-"*80)

if is_bidirectional and code_is_bidirectional and not enable_shorts:
    print("✓✓✓ PERFECT SETUP ✓✓✓")
    print("  • Model: BIDIRECTIONAL (with 'side' feature)")
    print("  • Code: BIDIRECTIONAL (evaluates both LONG and SHORT)")
    print("  • Config: enable_shorts=false (uses model's sides)")
    print("\n  Expected behavior:")
    print("    - Every bar: Model evaluates BOTH directions")
    print("    - Picks side with better expected value")
    print("    - 30-40+ trades per day")
    print("    - Mix of LONG and SHORT trades")

elif not is_bidirectional and not code_is_bidirectional:
    print("✓ LONG-ONLY SETUP (like Jan 12)")
    print("  • Model: LONG-ONLY (no 'side' feature)")
    print("  • Code: LONG-ONLY (simple prediction)")
    print("\n  Expected behavior:")
    print("    - Every bar: Simple prediction")
    print("    - Only LONG trades")
    print("    - 27 trades per day (like Jan 12)")

elif is_bidirectional and not code_is_bidirectional:
    print("⚠️  WARNING: PARTIAL SETUP")
    print("  • Model: BIDIRECTIONAL")
    print("  • Code: LONG-ONLY")
    print("  • This will cause incorrect predictions!")
    print("\n  Fix: Add bidirectional evaluation logic to code")

elif not is_bidirectional and code_is_bidirectional:
    print("⚠️  WARNING: CODE/MODEL MISMATCH")
    print("  • Model: LONG-ONLY")
    print("  • Code: BIDIRECTIONAL")
    print("  • Code expects 'side' feature that doesn't exist")
    print("\n  Fix: Either use bidirectional model or remove bidirectional code")

# 5. Summary
print("\n5. READY TO TRADE?")
print("-"*80)

if is_bidirectional and code_is_bidirectional and not enable_shorts:
    print("✅ YES - Bidirectional trading ready")
    print("\nStart command:")
    print('  cd ml_intraday_v3')
    print('  PYTHONPATH="." python3 live_trading/live_runner.py')
    print("\nWatch for:")
    print('  "Bidirectional choice: LONG/SHORT (EV_long=..., EV_short=...)"')

elif not is_bidirectional and not code_is_bidirectional:
    print("✅ YES - LONG-only trading ready (Jan 12 setup)")
    print("\nStart command:")
    print('  cd ml_intraday_v3')
    print('  PYTHONPATH="." python3 live_trading/live_runner.py')
    print("\nExpected:")
    print("  ~27 trades/day, all LONG")

else:
    print("❌ NO - Setup has mismatches (see warnings above)")

print("\n" + "="*80)
