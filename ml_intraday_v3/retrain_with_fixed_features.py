"""
Re-train model with corrected features.

This script trains a new model using the fixed feature calculations
(no annualization, correct ATR, etc.) so it can be used for live trading.
"""

import sys
from pathlib import Path

# Add parent to path
parent_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(parent_dir))

print("=" * 80)
print("RE-TRAINING MODEL WITH CORRECTED FEATURES")
print("=" * 80)
print()

# Instructions for the user
print("""
⚠️  TRAINING SETUP REQUIRED

Due to import path dependencies, please train using one of these methods:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 METHOD 1: Use Jupyter Notebook (RECOMMENDED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Open the notebook:
   jupyter notebook ml_intraday_v3_pipeline_runner_enhanced.ipynb

2. Run the "Build Features" section
   - This will use the CORRECTED feature calculations
   - Features will be saved to: data/processed/features_train.parquet

3. Run the "Train Model" section
   - Model will be trained on corrected features
   - Saved to: models/saved/model_bundle.pkl

4. Run the "Backtest" section
   - Tests the model with same corrected features
   - Results saved to: analysis/backtest_results.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📦 METHOD 2: Check Existing Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your current model location:
""")

# Check for existing models
model_dirs = [
    Path("models/saved"),
    Path("ml_intraday_v3/models/saved"),
    parent_dir / "ml_intraday_v3" / "models" / "saved"
]

found_models = []
for model_dir in model_dirs:
    if model_dir.exists():
        bundles = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.joblib"))
        if bundles:
            found_models.append((model_dir, bundles))

if found_models:
    print("✅ Found existing model(s):")
    for model_dir, bundles in found_models:
        print(f"\n   📁 {model_dir}/")
        for bundle in bundles:
            size = bundle.stat().st_size / 1024 / 1024
            mtime = bundle.stat().st_mtime
            import datetime
            mod_date = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
            print(f"      • {bundle.name} ({size:.1f} MB, modified {mod_date})")

    print("\n⚠️  WARNING: These models were trained with OLD (incorrect) features!")
    print("   You MUST re-train before using for live trading.")
else:
    print("❌ No existing models found")
    print("   You need to train a model first")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 WHAT CHANGED IN FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fixed in features/build.py and live_trading/feature_generator.py:

✅ Removed annualization from vol_20, parkinson_vol, vol_forecast
   (Were ~20x too large, now match training expectations)

✅ Fixed ATR calculation (now uses EMA instead of simple MA)

✅ Fixed vol_regime (uses rolling median of vol_20 values)

✅ Fixed trend_strength (uses SMA distance, not linear regression)

✅ Fixed autocorr_5 (uses close prices, not returns)

✅ Fixed volume_imbalance (price direction ratio, not volume ratio)

✅ Fixed price_vs_vwap (50-bar lookback, not 20)

✅ Fixed large_move (return vs volatility, not body vs ATR)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏱️  ESTIMATED TIME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Feature building:  ~2-5 minutes
• Model training:    ~5-15 minutes
• Backtesting:       ~2-5 minutes

Total: ~10-25 minutes (depending on data size and CPU)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 AFTER TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Once you've re-trained the model:

1. ✅ Live trading will use corrected features (already fixed)
2. ✅ Model will receive correct inputs (matching training)
3. ✅ Backtest results will be meaningful
4. ✅ Ready for paper trading on Monday!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("=" * 80)
print("Run this script finished. Use Method 1 (notebook) to re-train.")
print("=" * 80)
