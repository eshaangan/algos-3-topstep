# 🚀 Step-by-Step Training Guide

## Quick Start

```bash
cd "/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3"
jupyter notebook ml_intraday_v3_pipeline_runner_enhanced.ipynb
```

---

## 📋 Step-by-Step Workflow

### **Step 1: Setup & Imports** ✅
**What to run:** First few cells with imports

**Expected output:**
```python
import pandas as pd
import numpy as np
# ... etc
✓ All imports successful
```

**Troubleshooting:**
- If import errors: `pip install -r requirements.txt`
- If module not found: Check PYTHONPATH

---

### **Step 2: Load Data** 📊
**What to run:** Cell that loads raw bar data

**Look for:**
```python
# Loading bars from HDF5 or CSV
bars = pd.read_hdf('data/processed/mes_bars.h5')
# or
bars = pd.read_csv('data/processed/glbx-mdp3-20100606-20251219.ohlcv-1m.csv')

print(f"Loaded {len(bars):,} bars")
print(f"Date range: {bars.index[0]} to {bars.index[-1]}")
```

**Expected output:**
```
Loaded 1,234,567 bars
Date range: 2010-01-01 to 2025-12-19
```

**Check:**
- ✅ Date range makes sense
- ✅ No missing data warnings
- ✅ OHLC columns present

---

### **Step 3: Build Features** 🔧
**What to run:** Feature building section

**This cell does:**
```python
from features.build import build_all_features

# This will use your CORRECTED feature calculations!
features_df = build_all_features(bars, config)
```

**Expected output:**
```
Building features...
✓ Price features: 10 features
✓ Volatility features: 6 features
✓ Trend features: 8 features
✓ Volume features: 5 features
✓ Time features: 3 features
Total: 32 features built

Feature stats:
  - Rows: 1,234,567
  - Features: 32
  - NaN values: 0.02%

Saving to: data/processed/features_train.parquet
✓ Features saved
```

**What to check:**
- ✅ All feature groups built successfully
- ✅ Low NaN percentage (< 5%)
- ✅ Features saved to parquet file
- ⚠️  **IMPORTANT:** These features use your FIXED calculations!

**Time:** ~2-5 minutes

---

### **Step 4: Build Labels** 🎯
**What to run:** Triple barrier labeling section

**This cell does:**
```python
from labeling.triple_barrier import build_labels

labels_df = build_labels(
    bars=bars,
    config=labeling_config,
    volatility_lookback=20,
    pt_multiplier=2.9,  # Profit target
    sl_multiplier=3.4,  # Stop loss
    horizon_bars=24
)
```

**Expected output:**
```
Building triple barrier labels...
Parameters:
  - PT multiplier: 2.9× ATR
  - SL multiplier: 3.4× ATR
  - Horizon: 24 bars
  - Volatility: ATR-14

Progress: 100%|████████████████| 1234567/1234567

Label distribution:
  +1 (target hit): 35.2%
  -1 (stop hit):   42.8%
   0 (vertical):   22.0%

Saving to: data/processed/labels_train.parquet
✓ Labels saved
```

**What to check:**
- ✅ Label distribution looks reasonable (not 90% one class)
- ✅ Roughly 30-40% targets, 40-50% stops, 20-30% vertical
- ✅ Labels saved successfully

**Time:** ~3-5 minutes (depends on data size)

---

### **Step 5: Merge Features + Labels** 🔗
**What to run:** Data merging section

**This cell does:**
```python
# Merge features with labels
train_data = features_df.join(labels_df, how='inner')

# Remove NaN rows
train_data = train_data.dropna()

print(f"Training data shape: {train_data.shape}")
print(f"Features: {len(feature_columns)}")
print(f"Samples: {len(train_data):,}")
```

**Expected output:**
```
Training data shape: (1,150,000, 33)
Features: 32
Samples: 1,150,000

Memory usage: 2.3 GB
```

**What to check:**
- ✅ Sample count makes sense (should be most of your bars)
- ✅ No excessive NaN removal (< 10% lost)

---

### **Step 6: Create Train/Test Splits** ✂️
**What to run:** Time-series split section

**This cell does:**
```python
from sklearn.model_selection import TimeSeriesSplit

# Walk-forward validation
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(train_data)):
    print(f"Fold {fold}: Train {len(train_idx):,}, Test {len(test_idx):,}")
```

**Expected output:**
```
Creating time-series splits...

Fold 0: Train 200,000, Test 150,000
Fold 1: Train 350,000, Test 150,000
Fold 2: Train 500,000, Test 150,000
Fold 3: Train 650,000, Test 150,000
Fold 4: Train 800,000, Test 150,000

✓ Splits created
```

**What to check:**
- ✅ Train size grows, test size stays constant (walk-forward)
- ✅ No data leakage (test dates always after train dates)

---

### **Step 7: Train Model** 🤖
**What to run:** Model training section

**This cell does:**
```python
from training.train import train_on_splits
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=100,
    random_state=42,
    n_jobs=-1
)

results = train_on_splits(
    model=model,
    X=features,
    y=labels,
    splits=tscv
)
```

**Expected output:**
```
Training model...

Fold 0/5: Training...
  - Train accuracy: 0.68
  - Test accuracy: 0.52
  - F1 score: 0.48
  ✓ Fold 0 complete (2m 34s)

Fold 1/5: Training...
  - Train accuracy: 0.67
  - Test accuracy: 0.53
  - F1 score: 0.49
  ✓ Fold 1 complete (3m 12s)

... (continues for all folds)

Average test accuracy: 0.52 ± 0.01
Average F1 score: 0.48 ± 0.02

✓ Training complete!
Saving model bundle to: models/saved/model_bundle.pkl
```

**What to check:**
- ✅ Test accuracy > 0.50 (better than random)
- ✅ Not too much overfitting (train vs test gap < 0.15)
- ✅ Consistent across folds (low standard deviation)
- ⚠️  Training accuracy should NOT be > 0.80 (overfitting!)

**Time:** ~5-15 minutes (depends on data size and model complexity)

---

### **Step 8: Save Model Bundle** 💾
**What to run:** Model saving section

**This cell does:**
```python
import joblib

model_bundle = {
    'primary_model': model,
    'primary_preprocessor': preprocessor,
    'feature_columns': feature_columns,
    'label_schema': label_config,
    'metadata': {
        'train_date': datetime.now(),
        'n_features': len(feature_columns),
        'n_samples': len(train_data)
    }
}

output_path = 'models/saved/model_bundle.pkl'
joblib.dump(model_bundle, output_path)

print(f"✓ Model saved to: {output_path}")
print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
```

**Expected output:**
```
Saving model bundle...
✓ Model saved to: models/saved/model_bundle.pkl
  Size: 45.3 MB

Bundle contains:
  - primary_model: RandomForestClassifier
  - primary_preprocessor: StandardScaler
  - feature_columns: 32 features
  - label_schema: triple_barrier config
  - metadata: training info
```

**What to check:**
- ✅ File size reasonable (10-100 MB typical)
- ✅ File exists at specified path
- ✅ Contains all required components

---

### **Step 9: Backtest** 📈
**What to run:** Backtesting section

**This cell does:**
```python
from backtesting.engine import run_backtest

backtest_results = run_backtest(
    model=model,
    data=test_data,
    config=backtest_config
)

# Generate report
print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)
print(f"Total Trades: {backtest_results['n_trades']}")
print(f"Win Rate: {backtest_results['win_rate']:.1f}%")
print(f"Total P&L: ${backtest_results['total_pnl']:,.2f}")
print(f"Profit Factor: {backtest_results['profit_factor']:.2f}")
print(f"Sharpe Ratio: {backtest_results['sharpe']:.2f}")
print(f"Max Drawdown: ${backtest_results['max_dd']:,.2f}")
print("="*60)
```

**Expected output:**
```
Running backtest...

Progress: 100%|████████████████| 50000/50000

============================================================
BACKTEST RESULTS
============================================================
Total Trades: 1,234
Win Rate: 52.3%
Total P&L: $15,432.50
Profit Factor: 1.82
Sharpe Ratio: 1.45
Max Drawdown: $1,245.00
============================================================

Saving results to: analysis/backtest_results.csv
✓ Backtest complete
```

**What to check:**
- ✅ Total P&L > $0 (profitable!)
- ✅ Profit Factor > 1.5
- ✅ Max Drawdown < $1,500
- ✅ Sharpe > 1.0
- ✅ Enough trades (> 200)

**Time:** ~2-5 minutes

---

## 🎯 Complete Workflow Summary

```
1. Setup & Imports           [30 sec]
2. Load Data                 [1 min]
3. Build Features            [2-5 min]  ← Uses CORRECTED calculations!
4. Build Labels              [3-5 min]
5. Merge Data                [30 sec]
6. Create Splits             [30 sec]
7. Train Model               [5-15 min]
8. Save Model                [30 sec]
9. Backtest                  [2-5 min]

Total Time: 15-30 minutes
```

---

## ⚠️ Common Issues & Fixes

### **Issue 1: Import Errors**
```
ModuleNotFoundError: No module named 'xyz'
```
**Fix:**
```bash
pip install xyz
# or
pip install -r requirements.txt
```

### **Issue 2: Memory Error**
```
MemoryError: Unable to allocate array
```
**Fix:**
- Use smaller data subset for testing
- Reduce `lookback_bars` parameter
- Close other applications

### **Issue 3: Feature NaN Values**
```
Warning: 15% of features are NaN
```
**Fix:**
- Check if you have enough bars for lookback periods
- Some features need 100+ bars of history
- First 100 bars will have NaN for some features (expected)

### **Issue 4: Model Overfitting**
```
Train accuracy: 0.95
Test accuracy: 0.52
```
**Fix:**
- Increase `min_samples_leaf` (e.g., 100 → 500)
- Decrease `max_depth` (e.g., 10 → 5)
- Add regularization

### **Issue 5: Poor Backtest Results**
```
Total P&L: -$5,234.50
Win Rate: 35%
```
**Possible causes:**
- Model not trained well
- Features have errors (but we just fixed them!)
- Hyperparameters need tuning
- Labels might be wrong

---

## ✅ Success Checklist

After running the entire notebook:

```
□ Features built with CORRECTED calculations
□ Labels created with triple barrier method
□ Model trained on 5 folds
□ Model bundle saved to models/saved/
□ Backtest shows positive P&L
□ Backtest shows profit factor > 1.5
□ Max drawdown < $1,500
□ No Topstep rule violations in backtest
□ Ready for live trading!
```

---

## 🚀 Next Steps After Training

Once backtest looks good:

1. **Verify Model Bundle:**
   ```bash
   python tests/test_infrastructure_fixes.py
   ```

2. **Test Live Features:**
   ```bash
   python live_trading/test_monitoring.py
   ```

3. **Start Paper Trading:**
   ```bash
   cd ml_intraday_v3
   PYTHONPATH=".." python live_trading/live_runner.py
   ```

---

## 📞 Need Help?

If you get stuck at any step:
1. Check the error message carefully
2. Look at the "Common Issues" section above
3. Share the error output for debugging

**Good luck with training!** 🎯
