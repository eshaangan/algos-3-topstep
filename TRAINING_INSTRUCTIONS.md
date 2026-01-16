# Training Instructions - Bidirectional 24-Hour Model

## ✅ Configurations Already Updated

All configs have been updated for 24-hour bidirectional trading:
- ✓ `data.yaml` - 24-hour grid
- ✓ `labeling.yaml` - trend_scanning (adds 'side')
- ✓ `live_trading.yaml` - 24-hour sessions, 0.03 threshold
- ✓ `execution_spec.yaml` - 24-hour sessions, 0.03 threshold
- ✓ `backtest.yaml` - 0.03 threshold, no regime filters

**You just need to run the training notebook now!**

---

## Method 1: Jupyter Notebook (RECOMMENDED)

### Step 1: Open the Notebook
```bash
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep
jupyter notebook ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb
```

### Step 2: Configure Run ID
In the notebook's first cell, set:
```python
RUN_ID = "bidirectional_24h_20260114"  # Or use timestamp
BAR_SIZE = "5m"
CV_KIND = "purged_kfold"
```

### Step 3: Run All Cells
Click: `Cell > Run All`

The notebook will:
1. **Load data** (2022-2025, 24-hour) - ~5 min
2. **Reindex to grid** (full_range for 24-hour) - ~10 min
3. **Build features** (33 features) - ~15 min
4. **Generate labels** (trend_scanning with 'side') - ~30 min
5. **Compute weights** (uniqueness + magnitude) - ~10 min
6. **Train model** (LightGBM with 'side' feature) - ~20 min
7. **Validate** (K-Fold, Walk-Forward) - ~40 min
8. **Backtest** (with bidirectional logic) - ~20 min

**Total time**: ~2.5-3 hours

### Step 4: Verify 'side' Feature
After training completes, check:
```python
import pickle
bundle = pickle.load(open('ml_intraday_v3/models/saved/model_bundle.pkl', 'rb'))
print("Has 'side' feature:", 'side' in bundle['feature_columns'])
print("Feature columns:", bundle['feature_columns'])
```

Should show:
```
Has 'side' feature: True
Feature columns: ['log_ret', 'log_ret_3', 'atr', 'vol_20', ..., 'side']
```

---

## Method 2: Command Line (Alternative)

If the notebook has issues, use the CLI:

```bash
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep

# Stage 1: Build data
python -m ml_intraday_v3.cli build-data --run-id bidirectional_24h_20260114 --bar-size 5m

# Stage 2: Build features
python -m ml_intraday_v3.cli build-features --run-id bidirectional_24h_20260114 --bar-size 5m

# Stage 3: Generate labels (trend_scanning)
python -m ml_intraday_v3.cli generate-labels --run-id bidirectional_24h_20260114 --bar-size 5m

# Stage 4: Compute weights
python -m ml_intraday_v3.cli compute-weights --run-id bidirectional_24h_20260114 --bar-size 5m

# Stage 5: Train model
python -m ml_intraday_v3.cli train-model --run-id bidirectional_24h_20260114 --bar-size 5m

# Stage 6: Validate
python -m ml_intraday_v3.cli validate --run-id bidirectional_24h_20260114 --bar-size 5m --cv-kind purged_kfold

# Stage 7: Backtest
python -m ml_intraday_v3.cli backtest --run-id bidirectional_24h_20260114 --bar-size 5m
```

---

## What to Expect

### During Training
You'll see output like:
```
STAGE 3: Labeling (trend_scanning)
  ✓ Generated 182,453 events
  ✓ Has 'side' feature: True
  Side distribution:
    LONG (side=1): 87,621
    SHORT (side=-1): 94,832

STAGE 5: Training
  ✓ Model trained: LightGBMClassifier
  ✓ Features: 34 (33 base + 'side')
  ✓ Training score: 0.xxxx
```

### Model Output Location
```
ml_intraday_v3/runs/bidirectional_24h_20260114/bar_size=5m/
├── bars.parquet              # 24-hour bar data
├── features.parquet          # Features with 'side'
├── labels.parquet            # Labels from trend_scanning
├── model_bundle.pkl          # Trained model
├── backtest_results.json     # Performance metrics
└── equity_curve.csv          # Equity curve

ml_intraday_v3/models/saved/
└── model_bundle.pkl          # Copy for live trading
```

---

## Validation Checklist

After training completes, verify:

### ✓ Model has 'side' feature
```python
import pickle
bundle = pickle.load(open('ml_intraday_v3/models/saved/model_bundle.pkl', 'rb'))
assert 'side' in bundle['feature_columns'], "ERROR: 'side' feature missing!"
print("✓ Model is bidirectional")
```

### ✓ Labels show balanced sides
```python
import pandas as pd
labels = pd.read_parquet('ml_intraday_v3/runs/bidirectional_24h_20260114/bar_size=5m/labels.parquet')
print("Side distribution:")
print(labels['side'].value_counts())
# Should show roughly 45-55% split between -1 and 1
```

### ✓ Backtest shows bidirectional trades
```python
import json
with open('ml_intraday_v3/runs/bidirectional_24h_20260114/bar_size=5m/backtest_results.json') as f:
    results = json.load(f)
print(f"Total trades: {results['total_trades']}")
print(f"LONG trades: {results.get('long_trades', 'N/A')}")
print(f"SHORT trades: {results.get('short_trades', 'N/A')}")
```

---

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError`, run from repo root:
```bash
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Memory Issues
If training runs out of memory:
1. Edit `data.yaml` line 26: `min_date: "2024-01-01"` (use 1 year instead of 3)
2. Restart kernel and re-run notebook

### 'side' Feature Missing
If 'side' doesn't appear in features:
1. Check `labeling.yaml` line 14: `event_policy: "trend_scanning"`
2. Re-run "Generate Labels" stage
3. Verify labels.parquet has 'side' column

### Zero Events Generated
If trend_scanning generates 0 events:
1. Lower `tstat_threshold` in `labeling.yaml`: try 1.0 instead of 1.5
2. Lower `cusum_threshold_atr_mult`: try 0.5 instead of 0.8

---

## Next Steps After Training

Once training completes successfully:

1. **Validate model**:
   ```bash
   python check_model.py  # Verifies 'side' feature exists
   ```

2. **Paper trade** (Monday-Tuesday):
   ```bash
   cd ml_intraday_v3/live_trading
   python live_runner.py --paper --no-confirm
   ```

3. **Monitor logs** for bidirectional signals:
   ```bash
   tail -f logs/live_trading_*.log | grep -E "Bidirectional|score_ev"
   ```

4. **Go live** (Wednesday+) if paper trading looks good:
   ```bash
   python live_runner.py --live
   ```

---

## Expected Results

With 2022-2025 data and trend_scanning:
- **Events**: ~180k-200k
- **Training time**: 2.5-3 hours
- **Model features**: 34 (33 base + 'side')
- **Backtest trades**: 15-25 per day (24-hour trading)
- **Win rate**: 55-65% (bidirectional)
- **Sharpe ratio**: 1.5-2.5 (aggressive threshold)

Ready to start? Run:
```bash
jupyter notebook ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb
```
