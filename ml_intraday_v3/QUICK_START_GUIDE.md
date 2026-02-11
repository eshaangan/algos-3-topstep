# Quick Start Guide - Model Simplification Plan

## 🚀 Week 1 Execution (Day 7)

### Prerequisites

Ensure you have the required data files:
```bash
ls ml_intraday_v3/data/bars.h5
ls ml_intraday_v3/data/features_5m.parquet
ls ml_intraday_v3/data/events_5m.parquet
```

If missing, generate them:
```bash
# Generate features and events
python ml_intraday_v3/train_balanced_model.py
```

---

### Step 1: Run Baseline Training

```bash
cd "/Users/eshaanganguly/Documents/projects/algos 3 topstep"

python ml_intraday_v3/train_simple_baseline.py
```

**Expected Output**:
```
Loading data...
   Loaded 50,123 bars (2022-01-03 to 2025-12-31)
   Loaded 50,123 feature rows
   Loaded 12,456 events

   Filtered vertical barriers: 12,456 → 2,867 (77.0% removed)
   Label distribution:
      TARGET (+1): 1,433 (50.0%)
      STOP (-1): 1,434 (50.0%)

📊 Feature Selection:
   Target: 12 core features
   Available: 12

🔬 Training Logistic Regression Baseline...
   Fitting model with sample weights...

📈 Training Results:
   Accuracy: 0.623
   Balanced Accuracy: 0.622
   AUC: 0.653

📊 Test Results (Dec 2025 Holdout):
   Accuracy: 0.567
   Balanced Accuracy: 0.566
   AUC: 0.604
   Precision: 0.568
   Recall: 0.567
   F1 Score: 0.566

🎯 Decision Criteria:
   AUC: 0.604 (target: 0.58+)
   Accuracy: 0.567 (target: 0.54+)
   ✅ GO: Linear edge exists, proceed to Week 2

💾 Saved results to ml_intraday_v3/diagnostics/week1_baseline_results.json
```

---

### Step 2: Check Results

```bash
# View JSON results
cat ml_intraday_v3/diagnostics/week1_baseline_results.json

# View feature importance
cat ml_intraday_v3/diagnostics/week1_feature_importance.csv

# View confusion matrix
open ml_intraday_v3/diagnostics/week1_confusion_matrix.png
```

---

### Step 3: GO/NO-GO Decision

```bash
python -c "
import json

r = json.load(open('ml_intraday_v3/diagnostics/week1_baseline_results.json'))
acc = r['test']['accuracy']
auc = r['test']['auc']

print('=' * 50)
print('WEEK 1 GO/NO-GO DECISION')
print('=' * 50)
print(f'Test Accuracy: {acc:.3f} (target: 0.54+)')
print(f'Test AUC: {auc:.3f} (target: 0.58+)')
print('=' * 50)

if acc >= 0.54 and auc >= 0.58:
    print('✅ GO: Linear edge exists')
    print('   Proceed to Week 2 (filters + validation)')
elif auc >= 0.55:
    print('⚠️  PARTIAL: Edge exists but weak')
    print('   Consider calibration or threshold tuning')
else:
    print('❌ NO-GO: No linear edge detected')
    print('   Revisit features or approach')
"
```

---

## 🔧 Week 2 Execution (Days 8-14)

### Day 8-9: Enable Filters

**Edit**: `ml_intraday_v3/configs/execution_spec.yaml`

```yaml
filters:
  volatility:
    enabled: true  # Change from false
    min_percentile: 30
    max_percentile: 70
    lookback_bars: 100

  time_of_day:
    enabled: true  # Change from false
    start_hour: 9
    start_minute: 30
    end_hour: 13
    end_minute: 30
    timezone: "America/Chicago"
```

---

### Day 10-11: Model Comparison

```bash
python ml_intraday_v3/experiments/model_comparison.py
```

**Expected Output**:
```
📊 Model Comparison Results:
                  model  accuracy  balanced_accuracy   auc  precision  recall    f1
  Logistic Regression     0.567              0.566 0.604      0.568   0.567 0.566
       Shallow XGBoost     0.579              0.578 0.618      0.581   0.579 0.579
      Shallow LightGBM     0.574              0.573 0.612      0.576   0.574 0.574

🎯 Decision:
   Best Model: Shallow XGBoost (AUC: 0.618)
   LogReg AUC: 0.604
   ✅ Chosen Model: Logistic Regression
   Reason: Tree model only 2.3% better (< 3% threshold) - prefer simplicity
```

---

### Day 12-13: Walk-Forward Validation

```bash
# TO BE IMPLEMENTED
python ml_intraday_v3/experiments/walk_forward_validation.py
```

---

### Day 14: Paper Trading Simulation

```bash
# TO BE IMPLEMENTED
python ml_intraday_v3/experiments/paper_trading_simulation.py
```

---

## 🐛 Troubleshooting

### Data Not Found

```bash
# Regenerate features and events
python -m ml_intraday_v3.cli build-features \
    --run-dir runs/v3_2022_5m \
    --bar-sizes 5m

python -m ml_intraday_v3.cli build-labels \
    --run-dir runs/v3_2022_5m \
    --bar-sizes 5m
```

---

### Features Missing

Check which features exist:
```python
import pandas as pd
features = pd.read_parquet('ml_intraday_v3/data/features_5m.parquet')
print(features.columns.tolist())
```

---

### Low Accuracy/AUC

**If AUC < 0.55**:
- No linear edge exists
- Try alternative features
- Consider stopping if no improvement

**If AUC > 0.55 but accuracy < 0.54**:
- Calibration issue
- Apply Platt scaling or isotonic regression
- Optimize decision threshold

---

## 📊 Expected Performance Benchmarks

### Week 1 Baseline

| Metric | Minimum | Target | Excellent |
|--------|---------|--------|-----------|
| Test Accuracy | 0.54 | 0.57 | 0.60+ |
| Test AUC | 0.58 | 0.61 | 0.65+ |
| LONG Precision | 0.48 | 0.52 | 0.55+ |
| SHORT Precision | 0.48 | 0.52 | 0.55+ |

### Week 2 Walk-Forward

| Metric | Minimum | Target |
|--------|---------|--------|
| Positive Months | 4/6 (67%) | 5/6 (83%) |
| Cumulative PnL | $500 | $800+ |
| Avg Sharpe | 0.8 | 1.2+ |
| Max DD | <$1,500 | <$1,000 |

### Week 2 Paper Trading

| Metric | Minimum | Target |
|--------|---------|--------|
| Positive Days | 6/10 (60%) | 7/10 (70%) |
| Cumulative PnL | $300 | $500+ |
| Max DD | <$1,500 | <$1,000 |
| Violations | 0 | 0 |

---

## ⏱️ Time Estimates

| Task | Estimated Time |
|------|---------------|
| Week 1 baseline training | 5-10 minutes |
| Week 2 model comparison | 10-15 minutes |
| Week 2 walk-forward (to implement) | 30-60 minutes |
| Week 2 paper trading (to implement) | 10-20 minutes |

---

## 📁 Key Files Reference

### Configuration
- `ml_intraday_v3/configs/features.yaml` - Feature settings
- `ml_intraday_v3/configs/labeling.yaml` - Label settings
- `ml_intraday_v3/configs/training.yaml` - Training settings
- `ml_intraday_v3/configs/execution_spec.yaml` - Filter settings

### Scripts
- `ml_intraday_v3/train_simple_baseline.py` - Week 1 baseline
- `ml_intraday_v3/experiments/model_comparison.py` - Week 2 model comparison

### Diagnostics
- `ml_intraday_v3/diagnostics/week1_baseline_results.json`
- `ml_intraday_v3/diagnostics/week1_feature_importance.csv`
- `ml_intraday_v3/diagnostics/week1_confusion_matrix.png`

---

**Last Updated**: January 29, 2026
