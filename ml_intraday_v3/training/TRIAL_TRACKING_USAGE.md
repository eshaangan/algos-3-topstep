# Trial Tracking & PBO Analysis - Usage Guide

## Overview

This guide shows you how to use **TrialTracker** with your **real MES/ES data** to:
1. Track all hyperparameter configurations tested
2. Compute PBO (Probability of Backtest Overfitting)
3. Assess overfitting risk before deployment

---

## Quick Start

### 1. Enable Hyperparameter Search

Edit `configs/training.yaml`:

```yaml
hyperparam_search:
  enabled: true  # Enable grid search

  model:
    kind: "lgbm"

    param_grid:
      n_estimators: [300, 500, 1000]
      learning_rate: [0.01, 0.05, 0.1]
      max_depth: [4, 6, 8]
      # This creates 3 × 3 × 3 = 27 trials
```

### 2. Run Hyperparameter Search (Real Data)

```bash
# Make sure you're in the project root
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep

# Set PYTHONPATH
export PYTHONPATH="/Users/eshaanganguly/Documents/projects/algos 3 topstep"

# Run hyperparameter search with your real data
python3 -m ml_intraday_v3.training.hyperparam_search \
    --run-dir runs/run_20251224_123456 \
    --bar-size 1m \
    --config ml_intraday_v3/configs/training.yaml \
    --cv-kind cpcv
```

**What happens:**
- Loads your **real data** from `data/processed/mes_bars.h5`
- Tests all 27 configurations on CPCV splits
- Tracks **every trial** (not just the best one)
- Saves results to `runs/run_20251224_123456/trials/trials.json`

### 3. Analyze Results & Compute PBO

```bash
python3 -m ml_intraday_v3.training.analyze_pbo \
    --run-dir runs/run_20251224_123456 \
    --metric roc_auc
```

**Output:**
```
================================================================================
PBO Results
================================================================================

PBO: 0.342 (34.2%)
95% CI: [0.298, 0.387]

Lambda Statistics:
  Mean: 0.512
  Median: 0.501
  Std: 0.156

Risk Level: 🟠 MODERATE RISK
Interpretation: Some overfitting risk detected
Recommended Action: Monitor carefully, consider additional validation

✓ Report saved to: runs/run_20251224_123456/pbo_analysis/pbo_report.md
✓ Lambda distribution saved to: runs/run_20251224_123456/pbo_analysis/pbo_lambda_distribution.png
✓ PBO with CI saved to: runs/run_20251224_123456/pbo_analysis/pbo_with_confidence.png
```

---

## Detailed Workflow

### Step 1: Prepare Your Data

Ensure you have real data ready:

```bash
ls -lh data/processed/mes_bars.h5
# Should show your real MES data file
```

### Step 2: Design Hyperparameter Grid

**Small Grid (Fast, for testing):**
```yaml
hyperparam_search:
  enabled: true
  model:
    kind: "lgbm"
    param_grid:
      n_estimators: [300, 500]
      learning_rate: [0.05, 0.1]
      max_depth: [4, 6]
      # 2 × 2 × 2 = 8 trials
```

**Medium Grid (Balanced):**
```yaml
hyperparam_search:
  enabled: true
  model:
    kind: "lgbm"
    param_grid:
      n_estimators: [300, 500, 1000]
      learning_rate: [0.01, 0.05, 0.1]
      max_depth: [4, 6, 8]
      # 3 × 3 × 3 = 27 trials
```

**Large Grid (Comprehensive, slow):**
```yaml
hyperparam_search:
  enabled: true
  model:
    kind: "lgbm"
    param_grid:
      n_estimators: [300, 500, 1000]
      learning_rate: [0.01, 0.05, 0.1]
      max_depth: [4, 6, 8]
      num_leaves: [15, 31, 63]
      min_child_samples: [50, 100]
      # 3 × 3 × 3 × 3 × 2 = 162 trials
```

**WARNING:** Grid size = product of all list lengths!

### Step 3: Run Search

```bash
# Full command with all options
python3 -m ml_intraday_v3.training.hyperparam_search \
    --run-dir runs/run_20251224_123456 \
    --bar-size 1m \
    --config ml_intraday_v3/configs/training.yaml \
    --cv-kind cpcv \
    --log-level INFO
```

**Command options:**
- `--run-dir`: Your run directory (must exist with processed data)
- `--bar-size`: `1m` or `5m`
- `--config`: Path to training config YAML
- `--cv-kind`: `cpcv` (recommended) or `purged_kfold`
- `--log-level`: `DEBUG`, `INFO`, `WARNING`, or `ERROR`

**Runtime estimate:**
- Each trial trains on all CPCV paths (typically 10-15 paths)
- Example: 27 trials × 10 paths × 2 min/path = ~9 hours

### Step 4: Monitor Progress

The script outputs progress in real-time:

```
================================================================================
Trial 5/27
================================================================================
Model: lgbm
Hyperparameters: {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6}
Trial ID: abc123def456
Training completed: runs/run_20251224_123456/bar_size=1m/trials/abc123def456
✓ Trial 5 completed successfully
  Tracked 10 CPCV paths
```

If a trial fails, it continues with the next one.

### Step 5: Analyze PBO

```bash
python3 -m ml_intraday_v3.training.analyze_pbo \
    --run-dir runs/run_20251224_123456 \
    --metric roc_auc \
    --n-bootstrap 1000 \
    --confidence-level 0.95
```

**Options:**
- `--metric`: Metric to use (`roc_auc`, `sharpe`, etc.)
- `--n-bootstrap`: Bootstrap samples for CI (default: 1000)
- `--confidence-level`: CI level (default: 0.95)
- `--output-dir`: Custom output directory (default: `run_dir/pbo_analysis`)

---

## Interpreting Results

### PBO Values

| PBO Range | Risk Level | Interpretation | Action |
|-----------|------------|----------------|--------|
| < 0.3 | 🟢 **Low** | Configuration appears robust | Proceed with deployment |
| 0.3 - 0.5 | 🟠 **Moderate** | Some overfitting detected | Monitor closely, add validation |
| > 0.5 | 🔴 **High** | Likely overfitting | **DO NOT deploy** - reduce search space |

### Lambda (λ) Distribution

**Lambda** = percentile rank of selected config on out-of-sample data

- **λ > 0.5**: Config performs above median OOS (good sign)
- **λ < 0.5**: Config performs below median OOS (overfitting)

**Example:**
```
Lambda Statistics:
  Mean: 0.512   ← Slightly above median (good)
  Median: 0.501 ← Very close to 0.5 (neutral)
  Std: 0.156    ← Moderate variability
```

### Confidence Intervals

```
PBO: 0.342
95% CI: [0.298, 0.387]
```

- Narrow CI → more reliable estimate
- Wide CI → need more trials or paths
- CI excludes 0.5 → confident about risk level

---

## Real Data vs Demo

### ❌ Demo Script (Synthetic Data)

```bash
# This creates FAKE trials for testing
python3 ml_intraday_v3/training/demo_pbo_with_synthetic_trials.py \
    runs/demo_test \
    moderate_overfitting
```

**Use case:** Testing PBO code only

### ✅ Real Training (Your Data)

```bash
# This uses YOUR REAL MES/ES data
python3 -m ml_intraday_v3.training.hyperparam_search \
    --run-dir runs/run_20251224_123456 \
    --bar-size 1m
```

**Use case:** Production model selection

---

## Files Created

After running hyperparameter search + PBO analysis:

```
runs/run_20251224_123456/
├── trials/
│   ├── trials.json              ← All tracked trials
│   ├── abc123def456/            ← Trial 1 results
│   │   ├── summary.json
│   │   └── ...
│   ├── def456ghi789/            ← Trial 2 results
│   └── ...
├── pbo_analysis/
│   ├── pbo_report.md            ← Full report
│   ├── pbo_lambda_distribution.png
│   └── pbo_with_confidence.png
└── bar_size=1m/
    ├── events.parquet           ← Your real labels
    ├── features.parquet         ← Your real features
    └── ...
```

---

## Advanced Usage

### Custom Metric

```bash
python3 -m ml_intraday_v3.training.analyze_pbo \
    --run-dir runs/run_20251224_123456 \
    --metric sharpe_ratio \
    --higher-is-better true
```

### Lower-is-Better Metrics

```bash
python3 -m ml_intraday_v3.training.analyze_pbo \
    --run-dir runs/run_20251224_123456 \
    --metric log_loss \
    --higher-is-better false
```

### Programmatic Usage

```python
from pathlib import Path
from ml_intraday_v3.experiments.trial_tracker import TrialTracker
from ml_intraday_v3.experiments.diagnostics import compute_pbo_with_confidence

# Load trials
tracker = TrialTracker("runs/run_20251224_123456")
trials_df = tracker.to_dataframe()

# Compute PBO
pbo_result = compute_pbo_with_confidence(
    trials_df,
    metric_name='roc_auc',
    higher_is_better=True,
    n_bootstrap=1000
)

print(f"PBO: {pbo_result['pbo']:.3f}")
print(f"95% CI: [{pbo_result['pbo_lower']:.3f}, {pbo_result['pbo_upper']:.3f}]")
```

---

## Troubleshooting

### "No trials found"

**Cause:** Haven't run hyperparameter search yet

**Fix:**
```bash
python3 -m ml_intraday_v3.training.hyperparam_search \
    --run-dir runs/run_20251224_123456 \
    --bar-size 1m
```

### "Need at least 2 trials for PBO"

**Cause:** Grid too small or many trials failed

**Fix:** Increase grid size in `configs/training.yaml`

### "Need at least 2 CPCV paths"

**Cause:** Using `purged_kfold` instead of `cpcv`

**Fix:** Use `--cv-kind cpcv`

### Import errors

**Cause:** PYTHONPATH not set

**Fix:**
```bash
export PYTHONPATH="/Users/eshaanganguly/Documents/projects/algos 3 topstep"
```

---

## Next Steps

1. **Start small:** Test with 8-trial grid first
2. **Validate PBO:** Should see reasonable values (0.2-0.6)
3. **Expand grid:** Increase to 27+ trials
4. **Review plots:** Check lambda distribution visually
5. **Make decision:** Deploy if PBO < 0.5

---

## References

- **PBO Theory:** López de Prado, M. (2018). *Advances in Financial Machine Learning.* Chapter 11.
- **Implementation Docs:** `ml_intraday_v3/experiments/PBO_ENHANCED_README.md`
- **Tests:** `ml_intraday_v3/tests/test_pbo_enhanced.py`
