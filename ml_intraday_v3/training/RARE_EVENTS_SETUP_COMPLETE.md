# Rare Events Corrections - Setup Complete ✅

## Overview

Rare events corrections (King & Zeng 2001) have been successfully enabled in your ML Pipeline V3. The corrections will automatically apply when using logistic regression models (`model.kind = "logreg"`).

## Files Modified

### 1. Configuration: `ml_intraday_v3/configs/training.yaml`

Added rare events configuration section:
```yaml
rare_events:
  enabled: true
  tau: null  # Auto-estimate from data
  use_sample_weights: true
  weight_method: "king_zeng"
  correction_method: "king_zeng"
```

**Current Status**: ✅ ENABLED

### 2. Training Script: `ml_intraday_v3/training/train.py`

**Changes made:**
- ✅ Import `RelogitClassifier` from `rare_events` module
- ✅ Read rare events configuration from `training.yaml`
- ✅ Use `RelogitClassifier` instead of `LogisticRegression` when enabled
- ✅ Store uncorrected probabilities for comparison
- ✅ Save both corrected and uncorrected probabilities to disk

**Prediction columns saved:**
- `event_id` - Event identifier
- `y_true` - True labels
- `p_target` - **Corrected** probabilities (if rare events enabled)
- `p_target_uncorrected` - **Uncorrected** probabilities (only when rare events enabled)
- `y_pred` - Binary predictions
- `weight` - Sample weights

### 3. Pipeline Notebook: `ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb`

**New sections added:**
- ✅ Section 4.6.1: Rare Events Corrections - Overview
- ✅ Section 4.6.3: Rare Events Calibration Analysis

These sections will automatically load predictions from disk and show calibration improvements.

## How It Works

### When Training with `model.kind = "logreg"`:

1. **Rare events corrections are ENABLED** (per your request)
2. The pipeline uses `RelogitClassifier` instead of standard `LogisticRegression`
3. During training:
   - Estimates tau (prior probability) from training data (since `tau: null`)
   - Applies King & Zeng sample weighting to training samples
   - Fits the underlying logistic regression model
4. During prediction:
   - Gets uncorrected probabilities from the model
   - Applies King & Zeng probability correction formula
   - Returns corrected probabilities
5. Saves both versions to disk for comparison

### When Training with `model.kind = "lgbm"`:

- Rare events corrections **do not apply** (they're specific to logistic regression)
- Pipeline uses standard LightGBM as before

## Current Configuration

Since your current `training.yaml` has:
```yaml
model:
  kind: "lgbm"  # LightGBM (Phase 5 upgrade)
```

The rare events corrections won't apply **yet** because you're using LightGBM.

## To Use Rare Events Corrections

### Option 1: Switch to Logistic Regression

Change in `configs/training.yaml`:
```yaml
model:
  kind: "logreg"  # Changed from "lgbm"
  params:
    C: 1.0
    penalty: "l2"
    solver: "lbfgs"
    max_iter: 200
```

Then run your pipeline normally - rare events corrections will automatically apply!

### Option 2: Compare Both Models

1. Run with `kind: "lgbm"` (current setup)
2. Run again with `kind: "logreg"` (rare events will apply)
3. Compare results in the notebook

## Expected Results

When using rare events corrections with imbalanced data:

### Calibration Improvement
- **Brier score**: Lower (better calibration)
- **Mean predicted probability**: Closer to true rate
- **Calibration curve**: Closer to diagonal line

### Discrimination Preserved
- **ROC AUC**: Unchanged (±0.01)
- **Ranking**: Identical

### Practical Impact
- More accurate probability estimates
- Better position sizing decisions
- Improved risk management

## Validation in Notebook

After training, the notebook section 4.6.3 will automatically:
1. Load predictions from all CV folds
2. Compare corrected vs uncorrected probabilities
3. Show side-by-side calibration curves
4. Report Brier score improvement
5. Validate AUC preservation

## Configuration Reference

### `tau` Parameter
- `null`: Auto-estimate from training data (recommended)
- `0.05`: 5% win rate (for highly imbalanced)
- `0.35`: 35% win rate (moderately imbalanced)

### When to Enable
✅ Use when:
- Class imbalance > 80:20 (< 20% positive)
- Probability calibration matters
- Using logistic regression
- Known or estimable prior

❌ Skip when:
- Balanced data (40-60% positive)
- Only care about AUC/ranking
- Using tree-based models (LightGBM, XGBoost)

## Files You Can Reference

- **Quick Start**: `ml_intraday_v3/training/RARE_EVENTS_QUICKSTART.md`
- **Full Documentation**: `ml_intraday_v3/training/README_RARE_EVENTS.md`
- **Implementation**: `ml_intraday_v3/training/rare_events.py`
- **Tests**: `ml_intraday_v3/tests/test_rare_events.py`
- **Demo**: `ml_intraday_v3/training/rare_events_demo.py`

## Next Steps

1. **To activate now**: Change `model.kind` to `"logreg"` in `configs/training.yaml`
2. **Run pipeline**: Execute notebook sections 4.1-4.6
3. **View results**: Section 4.6.3 shows calibration analysis
4. **Compare**: Run with both "lgbm" and "logreg" to compare

Everything is ready to use! The rare events corrections will activate automatically when you switch to logistic regression.
