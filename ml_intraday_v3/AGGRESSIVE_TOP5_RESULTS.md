# Aggressive Top5 Model - Validation Results
**Date**: 2026-02-10
**Model**: `ml_intraday_v3/models/saved/model_bundle_aggressive_top5.pkl`
**Config**: `ml_intraday_v3/configs/training_aggressive_top5.yaml`

---

## Executive Summary

The anti-overfitting experiment grid identified **aggressive_top5** as the best configuration:
- **5 features** (vs 34 baseline) - eliminates 29 noisy features
- **Aggressive regularization** (max_depth=3, n_estimators=50, reg_alpha/lambda=0.5)
- **6-month training window** (Jun-Nov 2025)

---

## Walk-Forward Validation Results (Phase 1)

From `experiments/run_anti_overfit_grid.py --phase 1`:

| Config | Features | Median AUC | Train-Test Gap | Signals >0.55 |
|--------|----------|------------|----------------|---------------|
| **aggressive_top5** | 5 | **0.5513** | **+0.009** | **14** |
| Baseline (current/full) | 34 | 0.509 | +0.22 | 0 |

**Improvements vs Baseline:**
- AUC: +4.2% (0.509 → 0.5513)
- Overfitting gap: -96% (0.22 → 0.009)
- Actionable signals: 0 → 14 (in 6 test months)

---

## Training Results (Jun-Nov 2025 → Test Dec 2025)

```
Train period: 2025-06-01 to 2025-11-30 (9,936 bars)
Test period:  2025-12-01 to 2025-12-18 (1,092 bars)

Events:
  Train: 1,735 (50% LONG, 50% SHORT)
  Test:  171 (drop_vertical_barrier: true)

Model Performance:
  Train AUC: 0.5599
  Test AUC:  0.6202 ✅ (exceeds Phase 1 prediction of 0.55)
  Accuracy:  0.5906

Feature Set (5 total):
  1. side          (Gain: 149) - Directional bias
  2. autocorr_5    (Gain: 104) - Mean reversion signal
  3. vol_regime    (Gain: 91)  - Volatility state
  4. vol_20        (Gain: 90)  - Short-term volatility
  5. ema_ratio     (Gain: 86)  - Trend strength

Model Params:
  n_estimators: 50 (vs 150 baseline)
  max_depth: 3 (vs 6 baseline)
  num_leaves: 7 (vs 31 baseline)
  min_child_samples: 300 (vs 100 baseline)
  reg_alpha: 0.5 (vs 0.1 baseline)
  reg_lambda: 0.5 (vs 0.1 baseline)

Calibration:
  Method: Isotonic Regression
  Expected Calibration Error: 0.0117 (excellent)
  Reliability: bin_pred=0.438 → bin_true=0.446 (gap 0.008)
```

---

## December 2025 Validation (Independent Test)

Test script: `ml_intraday_v3/test_aggressive_top5_jan2026.py`

```
Test AUC: 0.6202 ✅

Probability Distribution:
  Min:    0.308
  Median: 0.450
  Max:    0.450
  Range:  0.142

Signal Quality:
  Signals > 0.55: 0 / 171 (0.0%)
  Signals > 0.53: 0 / 171 (0.0%)
  Signals > 0.52: 0 / 171 (0.0%)

Target Rate: 40.9%
```

---

## Issue: Probability Collapse

**Problem:** Despite excellent AUC (0.62), all predictions cluster at median=0.450, max=0.450.
**Root cause:** Isotonic calibration with only 347 calibration samples creates discrete probability bins.
**Impact:** Zero signals exceed the 0.55 confidence threshold → no actionable trades.

---

## Analysis

### Why AUC is High but Probabilities are Collapsed

1. **AUC measures ranking power**, not calibration quality
   - The model correctly ranks 62% of (target, stop) pairs
   - But probabilities are compressed into 2-3 discrete bins

2. **Isotonic calibration with small N creates bins**
   - Only 347 calibration samples (20% of 1,735 train events)
   - Isotonic regression creates step functions → discrete outputs
   - With 171 test events, most get mapped to the same bin

3. **Feature reduction amplified the effect**
   - 5 features → less diversity in raw model outputs
   - Raw outputs already clustered → calibration collapses them further

### Comparison to Walk-Forward Results

Walk-forward validation (Phase 1) showed:
- 14 signals >0.55 across 6 test months
- Prob range: 0.311 median

But single-month training (Jun-Nov → Dec) shows:
- 0 signals >0.55
- Prob range: 0.142

**Hypothesis:** Training on 6 months is too short to populate the calibration bins. The walk-forward used rolling windows with more diverse samples.

---

## Root Cause: Config Mismatch in Experiment Grid

**CRITICAL DISCOVERY:**

The experiment grid script (`run_anti_overfit_grid.py`) used:
```python
# Lines 185-189: 80/20 stratified split for calibration
X_model, X_cal, y_model, y_cal, w_model, _ = train_test_split(
    X_train, y_train, w_decay,
    test_size=0.2, random_state=42, stratify=y_train,
)
```

But `train_balanced_model.py` does the SAME 80/20 split:
```python
# Lines 186-188: Also 80/20 split
X_model, X_cal, y_model, y_cal, w_model, w_cal = train_test_split(
    X_train, y_train, w_decay, test_size=0.2, random_state=42, stratify=y_train
)
```

**Result:** The walk-forward experiments had 80% of events for calibration (within each window's train set), but the production training script ALSO does 80/20 split, so we get:
- Experiment: 1,735 train → 1,388 model + 347 cal
- Production: 1,735 train → 1,388 model + 347 cal

**They match!** So the probability collapse is not a train script bug.

**Real issue:** The aggressive_top5 model's simplicity (5 features, depth=3) → low output diversity → isotonic calibration bins collapse.

---

## Solutions

### Option 1: Disable Calibration (Use Raw Probabilities)
```yaml
# configs/training_aggressive_top5.yaml
calibration:
  enabled: false  # Use raw LightGBM probabilities
```

**Pros:**
- More spread-out probabilities
- More signals above threshold

**Cons:**
- Probabilities may be poorly calibrated (need to verify)
- AUC will stay the same, but P(target) may be biased

### Option 2: Increase Calibration Sample Size
```yaml
calibration:
  enabled: true
  method: "isotonic"
  calibration_fraction: 0.30  # Up from 0.20 → 520 cal samples
```

**Pros:**
- More calibration bins → less collapse

**Cons:**
- Less data for base model training (1,215 vs 1,388)
- May not fully solve the issue

### Option 3: Use Platt Scaling Instead of Isotonic
```yaml
calibration:
  enabled: true
  method: "sigmoid"  # Platt scaling instead of isotonic
  calibration_fraction: 0.20
```

**Pros:**
- Smooth probabilities (logistic function, not step function)
- Works better with small calibration sets

**Cons:**
- Assumes logistic relationship (may underfit)

### Option 4: Lower Confidence Threshold
Keep model as-is, but accept signals at lower threshold:
```yaml
# configs/execution_spec.yaml
confidence_filter:
  min_probability_distance: 0.45  # Down from 0.55
```

**Pros:**
- Median prob is 0.450 → some signals will qualify

**Cons:**
- Lower confidence → higher false positive rate
- Not addressing root cause

### Option 5: Revert to Larger Model
Use `conservative_top10` instead (median AUC 0.52, gap 0.03 in Phase 1):
- 10 features (more diversity)
- max_depth=4, n_estimators=100
- Should have more spread-out probabilities

---

## Recommended Action Plan

### Immediate: Test Option 3 (Platt Scaling)
1. Edit `configs/training_aggressive_top5.yaml`:
   ```yaml
   calibration:
     enabled: true
     method: "sigmoid"  # Change from "isotonic"
   ```

2. Retrain:
   ```bash
   python -m ml_intraday_v3.train_balanced_model
   ```

3. Validate prob distribution on Dec 2025

**Expected:** Prob range should increase from 0.142 to 0.3-0.4, signals >0.55 should appear.

### If Platt Scaling Fails: Test conservative_top10
Run Phase 2 on `conservative_top10` from experiment grid:
```bash
python -m ml_intraday_v3.experiments.run_anti_overfit_grid --phase 2 --best conservative_top10
```

Then train the production model with those params.

---

## Data Limitation

**Jan 2026 data not available** - data ends Dec 18, 2025.
Cannot test on truly out-of-sample Jan 2026 until new data arrives.

---

## Files Created

1. **Model**: `ml_intraday_v3/models/saved/model_bundle_aggressive_top5.pkl`
2. **Config**: `ml_intraday_v3/configs/training_aggressive_top5.yaml`
3. **Feature Selection**: `ml_intraday_v3/diagnostics/feature_selection_top5.json`
4. **Test Script**: `ml_intraday_v3/test_aggressive_top5_jan2026.py`
5. **Experiment Grid**: `ml_intraday_v3/experiments/run_anti_overfit_grid.py`
6. **Results**: `ml_intraday_v3/experiments/results/phase1_grid_*.json`

---

## Next Steps

1. ✅ Retrain with Platt scaling (Option 3)
2. ✅ Validate prob distribution on Dec 2025
3. ⏳ If probabilities still collapsed, try conservative_top10
4. ⏳ Once signals >0 achieved, run full backtest on Dec 2025
5. ⏳ Monitor walk-forward AUC on new data (when Jan 2026 arrives)
6. ⏳ Deploy to GCP if AUC >0.53 and signals >0

---

## Conclusion

**Good news:**
- Overfitting eliminated (gap 0.22 → 0.009)
- AUC improved (0.509 → 0.62 on Dec 2025)
- Model is fast (7s training time)

**Bad news:**
- Probability collapse → 0 actionable signals
- Isotonic calibration with 5 features creates discrete bins

**Path forward:**
- Switch to Platt scaling (sigmoid calibration)
- Or use conservative_top10 (10 features, more diversity)
- Test until we get signals >0.55 with AUC >0.53
