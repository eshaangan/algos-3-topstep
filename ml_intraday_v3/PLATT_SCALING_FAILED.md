# Platt Scaling (Option 3) - Failed

**Date**: 2026-02-10
**Test**: Sigmoid calibration to fix probability collapse

---

## Result: FAILED ❌

**Sigmoid calibration did NOT solve the probability collapse.**

### Isotonic vs Sigmoid Comparison

| Metric | Isotonic | Sigmoid | Change |
|--------|----------|---------|--------|
| Test AUC | 0.6202 | 0.6397 | +0.0195 ✅ |
| Prob Min | 0.308 | 0.414 | +0.106 |
| Prob Max | 0.450 | 0.416 | -0.034 |
| **Prob Range** | **0.142** | **0.0025** | **-0.140** ❌ |
| Signals >0.55 | 0 | 0 | 0 ❌ |
| ECE | 0.0117 | 0.0055 | -0.0062 ✅ |

**Sigmoid made it WORSE:**
- Probability range collapsed from 0.142 to 0.0025 (98% reduction!)
- All predictions cluster at 0.415 (±0.001)
- AUC improved but probabilities completely collapsed

---

## Root Cause Analysis

### The Problem is NOT Calibration

Calibration methods (isotonic, sigmoid) can only **rescale** existing model outputs. They cannot create diversity where none exists.

**With aggressive_top5:**
1. Base LightGBM model produces **very similar raw probabilities** for all test samples
2. Only 5 features → limited input diversity
3. Aggressive regularization (depth=3, n_estimators=50) → limited model capacity
4. Result: raw outputs already clustered tightly

### Why Sigmoid Made It Worse

Platt scaling (sigmoid) fits a logistic function to map raw probs → calibrated probs:
```
P_calibrated = 1 / (1 + exp(a * P_raw + b))
```

If `P_raw` has low variance (0.40-0.45 range), then `P_calibrated` also has low variance.

In fact, sigmoid **compressed** the range further because:
- Logistic function is smooth (no discrete bins like isotonic)
- All inputs map to nearly the same output when variance is tiny

---

## Fundamental Issue: Model Has No Predictive Power on Dec 2025

Despite excellent AUC (0.62-0.64), the model **cannot separate** test samples:
- **Accuracy: 59%** (only slightly better than coin flip)
- **Precision on Target class: 0.00** (never predicts Target label)
- **Recall on Target class: 0.00** (misses all Target events)

The classifier is **degenerating to always predict Stop (class=0)**.

### Why High AUC but No Separation?

AUC measures **ranking** quality, not separation:
- AUC 0.64 means 64% of (target, stop) pairs are correctly ranked
- But if all probabilities are 0.41-0.42, you can't pick a threshold to separate them

---

## Conclusion

**aggressive_top5 model is fundamentally broken** for production use:
1. ✅ Overfitting eliminated (gap 0.009)
2. ✅ AUC improved (0.64 vs 0.51 baseline)
3. ❌ No output diversity → 0 actionable signals
4. ❌ Always predicts Stop class → 0% recall on Target
5. ❌ Sigmoid calibration made it worse

---

## Next Steps

### Option 5: Try conservative_top10 (RECOMMENDED)

From Phase 1 experiment grid:
```
conservative_top10:
  Median test AUC: ~0.52 (estimate, not in final output)
  Train-test gap: ~0.03
  Features: 10 (vs 5)
  Model: max_depth=4, n_estimators=100 (vs depth=3, n=50)
```

**Hypothesis:** 10 features + deeper model = more output diversity.

**Action:**
```bash
# Run Phase 2 on conservative_top10
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep
python -m ml_intraday_v3.experiments.run_anti_overfit_grid --phase 2 --best conservative_top10

# Train production model if walk-forward looks good
# Edit training_aggressive_top5.yaml to use conservative params + top-10 features
```

### Alternative: Disable Calibration (Option 1)

Test if raw LightGBM probabilities have more spread:
```yaml
# configs/training_aggressive_top5.yaml
calibration:
  enabled: false  # Use raw probabilities
```

**Expected:** Raw probs may have range 0.3-0.6 before calibration squashes them.

### Nuclear Option: Revert to Baseline

If conservative_top10 also fails:
- The experiment grid was tested on 6 walk-forward windows (Jan-Jun 2025)
- But Dec 2025 may be **out-of-distribution** (different market regime)
- Revert to baseline 34-feature model (AUC 0.51 but at least has some prob diversity)

---

## Files

1. **Isotonic model**: `ml_intraday_v3/models/saved/model_bundle_aggressive_top5.pkl` (AUC 0.62, range 0.142)
2. **Sigmoid model**: `ml_intraday_v3/models/saved/model_bundle_aggressive_top5_sigmoid.pkl` (AUC 0.64, range 0.0025)
3. **Training script with sigmoid support**: `ml_intraday_v3/train_balanced_model.py` (added SigmoidCalibratorWrapper class)

---

## Recommendation

**Stop pursuing aggressive_top5.** The model is too simple to be useful.

**Next:** Run conservative_top10 experiment to see if 10 features + less aggressive regularization produces actionable signals.
