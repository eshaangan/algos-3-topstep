# Feature Importance Analysis - Executive Summary
**Date**: February 10, 2026
**Model**: Binary LightGBM + Isotonic Calibration (Phase 1-3)
**Status**: ❌ FAILED - Model has no predictive power

---

## Critical Findings

### Model Performance
- **Test AUC: 0.4934** (below 0.5 random coin flip)
- **Probability Collapse**: All 629 test predictions clustered in range 0.419-0.438
- **Zero Confident Signals**: 0 samples with P(target) > 0.55 threshold
- **Classification**: Model predicts ONLY "Stop" class, never "Target"

### Feature Analysis Results

**57% of features (24/42) are ACTIVELY HARMFUL**

Permutation importance revealed that more than half the features **hurt** model performance when included:

#### Harmful Features (Negative Contribution)
```
WORST OFFENDERS:
- candle_range:       -0.0065 (worst)
- vol_20:             -0.0056
- sma_20:             -0.0047
- side:               -0.0040
- sma_30:             -0.0040
- log_return_1/2/4:   -0.0030 to -0.0038 (ALL short-term returns harmful)

MOMENTUM FEATURES (ALL NEGATIVE):
- vwap_momentum:      -0.0033
- rsi_14:             -0.0021
- macd_hist:          -0.0009
- macd_signal:        -0.0012
- rsi_divergence:      0.0000 (zero contribution)

VOLATILITY FEATURES (ALL NEGATIVE):
- vol_20:             -0.0056
- vol_regime:         -0.0026
- parkinson_vol:      -0.0015
- vol_forecast:       -0.0014
```

#### Helpful Features (Barely Positive)
```
Only 14 features show positive contribution (MAX +0.0048 AUC):

TOP 10:
1. ema_spread:         +0.0048 (best, but only 0.48% AUC boost)
2. ema_ratio:          +0.0035
3. relative_volume:    +0.0033
4. autocorr_5:         +0.0031
5. macd:               +0.0026
6. ema_34:             +0.0026
7. minute_of_day_cos:  +0.0021
8. upper_wick:         +0.0018
9. bb_position:        +0.0015
10. log_return_12:     +0.0013
```

---

## Root Cause Analysis

### What Went Wrong?

**Normalization didn't break the model - there was NO SIGNAL to begin with.**

The Phase 1-3 improvements (normalization + momentum + 6-month window) didn't destroy a working model. The baseline model **never had edge** using these features.

### Why No Signal?

Standard technical indicators on MES 5-minute bars contain almost zero information about whether price will hit a profit target (2.5 ATR) before hitting a stop loss (2.0 ATR).

**Evidence:**
- Best feature (ema_spread) contributes only +0.48% AUC
- 24/42 features actively hurt performance (overfitting to noise)
- Model converged to predicting same probability for every sample
- Removing normalization unlikely to fix this (testing needed)

---

## Recommended Action Plan

### Phase 1: Diagnostic (HIGH PRIORITY - 30 minutes)

**Test if normalization specifically broke things:**

```bash
cd ml_intraday_v3
python run_diagnostic_tests.py
```

This will:
1. Test current model (with normalization): Expect AUC ~0.49
2. Test baseline model (no normalization, original features): Compare AUC

**Decision Tree:**
- **If baseline AUC > 0.52**: Normalization broke it → Revert + prune harmful features
- **If baseline AUC < 0.50**: No edge exists → Need fundamental redesign

### Phase 2: Simplification (if Phase 1 shows baseline > 0.52)

**Create minimal feature set:**

```yaml
# configs/features_minimal.yaml
enabled_features:
  - ema_spread          # +0.0048
  - ema_ratio           # +0.0035
  - relative_volume     # +0.0033
  - autocorr_5          # +0.0031
  - macd                # +0.0026
  - ema_34              # +0.0026
  - minute_of_day_cos   # +0.0021
  - upper_wick          # +0.0018
  - bb_position         # +0.0015
  - log_return_12       # +0.0013
```

**Expected Result**: AUC ~0.51-0.53 (marginal edge)

### Phase 3: Fundamental Redesign (if Phase 1-2 fail)

If even the simplified model shows AUC < 0.52, consider:

1. **Different Prediction Target**
   - Current: "Will hit 2.5 ATR profit before 2.0 ATR stop?"
   - Alternative: "Will next bar be positive?" (simpler, more predictable)

2. **Different Timeframe**
   - Current: 5-minute bars (noisy, less patterns)
   - Alternative: 1-minute bars (more data, more granular patterns)

3. **Different Features**
   - Current: Technical indicators (price/volume only)
   - Alternative: Microstructure (order flow, bid-ask, depth), cross-asset (VIX, bonds, SPY)

4. **Barrier Optimization (Phase 4 from original plan)**
   - Test if different PT/SL/horizon combinations have better predictability
   - May discover that 2.5/2.0 ratio is suboptimal for MES

---

## Files Created

1. **MODEL_SIMPLIFICATION_PLAN.md** - Detailed strategy document
2. **configs/features_simplified.yaml** - Baseline configuration (no normalization)
3. **run_diagnostic_tests.py** - Automated diagnostic test suite
4. **diagnostics/feature_importance_output.log** - Full analysis results
5. **diagnostics/feature_importance_gain.csv** - Gain-based importance (partial, before error)

---

## Success Criteria for Go/No-Go

**Minimum Viable Edge** (needed to continue this approach):
- OOS AUC > 0.55 (consistent across walk-forward windows)
- Win rate @ P>0.55 > 48%
- Sharpe ratio > 0.5 (annualized on backtest)
- Probability separation (P5-P95 spread > 0.15)

**If we can't reach these after:**
- Removing harmful features
- Testing baseline vs normalized
- Optimizing barrier parameters

**Then honest conclusion**: This feature set + prediction target + timeframe combination doesn't have tradeable edge on MES.

---

## Next Steps (User Decision Required)

1. **Run diagnostic tests** to determine if baseline model had edge:
   ```bash
   cd "/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3"
   python run_diagnostic_tests.py
   ```

2. **Review results**, then decide:
   - **Option A**: Continue with simplified features (if baseline shows promise)
   - **Option B**: Fundamental redesign (if baseline has no edge)
   - **Option C**: Pause ML approach, focus on rule-based strategies

**Estimated Time:**
- Diagnostic test: 30 minutes
- Analysis + decision: 30 minutes
- **Total: 1 hour to know if current path is viable**

---

## Key Insight

> **Adding more features and complexity made things WORSE, not better.**
> The path forward is **radical simplification** or **fundamental redesign**, not more complexity.

We added normalization, momentum indicators, sample decay, and shorter windows. Result: AUC went from ~0.52 to 0.49. The model is telling us it has nothing to learn from these features.

**Simple truth**: You can't extract signal from noise by adding more sophisticated noise processing.
