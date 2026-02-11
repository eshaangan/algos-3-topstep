# CRITICAL: Feature Configuration Mismatch - Root Cause of NaN Issue

**Date**: February 4, 2026
**Severity**: 🚨 CRITICAL - System Completely Broken
**Status**: ROOT CAUSE IDENTIFIED

---

## Executive Summary

The identical predictions bug is **NOT caused by insufficient buffer size**.

**Root Cause**: The `features.yaml` configuration file is **missing 19 out of 34 features** that the model was trained on.

**Impact**: Model receives NaN for 56% of its expected features, regardless of buffer size.

**Fix Required**: Update `features.yaml` to generate all 34 features the model expects, OR retrain model on the 21 features that are actually being generated.

---

## Discovery Process

While testing buffer size impact, discovered that:

1. ✅ 100-bar buffer generates clean features (0 NaN) when features exist
2. ✅ 300-bar buffer also generates clean features (0 NaN) when features exist
3. ❌ **BUT** features.yaml only generates 21 features, model expects 34

---

## Feature Mismatch Analysis

### Model Expects (34 features):
```python
['log_return_1', 'log_return_2', 'log_return_4', 'log_return_6',
 'log_return_12', 'log_return_24', 'true_range', 'atr_14', 'vol_20',
 'vol_regime', 'parkinson_vol', 'vol_forecast', 'ema_13', 'ema_34',
 'ema_spread', 'ema_ratio', 'sma_20', 'sma_30', 'trend_strength',
 'autocorr_5', 'bb_position', 'volume_imbalance', 'price_vs_vwap',
 'relative_volume', 'large_move', 'candle_body', 'candle_range',
 'body_pct', 'upper_wick', 'lower_wick', 'minute_of_day_sin',
 'minute_of_day_cos', 'day_of_week', 'is_synthetic']
```

### features.yaml Generates (21 features):
```python
['log_return_1', 'log_return_4', 'true_range', 'atr_14', 'vol_20',
 'vol_regime', 'parkinson_vol', 'vol_forecast', 'ema_13', 'ema_21',
 'ema_spread', 'ema_ratio', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
 'minute_of_day_sin', 'minute_of_day_cos', 'day_of_week', 'is_synthetic',
 'usable_for_training']
```

### Missing Features (19):
1. **Multi-bar returns** (5): `log_return_2`, `log_return_6`, `log_return_12`, `log_return_24`
2. **Moving averages** (3): `ema_34`, `sma_20`, `sma_30`
3. **Trend indicators** (2): `trend_strength`, `autocorr_5`
4. **Bollinger Bands** (1): `bb_position`
5. **Volume indicators** (3): `volume_imbalance`, `price_vs_vwap`, `relative_volume`
6. **Candle patterns** (5): `large_move`, `candle_body`, `candle_range`, `body_pct`, `upper_wick`, `lower_wick`

### Extra Features (6 - not in model):
- `ema_21`, `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `usable_for_training`

---

## Why This Causes Identical Predictions

```python
# What happens in LiveModelPredictor:

# 1. Model expects 34 features
model_features = ['log_return_1', 'log_return_2', ..., 'lower_wick']

# 2. Feature builder generates only 21
generated_features = build_features(bars, config=features_yaml)
# -> Only has: ['log_return_1', 'log_return_4', 'ema_13', ...]
# -> Missing: ['log_return_2', 'log_return_6', ..., 'lower_wick']

# 3. Try to create feature matrix
X = generated_features[model_features]
# -> KeyError! Missing columns!

# 4. Preprocessing fills NaN with median
X_with_nan = ... # 19 columns are NaN
X_imputed = X_with_nan.fillna(median_values)
# -> 19/34 features become CONSTANTS (median values)

# 5. Model receives identical input every time
model.predict(X_imputed)
# -> IDENTICAL PREDICTIONS: [0.457, 0.543]
```

---

## Evidence from Live Trading Logs

```
2026-02-04 19:12:39 Feature quality check:
{
  'nan_count': 19,
  'nan_columns': [
    'log_return_2', 'log_return_6', 'log_return_12', 'log_return_24',
    'ema_34', 'sma_20', 'sma_30', 'trend_strength', 'autocorr_5',
    'bb_position', 'volume_imbalance', 'price_vs_vwap', 'relative_volume',
    'large_move', 'candle_body', 'candle_range', 'body_pct',
    'upper_wick', 'lower_wick'
  ],
  'healthy': False
}
```

These are EXACTLY the 19 features missing from features.yaml!

---

## Why Buffer Size Didn't Matter

- ❌ **Initial Hypothesis**: 100 bars insufficient → features have NaN → need 300 bars
- ✅ **Actual Reality**: Features not generated at all → always NaN → buffer size irrelevant

The 300-bar fix would NOT have solved anything because:
- Missing features would still be missing with 300 bars
- NaN would persist regardless of buffer size
- Predictions would remain identical

---

## How This Happened

**Likely Scenario**: Model was trained with one features.yaml configuration, but the current features.yaml is different (either reverted to old version or manually edited).

**Timeline**:
1. Model trained in Oct-Nov 2025 with 34 features
2. features.yaml was modified (intentionally or accidentally)
3. Live trading started using modified features.yaml
4. Model expects 34 features, gets 21 + 19 NaN
5. Median imputation converts NaN to constants
6. Model produces identical predictions

---

## Solution Options

### Option A: Update features.yaml (RECOMMENDED)

Add missing feature generators to features.yaml:

**Files to modify**:
- `ml_intraday_v3/configs/features.yaml`

**Changes needed**:
```yaml
returns:
  enable_multi_horizon: true
  multi_horizon_bars: [2, 6, 12, 24]  # Add 2, 6 to existing 12, 24

moving_averages:
  enable_ema: true
  ema_spans: [13, 34]  # Add 34
  enable_sma: true
  sma_windows: [20, 30]  # Add SMA calculation

trend:
  enable_trend_strength: true
  enable_autocorr: true
  autocorr_lags: [5]

bollinger_bands:
  enable: true
  window: 20
  num_std: 2

volume:
  enable_volume_imbalance: true
  enable_vwap: true
  enable_relative_volume: true

candle_patterns:
  enable_large_move: true
  enable_body_metrics: true
  enable_wick_metrics: true
```

**Trade-offs**:
- ✅ Matches model training exactly
- ✅ Model works as designed
- ❌ Need to test feature calculations are correct
- ❌ May take 2-4 hours to implement and validate

---

### Option B: Retrain Model on 21 Features

Retrain model using only the 21 features that features.yaml actually generates.

**Files to modify**:
- Re-run `train_balanced_model.py` with current features.yaml
- Generate new model_bundle.pkl

**Trade-offs**:
- ✅ Guaranteed feature alignment
- ✅ No config changes needed
- ❌ Loses 19 potentially valuable features
- ❌ Need to re-validate performance (may be worse)
- ❌ Training takes ~1-2 hours

---

### Option C: Quick Fix - Disable Feature Quality Check (TEMPORARY)

Keep current setup, disable feature quality check, let median imputation handle NaN.

**Already Done**:
- `check_feature_quality: false` in live_trading.yaml

**Trade-offs**:
- ✅ System runs (already deployed)
- ❌ Model degraded (19/34 features are constants)
- ❌ Predictions likely suboptimal
- ❌ Not a real fix, just a workaround

**Status**: This is what's currently running in production! 🚨

---

## Recommended Action Plan

### Immediate (Next 2 hours)

1. **Verify Training Configuration**
   ```bash
   # Find the original features.yaml used for training
   cd runs/
   find . -name "features.yaml" | grep "oct.*nov"
   ```

2. **Compare Configurations**
   - Training features.yaml vs current features.yaml
   - Identify what changed and when

3. **Decision Point**:
   - If training config found → restore it (Option A)
   - If training config lost → retrain model (Option B)
   - If neither feasible → document degradation (Option C, current state)

### Short-Term (Next 24 hours)

4. **If Option A (Restore Config)**:
   - Copy training features.yaml to configs/
   - Test feature generation on sample data
   - Verify all 34 features generate correctly
   - Redeploy to GCP
   - Validate predictions vary

5. **If Option B (Retrain)**:
   - Backup current model
   - Run training with current features.yaml
   - Validate new model on Dec 2025 / Jan 2026
   - If performance acceptable → deploy
   - If performance degraded → revert to Option A

### Long-Term (Next Week)

6. **Prevent Future Mismatches**:
   - Add feature validation in model loading
   - Fail fast if features don't match
   - Version-lock features.yaml with model bundles
   - Add unit tests for feature alignment

---

## Critical Questions to Answer

1. **Where is the original features.yaml from training?**
   - Check `runs/` directories from Oct-Nov 2025
   - Check git history: `git log --all -- ml_intraday_v3/configs/features.yaml`

2. **When did features.yaml change?**
   - `git diff HEAD~10 ml_intraday_v3/configs/features.yaml`
   - Check if it was intentional or accidental

3. **Can we recover training configuration?**
   - Model bundle metadata
   - Training run directories
   - Git history

4. **What is acceptable performance degradation?**
   - If retraining with 21 features, what's minimum acceptable win rate?
   - Can we afford to lose predictive power from 19 features?

---

## Current Production Status

**System Status**: 🚨 DEGRADED - Running with 56% constant features

**What's Deployed**:
- Model expecting 34 features
- features.yaml generating 21 features
- 19 features filled with median constants
- Predictions likely suboptimal but system functional

**Risk Level**: MEDIUM
- System won't crash
- May generate some trades (if predictions vary enough)
- Performance likely worse than validation baseline
- Win rate / P&L may be significantly degraded

---

## Validation Results Update

**Previous Hypothesis**: Buffer size causes NaN → test 100 vs 300 bars
**Actual Finding**: Feature config mismatch → buffer size irrelevant

**Jan 2026 validation NOT run** because:
1. No point testing buffer size when features are missing
2. Need to fix feature config first
3. Then re-validate with correct features

**Next Steps**:
1. Fix feature configuration (Option A or B)
2. THEN re-run Jan 2026 validation
3. Compare against baseline with correct features

---

## Files for Investigation

**Check these files**:
```bash
# Find training run directory
ls -la runs/ | grep "oct.*nov\|2024\|2025"

# Check git history
git log --all --full-history -- ml_intraday_v3/configs/features.yaml

# Find model training script runs
ls -la runs/*/features.yaml

# Check model metadata
python -c "import joblib; print(joblib.load('ml_intraday_v3/model_bundle_retrained_oct2024_nov2025.pkl')['metadata'])"
```

---

## Summary

❌ **Buffer size fix (100→300) will NOT solve the problem**
✅ **Feature configuration mismatch is the root cause**
🚨 **System is running degraded in production right now**
⚡ **Need to restore correct features.yaml or retrain model ASAP**

**Critical Next Action**: Locate original training features.yaml and restore it.

---

**Last Updated**: February 4, 2026 13:36 UTC
**Discovered By**: Validation testing for buffer size impact
**Priority**: CRITICAL - Affects all predictions
**ETA to Fix**: 2-4 hours (Option A) or 1-2 hours (Option B)
