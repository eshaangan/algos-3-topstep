# Model Simplification Plan - February 10, 2026

## Problem Diagnosis

Feature importance analysis on the Phase 1-3 model (42 features, normalized + momentum) revealed:

- **Test AUC: 0.4934** (below 0.5 random baseline)
- **24/42 features (57%) are harmful** (negative permutation importance)
- **Only 14 features show positive contribution**, and the strongest adds only +0.0048 AUC
- **Probability collapse**: All predictions in narrow range (0.419-0.438)

### Root Cause

The normalization and momentum features didn't destroy signal - **there was no meaningful signal in the feature set to begin with**. We're trying to predict stop-vs-target outcomes using features that don't contain relevant information about that outcome.

## Simplification Strategy

### Option 1: Minimal Feature Set (RECOMMENDED)

Keep only the **top 10 features** with clearly positive permutation importance:

```yaml
features:
  enabled:
    - ema_spread         # +0.0048 (EMA 13-34 distance)
    - ema_ratio          # +0.0035 (EMA 13/34 ratio)
    - relative_volume    # +0.0033 (volume vs 20-bar avg)
    - autocorr_5         # +0.0031 (5-bar price autocorrelation)
    - macd               # +0.0026 (trend momentum)
    - ema_34             # +0.0026 (longer-term trend)
    - minute_of_day_cos  # +0.0021 (time of day cyclical)
    - upper_wick         # +0.0018 (rejection at highs)
    - bb_position        # +0.0015 (relative position in BB)
    - log_return_12      # +0.0013 (1-hour momentum)
```

**Expected Result**: Likely still AUC ~0.51-0.52 (marginal improvement over coin flip)

### Option 2: Ultra-Minimal Baseline (5 features)

Test if **ANY edge exists** using only the strongest signals:

```yaml
features:
  enabled:
    - ema_spread         # Trend structure
    - ema_ratio          # Trend slope
    - relative_volume    # Volume conviction
    - autocorr_5         # Mean reversion
    - minute_of_day_cos  # Time effects
```

**Expected Result**: AUC ~0.50-0.51 (pure baseline test)

### Option 3: Revert to Original Pre-Normalization (DIAGNOSTIC)

Test if normalization specifically caused the collapse:

```yaml
# Revert features.yaml to state BEFORE Phase 1-2
normalization:
  enabled: false
momentum:
  enabled: false

# Use original 36 features without modifications
```

Train on **14-month window** (original) instead of 6-month to isolate whether shorter window contributed to failure.

**Expected Result**: If AUC returns to ~0.52-0.54, then normalization was the problem. If AUC stays ~0.49, then the baseline model never had edge.

## Fundamental Question

**Does ANY feature set predict stop-vs-target on MES 5m bars?**

The permutation analysis suggests the answer might be **NO** - at least not with standard technical indicators. The tiny contributions (max +0.0048 AUC) are statistically indistinguishable from noise.

### Alternative Hypotheses

1. **Label Quality Issue**: Triple barrier labels with PT=2.5, SL=2.0, Hz=12-24 may not reflect tradeable outcomes
2. **Signal-to-Noise Too Low**: 5-minute bars may be too noisy; need 1-minute bars or tick data
3. **Missing Critical Features**: Need microstructure features (bid-ask spread, order flow, depth) or cross-asset features (VIX, bonds, SPY correlation)
4. **Wrong Prediction Target**: Predicting "will hit PT before SL" may be fundamentally unpredictable; consider simpler targets like "next bar up/down" or "next 3-bar max return"

## Recommended Next Steps

### Step 1: Baseline Diagnostic (30 minutes)

1. Revert features.yaml to pre-Phase-1 state (disable normalization + momentum)
2. Revert training.yaml to 14-month window
3. Retrain model
4. Compare AUC to current 0.4934

**If AUC recovers to 0.52+**: Normalization broke things, use Option 1 (10 features, no normalization)
**If AUC stays below 0.50**: Fundamental signal problem, proceed to Step 2

### Step 2: Test Minimal Feature Set (30 minutes)

1. Implement Option 2 (5 features only)
2. Train with NO normalization, NO sample decay
3. Use both 6-month and 14-month windows

**If AUC reaches 0.52+**: We have minimal edge, proceed to Phase 4 (barrier optimization)
**If AUC < 0.50**: No edge exists with current approach, proceed to Step 3

### Step 3: Fundamental Redesign (if Steps 1-2 fail)

Consider:
- **Different labels**: Switch from triple-barrier to simpler next-bar direction
- **Different timeframe**: Try 1-minute bars (more data, more patterns)
- **Different features**: Add order flow, microstructure, cross-asset correlations
- **Different model**: Try XGBoost, neural networks, or ensemble methods
- **Accept reality**: MES 5m may not be predictable with public technical indicators alone

## Files to Modify

### Immediate (Diagnostic):

1. **configs/features.yaml**
   - Set `normalization.enabled: false`
   - Set `momentum.enabled: false`
   - Remove new momentum features

2. **configs/training.yaml**
   - Change train_start back to 2024-01-01 (14 months)
   - Set `sample_decay.enabled: false`

3. **Retrain and compare**

### If Moving Forward with Simplification:

1. **configs/features.yaml**
   - Create new `feature_selection` section
   - List only the 10 features from Option 1
   - Add logic to build_features() to respect feature_selection

## Success Criteria

**Minimum Viable Edge**:
- OOS AUC > 0.55 (consistent across walk-forward windows)
- Win rate @ P>0.55 > 48%
- Sharpe ratio > 0.5 (annualized)

**If we can't reach these minimums** after:
1. Removing harmful features
2. Testing multiple train window sizes
3. Optimizing barrier parameters (Phase 4)

Then the honest conclusion is: **This feature set + prediction target + timeframe combination doesn't have tradeable edge on MES.**

## Timeline

- **Step 1 (Diagnostic)**: 30 mins - Run tonight
- **Step 2 (Minimal)**: 30 mins - If Step 1 confirms baseline was better
- **Step 3 (Decision)**: 1 hour - Review results, decide go/no-go on current approach

Total: 2 hours to know if this path has any viability.
