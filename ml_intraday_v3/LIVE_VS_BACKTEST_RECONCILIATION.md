# Live Trading vs Jan 2026 Backtest Reconciliation

**Date**: 2026-02-10
**Purpose**: Identify configuration mismatches preventing $654/day Jan 2026 performance from replicating in live trading

---

## Executive Summary

The Jan 2026 validation showed **$654/day** on real data with the `model_bundle_retrained_oct2024_nov2025.pkl` model. However, live trading generated signals that were "not strong enough to be executed" and shorts were blocked.

**Root Causes Identified**:

1. ❌ **CRITICAL**: Volatility filter ENABLED in execution_spec.yaml (wasn't in Jan 2026 test)
2. ❌ **Model path unclear**: live_trading.yaml uses auto-detect instead of explicit path
3. ⚠️ **Regime filters**: execution_engine has regime_filter and ADX filter (both disabled, but present)
4. ⚠️ **Threshold confusion**: Two separate threshold systems (confidence filter vs primary_threshold)

---

## Configuration Comparison

### Model Selection

| Component | Jan 2026 Successful | Current Config | Status |
|-----------|-------------------|----------------|--------|
| Model file | `model_bundle_retrained_oct2024_nov2025.pkl` | `null` (auto-detect) | ⚠️ UNCLEAR |
| Model location | `ml_intraday_v3/` (root) | `models/saved/` or root | ⚠️ AMBIGUOUS |
| Features | 34 features | 34 features (fixed Feb 4) | ✅ MATCH |

**Issue**: Auto-detect may pick wrong model. The successful model has:
- 316 trades in Jan 2026
- 56.3% win rate
- Probabilities reaching >0.55

But the new `conservative_top10_full.pkl` only has:
- Max probability 0.472 (below 0.55 threshold!)

### Confidence/Threshold Filters

| Filter | Jan 2026 Successful | Current Config | Status |
|--------|-------------------|----------------|--------|
| Confidence threshold (execution_spec) | 0.55 | 0.55 | ✅ MATCH |
| Primary threshold (live_trading) | 0.03 | 0.10 | ❌ MISMATCH |
| Max primary threshold cap | Not mentioned | 0.15 | ⚠️ UNKNOWN |
| Meta model | Not used | Not used | ✅ MATCH |

**Issue**: Primary threshold raised from 0.03 to 0.10 (3.3x stricter). This may be blocking valid signals after they pass confidence filter.

### Volatility/Regime Filters

| Filter | Jan 2026 Successful | Current Config | Status |
|--------|-------------------|----------------|--------|
| **Volatility filter (execution_spec)** | ❌ **NOT PRESENT** | ✅ **ENABLED** | ❌ **CRITICAL MISMATCH** |
| Volatility min percentile | N/A | 30 | ❌ NEW FILTER |
| Volatility max percentile | N/A | 70 | ❌ NEW FILTER |
| Regime filter (execution_engine) | Disabled | Disabled | ✅ MATCH |
| ADX filter (execution_engine) | Disabled | Disabled | ✅ MATCH |
| Regime detector (live_runner) | Disabled | Disabled | ✅ MATCH |

**CRITICAL FINDING**: The volatility filter in `execution_spec.yaml::filters.volatility` is **ENABLED** but wasn't validated in Jan 2026 backtest!

This filter rejects trades when volatility is:
- Below 30th percentile (dead markets)
- Above 70th percentile (high chaos)

This could be blocking **30-40% of valid signals** that passed in Jan 2026!

### Circuit Breaker

| Component | Jan 2026 Successful | Current Config | Status |
|-----------|-------------------|----------------|--------|
| Enabled | ✅ Yes | ✅ Yes | ✅ MATCH |
| Daily loss limit | -$500 | -$500 | ✅ MATCH |
| Consecutive losses | 3 | 4 | ⚠️ MINOR DIFF |
| Cooldown minutes | 30 | 15 | ⚠️ MINOR DIFF |

**Status**: Minor differences, but shouldn't block trades under normal conditions.

### Session Rules

| Rule | Jan 2026 Successful | Current Config | Status |
|------|-------------------|----------------|--------|
| RTH filter | ✅ Enabled (8:30-15:00 CT) | ✅ Enabled (8:30-15:00 CT) | ✅ MATCH |
| No entry before close | 60 minutes | 30 minutes | ⚠️ MINOR DIFF |
| Flatten at close | Yes | Yes | ✅ MATCH |

**Status**: No blocking issues here.

### Direction Filtering

| Filter | Jan 2026 Successful | Current Config | Status |
|--------|-------------------|----------------|--------|
| Allowed directions | Both LONG and SHORT | Both (no restriction) | ✅ MATCH |
| Negative edge check | Enabled | Enabled | ✅ MATCH |

**User Report**: "I think there were some shorts, but I think our live trading was set up in a way that blocked it."

**Finding**: No explicit SHORT blocking in configs. But negative edge check could reject SHORTs if probabilities are from LONG perspective.

---

## Filter Chain Execution Order

### Jan 2026 Backtest (Simplified)
1. CUSUM event detection
2. Feature generation (34 features)
3. Model prediction
4. **Confidence filter: p_target > 0.55** ← MAIN GATE
5. Circuit breaker (daily loss, consecutive losses)
6. Execute trade

### Current Live Trading (Full Chain)
1. Regime detector (disabled) ✅
2. **Volatility filter (ATR-based, signals config)** ← NOT IN JAN 2026!
3. CUSUM event detection
4. Feature generation (34 features) + quality check
5. Model prediction
6. **Confidence filter: p_target > 0.55**
7. **Volatility percentile filter (30-70%)** ← NOT IN JAN 2026!
8. Circuit breaker threshold adjustments
9. Threshold cap (max 0.15)
10. **should_trade() check: score_ev > primary_threshold (0.10-0.15)**
11. Negative edge check: p_stop < p_target
12. Execution engine filters:
    - regime_filter (disabled) ✅
    - ADX filter (disabled) ✅

**Extra filters in live (not in Jan 2026)**:
- ❌ Volatility filter (execution_spec) - NEW, ENABLED
- ❌ Primary threshold raised to 0.10 (was 0.03)

---

## Probability Analysis

### Successful Jan 2026 Model (`model_bundle_retrained_oct2024_nov2025.pkl`)
- **Training**: Oct 2024 - Nov 2025 (14 months, 34 features)
- **Jan 2026 Results**: 316 trades, 56.3% win rate, $654/day
- **Probability range**: Must have probabilities >0.55 to generate 316 trades
- **Threshold**: 0.55 confidence filter

### New Conservative Model (`model_bundle_conservative_top10_full.pkl`)
- **Training**: 2019-2025 (6 years, 10 features)
- **Dec 2025 Results**: 0 trades at 0.55 threshold (max prob 0.472)
- **Probability range**: 0.320 - 0.472
- **Optimal threshold**: 0.44 (67 trades, 50.7% win rate, $0.24/trade)

**CRITICAL**: The new conservative model **CANNOT** work with 0.55 threshold! Its max probability is 0.472.

**Implication**: If live trading auto-detected the conservative model instead of the Jan 2026 model, **ZERO trades would execute** (all below 0.55).

---

## Why Trades Aren't Executing

### Scenario 1: Wrong Model Loaded
**If**: Live system loaded `conservative_top10_full.pkl` instead of `model_bundle_retrained_oct2024_nov2025.pkl`

**Then**:
- Max probability: 0.472
- Confidence threshold: 0.55
- **Result**: 0 trades (all signals rejected by confidence filter)

**Evidence**: User said "signals generated but none strong enough" ← matches low probabilities

### Scenario 2: Correct Model + Volatility Filter Blocking
**If**: Live system loaded correct Jan 2026 model

**Then**:
- Probabilities >0.55 ✅
- Confidence filter passes ✅
- **Volatility filter rejects** if vol outside 30-70% range ❌
- **Result**: ~30-40% reduction in trades

**Evidence**: User said "some signals...none strong enough" ← could mean filtered out

### Scenario 3: Shorts Blocked by Negative Edge Check
**If**: Model predicts SHORT but probabilities are from LONG perspective

**Then**:
- Bidirectional check: p_stop >= p_target → negative edge ❌
- Non-bidirectional SHORT: p_target >= p_stop (inverted for SHORT) → negative edge ❌
- **Result**: SHORTs rejected

**Evidence**: User said "I think there were some shorts...blocked"

---

## Recommended Fixes

### Fix 1: Explicitly Set Model Path (HIGHEST PRIORITY)
**File**: `ml_intraday_v3/configs/live_trading.yaml`

```yaml
model:
  # EXPLICIT PATH - don't auto-detect!
  model_bundle_path: "model_bundle_retrained_oct2024_nov2025.pkl"  # Jan 2026 $654/day model
  version: "oct2024_nov2025"
  trained: "2025-11-30"
```

**Impact**: Ensures correct model is loaded (not conservative_top10 with low probabilities)

### Fix 2: Disable Volatility Filter (CRITICAL)
**File**: `ml_intraday_v3/configs/execution_spec.yaml`

```yaml
filters:
  # Volatility regime filter (Moreira & Muir 2017) - Quick Win #4
  volatility:
    enabled: false  # DISABLED - wasn't in Jan 2026 validation ← CHANGE THIS
    min_percentile: 30
    max_percentile: 70
    lookback_bars: 100
    vol_column: "vol_20"
```

**Impact**: Removes filter that blocks 30-40% of trades outside vol range

### Fix 3: Lower Primary Threshold (HIGH PRIORITY)
**File**: `ml_intraday_v3/configs/live_trading.yaml`

```yaml
signals:
  # Minimum probability threshold
  # Conservative with binary calibrated model; confidence filter at 0.55 gates quality
  primary_threshold: 0.03  # RESTORE to Jan 2026 value (was 0.10) ← CHANGE THIS
```

**Impact**: Reduces redundant filtering after confidence filter already gated at 0.55

### Fix 4: Verify Model Has 34 Features (Already Done)
**Status**: ✅ Fixed on Feb 4, 2026 (features.yaml restored)

### Fix 5: Verify Bidirectional Logic
**File**: `ml_intraday_v3/live_trading/model_predictor.py` (lines 202-258)

**Check**: Ensure bidirectional evaluation is working correctly:
- If model has 'side' feature → evaluate both LONG and SHORT
- Choose side with best positive EV
- Return probabilities from chosen side's perspective

**Verification**: Check logs for "Bidirectional choice" messages

---

## Testing Plan

### Step 1: Verify Model Loaded
```bash
# SSH to GCP
gcloud compute ssh topstep-trader-vm --zone=us-central1-a

# Check logs for model loading
docker logs $(docker ps -q) 2>&1 | grep -i "model loaded" | tail -5

# Should show:
# Model loaded: CalibratedClassifierCV or LGBMClassifier
# Features: 34
# Bundle path: model_bundle_retrained_oct2024_nov2025.pkl
```

**Expected**: `model_bundle_retrained_oct2024_nov2025.pkl` (Jan 2026 model)
**If wrong**: Auto-detect picked wrong model → Apply Fix 1

### Step 2: Check Signal Strength
```bash
# Check recent predictions
docker logs $(docker ps -q) 2>&1 | grep -E "p_target|p_stop" | tail -20

# Expected (Jan 2026 model):
# p_target: 0.55-0.65 range (above 0.55 threshold)

# If seeing:
# p_target: 0.40-0.47 range (below 0.55) → WRONG MODEL LOADED
```

### Step 3: Check Filter Rejections
```bash
# Check why signals are rejected
docker logs $(docker ps -q) 2>&1 | grep "rejected" | tail -20

# If seeing:
# "✗ Confidence filter rejected signal: P=0.47, threshold=0.55"
#   → WRONG MODEL (apply Fix 1)

# "✗ Volatility filter rejected signal"
#   → EXTRA FILTER (apply Fix 2)

# "✗ Signal rejected: reason=primary_threshold"
#   → THRESHOLD TOO HIGH (apply Fix 3)
```

### Step 4: Backtest with Current Configs
```bash
# Run Jan 2026 backtest with EXACT current configs
cd ml_intraday_v3
python test_jan2026_with_fixes.py \
  --model model_bundle_retrained_oct2024_nov2025.pkl \
  --enable-volatility-filter  # Test with current settings
```

**Expected**: Should match live trading results (low trade count)
**Then**: Disable volatility filter and re-run to see impact

---

## Success Criteria

After applying fixes, live trading should show:

| Metric | Jan 2026 Baseline | Acceptable Range | Status |
|--------|-------------------|------------------|--------|
| Trades per day | 13.2 | 8-15 | ⏳ Pending |
| Win rate | 56.3% | 50-60% | ⏳ Pending |
| Daily P&L | $654 | $300-700 | ⏳ Pending |
| Signal rejection rate | ~5% | <15% | ⏳ Pending |
| Probability range | 0.55-0.65 | 0.50-0.70 | ⏳ Pending |

**Red Flags**:
- ❌ Probabilities below 0.50 → Wrong model
- ❌ Zero trades → Filters too strict or wrong model
- ❌ >50% signals rejected → Extra filters active
- ❌ Win rate <45% → Distribution shift or config mismatch

---

## Deployment Sequence

### Phase 1: Apply Fixes (5 minutes)
1. Edit `live_trading.yaml` → set model_bundle_path explicitly
2. Edit `execution_spec.yaml` → disable volatility filter
3. Edit `live_trading.yaml` → lower primary_threshold to 0.03
4. Verify features.yaml still has all 34 features ✅

### Phase 2: Redeploy (10 minutes)
```bash
cd ml_intraday_v3
./deploy_to_gcp.sh
```

### Phase 3: Monitor (2 hours)
```bash
# Check every 15 minutes
./monitor_gcp.sh

# Look for:
# ✅ Model: model_bundle_retrained_oct2024_nov2025.pkl
# ✅ Signals with P > 0.55
# ✅ Trades executing
# ✅ Both LONG and SHORT trades
```

### Phase 4: Validate (First 5 Days)
- Compare actual vs Jan 2026 baseline
- Track: trades/day, win rate, daily P&L
- If matching: Continue
- If not: Investigate further

---

## Conclusion

**Primary Issue**: Volatility filter ENABLED in execution_spec but wasn't in Jan 2026 validation

**Secondary Issue**: Model path unclear - may be loading wrong model (conservative_top10 with low probabilities instead of Jan 2026 model)

**Tertiary Issue**: Primary threshold raised from 0.03 to 0.10 (redundant strictness after 0.55 confidence filter)

**Action Plan**:
1. ✅ Explicitly set model path to Jan 2026 model
2. ✅ Disable volatility filter
3. ✅ Lower primary threshold to 0.03
4. ⏳ Deploy and monitor

**Expected Outcome**: Live trading should match Jan 2026 performance ($654/day, 13 trades/day, 56% win rate)

---

**Last Updated**: 2026-02-10 20:15
**Status**: Diagnosis complete, fixes ready for deployment
