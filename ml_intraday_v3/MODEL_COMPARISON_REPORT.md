# Model Comparison Report - Jan 25, 2026

## Test Metrics Comparison

### Baseline Model (OLD) - Jan 4-23, 2026 Backtest
From: `backtest_results/databento_validation_20260125_000415/analysis_summary.json`

| Metric | Baseline | High Confidence | Aggressive |
|--------|----------|-----------------|------------|
| **Total Trades** | 129 | 129 | 129 |
| **Win Rate %** | 3.10% | 3.10% | 3.10% |
| **Profit Factor** | 0.028 | 0.028 | 0.028 |
| **Stop Hit Rate %** | 96.90% | 96.90% | 96.90% |
| **Target Hit Rate %** | 3.10% | 3.10% | 3.10% |
| **Total P&L** | -$2,368.71 | -$2,368.71 | -$2,368.71 |
| **Avg Win** | $17.20 | $17.20 | $17.20 |
| **Avg Loss** | -$19.50 | -$19.50 | -$19.50 |
| **LONG %** | **100.0%** | **100.0%** | **100.0%** |
| **SHORT %** | **0.0%** | **0.0%** | **0.0%** |

**Critical Issues**:
- ❌ 100% LONG bias (0% SHORT) - Bug in replay.py
- ❌ Only 3.1% win rate (96.9% stop-outs)
- ❌ Lost $2,368 in 3 weeks
- ❌ Essentially gambling against the model

---

### Simple Retrained Model - Dec 2025 Test
From: `retrain_output.log`

**Training**: Oct 2024 - Nov 2025 (22,924 samples)
**Testing**: Dec 2025 (1,091 samples)

| Metric | Train | Test |
|--------|-------|------|
| **AUC** | 0.6682 | 0.4959 |
| **Accuracy** | 0.6131 | 0.5078 |
| **has_side_feature** | ❌ False | ❌ False |

**Problems**:
- ❌ Test AUC 0.4959 = essentially random (coin flip is 0.50)
- ❌ No 'side' feature = can't trade directionally
- ❌ Bypassed V3 labeling pipeline = no trend_scanning

**Why It Failed**: Used simple next-bar labels instead of proper event-based labeling with trend_scanning.

---

### Full Pipeline Model - Training Only
From: `full_pipeline_output_complete.log`

**Training**: Oct 2024 - Nov 2025 (6,796 events)
**Testing**: Not tested (data leakage discovered)

| Metric | Value |
|--------|-------|
| **Events Generated** | 6,796 |
| **LONG Events** | 3,724 (54.8%) ✅ |
| **SHORT Events** | 3,072 (45.2%) ✅ |
| **Stop Labels** | 269 (4.0%) |
| **Vertical Labels** | 2,985 (43.9%) |
| **Target Labels** | 3,542 (52.1%) |
| **has_side_feature** | ✅ True |
| **Training Status** | ✅ Completed |

**Key Achievement**: Successfully generated balanced LONG/SHORT events!

**CRITICAL PROBLEM**: Model features include data leakage:
- `exit_price` (future information!)
- `ret_gross`, `ret_net` (future returns!)
- Cannot be used in production

---

## Side-by-Side Comparison

### Feature Sets

| Model | Features | Has 'side'? | Clean? |
|-------|----------|-------------|--------|
| **Old Baseline** | 34 | ✅ Yes | ✅ Yes |
| **Simple Retrained** | 34 | ❌ No | ✅ Yes |
| **Full Pipeline** | 42 | ✅ Yes | ❌ No (leakage) |

### Performance Summary

| Model | Win Rate | Directional Balance | Production Ready? |
|-------|----------|---------------------|-------------------|
| **Old Baseline** | 3.1% (Jan 2026) | 100% LONG, 0% SHORT | ❌ Broken |
| **Simple Retrained** | ~50% (random) | Unknown | ❌ No 'side' |
| **Full Pipeline** | Not tested | Should be 50/50 | ❌ Data leakage |

---

## Root Cause Analysis Summary

### Problem 1: Distribution Shift ✅ CONFIRMED
- **Evidence**: 13-month gap between training (Dec 2024) and test (Jan 2026)
- **Impact**: Win rate dropped from 58% → 3.1% (55-point collapse)
- **Solution**: Retrain on recent data (Oct 2024 - Nov 2025)
- **Status**: Identified, solution attempted but incomplete

### Problem 2: 100% LONG Bias ✅ FIXED (not validated)
- **Location**: `ml_intraday_v3/live_trading/replay.py:345`
- **Bug**: Used `score > 0` instead of `prediction['side']`
- **Impact**: ALL trades forced to LONG regardless of model prediction
- **Fix**:
```python
predicted_side = prediction.get("side", 1)
direction = "LONG" if predicted_side > 0 else "SHORT"
```
- **Status**: Code fixed, not yet validated in backtest

### Problem 3: 'side' Feature Generation ✅ SOLVED
- **Root Cause**: 'side' comes from labeling pipeline (`generate_events`), not feature generation
- **Config**: Already correct (`event_policy: "trend_scanning"`)
- **Solution**: Use full V3 pipeline with trend_scanning
- **Status**: Successfully generated 6,796 events with 54.8% LONG / 45.2% SHORT

### Problem 4: Data Leakage ✅ DETECTED & AVOIDED
- **Problem**: One training approach merged label metadata with features
- **Leaky Features**: `exit_price`, `ret_gross`, `ret_net`, `entry_price`, `pt_mult`, `sl_mult`
- **Impact**: Model learned from future information (would fail in production)
- **Status**: Detected before deployment, avoided

### Problem 5: Event-Bar Merge Challenge ❌ BLOCKER
- **Problem**: Can't merge sparse events (6,796) with dense bars (22,993)
- **Root Cause**: Timestamp mismatch between event times and bar times
- **Impact**: Can't complete clean retraining
- **Status**: Technical challenge, needs investigation of existing pipeline

---

## What We Know For Sure

### ✅ Confirmed Working
1. **Event Generation**: Successfully creates balanced LONG/SHORT events (54.8% / 45.2%)
2. **Triple-Barrier Labeling**: Generates proper labels (4.0% stop, 43.9% vertical, 52.1% target)
3. **Feature Generation**: Creates 35 clean features without leakage
4. **Directional Bug**: Identified and fixed in replay.py (needs validation)

### ❌ Confirmed Broken
1. **Old Baseline**: 100% LONG bias due to replay.py bug
2. **Simple Retrained**: No 'side' feature, performs randomly
3. **Full Pipeline Model**: Has data leakage, can't use in production

### ⚠️ Unknown / Needs Testing
1. **Directional Fix**: Code changed but not validated in backtest
2. **Clean Retraining**: Can't complete due to event-bar merge issue
3. **Dec 2025 Performance**: No clean model to test yet

---

## Critical Next Steps

### 1. Solve Event-Bar Merge (IMMEDIATE)
**Why Critical**: Blocker for clean retraining
**Approach**:
- Study `ml_intraday_v3/training/dataset.py` to see how existing pipeline handles this
- Events are PRIMARY (they have 't0' and 'side')
- Need to look up features at event 't0' times (not merge indexes)
- Likely need: `features.loc[events['t0']]` or similar

### 2. Complete Clean Retraining
**Once merge solved**:
- Train on Oct 2024 - Nov 2025 events (6,796 samples)
- Use ONLY clean features + 'side'
- Test on Dec 2025 (313 events)
- Target: >40% win rate, balanced LONG/SHORT

### 3. Validate Directional Fix
**Test replay.py changes**:
- Run backtest with OLD_BASELINE model
- Verify signals are NOT 100% LONG
- Check if LONG/SHORT distribution matches model's 'side' predictions

### 4. Backtest Clean Model
**Once trained**:
- Test on Dec 2025 held-out data
- Verify: Win rate >40%, LONG/SHORT ~50/50, no 100% bias
- Compare to baseline (-$2,368 P&L)

---

## Expected Outcomes

### If Everything Works
- **Win Rate**: Should improve from 3.1% to >40% (ideally >50%)
- **Directional Balance**: ~50/50 LONG/SHORT (vs 100/0 baseline)
- **P&L**: Positive instead of -$2,368
- **Profit Factor**: >1.0 instead of 0.028

### Key Improvement Drivers
1. **Recent Training Data**: Adapts to current market regime (not 13-month-old patterns)
2. **Fixed Directional Bug**: Model can trade both directions
3. **Proper 'side' Feature**: Bidirectional capability enabled
4. **No Data Leakage**: Clean features = realistic performance

---

## Model Bundle Comparison

### Old Baseline
```python
{
    'primary_feature_columns': 34 features,
    'has_side_feature': True,  # ✅ But unused due to bug
    'metadata': {
        'training_method': 'Dec 2024',
        'test_accuracy': Unknown
    }
}
```

### Simple Retrained
```python
{
    'primary_feature_columns': 34 features,
    'has_side_feature': False,  # ❌ Missing!
    'metadata': {
        'training_method': 'Oct2024_Nov2025_retrain',
        'test_auc': 0.4959,  # Random
        'test_accuracy': 0.5078
    }
}
```

### Full Pipeline (Leaky)
```python
{
    'primary_feature_columns': 42 features,  # ❌ Includes leaky features
    'has_side_feature': True,  # ✅ Present
    'metadata': {
        'training_method': 'Full_V3_Pipeline_Oct2024_Nov2025',
        'side_feature_included': True
    }
}
```

### Target Clean Model (Not Yet Built)
```python
{
    'primary_feature_columns': 34 clean features + 'side',  # 35 total
    'has_side_feature': True,  # ✅ Present
    'metadata': {
        'training_method': 'Clean_Retraining_Oct2024_Nov2025',
        'test_auc': TARGET > 0.60,
        'test_accuracy': TARGET > 0.50,
        'train_period': '2024-10-01 to 2025-11-30',
        'test_period': '2025-12-01 to 2025-12-18'
    }
}
```

---

## Conclusion

### What We Accomplished
1. ✅ Identified distribution shift (13-month gap)
2. ✅ Found 'side' feature source (labeling pipeline)
3. ✅ Generated balanced LONG/SHORT events (54.8% / 45.2%)
4. ✅ Fixed directional bias bug in replay.py
5. ✅ Detected and avoided data leakage

### What's Still Needed
1. ❌ Solve event-bar merge technical challenge
2. ❌ Complete clean retraining
3. ❌ Validate directional fix in backtest
4. ❌ Test new model on Dec 2025

### Estimated Time to Completion
- **If merge solved quickly**: 1-2 days to production-ready model
- **If merge takes longer**: 3-5 days

### Confidence Assessment
- **'side' feature works**: HIGH (proven by event generation)
- **Directional fix works**: MEDIUM (code looks correct, needs testing)
- **Performance improves**: MEDIUM-HIGH (addressing root causes)
- **Can complete retraining**: MEDIUM (blocked on merge issue)

### Bottom Line
We've identified and fixed the critical bugs, but hit a technical challenge in the final step. The path forward is clear:solve the event-bar merge, retrain with clean features, validate in backtest, then deploy.
