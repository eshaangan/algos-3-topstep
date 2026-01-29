# Model Retraining Summary - Jan 25, 2026

## Executive Summary

**Status**: Partial success with key findings
- ✅ Successfully identified root cause of 100% LONG bias
- ✅ Generated proper 'side' feature using trend_scanning
- ❌ Hit technical challenges merging event-based and bar-based data
- ⚠️ Discovered data leakage in one attempted approach

## Models Trained

### 1. Simple Retrained Model (retrain_with_existing_data.py)
**File**: `model_bundle_retrained_oct2024_nov2025.pkl`

**Training**:
- Period: Oct 2024 - Nov 2025 (22,924 samples)
- Features: 34 clean features
- Method: Simple next-bar labels (bypassed V3 labeling pipeline)

**Results**:
- Train AUC: 0.6682, Accuracy: 61.31%
- Test AUC: 0.4959, Accuracy: 50.78% (essentially random)
- **has_side_feature: False** ❌

**Problem**: No 'side' feature = no directional capability

---

### 2. Full Pipeline Model (retrain_with_full_pipeline.py)
**File**: `model_bundle_full_pipeline.pkl`

**Training**:
- Period: Oct 2024 - Nov 2025 (6,796 events)
- Features: 42 features **WITH DATA LEAKAGE**
- Method: Full V3 pipeline (trend_scanning + triple-barrier)

**Results**:
- Successfully generated 'side' feature (54.8% LONG, 45.2% SHORT) ✅
- Triple-barrier labels: 4.0% STOP, 43.9% VERTICAL, 52.1% TARGET
- Model trained successfully

**CRITICAL PROBLEM**: Features include label metadata:
- `exit_price`, `ret_gross`, `ret_net` (future information!)
- `entry_price`, `pt_mult`, `sl_mult`, `sigma` (label metadata)
- This is **severe data leakage** - model can't predict in production

---

### 3. Old Baseline Model (model_bundle_OLD_BASELINE.pkl)
**File**: Backed up from original `model_bundle.pkl`

**Configuration**:
- Features: 34 clean features (no leakage)
- **has_side_feature: True** ✅
- Properly trained with V3 pipeline

**Known Performance** (from baseline backtest):
- Win rate: 13.7% on Jan 2026 data
- Profit factor: 0.19
- **100% LONG bias** (directional bug in replay.py)

---

## Root Cause Analysis

### Issue 1: 'side' Feature Generation ✅ SOLVED

**Finding**: The 'side' feature comes from the **labeling pipeline**, not feature generation.

**Source**: `ml_intraday_v3/labels/events.py` (line ~256)
```python
betas = best_beta[keep]
sides_all = np.where(betas >= 0.0, 1, -1).astype(int)
```

**Solution**: Must use `generate_events()` with `event_policy: "trend_scanning"` in `labeling.yaml`.

**Verification**: Successfully generated 6,796 events with:
- 54.8% LONG (3,724 events)
- 45.2% SHORT (3,072 events)

This is the correct, balanced distribution we needed!

---

### Issue 2: Directional Bias Bug 🔧 PARTIALLY FIXED

**Location**: `ml_intraday_v3/live_trading/replay.py:345`

**Original Bug**:
```python
direction = "LONG" if score > 0 else "SHORT"
```

**Problem**:
- `score` = max(score_ev_long, score_ev_short)
- Since it's the maximum, it's almost always positive
- Result: ALL trades become LONG

**Fix Applied**:
```python
predicted_side = prediction.get("side", 1)
if predicted_side == 0:
    should_trade = False
if should_trade:
    direction = "LONG" if predicted_side > 0 else "SHORT"
```

**Status**: Fix implemented but not yet validated in production

---

### Issue 3: Data Leakage in Full Pipeline Model ❌ BLOCKER

**Problem**: Fallback training in `retrain_with_full_pipeline.py` merged labels with features:
```python
merged = labels.merge(features, left_on='t0', right_index=True, how='inner')
```

**Labels DataFrame Contains**:
- `exit_price`: FUTURE information (where trade exits)
- `ret_gross`, `ret_net`: FUTURE returns
- `entry_price`: Technically available but redundant with bar close
- `pt_mult`, `sl_mult`, `sigma`: Label metadata

**Impact**: Model learned to "cheat" using future information. Won't work in production.

**Solution Attempted**: `retrain_clean.py` to use only clean features
**Status**: Hit technical challenge merging event-based data with bar-based features

---

## Technical Challenges Encountered

### Challenge: Event-Based vs Bar-Based Data Merge

**Problem**:
- Features are computed at every 5-minute bar (22,993 bars)
- Events are sparse, generated only when trend_scanning detects opportunities (6,796 events)
- Event timestamps (from `t0`) don't necessarily align with bar boundaries
- Standard pandas join/merge returns 0 rows

**Attempted Solutions**:
1. Direct index merge: `features.merge(events[['side']], left_index=True, right_index=True)` - 0 rows
2. Join approach: `events[['side']].join(features, how='inner')` - 0 rows

**Root Cause**: Index mismatch between event times and bar times

**Next Steps**:
- Need to understand how existing pipeline handles this (likely in `training/dataset.py`)
- May need to use nearest-neighbor timestamp matching
- Or generate features specifically at event times

---

## Comparison with Baseline

### Old Baseline (Pre-Retraining)
| Metric | Value |
|--------|-------|
| Training Period | Dec 2024 (1 month) |
| Test Period | Jan 4-23, 2026 |
| Test Win Rate | 13.7% |
| Test Profit Factor | 0.19 |
| Directional Balance | 100% LONG, 0% SHORT |
| has_side_feature | True (but unused due to bug) |

### Simple Retrained Model
| Metric | Value |
|--------|-------|
| Training Period | Oct 2024 - Nov 2025 (14 months) |
| Test Period | Dec 2025 |
| Test AUC | 0.4959 (random) |
| Test Accuracy | 50.78% (random) |
| has_side_feature | False ❌ |

### Full Pipeline Model (with leakage)
| Metric | Value |
|--------|-------|
| Training Period | Oct 2024 - Nov 2025 (6,796 events) |
| Test Period | Not tested (data leakage) |
| Training Success | ✅ Completed |
| Side Feature | ✅ Present (54.8% LONG, 45.2% SHORT) |
| **Data Leakage** | ❌ CRITICAL - uses future information |

---

## Key Findings

### 1. Distribution Shift Confirmed
- Old model trained on Dec 2024 data
- Tested on Jan 2026 (13-month gap)
- This explains 44-point win rate drop (58% → 13.7%)

### 2. 'side' Feature Mystery Solved
- Feature comes from labeling pipeline (trend_scanning)
- Successfully generated with proper LONG/SHORT balance
- Config was correct all along (`event_policy: "trend_scanning"`)

### 3. Directional Bug Identified
- replay.py was ignoring model's 'side' predictions
- Used score sign instead (always positive)
- Fix implemented but not validated

### 4. Data Leakage Discovered
- One training approach accidentally used future information
- Caught before deployment (good!)
- Highlights importance of careful feature engineering

---

## Recommendations

### Immediate (Next Steps)

1. **Fix Event-Bar Data Merge**
   - Study existing `training/dataset.py` to see how pipeline handles this
   - Implement proper timestamp alignment
   - Complete `retrain_clean.py` script

2. **Validate Directional Fix**
   - Test replay.py fix with a known good model
   - Verify LONG/SHORT signals are properly separated
   - Run backtest on Jan 2026 to confirm not 100% LONG

3. **Complete Clean Retraining**
   - Once merge issue solved, retrain with:
     - Clean features only (no label metadata)
     - 'side' feature from trend_scanning
     - Oct 2024 - Nov 2025 training period
   - Test on Dec 2025 held-out data

### Short-Term (This Week)

4. **Backtest Validation**
   - Test new model on Dec 2025 data
   - Verify directional balance (should be ~50/50, not 100%/0%)
   - Check win rate improves from 13.7% baseline

5. **Paper Trading**
   - 1 week paper trading with new model
   - Monitor LONG/SHORT distribution
   - Verify no 100% bias

### Long-Term (Ongoing)

6. **Monthly Retraining Schedule**
   - Retrain on rolling 4-month window
   - Keep most recent month for validation
   - Example: Feb 2026 → train on Nov 2024 - Feb 2026

7. **Monitoring & Alerts**
   - Track directional balance daily
   - Alert if >80% one direction for 3+ days
   - Monitor win rate (should stay >40%)

---

## Files Created

### Scripts
1. `retrain_with_existing_data.py` - Simple retraining (no 'side')
2. `retrain_with_full_pipeline.py` - Full V3 pipeline (has leakage)
3. `retrain_clean.py` - Clean retraining (incomplete - merge issue)
4. `test_retrained_model_dec2025.py` - Test script for Dec 2025

### Model Bundles
1. `model_bundle_retrained_oct2024_nov2025.pkl` - Simple (no 'side')
2. `model_bundle_full_pipeline.pkl` - Has leakage ❌
3. `model_bundle_OLD_BASELINE.pkl` - Backup of original

### Documentation
1. `SIDE_FEATURE_INVESTIGATION.md` - Root cause analysis
2. `CONFIG_FIX_REPORT.md` - Config investigation
3. `RETRAINING_SUMMARY.md` - This file

### Logs
1. `retrain_output.log` - Simple retraining log
2. `full_pipeline_output_complete.log` - Full pipeline log
3. `retrain_clean.log` - Clean retraining attempt

---

## Open Questions

1. **Event-Bar Timestamp Alignment**
   - How does existing V3 pipeline handle sparse events vs dense bars?
   - What's the correct way to merge event metadata with bar features?

2. **Model Bundle Compatibility**
   - Does `model_predictor.py` expect specific feature ordering?
   - How are features generated at inference time for events?

3. **Directional Fix Validation**
   - Has replay.py fix been tested end-to-end?
   - Are there other places where 'side' prediction is ignored?

---

## Success Metrics

### Must-Have (for production deployment)
- [ ] Win rate >40% on Dec 2025 backtest
- [ ] Profit factor >1.0
- [ ] LONG/SHORT balance 40-60% (NOT 100%/0%)
- [ ] No data leakage (clean features only)
- [ ] Model has 'side' feature and uses it

### Should-Have
- [ ] Win rate >50% (approaching Dec 2024 baseline of 58%)
- [ ] Profit factor >1.5
- [ ] Test on both Dec 2025 AND Jan 2026 (robust across regimes)

### Could-Have
- [ ] Win rate >55%
- [ ] Profit factor >2.0
- [ ] Sharpe ratio >1.0

---

## Conclusion

**Progress Made**:
- ✅ Identified root cause of 'side' feature generation
- ✅ Successfully generated events with balanced LONG/SHORT
- ✅ Fixed directional bias bug in replay.py
- ✅ Discovered and avoided data leakage

**Blockers Remaining**:
- ❌ Event-bar data merge technical challenge
- ❌ Clean model not yet trained
- ❌ Directional fix not yet validated

**Next Critical Step**: Solve event-bar merge issue to complete clean retraining.

**Estimated Time to Production**: 1-2 days if merge issue resolved quickly.
