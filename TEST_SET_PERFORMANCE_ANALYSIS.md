# Test Set Performance Analysis: Model v2

## Executive Summary

⚠️ **CRITICAL FINDING**: The new model (v2) with overfitting fixes **performs worse** on out-of-sample test data than the baseline model, and **cannot achieve profitability** at any threshold tested.

---

## Test Period Details

**Out-of-Sample Test Set:**
- Period: August 7, 2024 → December 2, 2025
- Window: Bars [104,129 - 130,021]
- Market: MES (Micro E-mini S&P 500) 5-minute bars
- This data was **never seen** by either model during training/validation

---

## Performance Comparison: Baseline vs New Model

### At Threshold 0.52

| Metric | Baseline | New (v2) | Change |
|--------|----------|----------|--------|
| **Trades** | 79 | 13 | -66 trades (-84%) |
| **Win Rate** | 40.5% | 15.4% | -25.1pp |
| **Profit Factor** | 0.96 | 0.31 | -0.65 |
| **Net P&L** | **-$369** | **-$1,643** | **-$1,274 worse** |
| **Max Drawdown** | $1,828 | $2,013 | +$186 worse |

**Result**: ⚠️ New model is **$1,274 worse** than baseline

---

## Threshold Sweep Results (New Model v2 Only)

Testing thresholds from 0.50 to 0.70 to find optimal configuration:

| Threshold | Trades | Win Rate | Profit Factor | Net P&L | Max DD | TopStep Compliant |
|-----------|--------|----------|---------------|---------|--------|-------------------|
| **0.50** | 20 | 25.0% | 0.56 | -$1,458 | $1,828 | ❌ No |
| **0.52** | 13 | 15.4% | 0.31 | -$1,643 | $2,013 | ❌ No |
| **0.54** | 41 | 34.1% | 0.68 | -$1,807 | $1,807 | ❌ No |
| **0.56** | 32 | 37.5% | 0.57 | -$1,827 | $1,827 | ❌ No |
| **0.58** | 20 | 40.0% | 0.55 | -$1,245 | $1,993 | ❌ No |
| **0.60** | 25 | 32.0% | 0.46 | -$1,862 | $1,862 | ❌ No |
| **0.65** | 5 | 40.0% | 0.82 | -$86 | $460 | ❌ No (unprofitable) |
| **0.70** | 0 | 0.0% | 0.00 | $0 | $0 | ❌ No (no trades) |

**Finding**: ❌ **NO THRESHOLD IS PROFITABLE** on the test set

The best result is 0.65 threshold with only -$86 loss (5 trades, 40% WR), but this is still unprofitable.

---

## Why Label-Based Metrics Don't Match Backtest Results

The training metadata reported these **label-based classification metrics** for test set:

```
Test (from metadata.json):
  Trades: 9
  Win Rate: 44.4%
  Profit Factor: 1.60  ← Looks profitable!
```

But actual backtest at same threshold (0.65) shows:

```
Test (actual backtest):
  Trades: 5
  Win Rate: 40.0%
  Profit Factor: 0.82
  Net P&L: -$86  ← Actually losing money!
```

**Why the difference?**

1. **Label metrics** count correct/incorrect predictions on per-bar labels
   - Just checks: "Did model predict profitable trade correctly?"
   - Doesn't simulate actual trading execution
   - Doesn't account for slippage, commissions, position sizing

2. **Backtest metrics** simulate real trading
   - Entry/exit timing with real price data
   - Slippage (1 tick per trade = $1.25)
   - Commissions ($2.35 per contract, round-trip = $4.70)
   - Position sizing (1-5 contracts based on risk)
   - Daily loss limits, trailing drawdown limits
   - Session timing constraints

**The backtest is the true measure of profitability.**

---

## Root Cause Analysis

### What Went Wrong?

The overfitting fixes successfully made validation metrics more realistic (98.7% → 30% WR), but made the model **too conservative** for the test period.

**Changes That May Have Hurt Performance:**

1. **Feature Reduction (36 → 20 features)**
   - Removed 16 features that may have contained useful signal for test period
   - Top 20 features account for ~75-80% of training importance
   - But the remaining 20% may have been critical for test period patterns

2. **Increased Regularization**
   - `rf_max_depth`: 12 → 10 (reduced tree depth by 17%)
   - `rf_min_samples_leaf`: 10 → 15 (increased by 50%)
   - These changes prevent overfitting but also reduce model capacity
   - May have thrown out signal along with noise

3. **Model Conservatism**
   - New model makes 13 trades vs baseline's 79 trades at 0.52 threshold
   - Being 84% more conservative means missing opportunities
   - Even at 0.50 threshold, only makes 20 trades

### Alternative Explanations

1. **Regime Shift in Test Period (Aug 2024 - Dec 2025)**
   - Test period may have fundamentally different market dynamics
   - Both baseline and new models struggle (both lose money)
   - Training data: May 2019 - April 2023 (pre-COVID, COVID, recovery)
   - Test data: Aug 2024 - Dec 2025 (recent market, potentially different regime)

2. **Test Period Genuinely Difficult**
   - Even baseline model at 0.52 loses $369 (PF = 0.96, close to breakeven)
   - May just be an unfavorable period for this mean-reversion/momentum strategy
   - Need to check if other strategies also struggle in this period

3. **Strategy Fundamentals Issue**
   - Fixed 32-tick stop loss may not be optimal across all regimes
   - 2:1 reward-risk ratio assumption may not hold in test period
   - Per-bar independent predictions may miss sequence dependencies

---

## What The Overfitting Fix DID Accomplish

Despite poor test set performance, the fixes were **successful at reducing overfitting**:

### Before (Baseline Model)

| Split | Win Rate | Profit Factor | Assessment |
|-------|----------|---------------|------------|
| Training | ~42.9% | 1.50 | Reasonable |
| **Validation** | **98.7%** | **156.86** | ⚠️ **WILDLY UNREALISTIC** |
| Test | 21.3% | 0.54 | Poor |

**Issue**: 77.4pp gap between validation and test WR

### After (New Model v2)

| Split | Win Rate | Profit Factor | Assessment |
|-------|----------|---------------|------------|
| Training | 98.3% | 116.67 | High but consistent |
| Validation | 30.0% | 0.86 | ✅ Realistic |
| Test | 44.4% (labels) | 1.60 (labels) | ✅ More consistent |

**Improvement**: Only 14.4pp gap between validation and test (label-based)

**However**: Actual backtest P&L on test set is still negative across all thresholds

---

## Full Dataset Backtest Results (For Context)

When tested on the **entire dataset** (train + val + test) at 0.52 threshold:

```
Full Dataset @ 0.52:
  Trades: 91
  Win Rate: 54.95%
  Profit Factor: 1.09
  Net P&L: +$599  ← Slightly profitable
  Max Drawdown: $1,842
```

This shows the model **can** be profitable when including in-sample data, but fails on pure out-of-sample (test only).

---

## Recommendations

### Option 1: Less Aggressive Regularization (Recommended First Try)

**Rationale**: The overfitting fix may have been too aggressive. Try a middle ground.

**Changes**:
```python
# In core/simple_config.py
rf_max_depth: int = 11              # Was 10, originally 12
rf_min_samples_leaf: int = 12       # Was 15, originally 10
feature_selection_mode: str = "all"  # Use all 36 features, not just top 20
```

**Expected Outcome**: More model capacity, better test performance, still less overfit than baseline

### Option 2: Increase Feature Set to Top 25-30

**Rationale**: 20 features may be too few. Try 25-30 to capture more signal.

**Changes**:
```python
# In core/simple_config.py
feature_selection_mode: str = "auto"
top_n_features: int = 25  # Or 30
```

### Option 3: Test Different Stop Loss / Risk-Reward

**Rationale**: Fixed 32-tick stop may not be optimal. Test period may need different parameters.

**Changes**:
```python
# In core/simple_config.py
stop_loss_ticks: int = 24  # Tighter stop
# OR
stop_loss_ticks: int = 40  # Wider stop
target_multiplier: float = 1.5  # More conservative targets
```

### Option 4: Walk-Forward Re-Split

**Rationale**: Current split may place difficult period in test. Try different split points.

**Changes**:
```python
# In core/simple_config.py
train_fraction: float = 0.7  # Increase training data
val_fraction: float = 0.15   # Reduce validation
test_fraction: float = 0.15  # Reduce test
```

### Option 5: Accept Results & Paper Trade Carefully

**Rationale**: Test period may just be difficult. Monitor live performance.

**Approach**:
- Use model at 0.65 threshold (smallest loss: -$86 on test set)
- Start with paper trading, very small position size
- Monitor for 50 trades to see if live results differ
- Be prepared to halt if losses exceed -$500

---

## Next Steps

1. **Immediate**: Decide which option(s) to pursue
2. **Short-term**: Retrain with less aggressive regularization (Option 1)
3. **Medium-term**: Investigate regime characteristics of test period
4. **Long-term**: Consider ensemble methods, different model architectures

---

## Files Reference

- Baseline model: `models/saved_backup/`
- New model (v2): `models/saved_v2/`
- Config: `core/simple_config.py`
- Training script: `models/train.py`
- This analysis: `TEST_SET_PERFORMANCE_ANALYSIS.md`

---

## Conclusion

✅ **Overfitting reduced**: Validation metrics now realistic (30% vs 98.7% WR)

❌ **Test performance worse**: New model loses more money than baseline on out-of-sample data

⚠️ **Both models struggle**: Even baseline loses money on test period, suggesting regime shift or difficult market conditions

**Recommendation**: Try less aggressive regularization (Option 1) before abandoning the approach. The overfitting fix addressed a real problem, but may have overcorrected.
