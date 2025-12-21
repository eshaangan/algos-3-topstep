# Forward vs Backward Generalization Test Results

## Executive Summary

⚠️ **CRITICAL FINDING**: The model **cannot generalize in EITHER direction** - whether trained on old data and tested on new, or trained on new data and tested on old.

**This proves the problem is fundamental to the model/strategy, not just a regime shift issue.**

---

## Test 1: Forward Generalization (Normal)

**Train on OLD data (2010-2019), Test on NEW data (2022-2025)**

| Period | Trades | Win Rate | Profit Factor | Net P&L | Status |
|--------|--------|----------|---------------|---------|--------|
| **Training** (2010-2019) | 92 | 87.0% | 6.46 | +$9,732 | ✅ Excellent |
| **Test** (2022-2025) | 16 | 18.8% | 0.35 | -$1,526 | ❌ Terrible |

**Train-Test Gap:** 68.2 percentage points (87% → 18.8%)

**Conclusion:** ❌ **SEVERE OVERFITTING** - Cannot predict future from past

---

## Test 2: Backward Generalization (REVERSE)

**Train on NEW data (2019-2025), Test on OLD data (2010-2019)**

### At 0.65 Threshold (Optimized)
| Period | Trades | Win Rate | Profit Factor | Net P&L | Status |
|--------|--------|----------|---------------|---------|--------|
| **Training** (2019-2024) | 0 | 0.0% | 0.00 | $0 | ❌ Too conservative |
| **Test** (2010-2019) | 0 | 0.0% | 0.00 | $0 | ❌ No trades |

### At 0.50 Threshold (Most Permissive)
| Period | Trades | Win Rate | Profit Factor | Net P&L | Status |
|--------|--------|----------|---------------|---------|--------|
| **Test** (2010-2019) | 63 | 34.9% | 0.36 | -$1,849 | ❌ Losing |

**Conclusion:** ❌ **CANNOT GENERALIZE BACKWARD** - Cannot predict past from future

---

## Comparison: Same Time Period, Different Directions

### Testing on 2010-2019 Period

**Forward Model** (trained on 2010-2019):
- Self-test (in-sample): 92 trades, 87% WR, +$9,732 ✅

**Backward Model** (trained on 2019-2025, tested on 2010-2019):
- Out-of-sample test: 63 trades (@ 0.50), 34.9% WR, -$1,849 ❌

**Gap:** 52.1 percentage points - model overfits regardless of direction

---

## What This Tells Us

### 1. The Problem is NOT Regime-Specific

If the issue were only that markets changed after 2019, we would expect:
- ✅ Forward model fails (old → new): **TRUE** ✅
- ✅ Backward model succeeds (new → old): **FALSE** ❌

**Reality:** Both fail spectacularly.

**Conclusion:** The model doesn't learn generalizable patterns from ANY time period.

---

### 2. The Problem is Fundamental to the Approach

**Forward generalization:**
- Train 2010-2019 → Test 2022-2025
- 87% WR → 18.8% WR (68pp drop)
- In-sample excellent, out-of-sample terrible

**Backward generalization:**
- Train 2019-2025 → Test 2010-2019
- 0 trades @ 0.65, 34.9% WR @ 0.50 (terrible)
- Cannot predict old data either

**Conclusion:** The model architecture/features/strategy cannot learn time-invariant patterns.

---

### 3. Model Overfits Regardless of Data Amount

**6.6 years of MES data (2019-2025):**
- Out-of-sample: -$187 (Val + Test combined)

**15.5 years of ES data (2010-2025):**
- Out-of-sample: -$2,598 (Val + Test combined)
- **MORE data = WORSE results**

**Reverse (2019-2025 training):**
- Out-of-sample on 2010-2019: -$1,849 (@ 0.50 threshold)

**Conclusion:** More data doesn't fix overfitting - may actually make it worse by giving the model more patterns to memorize.

---

## Root Cause Analysis

### Why Both Directions Fail

1. **Model Memorizes, Not Learns**
   - High capacity Random Forest with 500 trees
   - Even with regularization (max_depth=10, min_samples_leaf=15)
   - Learns specific patterns unique to training period
   - Can't extract time-invariant signals

2. **Features May Not Be Predictive**
   - EMA spreads, volatility, volume ratios may not have consistent predictive power
   - What works in 2010-2019 doesn't work in 2022-2025
   - What works in 2019-2025 doesn't work in 2010-2019
   - Features capture noise, not signal

3. **Strategy Assumptions May Be Wrong**
   - Fixed 24-tick stop assumption may not hold across regimes
   - 2:1 reward-risk may not be achievable consistently
   - Per-bar independent predictions miss sequential dependencies
   - Mean-reversion/momentum signals may not be persistent

4. **Market Efficiency**
   - Simple technical patterns (EMAs, volatility, volume) are widely known
   - Any edge from these features likely arbitraged away
   - Or never existed in a statistically robust way

---

## Evidence Summary

| Test | Training Data | Test Data | Result | Interpretation |
|------|--------------|-----------|--------|----------------|
| **Forward** | 2010-2019 (old) | 2022-2025 (new) | ❌ Fails (18.8% WR) | Cannot predict future |
| **Backward** | 2019-2025 (new) | 2010-2019 (old) | ❌ Fails (34.9% WR) | Cannot predict past |
| **More Data** | 15.5 years ES | Out-of-sample | ❌ Worse (-$2,598) | More data ≠ better |
| **Less Data** | 6.6 years MES | Out-of-sample | ❌ Bad (-$187) | Less data also bad |

**Universal Finding:** The model cannot generalize to unseen data **regardless of**:
- Which time period it trains on
- How much data it has
- Which direction the split goes (forward or backward)

---

## What Doesn't Work

We've now tried:

1. ✅ **Overfitting fixes** (regularization, feature selection)
   - Result: Validation became realistic, but test still fails

2. ✅ **Stop loss optimization** (24 ticks vs 32 ticks)
   - Result: Small improvement on MES test (+$95), but ES test still loses

3. ✅ **More historical data** (15.5 years vs 6.6 years)
   - Result: Made overfitting WORSE

4. ✅ **Reverse generalization test** (train on new, test on old)
   - Result: Fails in reverse direction too

**All approaches failed.**

---

## Implications

### 1. Current Model/Strategy is Not Viable

The model shows:
- Severe overfitting in both directions
- No consistent edge across different time periods
- Performance that gets worse, not better, with more data

**This is not a model that will work on TopStep or live trading.**

### 2. Fundamental Rethink Required

We cannot fix this by:
- Adjusting regularization parameters
- Adding more data
- Optimizing thresholds or stop sizes
- Using different train/test splits

**Need a completely different approach:**
- Different features (alternative data, order flow, market microstructure)
- Different model architecture (LSTM, Transformer for sequential dependencies)
- Different strategy (not per-bar predictions, but regime detection or portfolio approaches)
- Or accept that this type of strategy may not have persistent edge

### 3. The Bar is High

For a profitable futures trading strategy:
- Need 55%+ win rate with proper risk management
- OR 40%+ win rate with excellent risk-reward (2.5-3:1)
- Must be robust across different market regimes
- Must have statistical significance (100+ trades minimum)

**Current model achieves none of these consistently.**

---

## Recommendations

### Option A: Complete Strategy Redesign (High Effort, Uncertain Outcome)

**What to try:**
1. **Regime detection first**
   - Classify markets into regimes (trending, ranging, high vol, low vol)
   - Train separate models per regime
   - Only trade when regime matches training conditions

2. **Different features**
   - Order flow imbalance
   - Bid-ask spread dynamics
   - Volume profile analysis
   - Alternative data (options flow, sentiment, etc.)

3. **Sequential models**
   - LSTM/GRU for temporal dependencies
   - Transformer with attention mechanisms
   - Capture patterns across multiple bars, not just per-bar

4. **Different targets**
   - Predict next N bars direction (regression)
   - Predict volatility regime changes
   - Portfolio optimization instead of single directional bets

**Effort:** Months of work
**Success Probability:** Low (10-20%)

---

### Option B: Accept Current Limitations (Realistic)

**What this means:**
1. **The model as-is will not work reliably**
   - Overfits to training data
   - Cannot generalize to new periods
   - Not suitable for TopStep evaluation

2. **Paper trading would likely show losses**
   - Test set showed -$1,526 (ES) or marginal +$95 (MES)
   - Real trading has additional slippage, execution issues
   - Likely worse than backtest

3. **Move on to different strategies**
   - Systematic strategies based on robust academic research
   - Simple trend-following with proper position sizing
   - Options strategies (selling premium with defined risk)
   - Or accept that algorithmic futures trading is very difficult

**Effort:** Minimal
**Success Probability:** N/A (not pursuing this path)

---

### Option C: Hybrid Approach (Middle Ground)

**What to try:**
1. **Use model as filter, not predictor**
   - Don't rely solely on model predictions
   - Use model to filter out obviously bad trades
   - Combine with other indicators (trend, volume, time of day)

2. **Conservative parameters**
   - Very high threshold (0.70+)
   - Trade only 5-10 times per month
   - Accept that most opportunities will be passed

3. **Extensive paper trading**
   - 50+ trades minimum before going live
   - Track all metrics vs backtest expectations
   - Stop immediately if results deviate

4. **Risk management focus**
   - Never risk more than 1% of capital per trade
   - Strict daily/weekly loss limits
   - Accept small sample size means high variance

**Effort:** Moderate (weeks of paper trading)
**Success Probability:** Low (20-30%), but learning experience

---

## Conclusion

The forward vs backward generalization test **conclusively proves** that the current model cannot learn time-invariant patterns from the data.

**Key Takeaways:**
1. ❌ **Training on old data** → Cannot predict new data (forward fails)
2. ❌ **Training on new data** → Cannot predict old data (backward fails)
3. ❌ **More data doesn't help** → Actually makes overfitting worse
4. ❌ **Optimization doesn't help** → Stop loss, thresholds, etc. are band-aids

**This is not a model problem. This is a strategy viability problem.**

The features, model architecture, and overall approach do not capture robust, repeatable edges in futures markets across different time periods.

**Recommendation:** Either pursue Option A (complete redesign, high effort, uncertain outcome) or Option B (accept limitations and move on to different approaches).

---

**Files:**
- ES forward model: `models/saved_es_v1/`
- ES reverse model: `models/saved_es_reverse/`
- MES model: `models/saved_v2/`

**Date:** 2025-12-20
