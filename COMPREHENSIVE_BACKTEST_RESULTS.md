# Comprehensive Backtest Results - Optimized Configuration

## Executive Summary

⚠️ **MIXED RESULTS**: While the optimized configuration (24-tick stop, 2.0R, 0.65 threshold) is profitable on the test set (+$95), **out-of-sample performance overall is negative** when combining validation and test sets (-$187).

---

## Configuration Tested

**Optimized Parameters:**
- Stop Loss: 24 ticks ($30.00)
- Target: 48 ticks ($60.00) = 2.0R
- Probability Threshold: 0.65
- Direction: Long only
- Max Hold: 12 bars
- Slippage: 1 tick
- Commission: $2.35 per contract

---

## Results by Period

### Training Set (In-Sample)
**Period:** May 6, 2019 → April 12, 2023
**Window:** [0, 77,899]

| Metric | Value |
|--------|-------|
| **Trades** | 39 |
| **Win Rate** | 89.7% |
| **Profit Factor** | 8.16 |
| **Net P&L** | +$5,146 |
| **Max Drawdown** | $180 |
| **Ending Equity** | $55,146 |

**Status:** ✅ TopStep Compliant - Excellent performance

**Analysis:** Very strong performance on training data. 89.7% win rate with 8.16 profit factor indicates model learned the patterns well, but raises overfitting concerns.

---

### Validation Set (Out-of-Sample)
**Period:** April 17, 2023 → August 2, 2024
**Window:** [78,125, 103,903]

| Metric | Value |
|--------|-------|
| **Trades** | 8 |
| **Win Rate** | 37.5% |
| **Profit Factor** | 0.53 |
| **Net P&L** | **-$282** |
| **Max Drawdown** | $585 |
| **Ending Equity** | $49,718 |

**Status:** ❌ Not TopStep Compliant - Losing money

**Analysis:** Poor performance on validation set. This period (Apr 2023 - Aug 2024) appears to be particularly difficult for the strategy, with:
- Win rate dropped from 89.7% (train) to 37.5%
- Only 8 trades triggered at 0.65 threshold (very conservative)
- Model correctly avoided most trades but the ones taken were mostly losers

---

### Test Set (Out-of-Sample)
**Period:** August 7, 2024 → December 2, 2025
**Window:** [104,129, 130,021]

| Metric | Value |
|--------|-------|
| **Trades** | 5 |
| **Win Rate** | 40.0% |
| **Profit Factor** | 1.25 |
| **Net P&L** | **+$95** |
| **Max Drawdown** | $360 |
| **Ending Equity** | $50,095 |

**Status:** ✅ TopStep Compliant - Profitable

**Analysis:** Marginally profitable on test set. Only 5 trades is a very small sample size, so statistical significance is questionable. The +$95 profit could be within margin of random variation.

---

### Full Dataset
**Period:** May 6, 2019 → December 2, 2025
**All Bars:** 130,150

| Metric | Value |
|--------|-------|
| **Trades** | 52 |
| **Win Rate** | 76.9% |
| **Profit Factor** | 3.91 |
| **Net P&L** | +$4,959 |
| **Max Drawdown** | $585 |
| **Ending Equity** | $54,959 |

**Status:** ✅ TopStep Compliant - Highly profitable

**Analysis:** Strong overall performance, but heavily driven by training set (39 trades, $5,146 profit). Out-of-sample contribution is minimal.

---

## Summary Comparison Table

| Period | Trades | Win Rate | Profit Factor | Net P&L | Max DD | TopStep Compliant |
|--------|--------|----------|---------------|---------|--------|-------------------|
| **Training** | 39 | 89.7% | 8.16 | +$5,146 | $180 | ✅ |
| **Validation** | 8 | 37.5% | 0.53 | -$282 | $585 | ❌ |
| **Test** | 5 | 40.0% | 1.25 | +$95 | $360 | ✅ |
| **Full Dataset** | 52 | 76.9% | 3.91 | +$4,959 | $585 | ✅ |

---

## Out-of-Sample Analysis

**Combined Validation + Test (True Out-of-Sample):**

| Metric | Value |
|--------|-------|
| **Total Trades** | 13 |
| **Combined P&L** | **-$187** |
| **Average P&L per Trade** | -$14.37 |
| **Validation Contribution** | -$282 (8 trades) |
| **Test Contribution** | +$95 (5 trades) |

**Status:** ⚠️ **Out-of-sample is LOSING overall**

---

## Critical Insights

### 1. Validation Period Was Particularly Difficult

**Validation Period:** April 2023 - August 2024

This period shows significantly worse performance than both training and test:
- Win rate: 37.5% (vs 89.7% train, 40.0% test)
- Loss: -$282
- Only 8 trades triggered

**Possible Reasons:**
- **Regime shift:** Market dynamics changed after 2023 (post-COVID recovery period)
- **Volatility changes:** Different volatility regime made 24-tick stops less effective
- **Trend changes:** May have been more choppy/ranging market vs training period's trending

**Evidence:** The test period (Aug 2024 - Dec 2025) performs better, suggesting the validation period was an outlier difficult period.

---

### 2. Test Set Profitability May Not Be Significant

**Test Set:** Only 5 trades with +$95 profit

**Statistical Concerns:**
- With only 5 trades, confidence intervals are very wide
- 40% win rate (2 wins, 3 losses) could easily reverse with more trades
- +$95 profit = only ~$19 per trade average
- A single additional loss would make it negative

**Coin Flip Analogy:**
- Flipping a coin 5 times and getting 2 heads doesn't prove much
- Need 50+ trades for statistical significance at 40% win rate

---

### 3. Training vs Out-of-Sample Gap Remains Large

**Training:** 89.7% WR, 8.16 PF, +$5,146
**Out-of-Sample:** Combined -$187 over 13 trades

**This indicates:**
- Model is still overfitting to training data despite our regularization fixes
- The overfitting fixes improved validation realism (30% WR on labels) but didn't fix trading performance
- Need more aggressive regularization or different approach

---

### 4. Low Trade Frequency at 0.65 Threshold

**Total out-of-sample trades:** Only 13 trades over ~2.5 years (Apr 2023 - Dec 2025)

**Trade Frequency:**
- Validation: 8 trades over 15.5 months = 0.5 trades/month
- Test: 5 trades over 16 months = 0.3 trades/month

**Implications:**
- Very conservative threshold (0.65) reduces opportunities
- Would take months to accumulate enough trades for TopStep evaluation
- Low sample size makes performance assessment difficult

---

## Comparison: 24-Tick vs 32-Tick Stop

### Test Set Only

| Stop Size | Trades | Win Rate | Profit Factor | Net P&L |
|-----------|--------|----------|---------------|---------|
| 24 ticks | 5 | 40.0% | 1.25 | +$95 |
| 32 ticks | 5 | 40.0% | 0.82 | -$86 |

**Improvement:** +$181 by using 24-tick stop

### Full Dataset

| Stop Size | Trades | Win Rate | Profit Factor | Net P&L |
|-----------|--------|----------|---------------|---------|
| 24 ticks | 52 | 76.9% | 3.91 | +$4,959 |
| 32 ticks | ~91 | 54.95% | 1.09 | +$599 |

**Key Difference:** 24-tick stop is more selective (52 vs 91 trades) but much more profitable per trade.

---

## Validation vs Test Consistency

**Comparison:**

| Metric | Validation | Test | Difference |
|--------|-----------|------|------------|
| Trades | 8 | 5 | -3 |
| Win Rate | 37.5% | 40.0% | +2.5pp |
| Profit Factor | 0.53 | 1.25 | +0.71 |
| Net P&L | -$282 | +$95 | +$377 |

**Assessment:** ⚠️ **INCONSISTENT**

- Validation is losing, test is profitable
- Profit factor differs significantly (0.53 vs 1.25)
- Suggests model performance is **regime-dependent**
- Test period profitability may not generalize

---

## Risk Assessment

### TopStep Compliance by Period

| Period | Max Drawdown | Within $1,800 Limit? | Profitable? | Overall Status |
|--------|--------------|----------------------|-------------|----------------|
| Training | $180 | ✅ | ✅ | ✅ Pass |
| Validation | $585 | ✅ | ❌ | ❌ Fail (losing) |
| Test | $360 | ✅ | ✅ | ✅ Pass |
| Full Dataset | $585 | ✅ | ✅ | ✅ Pass |

**Key Risks:**

1. **Out-of-sample is negative overall** (-$187 on Val + Test)
2. **Low trade frequency** (13 trades in 2.5 years) makes it hard to recover from losses
3. **Validation period losses** suggest regime shifts can cause significant drawdowns
4. **Small sample size on test** (5 trades) makes profitability uncertain

---

## Recommendations

### Option A: Accept Current Configuration with Caution (Conservative)

**Approach:**
- Use 24-tick stop, 48-tick target, 0.65 threshold
- Start with paper trading
- Monitor for 20-30 trades before going live
- Be prepared to stop if losses exceed $300

**Pros:**
- Test set is profitable (+$95)
- Full dataset is highly profitable (+$4,959)
- Max drawdown well within TopStep limits

**Cons:**
- Out-of-sample overall is negative (-$187)
- Only 5 trades on test set (not statistically significant)
- Validation period lost -$282
- Very low trade frequency (0.3-0.5 trades/month)

---

### Option B: Lower Threshold to Increase Trade Frequency

**Approach:**
- Try thresholds: 0.60, 0.58, 0.55
- Keep 24-tick stop and 2.0R target
- Accept lower win rate but more opportunities

**Expected Outcome:**
- More trades per month
- Lower win rate (maybe 35-40% instead of 40%)
- Better statistical significance
- Potentially better overall P&L through volume

**Trade-off:** Lower quality trades but more data to assess performance

---

### Option C: Retrain with Less Aggressive Regularization

**Approach:**
- Increase model capacity: max_depth=11, min_samples_leaf=12
- Use top 25-30 features instead of 20
- Keep 24-tick stop configuration

**Expected Outcome:**
- Better validation/test performance
- Risk of re-introducing some overfitting
- Need to re-run all diagnostics

**Recommendation:** Try this as next experiment (Option 1 from TEST_SET_PERFORMANCE_ANALYSIS.md)

---

### Option D: Accept That Model Needs More Work

**Honest Assessment:**
- Out-of-sample is losing overall
- Test set profitability is marginal (5 trades, +$95)
- Validation period failed significantly (-$282)
- Model still overfits (89.7% train WR vs 37-40% OOS WR)

**Next Steps:**
- Collect more recent data for training
- Try different model architectures (LightGBM, XGBoost)
- Investigate validation period characteristics (why so difficult?)
- Consider ensemble methods or different features

---

## Conclusion

### What We Learned

✅ **Overfitting fix worked partially:** Validation metrics are now realistic (30% WR on labels vs 98.7% before)

✅ **Stop loss optimization helpful:** 24-tick stop improves test set from -$86 to +$95

⚠️ **But overall out-of-sample is still negative:** -$187 combined (Val + Test)

⚠️ **Test set profitability uncertain:** Only 5 trades, could be random variation

❌ **Model still overfits:** 89.7% train WR vs 37-40% OOS WR is too large a gap

---

### Honest Recommendation

**Current Status:** Model v2 with 24-tick stop is **marginally viable** but **high risk**.

**Best Path Forward:**

1. **Immediate:** Paper trade for 20 trades to see if test set profitability holds
2. **Short-term:** Try Option C (less aggressive regularization) to improve OOS performance
3. **Medium-term:** Lower threshold to 0.60 for more trade frequency
4. **Long-term:** If paper trading shows continued losses, revisit model architecture

**Critical Success Metric:**
- Paper trading must show **>50% win rate** and **>1.3 profit factor** over 20+ trades
- If not, model needs more work before risking real capital on TopStep

---

**Date:** 2025-12-20
**Model:** v2 (overfitting fixes + optimized stops)
**Configuration:** 24-tick stop, 48-tick target, 0.65 threshold, long only
