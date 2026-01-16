# Overfitting Analysis - Bidirectional 24H Model

**Date**: 2026-01-14
**Status**: ❌ MODEL OVERFIT - NOT READY FOR LIVE TRADING

---

## Executive Summary

The bidirectional 24-hour model shows **severe overfitting** and **would fail Topstep combine** due to:
1. **Trailing drawdown breach**: $4,082 (limit: $2,500)
2. **Daily loss violations**: 8 days exceeded $1,000 loss
3. **Inconsistent fold performance**: 124% coefficient of variation

**Recommendation**: DO NOT use this model for live trading until issues are fixed.

---

## Backtest Performance Across Folds

| Fold | PnL | Win Rate | Trades | Bad Days (>$1k loss) |
|------|-----|----------|--------|----------------------|
| 0 | **-$1,474** | 47.9% | 359 | 0 |
| 1 | **-$1,080** | 53.2% | 453 | 1 |
| 2 | $7,716 | 53.2% | 832 | 1 |
| 3 | $22,067 | 55.3% | 929 | 1 |
| 4 | **$50,048** | 55.0% | 1,362 | **4** |
| 5 | $8,922 | 60.1% | 213 | 1 |
| **Total** | **$86,199** | **54.2%** | **4,148** | **8** |

### Key Issues:
- **2/6 folds lose money** (folds 0-1)
- **Fold 4 drives 58% of total profit** (unrealistic)
- **Coefficient of Variation**: 124% (>50% is concerning)
- **Performance range**: -$1,474 to $50,048 (swing of $51,522)

---

## Topstep 50K Combine Violations

### Daily Loss Limit Breaches ($1,000)

8 days exceeded the $1,000 daily loss limit:

| Date | Loss | Fold |
|------|------|------|
| 2023-04-25 | **-$1,865** | fold_2 |
| 2025-04-07 | -$1,549 | fold_5 |
| 2024-04-01 | -$1,440 | fold_4 |
| 2022-04-27 | -$1,341 | fold_0 |
| 2024-04-16 | -$1,324 | fold_4 |
| 2024-07-08 | -$1,267 | fold_4 |
| 2024-10-28 | -$1,092 | fold_4 |
| 2022-11-25 | -$1,062 | fold_1 |

**Any ONE of these days would instantly fail the Topstep combine.**

### Trailing Drawdown Breach

- **Max Drawdown**: $4,082
- **Topstep Limit**: $2,500
- **Breach Amount**: $1,582 (63% over limit)
- **Status**: ❌ WOULD FAIL COMBINE

### Risk Metrics

| Metric | Value | Topstep Limit | Status |
|--------|-------|---------------|--------|
| Max Daily Loss | -$1,865 | -$1,000 | ❌ FAIL |
| Max Drawdown | $4,082 | $2,500 | ❌ FAIL |
| Days with >$1k loss | 8 (1.7%) | 0 | ❌ FAIL |
| Win Rate | 54.2% | No limit | ⚠️ Low |
| Avg Trade | $20.78 | No limit | ⚠️ Low |

---

## Why This Happened

### 1. Overfitting to Training Data
- Model learned patterns that work in some periods (fold 4) but not others
- No generalization across different market regimes
- Likely overfit to specific market conditions in 2024

### 2. Aggressive Threshold (0.03)
- Taking too many marginal trades
- Low win rate (54.2%) combined with poor risk management
- Many small wins, occasional large losses

### 3. No Intraday Risk Management
- No circuit breakers for daily loss limit
- No position size reduction after losses
- No time-of-day filtering for high-risk periods

---

## What to Do Next

### Option 1: Paper Trade with Tight Risk Controls (RISKY)

If you still want to paper trade, add these MANDATORY risk controls:

1. **Daily Loss Circuit Breaker**: Stop trading at **-$500** (not -$1,000)
2. **Position Size Reduction**:
   - Start with **50% position size** (not 100%)
   - Reduce to 25% after -$250 daily loss
3. **Aggressive Threshold**: Increase to **0.07** (from 0.03)
4. **Time Filter**: Avoid trading during first/last hour (high volatility)

**Expected impact**:
- Fewer trades: 10-15/day (down from 20-30)
- Lower daily loss risk
- Still high chance of drawdown issues

### Option 2: Fix the Model First (RECOMMENDED)

Before going live, address the overfitting:

1. **Retrain with better regularization**:
   - Increase `min_child_samples` in LightGBM
   - Add `max_depth` limit (try 5-7)
   - Increase `min_split_gain`

2. **Ensemble across folds**:
   - Instead of using one model, use an ensemble of all 6 fold models
   - This smooths out the overfitting to specific periods

3. **Train on more recent data only**:
   - Current model uses 2022-2025 (3 years)
   - Try training on 2023-2025 only (2 years)
   - More recent data may be more relevant

4. **Add regime awareness**:
   - Train separate models for trending vs ranging markets
   - Use HMM or volatility regimes to select model

### Option 3: Use Walk-Forward Validation

Instead of K-Fold, use walk-forward:
- Train on 12 months, test on 1 month
- Roll forward monthly
- Only use windows that pass Topstep criteria
- This better simulates real trading

---

## Updated Notebook to Run

I can update `analysis/topstep_50k_combine_test.ipynb` to:
1. Point to your new model (`runs/bidirectional_24h_20260114`)
2. Run Monte Carlo simulations
3. Show realistic pass rate for Topstep combine
4. Identify which risk controls would help most

Would you like me to:
1. **Update and run the notebook** to see exact pass rate?
2. **Fix the overfitting** by retraining with better parameters?
3. **Add risk controls** to the live trading code and paper trade anyway?

---

## Bottom Line

**DO NOT GO LIVE** with this model yet. The overfitting is severe and you WILL hit:
- Daily loss violations (8 days already exceeded $1,000)
- Trailing drawdown breach ($4,082 > $2,500 limit)

You have two realistic options:
1. **Fix the model** (1-2 days of retraining)
2. **Add aggressive risk controls** and paper trade with 50% position size

Which would you prefer?
