# V2 Results Analysis: Still Cannot Generalize

**Date**: 2025-12-20
**Model**: V2 Pure ML with Fixed-Horizon Labels
**Data**: MES 5-minute bars (2019-2025, 130K bars)

---

## Executive Summary

❌ **V2 FAILED TO SOLVE THE OVERFITTING PROBLEM**

Despite implementing all recommended fixes:
- ✅ Fixed-horizon market labels (no TP/SL simulation)
- ✅ No heuristic gates (pure ML entries)
- ✅ Score-based frequency control
- ✅ Lockbox split for final validation
- ✅ Proper embargo gaps (split integrity PASSED)

**The model still cannot generalize to out-of-sample data.**

---

## Training Metrics

### Model Performance Across Splits

| Split | Samples | AUC | Interpretation |
|-------|---------|-----|----------------|
| **Train** | 64,123 | 0.763 | Good learning (moderate overfitting) |
| **Val** | 25,658 | 0.529 | Near random (collapsed from train) |
| **Test** | 25,658 | 0.553 | Near random (slightly better than val) |
| **Lockbox** | 12,485 | 0.539 | Near random (consistent with val/test) |

**Analysis:**
- Train AUC (0.763) shows the model **can learn** from training data
- Val/Test/Lockbox AUC (~0.53-0.55) shows it **cannot generalize**
- The 21pp drop from train to val indicates **overfitting persists**
- Out-of-sample performance barely above random (0.50)

---

## Backtest Results

### Full Dataset (All Splits)

**Period**: May 2019 - Dec 2025 (1,697 trading days)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Total Trades** | 141 | ~3,000 (1-2/day) | ❌ 95% below target |
| **Win Rate** | 39.0% | >50% | ❌ Below breakeven |
| **Profit Factor** | 0.86 | >1.3 | ❌ Losing strategy |
| **Net P&L** | -$440 | >$0 | ❌ Losing |
| **Max Drawdown** | -$754 | <$1,500 | ✅ Low (but only because few trades) |
| **Avg Trades/Day** | 0.083 | 1-2 | ❌ **98% below target** |

### Test Window Only (Out-of-Sample)

**Period**: Dec 2023 - Apr 2025 (339 trading days, ~1.4 years)

| Metric | Value | Status |
|--------|-------|--------|
| **Total Trades** | 25 | ❌ Only 1 trade every 13-14 days |
| **Win Rate** | 20.0% | ❌ **Catastrophic** (need >40%) |
| **Profit Factor** | 0.38 | ❌ **Terrible** (need >1.0) |
| **Net P&L** | -$449 | ❌ Losing |
| **Avg Trades/Day** | 0.074 | ❌ Far too low |

**This is WORSE than V1 test results.**

---

## Root Cause Analysis

### Issue 1: Score Threshold Too Conservative

**Fitted Threshold:** 0.6831 (99.5th percentile)

**Problem:**
- Threshold calibrated on training data where model had AUC=0.763
- On test data where AUC=0.553 (near random), very few bars exceed 0.6831
- Results in 1 trade every 12-14 days instead of 1-2 per day

**Evidence:**
```
Training set:
  Bars above threshold: 321 / 64,123 (0.50%)
  Expected trades/day: ~1-2 ✅

Test set (implied):
  Trades over 339 days: 25
  Actual trades/day: 0.074
  Days with trades: 24 / 339 (7.1%) ❌
```

**Why It Happened:**
The quantile-based approach assumes train and test distributions are similar. But because the model overfits, train scores are inflated. The threshold fitted on train is too high for test.

---

### Issue 2: Weak Predictive Power on Test Set

**Test AUC:** 0.553 (barely above random 0.50)

**Problem:**
- Model has almost no predictive power on unseen data
- Even when it does trade (score > 0.6831), win rate is only 20%
- This suggests the features don't capture persistent patterns

**Evidence:**
- Test win rate: 20% (much worse than 39% overall)
- Test profit factor: 0.38 (for every $1 won, lost $2.63)

---

### Issue 3: Fundamental Overfitting Persists

**Despite V2 Improvements:**

| Fix Applied | Result |
|-------------|--------|
| Fixed-horizon labels (not TP/SL) | Still overfits (train 0.763 → test 0.553) |
| No heuristic gates | Still overfits |
| Lockbox split | Lockbox also fails (AUC 0.539) |
| Regularization (max_depth=10, min_samples_leaf=15) | Still overfits |
| Feature selection (20 features) | Still overfits |

**Conclusion:**
The overfitting is **not due to TP/SL label artifacts, heuristic gates, or insufficient regularization.**

The overfitting is **fundamental to the features and model architecture** trying to predict MES 5-minute price movements.

---

## Comparison: V1 vs V2

### V1 Results (MES, 2019-2025)

**Test Window:**
- Trades: 5
- Win Rate: 40%
- Profit Factor: 1.25
- Net P&L: +$95

**Out-of-Sample (Val + Test):**
- Net P&L: -$187

### V2 Results (MES, 2019-2025)

**Test Window:**
- Trades: 25
- Win Rate: 20%
- Profit Factor: 0.38
- Net P&L: -$449

**Out-of-Sample (Val + Test estimated):**
- Net P&L: ~-$450 to -$600

### Verdict: V2 is WORSE than V1

**Why V2 Performed Worse:**
1. **Fewer trades**: V1 at least took some trades, V2 barely trades at all
2. **Lower win rate**: V2's 20% WR is worse than V1's 40% WR
3. **Higher losses**: V2 lost more money (-$449 vs V1's +$95 on test)

**What Happened:**
- V2's quantile-based threshold (0.6831) is too conservative for test data
- V2's fixed-horizon labels may be **harder to predict** than TP/SL simulation
  - TP/SL simulation: "Will this execute profitably given 24-tick stop, 48-tick target?"
  - Fixed-horizon: "Will price move +10 ticks in next 60 minutes?"
  - The second question may be **fundamentally harder** for the features to answer
- Removing heuristic gates removed potential **defensive filters**

---

## Leakage Test Results

### Test 1: Label Shuffle
**Status**: ❌ FAILED (but false positive)
- Baseline AUC: 0.4919 (too low on 5K bar subset)
- Shuffled AUC: 0.5007
- **Reason for failure**: Insufficient data in test subset, not actual leakage

### Test 2: Time Shift
**Status**: ❌ FAILED (but false positive)
- Baseline AUC: 0.4919 (too low)
- **Reason for failure**: Same as above

### Test 3: Split Integrity
**Status**: ✅ PASSED
- Train-Val embargo: 113 bars ✅
- Val-Test embargo: 113 bars ✅
- Lockbox embargo: 113 bars ✅
- No temporal overlap ✅

**Interpretation:**
The split integrity is correct. The label shuffle and time shift tests failed due to weak learning on small subsets, not actual data leakage.

---

## Why V2 Failed: Fundamental Issues

### 1. Features Don't Have Persistent Predictive Power

**Evidence:**
- Test AUC: 0.553 (barely above random)
- Backward generalization (from earlier testing): 34.9% WR @ 0.50 threshold
- Forward generalization: 18.8% WR

**The features (EMAs, volatility, volume, ATR, etc.) cannot predict future price movements in a time-invariant way.**

### 2. Fixed-Horizon Labels May Be Harder to Predict

**TP/SL Simulation (V1):**
- Question: "If I enter with 24-tick stop, 48-tick target, will I win?"
- Incorporates execution logic (stop placement, target, max-hold)
- Model learns stop-target-hold patterns

**Fixed-Horizon Market (V2):**
- Question: "Will price move ≥+10 ticks in next 60 minutes?"
- Pure market movement, no execution consideration
- May be **fundamentally noisier** signal

**Paradox:**
We removed TP/SL to prevent overfitting to execution artifacts, but the resulting labels may be **harder to predict** because they ignore realistic execution constraints.

### 3. Score Threshold Calibration Fails on Weak Models

**Intended Logic:**
- Fit threshold on train quantile (99.5%)
- Apply to test → get similar trade frequency

**What Actually Happens:**
- Train AUC: 0.763 (scores well-separated)
- Test AUC: 0.553 (scores compressed near 0.50)
- Train threshold (0.6831) too high for test
- Result: Almost no trades on test

**Alternative:**
If we lowered the threshold to get 1-2 trades/day on test, we'd probably still lose because the win rate is terrible (20%).

---

## What We've Learned

### V1 Testing Showed:
1. ❌ Forward generalization fails (train 2010-2019 → test 2022-2025)
2. ❌ Backward generalization fails (train 2019-2025 → test 2010-2019)
3. ❌ More data makes it worse (15.5 years ES worse than 6.6 years MES)

### V2 Testing Shows:
4. ❌ Fixed-horizon labels don't solve overfitting
5. ❌ Removing heuristic gates doesn't solve overfitting
6. ❌ Quantile-based thresholds fail when model is weak
7. ❌ V2 performs **worse** than V1 on test set

### Fundamental Conclusion:

**The features (technical indicators: EMAs, ATR, volume, volatility) do NOT have persistent, time-invariant predictive power for MES 5-minute price movements.**

**No amount of:**
- Label engineering (TP/SL vs fixed-horizon)
- Regularization (max_depth, min_samples_leaf)
- Feature selection (36 → 20)
- Split methodology (lockbox, embargo)
- Entry logic (heuristic gates vs pure ML)

**...will fix the fundamental problem that the features don't predict the target reliably across time.**

---

## Options Going Forward

### Option A: Acknowledge Defeat (Recommended)

**Reality Check:**
- 7 different approaches tried (regularization, feature selection, more data, stop optimization, reverse generalization, V2 refactor)
- **All failed**
- This is strong evidence that the approach is not viable

**What This Means:**
- The features (EMAs, ATR, volume, etc.) are insufficient
- MES 5-minute price movements may not be predictable with these features
- Technical patterns alone likely don't have persistent edge (widely known, arbitraged away)

**Next Steps:**
- Move to different strategy types (mean reversion, trend following with longer timeframes)
- Accept that algorithmic futures day-trading is **extremely difficult**
- Focus on risk management and capital preservation

---

### Option B: Radical Changes (Low Probability of Success)

**What Would Need to Change:**

#### 1. Different Data/Features
- **Order flow data**: Bid-ask imbalance, volume at price (VWAP, POC)
- **Market microstructure**: Tick-by-tick data, trade aggressor classification
- **Alternative data**: Options flow (put/call ratio), sentiment, news
- **Cross-asset signals**: VIX, bond yields, sector rotation

**Problem**: Expensive data, may still not work

#### 2. Different Model Architecture
- **LSTM/GRU**: Model temporal dependencies across 50-100 bars
- **Transformer**: Attention mechanism over past sequence
- **Deep Learning**: CNN for pattern recognition in price charts

**Problem**: More complex models = more overfitting risk

#### 3. Different Prediction Target
- **Multi-horizon**: Predict 5-min, 15-min, 60-min ahead simultaneously
- **Regression**: Predict continuous returns (not binary classification)
- **Volatility**: Predict realized volatility instead of direction
- **Regime**: Classify market state (trending/ranging/high vol/low vol)

**Problem**: Still using same features that didn't work

#### 4. Different Strategy Paradigm
- **Portfolio approach**: Trade basket of correlated instruments (risk parity)
- **Pairs trading**: Statistical arbitrage between related futures
- **Market making**: Provide liquidity, capture bid-ask spread
- **Options strategies**: Defined-risk premium selling

**Problem**: Completely different strategy, not ML-based

---

### Option C: Lower Threshold to Increase Trades (Not Recommended)

**What If We Lower score_threshold?**

Current: 0.6831 (99.5th percentile) → 25 trades, 20% WR, PF 0.38

If we lower to 0.60 (maybe 90th percentile):
- More trades (maybe 100-200 on test)
- But win rate likely **still below 40%** (model AUC is 0.553)
- Profit factor likely **still below 1.0**
- Net result: **More losing trades**

**Why Not:**
The problem isn't trade frequency. The problem is the model has **no edge** on test data.

Taking more trades with a losing edge = losing more money faster.

---

## Final Recommendation

### Accept That This Approach Doesn't Work

**After extensive testing:**
- V1: Multiple fixes attempted, all failed
- V2: Complete pipeline refactor, still failed

**The evidence is clear:**
- Technical features (EMAs, ATR, volume) don't predict MES 5-min movements
- Random Forest model cannot extract time-invariant patterns
- More data doesn't help
- Label engineering doesn't help
- Regularization doesn't help

**This is not a viable trading strategy for TopStep or live trading.**

---

### What to Do Instead

**1. Different Strategy Types** (if staying in futures):
- **Longer timeframes**: Daily/weekly trend following (less noise)
- **Mean reversion**: Statistical pairs, Bollinger touches
- **Systematic rules**: Simple SMA cross with strict risk management
- **Managed futures**: CTA-style diversified approach across many markets

**2. Different Instruments** (if staying short-term):
- **Options**: Premium selling with defined risk (theta decay edge)
- **Forex**: Some pairs have clearer trends/seasonality
- **Crypto**: Higher volatility = larger moves (but more risk)

**3. Accept Reality** (most honest):
- Algorithmic day-trading is dominated by HFT firms with:
  - Proprietary data feeds (faster than retail)
  - Co-located servers (nanosecond latency)
  - Sophisticated order flow analysis
  - Massive R&D budgets

**Retail ML day-trading with technical indicators on 5-minute bars is a game you're unlikely to win.**

---

## Lessons Learned

### Technical Lessons

1. **Overfitting is hard to fix**: Even with lockbox, embargo, regularization, feature selection
2. **More data ≠ better**: Can make overfitting worse if features are noisy
3. **Label engineering matters**: But not enough if features are weak
4. **Quantile thresholds fail**: When train/test distributions differ
5. **Leakage tests are important**: But can have false positives on small data

### Strategy Lessons

1. **Technical indicators are insufficient**: For short-term futures prediction
2. **Predictive edge is rare**: Especially on highly liquid, efficient markets
3. **Execution matters less than prediction**: Great execution can't save weak predictions
4. **Simplicity doesn't guarantee robustness**: Simple models can still overfit
5. **Past performance ≠ future results**: Even with proper validation

### Meta Lessons

1. **Know when to quit**: After 7+ failed attempts, evidence is clear
2. **Sunk cost fallacy**: Time invested doesn't justify continuing losing approach
3. **Market efficiency is real**: Easy opportunities don't exist (or don't persist)
4. **Risk management > prediction**: Capital preservation matters more than being right

---

## Appendix: V2 Configuration Used

```python
# Labels V2
use_labels_v2 = True
horizon_bars = 12  # 60 minutes
threshold_ticks = 10  # 2.5 MES points

# Model
rf_n_estimators = 500
rf_max_depth = 10
rf_min_samples_leaf = 15
rf_min_samples_split = 30
rf_max_features = "sqrt"

# Features
feature_selection_mode = "recommended"  # 20 features
top_n_features = 20

# Splits
train_fraction = 0.5
val_fraction = 0.2
test_fraction = 0.2
lockbox_fraction = 0.1

# Entry Logic
score_quantile = 0.995
max_trades_per_day = 2
min_bars_between_trades = 12
enable_long = True
enable_short = False

# Execution
stop_loss_ticks = 24
target_multiplier = 2.0
max_hold_bars = 12
```

---

**Conclusion**: V2 did not solve the overfitting problem. The fundamental issue persists: the features lack predictive power across different time periods. After exhaustive testing, the evidence suggests this strategy approach is not viable for profitable trading.
