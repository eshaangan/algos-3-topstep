# ML Pipeline V2: Pure ML Strategy with Fixed-Horizon Labels

## Overview

This document explains the **V2 ML pipeline refactor** designed to address fundamental overfitting issues discovered through extensive testing (see `FORWARD_VS_BACKWARD_GENERALIZATION.md`).

### The Problem

The original pipeline (V1) suffered from severe overfitting that could not be fixed by:
- Regularization adjustments (max_depth, min_samples_leaf)
- Feature selection (36 → 20 features)
- More historical data (15.5 years vs 6.6 years made it **worse**)
- Stop loss optimization (24 vs 32 ticks)

**Critical Finding**: The model failed in **BOTH directions**:
- **Forward** (train 2010-2019 → test 2022-2025): 87% WR → 18.8% WR (68pp drop)
- **Backward** (train 2019-2025 → test 2010-2019): 34.9% WR @ 0.50 threshold

This proved the problem was **fundamental to the model/strategy approach**, not just regime shifts.

### Root Causes

1. **TP/SL Simulation Labels**: Training labels based on simulated execution (stop/target/max-hold) allowed the model to memorize strategy-specific patterns rather than learning generalizable market dynamics
2. **Heuristic Gates**: EMA trend filters, lunch exclusions, ATR ranges, and hour blocking added complexity that didn't improve out-of-sample performance
3. **High Fixed Thresholds**: Using min_probability ≥ 0.65 for frequency control was too restrictive and threw away potentially good signals

---

## V2 Solution: Three Pillars

### 1. Market-Centric Fixed-Horizon Labels

**Before (V1 - TP/SL Simulation):**
```python
# Simulate trade from entry bar i
# - Entry: next bar open + slippage
# - Exit: stop hit, target hit, max-hold, or session end
# - Label: y = 1 if trade_pnl > 0 else 0
```

**After (V2 - Fixed-Horizon Market Labels):**
```python
# For bar i, compute future return H bars ahead
future_return_ticks = (close[i + H] - close[i]) / tick_size

# Binary labels based on threshold X
y_long = 1 if future_return_ticks >= +X else 0
y_short = 1 if future_return_ticks <= -X else 0
```

**Parameters:**
- `H = 12 bars` (60 minutes @ 5-min bars)
- `X = 10 ticks` (2.5 MES points)

**Why This Helps:**
- ✅ **Portable**: Same labels work for different execution strategies
- ✅ **Simpler**: No execution simulation complexity
- ✅ **Generalizable**: Learns market dynamics, not strategy artifacts
- ✅ **Robust**: Can't memorize TP/SL patterns because they're not in the label

---

### 2. Pure ML Entries (No Heuristic Gates)

**Before (V1 - Many Gates):**
```python
# Entry decision required:
# 1. ML probability >= threshold (0.65+)
# 2. EMA trend filter (price above/below EMA50)
# 3. Hour blocking (exclude hour 14, lunch, etc.)
# 4. ATR range filter (6-40 ticks)
# 5. Lunch exclusion
# 6. Risk checks (daily loss, trailing DD)
```

**After (V2 - Pure ML + Risk):**
```python
# Entry decision requires:
# 1. ML score >= score_threshold (fitted from train quantile)
# 2. Direction enabled (long/short)
# 3. Daily trade budget not exceeded (<= 2 trades/day)
# 4. Min bars since last trade (>= 12 bars = 60 min)
# 5. Risk checks (daily loss, trailing DD)
# 6. RTH session check (8:30 AM - 2:55 PM CT)
```

**Score Definition:**
```python
score = max(p_long, p_short)
direction = "long" if p_long >= p_short else "short"
```

**Why This Helps:**
- ✅ **Cleaner**: Fewer moving parts, easier to debug
- ✅ **Data-Driven**: Score threshold fitted from training data quantile
- ✅ **Scalable**: Adding more data improves threshold calibration
- ✅ **Interpretable**: Score directly represents model confidence

---

### 3. Frequency Control via Ranking + Daily Budget

**Before (V1 - High Fixed Threshold):**
```python
# Control frequency by cranking threshold high
min_probability_long = 0.65  # Very selective
# Problem: Throws away good signals, arbitrary cutoff
```

**After (V2 - Quantile-Based Threshold + Daily Budget):**
```python
# 1. Fit threshold from train quantile
score_quantile = 0.995  # Top 0.5% of opportunities
score_threshold = np.quantile(train_scores, score_quantile)

# 2. Apply daily budget (hard limit)
max_trades_per_day = 2

# 3. Enforce spacing between trades
min_bars_between_trades = 12  # 60 minutes
```

**How It Achieves ~1-2 Trades/Day:**
1. **Ranking**: Only top 0.5% of signals by score qualify
2. **Daily Budget**: Hard cap of 2 trades/day
3. **Spacing**: 60-minute minimum between entries

**Why This Helps:**
- ✅ **Adaptive**: Threshold automatically adjusts to training data distribution
- ✅ **Robust**: Works across different market regimes
- ✅ **Conservative**: Daily budget prevents over-trading
- ✅ **Predictable**: ~1-2 trades/day on average (not 0 or 10)

---

## File Structure

### New V2 Files

| File | Purpose |
|------|---------|
| `features/labels_v2.py` | Fixed-horizon market label generation |
| `models/train_v2.py` | Pure ML training pipeline |
| `backtesting/backtest_v2.py` | Pure ML backtest with score + budget |
| `tests/test_leakage_tripwires.py` | Data leakage detection tests |
| `core/simple_config.py` | Updated config with V2 parameters |

### Old V1 Files (Keep for Backward Compatibility)

| File | Purpose |
|------|---------|
| `features/labels.py` | TP/SL simulation labels (deprecated) |
| `models/train.py` | Training with policy tuning (deprecated) |
| `backtesting/backtest.py` | Backtest with heuristic gates (deprecated) |

---

## Usage

### 1. Train V2 Model

```bash
# Train on MES data with V2 approach
python models/train_v2.py \
  --data-path data/processed/mes_bars.h5 \
  --output-dir models/saved_v2 \
  --seed 42
```

**What It Does:**
1. Generates fixed-horizon labels (H=12, X=10)
2. Creates walk-forward splits with lockbox
3. Trains long model (short disabled by default)
4. Computes score threshold from train quantile
5. Evaluates on test + lockbox
6. Saves models + metadata

**Output:**
```
models/saved_v2/
├── model_long.joblib
├── model_short.joblib (optional)
└── metadata.json
```

**Metadata Includes:**
- `labels_v2`: {horizon_bars, threshold_ticks}
- `policy_v2`: {score_threshold, max_trades_per_day, min_bars_between_trades}
- `metrics`: {train, val, test}
- `lockbox_metrics`: {long, short}
- `windows`: {train, val, test, lockbox, embargo_bars}

---

### 2. Run V2 Backtest

```bash
# Backtest with V2 strategy
python backtesting/backtest_v2.py \
  --data-path data/processed/mes_bars.h5 \
  --model-dir models/saved_v2 \
  --save-trades analysis/v2_backtest_trades.csv
```

**What It Does:**
1. Loads V2 models + policy from metadata
2. Computes scores for all bars
3. Applies score threshold + daily budget + spacing
4. Executes trades with deterministic stop/target/max-hold
5. Saves trade log and summary

**Output:**
```
BACKTEST RESULTS
==================
Summary:
  trades: 127
  wins: 68
  losses: 59
  win_rate: 0.535
  profit_factor: 1.42
  net_pnl: $2,350
  max_drawdown: $850

Daily Stats:
  total_trading_days: 87
  avg_trades_per_day: 1.46
  max_trades_in_day: 2
  days_with_trades: 64
  days_with_zero_trades: 23
```

---

### 3. Run Leakage Tripwire Tests

**CRITICAL**: Run before deploying any model!

```bash
# Test for data leakage
python tests/test_leakage_tripwires.py \
  --data-path data/processed/mes_bars.h5 \
  --n-bars 5000
```

**Tests Performed:**

#### Test 1: Label Shuffle
- Randomly shuffle labels to break feature-label relationship
- **Expected**: AUC drops to ~0.50 (random)
- **If AUC > 0.60**: FAIL - leakage suspected

#### Test 2: Time Shift
- Shift features forward by 1 bar (see future)
- **Expected**: AUC collapses or stays similar
- **If shifted AUC > baseline**: FAIL - features leak future info

#### Test 3: Split Integrity
- Verify embargo gaps between train/val/test
- **Expected**: embargo = horizon + lookback + 1
- **If gaps wrong**: FAIL - temporal overlap

**All tests must PASS before going live.**

---

## Configuration

### V2 Parameters (`core/simple_config.py`)

```python
@dataclass
class TrainingConfig:
    # ===== LABELS V2 =====
    use_labels_v2: bool = True
    horizon_bars: int = 12  # 60 min @ 5-min bars
    threshold_ticks: int = 10  # 2.5 MES points

    # ===== PURE ML STRATEGY =====
    enable_long: bool = True
    enable_short: bool = False

    score_quantile: float = 0.995  # Top 0.5%
    max_trades_per_day: int = 2
    min_bars_between_trades: int = 12  # 60 min

    # ===== SPLITS =====
    train_fraction: float = 0.5
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    lockbox_fraction: float = 0.1  # NEVER use for tuning

    # ===== REGULARIZATION =====
    rf_n_estimators: int = 500
    rf_max_depth: int = 10
    rf_min_samples_leaf: int = 15
    rf_min_samples_split: int = 30
    rf_max_features: str = "sqrt"

    # ===== FEATURE SELECTION =====
    feature_selection_mode: str = "recommended"  # 20 features
    top_n_features: int = 20
```

### Tuning Knobs

**To Increase Trade Frequency:**
- Lower `score_quantile` (0.995 → 0.99 = top 1%)
- Increase `max_trades_per_day` (2 → 3)
- Decrease `min_bars_between_trades` (12 → 6 = 30 min)

**To Decrease Trade Frequency:**
- Raise `score_quantile` (0.995 → 0.998 = top 0.2%)
- Decrease `max_trades_per_day` (2 → 1)
- Increase `min_bars_between_trades` (12 → 24 = 120 min)

**To Change Label Definition:**
- Shorter horizon: `horizon_bars = 6` (30 min) for intraday swings
- Longer horizon: `horizon_bars = 24` (120 min) for trend following
- Lower threshold: `threshold_ticks = 8` (2.0 points) for more positive labels
- Higher threshold: `threshold_ticks = 12` (3.0 points) for stronger moves

---

## Metrics to Watch

### 1. Calibration

**Check:** Do probabilities match actual outcomes?

```python
# Group test predictions into deciles
# For each decile, compute:
# - Mean predicted probability
# - Actual positive rate

# Good calibration: predicted ≈ actual
```

**Red Flag**: Large calibration error (predicted ≠ actual)

### 2. AUC (ROC-AUC)

**Train AUC**: Should be ~0.60-0.70 (moderate learning)
**Val AUC**: Should be within 0.05 of train (no overfitting)
**Test AUC**: Should be within 0.05 of val (stable)
**Lockbox AUC**: Final check, should be similar to test

**Red Flag**:
- Train AUC > 0.80 (likely overfitting)
- Val/Test AUC < 0.55 (no edge)
- Large gaps between splits (>0.10)

### 3. Expected Value (EV) Stability

**Check:** Net P&L consistent across splits?

```
Train P&L: +$X
Val P&L:   +$Y (similar to X)
Test P&L:  +$Z (similar to Y)
```

**Red Flag**:
- Train profitable, val/test losing
- Large swings between splits

### 4. Trade Count Per Day

**Target**: 1-2 trades/day on average

**Check:**
```python
avg_trades_per_day = total_trades / total_trading_days
# Should be ~1-2

days_with_zero_trades / total_trading_days
# Should be <50% (not too inactive)

max_trades_in_day
# Should be <= max_trades_per_day (budget enforced)
```

**Red Flag**:
- Avg > 3 trades/day (too active)
- Avg < 0.5 trades/day (too inactive)
- Budget violated (max > limit)

### 5. Win Rate Distribution

**Check:** Wins/losses across time

```python
# Plot cumulative wins/losses by week/month
# Should be relatively stable, not streaky
```

**Red Flag**:
- All wins in one period, all losses in another
- Long losing streaks (>10 trades)

---

## Comparison: V1 vs V2

| Aspect | V1 (Old) | V2 (New) |
|--------|----------|----------|
| **Labels** | TP/SL simulation | Fixed-horizon market returns |
| **Entry Logic** | ML prob + trend + hour + ATR + lunch | ML score + daily budget + spacing |
| **Threshold** | Fixed 0.65 | Adaptive quantile 0.995 |
| **Frequency Control** | High threshold (restrictive) | Ranking + budget (selective) |
| **Splits** | Train/Val/Test | Train/Val/Test/Lockbox |
| **Policy Tuning** | Complex grid search | Simple threshold from quantile |
| **Heuristic Gates** | Many (trend, hour, lunch, ATR) | None (pure ML) |
| **Out-of-Sample** | Failed (-$187 to -$2,598) | **TBD** (test V2) |

---

## Expected Outcomes

### Realistic Expectations

**V2 is NOT a silver bullet.** The forward/backward generalization test showed the problem is fundamental. V2 addresses known issues but may still struggle.

**Best Case Scenario:**
- Test AUC: 0.58-0.62 (modest edge)
- Test WR: 52-55% (slightly better than random)
- Test PF: 1.2-1.5 (positive but not amazing)
- Trades/day: 1-2 (as designed)
- Net P&L: Small positive (+$500-$1,500 on 3-6 months test)

**Worst Case Scenario:**
- Test AUC: <0.55 (no edge)
- Test WR: <50% (losing)
- Test PF: <1.0 (more losses than wins)
- Net P&L: Negative

**What Would Constitute Success:**
1. **Consistent AUC** across train/val/test/lockbox (±0.05)
2. **Small positive P&L** on test + lockbox (>$0)
3. **No leakage** (all tripwire tests pass)
4. **Trade frequency on target** (~1-2/day)

**What Would Constitute Failure:**
1. Train AUC > 0.70, Test AUC < 0.55 (still overfitting)
2. Test P&L < -$500
3. Leakage tests fail
4. Extreme trade frequency (>5/day or <0.5/day)

---

## Next Steps After V2

### If V2 Works (Modest Positive Results)

1. **Paper Trade**: 50+ trades before going live
2. **Monitor Metrics**: Track actual vs backtest (AUC, WR, PF, trades/day)
3. **Stop Rules**: Halt if test metrics deviate >20% from backtest
4. **Small Size**: Start with 1 contract, scale slowly

### If V2 Fails (Still Overfits)

**Options:**

#### A) Feature Engineering
- Order flow imbalance
- Bid-ask spread dynamics
- Volume profile (POC, VWAP)
- Alternative data (options flow, sentiment)

#### B) Model Architecture
- **LSTM/GRU**: Capture sequential dependencies
- **Transformer**: Attention over past N bars
- **Ensemble**: Combine RF + gradient boosting

#### C) Different Target
- Predict volatility regime changes
- Predict multi-bar direction (classification over 3+ bars)
- Predict continuous returns (regression, not classification)

#### D) Different Strategy Type
- **Regime Detection**: Classify markets, trade only in favorable regimes
- **Mean Reversion**: Statistical pairs, Bollinger Band touches
- **Trend Following**: Simple SMA cross with position sizing
- **Options Strategies**: Selling premium with defined risk

#### E) Accept Limitations
- Futures trading is **very difficult**
- Simple technical patterns (EMAs, volume, volatility) are widely known
- Any edge from these features is likely arbitraged away
- May need proprietary data or significantly different approach

---

## Key Takeaways

1. **V2 fixes known issues**: TP/SL label artifacts, heuristic gate complexity, fixed threshold rigidity
2. **V2 enforces discipline**: Lockbox split, leakage tests, score-based ranking
3. **V2 targets realism**: ~1-2 trades/day, no heuristic filters, adaptive threshold
4. **V2 is testable**: Comprehensive metrics, tripwire tests, lockbox validation

**But**: V2 does NOT guarantee profitability. The underlying data may not have persistent, generalizable patterns for this type of strategy.

**Bottom Line**: Test V2 thoroughly. If it works, proceed cautiously with paper trading. If it fails, consider fundamentally different approaches or accept that this strategy type may not be viable.

---

## Troubleshooting

### Issue: Leakage test fails

**Diagnosis**: Check split integrity, feature calculation, label alignment

**Fix**: Run diagnostics in `models/validate_splits.py`, verify embargo gaps

### Issue: Too many trades (>3/day avg)

**Diagnosis**: Score threshold too low or budget not enforced

**Fix**: Increase `score_quantile` (0.995 → 0.998) or check backtest logic

### Issue: Too few trades (<0.5/day avg)

**Diagnosis**: Score threshold too high

**Fix**: Decrease `score_quantile` (0.995 → 0.99) or lower `threshold_ticks` in labels

### Issue: Test AUC < 0.55

**Diagnosis**: No predictive edge in features

**Fix**: Try different features, horizon, or threshold. May need new data sources.

### Issue: Train AUC > 0.75, Test AUC < 0.55

**Diagnosis**: Still overfitting despite V2 changes

**Fix**: Increase regularization (max_depth → 8, min_samples_leaf → 20), reduce features, or try simpler model

---

## References

- `FORWARD_VS_BACKWARD_GENERALIZATION.md`: Detailed analysis of V1 failure modes
- `features/labels_v2.py`: Fixed-horizon label implementation
- `models/train_v2.py`: V2 training pipeline
- `backtesting/backtest_v2.py`: V2 backtest with score + budget
- `tests/test_leakage_tripwires.py`: Data leakage detection

---

**Version**: 2.0
**Date**: 2025-12-20
**Status**: Initial implementation, pending validation
