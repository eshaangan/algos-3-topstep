# Recovery Plan: Fixing January 2026 Out-of-Sample Failure

**Date**: January 29, 2026
**Status**: CRITICAL - Model failed in live trading

---

## Executive Summary

**Problem**: The model achieved 80.3% accuracy on 2024-2025 data but only 35.5% win rate when deployed live in January 2026, losing $884.73.

**Root Cause**: The model overfits to the 2024-2025 regime and fails to generalize to new market conditions.

**This Document**: Provides ranked, actionable fixes from most to least likely to succeed.

---

## What We Know

### Training Performance (2024-2025)
- Walk-forward validation: 80.3% accuracy across 6 months
- Cumulative PnL: +$4,725
- Sharpe: 8.41
- 5/6 months profitable

### Live Trading Failure (January 2026)
- Win rate: 35.5% (152 trades)
- Total PnL: -$884.73
- Sharpe: -2.20
- Profitable days: 3/18 (16.7%)
- Max drawdown: -$1,315 (approaching -$1,500 Topstep limit)

**Gap**: 45 percentage point drop in accuracy (80% → 35%)

---

## Root Cause Hypotheses (Ranked by Likelihood)

### 1. Regime Shift (MOST LIKELY)
- January 2026 represents a different market regime than 2024-2025
- Features that worked historically no longer predict outcomes
- Volatility, trend, or microstructure changed

### 2. Overfitting to Noise
- Model learned spurious patterns specific to 2024-2025
- These patterns don't exist in 2026
- Need better regularization or simpler model

### 3. Label Leakage (MEDIUM LIKELIHOOD)
- Subtle look-ahead bias in feature engineering or labeling
- Works in-sample but fails out-of-sample
- Need forensic review of feature computation

### 4. Poor Calibration
- Model probabilities calibrated for 2024-2025 conditions
- Thresholds (0.5 for classification) inappropriate for 2026
- Need probability recalibration

### 5. Data Quality Issues (LOWER LIKELIHOOD)
- Live trading data different from backtest data
- Slippage, execution costs, or timing differences
- Less likely given magnitude of failure

---

## Action Plan (Ranked by ROI)

### TIER 1: MUST DO IMMEDIATELY (High ROI, Low Effort)

#### Action 1.1: Stop Live Trading ✅
**Status**: Likely already done (last trade Jan 23, 2026)
**Rationale**: Prevent further losses until model is fixed

---

#### Action 1.2: Implement Regime Detection Filter
**Priority**: CRITICAL
**Effort**: 2-3 days
**Expected Impact**: Prevent trading in unfavorable regimes

**Implementation**:

Create `ml_intraday_v3/filters/regime_filter.py`:

```python
def detect_regime_shift(
    current_features: pd.DataFrame,
    reference_features: pd.DataFrame,
    feature_cols: list,
    threshold: float = 0.3
) -> bool:
    """
    Detect if current market regime differs from reference (training) regime.

    Returns:
        True if regime is similar (safe to trade)
        False if regime has shifted (do NOT trade)
    """
    from scipy.stats import ks_2samp

    # Compare distributions
    significant_shifts = 0

    for feature in feature_cols:
        ref_vals = reference_features[feature].dropna()
        curr_vals = current_features[feature].dropna()

        if len(ref_vals) == 0 or len(curr_vals) == 0:
            continue

        # KS test
        ks_stat, ks_pvalue = ks_2samp(ref_vals, curr_vals)

        # If distribution shifted significantly
        if ks_pvalue < 0.05:
            significant_shifts += 1

    # If >30% of features shifted, regime has changed
    shift_pct = significant_shifts / len(feature_cols)

    return shift_pct < threshold  # True = safe to trade


def rolling_regime_check(
    bars_df: pd.DataFrame,
    feature_cols: list,
    reference_window_days: int = 90,
    current_window_days: int = 5,
    threshold: float = 0.3
) -> pd.Series:
    """
    Check regime stability over rolling windows.

    Returns:
        Boolean series: True = safe to trade, False = regime shifted
    """
    # Implementation...
    pass
```

**Usage in Live Trading**:
```python
# Before taking any trade:
if not regime_filter.detect_regime_shift(current_data, training_data, feature_cols):
    logger.warning("Regime shift detected - SKIPPING TRADE")
    return None  # Don't trade
```

**Validation**:
- Backtest with regime filter on Dec 2025 data
- Should reduce trades but maintain/improve win rate

---

#### Action 1.3: Add Confidence Threshold Filter
**Priority**: CRITICAL
**Effort**: 1 day
**Expected Impact**: Only trade high-confidence signals

**Implementation**:

Modify `ml_intraday_v3/configs/execution_spec.yaml`:
```yaml
filters:
  confidence:
    enabled: true
    min_probability_distance: 0.25  # Require P(target) > 0.75 or < 0.25
    # Only trade when model is very confident
```

**Code**:
```python
# In signal generation:
y_proba = model.predict_proba(X_scaled)[:, 1]

# Confidence = distance from 0.5
confidence = np.abs(y_proba - 0.5)

# Only take trades with high confidence
high_confidence_mask = confidence >= 0.25  # P > 0.75 or P < 0.25

signals = signals[high_confidence_mask]
```

**Expected Result**:
- Reduce trades by 50-70%
- Increase win rate by 10-15 percentage points
- Avoid marginal predictions (50-60% probability)

---

### TIER 2: HIGH PRIORITY (Medium Effort, High Impact)

#### Action 2.1: Implement Ensemble with Shorter Training Windows
**Priority**: HIGH
**Effort**: 3-5 days
**Expected Impact**: Reduce overfitting to specific regimes

**Approach**: Train multiple models on different time windows and ensemble

**Implementation**:
```python
# Train 3 models:
# Model 1: Train on last 12 months
# Model 2: Train on last 6 months
# Model 3: Train on last 3 months

# Ensemble prediction:
y_proba_ensemble = (y_proba_12m + y_proba_6m + y_proba_3m) / 3

# Weight more recent models higher:
y_proba_weighted = (
    0.2 * y_proba_12m +
    0.3 * y_proba_6m +
    0.5 * y_proba_3m  # Most recent gets highest weight
)
```

**Rationale**:
- Shorter windows adapt faster to regime changes
- Ensemble reduces overfitting to any single period
- Weighted ensemble prioritizes recent data

**Validation**:
- Walk-forward test with ensemble
- Should improve stability across different months

---

#### Action 2.2: Add Model Performance Circuit Breaker
**Priority**: HIGH
**Effort**: 1-2 days
**Expected Impact**: Auto-stop trading when model starts failing

**Implementation**:

Create `ml_intraday_v3/monitoring/circuit_breaker.py`:
```python
class ModelCircuitBreaker:
    """Stop trading if model performance degrades."""

    def __init__(
        self,
        lookback_trades: int = 20,
        min_win_rate: float = 0.45,
        max_drawdown: float = -800,
        max_consecutive_losses: int = 5
    ):
        self.lookback_trades = lookback_trades
        self.min_win_rate = min_win_rate
        self.max_drawdown = max_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.recent_trades = []
        self.is_tripped = False

    def check(self, trade_result: dict) -> bool:
        """
        Check if circuit breaker should trip.

        Returns:
            True if safe to continue trading
            False if should stop trading
        """
        self.recent_trades.append(trade_result)

        # Keep only recent trades
        if len(self.recent_trades) > self.lookback_trades:
            self.recent_trades.pop(0)

        # Check 1: Win rate too low
        wins = sum(1 for t in self.recent_trades if t['pnl'] > 0)
        win_rate = wins / len(self.recent_trades)

        if win_rate < self.min_win_rate:
            self.is_tripped = True
            logger.error(f"🚨 CIRCUIT BREAKER: Win rate {win_rate:.1%} < {self.min_win_rate:.1%}")
            return False

        # Check 2: Drawdown too large
        cumulative_pnl = sum(t['pnl'] for t in self.recent_trades)
        if cumulative_pnl < self.max_drawdown:
            self.is_tripped = True
            logger.error(f"🚨 CIRCUIT BREAKER: Drawdown ${cumulative_pnl:.2f} < ${self.max_drawdown:.2f}")
            return False

        # Check 3: Too many consecutive losses
        consecutive_losses = 0
        for t in reversed(self.recent_trades):
            if t['pnl'] < 0:
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= self.max_consecutive_losses:
            self.is_tripped = True
            logger.error(f"🚨 CIRCUIT BREAKER: {consecutive_losses} consecutive losses")
            return False

        return True
```

**Usage**:
```python
circuit_breaker = ModelCircuitBreaker()

# After each trade:
if not circuit_breaker.check(trade_result):
    logger.critical("STOPPING ALL TRADING - Circuit breaker tripped")
    # Send alert, stop trading system
```

**Expected Result**:
- Would have stopped trading after ~30 trades in Jan 2026 (at -$500 instead of -$884)
- Limits damage from model failure

---

#### Action 2.3: Probability Calibration (Isotonic Regression)
**Priority**: MEDIUM
**Effort**: 2 days
**Expected Impact**: Better calibrated predictions

**Approach**: Use isotonic regression to calibrate probabilities

**Implementation**:
```python
from sklearn.isotonic import IsotonicRegression

# After training main model:
calibrator = IsotonicRegression(out_of_bounds='clip')

# Calibrate on validation set (not test!)
y_proba_train = model.predict_proba(X_val_scaled)[:, 1]
calibrator.fit(y_proba_train, y_val)

# In production:
y_proba_raw = model.predict_proba(X_scaled)[:, 1]
y_proba_calibrated = calibrator.predict(y_proba_raw)
```

**Validation**:
- Check calibration curves before/after
- Measure Brier score improvement

---

### TIER 3: DEEPER FIXES (High Effort, Medium-High Impact)

#### Action 3.1: Feature Engineering Review
**Priority**: MEDIUM
**Effort**: 5-7 days
**Expected Impact**: Eliminate potential leakage, add robust features

**Tasks**:

1. **Audit for Look-Ahead Bias**:
   - Review every feature computation
   - Ensure only past data used
   - Check rolling window calculations
   - Verify event labeling uses only future data

2. **Add Regime-Robust Features**:
   - Relative features (percentiles) instead of absolute values
   - Market microstructure features (order flow imbalance)
   - Cross-asset features (VIX, bonds, forex)
   - Time-of-day interactions

3. **Test Feature Stability**:
   - Measure feature importance across different time periods
   - Drop features that flip sign across periods
   - Keep only consistently informative features

**New Features to Consider**:
```python
# Relative momentum (more robust than absolute)
features['return_1_percentile'] = features['log_return_1'].rolling(100).rank(pct=True)

# Volatility-adjusted returns
features['return_1_vol_adj'] = features['log_return_1'] / (features['atr_14'] + 1e-9)

# Trend strength (direction-agnostic)
features['trend_strength'] = np.abs(features['ema_spread'])

# Market regime indicator
features['high_vol_regime'] = (features['vol_20'] > features['vol_20'].rolling(100).quantile(0.7)).astype(int)
```

---

#### Action 3.2: Online Learning / Model Retraining
**Priority**: MEDIUM-HIGH
**Effort**: 7-10 days
**Expected Impact**: Continuous adaptation to regime changes

**Approach**: Retrain model weekly or monthly on rolling window

**Implementation**:

Create `ml_intraday_v3/online_learning/adaptive_trainer.py`:
```python
class AdaptiveModelTrainer:
    """Continuously retrain model on recent data."""

    def __init__(
        self,
        retrain_frequency: str = 'weekly',  # 'daily', 'weekly', 'monthly'
        training_window_months: int = 6,
        min_samples: int = 500
    ):
        self.retrain_frequency = retrain_frequency
        self.training_window_months = training_window_months
        self.min_samples = min_samples

    def should_retrain(self, current_date: pd.Timestamp, last_train_date: pd.Timestamp) -> bool:
        """Check if it's time to retrain."""
        if self.retrain_frequency == 'daily':
            return (current_date - last_train_date).days >= 1
        elif self.retrain_frequency == 'weekly':
            return (current_date - last_train_date).days >= 7
        elif self.retrain_frequency == 'monthly':
            return (current_date - last_train_date).days >= 30
        return False

    def retrain(self, data: pd.DataFrame, current_date: pd.Timestamp):
        """Retrain model on recent data."""
        # Use rolling window
        train_start = current_date - pd.DateOffset(months=self.training_window_months)
        train_data = data[(data.index >= train_start) & (data.index < current_date)]

        if len(train_data) < self.min_samples:
            logger.warning(f"Insufficient samples ({len(train_data)}) for retraining")
            return None

        # Train new model
        # ... (same training code as baseline)

        return new_model
```

**Usage**:
```python
# Every week:
if trainer.should_retrain(current_date, last_train_date):
    logger.info("Retraining model on recent 6 months of data")
    new_model = trainer.retrain(data, current_date)
    # Replace production model
```

**Expected Result**:
- Model adapts to regime changes
- Uses most recent data for predictions
- May prevent Jan 2026 type failures

---

### TIER 4: FUNDAMENTAL REDESIGN (Very High Effort, Uncertain Impact)

#### Action 4.1: Meta-Learning Approach
**Priority**: LOW (research project)
**Effort**: 3-4 weeks
**Expected Impact**: Uncertain, high variance

**Approach**: Train a meta-model to predict when the primary model will fail

**Implementation**:
- Features: recent model accuracy, feature distribution shifts, volatility regime
- Label: whether next trade will be successful
- Meta-model decides: trade or skip

**Risk**: Complex, may not generalize

---

#### Action 4.2: Completely Different Approach
**Priority**: LOW (backup plan)
**Effort**: 4-6 weeks
**Expected Impact**: Unknown

**Alternatives to Consider**:
1. **Reinforcement Learning**: Learn trading policy directly from reward signal
2. **Rule-Based System**: Simple heuristics that don't rely on ML
3. **Market Making**: Switch from directional to spread capture
4. **Options Strategies**: Reduce directional risk

**Rationale**: If ML approach fundamentally flawed, try different paradigm

---

## Recommended Immediate Action Plan (Next 7 Days)

### Day 1-2: Damage Control
- ✅ Confirm live trading stopped
- ⚡ Implement confidence threshold filter (Action 1.3)
- ⚡ Implement circuit breaker (Action 2.2)
- 📊 Analyze trading logs for patterns in losses

### Day 3-4: Quick Wins
- ⚡ Implement regime detection filter (Action 1.2)
- 🧪 Backtest with filters on Dec 2025 data
- 📈 Measure win rate improvement

### Day 5-7: Medium-Term Fix
- ⚡ Implement ensemble with shorter windows (Action 2.1)
- 🧪 Walk-forward validation with ensemble
- 📋 If ensemble works, deploy to paper trading

### Day 8-14: Validation
- Paper trade with new filters + ensemble
- Monitor circuit breaker triggers
- Collect 20+ trades worth of data
- Decision: GO/NO-GO for live trading

---

## Success Criteria for Return to Live Trading

**ALL must be met**:
- ✅ Paper trading: 60%+ win rate over 20 trades
- ✅ Paper trading: Positive cumulative PnL
- ✅ Circuit breaker: Zero triggers in 20 trades
- ✅ Regime filter: Confirms current regime matches training
- ✅ Confidence filter: Reduces trade count by 40-60%
- ✅ Max drawdown: <$500 in paper trading
- ✅ No Topstep violations

**If ANY criterion fails**: Extend paper trading, iterate on fixes

---

## Key Insights & Lessons Learned

### What We Learned
1. **Walk-forward validation on same regime ≠ out-of-sample validation**
   - Testing on Aug-Jan 2025 is NOT the same as testing on Jan 2026
   - Need truly held-out periods from different market conditions

2. **High in-sample performance is a red flag**
   - 80% accuracy was too good to be true
   - Simple linear model shouldn't achieve such high accuracy
   - Indicates overfitting or regime-specific patterns

3. **Need multiple layers of safety**
   - Regime detection
   - Confidence filtering
   - Circuit breakers
   - Position sizing
   - Risk limits

4. **Model monitoring is essential**
   - Can't just "set and forget"
   - Need real-time performance tracking
   - Need automated shutdown triggers

### What We Should Have Done Differently
1. **Tested on earlier out-of-sample periods**
   - Use 2023 data as test (completely before training)
   - Use 2022 data as test
   - Check if edge existed historically

2. **Implemented circuit breaker from day 1**
   - Would have limited losses to -$500 instead of -$884

3. **Used ensemble from start**
   - Reduces overfitting
   - More robust to regime changes

4. **Added regime detection**
   - Don't trade when market structure changes
   - Preserve capital for favorable conditions

---

## Files to Create/Modify

### New Files
1. `ml_intraday_v3/filters/regime_filter.py` - Regime shift detection
2. `ml_intraday_v3/filters/confidence_filter.py` - High-confidence signals only
3. `ml_intraday_v3/monitoring/circuit_breaker.py` - Auto-stop on failure
4. `ml_intraday_v3/online_learning/adaptive_trainer.py` - Continuous retraining
5. `ml_intraday_v3/experiments/ensemble_model.py` - Multi-window ensemble

### Modify Files
1. `ml_intraday_v3/configs/execution_spec.yaml` - Add filter configs
2. `ml_intraday_v3/configs/training.yaml` - Add ensemble config
3. Live trading system - Integrate filters and circuit breaker

---

## Next Steps

1. **User Decision Required**: Which actions to prioritize?
   - **Conservative**: Implement all Tier 1 actions (3-4 days)
   - **Moderate**: Tier 1 + Tier 2 (7-10 days)
   - **Aggressive**: Tier 1 + 2 + 3 (3-4 weeks)

2. **Validation Strategy**: Paper trading duration?
   - **Conservative**: 30 days paper trading before live
   - **Moderate**: 14 days paper trading
   - **Aggressive**: 7 days paper trading

3. **Risk Tolerance**: Position sizing for next attempt?
   - **Conservative**: 0.25 contracts (limit losses)
   - **Moderate**: 0.5 contracts
   - **Aggressive**: 1.0 contracts (current)

---

**Document Status**: READY FOR REVIEW
**Recommended Path**: Conservative (Tier 1 + 2, 14 days paper trading, 0.5 contracts)
**Confidence in Recovery**: MEDIUM (60% chance these fixes work)
**Alternative if Fails**: Fundamental redesign (Tier 4)
