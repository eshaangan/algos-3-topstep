# SHORT Signal Investigation - Current Status

**Date**: 2026-01-25  
**Status**: Bug Identified, Fix Applied, But Still Not Working

---

## What We Found

### ✅ ROOT CAUSE IDENTIFIED
**Location**: `ml_intraday_v3/live_trading/replay.py:345` (original)

**The Bug**:
```python
# WRONG - derives direction from score sign
direction = "LONG" if score > 0 else "SHORT"
```

**Why It's Wrong**:
- `score` is the BEST EV (max of score_ev_long, score_ev_short)
- Model correctly chooses side=-1 for SHORT opportunities
- But replay.py ignores `prediction['side']` and uses `score > 0` instead
- Since score = best EV, it's almost always positive → always LONG

---

## What We Fixed

**File Modified**: `ml_intraday_v3/live_trading/replay.py`

**New Code (lines 347-356)**:
```python
# FIXED: Use the model's predicted side instead of deriving from score
predicted_side = prediction.get("side", 1)  # 1=LONG, -1=SHORT, 0=no trade

if predicted_side == 0:
    # Neither side has positive EV - override should_trade
    should_trade = False
    reason = "no_positive_ev_either_side"

if should_trade:
    direction = "LONG" if predicted_side > 0 else "SHORT"
```

---

## Current Problem

**Result**: Still 100% LONG, 0% SHORT after fix

**Possible Causes**:
1. ❓ Model not returning 'side' field in predictions
2. ❓ 'side' field always equals 1 (LONG)
3. ❓ Different code path being used (not replay.py)
4. ❓ Model truly believes all setups are LONG (Jan 2026 was bullish)

---

## Model Configuration Verified

✅ Model bundle has `has_side_feature: True`
✅ Model has 'side' feature at index 33
✅ Model type: LGBMClassifier (single model, not dual)
✅ Uses `has_side_feature` path in model_predictor.py (lines 189-237)
✅ That path DOES set `'side': chosen_side` (line 234)

**Expected Behavior**:
- model_predictor evaluates both LONG (side=1) and SHORT (side=-1)
- Calculates score_ev_long and score_ev_short
- Chooses side with better EV
- Returns `'side': chosen_side` in prediction dict

---

## Next Debugging Steps

### Option A: Direct Model Test
Create a standalone script to test model predictions:
```python
# Test if model can generate SHORT predictions

from ml_intraday_v3.live_trading.model_predictor import LiveModelPredictor
import pandas as pd

predictor = LiveModelPredictor("ml_intraday_v3/models/saved/model_bundle.pkl")

# Create features for a bearish scenario
features = pd.Series(0.0, index=predictor.feature_columns)
features['ema_spread'] = -0.01  # Negative = bearish
features['log_return_1'] = -0.005  # Down move
features['trend_strength'] = -0.02  # Downtrend

# Test prediction
pred = predictor.predict(features, use_meta=False, side=None)
print(f"Predicted side: {pred.get('side')}")
print(f"Score EV LONG: {pred.get('score_ev_long')}")
print(f"Score EV SHORT: {pred.get('score_ev_short')}")
```

### Option B: Check Jan 2026 Market Regime
Maybe Jan 2026 was actually bullish and there genuinely are no SHORT setups:
```python
# Check market direction during Jan 2026
import pandas as pd

bars = pd.read_parquet("runs/databento_backtest_*/bar_size=5m/bars.parquet")
returns = bars['close'].pct_change()
print(f"Mean return: {returns.mean():.6f}")
print(f"Cumulative return: {returns.sum():.4f}")
print(f"Up days: {(returns > 0).sum()}")
print(f"Down days: {(returns < 0).sum()}")
```

### Option 3: Add Simpler Debug Logging
Instead of using loop counter `i`, add unconditional logging:
```python
# In replay.py after prediction =predictor.predict():
logger.warning(f"PREDICTION DEBUG: side={prediction.get('side')}, score={prediction.get('score_ev')}")
```

---

## Hypothesis Ranking

**Most Likely → Least Likely**:

1. **60% - Model Legitimately Predicts LONG** 🤔
   - Jan 2026 data was bullish period
   - score_ev_short genuinely < score_ev_long for all bars
   - Not a bug, just bad market conditions for SHORT trades
   - **Test**: Check market returns during Jan 2026

2. **30% - 'side' Field Not Being Set** 🐛
   - Model predictor has a bug in has_side_feature path
   - 'side' not actually in prediction dict
   - Fix in replay.py works but gets default value of 1
   - **Test**: Direct model test with bearish features

3. **8% - Wrong Code Path** 🔀
   - Backtest uses different signal generation code
   - replay.py fix not being applied
   - **Test**: Add logging to verify code path

4. **2% - Feature Values Force LONG** ⚙️
   - 'side' feature in model always = 1
   - Or features systematically bullish
   - **Test**: Inspect actual feature values from bars

---

## Files Involved

**Modified**:
- `ml_intraday_v3/live_trading/replay.py` (fix applied, backup: replay.py.backup)

**To Investigate**:
- `ml_intraday_v3/live_trading/model_predictor.py` (lines 189-237)
- `ml_intraday_v3/models/saved/model_bundle.pkl` (model configuration)
- `ml_intraday_v3/runs/databento_backtest_*/bar_size=5m/bars.parquet` (market data)

---

## Recommendation

**Try Option B first** (check market regime):
- If Jan 2026 was genuinely bullish, 100% LONG is expected
- This would mean the model is working correctly
- The poor performance is due to OTHER issues (stops too tight, wrong entry timing, etc.)

**Then Option A** (direct model test):
- Synthetically create bearish features
- Force-feed them to predictor
- See if it CAN predict SHORT

**Finally Option C** (better logging):
- Add unconditional warning-level logs
- Re-run backtest
- Capture actual prediction values

