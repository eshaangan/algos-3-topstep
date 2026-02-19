# Paper Trading Setup Guide

## Overview

Paper trading validates the production model on live data WITHOUT real trades. This is Phase 5 of the deployment plan.

**Duration**: 1-2 weeks  
**Goal**: Verify model performance before risking real capital on Topstep combine

---

## Quick Start

### Option 1: Test on Historical Data (Immediate)

```bash
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep

# Run paper trading on Jan-Feb 2026 OOS data
python3 << 'ENDPY'
import pickle
import pandas as pd
import numpy as np

# Load model
with open('ml_intraday_v3/models/final/batch1_exp_00159_production_model.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
features = bundle['feature_columns']

# Load data
df = pd.read_parquet('data/processed/jan_feb_2026_oos_for_validation.parquet')

# Build features (same as training)
df['returns'] = df['close'].pct_change()
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
df['price_ma5'] = df['close'].rolling(5).mean()
df['price_ma20'] = df['close'].rolling(20).mean()
df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
df['atr_ratio'] = df['atr'] / df['close']

# Generate predictions
df_clean = df.dropna()
X = df_clean[features].values
predictions = model.predict_proba(X)

df_clean['p_target'] = predictions[:, 1]
df_clean['signal'] = df_clean['p_target'] > 0.55

print(f"Signals generated: {df_clean['signal'].sum()} / {len(df_clean)}")
print(f"Signal rate: {100*df_clean['signal'].sum()/len(df_clean):.1f}%")
print(f"\nReady for live paper trading!")
ENDPY
```

**Expected Output**:
- Signals generated: ~3,400 / 7,400
- Signal rate: ~46%
- Ready for live!

---

### Option 2: Live Paper Trading (Real-Time)

**Requirements**:
1. Live data feed (Databento, Interactive Brokers, etc.)
2. Model integration in `live_trading/model_predictor.py`
3. Feature generation matching training exactly

**Integration Steps**:

1. **Update `live_trading/model_predictor.py`**:

```python
# Replace existing model loading with:
import pickle

class ModelPredictor:
    def __init__(self):
        # Load production model
        with open('ml_intraday_v3/models/final/batch1_exp_00159_production_model.pkl', 'rb') as f:
            bundle = pickle.load(f)
        
        self.model = bundle['model']
        self.feature_cols = bundle['feature_columns']
        self.confidence_threshold = 0.55  # Adjust to 0.60 to reduce trade frequency
    
    def predict(self, df):
        """Generate predictions on live data."""
        # Build features (MUST match training exactly!)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_ma5'] = df['close'].rolling(5).mean()
        df['price_ma20'] = df['close'].rolling(20).mean()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        
        # ATR
        tr = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        })
        df['atr'] = tr.max(axis=1).rolling(14).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Predict
        X = df[self.feature_cols].values
        predictions = self.model.predict_proba(X)
        
        df['p_target'] = predictions[:, 1]
        df['p_stop'] = predictions[:, 0]
        
        return df
    
    def should_trade(self, row):
        """Determine if we should take this signal."""
        if row['p_target'] > self.confidence_threshold:
            return {'side': 1, 'confidence': row['p_target']}  # Long
        elif row['p_stop'] > self.confidence_threshold:
            return {'side': -1, 'confidence': row['p_stop']}  # Short
        return {'side': 0, 'confidence': 0}
```

2. **Run in Paper Trading Mode**:

```bash
# In live_runner.py, set:
PAPER_TRADING = True  # Don't execute real orders

# Run
python ml_intraday_v3/live_trading/live_runner.py
```

---

## Monitoring Checklist

### Daily Checks (10 min/day)

✅ **Signal Frequency**:
- Expected: 8-15 signals/day
- If <5/day → Model too conservative (lower threshold to 0.50)
- If >20/day → Model too aggressive (raise threshold to 0.60)

✅ **Prediction Distribution**:
- Mean p(target): ~0.53-0.57
- Std: ~0.15-0.20
- If predictions cluster at 0.0 or 1.0 → Feature drift or bug

✅ **"Would-Be" PnL**:
- Track what PnL WOULD have been if trades were real
- Compare to OOS test ($1,178 in 28 days = $42/day average)

✅ **Win Rate**:
- Expected: 45-50%
- If <40% → Investigate (market regime change? feature issue?)

✅ **Max Drawdown**:
- Keep below $500 in paper trading
- If exceeds $800 → Pause and investigate

---

### Weekly Review (30 min/week)

📊 **Performance vs Expectations**:
```
Expected (from OOS test):
- PnL: $42/day average
- Win Rate: 45.5%
- Trades/Day: 33 (HIGH - consider raising threshold)
- Max DD: <$700

Actual paper trading:
- PnL: $___/day
- Win Rate: ___%
- Trades/Day: ___
- Max DD: $___
```

📈 **Feature Drift Detection**:
- Compare feature distributions to training data
- Check for NaN or inf values
- Verify feature calculations match training

🔧 **Adjustments**:
- If trade frequency too high (>20/day): Raise threshold to 0.60
- If win rate too low (<42%): Check for bugs, feature drift
- If DD too high: Review risk parameters (PT/SL)

---

## Success Criteria (Green Light for Live Trading)

Must achieve ALL of these for **at least 1 week**:

1. ✅ **Positive PnL** (any amount >$0)
2. ✅ **Win Rate >45%**
3. ✅ **Max DD <$500**
4. ✅ **Signal frequency 8-15/day** (stable)
5. ✅ **No technical errors** (bugs, crashes, missing data)
6. ✅ **Predictions well-distributed** (0.4-0.7 range, not clustered)

---

## Red Flags (Stop and Investigate)

❌ **Immediate Stop**:
- Win rate drops below 35%
- Max drawdown exceeds $800
- Signal frequency drops to <3/day or exceeds 30/day
- Predictions cluster at extremes (all 0.0 or 1.0)
- Repeated technical errors

⚠️ **Warning Signs**:
- Win rate 40-42% (borderline)
- Drawdown $600-800 (approaching limit)
- Feature values outside training distribution
- Slow performance degradation over days

---

## Expected Results

Based on OOS test (Jan 1 - Feb 9, 2026):

| Metric | Expected Value |
|--------|----------------|
| **Daily PnL** | ~$42/day average |
| **Win Rate** | 45-47% |
| **Trades/Day** | 8-15 (adjust threshold if needed) |
| **Max DD** | <$700 |
| **Profit Factor** | 1.05-1.10 |

**If paper trading matches these**, you're ready for Topstep combine!

---

## Troubleshooting

### Issue: No signals generated

**Check**:
1. Features calculated correctly? (print feature values)
2. Model loaded? (print model type)
3. Threshold too high? (try 0.50 temporarily)
4. Data has NaN? (check for missing values)

### Issue: Win rate too low (<40%)

**Check**:
1. Feature drift (compare distributions to training)
2. Market regime changed (check VIX, trend)
3. Bug in exit logic (PT/SL calculated correctly?)
4. Slippage/fees not accounted for?

### Issue: Too many trades (>25/day)

**Fix**: Raise confidence threshold to 0.60 or 0.65

### Issue: Too few trades (<5/day)

**Fix**: Lower confidence threshold to 0.50 or 0.52

---

## After Paper Trading: Go Live

Once paper trading succeeds for 1-2 weeks:

1. **Deploy to Topstep Combine** (50k account)
2. **Set Conservative Risk**:
   - $100/trade (1 MES contract)
   - $400 daily loss limit
   - $1,000 max drawdown
3. **Monitor Daily** (20 min/day)
4. **Circuit Breakers**:
   - 5 consecutive losses → pause
   - Daily loss hits $300 → stop for day
   - 3 consecutive red days → review strategy

---

## Files

- **Paper Trading System**: `ml_intraday_v3/paper_trading/paper_trader.py`
- **Production Model**: `ml_intraday_v3/models/final/batch1_exp_00159_production_model.pkl`
- **OOS Test Data**: `data/processed/jan_feb_2026_oos_for_validation.parquet`
- **Integration Example**: `ml_intraday_v3/models/final/integration_example.py`

---

**Ready to start? Run the Quick Start code above!**
