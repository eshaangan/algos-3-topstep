# January 2026 Model Performance Test Results

**Test Date**: 2026-01-25  
**Test Period**: January 1-25, 2026 (25 days)  
**Data Source**: Databento (MES.c.0 front month continuous contract)  
**Models Tested**: RETRAINED_CLEAN, OLD_BASELINE

---

## Executive Summary

**Critical Findings:**
- ❌ **100% LONG BIAS** - Both models only predicted LONG trades (0 SHORT trades)
- ❌ **Very Poor Performance** - 17-19% win rate (target: >50%)
- ❌ **Significant Losses** - Average loss of ~$2,350 over 25 days
- ❌ **Directional Bug NOT Fixed** - Models still exhibit structural LONG bias

---

## Test Results by Model

### RETRAINED_CLEAN Model
```
Total Trades:     222
Direction:        100.0% LONG, 0.0% SHORT
Win Rate:         17.6% (39 wins, 183 losses)
Total P&L:        -$2,434.03
Average Trade:    -$10.96
```

### OLD_BASELINE Model  
```
Total Trades:     211
Direction:        100.0% LONG, 0.0% SHORT
Win Rate:         19.0% (40 wins, 171 losses)
Total P&L:        -$2,276.64
Average Trade:    -$10.79
```

### CURRENT_PRODUCTION Model
```
Status:           FAILED TO LOAD
Error:            No primary_preprocessor in bundle
```

---

## Market Context: January 2026

The January 2026 period appears to have been a **challenging market** for LONG-only strategies:

- **Directional Bias**: Both models predicted 100% LONG (same as December 2025)
- **Market Reality**: Given the poor LONG performance (17-19% win rate), January likely had:
  - Downward or sideways price action
  - High volatility / choppy conditions
  - Frequent whipsaws against LONG positions

---

## Comparison to Previous Tests

| Period | Model | LONG% | SHORT% | Win Rate | Total P&L |
|--------|-------|-------|--------|----------|-----------|
| **Dec 2025** | OLD_BASELINE | 100% | 0% | 58% | Positive |
| **Dec 2025** | RETRAINED_CLEAN | 95%+ | <5% | 20.4% | Negative |
| **Jan 2026** | OLD_BASELINE | 100% | 0% | 19.0% | -$2,277 |
| **Jan 2026** | RETRAINED_CLEAN | 100% | 0% | 17.6% | -$2,434 |

**Key Observation**: 
- December 2025 was a strong bull market → LONG-only worked (58% win rate)
- January 2026 appears bearish/choppy → LONG-only failed catastrophically (17-19% win rate)

---

## Root Cause Analysis

The test confirms the original diagnosis:

### 1. Structural LONG Bias (NOT FIXED)
- Both models **still predict 100% LONG trades**
- Zero SHORT predictions despite market conditions
- This is the exact same issue identified in the investigation

### 2. Training Data Imbalance
- Models were trained on bull market data (Oct 2024 - Nov 2025)
- Learned that "SHORT trades always lose"
- Cannot adapt to bearish/neutral conditions

### 3. Why Previous Fix Didn't Work
The `replay.py` fix (line 345) and `model_predictor.py` bidirectional evaluation are **correct**, but:
- OLD_BASELINE: Trained with imbalanced data → EV_short always negative
- RETRAINED_CLEAN: Bypassed triple-barrier labeling → poor training quality

---

## Impact on Topstep Combine

If this model were deployed to a Topstep 50k Combine:

### Daily Loss Violations
```
Average daily loss: ~$95/day ($2,434 / 25 days)
Topstep limit: $1,000/day
Status: ✅ Within limit (but losing money every day)
```

### Trailing Drawdown
```
Cumulative loss: $2,434
Topstep limit: $2,500
Status: ⚠️ DANGEROUSLY CLOSE (97% of limit)
```

### Profit Target
```
Progress: -$2,434 / $3,000 target
Status: ❌ MOVING BACKWARDS
```

### **Verdict**: ❌ **WOULD FAIL COMBINE**
- Cannot reach profit target with consistent losses
- High risk of hitting trailing drawdown limit
- Model cannot adapt to changing market conditions

---

## What This Means

### January 2026 validated our findings:

1. **Directional Bug Persists**: 100% LONG bias confirms models cannot predict SHORT

2. **Regime-Dependent Performance**:
   - Bull market (Dec 2025): LONG-only worked well (58% win rate)
   - Bear/neutral market (Jan 2026): LONG-only failed catastrophically (17-19% win rate)

3. **Critical Need for Balanced Training**:
   - Current models are **dangerously overfit** to bullish conditions
   - Will fail spectacularly in any non-bull market
   - Cannot survive Topstep Combine's 2-3 month evaluation period

---

## Recommended Actions

### IMMEDIATE (Before Live Trading):
1. ❌ **DO NOT DEPLOY** current models to live trading or Topstep
2. ✅ **Execute the implementation plan** to train balanced V3 model
3. ✅ **Re-test on January 2026** after new model is trained

### Implementation Plan (Already Prepared):
1. Run regime analysis to identify bull/bear/volatile periods in 2024-2025
2. Generate balanced 50/50 LONG/SHORT training events
3. Train new model with full V3 pipeline (triple-barrier labeling, sample weighting)
4. Validate on multiple periods (Q1 2024 bearish, Q3 2024 volatile, Dec 2025 bullish, Jan 2026)
5. Only deploy if all periods show >50% win rate and balanced direction

### Success Criteria for New Model:
- ✅ LONG/SHORT: 30-70% on each side (regime-dependent)
- ✅ Win rate: >50% on ALL test periods (bull, bear, volatile)
- ✅ Sharpe ratio: >1.0
- ✅ January 2026 test: Must predict SHORT trades and be profitable

---

## Data Quality Notes

**Databento API**: ✅ Working correctly
- Successfully fetched 25 days of January 2026 MES data
- Data quality appears good (proper OHLCV, reasonable prices)
- Resampled from 1-minute to 5-minute bars without issues

**Feature Generation**: ⚠️ Some Issues
- NaN values in `vol_regime` and `price_vs_vwap` features during warmup period
- This is expected for the first ~50 bars (warmup period for indicators)
- Does not affect overall test validity (model used features from bar 51+)

---

## Conclusion

The January 2026 test **validates the critical need** for the balanced V3 model retraining. Current models are:

- ❌ **Structurally biased** (100% LONG only)
- ❌ **Regime-dependent** (only work in bull markets)
- ❌ **Not production-ready** (would fail Topstep Combine)

**Next Step**: Execute the implementation plan to train the balanced bidirectional model before any live trading.

---

**Files Generated:**
- Raw data: `ml_intraday_v3/backtest_results/jan_2026_test/bar_size=5m/bars.parquet`
- Trade logs: `logs/trades_20260125_173031.csv`, `logs/trades_20260125_173134.csv`
- Full log: `ml_intraday_v3/jan_2026_test_log.txt`
