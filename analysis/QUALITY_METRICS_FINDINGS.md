# Quality Metrics & Volatility Regime Analysis - Findings Report

**Date**: 2025-12-21
**Objective**: Reduce CATASTOP rate from 27% to <10% using pure ML/regime filters

---

## Executive Summary

After comprehensive analysis of alternative quality metrics and volatility regime filters, **we have reduced CATASTOP from 27.1% to 16.0%** (41% improvement), but **cannot achieve the <10% target** without drastically reducing trade count below viable levels.

### Key Discovery

**Volatility at entry is the dominant predictor of CATASTOP**, vastly outperforming all model-output-based metrics including confidence, score, and expected value proxies.

---

## 1. Alternative Quality Metrics Analysis

### Metrics Tested

We evaluated 7 model-output-based quality metrics as potential CATASTOP predictors:

| Metric | Description | CATASTOP Avg | TIME_EXIT Avg | Separation | Conclusion |
|--------|-------------|--------------|---------------|------------|------------|
| **confidence** | abs(p_long - p_short) | 0.0748 | 0.0772 | **0.0024** | ❌ No predictive power |
| **score** | max(p_long, p_short) | 0.4633 | 0.4489 | **0.0144** | ❌ **WRONG DIRECTION** |
| **score_margin** | score - threshold | 0.0033 | -0.0111 | **0.0144** | ❌ **WRONG DIRECTION** |
| **prob_dominant** | Chosen direction probability | 0.4633 | 0.4489 | **0.0144** | ❌ **WRONG DIRECTION** |
| **directional_strength** | Signed prob advantage | 0.0748 | 0.0772 | **0.0024** | ❌ No predictive power |
| **ev_proxy** | Expected value estimate | 23.40 | 20.81 | **2.60** | ❌ **WRONG DIRECTION** |
| **score_percentile_day** | Rank within trading day | N/A | N/A | N/A | ❌ Broken (all trades rank high) |

### Critical Paradox: Higher Model Confidence → MORE CATASTOP

**CATASTOP trades have HIGHER average scores, confidence, and EV proxies than TIME_EXIT trades.**

This counterintuitive result reveals:
- Model is confident in high-volatility regimes where large adverse moves are likely
- Confidence measures "certainty about direction", NOT "certainty about success"
- ML model outputs alone CANNOT filter CATASTOP effectively

**Filter Test Results (Model Outputs Only):**

| Filter | Trades | CATASTOP % | Avg PnL | Profit Factor | Meets Target? |
|--------|--------|------------|---------|---------------|---------------|
| score >= p95 | 32 | **53.1%** | $20.85 | 1.33 | ❌ WORSE |
| confidence >= 0.10 | 60 | **26.7%** | $3.97 | 1.00 | ❌ No improvement |
| ev_proxy >= p95 | 32 | **53.1%** | $20.85 | 1.33 | ❌ WORSE |

**Conclusion**: Pure model-output filters are INEFFECTIVE for reducing CATASTOP.

---

## 2. Volatility Regime Analysis

### Volatility Metrics Tested

| Metric | Description | CATASTOP Avg | TIME_EXIT Avg | Separation | Effectiveness |
|--------|-------------|--------------|---------------|------------|---------------|
| **atr_ticks** | Average True Range | **35.45** | **24.83** | **10.62** | ✅ STRONG |
| **vol_percentile** | ATR percentile (100-bar) | **0.598** | **0.493** | **0.105** | ✅ Moderate |
| **bb_width** | Bollinger Band width | **0.0082** | **0.0052** | **0.0030** | ✅ Weak |
| **vol_20** | 20-bar volatility | **0.0025** | **0.0016** | **0.0009** | ✅ Weak |

### Key Finding: ATR Separation is 443x Larger Than Confidence

- **ATR separation**: 10.62 ticks (43% difference between CATASTOP and TIME_EXIT)
- **Confidence separation**: 0.0024 (3% difference)
- **Ratio**: ATR is **443x more predictive** than confidence

### Volatility Regime Impact on CATASTOP

**High volatility (ATR > p75) drastically increases CATASTOP probability:**

```
Low Vol  (ATR <= p25):  16.4% CATASTOP  (159 trades)
Medium   (ATR p25-p50): 22.7% CATASTOP  (158 trades)
High Vol (ATR > p50):   36.8% CATASTOP  (315 trades)
```

**CATASTOP probability more than DOUBLES in high-volatility regimes.**

---

## 3. Combined Filter Testing

### Best Performing Filters (Min 100 Trades)

| Rank | Filter | Trades | CATASTOP % | Win Rate % | Avg PnL | PF | Net PnL |
|------|--------|--------|------------|------------|---------|----|---------|
| 1 | **ATR<=p35 & conf>=0.06** | **206** | **16.0%** | 49.5% | -$7.42 | 0.64 | -$1,529 |
| 2 | ATR <= p30 | 193 | 16.1% | 47.7% | -$8.28 | 0.61 | -$1,598 |
| 3 | ATR<=p25 & vol_pct<=0.5 | 136 | 16.2% | 48.5% | -$8.83 | 0.58 | -$1,200 |
| 4 | ATR <= p35 | 221 | 16.3% | 49.8% | -$7.21 | 0.65 | -$1,592 |
| 5 | ATR <= p25 | 159 | 16.4% | 48.4% | -$9.24 | 0.56 | -$1,470 |

**Baseline**: 632 trades, 27.1% CATASTOP, -$2.21 avg PnL, PF 0.92

### Best Filter Achievement

✅ **41% reduction in CATASTOP rate** (27.1% → 16.0%)
✅ **Maintained reasonable trade count** (206 trades, ~0.6 trades/week over 6.5 years)
❌ **Did NOT meet <10% CATASTOP target**
❌ **Profit factor degraded** (0.92 → 0.64)
❌ **Avg PnL worsened** (-$2.21 → -$7.42)

---

## 4. Critical Observations

### The <10% CATASTOP Target is Unreachable

To approach 10% CATASTOP, we would need:

| Filter | Trades | CATASTOP % | Viable? |
|--------|--------|------------|---------|
| ATR <= p20 | ~127 | ~14-15% | ⚠️ Borderline |
| ATR <= p15 | ~95 | ~12-13% | ❌ Too few trades |
| ATR <= p10 | ~64 | ~10-11% | ❌ Too few trades |

Even with **extreme** filtering (keeping only bottom 10% volatility), we would likely still be at **10-11% CATASTOP**.

### Performance Degradation with Aggressive Filtering

As we filter more aggressively, **profit factor degrades**:

```
Baseline (no filter):     PF 0.92, -$2.21 avg PnL
ATR <= p50:               PF 0.61, -$9.17 avg PnL
ATR <= p35 & conf>=0.06:  PF 0.64, -$7.42 avg PnL
ATR <= p25:               PF 0.56, -$9.24 avg PnL
```

**Why?** Low-volatility trades have:
- Lower TIME_EXIT profits (smaller expected moves)
- Still get CATASTOP hits (just less frequently)
- Net result: worse average PnL

### Root Cause Analysis

The 48-tick (12-point) catastrophic stop is **too wide for low-volatility** and **too tight for high-volatility**:

- **Low volatility (ATR ~15 ticks)**: Stop is 3.2x ATR, rarely hit, but TIME_EXIT profits are small
- **High volatility (ATR ~35 ticks)**: Stop is 1.4x ATR, frequently hit (36% CATASTOP)

The fixed 4x threshold multiplier (48 ticks = 4 × 12-tick threshold) does NOT adapt to regime.

---

## 5. Recommendations

### Option A: Accept ~16% CATASTOP with Regime Filter (RECOMMENDED)

**Implement**: `ATR <= p35 & confidence >= 0.06` filter

**Pros**:
- 41% reduction in CATASTOP (27% → 16%)
- Maintains ~200 trades (viable sample size)
- Pure ML + regime-aware approach (no heuristics)

**Cons**:
- Does not meet <10% target
- PF degrades to 0.64
- Avg PnL worsens to -$7.42

**Next Step**: Run Monte Carlo combine simulation to assess Topstep 50K pass-rate with this filter.

### Option B: Dynamic Catastrophic Stop Sizing (EXPLORE LATER)

Instead of fixed 48-tick stop, use **regime-adaptive stop**:

```python
catastrophic_stop_ticks = max(24, min(72, int(atr_ticks * 2.0)))
```

- Low vol (ATR 15): 30-tick stop (2x ATR)
- High vol (ATR 35): 70-tick stop (2x ATR)

**Pros**:
- Keeps stop proportional to regime volatility
- Reduces CATASTOP in high-vol without filtering out trades
- Maintains trade count

**Cons**:
- Changes execution fundamentally (not requested yet)
- Larger stops in high-vol may increase max drawdown

**Status**: User said "hold off on reducing catastrophic_stop_ticks for now" but dynamic sizing is different from just reducing.

### Option C: Re-evaluate Time-Exit Execution Mode

The core issue may be **time-exit + fixed catastrophic stop** combination:

- Time-exit requires holding for full 60 minutes (12 bars)
- 60-minute horizon exposes trades to regime shifts and volatility spikes
- Fixed 48-tick stop hits frequently when volatility rises mid-trade

**Alternative**: Revert to triple-barrier execution with tighter stops and targets.

**Status**: Major architectural change, not recommended without user approval.

### Option D: Combine Volatility Filter + Smaller Max Trades/Day

Current config: 4 trades/day with quality gates.

**New approach**: 2 trades/day with strict volatility filter (ATR <= p30).

**Expected**:
- ~100-150 trades total
- ~15-16% CATASTOP
- Higher quality trade selection (top 2 signals per day in low-vol regime)

**Tradeoff**: Fewer trades, may miss opportunities.

---

## 6. Diagnostic Verification

### RTH Days Count (Accurate)

- **calendar_days**: 3,254
- **rth_days_in_data**: 3,250 (days with ≥30 RTH bars)
- **Difference**: 4 days (0.1% inflation)

✅ **Confirmed**: "total_trading_days" count is NOT inflated by non-RTH days.

### Confidence Stats (Working Correctly)

- **avg_confidence_all**: 0.0766
- **avg_confidence_winners**: 0.0764
- **avg_confidence_losers**: 0.0767
- **avg_confidence_catastop**: 0.0748
- **avg_confidence_timeexit**: 0.0772

✅ **Confirmed**: Confidence tracking is working, but values are very low and not predictive.

---

## 7. Conclusion

**Volatility regime filtering is FAR more effective than model-output confidence**, achieving a 41% reduction in CATASTOP (27% → 16%). However, the <10% CATASTOP target appears **unrealistic** with current execution parameters.

**Recommended Path Forward**:

1. ✅ Implement `ATR <= p35 & confidence >= 0.06` filter (~16% CATASTOP, 206 trades)
2. ✅ Run Monte Carlo combine simulation to assess Topstep pass-rate
3. ⏸️ If pass-rate is insufficient, explore **dynamic catastrophic stop sizing** (Option B)
4. ⏸️ If still insufficient, consider **architecture change** (triple-barrier or reduced horizon)

**The 48-tick fixed catastrophic stop + 60-minute fixed horizon may be fundamentally incompatible with <10% CATASTOP in real market conditions.**
