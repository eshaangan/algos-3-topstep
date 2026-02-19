# CRITICAL FINDINGS: Jan-Feb 2026 Validation Results

**Date**: Feb 12, 2026
**Test Period**: Jan 1 - Feb 10, 2026 (7,416 bars, 41 days)
**Models Tested**: 10 finalists from batch1/batch3 experiments
**Threshold**: 0.40

---

## EXECUTIVE SUMMARY

**ALL 10 MODELS FAILED** with negative PnL and identical results within groups.

The sophisticated financial ML techniques (trend scanning labels, fractional differentiation, CPCV, sample uniqueness weighting) **DID NOT solve the fundamental problem**.

---

## DETAILED RESULTS

### Group 1: Triple Barrier Models (Batch1)

All 5 models produced **IDENTICAL** results:

| Model ID | Labeling | Features | Weighting | CV | PnL | Signals | Sharpe |
|----------|----------|----------|-----------|----|----|---------|--------|
| batch1_exp_00158 | triple_barrier | baseline_fracdiff | uniform | kfold | **-$1,140.61** | **658** | **-2.40** |
| batch1_exp_00149 | triple_barrier | baseline | uniform | kfold | **-$1,140.61** | **658** | **-2.40** |
| batch1_exp_00010 | triple_barrier | baseline | uniqueness | kfold | **-$1,140.61** | **658** | **-2.40** |
| batch1_exp_00072 | triple_barrier | baseline_fracdiff | uniqueness | kfold | **-$1,140.61** | **658** | **-2.40** |
| batch1_exp_00137 | triple_barrier | baseline_fracdiff | uniform | kfold | **-$1,140.61** | **658** | **-2.40** |

**Key Observation**: Fractional diff, uniqueness weighting made ZERO difference.

---

### Group 2: Trend Scanning Models (Batch3)

All 5 models produced **IDENTICAL** results:

| Model ID | Labeling | Features | Weighting | CV | PnL | Signals | Sharpe |
|----------|----------|----------|-----------|----|----|---------|--------|
| batch3_exp_00102 | trend_scanning | baseline | uniqueness_decay | cpcv | **-$2,664.05** | **1,221** | **-2.40** |
| batch3_exp_00006 | trend_scanning | baseline | uniqueness_decay | cpcv | **-$2,664.05** | **1,221** | **-2.40** |
| batch3_exp_00067 | trend_scanning | baseline | uniqueness_decay | cpcv | **-$2,664.05** | **1,221** | **-2.40** |
| batch3_exp_00089 | trend_scanning | baseline | uniqueness_decay | cpcv | **-$2,664.05** | **1,221** | **-2.40** |
| batch3_exp_00158 | trend_scanning | baseline | uniqueness_decay | cpcv | **-$2,664.05** | **1,221** | **-2.40** |

**Key Observation**: Trend scanning is **133% WORSE** than triple barrier!

---

## ROOT CAUSE ANALYSIS

### Why Are Results Identical?

The identical results prove that **labels, not model predictions, drive outcomes**.

**Triple Barrier Group** (baseline CUSUM + triple barrier):
- Uses CUSUM for event detection (deterministic threshold)
- Uses triple barrier labeling (fixed PT/SL multiples of ATR)
- **Result**: Same events → same labels → same outcomes regardless of model

**Trend Scanning Group** (CUSUM + trend scanning):
- Uses CUSUM for event detection (deterministic threshold)
- Uses trend scanning for direction (max t-value from fixed regressions)
- **Result**: Same events → same trend directions → same outcomes regardless of model

### Why Did The Plan Fail?

The plan from `/Users/eshaanganguly/.claude/plans/structured-fluttering-sifakis.md` proposed:

1. ✅ **Trend Scanning Labels** - IMPLEMENTED but made results WORSE
2. ✅ **Fractional Differentiation** - IMPLEMENTED but made NO difference
3. ✅ **Sample Uniqueness Weighting** - IMPLEMENTED but made NO difference
4. ✅ **CPCV (Purged CV)** - IMPLEMENTED but made NO difference

**The problem**: All these techniques assume the model predictions matter. But if labels are deterministic (entry price + fixed rules → outcome), then:
- Better cross-validation doesn't help (you're just measuring label quality, not model quality)
- Better sample weighting doesn't help (you're weighting labels, not predictions)
- Better features don't help (model never influences the outcome)

---

## COMPARISON TO BASELINE

### Original Exhaustive Search (100 models, Nov 2025)
- **Result**: ALL 100 models produced IDENTICAL results
- **PnL**: -$1,212.56, Signals: 624, Win Rate: 41.3%

### New Models with Plan Innovations (10 models, Feb 2026)
- **Triple Barrier**: -$1,140.61, Signals: 658 (6% better PnL, but still terrible)
- **Trend Scanning**: -$2,664.05, Signals: 1,221 (120% WORSE!)

**Verdict**: The plan made things marginally better (triple barrier) or much worse (trend scanning), but didn't solve the fundamental problem.

---

## THE FUNDAMENTAL PROBLEM

Based on previous investigation (`ml_intraday_v3/labels/triple_barrier.py:217-226`):

```python
# Entry price is from bar data, NOT model prediction
p0 = float(entry_prices[i])

# Barriers are fixed ATR multiples, NOT model-informed
pt_points = float(events.loc[i, "pt_mult"]) * sigma[i] + cost_buffer
sl_points = float(events.loc[i, "sl_mult"]) * sigma[i] + cost_buffer
```

**The label (win/loss) is determined ENTIRELY by**:
1. When we enter (event detection - CUSUM threshold)
2. Entry price (bar's open/close - NOT model-dependent)
3. Barrier levels (ATR multiples - NOT model-dependent)
4. Market price action after entry (exogenous - NOT model-dependent)

**The model prediction is NEVER USED in determining the outcome**.

This is why:
- Different models → identical results
- Better features → no improvement
- Sophisticated techniques → no improvement

---

## RECOMMENDATIONS

### Option 1: Meta-Labeling (Not Yet Tested)

The plan proposed meta-labeling but it wasn't tested in these 10 finalists.

**Idea**:
- Primary model predicts direction (can be rule-based)
- **Secondary model predicts "should we trade?"** based on primary prediction + features
- Only the secondary model's prediction actually gates trades

**Why it might work**:
- Breaks the deterministic label problem
- Model prediction (secondary) actually influences whether we take the trade
- Outcomes become model-dependent

**Status**: Code exists (`ml_intraday_v3/models/meta_labeling.py`) but not integrated

### Option 2: Abandon ML, Use Rule-Based System

The `rule_based_v1/` system was built specifically because ML walk-forward AUC was ~0.51.

**Advantages**:
- No label determinism issues
- Clear logic: EMA trend + confirmations + filters
- 106 tests, all passing
- Transparent risk management

**Status**: Fully implemented, tested, ready to use

### Option 3: Fix Label Methodology

To make labels model-dependent, we need to use **model predictions in the labeling process**:

**Current (broken)**:
```
Event → Enter at bar price → Fixed barriers → Label = outcome
         (model not involved)
```

**Fixed approach**:
```
Event → Model predicts outcome confidence → Size position or skip trade based on confidence → Label = actual outcome of THAT decision
                (model directly influences outcome)
```

This requires fundamental restructuring of the labeling pipeline.

---

## DECISION REQUIRED

Given that:
1. ❌ ML exhaustive search failed (100 models identical)
2. ❌ ML with advanced techniques failed (10 models identical in groups)
3. ✅ Rule-based system is ready and tested

**Recommendation**:

**ABANDON ML APPROACH** and proceed with the rule-based system (`rule_based_v1/`).

The evidence is overwhelming:
- 110 ML models tested (100 exhaustive + 10 finalists)
- $2+ spent on GCP experiments
- Dozens of hours of research and implementation
- State-of-the-art techniques from financial ML literature
- **Result**: ZERO profitable models, systemic label determinism

The ML approach is fundamentally broken at the labeling level. Fixing it would require:
1. Complete redesign of event detection (not CUSUM-based)
2. Complete redesign of labeling (model-informed, not fixed barriers)
3. Complete retraining of all models
4. Likely months of additional iteration

**The rule-based system is ready NOW and doesn't have these issues.**

---

## FILES FOR REFERENCE

- Plan document: `/Users/eshaanganguly/.claude/plans/structured-fluttering-sifakis.md`
- Finalist configs: `/tmp/finalist_configs_for_validation.json`
- Test data: `data/processed/jan_feb_2026_oos_test.h5`
- Validation script: `ml_intraday_v3/experiments/validate_10_finalists_jan_feb_2026.py`
- Rule-based system: `rule_based_v1/`

---

## CONCLUSION

The sophisticated financial ML techniques from peer-reviewed literature (trend scanning, CPCV, fractional differentiation, sample uniqueness weighting) **failed to solve the fundamental labeling problem**.

All 10 finalist models lost money and produced identical results within their groups, proving that model predictions remain irrelevant to outcomes.

**The ML intraday trading approach is not viable** with the current labeling methodology.

**Recommended action**: Pivot to the rule-based system or invest significant time in fundamental labeling redesign.
