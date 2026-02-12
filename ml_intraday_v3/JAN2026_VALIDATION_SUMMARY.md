# Jan 2026 Out-of-Sample Validation Summary
**Date**: 2026-02-11
**Test Period**: Jan 1-30, 2026 (5,748 bars, 934 events)
**Models**: Baseline vs Candidate (exhaustive_exp_00336)
**Thresholds Tested**: 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60

---

## 🚨 **Executive Summary: STRONG NO-GO**

**Both models FAIL on true out-of-sample Jan 2026 data.**

The candidate model shows **NEGATIVE expected value at ALL thresholds** tested. Even when lowering the confidence threshold to enable trading, the model produces consistent losses.

---

## 📊 **Candidate Model Results by Threshold**

| Threshold | Signals | Signal Rate | Win Rate | Total PnL | Mean PnL | Sharpe | Max DD | Profit Factor |
|-----------|---------|-------------|----------|-----------|----------|--------|--------|---------------|
| **0.30** | 928 | 100.0% | 41.7% | **-$1,885** | -$2.03 | -2.54 | -$1,891 | 0.69 |
| **0.35** | 373 | 40.2% | 44.0% | **-$586** | -$1.57 | -2.12 | -$587 | 0.74 |
| **0.40** | 208 | 22.4% | 42.8% | **-$297** | -$1.43 | -1.85 | -$309 | 0.77 |
| **0.45** | 9 | 1.0% | 33.3% | **-$46** | -$5.12 | -7.68 | -$63 | 0.33 |
| **0.50+** | 0 | 0.0% | N/A | N/A | N/A | N/A | N/A | N/A |

### Key Findings

1. **No Positive Expectancy**: At every threshold, total PnL is negative
2. **Best Threshold (0.40)**: Still loses -$297 on 208 signals
3. **Win Rate Insufficient**: Even at 44% win rate (threshold 0.35), average loser > average winner
4. **Profit Factor < 1.0**: All thresholds show PF < 1.0 (need >1.0 for profitability)
5. **Poor Sharpe Ratios**: All Sharpe ratios are negative (worst: -7.68 at 0.45)

---

## 📉 **Baseline Model Comparison**

| Threshold | Signals | Win Rate | Total PnL | Sharpe |
|-----------|---------|----------|-----------|--------|
| 0.30 | 928 | 41.7% | -$1,885 | -2.54 |
| 0.35 | 928 | 41.7% | -$1,885 | -2.54 |
| 0.40 | 928 | 41.7% | -$1,885 | -2.54 |
| 0.45 | 890 | 41.8% | -$1,775 | -2.56 |
| 0.50 | 143 | 43.4% | -$274 | -2.05 |
| **0.55** | **6** | **50.0%** | **+$2.24** | **0.37** |
| 0.60 | 0 | N/A | N/A | N/A |

**Baseline Observation**: The baseline only shows positive PnL at threshold 0.55 with 6 signals (+$2.24) - essentially random luck, not systematic edge.

---

## 🔍 **Detailed Analysis: Threshold 0.40 (Best for Candidate)**

This was identified as the "optimal" threshold for the candidate model, but results are still unacceptable:

```
Model: CANDIDATE
Threshold: 0.40
----------------------------------------
Signals:        208 (22.4% of events)
Win Rate:       42.8% (89W / 119L)
Avg Winner:     $11.15
Avg Loser:      $-10.84
Total PnL:      -$297.41
Mean PnL:       -$1.43 per trade
Sharpe Ratio:   -1.85
Max Drawdown:   -$308.73
Profit Factor:  0.77 (need > 1.0)
```

### Why This Fails

1. **Win Rate Too Low**: 42.8% with avg winner ≈ avg loser means you need 50%+ to break even
2. **Negative Mean**: -$1.43 per trade × 208 trades = -$297 total
3. **Large Drawdown**: -$309 max DD would violate Topstep daily loss limit ($500) rapidly
4. **Profit Factor**: 0.77 means for every $1 won, you lose $1.30

---

## 🎯 **Optimal Threshold Analysis**

The script analyzed all thresholds to find the "best" settings:

### Best by Sharpe Ratio: Threshold 0.40
- Sharpe: **-1.85** (still negative)
- Total PnL: -$297
- Signals: 208

### Best by Total PnL: Threshold 0.45
- Total PnL: **-$46** (least loss, but only 9 signals)
- Sharpe: -7.68 (terrible)
- Signals: 9

### Best by Profit Factor: Threshold 0.40
- Profit Factor: **0.77** (still < 1.0)
- Total PnL: -$297
- Signals: 208

**Conclusion**: There is NO viable threshold. Even the "best" configurations lose money.

---

## 🚫 **Why the Candidate Model Fails**

### 1. Distribution Mismatch
The model was trained on 12-month rolling window ending ~Nov 2025, but Jan 2026 exhibits different market behavior:
- Different volatility regime
- Different price action patterns
- Features (especially momentum) don't generalize

### 2. Overfitting to CV Splits
Exhaustive search reported:
- **CV AUC**: 0.68 (looked promising)
- **True Holdout AUC (Dec 2025)**: 0.60 (degraded)
- **Jan 2026 Performance**: Negative EV (complete failure)

The model optimized for CV performance, not real-world profitability.

### 3. Calibration Breakdown
The isotonic calibration was fit on historical data. On Jan 2026:
- Probabilities compressed to [0.33, 0.45]
- No signals above 0.50 threshold
- Calibration doesn't reflect true win probabilities

### 4. Win Rate Math Doesn't Work
With PT=4.0 and SL=3.0 labels:
- Need ~43% win rate to break even (3/(3+4) = 0.43)
- Candidate achieves 42.8% at best threshold
- Just below breakeven, compounded over 208 trades = -$297

---

## ⚖️ **Comparison to Dec 2025 Validation**

| Metric | Dec 2025 | Jan 2026 |
|--------|----------|----------|
| AUC | 0.596 | Not calculated* |
| Signals (p>0.55) | 0 | 0 |
| Best Threshold | 0.40 | 0.40 |
| Total PnL @ 0.40 | N/A | -$297 |
| Win Rate @ 0.40 | N/A | 42.8% |

*Jan 2026 focused on PnL metrics rather than AUC since we know from Dec that AUC doesn't correlate with profitability.

---

## 🔄 **Next Steps & Recommendations**

### Option 1: Test Alternative Models from Exhaustive Search ✅ **RECOMMENDED**

The top-20 ranked models from the exhaustive search have different configurations:

```bash
# View top 20 candidates
cat /tmp/phasefull_results/phasefull_ranked_top20.csv

# Promote alternative candidate
cd ml_intraday_v3/experiments
python promote_top_exhaustive_model.py --exp_id exhaustive_exp_XXXXX

# Validate on Jan 2026
python validate_jan2026_threshold_sweep.py
```

**Look for models with:**
- Different model families (LightGBM vs RandomForest)
- Different calibration methods (Platt scaling vs isotonic)
- Different training windows (6 months vs 12 months)
- Different labeling schemes (PT/SL/Hz combinations)
- Different sample weighting strategies

### Option 2: Return to Rule-Based System 🔄 **FALLBACK**

Your project notes mention `rule_based_v1/` which was developed after ML showed AUC ~0.51:

```
Current status:
- Baseline ML: AUC 0.516 (Jan 2026: loses money)
- Candidate ML: AUC 0.596 (Jan 2026: loses money)
- Rule-Based: Unknown (but designed for stability)
```

**Evaluate**: Test rule-based system on Jan 2026 to compare.

### Option 3: Ensemble Approach 🔬 **EXPERIMENTAL**

Combine top-3 models from exhaustive search:
- Use voting/averaging for predictions
- May reduce variance and improve stability
- Test on Jan 2026 before committing

### Option 4: Retrain with More Recent Data 📅 **MEDIUM-TERM**

Current models trained on older data (ending Nov 2025). Consider:
- Add Dec 2025 and Jan 2026 to training set
- Use only last 6 months (July 2025 - Jan 2026)
- Add regime detection features
- Re-run exhaustive search on updated training set

---

## 📁 **Files & Artifacts**

### Created
1. **Validation Script**: `ml_intraday_v3/experiments/validate_jan2026_threshold_sweep.py`
2. **Results CSV**: `ml_intraday_v3/diagnostics/threshold_sweep_jan2026_20260211_225212.csv`
3. **This Summary**: `ml_intraday_v3/JAN2026_VALIDATION_SUMMARY.md`

### Data Fetched
- **Jan 2026 MES bars**: 5,748 5-minute bars from Databento
- **Events**: 934 (after dropping vertical barriers)
- **Date Range**: 2026-01-01 to 2026-01-30

### To Re-Run
```bash
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep
python ml_intraday_v3/experiments/validate_jan2026_threshold_sweep.py
```

---

## 🎓 **Lessons Learned**

1. **CV Performance ≠ Live Performance**: A model with 0.68 CV AUC can still lose money in production
2. **Calibration is Fragile**: Isotonic calibration breaks down under distribution shift
3. **Threshold Flexibility is Limited**: Lowering thresholds doesn't fix fundamental lack of edge
4. **Win Rate Math is Unforgiving**: With asymmetric PT/SL, you need precise win rates or you bleed money
5. **Market Regime Matters**: Jan 2026 may be sufficiently different from training period to break the model

---

## ✅ **Bottom Line**

**DO NOT PROMOTE exhaustive_exp_00336 to production.**

The candidate model:
- ❌ Loses money at ALL confidence thresholds on Jan 2026
- ❌ Best result: -$297 on 208 signals at threshold 0.40
- ❌ Profit factor < 1.0 everywhere
- ❌ Negative Sharpe ratios
- ❌ Cannot generate high-confidence signals (p > 0.50)

**Recommendation**: Test alternative models from the top-20 exhaustive search results. If all fail Jan 2026 validation, consider returning to rule-based system or retraining with updated data.

---

**End of Report**
