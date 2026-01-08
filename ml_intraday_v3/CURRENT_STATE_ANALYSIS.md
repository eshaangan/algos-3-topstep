# ML Pipeline V3 - Current State Analysis (January 7, 2026)

## Executive Summary

**Status: NOT READY FOR LIVE TRADING**

The current model fails validation with:
- Total backtest loss: -$1,389 (1m), -$3,572 (5m)
- 3 trailing drawdown breaches across all folds
- Inconsistent performance (1 profitable fold out of 6)

---

## 1. Data Overview

| Metric | 1m Bars | 5m Bars |
|--------|---------|---------|
| Total Bars | 1,037,010 | 209,849 |
| Date Range | 2010-06-07 to 2025-12-19 | Same |
| Events | 670,161 | 72,040 |
| Features | 34 | 35 |
| **Usable for Training** | **25.5%** | **100%** |

### Critical Issue: 1m Feature Usability
Only 25.5% of 1m bars are usable for training due to NaN features. This is a **major problem** that needs investigation.

---

## 2. Label Distribution (Triple Barrier)

### 1m Events
| Label | Count | Percentage |
|-------|-------|------------|
| -1 (Stop) | 81,037 | 12.1% |
| 0 (Vertical) | 517,633 | 77.2% |
| +1 (Target) | 71,491 | 10.7% |

### Time Period Analysis - Key Insight!

| Year | Events | Target Rate | Stop Rate |
|------|--------|-------------|-----------|
| 2010-2017 | ~400k | 4-7% | 8-20% |
| 2018-2021 | ~160k | 10-18% | 6-17% |
| **2022-2025** | **~98k** | **22-31%** | **4-9%** |

**Critical Finding**: The market regime changed significantly around 2018. Training on all data (2010-2025) dilutes the model with patterns from a different market regime where success rates were much lower.

---

## 3. HMM Regime Detection - FAILED

| Metric | Value | Target |
|--------|-------|--------|
| Transition Rate | 35.0% | <5% |
| Avg Regime Duration | 2.9 bars | >20 bars |
| 1-bar Regimes | 64.7% | <10% |
| State 0 Mean Return | +0.0056% | - |
| State 1 Mean Return | -0.0048% | - |

**Conclusion**: HMM regime detection does not work for intraday (1m/5m) returns. The returns are too noisy and the regime separation is negligible.

---

## 4. Model Performance

### Training Metrics (5m, 6-fold CV)
| Metric | Value |
|--------|-------|
| Accuracy | 84.9% |
| Balanced Accuracy | 40.5% |
| F1 Macro | 40.3% |
| ROC AUC (target vs rest) | 78.7% |

The balanced accuracy of 40.5% is close to random (33% for 3 classes), indicating poor predictive power for minority classes.

### Backtest Results

#### 1m Bars
| Fold | PnL | Win Rate | Profit Factor | Trades | DD Breach |
|------|-----|----------|---------------|--------|-----------|
| 0 | -$554 | 37.1% | 0.79 | 170 | No |
| 1 | -$76 | 32.0% | 0.82 | 25 | No |
| 2 | -$138 | 39.4% | 0.86 | 33 | No |
| 3 | -$2,020 | 40.2% | 0.75 | 978 | No |
| 4 | -$2,116 | 46.1% | 0.85 | 1024 | **YES** |
| 5 | +$3,515 | 66.7% | 4.24 | 54 | No |
| **TOTAL** | **-$1,389** | - | - | 2,284 | 1 |

#### 5m Bars
| Fold | PnL | Win Rate | Profit Factor | Trades | DD Breach |
|------|-----|----------|---------------|--------|-----------|
| 0-2 | Various | - | - | 16 | No |
| 3 | -$1,116 | 46.8% | 0.73 | 186 | No |
| 4 | -$2,578 | 38.8% | 0.35 | 49 | **YES** |
| 5 | +$342 | 51.8% | 1.03 | 166 | **YES** |
| **TOTAL** | **-$3,572** | - | - | 417 | 2 |

---

## 5. Root Cause Analysis

### Issue 1: Feature NaN Problem (1m bars)
- Only 25.5% of 1m bars have complete features
- Likely caused by rolling window calculations not having enough warmup data
- **Impact**: Massive data loss, model trained on incomplete picture

### Issue 2: Training on Wrong Time Period
- 2010-2017 data has very different characteristics (4-7% target rate)
- 2022+ data has 22-31% target rate
- Training on all data creates a confused model

### Issue 3: HMM Regime Weighting is Useless
- Noisy HMM with 2.9 bar average duration
- w_regime essentially random (mean=0.516, std=0.446)
- Adds noise rather than signal

### Issue 4: Inconsistent Fold Performance
- Fold 5 shows +$3,515 while others lose money
- Suggests either:
  - Data leakage in fold 5
  - Fold 5 covers a favorable market period
  - Random variance

---

## 6. Improvement Plans

### Plan A: Date-Filtered Training (RECOMMENDED)
**Description**: Train only on 2022+ data where market characteristics match current conditions.

**Steps**:
1. Add date filter to configs: `min_date: "2022-01-01"`
2. Re-run pipeline with filtered data
3. Expected: ~98k events with higher quality patterns

**Pros**: Simple, addresses root cause
**Cons**: Less training data

### Plan B: Fix 1m Feature NaN Issue
**Description**: Investigate and fix why 74.5% of 1m features are unusable.

**Steps**:
1. Identify which features have NaNs
2. Increase rolling window warmup periods
3. Use forward-fill or alternative imputation
4. Re-run feature generation

**Pros**: Recovers massive amount of data
**Cons**: May require significant code changes

### Plan C: Remove HMM, Simplify Weights
**Description**: Since HMM doesn't work for intraday, remove it entirely.

**Steps**:
1. Set `hmm_regime.enabled: false` in config
2. Use only uniqueness weights
3. Re-run weights and training

**Pros**: Removes noise source
**Cons**: Loses potential regime signal

### Plan D: 5m Bars with Walk-Forward
**Description**: Use 5m bars (100% usable features) with walk-forward validation.

**Steps**:
1. Focus on 5m bars only
2. Use expanding walk-forward instead of K-fold
3. Test on most recent period (2024-2025)

**Pros**: Clean data, proper time-series validation
**Cons**: Fewer events (72k vs 670k)

### Plan E: Ensemble Multiple Approaches
**Description**: Combine date-filtered + 5m bars + simplified weights.

**Steps**:
1. Implement Plan A + C + D together
2. Train ensemble of models
3. Use voting or averaging for final signal

**Pros**: Robust, multiple validation angles
**Cons**: More complex

---

## 7. Priority Recommendations

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| 1 | **Date-filter to 2022+** | High - matches current market |
| 2 | **Disable HMM regime** | Medium - removes noise |
| 3 | **Fix 1m feature NaNs** | High - recovers 75% data |
| 4 | **Use 5m bars** | Medium - clean baseline |
| 5 | **Walk-forward validation** | High - proper time-series CV |

---

## 8. Next Steps

1. **Immediate**: Update notebooks to point to `runs/v3_full`
2. **Short-term**: Run Plan A (date-filtered training)
3. **Medium-term**: Investigate and fix feature NaN issue
4. **Long-term**: Implement walk-forward validation

---

## 9. Files Updated This Session

| File | Action |
|------|--------|
| `runs/v3_full/` | Created - full pipeline run |
| `features/hmm_regime_gpu.py` | Implemented GPU HMM |
| `weights/hmm_weights.py` | Implemented regime weights |
| `cli.py` | Integrated HMM with build-weights |

---

## 10. Artifacts Location

```
runs/v3_full/
├── run_manifest.json
├── bar_size=1m/
│   ├── bars.parquet (17.4MB, 1,037,010 rows)
│   ├── events.parquet (36.5MB, 670,161 events)
│   ├── features.parquet (132.6MB, 34 features)
│   ├── hmm_regimes_gpu.parquet
│   ├── weights.parquet
│   ├── cv_splits.json (15 CPCV paths)
│   ├── training/purged_kfold/ (6 folds + models)
│   └── backtests/purged_kfold/ (6 fold results)
└── bar_size=5m/
    └── (same structure)
```

---

*Generated: January 7, 2026*
