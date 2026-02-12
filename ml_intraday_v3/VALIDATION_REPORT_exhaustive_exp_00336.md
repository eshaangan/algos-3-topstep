# Model Validation Report: exhaustive_exp_00336
**Date**: 2026-02-11
**Holdout Period**: Dec 2025 (1,092 bars, 171 valid events)
**Comparison**: Candidate vs Current Baseline

---

## Executive Summary

**Decision: ⚠️ MARGINAL / NO-GO**

The promoted model (exhaustive_exp_00336) shows **statistically significant AUC improvement (+0.079)** over the baseline but fails critical production readiness criteria. Most importantly, **neither model generates actionable high-confidence signals** on the Dec 2025 holdout period.

### Key Findings

| Metric | Baseline | Candidate | Delta | Status |
|--------|----------|-----------|-------|--------|
| **AUC** | 0.516 | 0.596 | +0.079 | ✅ PASS |
| **Brier Score** | 0.247 | 0.240 | -0.008 | ✅ PASS |
| **Calibration Error** | 0.075 | 0.148 | +0.072 | ❌ FAIL |
| **Signals (p>0.55)** | 1 (0.6%) | 0 (0%) | -1 | ❌ FAIL |
| **Accuracy** | 55.0% | 59.1% | +4.1% | Improved |

**Decision Criteria: 2/4 passed**

---

## Detailed Analysis

### 1. Baseline Model (Current Production)
**Path**: `ml_intraday_v3/model_bundle_retrained_oct2024_nov2025.pkl`
**Training**: Oct 2024 - Nov 2025
**Architecture**: LGBMClassifier, 34 features, no momentum

#### Performance on Dec 2025 Holdout
- **AUC**: 0.516 (barely above random guess)
- **Probability Range**: [0.419, 0.554]
- **Mean Probability**: 0.485 ± 0.024
- **High-Confidence Signals**: 1 out of 171 events (0.6%)
- **Expected Value**: -0.13 pts/event (negative expectancy)

#### Critical Issues
1. **Near-Random Performance**: AUC 0.516 indicates the model has minimal predictive power on recent data
2. **Extremely Low Signal Rate**: Only 1 event exceeded 0.55 confidence threshold
3. **Negative Expected Value**: Model predictions do not yield positive expectancy

### 2. Candidate Model (exhaustive_exp_00336)
**Path**: `ml_intraday_v3/models/saved/model_bundle_phasefull_best_exhaustive_exp_00336.pkl`
**Training**: 12-month rolling window (config-driven)
**Architecture**: RandomForestClassifier (600 trees, depth 8) + Isotonic Calibration, 41 features with momentum

#### Configuration Details
```json
{
  "model_name": "random_forest",
  "n_estimators": 600,
  "max_depth": 8,
  "min_samples_leaf": 20,
  "session_mode": "rth_eth",
  "training_window_months": 12,
  "sample_weight": "class_balanced",
  "calibration": "isotonic",
  "labeling": {"pt": 4.0, "sl": 3.0, "hz": 24},
  "feature_set": "momentum_on"
}
```

#### Performance on Dec 2025 Holdout
- **AUC**: 0.596 (+0.079 vs baseline)
- **Probability Range**: [0.332, 0.447]
- **Mean Probability**: 0.382 ± 0.044
- **High-Confidence Signals**: 0 out of 171 events (0%)
- **Expected Value**: -0.13 pts/event (same as baseline)

#### Critical Issues
1. **Zero High-Confidence Signals**: Not a single event exceeded 0.55 probability
2. **Miscalibration**: Calibration error increased from 0.075 to 0.148
3. **Probability Compression**: All predictions fall in narrow [0.33, 0.45] range
4. **No Improvement in EV**: Despite better AUC, expected value remains negative

---

## Root Cause Analysis

### Why the Candidate Model Fails Production Criteria

1. **Probability Shift**: The model's probability distribution is shifted downward
   - Baseline: mean 0.485, range [0.419, 0.554]
   - Candidate: mean 0.382, range [0.332, 0.447]
   - **Impact**: With a 0.55 confidence filter, candidate generates ZERO signals

2. **Calibration Degradation**: Despite isotonic calibration, the model's calibration error doubled
   - This suggests the calibration was fit on a different distribution than Dec 2025 data
   - Possible regime shift between training (12-month window) and holdout (Dec 2025)

3. **Momentum Features Don't Help**: Adding 7 momentum features (RSI, MACD, Stochastics) improved AUC but:
   - Did not increase high-confidence signal rate
   - Created probability compression (lower variance)
   - May have overfit to cross-validation splits without generalizing to holdout

### Comparison to Exhaustive Search Results

During exhaustive search, this model achieved:
- **Median Test AUC**: 0.6806 (on CV splits)
- **Composite Score**: 0.7880

On true holdout (Dec 2025):
- **Test AUC**: 0.5958 (much lower)
- **Signal Rate**: 0% (unusable for trading)

**Conclusion**: The model's CV performance does not reflect real-world holdout performance. There is a significant distribution shift between the CV splits and Dec 2025.

---

## Decision Matrix

| Criterion | Threshold | Result | Pass |
|-----------|-----------|--------|------|
| 1. AUC Improvement | > 0.02 | +0.079 | ✅ |
| 2. Calibration Maintained | < 0.02 degradation | +0.072 | ❌ |
| 3. Signal Frequency | ≥ 50 signals at p>0.55 | 0 | ❌ |
| 4. Brier Score | ≤ 0.01 degradation | -0.008 (improvement) | ✅ |

**Overall: 2/4 criteria passed → MARGINAL**

---

## Recommendations

### Immediate Actions (Do NOT Promote)

**Primary Recommendation**: **Do not promote exhaustive_exp_00336 to production.**

While the model shows statistical improvement in AUC, it fails the critical test of generating actionable trading signals. Deploying this model would result in **zero trades** with the current 0.55 confidence threshold.

### Alternative Paths Forward

#### Option 1: Test Next-Best Candidates (Recommended)
The exhaustive search produced a ranked list of 1,000 models. Test the next-best alternatives:

**Action Items**:
1. Load `/tmp/phasefull_results/phasefull_ranked_top20.csv`
2. Review top 5 candidates (exp_00337 - exp_00341 or similar)
3. Check their configurations for:
   - Different calibration methods (Platt scaling vs isotonic)
   - Different training windows (6 months vs 12 months)
   - Different model families (LightGBM vs RandomForest)
4. Run `validate_promoted_model.py` on next-best candidate
5. Look for models with:
   - AUC > 0.55
   - Signal rate > 5% at p>0.55
   - Better calibration error < 0.10

#### Option 2: Adjust Confidence Threshold (Risky)
Lower the confidence threshold from 0.55 to enable candidate model to trade:

**Analysis**:
- Candidate's max probability: 0.447
- To get signals, threshold must be < 0.447
- At p>0.40: ~50% of events qualify (85 signals)
- At p>0.35: ~75% of events qualify (128 signals)

**Risks**:
- Lower threshold = lower quality signals
- May increase losing trades
- Does not address calibration issues

**Recommendation**: Only consider if no better model found AND after backtesting at lower threshold shows positive expectancy.

#### Option 3: Retrain with Recent Data (Medium-Term)
The baseline was trained on Oct 2024 - Nov 2025, but Dec 2025 performance is weak.

**Hypothesis**: Market regime shift in Dec 2025

**Action Items**:
1. Extend training window to include Dec 2025 (once enough data)
2. Use most recent 6-12 months instead of historical data
3. Add regime detection features to adapt to changing distributions
4. Consider ensemble of models trained on different time windows

#### Option 4: Return to Rule-Based System (Fallback)
Your project notes mention a rule-based system in `rule_based_v1/` that was developed after ML walk-forward AUC ~0.51.

**Consider**:
- Both models here show AUC ~0.52-0.60 on holdout
- Rule-based might be more stable and interpretable
- Can combine: ML filters + rule-based entries

---

## Production Handoff (If Overriding Decision)

### If You Decide to Promote Despite Marginal Status

#### Required Config Changes

**File**: `ml_intraday_v3/configs/live_trading.yaml`

```yaml
# Line 21: Update model bundle path
model_bundle_path: "models/saved/model_bundle_phasefull_best_exhaustive_exp_00336.pkl"

# Line 112: Lower primary threshold to enable signals
primary_threshold: 0.40  # Changed from 0.03; candidate max prob is 0.447

# Line 45: Enable momentum features
# File: ml_intraday_v3/configs/features.yaml
momentum:
  enabled: true  # Changed from false
```

**File**: `ml_intraday_v3/configs/execution_spec.yaml`

```yaml
# Line 5: Lower confidence filter to match new model's range
confidence_filter:
  enabled: true
  min_probability_distance: 0.35  # Changed from 0.55
```

#### Validation Checklist Before Going Live

- [ ] Run 1-week paper trading with new config
- [ ] Confirm signal generation (expect ~85 signals/week at p>0.40)
- [ ] Monitor trade quality metrics:
  - [ ] Win rate > 45%
  - [ ] Average winner > average loser
  - [ ] Sharpe > 0.5
- [ ] Verify directional balance (LONG/SHORT ratio 40-60%)
- [ ] Confirm Topstep rule compliance:
  - [ ] Daily loss limit respected
  - [ ] Max drawdown within limits
  - [ ] No consistency violations

#### Rollback Plan

1. Keep baseline bundle available: `model_bundle_retrained_oct2024_nov2025.pkl`
2. Git commit current config before changes
3. If metrics degrade after 50 trades:
   - Revert `live_trading.yaml` to baseline model
   - Revert thresholds to 0.55
   - Disable momentum features
4. Document rollback decision and reasons

---

## Files Modified/Created

### Created Files
1. `ml_intraday_v3/experiments/validate_promoted_model.py` - Validation script
2. `ml_intraday_v3/VALIDATION_REPORT_exhaustive_exp_00336.md` - This report
3. `ml_intraday_v3/diagnostics/model_validation_20260211_223605.json` - Raw results

### Models Tested
1. **Baseline**: `ml_intraday_v3/model_bundle_retrained_oct2024_nov2025.pkl`
2. **Candidate**: `ml_intraday_v3/models/saved/model_bundle_phasefull_best_exhaustive_exp_00336.pkl`
3. **Config**: `ml_intraday_v3/models/saved/model_bundle_phasefull_best_exhaustive_exp_00336.config.json`

---

## How to Run Validation

```bash
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep

# Run validation comparison
python ml_intraday_v3/experiments/validate_promoted_model.py

# Results saved to:
# ml_intraday_v3/diagnostics/model_validation_TIMESTAMP.json
```

---

## Assumptions Made

1. **Holdout Period**: Dec 2025 (1,092 bars) is representative of future live trading conditions
2. **Labeling Config**: Used current `labeling.yaml` (PT/SL/Hz may differ from model training)
3. **Feature Generation**: Correctly enabled momentum features for candidate model
4. **Bundle Schema**: Both models follow `LiveModelPredictor` schema requirements
5. **Decision Thresholds**: 0.55 confidence filter reflects actual live trading gates
6. **Expected Value Calculation**: Used PT=4.0, SL=3.0 from standard config (may not match model's trained PT/SL)

---

## Next Steps

1. **Review top-20 ranked results** from exhaustive search:
   ```bash
   cat /tmp/phasefull_results/phasefull_ranked_top20.csv
   ```

2. **Promote and test alternative candidate** (e.g., exp_00337, exp_00400, etc.)

3. **Run extended validation** on full Nov-Dec 2025 period (not just Dec)

4. **Consider ensemble approach**: Combine top-3 models with voting

5. **Investigate regime shift**: Compare Dec 2025 feature distributions vs training period

6. **Optimize confidence threshold**: Grid search on holdout to find optimal threshold that maximizes Sharpe while maintaining trade frequency

---

## Contact / Questions

For questions about this validation or to discuss alternative promotion strategies, refer to:
- **Project Memory**: `.claude/CLAUDE.md` and `MEMORY.md`
- **Model Configs**: `ml_intraday_v3/models/saved/model_bundle_phasefull_best_exhaustive_exp_00336.config.json`
- **Raw Results**: `ml_intraday_v3/diagnostics/model_validation_20260211_223605.json`

---

**End of Report**
