# PBO Quick Start Guide

## For Pipeline Notebook Users

This guide helps you use the new PBO (Probability of Backtest Overfitting) sections in the pipeline notebook.

---

## What is PBO?

PBO measures the probability that your "best" model configuration was selected due to overfitting rather than true predictive skill.

**Quick Interpretation:**
- **PBO < 0.3**: Configuration is robust, safe to proceed
- **PBO 0.3-0.5**: Moderate overfitting risk, monitor carefully
- **PBO > 0.5**: High overfitting risk, do NOT deploy

---

## New Notebook Sections

### Section 4.6.4: Enhanced PBO Analysis
- **Location:** After "Rare Events Calibration Analysis", before "Run Backtest"
- **Purpose:** Primary PBO computation and reporting
- **Outputs:**
  - `pbo_analysis.png` - Visualizations
  - `pbo_report.md` - Detailed report

### Section 5.3: PBO Validation Across Multiple Runs
- **Location:** After "Equity Curve Uncertainty", before "Final Summary"
- **Purpose:** Multi-run validation and stopping criteria
- **Outputs:**
  - `pbo_vs_trials.png` - Trend plot
  - `pbo_summary.csv` - Summary table

---

## How to Use

### Option 1: Quick Run (Trials Already Tracked)

If your run directory already has `trials/trials.json`:

1. Open the notebook
2. Run Section 4.6.4 cell
3. Review PBO results and interpretation
4. Run Section 5.3 cell (optional)
5. Check stopping criteria recommendations

**Done!** The sections will automatically load trials and compute PBO.

---

### Option 2: First-Time Setup (Track Trials)

If you haven't tracked trials yet, follow these steps:

#### Step 1: Track Trials During Training

Add this to your training code (before Section 4.6):

```python
from ml_intraday_v3.experiments.trial_tracker import TrialTracker

# Initialize tracker
tracker = TrialTracker(RUN_DIR)

# During hyperparameter search:
for config in all_configurations:
    # Log trial
    trial_id = tracker.log_trial(
        config=config,
        model_type='logreg',  # or 'lgbm', 'relogit', etc.
        hyperparameters=config['model']['params'],
        features=config.get('features', {}).get('use_columns'),
    )

    # Train model and run CPCV
    for path_id in range(n_cpcv_paths):
        # Get IS/OOS metrics for this path
        is_metric = ...  # In-sample ROC-AUC on other paths
        oos_metric = ... # Out-of-sample ROC-AUC on this path

        # Update trial metrics
        tracker.update_path_metrics(
            trial_id=trial_id,
            path_id=f'path_{path_id}',
            is_metric=is_metric,
            oos_metric=oos_metric
        )

# Save trials to disk
tracker.save()
```

#### Step 2: Run PBO Sections

After trials are tracked:
1. Run Section 4.6.4 to compute PBO
2. Review results and interpretation
3. Run Section 5.3 for trend analysis

---

## Reading the Results

### Section 4.6.4 Output

**Console Output:**
```
==================================================================
PBO Results:
==================================================================
PBO = 0.35 (35.0%)
95% CI: [0.25, 0.45]

Lambda Statistics:
  - Mean: 0.48
  - Median: 0.50
  - Std: 0.15

Trials: 20
CPCV Paths: 5

Risk Level: 🟠 MODERATE RISK
Interpretation: Some overfitting risk - Monitor performance carefully
==================================================================
```

**Visualizations:**
- Left plot: Lambda distribution (should peak above 0.5 for good configs)
- Right plot: PBO with confidence intervals (lower is better)

**Report (`pbo_report.md`):**
- Detailed interpretation
- Recommendations based on risk level
- Methodology explanation

---

### Section 5.3 Output

**PBO Trend Plot:**
- X-axis: Number of trials tested
- Y-axis: PBO value
- Shows if PBO increases with more trials (selection bias)

**Stopping Criteria:**
```
🔴 STOP: PBO > 0.5 (high overfitting risk)
   Actions:
   1. Reduce hyperparameter search space
   2. Increase training sample size
   3. Use simpler models
```

---

## Common Scenarios

### Scenario 1: No Trials Found

**Message:**
```
⚠️  No trials found at: RUN_DIR/trials/trials.json

To use PBO analysis, track trials during training...
```

**Action:** Follow "Option 2: First-Time Setup" above to track trials.

---

### Scenario 2: Low PBO (Good!)

**Result:**
```
PBO = 0.15 (15.0%)
Risk Level: 🟢 LOW RISK
```

**Action:**
- Configuration appears robust
- Safe to proceed to backtesting
- Continue to walk-forward validation

---

### Scenario 3: High PBO (Warning!)

**Result:**
```
PBO = 0.65 (65.0%)
Risk Level: 🔴 HIGH RISK
```

**Action:**
1. **STOP** - Do not deploy this configuration
2. **Reduce search space** - Test fewer hyperparameters
3. **Increase sample size** - Get more training data if possible
4. **Simplify model** - Use fewer features or simpler model
5. **Consider ensemble** - Combine multiple configs instead of selecting one

---

### Scenario 4: Increasing PBO Trend

**Result:**
```
Starting PBO (n=2): 0.20
Final PBO (n=20): 0.55
Change: +0.35

⚠️  Warning: PBO increased with more trials (selection bias)
   Consider stopping hyperparameter search.
```

**Action:**
- Selection bias detected
- More trials = more overfitting
- Stop hyperparameter search
- Use simpler selection criteria

---

## Best Practices

### 1. Always Track ALL Trials
❌ Don't: Only save the "winning" configuration
✅ Do: Log every single trial/configuration tested

**Why:** PBO requires the full distribution to detect selection bias.

### 2. Use CPCV for Validation
❌ Don't: Use simple train/test split
✅ Do: Use CPCV with multiple paths (already in pipeline)

**Why:** PBO needs multiple validation paths to compute lambda values.

### 3. Check PBO Before Backtesting
❌ Don't: Run expensive backtests on overfitted configs
✅ Do: Check PBO first, only backtest robust configs

**Why:** Saves time and prevents false confidence from overfitted results.

### 4. Monitor PBO During Search
❌ Don't: Run 100s of trials blindly
✅ Do: Check PBO periodically, stop if PBO > 0.5

**Why:** Prevents excessive selection bias.

### 5. Document PBO in Reports
❌ Don't: Ignore PBO in final decisions
✅ Do: Include PBO in deployment criteria

**Why:** PBO is a critical risk metric for overfitting.

---

## Troubleshooting

### "Cannot compute PBO: insufficient_trials"
**Cause:** Less than 2 trials tracked
**Fix:** Track more trials during hyperparameter search

### "Cannot compute PBO: no_is_oos_columns"
**Cause:** Trials missing IS/OOS metrics
**Fix:** Ensure `update_path_metrics()` is called for all paths

### "PBO is always None"
**Cause:** Check error message for specific reason
**Fix:** Review trial structure with `tracker.get_summary_stats()`

---

## Additional Resources

1. **Full Documentation:**
   - `/ml_intraday_v3/experiments/PBO_ENHANCED_README.md`

2. **Implementation Details:**
   - `/ml_intraday_v3/experiments/trial_tracker.py`
   - `/ml_intraday_v3/experiments/diagnostics.py`

3. **Integration Summary:**
   - `/ml_intraday_v3/training/PBO_INTEGRATION_SUMMARY.md`

4. **Reference:**
   - López de Prado, M. (2018). *Advances in Financial Machine Learning.* Chapter 11.

---

## Quick Checklist

Before deploying a model configuration, verify:

- [ ] Tracked ALL trials during hyperparameter search
- [ ] PBO < 0.3 (or at least < 0.5 with strong justification)
- [ ] Lambda distribution shows most paths have λ > 0.5
- [ ] PBO did not increase significantly with more trials
- [ ] Reviewed PBO report recommendations
- [ ] Validated on walk-forward data
- [ ] Documented PBO value in deployment decision

---

**Remember:** PBO is one input to the deployment decision, not the only input. Always validate on walk-forward data and monitor live performance.

---

**End of Quick Start Guide**
