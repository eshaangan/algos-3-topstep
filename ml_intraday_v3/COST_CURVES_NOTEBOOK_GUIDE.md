# Cost Curves in Pipeline Notebook - Quick Guide

## Overview

Your notebook `ml_intraday_v3_pipeline_runner_enhanced.ipynb` now includes **3 new sections** for cost curve analysis:

1. **Section 4.6.1**: Model Performance Analysis (after training)
2. **Section 4.7.1**: Backtest Performance Analysis (after backtesting)
3. **Section 5.5**: Model Comparison (before final summary)

## What You'll See

### Section 4.6.1: Model Performance Analysis - Cost Curves

**Location**: Right after model training (section 4.6)

**What it does:**
- Computes cost curve from model predictions
- Shows AUCC (Area Under Cost Curve) - lower is better
- Generates bootstrap 95% confidence intervals
- Maps risk/reward ratios to optimal thresholds
- Identifies optimal RR ratio for your strategy

**Visual outputs:**
- Left plot: Cost curve with confidence bands
- Right plot: Trading-specific curve with RR markers (1.0, 2.0, 3.0)

**Key metrics printed:**
```
AUCC: 0.0234 (lower is better)
Optimal Risk/Reward Ratio: 2.50
Optimal threshold: 0.623
```

**Required variables:**
- `y_test` - True labels from test set
- `y_pred_proba` - Model probability predictions

---

### Section 4.7.1: Backtest Performance - Cost Curve Analysis

**Location**: Right after backtest execution (section 4.7)

**What it does:**
- Analyzes actual trade outcomes
- Labels trades as profitable (1) or unprofitable (0)
- Computes cost curves on realized P&L
- Shows optimal RR ratio based on actual trading

**Visual outputs:**
- Left plot: Backtest cost curve
- Right plot: NC vs Risk/Reward ratio (find the minimum)

**Key metrics printed:**
```
Profitable trades: 245 (58.3%)
AUCC: 0.0156
Optimal RR: 2.00
Expected TPR: 95.1% (catch profitable trades)
Expected FPR: 8.2% (take unprofitable trades)
```

**Required variables:**
- `trades_df` - DataFrame with backtest results
  - Must have `pnl` column
  - Should have probability column (e.g., `prob`, `signal_prob`)

---

### Section 5.5: Model Comparison - Cost Curves

**Location**: Before final summary report (section 6)

**What it does:**
- Compares multiple models side-by-side
- Ranks models by AUCC (lower is better)
- Shows which model dominates across cost ratios
- Visualizes performance differences

**Visual outputs:**
- Left plot: All model cost curves overlaid
- Right plot: Difference between models (if 2 models)

**Key metrics printed:**
```
AUCC Comparison:
⭐ 1. Primary Model    : AUCC = 0.0156
   2. Baseline (Prior): AUCC = 0.2500

Primary Model outperforms Baseline for 100.0% of cost ratios
```

**Required variables:**
- `y_test` - True labels
- `y_pred_proba` - Primary model predictions
- (Optional) `y_pred_proba_meta` - Meta-labeling predictions
- (Optional) `y_pred_proba_baseline` - Baseline predictions

---

## How to Use

### Step 1: Run the Pipeline Normally

Execute all cells in order through section 4.6 (Train Models):
```
0) Parameters & Imports
1) Helper Functions
...
4.6) Train Models  ← Run this first
```

### Step 2: Run Section 4.6.1 - Model Performance

After training completes, run the new section 4.6.1.

**Expected runtime**: 10-30 seconds (due to bootstrap)

**You'll see:**
- AUCC metric
- Cost curve plots with confidence intervals
- Optimal RR ratio for your model
- Performance table by RR ratio

### Step 3: Run Backtest (Section 4.7)

Continue with the backtest:
```
4.7) Run Backtest  ← Run this
```

### Step 4: Run Section 4.7.1 - Backtest Analysis

After backtest completes, run section 4.7.1.

**Expected runtime**: 5-10 seconds

**You'll see:**
- Trade statistics (win rate, P&L)
- Cost curve on actual trade outcomes
- Optimal RR ratio based on realized performance
- Performance by RR ratio table

### Step 5: (Optional) Compare Models - Section 5.5

If you have multiple model variants, run section 5.5.

**To compare models, define:**
```python
# In section 4.6, save different model predictions:
y_pred_proba_v1 = model_v1.predict_proba(X_test)[:, 1]
y_pred_proba_v2 = model_v2.predict_proba(X_test)[:, 1]
```

Then section 5.5 will automatically compare them.

---

## Interpreting Results

### AUCC (Area Under Cost Curve)

- **Lower is better** (opposite of ROC AUC!)
- AUCC ≈ 0.00: Perfect classifier
- AUCC ≈ 0.25: Good classifier
- AUCC ≈ 0.50: Random classifier
- AUCC > 0.50: Poor classifier (worse than random)

### Optimal Risk/Reward Ratio

The RR ratio that minimizes normalized cost (NC).

**Example interpretations:**
- Optimal RR = 1.0: Use equal PT/SL distances (1:1)
- Optimal RR = 2.0: Use 2:1 profit target to stop loss
- Optimal RR = 3.0: Use 3:1 profit target to stop loss

### TPR vs FPR at Optimal

- **TPR** (True Positive Rate): % of profitable trades caught
  - Higher is better (want to catch winners)
- **FPR** (False Positive Rate): % of unprofitable trades taken
  - Lower is better (want to avoid losers)

**Example:**
```
Optimal: TPR=0.92, FPR=0.15
```
Means: Catch 92% of profitable trades, but also take 15% of unprofitable trades.

---

## Troubleshooting

### Error: "Model predictions not found"

**Solution**: Make sure you've run section 4.6 (Train Models) first.

Check that these variables exist:
```python
print("y_test:", type(y_test) if 'y_test' in dir() else "NOT FOUND")
print("y_pred_proba:", type(y_pred_proba) if 'y_pred_proba' in dir() else "NOT FOUND")
```

### Error: "Probability column not found in trades_df"

**Solution**: Your backtest results need to include prediction probabilities.

Modify your backtest code to store probabilities:
```python
# When creating trades_df, add:
trades_df['prob'] = signal_probabilities  # Your model predictions
```

### Slow Performance (Bootstrap)

**Solution**: Reduce bootstrap samples in section 4.6.1:

Change:
```python
n_bootstrap=500  # Default
```
To:
```python
n_bootstrap=100  # Faster (less accurate CI)
```

Or skip bootstrap entirely:
```python
# Comment out bootstrap section and just use:
curve_df = compute_cost_curve(y_test, y_pred_proba)
plot_cost_curve(curve_df)
```

### Want More Models to Compare

**Solution**: In section 4.6, train multiple variants:

```python
# Train baseline
baseline = DummyClassifier(strategy='prior')
baseline.fit(X_train, y_train)
y_pred_proba_baseline = baseline.predict_proba(X_test)[:, 1]

# Train meta-labeling
# (your meta-labeling code here)
# y_pred_proba_meta = ...

# Section 5.5 will automatically detect and compare all variants
```

---

## Customization

### Change Risk/Reward Ratios

Default is `[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]`.

To customize, edit in sections 4.6.1 or 4.7.1:
```python
risk_reward_ratios=[1.0, 2.0, 3.0, 4.0, 5.0]  # Your custom RR ratios
```

### Change Confidence Level

Default is 95% CI. To change:
```python
confidence_level=0.90  # 90% CI (narrower bands)
# or
confidence_level=0.99  # 99% CI (wider bands)
```

### Save Plots

Add after any `plt.show()`:
```python
plt.savefig('cost_curve_analysis.png', dpi=150, bbox_inches='tight')
```

---

## Next Steps

1. **Run the sections** - Execute 4.6.1, 4.7.1, and 5.5 after their prerequisites
2. **Interpret results** - Use AUCC and optimal RR to guide strategy tuning
3. **Compare variants** - Train multiple models and compare with section 5.5
4. **Integrate with risk** - Use optimal RR ratio in your position sizing logic

## References

- Main implementation: `ml_intraday_v3/analysis/cost_curves.py`
- Tests: `ml_intraday_v3/tests/test_cost_curves.py`
- Demo script: `ml_intraday_v3/analysis/cost_curves_demo.py`
- Full guide: `ml_intraday_v3/analysis/README_COST_CURVES.md`

Paper: Drummond & Holte (2006) "Cost curves: An improved method for visualizing classifier performance"
