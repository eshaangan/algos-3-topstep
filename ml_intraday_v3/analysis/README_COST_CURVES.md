# Cost Curves for Classifier Performance Visualization

Implementation of cost curves as described in **Drummond & Holte (2006)**: *"Cost curves: An improved method for visualizing classifier performance"*.

## Overview

Cost curves visualize classifier performance across **all class distributions and misclassification cost ratios** by plotting the Normalized Expected Cost (NC) against Probability Cost (PC).

### Key Advantages over ROC Curves

1. **Direct cost interpretation**: NC directly represents the normalized expected cost
2. **All operating points**: Shows performance across all cost ratios simultaneously
3. **Statistical confidence**: Easy to add bootstrap confidence intervals
4. **Dominance relationships**: Clear visualization of which model dominates
5. **Trading context**: Natural mapping to risk/reward ratios

## Components

### Core Module: `cost_curves.py`

Located at: `ml_intraday_v3/analysis/cost_curves.py`

**Main Functions:**
- `compute_cost_curve()` - Compute cost curve from predictions
- `bootstrap_cost_curve()` - Add confidence intervals via bootstrap
- `plot_cost_curve()` - Visualize cost curves
- `plot_cost_difference()` - Show difference between two models
- `compute_trading_cost_curve()` - Trading-specific cost ratios
- `compare_models_cost_curves()` - Compare multiple models
- `compute_area_under_cost_curve()` - Summary metric (AUCC)

### Integration: `diagnostics.py`

Located at: `ml_intraday_v3/experiments/diagnostics.py`

**Diagnostic Functions:**
- `compute_model_cost_curves()` - Complete diagnostics for one model
- `compute_trading_cost_diagnostics()` - Trading-specific analysis
- `compare_models_cost_diagnostics()` - Multi-model comparison with AUCC ranking

### Tests

Located at: `ml_intraday_v3/tests/test_cost_curves.py`

**Test Coverage:**
- Perfect/random/inverted classifiers
- Mathematical properties (PC/NC bounds, monotonicity)
- Bootstrap confidence intervals
- Trading-specific cost ratios
- Relationship to ROC curves
- Input validation

**Run tests:**
```bash
python -m pytest ml_intraday_v3/tests/test_cost_curves.py -v
```

## Quick Start

### 1. Basic Cost Curve

```python
from ml_intraday_v3.analysis.cost_curves import compute_cost_curve, plot_cost_curve
import matplotlib.pyplot as plt

# Your predictions
y_true = [0, 0, 1, 1, 1]
y_prob = [0.2, 0.3, 0.8, 0.7, 0.9]

# Compute cost curve
curve_df = compute_cost_curve(y_true, y_prob)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
plot_cost_curve(curve_df, ax=ax, label="My Model")
plt.show()
```

### 2. With Bootstrap Confidence Intervals

```python
from ml_intraday_v3.analysis.cost_curves import bootstrap_cost_curve, plot_cost_curve

mean_curve, lower, upper = bootstrap_cost_curve(
    y_true, y_prob,
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=42
)

fig, ax = plt.subplots()
plot_cost_curve(
    mean_curve,
    ax=ax,
    show_confidence=True,
    confidence_bounds=(lower, upper)
)
```

### 3. Compare Multiple Models

```python
from ml_intraday_v3.analysis.cost_curves import compare_models_cost_curves

models = {
    'Model A': (y_true, y_prob_a),
    'Model B': (y_true, y_prob_b),
    'Baseline': (y_true, y_prob_baseline)
}

fig, curves = compare_models_cost_curves(models, show_confidence=False)
plt.show()
```

### 4. Trading-Specific Analysis

```python
from ml_intraday_v3.analysis.cost_curves import compute_trading_cost_curve

# Map risk/reward ratios to cost curves
curve_df = compute_trading_cost_curve(
    y_true, y_prob,
    risk_reward_ratios=[1.0, 1.5, 2.0, 2.5, 3.0]
)

# Find optimal RR ratio
optimal_idx = curve_df['nc'].idxmin()
optimal_rr = curve_df.loc[optimal_idx, 'risk_reward_ratio']
print(f"Optimal Risk/Reward: {optimal_rr}")
```

### 5. Integration with Diagnostics

```python
from ml_intraday_v3.experiments.diagnostics import (
    compute_model_cost_curves,
    compute_trading_cost_diagnostics
)

# Full model diagnostics
result = compute_model_cost_curves(
    y_true, y_prob,
    model_name="Primary Model",
    include_bootstrap=True,
    n_bootstrap=1000
)

print(f"AUCC: {result['aucc']:.4f}")

# Trading diagnostics
trading = compute_trading_cost_diagnostics(y_true, y_prob)
print(f"Optimal RR: {trading['optimal_rr']}")
```

## Demo Script

A comprehensive demonstration is available:

```bash
cd ml_intraday_v3/analysis
python cost_curves_demo.py
```

**Generates:**
- `demo_cost_curve_basic.png` - Basic cost curve
- `demo_cost_curve_bootstrap.png` - With confidence intervals
- `demo_cost_curve_comparison.png` - Multiple models
- `demo_cost_curve_difference.png` - Difference plot
- `demo_trading_cost_curve.png` - Trading-specific

## Understanding Cost Curves

### Key Concepts

**Probability Cost (PC):**
```
PC = c * π₁ / (c * π₁ + (1-c) * π₀)
```
where:
- `c` = cost ratio = `cost(FP) / (cost(FP) + cost(FN))`
- `π₁` = prior probability of positive class
- `π₀` = prior probability of negative class

**Normalized Expected Cost (NC):**
```
NC = (1 - TPR) * PC + FPR * (1 - PC)
```
where:
- `TPR` = True Positive Rate at optimal threshold
- `FPR` = False Positive Rate at optimal threshold

**Area Under Cost Curve (AUCC):**
- Summary metric analogous to ROC AUC
- **Lower is better** (opposite of ROC AUC)
- AUCC ≈ 0: Perfect classifier
- AUCC ≈ 0.5: Random classifier

### Trading Interpretation

For trading, we map **Risk/Reward (RR) ratios** to cost ratios:

```
cost_ratio = 1 / (1 + RR)
```

Examples:
- RR = 1.0 (1:1 risk/reward) → cost_ratio = 0.5
- RR = 2.0 (2:1 risk/reward) → cost_ratio = 0.33
- RR = 3.0 (3:1 risk/reward) → cost_ratio = 0.25

This allows direct evaluation of classifier performance under different trading strategies.

## Integration with Pipeline

### In Notebook Analysis

Add to `ml_intraday_v3_pipeline_runner_enhanced.ipynb`:

```python
from ml_intraday_v3.analysis.cost_curves import compare_models_cost_curves

# After model training
models_for_comparison = {
    'Primary Model': (y_test, primary_probs),
    'Meta Model': (y_test, meta_probs),
    'Baseline': (y_test, baseline_probs)
}

fig, curves = compare_models_cost_curves(models_for_comparison)
```

### In Backtest Reports

```python
from ml_intraday_v3.experiments.diagnostics import compute_trading_cost_diagnostics

# After backtest
y_true = (trades_df['pnl'] > 0).astype(int)
y_prob = trades_df['signal_prob']

diagnostics = compute_trading_cost_diagnostics(
    y_true, y_prob,
    risk_reward_ratios=[1.0, 1.5, 2.0, 2.5, 3.0],
    trade_returns=trades_df['pnl']
)

print(f"Optimal RR: {diagnostics['optimal_rr']}")
print(f"At optimal: TPR={diagnostics['optimal_tpr']:.3f}, FPR={diagnostics['optimal_fpr']:.3f}")
```

## References

Drummond, C., & Holte, R. C. (2006). **Cost curves: An improved method for visualizing classifier performance.** *Machine Learning*, 65(1), 95-130.

Key points from the paper:
1. Cost curves show expected cost across all misclassification cost ratios
2. Lower curves indicate better performance
3. Crossing curves indicate context-dependent performance
4. Strictly dominated classifiers never perform best
5. Statistical tests easier than with ROC curves

## File Structure

```
ml_intraday_v3/
├── analysis/
│   ├── cost_curves.py              # Main implementation
│   ├── cost_curves_demo.py         # Demonstration script
│   └── README_COST_CURVES.md       # This file
├── experiments/
│   └── diagnostics.py              # Integration with diagnostics
└── tests/
    └── test_cost_curves.py         # Comprehensive tests
```

## Performance Notes

- Cost curve computation is O(N log N) due to ROC curve computation
- Bootstrap with 1000 samples takes ~2-5 seconds for N=1000 samples
- For large datasets (N > 100K), consider subsampling for bootstrap
- AUCC computation is O(K) where K is number of PC points (default 100)

## Future Enhancements

Potential additions (not yet implemented):
1. **Magnitude weighting**: Weight costs by trade return magnitude
2. **Regime-specific curves**: Separate curves by volatility regime
3. **Time-series bootstrap**: Account for temporal dependence
4. **Multi-class extension**: Extend to 3+ class problems
5. **Dynamic cost ratios**: Cost ratios that vary over time

## Questions?

For issues or questions about cost curves:
1. Check the demo script: `ml_intraday_v3/analysis/cost_curves_demo.py`
2. Run the tests: `pytest ml_intraday_v3/tests/test_cost_curves.py -v`
3. Review the paper: Drummond & Holte (2006)
