# Enhanced CPCV - Implementation Guide

## Overview

This guide covers the complete implementation of **Enhanced Combinatorial Purged Cross-Validation (CPCV)** with comprehensive path evaluation, distribution analysis, validation gates, and visualization.

Enhanced CPCV extends the baseline CPCV implementation (López de Prado 2018, Chapter 7) with:

- **Multiple path selection strategies** (lexicographic, balanced, random)
- **Comprehensive path performance tracking** across all metrics
- **Distribution analysis** (quartiles, percentiles, IS/OOS ratios)
- **Automated validation gates** for quality control
- **Rich visualization** of performance distributions

---

## What is CPCV?

**Combinatorial Purged Cross-Validation (CPCV)** is a cross-validation method designed for financial time series that:

1. **Combinatorially generates validation paths** from different test fold combinations
2. **Purges overlapping events** from training sets to prevent leakage
3. **Applies embargo** after test periods to prevent information leakage
4. **Evaluates stability** of model performance across different validation splits

### Why CPCV Matters

Traditional k-fold CV can be misleading for financial ML because:
- Standard CV doesn't account for **temporal overlap** between events
- A single train/test split may not represent **out-of-sample variability**
- **Selection bias** occurs when picking the "best" configuration based on one split

CPCV addresses these issues by:
- Testing performance across **multiple independent validation paths**
- Providing **distribution of performance** (not just point estimates)
- Enabling **robust model selection** based on quartiles/percentiles

---

## Path Selection Strategies

Enhanced CPCV offers three path selection strategies:

### 1. Lexicographic (Simple, Deterministic)

Selects the first N paths in natural combinatorial order.

**When to use:**
- Quick baseline evaluation
- When you need deterministic, reproducible path selection
- When computational budget is limited

**Configuration:**
```yaml
cpcv:
  path_selection: "lexicographic"
  max_paths: 10
```

### 2. Balanced (Recommended)

Uses a greedy algorithm to ensure each fold appears in the test set approximately equally.

**How it works:**
- Starts with an empty set of selected paths
- Iteratively selects the path that best balances test fold coverage
- Minimizes variance in fold representation

**When to use:**
- For production model validation (recommended)
- When you want comprehensive coverage across all time periods
- When max_paths < total combinations

**Configuration:**
```yaml
cpcv:
  path_selection: "balanced"
  max_paths: 15
```

**Example:**
```
With K=6 folds, test_groups=2, max_paths=10:
- Total combinations = C(6,2) = 15
- Balanced selection ensures each of 6 folds appears ~3-4 times in test sets
- Lexicographic might only test folds 0-3 heavily
```

### 3. Random (Sensitivity Analysis)

Randomly samples paths with a seed for reproducibility.

**When to use:**
- For sensitivity analysis (try multiple random selections)
- When you want to verify results aren't path-selection dependent
- For bootstrap-style resampling of validation paths

**Configuration:**
```yaml
cpcv:
  path_selection: "random"
  random_state: 42
  max_paths: 10
```

---

## Performance Evaluation

### Evaluate All Paths

The `evaluate_cpcv_paths()` function evaluates model performance across all CPCV paths:

```python
from ml_intraday_v3.validation.cpcv import (
    build_cpcv_paths,
    evaluate_cpcv_paths
)
from sklearn.linear_model import LogisticRegression

# Build paths
paths = build_cpcv_paths(
    events_df=events_df,
    n_groups=6,
    test_groups=2,
    max_paths=15,
    selection="balanced",
    pct_embargo=0.01
)

# Define model factory
def model_factory():
    return LogisticRegression(max_iter=1000, class_weight='balanced')

# Evaluate across all paths
perf_df = evaluate_cpcv_paths(
    paths=paths,
    model_factory=model_factory,
    X=X_features,
    y=y_labels,
    events_df=events_df,
    metrics=['roc_auc', 'accuracy', 'sharpe'],
    sample_weight=sample_weights  # Optional
)

print(perf_df.head())
```

**Output DataFrame:**
```
   path_id  metric_name  metric_value is_oos  n_samples  test_folds
0        0     roc_auc         0.623     IS       8234  [0, 1]
1        0     roc_auc         0.578    OOS       1876  [0, 1]
2        0     accuracy        0.612     IS       8234  [0, 1]
3        0     accuracy        0.591    OOS       1876  [0, 1]
...
```

### Supported Metrics

The `evaluate_cpcv_paths()` function supports:

- **roc_auc**: ROC-AUC (for classification)
- **accuracy**: Classification accuracy
- **precision**: Precision score
- **recall**: Recall score
- **f1**: F1 score
- **sharpe**: Sharpe ratio (requires trades data)
- **mean_return**: Average return per event
- **win_rate**: Fraction of winning events

---

## Distribution Analysis

### Compute Quartiles and Percentiles

The `analyze_cpcv_distributions()` function computes comprehensive statistics:

```python
from ml_intraday_v3.validation.cpcv import analyze_cpcv_distributions

stats = analyze_cpcv_distributions(
    perf_df=perf_df,
    metrics=['roc_auc', 'sharpe']
)

# Access statistics
print(f"OOS Median Sharpe: {stats['sharpe']['OOS']['median']:.3f}")
print(f"OOS Q25 Sharpe: {stats['sharpe']['OOS']['q25']:.3f}")
print(f"OOS Q75 Sharpe: {stats['sharpe']['OOS']['q75']:.3f}")
print(f"IS/OOS Ratio: {stats['sharpe']['is_oos_ratio']:.3f}")
```

**Statistics Returned:**

For each metric and split (IS/OOS):
- **mean**: Average across paths
- **median**: 50th percentile (robust central tendency)
- **std**: Standard deviation
- **q25**: 25th percentile (lower quartile)
- **q75**: 75th percentile (upper quartile)
- **p05**: 5th percentile
- **p95**: 95th percentile
- **min**: Minimum value
- **max**: Maximum value
- **is_oos_ratio**: Median IS / Median OOS

**Example Output:**
```json
{
  "sharpe": {
    "IS": {
      "mean": 1.234,
      "median": 1.198,
      "std": 0.156,
      "q25": 1.089,
      "q75": 1.367,
      "p05": 0.923,
      "p95": 1.521,
      "min": 0.812,
      "max": 1.678
    },
    "OOS": {
      "mean": 0.876,
      "median": 0.892,
      "std": 0.234,
      "q25": 0.712,
      "q75": 1.045,
      "p05": 0.534,
      "p95": 1.234,
      "min": 0.423,
      "max": 1.345
    },
    "is_oos_ratio": 1.343
  }
}
```

---

## Validation Gates

### Automated Quality Checks

The `check_cpcv_gates()` function performs automated validation checks:

```python
from ml_intraday_v3.validation.cpcv import check_cpcv_gates

# Define custom gate configuration (or use defaults from validation.yaml)
gates_config = {
    'median_sharpe': {'threshold': 0.8, 'enabled': True},
    'p25_sharpe': {'threshold': 0.2, 'enabled': True},
    'median_roc_auc': {'threshold': 0.55, 'enabled': True},
    'is_oos_sharpe_ratio': {'threshold': 1.5, 'enabled': True}
}

gates_result = check_cpcv_gates(
    perf_df=perf_df,
    stats=stats,  # Optional; will compute if None
    gates_config=gates_config
)

# Check results
if gates_result['all_passed']:
    print("✅ All validation gates passed!")
else:
    print(f"⚠️  {len(gates_result['gates_failed'])} gates failed")
    for failure in gates_result['gates_failed']:
        print(f"  - {failure['gate_name']}: {failure['actual']:.3f} vs {failure['threshold']:.3f}")
```

### Default Validation Gates

From `validation.yaml`:

| Gate | Threshold | Interpretation |
|------|-----------|----------------|
| **median_sharpe** | ≥ 0.8 | Typical OOS performance meets minimum skill bar |
| **p25_sharpe** | ≥ 0.2 | Worst quartile still shows positive performance |
| **median_roc_auc** | ≥ 0.55 | Classification better than random |
| **is_oos_sharpe_ratio** | ≤ 1.5 | IS not significantly better than OOS (low overfitting) |
| **pbo** | ≤ 0.5 | More likely skill than luck |
| **dsr** | ≥ 0.5 | Moderate evidence of skill |

### Gate Failure Actions

Configure what happens on gate failure in `validation.yaml`:

```yaml
cpcv:
  validation_gates:
    # ... gate definitions ...
  gate_failure_action: "warn"  # or "fail", "log"
```

- **warn**: Print warning but continue (recommended for research)
- **fail**: Halt execution and raise error (recommended for production)
- **log**: Log to file but continue silently

---

## Visualization

### Performance Distribution Plots

```python
from ml_intraday_v3.validation.cpcv import plot_cpcv_performance
import matplotlib.pyplot as plt

# Create box + violin plot
fig = plot_cpcv_performance(
    perf_df=perf_df,
    metric='sharpe',
    show_thresholds=True
)

plt.savefig('cpcv_sharpe_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Plot Features:**
- **Box plot**: Shows quartiles (Q1, median, Q3) and outliers
- **Violin plot**: Shows full distribution shape
- **Threshold lines**: Marks validation gate thresholds
- **Statistics box**: Displays key statistics
- **IS vs OOS**: Side-by-side comparison

---

## Complete Workflow Example

### End-to-End CPCV Evaluation

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml_intraday_v3.validation.cpcv import (
    build_cpcv_paths,
    evaluate_cpcv_paths,
    analyze_cpcv_distributions,
    check_cpcv_gates,
    plot_cpcv_performance
)

# 1. Load data
events_df = pd.read_parquet('data/events.parquet')
X = pd.read_parquet('data/features.parquet')
y = pd.read_parquet('data/labels.parquet')['label']
sample_weights = pd.read_parquet('data/weights.parquet')['weight']

# 2. Build CPCV paths with balanced selection
paths = build_cpcv_paths(
    events_df=events_df,
    n_groups=6,
    test_groups=2,
    max_paths=15,
    selection="balanced",
    pct_embargo=0.01,
    random_state=42
)

print(f"✓ Built {len(paths)} CPCV paths")

# 3. Evaluate performance across all paths
def model_factory():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=50,
        random_state=42
    )

perf_df = evaluate_cpcv_paths(
    paths=paths,
    model_factory=model_factory,
    X=X,
    y=y,
    events_df=events_df,
    metrics=['roc_auc', 'accuracy', 'sharpe'],
    sample_weight=sample_weights
)

print(f"✓ Evaluated {len(perf_df)} metric values across paths")

# 4. Analyze distributions
stats = analyze_cpcv_distributions(
    perf_df=perf_df,
    metrics=['roc_auc', 'sharpe']
)

print("\n📊 Distribution Statistics:")
print(f"  OOS Sharpe Median: {stats['sharpe']['OOS']['median']:.3f}")
print(f"  OOS Sharpe Q25: {stats['sharpe']['OOS']['q25']:.3f}")
print(f"  OOS Sharpe Q75: {stats['sharpe']['OOS']['q75']:.3f}")
print(f"  IS/OOS Ratio: {stats['sharpe']['is_oos_ratio']:.3f}")

# 5. Check validation gates
gates_result = check_cpcv_gates(
    perf_df=perf_df,
    stats=stats
)

print(f"\n🚦 Validation Gates:")
print(f"  Passed: {len(gates_result['gates_passed'])}")
print(f"  Failed: {len(gates_result['gates_failed'])}")

if gates_result['all_passed']:
    print("  ✅ All gates passed - model is deployment-ready")
else:
    print("  ⚠️  Some gates failed:")
    for failure in gates_result['gates_failed']:
        print(f"    - {failure['gate_name']}: {failure['actual']:.3f} vs threshold {failure['threshold']:.3f}")

# 6. Visualize
fig = plot_cpcv_performance(
    perf_df=perf_df,
    metric='sharpe',
    show_thresholds=True
)
fig.savefig('cpcv_sharpe_distribution.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved visualization to cpcv_sharpe_distribution.png")

# 7. Save results
perf_df.to_csv('cpcv_performance_results.csv', index=False)
with open('cpcv_statistics.json', 'w') as f:
    import json
    json.dump(stats, f, indent=2)

print("\n✓ Saved results to CSV and JSON")
```

---

## Integration with Training Pipeline

### Use CPCV for Hyperparameter Selection

```python
from ml_intraday_v3.validation.cpcv import build_cpcv_paths, evaluate_cpcv_paths

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1]
}

# Build CPCV paths once
paths = build_cpcv_paths(
    events_df=events_df,
    n_groups=6,
    test_groups=2,
    max_paths=15,
    selection="balanced",
    pct_embargo=0.01
)

best_config = None
best_median_sharpe = -np.inf

# Grid search
for n_est in param_grid['n_estimators']:
    for max_d in param_grid['max_depth']:
        for lr in param_grid['learning_rate']:

            # Define model with current params
            def model_factory():
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    learning_rate=lr,
                    random_state=42
                )

            # Evaluate on CPCV paths
            perf_df = evaluate_cpcv_paths(
                paths=paths,
                model_factory=model_factory,
                X=X, y=y,
                events_df=events_df,
                metrics=['sharpe'],
                sample_weight=sample_weights
            )

            # Compute median OOS Sharpe
            oos_sharpe = perf_df[
                (perf_df['metric_name'] == 'sharpe') &
                (perf_df['is_oos'] == 'OOS')
            ]['metric_value']

            median_sharpe = oos_sharpe.median()

            print(f"Config n_est={n_est}, max_depth={max_d}, lr={lr}: "
                  f"Median OOS Sharpe = {median_sharpe:.3f}")

            # Track best based on median OOS Sharpe
            if median_sharpe > best_median_sharpe:
                best_median_sharpe = median_sharpe
                best_config = {
                    'n_estimators': n_est,
                    'max_depth': max_d,
                    'learning_rate': lr
                }

print(f"\n🏆 Best Configuration:")
print(f"  {best_config}")
print(f"  Median OOS Sharpe: {best_median_sharpe:.3f}")
```

---

## Configuration Reference

### validation.yaml CPCV Section

```yaml
cpcv:
  enabled: true
  n_groups: 6
  test_groups: 2
  max_paths: 15

  path_selection: "balanced"  # or "lexicographic", "random"
  random_state: 42

  apply_purging: true
  apply_embargo: true
  use_purged_cv_embargo: true

  validation_gates:
    median_sharpe:
      threshold: 0.8
      enabled: true
      description: "Median OOS Sharpe ratio across CPCV paths"

    p25_sharpe:
      threshold: 0.2
      enabled: true
      description: "25th percentile OOS Sharpe"

    median_roc_auc:
      threshold: 0.55
      enabled: true
      description: "Median OOS ROC-AUC"

    is_oos_sharpe_ratio:
      threshold: 1.5
      enabled: true
      description: "IS/OOS ratio (< 1.5 preferred)"

    pbo:
      threshold: 0.5
      enabled: true
      description: "PBO < 0.5 preferred"

    dsr:
      threshold: 0.5
      enabled: true
      description: "DSR > 0.5 preferred"

  gate_failure_action: "warn"  # or "fail", "log"
```

---

## Best Practices

### 1. Use Balanced Path Selection

For production model validation, always use `selection="balanced"` to ensure comprehensive coverage across all time periods.

### 2. Focus on Quartiles, Not Just Median

Don't just optimize for median performance—check the 25th percentile (Q1) to ensure worst-case scenarios are acceptable.

```python
# ❌ Bad: Only check median
median_sharpe = stats['sharpe']['OOS']['median']

# ✅ Good: Check quartiles
median_sharpe = stats['sharpe']['OOS']['median']
q25_sharpe = stats['sharpe']['OOS']['q25']

if median_sharpe > 0.8 and q25_sharpe > 0.2:
    print("Performance is stable across paths")
```

### 3. Monitor IS/OOS Ratio

An IS/OOS ratio > 1.5 suggests overfitting, even if OOS performance looks acceptable.

```python
if stats['sharpe']['is_oos_ratio'] > 1.5:
    print("⚠️  Warning: IS performance significantly better than OOS")
    print("   This suggests overfitting - consider simplifying model")
```

### 4. Combine with PBO and DSR

Use CPCV alongside PBO and DSR for comprehensive overfitting assessment:

```python
# Compute all three
cpcv_stats = analyze_cpcv_distributions(perf_df, metrics=['sharpe'])
pbo_result = compute_pbo(trials_df)
dsr_result = compute_dsr(returns, n_trials=len(trials_df))

print(f"CPCV OOS Median Sharpe: {cpcv_stats['sharpe']['OOS']['median']:.3f}")
print(f"PBO: {pbo_result['pbo']:.3f}")
print(f"DSR: {dsr_result['dsr']:.3f}")

# Deploy only if all look good
if (cpcv_stats['sharpe']['OOS']['median'] > 0.8 and
    pbo_result['pbo'] < 0.5 and
    dsr_result['dsr'] > 0.5):
    print("✅ All diagnostics look good for deployment")
```

### 5. Save All Path Results

Always save the full `perf_df` DataFrame for later analysis and auditing.

```python
# Save detailed results
perf_df.to_parquet('runs/{run_id}/cpcv_performance_all_paths.parquet')

# Save summary statistics
import json
with open('runs/{run_id}/cpcv_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)
```

---

## Troubleshooting

### "All paths have same performance"

**Cause:** Dataset too small or too homogeneous

**Fix:**
- Increase dataset size
- Check if there's sufficient variation across time periods
- Verify folds are properly separated in time

### "High variance across paths"

**Cause:** Non-stationary performance or regime-dependent behavior

**Actions:**
- Investigate individual paths to identify problematic periods
- Consider regime-aware modeling
- Add regime features to the model
- Use walk-forward validation for non-stationary data

### "IS/OOS ratio is very high"

**Cause:** Overfitting

**Actions:**
- Simplify model (reduce capacity, increase regularization)
- Add more features to reduce noise fitting
- Increase sample weight on more unique events
- Check for leakage in features

### "Many gates failing"

**Interpretation:** Model not ready for deployment

**Actions:**
1. Check if thresholds are too strict for your use case
2. Review individual path results to understand failures
3. Consider model improvements before deployment
4. If appropriate, adjust gate thresholds (but be conservative!)

---

## References

1. **López de Prado, M. (2018).**
   *Advances in Financial Machine Learning.*
   Wiley. Chapter 7: "Cross-Validation in Finance."

2. **Bailey, D.H., Borwein, J., López de Prado, M., & Zhu, Q.J. (2014).**
   "Pseudomathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance."
   *Notices of the AMS*, 61(5), 458-471.

3. **Implementation:**
   - `ml_intraday_v3/validation/cpcv.py`
   - `ml_intraday_v3/configs/validation.yaml`
   - `ml_intraday_v3/tests/test_cpcv_enhanced.py`

---

## Quick Reference

### Function Signatures

```python
# Build CPCV paths
build_cpcv_paths(
    events_df: pd.DataFrame,
    n_groups: int,
    test_groups: int,
    max_paths: int,
    selection: str = "lexicographic",  # or "balanced", "random"
    random_state: Optional[int] = None,
    pct_embargo: float = 0.01,
    apply_purging: bool = True
) -> List[Dict]

# Evaluate performance across paths
evaluate_cpcv_paths(
    paths: List[Dict],
    model_factory: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    events_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    sample_weight: Optional[pd.Series] = None
) -> pd.DataFrame

# Analyze distributions
analyze_cpcv_distributions(
    perf_df: pd.DataFrame,
    metrics: Optional[List[str]] = None
) -> Dict[str, Dict]

# Check validation gates
check_cpcv_gates(
    perf_df: pd.DataFrame,
    stats: Optional[Dict] = None,
    gates_config: Optional[Dict] = None
) -> Dict[str, Any]

# Plot performance
plot_cpcv_performance(
    perf_df: pd.DataFrame,
    metric: str = 'sharpe',
    ax: Optional[plt.Axes] = None,
    show_thresholds: bool = True
) -> plt.Figure
```

---

*Enhanced CPCV implementation complete and validated. For questions or issues, see tests and validation module.*
