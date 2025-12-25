# Enhanced PBO (Probability of Backtest Overfitting) - Complete Guide

## Overview

This implementation provides a comprehensive system for detecting backtest overfitting using the Probability of Backtest Overfitting (PBO) metric as described by López de Prado (2018).

**Key Innovation**: Unlike naive implementations that only track winning configurations, this system tracks **ALL trials/configurations tested**, which is critical for unbiased PBO computation.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [What is PBO?](#what-is-pbo)
3. [Why Track All Trials?](#why-track-all-trials)
4. [Components](#components)
5. [Usage Guide](#usage-guide)
6. [Configuration](#configuration)
7. [Interpretation](#interpretation)
8. [API Reference](#api-reference)
9. [Examples](#examples)
10. [Validation](#validation)

---

## Quick Start

### 30-Second Example

```python
from ml_intraday_v3.experiments.trial_tracker import TrialTracker
from ml_intraday_v3.experiments.diagnostics import compute_pbo_with_confidence, generate_pbo_report

# 1. Create trial tracker
tracker = TrialTracker(run_dir)

# 2. During hyperparameter search, log EVERY trial
for config in all_configurations:
    trial_id = tracker.log_trial(
        config=config,
        model_type='logreg',
        hyperparameters=config['model']['params'],
    )

    # Run CPCV and update metrics for each path
    for path_id, (is_metric, oos_metric) in cpcv_results.items():
        tracker.update_path_metrics(trial_id, path_id, is_metric, oos_metric)

# 3. Save trials
tracker.save()

# 4. Compute PBO with confidence intervals
trials_df = tracker.to_dataframe()
pbo_result = compute_pbo_with_confidence(trials_df, n_bootstrap=1000)

# 5. Generate report
report = generate_pbo_report(pbo_result, save_path='pbo_report.md')
print(f"PBO = {pbo_result['pbo']:.3f} (CI: [{pbo_result['pbo_lower']:.3f}, {pbo_result['pbo_upper']:.3f}])")
```

---

## What is PBO?

**PBO (Probability of Backtest Overfitting)** measures the likelihood that your "best" configuration was selected due to overfitting rather than true predictive skill.

### The Problem

When testing multiple configurations (e.g., hyperparameter search):
- You select the best configuration based on in-sample (IS) validation performance
- But IS performance is biased upward due to selection
- How likely is the selected config to perform below median out-of-sample (OOS)?

### The Solution

PBO quantifies this risk:

**For each CPCV path:**
1. Select best configuration using IS performance on all other paths
2. Measure that configuration's OOS rank on the held-out path
3. Lambda (λ) = percentile rank (1.0 = best, 0.0 = worst)

**PBO = fraction of paths where λ < 0.5 (below median)**

### Interpretation

| PBO Value | Interpretation | Action |
|-----------|----------------|--------|
| **< 0.3** | 🟢 Low Risk | Configuration appears robust |
| **0.3 - 0.5** | 🟠 Moderate Risk | Monitor carefully, validate further |
| **> 0.5** | 🔴 High Risk | Likely overfitting, reduce search space or increase sample size |

---

## Why Track All Trials?

**Critical Insight**: Selection bias occurs when only tracking winning configurations.

### ❌ Naive Approach (Biased)

```python
# WRONG: Only track the winning configuration
best_config = select_best_by_validation()
save_config(best_config)  # Selection bias!
```

**Problem**: You don't know how many configurations were tested or how they performed. PBO cannot be computed accurately.

### ✅ Correct Approach (Unbiased)

```python
# CORRECT: Track ALL configurations tested
for config in all_configurations:
    log_trial(config, is_perf, oos_perf)  # Track everything

# Now you can compute unbiased PBO
pbo = compute_pbo(all_trials)
```

**Why it matters**:
- Testing 100 configs has higher selection bias than testing 10
- PBO requires knowing the full distribution of performance
- Without tracking all trials, PBO is meaningless

---

## Components

### 1. `trial_tracker.py` - Trial Tracking System

**Purpose**: Track all configurations/trials tested during model development.

**Key Classes:**
- `Trial`: Dataclass representing a single trial
- `TrialTracker`: Manages trial logging, storage, and retrieval

**Features:**
- Automatic config hashing for deduplication
- JSON persistence
- Conversion to DataFrame for analysis
- Summary statistics

### 2. `diagnostics.py` - Enhanced PBO Computation

**Purpose**: Compute PBO with bootstrap confidence intervals and visualizations.

**Key Functions:**
- `compute_pbo_enhanced()`: Core PBO algorithm
- `compute_pbo_with_confidence()`: PBO with bootstrap CI
- `plot_pbo_distribution()`: Visualize lambda distribution
- `plot_pbo_with_confidence()`: Visualize PBO with CI
- `generate_pbo_report()`: Generate markdown report

### 3. `validation.yaml` - Configuration

**Purpose**: Configure PBO computation parameters.

**Key Settings:**
- `track_all_trials`: Enable comprehensive trial tracking
- `metric`: Primary metric for PBO (e.g., 'roc_auc')
- `n_bootstrap`: Bootstrap samples for CI
- `warning_threshold`: PBO threshold for alerts

---

## Usage Guide

### Step 1: Initialize Trial Tracker

```python
from ml_intraday_v3.experiments.trial_tracker import TrialTracker

tracker = TrialTracker(run_dir='/path/to/run')
```

**Creates**:
- `{run_dir}/trials/` directory
- Loads existing trials if present

### Step 2: Log Trials During Hyperparameter Search

```python
# Example: Grid search over C and penalty
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    for penalty in ['l1', 'l2']:
        config = {
            'model': {'kind': 'logreg', 'params': {'C': C, 'penalty': penalty}},
            'seed': 42
        }

        # Log trial
        trial_id = tracker.log_trial(
            config=config,
            model_type='logreg',
            hyperparameters={'C': C, 'penalty': penalty},
            features=['rsi', 'macd', 'bbands'],  # Optional
            metadata={'note': 'baseline_search'}   # Optional
        )

        # Train and evaluate on CPCV paths
        # ... your training code ...

        # Update with CPCV results
        for path_id in range(n_cpcv_paths):
            is_metric = ...  # In-sample ROC-AUC on other paths
            oos_metric = ... # Out-of-sample ROC-AUC on this path

            tracker.update_path_metrics(
                trial_id=trial_id,
                path_id=f'path_{path_id}',
                is_metric=is_metric,
                oos_metric=oos_metric
            )

# Save to disk
tracker.save()
```

### Step 3: Compute PBO

```python
from ml_intraday_v3.experiments.diagnostics import compute_pbo_with_confidence

# Convert trials to DataFrame
trials_df = tracker.to_dataframe()

# Compute PBO with 95% confidence intervals
pbo_result = compute_pbo_with_confidence(
    trials_df,
    metric_name='roc_auc',
    higher_is_better=True,
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=42
)

print(f"PBO = {pbo_result['pbo']:.3f}")
print(f"95% CI: [{pbo_result['pbo_lower']:.3f}, {pbo_result['pbo_upper']:.3f}]")
print(f"Trials tracked: {pbo_result['n_trials']}")
print(f"CPCV paths: {pbo_result['n_paths']}")
```

### Step 4: Generate Report and Visualizations

```python
from ml_intraday_v3.experiments.diagnostics import (
    generate_pbo_report,
    plot_pbo_distribution,
    plot_pbo_with_confidence
)

# Generate markdown report
report = generate_pbo_report(
    pbo_result,
    save_path='runs/my_run/pbo_report.md'
)

# Create visualizations
fig1 = plot_pbo_distribution(
    lambda_values=pbo_result['lambda_values'],
    pbo_value=pbo_result['pbo']
)
fig1.savefig('pbo_lambda_distribution.png', dpi=150, bbox_inches='tight')

fig2 = plot_pbo_with_confidence(pbo_result)
fig2.savefig('pbo_with_ci.png', dpi=150, bbox_inches='tight')
```

---

## Configuration

Update `ml_intraday_v3/configs/validation.yaml`:

```yaml
overfitting_diagnostics:
  pbo:
    enabled: true
    track_all_trials: true        # CRITICAL: Track all trials, not just winners
    metric: "roc_auc"              # Primary metric for PBO
    higher_is_better: true         # True for AUC, Sharpe; False for loss
    n_bootstrap: 1000              # Bootstrap samples for CI
    confidence_level: 0.95         # 95% confidence intervals
    random_state: 42               # Reproducibility
    warning_threshold: 0.5         # Alert if PBO > 0.5
    min_trials: 2                  # Minimum trials to compute PBO
```

---

## Interpretation

### Understanding Lambda (λ)

Lambda represents the **percentile rank** of the selected configuration on each held-out path:

- **λ = 1.0**: Best performing configuration on that path
- **λ = 0.7**: 70th percentile (better than 70% of configurations)
- **λ = 0.5**: Median performance
- **λ = 0.3**: 30th percentile (worse than 70% of configurations)
- **λ = 0.0**: Worst performing configuration

### PBO Formula

```
PBO = (1/N) * Σ I(λᵢ < 0.5)
```

where:
- N = number of CPCV paths
- I = indicator function (1 if true, 0 if false)
- λᵢ = lambda value for path i

### Example Scenarios

**Scenario 1: No Overfitting (PBO = 0.0)**
```
Lambda values: [0.8, 0.9, 0.7, 0.85, 0.75]
All λ > 0.5 → PBO = 0.0
Interpretation: Selected config consistently performs above median OOS
```

**Scenario 2: Moderate Overfitting (PBO = 0.4)**
```
Lambda values: [0.6, 0.3, 0.7, 0.4, 0.9]
2 out of 5 paths have λ < 0.5 → PBO = 0.4
Interpretation: Some overfitting, monitor carefully
```

**Scenario 3: Severe Overfitting (PBO = 0.8)**
```
Lambda values: [0.2, 0.4, 0.3, 0.6, 0.1]
4 out of 5 paths have λ < 0.5 → PBO = 0.8
Interpretation: High risk, likely overfitting to validation
```

### Decision Framework

```
if PBO < 0.3:
    ✓ Proceed with selected configuration
    ✓ Configuration appears robust

elif 0.3 ≤ PBO ≤ 0.5:
    ⚠ Monitor performance closely
    ⚠ Consider additional validation
    ⚠ Be conservative with position sizing

else:  # PBO > 0.5
    ✗ Do not deploy configuration
    ✗ Reduce hyperparameter search space
    ✗ Increase sample size
    ✗ Use simpler models
    ✗ Consider ensemble methods
```

---

## API Reference

### TrialTracker

```python
class TrialTracker:
    def __init__(self, run_dir: Path | str)

    def log_trial(
        self,
        config: Dict[str, Any],
        model_type: str,
        hyperparameters: Dict[str, Any],
        features: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trial_id: Optional[str] = None,
    ) -> str

    def update_path_metrics(
        self,
        trial_id: str,
        path_id: str,
        is_metric: float,
        oos_metric: float,
    )

    def save(self)

    def to_dataframe(self) -> pd.DataFrame

    def get_summary_stats(self) -> Dict[str, Any]
```

### PBO Functions

```python
def compute_pbo_enhanced(
    trials_df: pd.DataFrame,
    metric_name: str = 'roc_auc',
    higher_is_better: bool = True,
    min_trials: int = 2,
) -> Dict

def compute_pbo_with_confidence(
    trials_df: pd.DataFrame,
    metric_name: str = 'roc_auc',
    higher_is_better: bool = True,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> Dict

def plot_pbo_distribution(
    lambda_values: List[float],
    pbo_value: float,
    ax: Optional[plt.Axes] = None,
    title: str = "PBO Distribution of Lambda Values",
) -> plt.Figure

def generate_pbo_report(
    pbo_result: Dict,
    save_path: Optional[str] = None,
) -> str
```

---

## Examples

### Example 1: Simple Hyperparameter Search

```python
from ml_intraday_v3.experiments.trial_tracker import TrialTracker
from ml_intraday_v3.experiments.diagnostics import compute_pbo_with_confidence

# Initialize
tracker = TrialTracker('runs/my_run')

# Grid search
C_values = [0.01, 0.1, 1.0, 10.0]
for C in C_values:
    config = {'model': {'params': {'C': C}}}
    trial_id = tracker.log_trial(config, 'logreg', {'C': C})

    # Simulate CPCV (replace with real evaluation)
    for path_idx in range(5):
        is_auc = 0.70 + np.random.rand() * 0.10
        oos_auc = is_auc - 0.05 + np.random.rand() * 0.03
        tracker.update_path_metrics(trial_id, f'path_{path_idx}', is_auc, oos_auc)

tracker.save()

# Compute PBO
trials_df = tracker.to_dataframe()
pbo = compute_pbo_with_confidence(trials_df)
print(f"PBO = {pbo['pbo']:.3f} [{pbo['pbo_lower']:.3f}, {pbo['pbo_upper']:.3f}]")
```

### Example 2: Integration with Training Pipeline

```python
# In your training script
from ml_intraday_v3.experiments.trial_tracker import TrialTracker

def train_with_pbo_tracking(run_dir, hyperparameter_grid, cpcv_splitter):
    tracker = TrialTracker(run_dir)

    for params in hyperparameter_grid:
        # Create config
        config = build_config(params)
        trial_id = tracker.log_trial(
            config=config,
            model_type=config['model']['kind'],
            hyperparameters=params
        )

        # Run CPCV
        for path_idx, (train_idx, test_idx) in enumerate(cpcv_splitter):
            # Train model
            model = train_model(train_idx, params)

            # Evaluate
            is_metric = evaluate_is(model, train_idx)
            oos_metric = evaluate_oos(model, test_idx)

            # Log metrics
            tracker.update_path_metrics(
                trial_id,
                f'path_{path_idx}',
                is_metric,
                oos_metric
            )

    tracker.save()
    return tracker
```

---

## Validation

### Synthetic Data Tests

The implementation includes comprehensive tests with synthetic data:

1. **No overfitting**: IS ≈ OOS → PBO should be low (<0.3)
2. **Moderate overfitting**: IS slightly > OOS → PBO moderate (0.3-0.7)
3. **Severe overfitting**: IS >> OOS → PBO high (>0.5)
4. **Selection bias**: PBO increases with more trials

Run tests:
```bash
cd ml_intraday_v3/tests
pytest test_pbo_enhanced.py -v
```

### Expected Behaviors

✅ **PBO increases with more trials** (selection bias)
✅ **PBO is robust to random noise** (bootstrap CI)
✅ **PBO correctly identifies overfitting scenarios**
✅ **Lambda values sum to expected distribution**

---

## References

1. **López de Prado, M. (2018).** *Advances in Financial Machine Learning.* Chapter 11: The Dangers of Backtesting.

2. **Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014).** "Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance." *Notices of the AMS*, 61(5), 458-471.

3. **Bailey, D. H., & López de Prado, M. (2014).** "The deflated Sharpe ratio: correcting for selection bias, backtest overfitting, and non-normality." *Journal of Portfolio Management*, 40(5), 94-107.

---

## Troubleshooting

### Problem: PBO is None

**Possible causes:**
- Empty trials DataFrame
- Insufficient trials (< min_trials)
- Missing IS/OOS columns

**Solution:**
```python
# Check trials
print(tracker.get_summary_stats())

# Validate DataFrame
trials_df = tracker.to_dataframe()
print(trials_df.columns)
print(f"Trials: {len(trials_df)}")
```

### Problem: PBO Always High

**Possible causes:**
- Too many trials (high selection bias)
- Genuine overfitting to validation
- Sample size too small

**Solutions:**
- Reduce hyperparameter search space
- Increase training sample size
- Use simpler models
- Implement early stopping based on PBO

### Problem: Lambda Values Not Sensible

**Check:**
- Correct `higher_is_better` setting
- IS/OOS metrics on same scale
- No data leakage in metric computation

---

## Best Practices

1. **Always track all trials** - Never cherry-pick what to log
2. **Use CPCV** - PBO requires multiple validation paths
3. **Monitor PBO during search** - Stop if PBO > threshold
4. **Set realistic thresholds** - Default 0.5 is conservative
5. **Bootstrap for uncertainty** - Especially with few paths
6. **Validate on walk-forward** - PBO is one input, not the only input
7. **Document assumptions** - Record metric definitions and preprocessing

---

## License

Part of ML Intraday V3 Pipeline. See project LICENSE.
