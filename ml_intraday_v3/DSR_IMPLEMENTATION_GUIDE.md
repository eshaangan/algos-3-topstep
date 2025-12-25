# Deflated Sharpe Ratio (DSR) - Implementation Guide

## Overview

This guide covers the complete implementation of the **Deflated Sharpe Ratio (DSR)** from Bailey & López de Prado (2014), including:

- Complete DSR formula accounting for selection bias and non-normality
- Annualization support for different return types
- Visualization and reporting
- Integration with backtest analysis and CPCV

---

## What is DSR?

**Deflated Sharpe Ratio (DSR)** is a selection-bias-adjusted version of the Sharpe ratio that accounts for:

1. **Selection bias** from testing multiple configurations (N trials)
2. **Non-normality** of returns (skewness and kurtosis)
3. **Statistical uncertainty** in Sharpe ratio estimation

### Why DSR Matters

When you test many configurations and select the best, the "best" Sharpe ratio is **biased upward** by chance. DSR corrects for this bias and provides a probabilistic interpretation similar to a p-value.

**Example:**
- Test 100 configurations, pick best with Sharpe = 2.0
- DSR might be only 0.60, indicating "moderate evidence" rather than "strong evidence"
- This prevents overconfidence in backtest results

---

## Complete DSR Formula

```
DSR = Φ((SR - SR*) / σ_SR)
```

where:

### SR (Observed Sharpe Ratio)
```
SR = μ / σ
```
- μ = mean return
- σ = standard deviation of returns

### SR* (Expected Maximum Sharpe from Selection Bias)
```
SR* = SR_0 + σ_SR × Z_{1-1/N}
```
- SR_0 = target/benchmark Sharpe (typically 0.0)
- Z_{1-1/N} = normal quantile at probability 1 - 1/N
- N = number of configurations tested

### σ_SR (Standard Error of Sharpe Ratio)
```
σ_SR = sqrt((1 - γ×SR + (κ-1)/4 × SR²) / (n-1))
```
- γ = skewness of returns = mean(z³)
- κ = kurtosis of returns = mean(z⁴)
- z = standardized returns = (r - μ) / σ
- n = number of observations

### Φ (Standard Normal CDF)
Converts z-score to probability

---

## Implementation

### Basic Usage

```python
from ml_intraday_v3.experiments.diagnostics import compute_dsr

# Your backtest returns
returns = [0.01, -0.005, 0.02, 0.008, ...]  # Per-trade P&L

# Compute DSR
result = compute_dsr(
    returns=returns,
    n_trials=27,  # Number of configurations tested
    target_sharpe=0.0  # Benchmark (0 = test against no skill)
)

print(f"DSR: {result['dsr']:.4f}")
print(f"Sharpe: {result['sharpe']:.4f}")
print(f"SR* (expected max from luck): {result['sr_star']:.4f}")
```

### With Annualization

```python
# Per-trade returns with ~5 trades/day
result = compute_dsr(
    returns=returns,
    n_trials=27,
    annualization_factor=np.sqrt(252 * 5)  # ≈ 35.5
)

print(f"Annualized Sharpe: {result['sharpe']:.4f}")
print(f"Raw Sharpe: {result['sharpe_raw']:.4f}")
```

### Getting n_trials from TrialTracker

```python
from ml_intraday_v3.experiments.trial_tracker import TrialTracker

# Load tracked trials
tracker = TrialTracker("runs/run_20251224_123456")
n_trials = len(tracker.trials)

# Compute DSR with accurate trial count
result = compute_dsr(returns, n_trials=n_trials)
```

---

## DSR Interpretation

| DSR Range | Interpretation | Confidence | Action |
|-----------|----------------|------------|--------|
| > 0.95 | Strong evidence of skill | p < 0.05 equivalent | High confidence for deployment |
| 0.90 - 0.95 | Good evidence | p < 0.10 equivalent | Proceed with monitoring |
| 0.50 - 0.90 | Moderate evidence | More likely skill than luck | Proceed with caution |
| < 0.50 | Weak evidence | Likely overfitting or luck | DO NOT deploy |

### Example Interpretations

**DSR = 0.98** 🟢
- Strong evidence this is genuine skill, not luck
- Equivalent to p-value < 0.02
- High confidence for deployment

**DSR = 0.65** 🟠
- Moderate evidence of skill
- More likely skill than luck, but not conclusive
- Proceed with caution and additional validation

**DSR = 0.35** 🔴
- Weak evidence
- Results likely due to overfitting or random chance
- DO NOT deploy

---

## Annualization

### Per-Trade Returns

```python
# If you have ~5 trades per day on average
trades_per_day = 5
annualization_factor = np.sqrt(252 * trades_per_day)  # ≈ 35.5

result = compute_dsr(
    returns=per_trade_pnl,
    n_trials=n_trials,
    annualization_factor=annualization_factor
)
```

### Time-Based Returns

```python
# For 1m bars, 390 bars/day, holding ~5 bars average
bars_per_day = 390
avg_holding_bars = 5
annualization_factor = np.sqrt(252 * bars_per_day / avg_holding_bars)  # ≈ 140

result = compute_dsr(
    returns=bar_returns,
    n_trials=n_trials,
    annualization_factor=annualization_factor
)
```

---

## Visualization and Reporting

### Generate Report

```python
from ml_intraday_v3.experiments.diagnostics import generate_dsr_report

# Compute DSR
result = compute_dsr(returns, n_trials=27)

# Generate markdown report
report = generate_dsr_report(
    result,
    save_path='dsr_report.md'
)

print(report)
```

### Visualizations

```python
from ml_intraday_v3.experiments.diagnostics import (
    plot_dsr_distribution,
    plot_dsr_with_confidence
)

# Compute DSR for multiple CPCV paths
dsr_results = []
for path in cpcv_paths:
    result = compute_dsr(path_returns, n_trials=27)
    dsr_results.append(result)

# Plot distribution across paths
fig1 = plot_dsr_distribution(dsr_results)
fig1.savefig('dsr_distribution.png')

# Plot with confidence intervals
fig2 = plot_dsr_with_confidence(dsr_results)
fig2.savefig('dsr_with_ci.png')
```

---

## Integration with Backtest Analysis

### Compute DSR Across CPCV Paths

```bash
# Run DSR analysis on backtest results
python -m ml_intraday_v3.analysis.compute_dsr \
    --run-dir runs/run_20251224_123456 \
    --bar-size 1m \
    --n-trials 27 \
    --trades-per-day 5
```

**Output:**
```
================================================================================
DSR Analysis Across CPCV Paths
================================================================================

Loaded n_trials from TrialTracker: 27
Annualization factor: 35.50 (for 5.0 trades/day)
Found 10 CPCV paths

Path 1/10: path_0
  Loaded 1245 trades
  DSR: 0.7823
  Sharpe: 1.45
  SR*: 0.38
  Skewness: -0.15, Kurtosis: 3.42

...

Aggregate Statistics:
  DSR Mean: 0.7453
  DSR Median: 0.7512
  DSR Std: 0.0621
  DSR Range: [0.6234, 0.8456]
  % DSR > 0.5: 100.0%
  % DSR > 0.95: 0.0%

Risk Assessment:
  Risk Level: 🟠 MODERATE EVIDENCE
  Interpretation: More likely skill than luck, but not conclusive
  Action: Proceed with caution, monitor closely

✓ Report saved to: runs/.../dsr_analysis/dsr_report_median.md
✓ DSR distribution saved to: runs/.../dsr_analysis/dsr_distribution_across_paths.png
✓ DSR with CI saved to: runs/.../dsr_analysis/dsr_with_confidence.png
```

---

## Key Insights from DSR

### 1. Selection Bias Penalty

Testing more configurations **increases** the expected maximum Sharpe from luck:

| N Trials | Expected Max SR* (σ_SR = 0.1) |
|----------|-------------------------------|
| 1 | 0.00 |
| 10 | 0.23 |
| 100 | 0.37 |
| 1000 | 0.49 |

**Implication:** The same observed Sharpe yields **lower DSR** with more trials tested.

### 2. Non-Normality Impact

Skewness and kurtosis affect the standard error:

- **High negative skew** (occasional large losses): Increases σ_SR → harder to achieve high DSR
- **High kurtosis** (fat tails): Increases uncertainty → harder to achieve high DSR
- **Normal returns** (γ ≈ 0, κ ≈ 3): Standard formula applies

### 3. Sample Size Matters

More observations → lower σ_SR → easier to achieve high DSR (if skill is real)

**Minimum recommended:** n ≥ 100 trades for reliable DSR

---

## Comparison with Other Metrics

| Metric | What it Measures | Selection Bias Adjusted? | Non-Normality Adjusted? |
|--------|------------------|--------------------------|-------------------------|
| **Sharpe Ratio** | Risk-adjusted return | ❌ No | ❌ No |
| **DSR** | Skill after bias correction | ✅ Yes | ✅ Yes |
| **PBO** | Overfitting probability | ✅ Yes | ❌ No |
| **t-statistic** | Statistical significance | ❌ No | ❌ No |

**Use DSR when:**
- You tested multiple configurations
- Returns are non-normal (skewed or heavy-tailed)
- You want a probabilistic interpretation

**Use PBO when:**
- You have CPCV results
- You want to assess IS/OOS gap
- You want path-level overfitting analysis

**Use both for comprehensive assessment!**

---

## Validation Tests

Run validation tests to verify implementation:

```bash
cd ml_intraday_v3/tests
pytest test_dsr_validation.py -v
```

**Tests verify:**
- ✅ DSR formula correctness
- ✅ Selection bias adjustment
- ✅ Non-normality adjustments
- ✅ Annualization
- ✅ Edge cases

---

## Best Practices

### 1. Always Track n_trials Accurately

```python
# ❌ Bad: Guessing
result = compute_dsr(returns, n_trials=10)  # Just a guess

# ✅ Good: Using TrialTracker
tracker = TrialTracker(run_dir)
result = compute_dsr(returns, n_trials=len(tracker.trials))
```

### 2. Use Conservative Estimates if Unsure

```python
# If you don't know exact count, use conservative (high) estimate
result = compute_dsr(returns, n_trials=50)  # Conservative
```

Higher n_trials → stricter test → less risk of false positives

### 3. Combine with Other Diagnostics

```python
# Compute both DSR and PBO
dsr_result = compute_dsr(returns, n_trials=n_trials)
pbo_result = compute_pbo_with_confidence(trials_df)

print(f"DSR: {dsr_result['dsr']:.3f}")
print(f"PBO: {pbo_result['pbo']:.3f}")

# Deploy only if BOTH look good
if dsr_result['dsr'] > 0.95 and pbo_result['pbo'] < 0.3:
    print("✅ Strong evidence for deployment")
else:
    print("⚠️ Additional validation needed")
```

### 4. Report DSR in All Results

Include DSR in your backtest summary alongside Sharpe ratio:

```
Backtest Results:
  Sharpe Ratio: 1.85
  DSR: 0.78 (Moderate evidence, n_trials=27)
  PBO: 0.32 (Moderate risk)
```

---

## Troubleshooting

### "DSR is None - nonpositive_variance"

**Cause:** Variance of Sharpe ratio formula yielded negative value (rare)

**Fix:**
- Check for extreme skewness or kurtosis
- Increase sample size
- Verify returns are not corrupted

### "DSR is surprisingly low"

**Possible causes:**
1. **High n_trials:** Tested many configurations → large selection bias penalty
2. **Non-normal returns:** High skewness/kurtosis increases uncertainty
3. **Small sample size:** Few observations → large σ_SR
4. **Genuine overfitting:** Results actually are overfitted

**Actions:**
- Review n_trials (is it accurate?)
- Check return distribution (plot histogram, Q-Q plot)
- Increase sample size if possible
- Consider reducing search space

### "DSR varies widely across CPCV paths"

**Interpretation:** Inconsistent performance across validation splits

**Actions:**
- Review individual path results
- Check for regime changes or non-stationarity
- Consider walk-forward validation
- Be cautious about deployment

---

## References

1. **Bailey, D.H., & López de Prado, M. (2014).**
   "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
   *Journal of Portfolio Management*, 40(5), 94-107.
   [DOI: 10.3905/jpm.2014.40.5.094](https://doi.org/10.3905/jpm.2014.40.5.094)

2. **López de Prado, M. (2018).**
   *Advances in Financial Machine Learning.*
   Wiley. Chapters 11-12.

3. **Implementation:**
   - `ml_intraday_v3/experiments/diagnostics.py::compute_dsr()`
   - `ml_intraday_v3/configs/metrics_contract.json`
   - `ml_intraday_v3/analysis/compute_dsr.py`

---

## Quick Reference

### Function Signature

```python
compute_dsr(
    returns: Iterable[float],
    n_trials: int,
    target_sharpe: float = 0.0,
    annualization_factor: Optional[float] = None
) -> Dict
```

### Return Keys

- `dsr`: Deflated Sharpe Ratio [0, 1]
- `sharpe`: Observed Sharpe (annualized if factor provided)
- `sharpe_raw`: Raw Sharpe before annualization
- `sr_star`: Expected max Sharpe from selection bias
- `sr_std`: Standard error of Sharpe
- `skewness`: Skewness of returns
- `kurtosis`: Kurtosis of returns
- `n_obs`: Number of observations
- `n_trials`: Number of trials tested
- `z_score`: Standardized score
- `target_sharpe`: Benchmark Sharpe used

### CLI Usage

```bash
python -m ml_intraday_v3.analysis.compute_dsr \
    --run-dir RUNS/RUN_ID \
    --bar-size {1m|5m} \
    [--n-trials N] \
    [--trades-per-day X] \
    [--target-sharpe Y]
```

---

*Implementation complete and validated. For questions or issues, see tests and diagnostics module.*
