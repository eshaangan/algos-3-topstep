## Regime-Aware Feature Scaling - Implementation Guide

## Overview

This guide covers the complete implementation of **Regime-Aware Feature Scaling** to prevent distribution shifts between train/test when market conditions change.

Regime-aware scaling normalizes features using **regime-specific statistics** rather than global statistics, ensuring that features remain properly scaled even when market regimes differ between training and testing periods.

### Key Components

1. **Regime Detection** - Classify market conditions (volatility, trend)
2. **Regime-Aware Scaler** - Normalize features per regime
3. **Probability Calibration** - Ensure reliable probability estimates

---

## Why Regime-Aware Scaling Matters

### The Problem: Distribution Shift

Standard scaling computes global statistics across all data:
```python
# Standard scaling
X_scaled = (X - mean_global) / std_global
```

**Problem**: If market volatility changes between train/test, the feature distributions shift:
- Train data: Low volatility period (std ≈ 0.5)
- Test data: High volatility period (std ≈ 2.0)
- Result: Test features have different distributions → model performance degrades

### The Solution: Regime-Aware Scaling

Regime-aware scaling computes statistics **per market regime**:
```python
# Regime-aware scaling
if regime == "low_vol":
    X_scaled = (X - mean_low_vol) / std_low_vol
elif regime == "high_vol":
    X_scaled = (X - mean_high_vol) / std_high_vol
```

**Benefit**: Features have consistent distributions within each regime, regardless of which regimes appear in train vs test.

---

## Quick Start

### Basic Usage

```python
import numpy as np
import pandas as pd
from ml_intraday_v3.features.regime_detector import detect_combined_regime
from ml_intraday_v3.features.regime_scaler import RegimeAwareScaler

# 1. Load your price data
prices = pd.Series([...])  # Your price series

# 2. Detect regimes
vol_regime, trend_regime, combined_regime = detect_combined_regime(
    prices,
    vol_window=20,
    trend_window=50
)

# 3. Prepare features
X = pd.DataFrame({
    'feature_0': [...],
    'feature_1': [...],
    'feature_2': [...]
})

# 4. Fit regime-aware scaler
scaler = RegimeAwareScaler()
scaler.fit(X, regime_labels=combined_regime)

# 5. Transform features
X_scaled = scaler.transform(X, regime_labels=combined_regime)

# Now X_scaled has mean≈0, std≈1 within each regime
```

---

## Regime Detection

### Volatility Regimes

Classify volatility as low/medium/high based on rolling standard deviation:

```python
from ml_intraday_v3.features.regime_detector import detect_volatility_regime

# Detect volatility regime
vol_regime = detect_volatility_regime(
    returns=returns,  # Return series (not prices!)
    window=20,        # Rolling window (bars)
    n_regimes=3,      # 3 = low/medium/high
    method="quantile" # Classification method
)

# Output: pd.Series with values 0 (low), 1 (medium), 2 (high)
print(vol_regime.value_counts())
```

**Parameters:**
- `returns`: Return series (pct_change of prices)
- `window`: Rolling window for volatility computation (default: 20 bars)
- `n_regimes`: Number of regimes (2 or 3)
- `method`: "quantile" (only supported method currently)

### Trend Regimes

Classify trend as downtrend/sideways/uptrend based on price slope:

```python
from ml_intraday_v3.features.regime_detector import detect_trend_regime

# Detect trend regime
trend_regime = detect_trend_regime(
    prices=prices,    # Price series (not returns!)
    window=50,        # Rolling window (bars)
    n_regimes=3,      # 3 = down/sideways/up
    method="slope"    # "slope" or "ma_diff"
)

# Output: pd.Series with values 0 (down), 1 (sideways), 2 (up)
print(trend_regime.value_counts())
```

**Parameters:**
- `prices`: Price series (not returns!)
- `window`: Rolling window for trend computation (default: 50 bars)
- `n_regimes`: Number of regimes (2 or 3)
- `method`: "slope" (linear regression) or "ma_diff" (MA deviation)

### Combined Regimes

Combine volatility + trend for more granular regimes:

```python
from ml_intraday_v3.features.regime_detector import detect_combined_regime, get_regime_labels

# Detect combined regime
vol_regime, trend_regime, combined_regime = detect_combined_regime(
    prices=prices,
    vol_window=20,
    trend_window=50,
    n_vol_regimes=3,
    n_trend_regimes=3
)

# Combined regime ranges from 0 to 8 (3×3-1)
# Get human-readable labels
labels = get_regime_labels(n_vol_regimes=3, n_trend_regimes=3)
print(labels[0])  # "low_vol_downtrend"
print(labels[4])  # "medium_vol_sideways"
print(labels[8])  # "high_vol_uptrend"
```

---

## RegimeAwareScaler

### Sklearn-Compatible API

The `RegimeAwareScaler` follows sklearn's transformer API:

```python
from ml_intraday_v3.features.regime_scaler import RegimeAwareScaler

scaler = RegimeAwareScaler(
    method="standard",              # Scaling method
    min_samples_per_regime=10,      # Min samples per regime
    fallback_to_global=True         # Use global stats for small regimes
)

# Fit on training data
scaler.fit(X_train, regime_labels=regime_train)

# Transform training data
X_train_scaled = scaler.transform(X_train, regime_labels=regime_train)

# Transform test data (using fitted statistics)
X_test_scaled = scaler.transform(X_test, regime_labels=regime_test)

# Or fit and transform in one step
X_train_scaled = scaler.fit_transform(X_train, regime_labels=regime_train)
```

### Train/Test Workflow

```python
# 1. Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # Time-series: no shuffle!
)

# 2. Split regime labels
regime_train = combined_regime[:len(X_train)]
regime_test = combined_regime[len(X_train):]

# 3. Fit scaler on train only
scaler = RegimeAwareScaler()
scaler.fit(X_train, regime_labels=regime_train)

# 4. Transform both train and test
X_train_scaled = scaler.transform(X_train, regime_labels=regime_train)
X_test_scaled = scaler.transform(X_test, regime_labels=regime_test)

# 5. Train model on scaled features
model.fit(X_train_scaled, y_train)

# 6. Predict on scaled test features
y_pred = model.predict(X_test_scaled)
```

### Inverse Transform

```python
# Transform features
X_scaled = scaler.transform(X, regime_labels=regime)

# Inverse transform back to original scale
X_reconstructed = scaler.inverse_transform(X_scaled, regime_labels=regime)

# Should match original (within floating point tolerance)
np.testing.assert_array_almost_equal(X, X_reconstructed)
```

### Inspecting Fitted Statistics

```python
# Get regime statistics
stats = scaler.get_regime_stats()

print(f"Regimes seen: {stats['regimes_seen']}")
print(f"Number of features: {stats['n_features']}")

# Per-regime statistics
for regime_id, regime_stats in stats['regime_stats'].items():
    print(f"Regime {regime_id}:")
    print(f"  Mean: {regime_stats['mean']}")
    print(f"  Std: {regime_stats['std']}")
    print(f"  N samples: {regime_stats['n_samples']}")
    print(f"  Fallback: {regime_stats.get('fallback', False)}")
```

---

## Probability Calibration

### Why Calibration Matters

Raw model probabilities are often poorly calibrated:
- Overconfident: P(predicted=0.9) but actual frequency is only 70%
- Underconfident: P(predicted=0.6) but actual frequency is 90%

Calibration ensures **reliability**: P(actual=1 | predicted=p) ≈ p

### Calibrate Probabilities

```python
from ml_intraday_v3.features.calibration import calibrate_probabilities

# Get uncalibrated probabilities from model
y_prob_uncalibrated = model.predict_proba(X_test)[:, 1]

# Calibrate using isotonic regression
y_prob_calibrated, calibrator = calibrate_probabilities(
    y_prob=y_prob_uncalibrated,
    y_true=y_test,
    method="isotonic",  # or "platt"
    return_calibrator=True
)

# Apply calibrator to new data
y_prob_new = model.predict_proba(X_new)[:, 1]
y_prob_new_calibrated = calibrator.predict(y_prob_new)
```

### Evaluate Calibration

```python
from ml_intraday_v3.features.calibration import evaluate_calibration

# Evaluate calibration quality
metrics = evaluate_calibration(
    y_prob=y_prob_calibrated,
    y_true=y_test,
    n_bins=10
)

print(f"Brier Score: {metrics['brier_score']:.4f}")  # Lower is better
print(f"Expected Calibration Error (ECE): {metrics['ece']:.4f}")  # Lower is better
print(f"Maximum Calibration Error (MCE): {metrics['mce']:.4f}")  # Lower is better
```

**Metrics:**
- **Brier Score**: Mean squared error between predicted probabilities and outcomes (lower is better, 0 is perfect)
- **ECE**: Expected Calibration Error - average calibration error across bins (lower is better)
- **MCE**: Maximum Calibration Error - worst calibration error in any bin (lower is better)

### Visualize Calibration

```python
from ml_intraday_v3.features.calibration import plot_calibration_curve
import matplotlib.pyplot as plt

# Plot before/after calibration
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plot_calibration_curve(
    y_prob=y_prob_uncalibrated,
    y_true=y_test,
    ax=axes[0],
    label="Before Calibration"
)

plot_calibration_curve(
    y_prob=y_prob_calibrated,
    y_true=y_test,
    ax=axes[1],
    label="After Calibration"
)

plt.tight_layout()
plt.show()
```

### Compare Calibration

```python
from ml_intraday_v3.features.calibration import compare_calibration

comparison = compare_calibration(
    y_prob_before=y_prob_uncalibrated,
    y_prob_after=y_prob_calibrated,
    y_true=y_test
)

print("Before calibration:")
print(f"  ECE: {comparison['before']['ece']:.4f}")
print(f"  Brier: {comparison['before']['brier_score']:.4f}")

print("\nAfter calibration:")
print(f"  ECE: {comparison['after']['ece']:.4f}")
print(f"  Brier: {comparison['after']['brier_score']:.4f}")

print("\nImprovement:")
print(f"  ECE: {comparison['improvement']['ece']:.4f}")
print(f"  Brier: {comparison['improvement']['brier_score']:.4f}")
```

---

## Complete Workflow Example

### End-to-End Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml_intraday_v3.features.regime_detector import detect_combined_regime
from ml_intraday_v3.features.regime_scaler import RegimeAwareScaler
from ml_intraday_v3.features.calibration import (
    calibrate_probabilities,
    plot_calibration_curve
)

# 1. Load data
prices = pd.read_parquet('data/bars.parquet')['close']
features = pd.read_parquet('data/features.parquet')
labels = pd.read_parquet('data/labels.parquet')['label']

# 2. Detect regimes
vol_regime, trend_regime, combined_regime = detect_combined_regime(
    prices=prices,
    vol_window=20,
    trend_window=50,
    n_vol_regimes=3,
    n_trend_regimes=3
)

# 3. Train/test split (time-series: no shuffle!)
train_size = int(0.8 * len(features))

X_train = features[:train_size]
X_test = features[train_size:]
y_train = labels[:train_size]
y_test = labels[train_size:]
regime_train = combined_regime[:train_size]
regime_test = combined_regime[train_size:]

# 4. Fit regime-aware scaler on train only
scaler = RegimeAwareScaler()
scaler.fit(X_train, regime_labels=regime_train)

# 5. Transform both train and test
X_train_scaled = scaler.transform(X_train, regime_labels=regime_train)
X_test_scaled = scaler.transform(X_test, regime_labels=regime_test)

# 6. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Get uncalibrated probabilities
y_prob_uncalibrated = model.predict_proba(X_test_scaled)[:, 1]

# 8. Calibrate probabilities
y_prob_calibrated, calibrator = calibrate_probabilities(
    y_prob=y_prob_uncalibrated,
    y_true=y_test,
    method="isotonic",
    return_calibrator=True
)

# 9. Visualize calibration
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_calibration_curve(y_prob_uncalibrated, y_test, ax=axes[0], label="Uncalibrated")
plot_calibration_curve(y_prob_calibrated, y_test, ax=axes[1], label="Calibrated")
plt.savefig('calibration_comparison.png')

# 10. Make predictions on new data (production)
X_new = pd.read_parquet('data/new_features.parquet')
regime_new = detect_combined_regime(new_prices, ...)[-1]  # Combined regime

# Scale using fitted scaler
X_new_scaled = scaler.transform(X_new, regime_labels=regime_new)

# Predict probabilities
y_prob_new = model.predict_proba(X_new_scaled)[:, 1]

# Calibrate probabilities
y_prob_new_calibrated = calibrator.predict(y_prob_new)

print(f"Calibrated probabilities: {y_prob_new_calibrated}")
```

---

## Configuration

### In `data.yaml`

```yaml
feature_scaling:
  method: "regime_aware"  # or "standard", "robust"

  regime_detection:
    enabled: true

    volatility:
      window: 20
      n_regimes: 3
      method: "quantile"

    trend:
      window: 50
      n_regimes: 3
      method: "slope"

    use_combined: true
    min_samples_per_regime: 10
    fallback_to_global: true

  calibration:
    enabled: true
    method: "isotonic"
    n_bins: 10
    compute_metrics: true
```

---

## Validation Tests

### Run Tests

```bash
cd ml_intraday_v3
python -m pytest tests/test_regime_scaler.py -v
```

**Tests verify:**
- ✅ Regime detection (volatility, trend, combined)
- ✅ RegimeAwareScaler fit/transform/inverse_transform
- ✅ Handling of unseen regimes
- ✅ Insufficient samples fallback
- ✅ Sklearn compatibility
- ✅ Probability calibration (isotonic, Platt)
- ✅ No distribution shift between train/test
- ✅ End-to-end workflow

---

## Best Practices

### 1. Choose Appropriate Regime Windows

```python
# For intraday 1-minute bars:
vol_window = 20   # ~20 minutes of volatility history
trend_window = 50  # ~50 minutes of trend history

# For daily data:
vol_window = 20   # ~1 month of volatility
trend_window = 50  # ~2.5 months of trend
```

### 2. Validate Regime Balance

```python
from ml_intraday_v3.features.regime_detector import validate_regime_consistency

# Check that regimes are reasonably balanced
is_valid = validate_regime_consistency(
    regime=combined_regime,
    min_samples_per_regime=10
)

if not is_valid:
    print("⚠️  Warning: Regime distribution is imbalanced!")
    print(combined_regime.value_counts())
```

### 3. Always Use Time-Series Splits

```python
# ❌ Bad: Random shuffle breaks temporal ordering
X_train, X_test = train_test_split(X, y, shuffle=True)

# ✅ Good: Preserve temporal ordering
train_size = int(0.8 * len(X))
X_train = X[:train_size]
X_test = X[train_size:]
```

### 4. Fit Scaler on Train Only

```python
# ❌ Bad: Leakage - fitting on all data
scaler.fit(X, regime_labels=regime)  # Includes test data!
X_train_scaled = scaler.transform(X_train, regime_train)

# ✅ Good: Fit on train only
scaler.fit(X_train, regime_labels=regime_train)
X_train_scaled = scaler.transform(X_train, regime_train)
X_test_scaled = scaler.transform(X_test, regime_test)
```

### 5. Monitor Calibration in Production

```python
# Periodically re-evaluate calibration
from ml_intraday_v3.features.calibration import evaluate_calibration

metrics = evaluate_calibration(y_prob_pred, y_true_actual, n_bins=10)

if metrics['ece'] > 0.1:
    print("⚠️  Calibration degraded - consider recalibrating!")
```

---

## Troubleshooting

### "Warning: Regime has insufficient samples"

**Cause:** Some regimes have fewer than `min_samples_per_regime` samples

**Fix:**
1. Reduce `min_samples_per_regime` (but keep >= 5)
2. Reduce number of regimes (use 2 instead of 3)
3. Increase dataset size
4. Use `fallback_to_global=True` (default)

### "Feature distributions differ between train/test"

**Cause:** Regime detection may be inconsistent or regimes are missing

**Fix:**
1. Ensure regimes are detected the same way for train/test
2. Check for unseen regimes in test (scaler will use global stats)
3. Validate regime consistency with `validate_regime_consistency()`

### "Calibration doesn't improve ECE"

**Cause:** Model is already well-calibrated, or insufficient calibration data

**Fix:**
1. Check if model is already calibrated (some models like logistic regression are naturally calibrated)
2. Increase calibration sample size
3. Try different calibration method (isotonic vs Platt)

---

## References

1. **López de Prado, M. (2018).**
   *Advances in Financial Machine Learning.*
   Wiley. Chapter 19: "Microstructural Features."

2. **Niculescu-Mizil, A., & Caruana, R. (2005).**
   "Predicting Good Probabilities with Supervised Learning."
   *ICML 2005.*

3. **Implementation:**
   - `ml_intraday_v3/features/regime_detector.py`
   - `ml_intraday_v3/features/regime_scaler.py`
   - `ml_intraday_v3/features/calibration.py`
   - `ml_intraday_v3/configs/data.yaml`
   - `ml_intraday_v3/tests/test_regime_scaler.py`

---

*Regime-aware feature scaling implementation complete and validated.*
