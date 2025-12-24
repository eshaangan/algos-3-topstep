# Rare Events Corrections for Logistic Regression

Implementation of rare events corrections following **King & Zeng (2001)**: "Logistic Regression in Rare Events Data".

## Overview

When training logistic regression on rare events data (highly imbalanced classes), standard maximum likelihood estimation produces biased probability estimates. This is particularly problematic for trading applications where profitable trades may be rare (5-40% of total).

This module implements corrections to:
1. **Adjust predicted probabilities** to match true population priors
2. **Weight training samples** to account for class imbalance
3. **Correct the intercept** when sample differs from population

## Key Components

### 1. Probability Correction

```python
from ml_intraday_v3.training.rare_events import correct_rare_events_probabilities

# After getting predictions from standard logistic regression
prob_uncorrected = lr.predict_proba(X_test)[:, 1]

# Apply King & Zeng correction
prob_corrected = correct_rare_events_probabilities(
    prob_uncorrected,
    y_test,
    tau=0.05  # True prior probability (5% profitable trades)
)
```

**Correction formula:**
```
P_corrected = (P * τ) / (P * τ + (1-P) * (1-τ))
```

where:
- `P` = uncorrected predicted probability
- `τ` (tau) = true prior probability of positive class

### 2. Sample Weighting

```python
from ml_intraday_v3.training.rare_events import compute_rare_event_weights

# Compute weights for rare event correction
weights = compute_rare_event_weights(
    y_train,
    method="king_zeng",  # or "inverse_freq", "balanced"
    tau=0.05
)

# Use in training
lr.fit(X_train, y_train, sample_weight=weights)
```

**King & Zeng weights:**
- w = 1.0 for rare events (positive class)
- w = τ/(1-τ) for common events (negative class)

### 3. RelogitClassifier

Drop-in replacement for `sklearn.linear_model.LogisticRegression` with automatic rare events corrections:

```python
from ml_intraday_v3.training.rare_events import RelogitClassifier

# Use just like LogisticRegression
clf = RelogitClassifier(
    tau=0.05,  # Population prior (or None to estimate)
    use_sample_weights=True,  # Apply sample weighting
    weight_method="king_zeng",
    correction_method="king_zeng",
    C=1.0,
    random_state=42
)

clf.fit(X_train, y_train)
prob = clf.predict_proba(X_test)  # Automatically corrected probabilities
```

### 4. Intercept Correction

For when training sample has different class balance than population:

```python
from ml_intraday_v3.training.rare_events import apply_prior_correction_to_intercept

# Scenario: trained on balanced data (50%), but population is 5% positive
intercept_corrected = apply_prior_correction_to_intercept(
    intercept=lr.intercept_[0],
    tau_population=0.05,  # True population rate
    y_sample_mean=0.50    # Sample rate
)

# Update model
lr.intercept_ = np.array([intercept_corrected])
```

## When to Use

### Use rare events corrections when:

1. **Class imbalance > 90:10** (less than 10% positive class)
2. **Training on case-control data** (artificially balanced sample)
3. **Probability calibration matters** (not just ranking)
4. **True population prior is known** or can be estimated

### Don't use when:

1. **Classes are balanced** (40:60 to 60:40)
2. **Only care about ranking** (AUC), not probabilities
3. **Population prior is unknown** and can't be estimated

## Trading Application

For trading with rare profitable trades:

```python
# Estimate true prior from historical data
tau = trades_df['profitable'].mean()  # e.g., 0.35 (35% win rate)

# Option 1: Use RelogitClassifier
clf = RelogitClassifier(tau=tau, random_state=42)
clf.fit(X_train, y_train)
prob = clf.predict_proba(X_test)[:, 1]

# Option 2: Standard LR + post-hoc correction
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
prob_raw = lr.predict_proba(X_test)[:, 1]
prob_corrected = correct_rare_events_probabilities(prob_raw, y_test, tau=tau)
```

## Configuration

Add to `ml_intraday_v3/configs/training.yaml`:

```yaml
rare_events:
  enabled: true

  # Correction method
  correction_method: "king_zeng"  # or "relogit", "simple"

  # Prior probability estimation
  estimate_prior: true
  prior_tau: null  # null = auto-estimate, or specify (e.g., 0.05)
  prior_method: "train"  # "train", "val", or "pooled"

  # Sample weighting
  use_sample_weights: true
  weight_method: "king_zeng"  # or "inverse_freq", "balanced"

  # Intercept correction (if using case-control data)
  apply_intercept_correction: false
  population_tau: null  # Required if apply_intercept_correction=true
```

## Validation

The corrections improve probability calibration, which can be measured by:

1. **Brier Score** - Lower is better
2. **Calibration Curve** - Should be closer to diagonal
3. **Mean Probability** - Should match true rate

Example validation:

```python
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Compare uncorrected vs corrected
brier_uncorr = brier_score_loss(y_test, prob_uncorrected)
brier_corr = brier_score_loss(y_test, prob_corrected)

print(f"Brier score improvement: {brier_uncorr - brier_corr:.4f}")

# Plot calibration curves
fraction_uncorr, mean_pred_uncorr = calibration_curve(
    y_test, prob_uncorrected, n_bins=10
)
fraction_corr, mean_pred_corr = calibration_curve(
    y_test, prob_corrected, n_bins=10
)

plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
plt.plot(mean_pred_uncorr, fraction_uncorr, label='Uncorrected')
plt.plot(mean_pred_corr, fraction_corr, label='Corrected')
plt.legend()
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.show()
```

## Demo

Run the demonstration script:

```bash
cd ml_intraday_v3/training
python rare_events_demo.py
```

This generates:
- Comparison of corrected vs uncorrected probabilities
- Calibration curves
- Performance metrics (Brier score, ROC AUC)
- Sample weighting examples
- RelogitClassifier vs standard LR comparison

## API Reference

### Functions

**`correct_rare_events_probabilities(y_prob, y_true, tau=None, method="king_zeng")`**
- Correct predicted probabilities for rare events bias
- Returns: Corrected probabilities

**`compute_rare_event_weights(y_true, method="king_zeng", tau=None)`**
- Compute sample weights for rare events logistic regression
- Returns: Sample weights array

**`apply_prior_correction_to_intercept(intercept, tau_population, y_sample_mean)`**
- Apply prior correction to logistic regression intercept
- Returns: Corrected intercept value

**`estimate_population_prior(y_train, y_val=None, method="train")`**
- Estimate population prior probability
- Returns: Estimated tau

### Classes

**`RelogitClassifier`**

sklearn-compatible classifier with automatic rare events corrections.

**Parameters:**
- `tau` (float, optional): True prior probability
- `use_sample_weights` (bool): Whether to use sample weighting
- `weight_method` (str): Weighting method
- `correction_method` (str): Probability correction method
- `C`, `penalty`, `solver`, `max_iter`, `random_state`: Same as LogisticRegression

**Methods:**
- `fit(X, y, sample_weight=None)`: Fit the model
- `predict(X)`: Predict class labels
- `predict_proba(X)`: Predict probabilities (corrected)
- `decision_function(X)`: Get decision function values

## References

1. **King, G., & Zeng, L. (2001).** "Logistic regression in rare events data."
   *Political analysis*, 9(2), 137-163.

2. **Owen, A. B. (2007).** "Infinitely imbalanced logistic regression."
   *Journal of Machine Learning Research*, 8(4).

3. **Firth, D. (1993).** "Bias reduction of maximum likelihood estimates."
   *Biometrika*, 80(1), 27-38.

## Integration with Pipeline

The rare events corrections integrate seamlessly with the ML Pipeline V3:

1. **Training:** Use RelogitClassifier instead of LogisticRegression
2. **Prediction:** Probabilities are automatically corrected
3. **Backtest:** Use corrected probabilities for position sizing
4. **Config:** Enable via `training.yaml` configuration

Example integration:

```python
# In training script
from ml_intraday_v3.training.rare_events import RelogitClassifier

if config['rare_events']['enabled']:
    clf = RelogitClassifier(
        tau=config['rare_events'].get('prior_tau'),
        use_sample_weights=config['rare_events']['use_sample_weights'],
        weight_method=config['rare_events']['weight_method'],
        random_state=config['seed']
    )
else:
    clf = LogisticRegression(random_state=config['seed'])

clf.fit(X_train, y_train)
```

## File Structure

```
ml_intraday_v3/training/
├── rare_events.py              # Main implementation
├── rare_events_demo.py         # Demonstration script
└── README_RARE_EVENTS.md       # This file

ml_intraday_v3/tests/
└── test_rare_events.py         # Comprehensive tests

ml_intraday_v3/configs/
└── training.yaml               # Add rare_events section
```

## Performance Notes

- Corrections are O(N) - very fast
- RelogitClassifier has same training time as LogisticRegression
- Sample weighting may slightly increase training time
- Probability correction is instantaneous (simple formula)

## FAQ

**Q: When should I use tau estimation vs providing it?**
- If data is a random sample from population, use tau=None (auto-estimate)
- If data is case-control or balanced, provide true population tau

**Q: Does this hurt discrimination (AUC)?**
- No! Corrections only affect probability calibration, not ranking
- ROC AUC should be identical or very similar

**Q: Should I use sample weighting?**
- Yes, if classes are imbalanced (<20% positive)
- Helps model learn better decision boundaries

**Q: What if I don't know the true prior?**
- Estimate from validation data if available
- Use domain knowledge (e.g., historical win rate)
- Conservative: use training sample mean (may underestimate)

**Q: Can I use this with other models?**
- Probability correction: Yes, works with any classifier
- Sample weighting: Yes, most sklearn models support it
- RelogitClassifier: Specific to logistic regression
