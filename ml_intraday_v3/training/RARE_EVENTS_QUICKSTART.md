# Rare Events - Quick Start Guide

## 30-Second Start

```python
from ml_intraday_v3.training.rare_events import RelogitClassifier

# Replace LogisticRegression with RelogitClassifier
clf = RelogitClassifier(tau=0.05, random_state=42)  # 5% win rate
clf.fit(X_train, y_train)
prob = clf.predict_proba(X_test)[:, 1]  # Automatically corrected!
```

## When to Use

✅ Use if: Positive class < 20% (rare events)
✅ Use if: Probabilities matter (not just ranking)
✅ Use if: Known or estimable population prior

❌ Skip if: Balanced data (40-60% positive)
❌ Skip if: Only care about AUC
❌ Skip if: Using non-probabilistic models

## Three Ways to Use

### 1. RelogitClassifier (Easiest)

```python
from ml_intraday_v3.training.rare_events import RelogitClassifier

clf = RelogitClassifier(
    tau=0.05,              # 5% historical win rate (or None to auto-estimate)
    use_sample_weights=True,
    random_state=42
)
clf.fit(X_train, y_train)
prob = clf.predict_proba(X_test)[:, 1]
```

### 2. Post-hoc Correction

```python
from sklearn.linear_model import LogisticRegression
from ml_intraday_v3.training.rare_events import correct_rare_events_probabilities

# Train normally
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
prob_raw = lr.predict_proba(X_test)[:, 1]

# Correct probabilities
prob_corrected = correct_rare_events_probabilities(
    prob_raw,
    y_test,
    tau=0.05  # True prior
)
```

### 3. Sample Weighting Only

```python
from sklearn.linear_model import LogisticRegression
from ml_intraday_v3.training.rare_events import compute_rare_event_weights

# Compute weights
weights = compute_rare_event_weights(y_train, method="king_zeng", tau=0.05)

# Train with weights
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train, sample_weight=weights)
```

## Demo

```bash
cd ml_intraday_v3/training
python rare_events_demo.py
```

Output: Calibration plots, performance comparisons, validation metrics

## Configuration

Add to `configs/training.yaml`:

```yaml
model:
  type: "relogit"
  relogit_params:
    tau: null  # Auto-estimate
    use_sample_weights: true
    weight_method: "king_zeng"
```

## Validation

```python
from sklearn.metrics import brier_score_loss

# Compare calibration
brier_before = brier_score_loss(y_test, prob_uncorrected)
brier_after = brier_score_loss(y_test, prob_corrected)

print(f"Calibration improvement: {brier_before - brier_after:.4f}")
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | `None` | True prior (None = auto-estimate) |
| `use_sample_weights` | `True` | Apply sample weighting |
| `weight_method` | `"king_zeng"` | Weighting method |
| `correction_method` | `"king_zeng"` | Probability correction |

## Common Scenarios

**Scenario 1: Very imbalanced (5% positive)**
```python
clf = RelogitClassifier(tau=0.05, random_state=42)
```

**Scenario 2: Unknown prior**
```python
clf = RelogitClassifier(tau=None, random_state=42)  # Estimates from data
```

**Scenario 3: Only probability correction**
```python
clf = RelogitClassifier(tau=0.05, use_sample_weights=False, random_state=42)
```

## Cheat Sheet

```python
# Import
from ml_intraday_v3.training.rare_events import RelogitClassifier

# Create
clf = RelogitClassifier(tau=0.05, random_state=42)

# Train
clf.fit(X_train, y_train)

# Predict
y_prob = clf.predict_proba(X_test)[:, 1]  # Corrected!
y_pred = clf.predict(X_test)

# Validate
from sklearn.metrics import brier_score_loss, roc_auc_score
print(f"Brier: {brier_score_loss(y_test, y_prob):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
```

## Troubleshooting

**Q: Probabilities sum > 1?**
A: Check that you're using `predict_proba()` not raw predictions

**Q: AUC dropped?**
A: AUC should be identical. Check implementation.

**Q: Don't know tau?**
A: Set `tau=None` to auto-estimate from training data

**Q: Warnings about deprecated 'penalty'?**
A: Safe to ignore - sklearn version compatibility

## Full Documentation

- **README:** `ml_intraday_v3/training/README_RARE_EVENTS.md`
- **Demo:** `ml_intraday_v3/training/rare_events_demo.py`
- **Tests:** `ml_intraday_v3/tests/test_rare_events.py`
- **Config:** `ml_intraday_v3/configs/training_rare_events_example.yaml`
