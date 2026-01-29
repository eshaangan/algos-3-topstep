# Investigation: Why 'side' Feature is Missing

## 🔍 Root Cause Found

The **'side' feature comes from the labeling pipeline**, NOT the feature generation pipeline.

### Where 'side' is Generated

**File**: `ml_intraday_v3/labels/events.py`  
**Function**: `generate_events()`  
**Line**: ~256

```python
# Trend scanning algorithm computes forward-looking trend slopes
betas = best_beta[keep]
sides_all = np.where(betas >= 0.0, 1, -1).astype(int)

# ... later added to events DataFrame
"side": side_vals,  # -1 = SHORT, +1 = LONG
```

### How Trend Scanning Works

The `trend_scanning` event policy in `labeling.yaml`:

1. **Computes t-statistics** for linear trend slopes over multiple horizons (12, 24 bars)
2. **Selects best horizon** with highest absolute t-stat for each potential event
3. **Determines direction**:
   - If beta (slope) >= 0 → `side = +1` (LONG)
   - If beta (slope) < 0 → `side = -1` (SHORT)
4. **Filters events** where abs(t-stat) >= threshold (1.5 in current config)

This creates events with a 'side' field that indicates which direction the forward price trend is moving.

### Why Simple Retraining Failed

**What I did** (`retrain_with_existing_data.py`):
```python
# Simple next-bar direction labels
labels = (close_prices.shift(-1) > close_prices).astype(int)
# Result: 0 or 1, no 'side' feature
```

**What should have been done**:
```python
# Use full labeling pipeline
from ml_intraday_v3.labels.events import generate_events
from ml_intraday_v3.labels.triple_barrier import apply_triple_barrier

# Generate events with trend_scanning (creates 'side' feature)
events = generate_events(bars, bar_size, labeling_config, execution_spec)

# Apply triple-barrier labeling (stop/vertical/target)
labels = apply_triple_barrier(bars, events, labeling_config)

# Result: labels with 'side' feature included
```

## 📊 Config Validation

### ✅ labeling.yaml is Correct

```yaml
primary_labeling:
  event_policy: "trend_scanning"  # ✅ CORRECT
  
  trend_scanning:
    tstat_threshold: 1.5           # ✅ Reasonable
    use_cusum_prefilter: true      # ✅ Reduces noise
    cusum_threshold_atr_mult: 0.8  # ✅ Lowered for more signals
```

This config **IS** set up to generate 'side' feature via trend_scanning.

### ❌ My Simple Script Bypassed This

The quick retraining script I created:
- ✅ Loaded data correctly
- ✅ Generated features correctly
- ❌ **Created simple labels, bypassing labeling pipeline**
- ❌ **No 'side' feature generated**
- ❌ **No triple-barrier labeling**

## 🔧 Proper Solution: Full V3 Pipeline

### Required Steps

1. **Load bars** (already have in `data/processed/`)

2. **Generate features** using `features/build.py`
   - Creates 34 features
   - But NOT 'side' - that comes from labeling

3. **Generate events** using `labels/events.py`
   - Uses trend_scanning algorithm
   - Creates 'side' field for each event
   - Filters based on t-stat threshold

4. **Apply triple-barrier labeling** using `labels/triple_barrier.py`
   - For each event, apply stop/target/vertical barriers
   - Label: 0=stop hit, 1=vertical (time), 2=target hit
   - Preserves 'side' field from events

5. **Merge features + labels**
   - Features: 34 columns from feature generation
   - Labels: includes 'side', outcome, and metadata
   - Final dataset: 35 columns (34 features + 'side')

6. **Train model** using `training/train.py`
   - Model sees 'side' as input feature
   - Learns to predict outcome conditional on direction
   - At inference: evaluate BOTH directions, pick best EV

### Why This Matters

**Without 'side' feature**:
- Model learns: P(up | features)
- Prediction: Always same direction based on features
- Result: 100% LONG bias (if bullish features dominate training)

**With 'side' feature**:
- Model learns: P(stop|features, side=-1), P(vertical|features, side=-1), P(target|features, side=-1)
- Model learns: P(stop|features, side=+1), P(vertical|features, side=+1), P(target|features, side=+1)
- Prediction: Evaluate EV_long and EV_short, pick best
- Result: Balanced LONG/SHORT based on which has higher EV

## 🎯 Next Steps

### Option 1: Use Existing Full Pipeline

The V3 pipeline already exists:

```bash
# Full training with proper labeling
python ml_intraday_v3/training/train.py \
    --run-dir runs/full_retrain_oct2024_nov2025 \
    --bar-size 5m \
    --config configs/training.yaml
```

This handles:
- ✅ Event generation with trend_scanning
- ✅ Triple-barrier labeling
- ✅ Sample uniqueness weighting
- ✅ Purged k-fold CV
- ✅ Isotonic calibration
- ✅ Meta-model training

### Option 2: Simplified Full Pipeline Script

Create a streamlined version that:
1. Uses existing bars
2. Runs full labeling pipeline
3. Trains with proper 'side' feature
4. Validates on Dec 2025

## 📋 Summary

| Aspect | Simple Script | Full Pipeline |
|--------|--------------|---------------|
| Data Loading | ✅ Correct | ✅ Correct |
| Feature Generation | ✅ Correct (34 features) | ✅ Correct (34 features) |
| Label Generation | ❌ Simple next-bar | ✅ Triple-barrier |
| Event Generation | ❌ Skipped | ✅ Trend scanning |
| 'side' Feature | ❌ Missing | ✅ Generated |
| Sample Weighting | ❌ None | ✅ Uniqueness weights |
| Cross-Validation | ❌ Simple split | ✅ Purged k-fold |
| Calibration | ❌ None | ✅ Isotonic |
| Meta-Model | ❌ None | ✅ Signal filter |
| **Result** | **50.8% accuracy** | **Should be better** |

## 🚀 Recommendation

**Use the full V3 pipeline** - it's already implemented and tested. My simplified script was a useful diagnostic but skipped critical components that enable bidirectional trading.

The configs are correct. The code is correct. We just need to use it properly.
