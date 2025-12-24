# Cost Curves - Now Loading from Disk ✅

## What Changed

All three cost curve sections (4.6.1, 4.7.1, 5.5) now **load predictions from disk** instead of expecting in-memory variables.

This matches how your pipeline works - sections run CLI commands that save to disk rather than creating Python variables.

## Required Files

### Section 4.6.1: Model Performance

**Looks for predictions in this order:**

1. `RUN_DIR/5m/predictions/test_predictions.parquet`
   - Columns needed: `label` and `prob` (or `y_true` and `y_pred`)

2. `RUN_DIR/5m/cv_results.parquet` (fallback)
   - Columns needed: `y_true` and `y_pred_proba`

**Expected columns:**
- Labels: `label` or `y_true` (binary: 0/1)
- Probabilities: `prob`, `y_pred`, or `y_pred_proba` (float: 0.0-1.0)

### Section 4.7.1: Backtest Performance

**Looks for trades in this order:**

1. `RUN_DIR/5m/backtest/trades.parquet`
2. `RUN_DIR/5m/trades.parquet` (fallback)
3. `RUN_DIR/backtest_trades.parquet` (fallback)

**Expected columns:**
- `pnl`: Trade P&L in dollars (used to determine profitable/unprofitable)
- `prob`, `signal_prob`, or `y_pred_proba`: Model probability predictions

### Section 5.5: Model Comparison

**Looks for:**

1. Primary model: Same as 4.6.1 (predictions/ or cv_results)
2. Meta model (optional): `RUN_DIR/5m/predictions/meta_predictions.parquet`
3. Creates baseline (class prior) if only one model found

## How to Use Now

### Step 1: Run Training (Section 4.6)

```python
# This runs and saves to disk
# Predictions should be saved to:
# RUN_DIR/5m/predictions/test_predictions.parquet
```

### Step 2: Run Cost Curve Analysis (Section 4.6.1)

Now it will:
- ✅ Load predictions from disk
- ✅ Compute cost curves
- ✅ Show bootstrap confidence intervals
- ✅ Find optimal risk/reward ratio

**If it fails**, you'll see exactly which file is missing and where it looked.

### Step 3: Run Backtest (Section 4.7)

```python
# This runs and saves to disk
# Trades should be saved to:
# RUN_DIR/5m/backtest/trades.parquet
```

### Step 4: Run Backtest Cost Curves (Section 4.7.1)

Now it will:
- ✅ Load backtest trades from disk
- ✅ Extract profitable/unprofitable labels from `pnl`
- ✅ Use model probabilities if available
- ✅ Show optimal RR for actual trading

### Step 5: Compare Models (Section 5.5)

Now it will:
- ✅ Load all available model predictions
- ✅ Create baseline for comparison
- ✅ Rank by AUCC
- ✅ Show dominance analysis

## Error Messages

If a section can't find the required files, you'll see:

```
⚠️  Could not find predictions at:
  /path/to/RUN_DIR/5m/predictions/test_predictions.parquet
  /path/to/RUN_DIR/5m/cv_results.parquet

Make sure section 4.6 (Train Models) has completed successfully.
```

This tells you:
- Exactly what file it needs
- Where it looked
- What to fix

## Ensuring Predictions Are Saved

### For Training (Section 4.6.1 to work)

Make sure your training code saves predictions:

```python
# In your training script
predictions_df = pd.DataFrame({
    'label': y_test,
    'prob': y_pred_proba
})
predictions_path = RUN_DIR / bar_size / "predictions" / "test_predictions.parquet"
predictions_path.parent.mkdir(parents=True, exist_ok=True)
predictions_df.to_parquet(predictions_path)
```

### For Backtest (Section 4.7.1 to work)

Make sure your backtest saves probabilities:

```python
# In your backtest script
trades_df = pd.DataFrame({
    'pnl': trade_pnls,
    'prob': signal_probs,  # ← Add this!
    # ... other columns
})
trades_path = RUN_DIR / bar_size / "backtest" / "trades.parquet"
trades_df.to_parquet(trades_path)
```

## Quick Test

To verify the sections will work, run:

```bash
cd "/Users/eshaanganguly/Documents/projects/algos 3 topstep"
python3 << 'EOF'
from pathlib import Path

RUN_DIR = Path("runs/latest")  # or your actual run dir
bar_size = "5m"

# Check for predictions
pred_path = RUN_DIR / bar_size / "predictions" / "test_predictions.parquet"
cv_path = RUN_DIR / bar_size / "cv_results.parquet"

print("Checking for required files:")
print(f"  Predictions: {pred_path.exists()} - {pred_path}")
print(f"  CV results:  {cv_path.exists()} - {cv_path}")

if pred_path.exists() or cv_path.exists():
    print("\n✅ Section 4.6.1 will work!")
else:
    print("\n⚠️  Need to run training first")

# Check for backtest
bt_path = RUN_DIR / bar_size / "backtest" / "trades.parquet"
print(f"  Backtest:    {bt_path.exists()} - {bt_path}")

if bt_path.exists():
    print("\n✅ Section 4.7.1 will work!")
else:
    print("\n⚠️  Need to run backtest first")
EOF
```

## Summary

✅ **Fixed**: All sections now load from disk
✅ **Clear errors**: Shows exact paths when files are missing
✅ **Fallbacks**: Tries multiple locations
✅ **Helpful**: Tells you what columns are needed

The cost curves will now work with your CLI-based pipeline!

## Next Steps

1. **Reload the notebook** (Kernel → Restart if already open)
2. **Run section 4.6** (Train Models)
3. **Run section 4.6.1** - should now find predictions and work!
4. **Run section 4.7** (Backtest)
5. **Run section 4.7.1** - should now find trades and work!
6. **Run section 5.5** - model comparison

If any section fails, it will show you exactly which file is missing and where it looked for it.
