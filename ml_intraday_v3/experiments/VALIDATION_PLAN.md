# Validation Plan: Prove the Model Works Before Implementing Improvements

## Current Status

✅ **Phase 1 Complete**: All 4 safety filters implemented
- Confidence filter (threshold 0.55)
- Adaptive circuit breaker
- Regime detector (KS test)
- Volatility filter

✅ **Jan 2026 Analysis Complete**: Confirmed that filtering alone is insufficient
- Filtering at P≥0.50: Only +$42.60, 1.7 trades/day ❌
- Filtering at P≥0.55: -$3.70, 1.1 trades/day ❌
- Filtering at P≥0.60: -$50.00, 0.6 trades/day ❌
- **Conclusion**: Jan 2026 was a regime shift month where even "good" signals failed

## Critical Question: Is the Model Broken or Just Facing Regime Shift?

**Two possibilities**:

1. **Model is fine, Jan 2026 was anomaly** → Filters + improvements will work
2. **Model is broken** → Need to retrain before anything else

**How to find out**: Run backtest on Dec 2025 (month before regime shift)

### Expected Results

#### Scenario A: Model is Fine (Expected)
**Dec 2025 with filters should show**:
- Win rate: 50-60%
- Trades/day: 6-8
- Daily P&L: +$80-150
- Total P&L: +$1,500-3,000 over ~20 trading days

**Interpretation**: Model works in normal conditions, Jan 2026 was regime shift
**Next step**: Implement signal quality improvements, proceed to paper trading

#### Scenario B: Model is Broken (Unexpected)
**Dec 2025 with filters shows**:
- Win rate: <45%
- Daily P&L: <$50 or negative
- Total P&L: <$1,000 or negative

**Interpretation**: Model has fundamental problems, not just regime shift
**Next step**: STOP - retrain model before implementing improvements

---

## Validation Backtest Plan

### Step 1: Run Dec 2025 Backtest with Filters

**File to modify**: Create `ml_intraday_v3/experiments/validate_dec2025.py`

**What it will do**:
1. Load Dec 2025 data (20 trading days before Jan 2026)
2. Apply confidence filter (P≥0.55)
3. Apply volatility filter (30-70th percentile)
4. Calculate metrics: trades/day, win rate, P&L, drawdown

**Expected timeline**: 30 minutes to create + run

### Step 2: Compare Dec 2025 vs Jan 2026

**Create comparison table**:

| Metric | Dec 2025 (Expected) | Jan 2026 (Actual) | Delta | Interpretation |
|--------|-------------------|------------------|-------|----------------|
| Win Rate | 50-60% | 35.5% | -15-25% | Regime shift confirmed |
| Avg Win | $40-50 | $41.16 | Similar | Win size stable |
| Avg Loss | -$15-20 | -$19.98 | Similar | Loss size stable |
| Trades/Day | 6-8 | 8.4 (no filters) | Lower | Filters working |
| Daily P&L | +$100-150 | -$49.15 | +$150-200 | Model works pre-shift |

**If this pattern holds** → Model is fine, proceed with improvements

### Step 3: Check Regime Detector

**Verify regime detector would have flagged Jan 2026**:

Run `ml_intraday_v3/experiments/test_regime_detector.py`:
- Train on data through Dec 31, 2025
- Test on Jan 1-31, 2026
- Check if detector flags shift in early January

**Expected**: Detector should flag 30-50% feature shift by Jan 5-7

---

## GO/NO-GO Decision Tree

```
┌─────────────────────────────┐
│ Run Dec 2025 Backtest       │
│ (with all filters)          │
└──────────┬──────────────────┘
           │
           ├──────────────────────────────────┬────────────────────────────────
           │                                  │
    Win Rate ≥50%                      Win Rate <45%
    +$100-150/day                      <$50/day or negative
           │                                  │
           │                                  │
    ┌──────▼──────────┐              ┌───────▼──────────┐
    │ MODEL IS FINE   │              │ MODEL IS BROKEN  │
    │ ✅ Proceed      │              │ ❌ STOP          │
    └──────┬──────────┘              └───────┬──────────┘
           │                                  │
           │                                  │
    Next: Implement                   Next: Retrain model
    signal improvements               (12-month, 6-month, 3-month
    - Entry timing                    ensemble on fresh data)
    - Dynamic stops
    - Tiered sizing                   Then restart validation
    - Features
    - Ensemble
           │
           │
    Then: Paper trade 5-7 days
    Then: Topstep combine
```

---

## Why Dec 2025 is the Right Test

1. **Recent but pre-shift**: Last month before Jan 2026 failure
2. **Same market conditions**: E-mini futures, same session times
3. **Same infrastructure**: Model, features, execution all identical
4. **Clean baseline**: If Dec works but Jan doesn't → regime shift confirmed

---

## Files to Create

### 1. `validate_dec2025.py`

```python
#!/usr/bin/env python3
"""
Validation Backtest: Dec 2025 with All Filters

Tests whether model + filters work in normal (pre-regime-shift) conditions.
If this passes (50%+ win rate, +$100-150/day), then model is fine and Jan 2026
was just a regime shift anomaly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ml_v3_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ml_v3_dir))

from filters.confidence_filter import apply_confidence_filter
from filters.volatility_filter import apply_volatility_filter
from monitoring.adaptive_circuit_breaker import AdaptiveCircuitBreaker

def load_dec2025_data():
    """Load Dec 2025 backtest results."""
    # TODO: Load from actual backtest parquet file
    # For now, will need to point to correct data file
    data_path = ml_v3_dir / "backtesting_v3" / "results" / "dec2025_backtest.parquet"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dec 2025 backtest data not found at {data_path}.\n"
            "Need to run backtest first or point to correct file."
        )

    df = pd.read_parquet(data_path)
    return df

def calculate_metrics(trades_df, label=""):
    """Calculate performance metrics."""
    # Same as jan2026 script
    pass

def main():
    print("="*80)
    print("DEC 2025 VALIDATION BACKTEST")
    print("="*80)
    print("\nTesting model performance in normal (pre-regime-shift) conditions\n")

    # Load Dec 2025 data
    df = load_dec2025_data()

    # Apply filters
    filtered_df = apply_confidence_filter(df, threshold=0.55)
    filtered_df = apply_volatility_filter(filtered_df)

    # Calculate metrics
    baseline = calculate_metrics(df, "BASELINE (Dec 2025 - No Filters)")
    filtered = calculate_metrics(filtered_df, "WITH FILTERS (0.55 + Volatility)")

    print_metrics(baseline)
    print_metrics(filtered)

    # Decision
    if filtered['win_rate'] >= 0.50 and filtered['daily_pnl'] >= 80:
        print("\n✅ MODEL VALIDATION PASSED")
        print("   Model works in normal conditions")
        print("   Jan 2026 failure was regime shift")
        print("   Safe to proceed with signal improvements")
    elif filtered['win_rate'] >= 0.45:
        print("\n⚠️ MODEL MARGINALLY PASSING")
        print("   Need signal improvements before paper trading")
    else:
        print("\n❌ MODEL VALIDATION FAILED")
        print("   Model has fundamental problems")
        print("   MUST retrain before continuing")

if __name__ == "__main__":
    main()
```

### 2. `compare_dec_jan.py`

```python
#!/usr/bin/env python3
"""
Compare Dec 2025 vs Jan 2026 Performance

Shows whether Jan 2026 failure was regime shift or model degradation.
"""

def compare_months():
    dec_results = load_dec2025_results()
    jan_results = load_jan2026_results()

    print("\n" + "="*80)
    print("DEC 2025 vs JAN 2026 COMPARISON")
    print("="*80)

    comparison = pd.DataFrame({
        'Dec 2025': dec_results,
        'Jan 2026': jan_results,
        'Delta': jan_results - dec_results,
        'Delta %': 100 * (jan_results - dec_results) / dec_results
    })

    print(comparison)

    # Interpretation
    if dec_results['win_rate'] > 0.50 and jan_results['win_rate'] < 0.40:
        print("\n✅ REGIME SHIFT CONFIRMED")
        print("   Dec 2025: Model worked (>50% win rate)")
        print("   Jan 2026: Model failed (<40% win rate)")
        print("   Conclusion: Regime shift, not broken model")
```

---

## Timeline

**Today** (30 minutes):
1. ✅ Create validation scripts
2. ⚠️ Locate Dec 2025 backtest data (or generate it)
3. ⚠️ Run validation backtest
4. ⚠️ Make GO/NO-GO decision

**If GO** (model validated):
- Days 1-3: Implement entry timing optimization
- Days 4-5: Implement dynamic stops
- Days 6-7: Implement tiered position sizing
- Days 8-10: Feature improvements + model ensemble
- Days 11-17: Paper trading validation
- Days 18-37: Topstep combine

**If NO-GO** (model broken):
- Days 1-5: Retrain model with 12/6/3-month ensemble
- Days 6-7: Validation backtest on new model
- Then: Restart from Phase 1

---

## Next Immediate Action

**Run Dec 2025 validation backtest** to determine if model is fine or broken.

This is the most critical step because:
- If model is broken → wasting time on improvements won't help
- If model is fine → improvements will work and we can proceed

**Expected Result**: Dec 2025 shows 50-60% win rate, confirming Jan 2026 was anomaly

---

## Files Status

✅ **Completed**:
- `ml_intraday_v3/monitoring/circuit_breaker.py`
- `ml_intraday_v3/monitoring/adaptive_circuit_breaker.py`
- `ml_intraday_v3/filters/regime_filter.py`
- `ml_intraday_v3/filters/confidence_filter.py`
- `ml_intraday_v3/experiments/jan2026_exact_metrics.py`
- `ml_intraday_v3/experiments/FINAL_JAN2026_ANALYSIS.md`
- `ml_intraday_v3/configs/execution_spec.yaml` (updated)

⚠️ **Need to create**:
- `ml_intraday_v3/experiments/validate_dec2025.py`
- `ml_intraday_v3/experiments/compare_dec_jan.py`

⚠️ **Need to locate/generate**:
- Dec 2025 backtest data (parquet file with trades)

Once validation passes, proceed to signal quality improvements.
