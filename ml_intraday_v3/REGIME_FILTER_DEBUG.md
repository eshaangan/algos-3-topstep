# Multi-Period Analysis Report with Regime Filter

**Date**: January 25, 2026
**Model**: `BALANCED_V3`
**Feature**: Trend-Regime Filter (SMA-200 Proxy)

## Summary of Results

We applied a Regime Filter that blocks counter-trend trades based on a long-term SMA (approx 50-200 day equivalent).

| Period | Regime | Resulting Bias | Win Rate | P&L | Interpretation |
|--------|--------|----------------|----------|-----|----------------|
| **Q1 2024** | **Bull** | **100% SHORT** (Blocked?) | 3.0% | -$2,586 | ❌ **FAILED**. The filter did NOT block the shorts. Why? |
| **Q3 2024** | **Volatile** | **95% SHORT** | 27.8% | -$2,661 | ❌ **FAILED**. Still heavily shorting. |
| **2025** | **Bull** | **98% SHORT** | 27.3% | -$2,841 | ❌ **FAILED**. Still heavily shorting. |
| **Jan 2026** | **Bear/Chop** | **97% LONG** | 35.9% | -$2,052 | ❌ **FAILED**. Still heavily longing. |

## Critical Issue Diagnosis

The **Regime Filter did not work as intended**. The trade logs show that the model continued to execute trades *against* the major trend (Shorting in 2024/2025, Longing in Jan 2026).

### Why?
1.  **Regime Calculation Error**: In `replay.py`, we calculated `bars['sma_long']`. However, if the `bars` dataframe passed to `replay.py` is *sliced* (e.g., just "Q1 2024"), the rolling mean calculation might be starting from index 0 of that slice, resulting in NaNs or incorrect values for the first 200 days of the slice.
2.  **Logic Check**:
    -   In 2025 (Bull Run), Price > SMA. Regime = 1 (Bull).
    -   Filter Logic: `if regime == 1 and direction == "SHORT": return False`.
    -   Result: Should block shorts.
    -   Actual: 217 SHORT trades executed.
    
    This implies `regime` was likely `0` or `NaN` (Neutral), bypassing the filter.

### Corrective Action
We must ensure the SMA is calculated on the **full historical dataset** before slicing for specific test periods.

1.  **Modify `test_multi_period.py`**: Calculate SMA on the full `bars_hist` dataframe *before* passing slices to `run_backtest`.
2.  **Verify Regime Values**: Log the regime value in the backtest to confirm it's being set correctly.

---

## Next Steps
1.  Fix the SMA calculation in `test_multi_period.py` (pre-calculate `regime` column on full history).
2.  Pass the `regime` column into the backtest buffer.
3.  Rerun the test.
