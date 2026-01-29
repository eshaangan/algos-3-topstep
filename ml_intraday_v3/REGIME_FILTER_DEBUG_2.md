# Multi-Period Analysis Report with Regime Filter (Attempt 2)

**Date**: January 25, 2026
**Model**: `BALANCED_V3`
**Feature**: Trend-Regime Filter (SMA-50 Days ~ 13,800 bars)

## Results

| Period | Trend (SMA50) | Filter Action | Resulting Bias | P&L | Status |
|--------|---------------|---------------|----------------|-----|--------|
| **2024 Full** | Bullish | Block Shorts | **93% SHORT** | -$1,961 | ❌ **FAILED**. Filter didn't block shorts. |
| **2025 Full** | Bullish | Block Shorts | **94% SHORT** | -$3,806 | ❌ **FAILED**. Filter didn't block shorts. |

## Diagnosis: Why is the filter failing?

The Regime Filter is clearly NOT engaging.
`logger.info("Enforcing Regime Filter for Replay: ENABLED")` was printed.

Possible Reasons:
1.  **Regime Value is 0 (Neutral)**: If `sma_long` calculation results in NaNs or fails, `regime` defaults to 0. `LiveExecutionEngine` logic: `if regime is not None:` (0 is not None). `if regime == 1... elif regime == -1...`. **If regime is 0, it does nothing.**
2.  **SMA Calculation**: 50 days (13,800 bars) is a lot. Maybe the rolling mean is all NaNs?
3.  **Data Alignment**: We pre-calculated `regime` in `bars_hist`. Then we sliced it.
    - `p['data']` is `bars_hist`.
    - `bars = p['data']`.
    - `bars['sma_long'] = ...` (calculated on full history).
    - Then sliced. This should work.

Let's check the logs for "Regime distribution".
I didn't see the log output in the tail. Let me check the full log file content or grep it.

Hypothesis: `regime` is 0 because `sma_long` is NaN or the comparison failed.

Let's debug by printing regime distribution inside the test script more explicitly.
Also, 13,800 bars is 50 days. The dataset starts in 2020? Or 2024?
The log said "Loading data...". `bars_hist` from `mes_bars_databento_rth.h5`.
I need to check the start date of `bars_hist`.

If `bars_hist` starts in Jan 2024, then the first 50 days (until mid-Feb) will be NaN.
Then 2024 test period starts in Jan/Feb.
So for Q1 2024, regime is likely 0 (Neutral).

Solution:
1.  Check start date of `bars_hist`.
2.  Reduce SMA period to something much faster for testing (e.g., 5 Days = 1,380 bars) to ensure we have a regime. 50-day SMA is "major trend", but 5-day is "weekly trend".
3.  Or verify if we have 2023 data.

Let's check the start date of `bars_hist`.
