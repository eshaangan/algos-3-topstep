# Regime Filter Debug Report

**Date**: January 25, 2026

## Issue
The Regime Filter is enabled and `regime` values are present (e.g., `{1: 14267, -1: 3755}` for Q1 2024), yet **counter-trend trades are still executing**.

## Investigation
In `LiveExecutionEngine.execute_signal`:
```python
        # Check Regime Filter
        if self.regime_filter_enabled:
            regime = current_regime
            if regime is None and 'regime' in bars_df.columns and not bars_df.empty:
                regime = bars_df.iloc[-1]['regime']
            
            # Apply filter logic if regime is known
            if regime is not None:
                if regime == 1 and direction == "SHORT":
                    logger.info(f"Regime Filter: SHORT signal blocked...")
                    return False, "regime_filter_bull"
```

In `replay.py`:
```python
        # Get current regime
        current_regime = int(row['regime']) if 'regime' in row else 0

        # ... (prediction logic)

        success, exec_reason = execution_engine.execute_signal(
                # ...
                current_regime=current_regime,
            )
```

**Hypothesis**: The `regime` is somehow passing as `0` or `None` incorrectly, OR the logic is flawed.

Wait. In `test_multi_period.py`:
```python
    # Pre-calculate Regime (SMA 200 Days approx = 55,200 bars)
    # Using 50 Days (approx 13,800 bars) as a responsive proxy
    sma_period = 13800
    # ...
    bars_hist['sma_long'] = bars_hist['close'].rolling(window=sma_period).mean()
```

If we are running "Q1 2024 (Bearish)" starting `2024-02-01`.
And `bars_hist` starts `2019-05-06`.
Then `2024-02-01` is definitely after 13,800 bars.
The log said `Regime counts in slice (pre-calc): {1: 14267, -1: 3755}`.
So `regime` is definitely `1` or `-1` for most bars.

**Why are SHORTS executing when Regime is 1 (Bull)?**
Maybe the regime for those specific bars was `-1`?
In Q1 2024 (Jan-Mar), the market was RALLYING (Bull). So regime should be 1.
If regime is 1, Shorts should be blocked.
But we see 66 SHORT trades in Q1 2024.
If regime was 1, they should be blocked.
If they executed, regime must have been `-1` or `0`.
If regime was `-1` (Bear), then Shorts are allowed. But Q1 2024 was a rally. Why would SMA50 say Bear?
SMA50 is a lagging indicator. If market was crashing in late 2023, SMA50 might still be above price (Bear) even as price rallies.
BUT, price > SMA = Bull.
If price rallies from below SMA, it crosses SMA.
Once Price > SMA, regime = 1.

Let's check the specific trade logs for `regime` values.
I will verify what `regime` value was used for a specific trade.
The `trade_log` doesn't strictly log the regime, but I can add a log line in `execution_engine` to show the regime value when a trade is *accepted*.

**Plan**:
1. Add logging to `execute_signal` to print the `regime` value for every accepted trade.
2. Run a short test on Q1 2024.
