# Regime Filter Debug Report 3

**Date**: January 25, 2026

## Findings
1.  **Regime Pre-calculation Works**: `{1: 90572, -1: 26965, 0: 13799}` globally.
2.  **Q1 2024 Slice**: `{1: 3163}`. This means **every single bar in Q1 2024 has Regime = 1 (Bull)**.
3.  **Filter Logic**:
    ```python
    if regime == 1 and direction == "SHORT":
        # Block SHORT
    ```
4.  **Observation**: I see NO "Regime Filter:" logs in the output.
    - `grep "Regime Filter:"` returned nothing.
    - `grep "Regime"` only showed setup logs.

## Conclusion
The `execute_signal` method logic for logging is **not being reached** or `regime_filter_enabled` is somehow False inside the instance.

In `replay.py`:
```python
    # Force Enable Regime Filter for this test
    if 'regime_filter' not in live_cfg:
        live_cfg['regime_filter'] = {}
    live_cfg['regime_filter']['enabled'] = True
    logger.info("Enforcing Regime Filter for Replay: ENABLED")
    
    # ...
    execution_engine = LiveExecutionEngine(
        # ...
        config=live_cfg,  # Pass live trading config
    )
```

In `LiveExecutionEngine.__init__`:
```python
        # Regime Filter configuration (NEW)
        regime_filter_cfg = self.config.get("regime_filter", {})
        self.regime_filter_enabled = regime_filter_cfg.get("enabled", False)
```

This looks correct. `live_cfg` is modified before passing to `LiveExecutionEngine`.

**Why no logs?**
If `execute_signal` returns early, we won't see logs?
```python
        # Check risk gates
        can_trade, reason = self.risk_manager.can_trade(timestamp)
        if not can_trade:
            logger.warning(f"Trade rejected by risk manager: {reason}")
            return False, f"risk_{reason}"
            
        # Check Regime Filter
        if self.regime_filter_enabled:
             # ... logging ...
```
If risk manager rejects (e.g. `halted` or `daily_loss_limit`), we return early.
Most logs in the tail were `Trade rejected by risk manager: halted`.
If the account is halted (daily loss reached), the engine rejects *before* checking regime.

This explains why we see no Regime Filter logs -- the account blows up (daily loss) before the filter can block the trades?
**Wait.**
If the filter works, it should prevent the losing trades that cause the blowout.
So the *first* trades of the day should be filtered.
If the first trades are counter-trend (Shorts in Bull), they should be blocked.
If they are blocked, no loss occurs.
If no loss occurs, account is not halted.

So:
1.  The filter is NOT blocking the *first* trades.
2.  Those trades execute -> Loss -> Account Halted -> Subsequent trades rejected by Risk Manager.

**Why is it not blocking the first trades?**
Maybe `current_regime` passed to `execute_signal` is 0?
In `replay.py`:
```python
        # Get current regime
        current_regime = int(row['regime']) if 'regime' in row else 0
```
And `bars` has `regime`.
And we saw `Regime counts in slice: {1: 3163}`. So `regime` is 1.

Let's look closely at `replay.py` loop again.
```python
    # Replay loop
    for ts, row in bars.iterrows():
        # ...
        prediction = predictor.predict(...)
        
        should_trade, reason = predictor.should_trade(...)
        
        if should_trade:
             # ...
             success, exec_reason = execution_engine.execute_signal(..., current_regime=current_regime)
```

If `predictor.should_trade` returns `False`, `execute_signal` is never called.
But we know trades ARE happening (we see trade logs). So `should_trade` is True.

I suspect `bars.iterrows()` issue?
In `replay.py`:
```python
    bars_path = run_dir / f"bar_size={bar_size}" / "bars.parquet"
    bars = pd.read_parquet(bars_path)
```
Wait. In `test_multi_period.py`, we overwrite `bars` with `bars_hist` slice which has `regime`.
BUT inside `replay_session` function in `replay.py`:
```python
def replay_session(...):
    # ...
    bars_path = run_dir / f"bar_size={bar_size}" / "bars.parquet"
    bars = pd.read_parquet(bars_path)
    # ...
```
The `replay_session` function **loads bars from disk** (parquet file).
In `test_multi_period.py`, we save `bars` (which has `regime`) to parquet:
```python
        # Save bars
        bars.to_parquet(bar_dir / "bars.parquet")
```
Does `to_parquet` save the custom columns? Yes.
Does `read_parquet` read them back? Yes.

**BUT**, `replay.py` has this logic *inside the loop*:
```python
    # Check if regime is already pre-calculated in bars
    if 'regime' not in bars.columns:
        # ... calculate ...
    else:
        logger.info("Using pre-calculated regime column.")
        regime_counts = bars['regime'].value_counts()
        logger.info(f"Regime counts in slice (pre-calc): {regime_counts.to_dict()}")
```
Wait. This logic I added is **outside** `replay_session` function?
No, I added it to `replay.py`. But `replay.py` has a `replay_session` function.
Where did I add it?
I used `StrReplace` to replace lines.
Let's check where the replacement happened.
I searched for `# Replay loop`.
It's inside `replay_session`.

**Key Check**:
In `test_multi_period.py`:
```python
    # Ensure sufficient data
    if len(bars_hist) > sma_period:
        bars_hist['sma_long'] = ...
        bars_hist['regime'] = ...
```
Then:
```python
        # Slice data
        df = p['data']
        # ...
        df_slice = df[...]
        res = run_backtest(..., df_slice, ...)
```
Then in `run_backtest`:
```python
        # Save bars
        bars.to_parquet(bar_dir / "bars.parquet")
```
And then calls `replay_session`.

So `bars.parquet` SHOULD have the `regime` column.
And `replay_session` reads it.
And `replay_session` logs: `Regime counts in slice (pre-calc): {1: 3163}`.
This confirms `bars` has the column and values are `1`.

So `current_regime` should be 1.
And `execute_signal` is called with `current_regime=1`.

If `regime=1` (Bull) and `direction="SHORT"`, it should block.
Why didn't it block?
Maybe `direction` passed is not "SHORT"?
In `replay.py`:
```python
        if should_trade:
            # Determine direction from predicted side
            direction = "LONG" if predicted_side > 0 else "SHORT"
```
If `predicted_side` is negative, direction is SHORT.

Let's look at `execute_signal` again.
```python
    def execute_signal(
        self,
        # ...
        current_regime: Optional[int] = None,
    ) -> Tuple[bool, str]:
        # ...
        # Check Regime Filter
        if self.regime_filter_enabled:
             regime = current_regime
             # ...
             if regime is not None:
                if regime == 1 and direction == "SHORT":
                     # BLOCK
```

Is it possible `regime` is treated as boolean? `if regime == 1`.
If `regime` is float `1.0`, it matches `1`.
Parquet might load integers as floats if there were NaNs.
But I filled NaNs: `bars_hist['regime'] = bars_hist['regime'].fillna(0).astype(int)`.
So it should be int.

**Mystery**: Why did I see no logs?
Maybe I was looking at `tail -30`.
If the account halts quickly, the logs for the initial trades might be scrolled up.
I should `grep "Regime Filter:"` on the *entire* log file.
I did `grep "Regime Filter:" ... | head -n 20` and it returned nothing (Exit code 1).

This implies the code inside `if self.regime_filter_enabled:` is NOT being executed.
Or `self.regime_filter_enabled` is False.

In `replay.py`, I set:
```python
    if 'regime_filter' not in live_cfg:
        live_cfg['regime_filter'] = {}
    live_cfg['regime_filter']['enabled'] = True
```
This modifies the dictionary passed to `LiveExecutionEngine`.

Wait. `replay_session` loads `live_cfg` from yaml:
```python
    live_cfg = _load_yaml(config_dir / "live_trading.yaml")
```
THEN I modify it.
THEN I pass it to `LiveExecutionEngine`.

Is it possible `LiveExecutionEngine` init logic is wrong?
```python
        regime_filter_cfg = self.config.get("regime_filter", {})
        self.regime_filter_enabled = regime_filter_cfg.get("enabled", False)
```
Looks correct.

Let's add a log in `LiveExecutionEngine.__init__` to confirm it sees the flag.
And print `regime` inside `execute_signal` regardless of filter enabled status.
