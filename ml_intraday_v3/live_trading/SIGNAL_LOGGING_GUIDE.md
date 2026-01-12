# Signal Logging Enhancement

## Overview

Enhanced signal logging to track **all signals** (both executed and rejected) with their prediction scores. This provides visibility into model performance and filtering effectiveness.

---

## What Was Added

### 1. **Enhanced MetricsTracker** (`monitoring/metrics_tracker.py`)

**New Features:**
- `record_signal()` now accepts signal score, timestamp, direction, and rejection reason
- New `signal_history` list tracks all signals
- New `save_signal_log()` method writes signals to CSV

**Signature:**
```python
def record_signal(
    self,
    executed: bool,
    score: float,              # NEW: prediction score
    timestamp: Optional[pd.Timestamp] = None,  # NEW: when signal occurred
    direction: Optional[str] = None,           # NEW: LONG/SHORT
    reason: Optional[str] = None,              # NEW: rejection reason
):
```

### 2. **Upgraded Live Runner Logging** (`live_trading/live_runner.py`)

**Changes:**
- Signal scores now logged at **INFO level** (was DEBUG)
- All signal rejections logged with score and reason
- Executed trades show score in log message
- Signal log saved periodically alongside metrics

**Example Logs:**
```
[INFO] Signal generated: score=0.243, p_target=0.65, p_stop=0.35
[INFO] ✗ Signal rejected: score=0.082, reason=primary_threshold (score=0.082 < 0.100)
[INFO] ✓ Trade executed: LONG 1 contracts, score=0.243
```

### 3. **Signal Log CSV** (`logs/signals_YYYYMMDD_HHMMSS.csv`)

**Columns:**
- `timestamp`: When signal was generated
- `score`: Model prediction score (score_ev)
- `executed`: True if trade executed, False if rejected
- `direction`: LONG/SHORT (if executed)
- `reason`: Execution status or rejection reason

**Example:**
```csv
timestamp,score,executed,direction,reason
2026-01-09 10:30:00,0.082,False,,primary_threshold
2026-01-09 10:35:00,0.243,True,LONG,executed
2026-01-09 10:40:00,0.156,False,LONG,max_concurrent_positions
2026-01-09 10:45:00,-0.198,True,SHORT,executed
```

---

## Signal Rejection Reasons

Your signals can be rejected for multiple reasons:

### Model Threshold Filter
- `primary_threshold`: Score below configured threshold (default 0.10)
- `meta_threshold`: Meta model score too low (if enabled)

### Risk Manager Gates
- `halted`: Trading halted from previous breach
- `max_trades`: Hit max trades per day (20)
- `consecutive_losses`: Too many losses in a row (5)
- `min_time`: Not enough time since last trade (60s)
- `risk_daily_loss`: Daily loss limit breached ($1,000)
- `risk_drawdown`: Max drawdown breached ($2,500)

### Execution Engine Limits
- `max_concurrent_positions`: Already at 5 open positions
- `no_bars`: Data buffer empty
- `execution_error`: API/broker error

---

## Usage Examples

### Analyzing Signal Quality

```python
import pandas as pd

# Load signal log
signals = pd.read_csv("logs/signals_20260109_103000.csv")

# Overall statistics
print(f"Total signals: {len(signals)}")
print(f"Executed: {signals['executed'].sum()}")
print(f"Rejected: {(~signals['executed']).sum()}")
print(f"Execution rate: {signals['executed'].mean():.1%}")

# Score distribution
print("\nExecuted signals:")
print(signals[signals['executed']]['score'].describe())

print("\nRejected signals:")
print(signals[~signals['executed']]['score'].describe())

# Rejection reasons
print("\nTop rejection reasons:")
print(signals[~signals['executed']]['reason'].value_counts())
```

### Comparing Executed vs Rejected Scores

```python
import matplotlib.pyplot as plt

executed = signals[signals['executed']]['score']
rejected = signals[~signals['executed']]['score']

plt.figure(figsize=(10, 6))
plt.hist(rejected, bins=30, alpha=0.5, label='Rejected')
plt.hist(executed, bins=30, alpha=0.5, label='Executed')
plt.axvline(0.10, color='red', linestyle='--', label='Threshold')
plt.xlabel('Signal Score')
plt.ylabel('Count')
plt.title('Signal Score Distribution: Executed vs Rejected')
plt.legend()
plt.show()
```

### Finding Missed Opportunities

```python
# Signals that were rejected but had strong scores
missed = signals[~signals['executed'] & (signals['score'].abs() > 0.15)]
print(f"\nStrong signals rejected: {len(missed)}")
print(missed[['timestamp', 'score', 'reason']])
```

---

## Integration with Existing Logs

Signal logs complement existing logs:

1. **Main log** (`logs/live_trading_*.log`): Full system activity
2. **Signal log** (`logs/signals_*.csv`): All model predictions
3. **Trade log** (`logs/trades_*.csv`): Executed trades only
4. **Metrics log** (`logs/metrics_*.csv`): Equity/performance snapshots
5. **Alert log** (`logs/alerts_*.log`): Risk warnings

---

## Configuration

No config changes needed - signal logging is always active when running live trading.

To adjust what gets filtered, modify `ml_intraday_v3/configs/live_trading.yaml`:

```yaml
signals:
  primary_threshold: 0.10  # Lower = more signals pass
  use_meta_model: false    # Enable for 2-stage filtering
  meta_threshold: 0.50     # Meta model threshold
```

---

## Files Modified

1. `ml_intraday_v3/monitoring/metrics_tracker.py`
   - Enhanced `record_signal()` method
   - Added `signal_history` tracking
   - Added `save_signal_log()` method

2. `ml_intraday_v3/live_trading/live_runner.py`
   - Upgraded signal logging from DEBUG to INFO
   - Pass score to `record_signal()` for all signals
   - Call `save_signal_log()` periodically

3. `ml_intraday_v3/live_trading/replay.py`
   - Updated to pass score to `record_signal()`
   - Track rejected signals

4. `ml_intraday_v3/live_trading/test_monitoring.py`
   - Updated test calls to new signature

---

## Next Steps

After your next paper trading session:

1. Check `logs/signals_*.csv` to see all generated signals
2. Analyze score distribution vs execution rate
3. Look for patterns in rejection reasons
4. Consider adjusting threshold if too many/few signals
5. Identify if risk gates are blocking good signals

This data will help you optimize the signal filtering pipeline!
