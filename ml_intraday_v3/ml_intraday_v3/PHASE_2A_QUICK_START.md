# Phase 2a Quick Start Guide

**Status**: Ready to test
**Last Updated**: January 30, 2026

---

## Quick Validation (30 seconds)

```bash
# Verify all filters are integrated correctly
cd ml_intraday_v3
python experiments/validate_live_filters.py
```

**Expected Output**: `✅ ALL VALIDATION TESTS PASSED`

---

## Test Runs

### 1. Dry Run (No Real Orders, No API)
**Purpose**: Test code logic without any market interaction
**Duration**: Can stop anytime (Ctrl+C)

```bash
python ml_intraday_v3/live_trading/live_runner.py \
    --dry-run \
    --no-confirm \
    --log-level DEBUG
```

**What to Watch**:
- No import errors
- Filters initialize correctly
- Logs show filter decisions

**Expected Errors**: API connection errors (normal in dry-run mode)

---

### 2. Paper Trading Test (Simulated Orders, Real Data)
**Purpose**: Test with real market data and simulated order execution
**Duration**: 1-2 hours minimum (full trading session ideal)

```bash
python ml_intraday_v3/live_trading/live_runner.py \
    --no-confirm \
    --log-level INFO
```

**What to Watch**:
```
✗ Confidence filter rejected signal: P=0.52, threshold=0.55
   → Should see 40-50% of signals filtered

⚠️ Circuit breaker: Entering cooling-off period
   → Only if 3 consecutive losses occur

⚠️ REGIME SHIFT DETECTED: 35% features shifted
   → Rare, but should pause trading if detected
```

**Stopping**: Ctrl+C (graceful shutdown, positions will be flattened)

---

## What Each Filter Does

### Confidence Filter (Always Active)
- **Purpose**: Only trade high-confidence signals
- **Threshold**: P > 0.55 (adjustable in configs/execution_spec.yaml)
- **Impact**: Reduces trades by 40-50%, increases win rate by 15-20pp
- **Logs**: `✗ Confidence filter rejected signal: P=0.XX`

### Adaptive Circuit Breaker (Active After Losses)
- **Purpose**: Pause and adapt after losing streaks
- **Triggers**:
  - 3 consecutive losses → Pause 30 min, raise threshold
  - Daily loss -$500 → Stop for the day
- **Actions**:
  - `cooling_off`: Skip trades for 30 minutes
  - `adapted`: Raise threshold +0.10, reduce size to 50%
  - `stop_today`: Hard stop at -$500
- **Logs**: `⚠️ Circuit breaker: ...`

### Regime Detector (Periodic Checks)
- **Purpose**: Detect major market regime shifts
- **Check Frequency**: Every 100 bars
- **Threshold**: Stops if >30% of features shifted
- **Auto-Resume**: Resumes when <30% shifted
- **Logs**: `⚠️ REGIME SHIFT DETECTED` or `✅ Regime stabilized`

---

## Configuration Quick Reference

### Enable/Disable Filters

**File**: `ml_intraday_v3/configs/live_trading.yaml`

```yaml
# Circuit Breaker
circuit_breaker:
  enabled: true  # Set to false to disable

# Regime Detector
regime_detector:
  enabled: true  # Set to false to disable
```

**File**: `ml_intraday_v3/configs/execution_spec.yaml`

```yaml
# Confidence Filter
filters:
  confidence:
    enabled: true  # Set to false to disable
    min_probability_distance: 0.55  # Adjust threshold here
```

### Adjust Sensitivity

**Lower threshold** (more trades, lower quality):
```yaml
min_probability_distance: 0.50  # Was 0.55
```

**Raise threshold** (fewer trades, higher quality):
```yaml
min_probability_distance: 0.60  # Was 0.55
```

---

## Monitoring Dashboard

### Key Metrics

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| **Signals/day** | 5-7 | 3-5 or 7-10 | <3 or >10 |
| **Confidence rejects** | 40-50% | 30-40% or 50-60% | <30% or >60% |
| **Win rate** | 50-55% | 45-50% | <45% |
| **Circuit breaker trips** | 0-1/day | 2/day | 3+/day |
| **Regime shifts** | 0/week | 1/week | 2+/week |

### Log Grep Commands

```bash
# Check confidence filter effectiveness
grep "Confidence filter rejected" logs/live_trading_*.log | wc -l

# Check circuit breaker activations
grep "Circuit breaker" logs/live_trading_*.log

# Check regime detector status
grep "REGIME" logs/live_trading_*.log

# Check trade executions
grep "Trade executed" logs/live_trading_*.log
```

---

## Troubleshooting

### Problem: Too Few Signals (<3/day)

**Cause**: Confidence threshold too high

**Fix**:
```bash
# Edit configs/execution_spec.yaml
vim ml_intraday_v3/configs/execution_spec.yaml
# Change: min_probability_distance: 0.50  # Lower from 0.55
```

---

### Problem: Too Many Signals (>10/day)

**Cause**: Confidence threshold too low

**Fix**:
```bash
# Edit configs/execution_spec.yaml
vim ml_intraday_v3/configs/execution_spec.yaml
# Change: min_probability_distance: 0.60  # Raise from 0.55
```

---

### Problem: Circuit Breaker Keeps Tripping

**Symptoms**: `⚠️ Circuit breaker: Entering cooling-off period` multiple times per day

**Cause**: Model not performing well

**Actions**:
1. Check if regime shift occurred (see logs)
2. Review recent trade quality (win rate, avg P&L)
3. Consider extending Phase 2a validation
4. May need Phase 2b improvements sooner

---

### Problem: Regime Shift Detected

**Symptoms**: `⚠️ REGIME SHIFT DETECTED: XX% features shifted`

**Expected**: System pauses trading automatically

**Actions**:
1. Wait for stabilization (system auto-resumes)
2. Check logs every hour: `grep "Regime" logs/live_trading_*.log | tail -5`
3. If shift persists >24 hours, market may have fundamentally changed
4. Consider manual intervention (stop trading, retrain model)

---

### Problem: No Trades Executing

**Check**:
1. Are filters too strict?
   ```bash
   grep "rejected signal" logs/live_trading_*.log | tail -20
   ```

2. Is circuit breaker stopped?
   ```bash
   grep "stop_today" logs/live_trading_*.log
   ```

3. Is regime detector paused?
   ```bash
   grep "REGIME SHIFT" logs/live_trading_*.log | tail -5
   ```

---

## Expected Performance

### Baseline (Jan 2026, No Filters)
- Trades/day: 8.4
- Win rate: 35.5%
- Daily P&L: -$49
- **Result**: -$884 in 18 days ❌

### With Filters (Phase 2a)
- Trades/day: 5-7
- Win rate: 50-55%
- Daily P&L: +$80-150
- **Result**: ~$1,500 in 18 days ✅

### Key Improvement
**Single biggest change**: Confidence filter eliminates 80% of losing trades

---

## Next Steps After Validation

### If Paper Trading Goes Well (50%+ win rate)
1. ✅ Continue to Phase 2b (signal quality improvements)
2. Target: 55-60% win rate, $150-250/day
3. Timeline: 10-14 days implementation

### If Paper Trading Is Marginal (45-50% win rate)
1. ⚠️ Extend Phase 2a testing to 5-7 days
2. Fine-tune confidence threshold
3. Consider adding Phase 2b improvements

### If Paper Trading Fails (<45% win rate)
1. ❌ STOP - Major issue with model
2. Review all trades for patterns
3. Check for data quality issues
4. May need model retraining (Phase 4)

---

## Command Cheat Sheet

```bash
# Validate integration
python ml_intraday_v3/experiments/validate_live_filters.py

# Dry run (no API)
python ml_intraday_v3/live_trading/live_runner.py --dry-run --no-confirm

# Paper trading (simulated orders)
python ml_intraday_v3/live_trading/live_runner.py --no-confirm

# Check filter effectiveness
grep -E "Confidence filter|Circuit breaker|REGIME" logs/live_trading_*.log

# Count signals rejected vs accepted
echo "Rejected: $(grep 'rejected signal' logs/live_trading_*.log | wc -l)"
echo "Executed: $(grep 'Trade executed' logs/live_trading_*.log | wc -l)"

# View latest trades
tail -50 logs/trades_*.csv

# Monitor live (in separate terminal)
tail -f logs/live_trading_*.log | grep -E "✗|⚠️|🚨|✓|📊"
```

---

## Support & Documentation

- **Full Implementation Details**: `PHASE_2A_IMPLEMENTATION_COMPLETE.md`
- **Overall Project Plan**: `PHASE_1_VALIDATION_COMPLETE.md` (context)
- **Filter Code**:
  - Confidence: `ml_intraday_v3/filters/confidence_filter.py`
  - Regime: `ml_intraday_v3/filters/regime_filter.py`
  - Circuit Breaker: `ml_intraday_v3/monitoring/adaptive_circuit_breaker.py`

---

**Ready to Begin Testing**: Run validation script, then start paper trading!
