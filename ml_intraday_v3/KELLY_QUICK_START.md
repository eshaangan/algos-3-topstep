# Kelly Criterion Quick Start Guide

**Status**: ✅ Implementation complete, ready for paper trading

---

## What Was Implemented

✅ **Kelly Criterion position sizing** - Dynamically scales contracts based on live performance
✅ **Learning phase** - Uses 1 contract for first 20 trades to build statistics
✅ **Confidence boost** - Increases size on high-conviction signals (score_ev ≥ 0.15)
✅ **Safety mechanisms** - Multiple fallbacks, caps, and instant kill-switch
✅ **Full integration** - Works seamlessly with existing live trading system
✅ **Comprehensive testing** - 33/33 tests passing

**Current state**: Kelly is **DISABLED by default** in `live_trading.yaml`.

---

## Quick Enable (3 Steps)

### 1. Edit Configuration

```bash
nano ml_intraday_v3/configs/live_trading.yaml
```

Find the `kelly_sizing` section (around line 108) and change:
```yaml
kelly_sizing:
  enabled: false  # ← Change to true
```

To:
```yaml
kelly_sizing:
  enabled: true  # ← Now enabled
```

Save and exit (Ctrl+O, Enter, Ctrl+X).

### 2. Verify Settings

Double-check these values in the same section:
```yaml
kelly_sizing:
  enabled: true                 # ✓ Just changed
  min_trades_for_kelly: 20      # ✓ 20 trades before Kelly activates
  kelly_fraction: 0.25          # ✓ 1/4 Kelly (conservative)
  max_contracts_per_trade: 5    # ✓ Hard cap at 5 contracts
  confidence_boost:
    enabled: true               # ✓ Boost on high-conviction signals
    boost_factor: 1.5           # ✓ 1.5x multiplier
    boost_threshold: 0.15       # ✓ When score_ev ≥ 0.15
```

### 3. Restart Live Runner

```bash
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep
python -m ml_intraday_v3.live_trading.live_runner \
  --config-dir ml_intraday_v3/configs \
  --bundle-path <path_to_your_model_bundle>
```

Look for:
```
[INFO] Kelly sizing enabled: fraction=0.25
```

---

## What to Expect

### Phase 1: Learning (Trades 1-20)

**Behavior**:
- All trades use **1 contract** (fixed)
- Kelly collects statistics but doesn't size yet
- Logs show: `"learning_phase_5/20"`, `"learning_phase_15/20"`, etc.

**Example Log**:
```
[INFO] Kelly sizing decision: contracts=1, raw_kelly=0.000, fractional=0.000, reason=learning_phase_15/20
[INFO] ✓ Trade executed: LONG 1 contract, score=0.123
```

### Phase 2: Kelly Active (Trades 21+)

**Behavior**:
- Contracts vary from **1 to 5** based on Kelly calculation
- High score_ev signals (≥0.15) get confidence boost (1.5x)
- Negative Kelly falls back to 1 contract

**Example Logs**:

**Normal Kelly sizing** (positive expectancy):
```
[INFO] Kelly sizing decision: contracts=2, raw_kelly=0.220, fractional=0.055, score_ev=0.120, reason=kelly_0.055_score_0.120
[INFO] ✓ Trade executed: LONG 2 contracts, score=0.120
```

**With confidence boost** (high conviction):
```
[INFO] Confidence boost applied: score_ev=0.183 >= 0.15, boost_factor=1.50, contracts: 2 → 3
[INFO] Kelly sizing decision: contracts=3, raw_kelly=0.220, fractional=0.055, score_ev=0.183, reason=kelly_0.055_score_0.183
[INFO] ✓ Trade executed: LONG 3 contracts, score=0.183
```

**Negative expectancy** (losing system):
```
[INFO] Kelly sizing decision: contracts=1, raw_kelly=-0.087, fractional=-0.022, score_ev=0.120, reason=negative_expectancy_kelly_-0.087
[INFO] ✓ Trade executed: LONG 1 contract, score=0.120
```

---

## Monitoring Kelly Performance

### 1. Real-Time Console Logs

Watch for sizing decisions in the console:
```bash
tail -f <live_runner_log_file>
```

Look for lines with `"Kelly sizing decision:"`.

### 2. Kelly Sizing CSV

After the session, check:
```bash
ls -lh logs/kelly_sizing_*.csv
```

Open in pandas or Excel to analyze:
```python
import pandas as pd

kelly_log = pd.read_csv("logs/kelly_sizing_20260109_084537.csv")

# Summary stats
print(kelly_log['contracts'].describe())
print(f"Avg contracts: {kelly_log['contracts'].mean():.2f}")
print(f"Contracts breakdown:\n{kelly_log['contracts'].value_counts()}")

# Confidence boost frequency
boosted = kelly_log[kelly_log['score_ev'] >= 0.15]
print(f"Confidence boost applied: {len(boosted)}/{len(kelly_log)} trades ({len(boosted)/len(kelly_log)*100:.1f}%)")
```

### 3. Metrics Tracker

Check the metrics CSV:
```bash
ls -lh logs/metrics_*.csv
```

Compare execution_rate, avg_trade, profit_factor before/after Kelly activation.

---

## Quick Disable (If Needed)

### Option 1: Config Kill-Switch (Recommended)

```bash
nano ml_intraday_v3/configs/live_trading.yaml
```

Change:
```yaml
kelly_sizing:
  enabled: false  # ← Back to disabled
```

Restart `live_runner.py` → immediately reverts to 1 contract per trade.

### Option 2: Force 1 Contract (Emergency)

If you can't restart, edit config while running:
```yaml
kelly_sizing:
  enabled: true
  max_contracts_per_trade: 1  # ← Force cap at 1
```

Next trade will use max 1 contract even if Kelly says more.

---

## Understanding Sizing Reasons

| Reason Code | Meaning | Action |
|-------------|---------|--------|
| `learning_phase_15/20` | Collecting data (15 trades done) | Normal, wait for 20 trades |
| `kelly_0.055_score_0.120` | Normal Kelly sizing | Good, system is working |
| `kelly_0.137_score_0.183` | Higher Kelly + high conviction | Good, strong signal |
| `kelly_0.055_capped_by_config` | Kelly wants more than 5 contracts | Config cap working |
| `kelly_0.055_capped_by_margin` | Insufficient equity | Consider adding capital |
| `negative_expectancy_kelly_-0.087` | Losing system, Kelly negative | Normal fallback to 1 |
| `consecutive_negative_kelly_3` | 3+ negative Kellys in a row | Safety mechanism activated |
| `disabled` | Kelly turned off in config | Expected if disabled |

---

## Performance Comparison

### Week 1: Baseline (Kelly Disabled)

**Objective**: Collect baseline statistics with fixed 1 contract sizing.

**Steps**:
1. Keep `kelly_sizing.enabled: false`
2. Run paper trading for 5-7 days
3. Collect 30-50 trades
4. Record metrics:
   - Total P&L
   - Sharpe ratio
   - Max drawdown
   - Win rate
   - Profit factor
   - Avg trade

### Week 2: Kelly Test (Kelly Enabled)

**Objective**: Test Kelly sizing and compare to baseline.

**Steps**:
1. Set `kelly_sizing.enabled: true`
2. Run paper trading for 5-7 days
3. Collect 30-50 trades
4. Record same metrics

### Week 3: Analysis

**Compare**:
```python
import pandas as pd

# Load both periods
baseline_trades = pd.read_csv("logs/trades_baseline.csv")
kelly_trades = pd.read_csv("logs/trades_kelly.csv")

# Compare P&L
baseline_pnl = baseline_trades['pnl'].sum()
kelly_pnl = kelly_trades['pnl'].sum()
print(f"Baseline P&L: ${baseline_pnl:.2f}")
print(f"Kelly P&L: ${kelly_pnl:.2f}")
print(f"Improvement: {(kelly_pnl - baseline_pnl) / abs(baseline_pnl) * 100:+.1f}%")

# Compare Sharpe
baseline_sharpe = baseline_trades['pnl'].mean() / baseline_trades['pnl'].std() * (252**0.5)
kelly_sharpe = kelly_trades['pnl'].mean() / kelly_trades['pnl'].std() * (252**0.5)
print(f"Baseline Sharpe: {baseline_sharpe:.2f}")
print(f"Kelly Sharpe: {kelly_sharpe:.2f}")

# Compare drawdown (simplified)
baseline_cum = baseline_trades['pnl'].cumsum()
kelly_cum = kelly_trades['pnl'].cumsum()
baseline_dd = (baseline_cum - baseline_cum.cummax()).min()
kelly_dd = (kelly_cum - kelly_cum.cummax()).min()
print(f"Baseline Max DD: ${baseline_dd:.2f}")
print(f"Kelly Max DD: ${kelly_dd:.2f}")
```

**Decision Criteria**:
- ✅ **Keep Kelly** if: Kelly Sharpe > Baseline Sharpe AND Kelly DD < 1.5× Baseline DD
- ⚠️ **Tune Kelly** if: Higher returns but much higher DD (reduce kelly_fraction to 0.20)
- ❌ **Disable Kelly** if: Worse Sharpe OR significantly higher DD with no benefit

---

## Advanced Tuning

If Kelly is working but you want to adjust behavior:

### Make Kelly More Conservative

Reduce `kelly_fraction` (currently 0.25):
```yaml
kelly_sizing:
  kelly_fraction: 0.20  # 1/5 Kelly instead of 1/4 (more conservative)
```

Or reduce `max_contracts_per_trade`:
```yaml
kelly_sizing:
  max_contracts_per_trade: 3  # Cap at 3 instead of 5
```

### Make Kelly More Aggressive

Increase `kelly_fraction`:
```yaml
kelly_sizing:
  kelly_fraction: 0.33  # 1/3 Kelly instead of 1/4 (more aggressive)
```

**⚠️ Warning**: Do NOT use full Kelly (1.0) - far too risky for Topstep.

### Adjust Confidence Boost

Tighten threshold (fewer boosts, higher quality):
```yaml
kelly_sizing:
  confidence_boost:
    boost_threshold: 0.20  # Only boost on very strong signals
```

Or increase multiplier:
```yaml
kelly_sizing:
  confidence_boost:
    boost_factor: 2.0  # 2x instead of 1.5x
```

### Change Rolling Window

Use shorter window (adapts faster to recent performance):
```yaml
kelly_sizing:
  rolling_window_trades: 30  # Last 30 trades instead of 50
```

Or remove window (use all trades since session start):
```yaml
kelly_sizing:
  rolling_window_trades: null  # Use all trades
```

---

## Troubleshooting

### Problem: Kelly always returns 1 contract after learning phase

**Cause**: Negative Kelly (losing system).

**Diagnosis**:
```bash
grep "negative_expectancy" logs/<latest_log_file>
```

**Solution**: This is correct behavior! Kelly is protecting you from scaling up a losing system. Focus on improving model predictions or wait for performance to improve.

### Problem: Kelly log not being created

**Cause**: `save_kelly_log()` not called or Kelly disabled.

**Diagnosis**:
```bash
grep "Kelly sizing enabled" logs/<latest_log_file>
```

**Solution**: Verify `kelly_sizing.enabled: true` in live_trading.yaml and restart.

### Problem: Contracts never exceed 1 despite positive Kelly

**Cause**: Tight caps (margin, position limit, or config max).

**Diagnosis**: Check logs for reason codes:
```bash
grep "capped_by" logs/<latest_log_file>
```

**Solution**:
- If `capped_by_margin`: Need more equity (unlikely with $50k+)
- If `capped_by_config`: Increase `max_contracts_per_trade`
- If `capped_by_position_limit`: Increase in risk.yaml (already at 5)

### Problem: Too much variance in contract sizes

**Cause**: Kelly fraction too high or confidence boost too aggressive.

**Solution**: Reduce `kelly_fraction` to 0.20 or disable confidence boost:
```yaml
kelly_sizing:
  kelly_fraction: 0.20
  confidence_boost:
    enabled: false  # Disable boost
```

---

## Testing Checklist

Before enabling in live paper trading:

- [x] Unit tests pass (`pytest ml_intraday_v3/tests/test_kelly_sizer.py`)
- [x] Smoke tests pass (`python ml_intraday_v3/tests/test_kelly_integration_smoke.py`)
- [ ] Replay test with historical data (optional but recommended)
- [ ] 1 week baseline paper trading (Kelly disabled)
- [ ] 1 week Kelly paper trading (Kelly enabled)
- [ ] Performance comparison analysis
- [ ] Decision: Keep, tune, or disable

---

## Summary

Kelly Criterion is ready to use:

1. **Enable**: Edit `live_trading.yaml`, set `enabled: true`
2. **Monitor**: Watch console logs and `kelly_sizing_*.csv`
3. **Validate**: Compare 1 week baseline vs 1 week Kelly
4. **Tune**: Adjust `kelly_fraction` or `boost_threshold` as needed
5. **Rollback**: Instant disable via config if needed

**Current recommendation**: Start with Kelly **disabled**, collect 1 week baseline, then enable for comparison.

---

**Last updated**: January 9, 2026
**Status**: ✅ Ready for paper trading validation
