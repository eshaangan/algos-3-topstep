# Kelly Criterion - Monday Paper Trading Checklist

**Date**: Starting Monday, January 13, 2026
**Status**: ✅ **READY FOR PAPER TRADING**

---

## ✅ Pre-Flight Checklist

### Configuration Status

- [x] **Kelly enabled** (`live_trading.yaml` line 112: `enabled: true`)
- [x] **Position limits updated** (5 per position, 15 concurrent)
- [x] **All tests passing** (34/34 tests)
- [x] **Documentation complete** (3 guide files)
- [x] **Execution engine updated** (Kelly integration)
- [x] **Metrics tracker updated** (Kelly logging)

### Kelly Settings (Conservative)

```yaml
kelly_sizing:
  enabled: true                   # ✓ ENABLED
  min_trades_for_kelly: 20        # ✓ Learn from first 20 trades
  kelly_fraction: 0.25            # ✓ 1/4 Kelly (conservative)
  rolling_window_trades: 50       # ✓ Use last 50 trades
  max_contracts_per_trade: 5      # ✓ Hard cap at 5
  min_contracts: 1                # ✓ Floor at 1

  confidence_boost:
    enabled: true                 # ✓ Boost on high conviction
    boost_factor: 1.5             # ✓ 1.5x multiplier
    boost_threshold: 0.15         # ✓ When score_ev ≥ 0.15

  negative_kelly_threshold: 3     # ✓ Fallback after 3 consecutive
  log_sizing_decisions: true      # ✓ Full logging
```

### Position Limits

```yaml
risk.yaml:
  max_contracts_per_position: 5   # ✓ Allows Kelly scaling
  max_concurrent_positions: 15    # ✓ Increased capacity
  max_total_contracts: 75         # ✓ 15 × 5
```

---

## 🎯 What to Expect on Monday

### Phase 1: Learning (Trades 1-20)

**Behavior**:
- All trades will use **1 contract** (fixed)
- Kelly will collect statistics (win rate, avg win/loss)
- No scaling yet - building confidence

**Console Logs**:
```
[INFO] Kelly sizing enabled: fraction=0.25
[INFO] Kelly sizing decision: contracts=1, reason=learning_phase_1/20
[INFO] Kelly sizing decision: contracts=1, reason=learning_phase_10/20
[INFO] Kelly sizing decision: contracts=1, reason=learning_phase_19/20
```

**What You'll See**:
- Trades execute normally with 1 contract
- `logs/kelly_sizing_*.csv` will show "learning_phase" reasons
- **No difference from fixed sizing** during this phase

### Phase 2: Kelly Activation (Trade 21+)

**Behavior**:
- Contracts will vary from **1 to 5** based on performance
- High score_ev signals (≥0.15) get 1.5x confidence boost
- Position size adjusts dynamically

**Console Logs**:
```
[INFO] Kelly sizing decision: contracts=2, raw_kelly=0.220, fractional=0.055, reason=kelly_0.055_score_0.120
[INFO] ✓ Trade executed: LONG 2 contracts, score=0.120

[INFO] Confidence boost applied: score_ev=0.183 >= 0.150, boost_factor=1.50, contracts: 3 → 4
[INFO] Kelly sizing decision: contracts=4, raw_kelly=0.220, fractional=0.055, reason=kelly_0.055_score_0.183
[INFO] ✓ Trade executed: LONG 4 contracts, score=0.183
```

**What You'll See**:
- Trades with **2-5 contracts** depending on Kelly calculation
- Higher contracts on **high conviction signals** (score_ev ≥ 0.15)
- Kelly log shows varying contract sizes

### Expected Contract Distribution (Based on Integration Test)

After 50 trades, you might see:
```
1 contract:  20 trades (learning phase)
2 contracts:  ~5 trades (low Kelly)
3 contracts: ~15 trades (normal Kelly)
4 contracts:  ~7 trades (high Kelly)
5 contracts:  ~3 trades (high Kelly + boost, capped)
```

---

## 📊 Monitoring Kelly Performance

### Real-Time Monitoring

**Watch console logs for**:
1. Kelly initialization:
   ```
   [INFO] Kelly sizing enabled: fraction=0.25
   ```

2. Learning phase progress:
   ```
   [INFO] Kelly sizing decision: contracts=1, reason=learning_phase_15/20
   ```

3. Kelly activation at trade 21:
   ```
   [INFO] Kelly sizing decision: contracts=3, raw_kelly=0.220, fractional=0.055, reason=kelly_0.055_score_0.120
   ```

4. Confidence boosts:
   ```
   [INFO] Confidence boost applied: score_ev=0.183 >= 0.150, boost_factor=1.50, contracts: 3 → 4
   ```

5. Position caps:
   ```
   [INFO] Kelly sizing decision: contracts=5, reason=kelly_0.144_capped_by_position_limit
   ```

### Daily Review

**Check these files at end of each day**:

1. **Kelly Log** (`logs/kelly_sizing_*.csv`):
   ```bash
   tail -20 logs/kelly_sizing_*.csv
   ```

   Look for:
   - Avg contracts per trade
   - How many trades got boosted
   - Any caps triggered (position_limit, config, margin)

2. **Trade Log** (`logs/trades_*.csv`):
   ```bash
   tail -20 logs/trades_*.csv
   ```

   Look for:
   - P&L per trade
   - Contracts column varying
   - Exit reasons (stop/target)

3. **Metrics** (`logs/metrics_*.csv`):
   ```bash
   tail -5 logs/metrics_*.csv
   ```

   Look for:
   - Total P&L trending
   - Win rate
   - Execution rate

### Quick Analysis Script

```python
import pandas as pd

# Load Kelly log
kelly = pd.read_csv("logs/kelly_sizing_<date>.csv")

print("=== KELLY PERFORMANCE ===")
print(f"Total trades: {len(kelly)}")
print(f"Learning phase: {len(kelly[kelly['reason'].str.contains('learning')])}")
print(f"Kelly active: {len(kelly[~kelly['reason'].str.contains('learning')])}")
print(f"\nAvg contracts: {kelly['contracts'].mean():.2f}")
print(f"Contracts breakdown:\n{kelly['contracts'].value_counts().sort_index()}")

# Confidence boost analysis
if 'score_ev' in kelly.columns:
    boosted = kelly[kelly['score_ev'] >= 0.15]
    print(f"\nHigh conviction trades (score_ev ≥ 0.15): {len(boosted)}/{len(kelly)} ({len(boosted)/len(kelly)*100:.1f}%)")
```

---

## 🚨 Safety & Emergency Procedures

### If You Need to Disable Kelly Immediately

**Option 1: Config Kill-Switch** (Recommended)
```bash
nano ml_intraday_v3/configs/live_trading.yaml
```

Change line 112:
```yaml
enabled: false  # ← EMERGENCY DISABLE
```

Restart live runner → immediately reverts to 1 contract.

**Option 2: Cap at 1 Contract** (Temporary)
```bash
nano ml_intraday_v3/configs/live_trading.yaml
```

Change line 127:
```yaml
max_contracts_per_trade: 1  # ← Force 1 contract
```

No restart needed → next trade uses max 1 contract.

### When to Disable Kelly

**Consider disabling if**:
- Negative Kelly for 5+ consecutive trades
- Drawdown exceeds your comfort level
- System is losing more than expected
- You want to revert to baseline for comparison

**Don't panic if**:
- First few Kelly trades lose (variance is normal)
- Kelly sizes 1 contract due to negative expectancy (it's protecting you!)
- Contracts vary trade-to-trade (that's Kelly working correctly)

### Risk Manager Override

**Remember**: Kelly **CANNOT override** RiskManager:
- $1,000 daily loss limit → trading stops
- $2,500 trailing drawdown → trading stops
- 5 consecutive losses → trading stops

Kelly respects all these limits and will halt if any are triggered.

---

## 📈 Week 1 Goals

### Data Collection

**Objective**: Collect 30-50 trades with Kelly enabled

**Target timeline**:
- **Days 1-3**: Collect 20 trades (learning phase)
- **Days 4-7**: Collect 10-30 trades (Kelly active)
- **Total**: 30-50 trades by Friday

### Metrics to Track

| Metric | Target | Notes |
|--------|--------|-------|
| **Total trades** | 30-50 | Need enough for statistical significance |
| **Win rate** | 50-55% | Baseline from backtest |
| **Avg contracts** | 1.5-3.0 | After learning phase |
| **Total P&L** | Positive | Compare to fixed 1 contract |
| **Max drawdown** | < $1,000 | Within daily loss limit |
| **Execution rate** | > 80% | Signals → trades conversion |

### Daily Log Template

```
Date: Monday, January 13, 2026
Trades today: X
Learning phase: Yes/No
Kelly active: Yes/No
Avg contracts: X.XX
P&L today: $XXX
Notes:
-
-
```

---

## 🔍 Troubleshooting

### Problem: Kelly always returns 1 contract after learning phase

**Diagnosis**:
```bash
grep "negative_expectancy" logs/kelly_sizing_*.csv
```

**Cause**: Kelly is negative (losing system)

**Solution**: This is **correct behavior**! Kelly is protecting you from scaling up a losing system. Focus on model predictions or wait for performance to improve.

**Action**: No action needed - Kelly is working as designed.

### Problem: Contracts never exceed 2-3 despite good performance

**Diagnosis**:
```bash
grep "capped_by" logs/kelly_sizing_*.csv
```

**Cause**:
- Low Kelly fraction (expected with 1/4 Kelly)
- Margin cap (unlikely with $50k)
- Position limit (check if many trades hit 5)

**Solution**:
- If you want larger sizes, increase `kelly_fraction` to 0.33 (after Week 1)
- Current settings are conservative by design

**Action**: Monitor for Week 1, tune later if needed.

### Problem: Too much variance in contract sizes

**Diagnosis**: Check standard deviation
```python
import pandas as pd
kelly = pd.read_csv("logs/kelly_sizing_*.csv")
print(f"Std dev: {kelly['contracts'].std():.2f}")
```

**Cause**: Confidence boost or high Kelly

**Solution**:
```yaml
# Reduce boost or disable it
confidence_boost:
  enabled: false
```

**Action**: Only adjust if variance is causing anxiety.

---

## 📋 Monday Morning Startup Checklist

### Before Market Open

- [ ] Verify Kelly enabled: `grep "enabled: true" ml_intraday_v3/configs/live_trading.yaml`
- [ ] Check model bundle exists and is recent
- [ ] Clear old logs if desired: `rm logs/*.csv` (optional)
- [ ] Review Topstep account balance ($50k+)
- [ ] Test internet connection
- [ ] Close unnecessary applications

### Start Live Runner

```bash
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep

python -m ml_intraday_v3.live_trading.live_runner \
  --config-dir ml_intraday_v3/configs \
  --bundle-path <path_to_your_latest_model_bundle>
```

**Replace `<path_to_your_latest_model_bundle>`** with actual path.

### Verify Kelly Initialization

**Look for these lines in startup**:
```
[INFO] Kelly sizing enabled: fraction=0.25
[INFO] KellySizer initialized: enabled=True, kelly_fraction=0.25, min_trades=20
```

If you see this, Kelly is ready! ✅

### First Trade Verification

**Watch for**:
```
[INFO] Kelly sizing decision: contracts=1, reason=learning_phase_1/20
[INFO] ✓ Trade executed: LONG 1 contract, score=0.XXX
```

This confirms Kelly is working correctly.

---

## 📞 Quick Reference

### File Locations

```
Config:
  ml_intraday_v3/configs/live_trading.yaml  (Kelly settings)
  ml_intraday_v3/configs/risk.yaml          (Position limits)

Logs:
  logs/kelly_sizing_*.csv                    (Kelly decisions)
  logs/trades_*.csv                          (Completed trades)
  logs/metrics_*.csv                         (Session metrics)

Docs:
  ml_intraday_v3/KELLY_IMPLEMENTATION_SUMMARY.md  (Full details)
  ml_intraday_v3/KELLY_QUICK_START.md            (Quick guide)
  ml_intraday_v3/KELLY_TEST_RESULTS.md           (Test results)
  ml_intraday_v3/KELLY_MONDAY_CHECKLIST.md       (This file)
```

### Kelly Settings Quick Reference

```yaml
enabled: true                   # Master on/off
min_trades_for_kelly: 20        # Learning phase length
kelly_fraction: 0.25            # 1/4 Kelly (conservative)
max_contracts_per_trade: 5      # Hard cap
confidence_boost:
  enabled: true
  boost_factor: 1.5
  boost_threshold: 0.15
```

### Emergency Disable

```bash
# Edit config
nano ml_intraday_v3/configs/live_trading.yaml

# Change line 112
enabled: false

# Restart live runner
```

---

## ✅ You're Ready!

**Kelly is enabled and ready for Monday paper trading.**

### What Happens Monday

1. **Trades 1-20**: Fixed 1 contract (learning phase)
2. **Trade 21+**: Dynamic 2-5 contracts (Kelly active)
3. **High conviction signals**: Get 1.5x boost
4. **All trades**: Logged to `kelly_sizing_*.csv`

### Your Action Items

✅ **Monday morning**: Start live runner
✅ **Throughout week**: Monitor console logs
✅ **Daily**: Check Kelly log for contract distribution
✅ **Friday**: Analyze week 1 results

### Support

- **Full implementation details**: `KELLY_IMPLEMENTATION_SUMMARY.md`
- **Quick enable/disable**: `KELLY_QUICK_START.md`
- **Test results**: `KELLY_TEST_RESULTS.md`

---

**Good luck with Monday's paper trading!** 🚀

Kelly will start conservative (1 contract for 20 trades), learn from your live performance, then scale up strategically. The system is designed to protect you while maximizing returns on strong signals.

**Remember**: You can disable Kelly instantly at any time if needed. It's just one line in the config file.

---

**Checklist created**: January 9, 2026
**Paper trading start**: Monday, January 13, 2026
**Kelly status**: ✅ **ENABLED AND READY**
