# 🚀 Monday Morning Startup Checklist

**Target: Be Ready by 8:30 AM CT (Market Open)**

---

## ⏰ TIMELINE

```
7:30 AM - Wake up, coffee ☕
7:45 AM - Start checklist
8:00 AM - System startup
8:15 AM - Buffer filling
8:30 AM - READY TO TRADE! 🎯
```

---

## 📋 PRE-MARKET CHECKLIST (7:45 AM - 8:00 AM)

### **1. Environment Check** ✅

```bash
# Open terminal
cd "/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3"

# Verify you're in the right directory
pwd
# Should show: .../algos 3 topstep/ml_intraday_v3
```

**Checkpoint:** ✅ In correct directory

---

### **2. Verify Critical Files Exist** ✅

```bash
# Check model bundle
ls -lh models/saved/*.pkl

# Expected output:
# model_bundle.pkl (30-100 MB)

# Check .env file
cat ../.env | grep -E "DATABENTO|TOPSTEP"

# Expected output:
# DATABENTO_API_KEY=db-***
# TOPSTEPX_API_KEY=***
# TOPSTEPX_ACCOUNT_ID=15266746
```

**Checkpoint:**
- ✅ Model bundle exists
- ✅ API keys present
- ✅ Account ID correct (150K practice account)

---

### **3. Run Pre-Flight Tests** ✅

```bash
# Test 1: Infrastructure validation
echo "Running infrastructure tests..."
python tests/test_infrastructure_fixes.py

# Expected: ALL TESTS PASSED
```

**If tests fail:**
```bash
# Check what failed and fix:
# - Feature calculations: Should show no annualization
# - Position limits: Should be 5, not 1
# - Validation methods: Should exist in data_fetcher.py
```

```bash
# Test 2: Quick monitoring test
echo "Testing monitoring system..."
python live_trading/test_monitoring.py

# Expected: Dashboard renders, files saved
```

**Checkpoint:**
- ✅ All infrastructure tests pass
- ✅ Monitoring works

---

### **4. Check Market Status** ✅

```bash
# Verify it's a trading day (not weekend/holiday)
date

# Monday-Friday expected
# Market hours: 8:30 AM - 3:00 PM CT
```

**Checkpoint:** ✅ It's a trading day

---

## 🚀 SYSTEM STARTUP (8:00 AM - 8:15 AM)

### **5. Start Live Trading System** ✅

```bash
# Set Python path and start
PYTHONPATH=".." python live_trading/live_runner.py
```

**What you'll see:**
```
================================================================================
LIVE TRADING SYSTEM STARTUP
================================================================================

Loading configurations...
✓ Live trading config loaded
✓ Risk config loaded
✓ Execution spec loaded
✓ Label schema loaded

Initializing components...
✓ Data fetcher initialized
✓ Feature generator initialized (32 features)
✓ Model predictor initialized
✓ Execution engine initialized
✓ Monitoring systems initialized

Running startup checks...
✓ API connection: PASS
✓ Model loaded: PASS
✓ Risk config valid: PASS
✓ Data connection: PASS
✓ Buffer initializing...

All startup checks passed
================================================================================
```

**Checkpoint:** ✅ All startup checks PASS

---

### **6. Manual Confirmation** ✅

**You'll be prompted:**
```
⚠️  MANUAL CONFIRMATION REQUIRED

You are about to start LIVE TRADING:
  Account: 15266746 (Practice - $149,925)
  Symbol: MES
  Contracts: 1 per trade
  Max Positions: 5
  Risk Limits: $2,000 daily / $2,500 drawdown

Type 'YES' to confirm and start trading:
```

**Before typing YES, verify:**
- ✅ Correct account (15266746 - practice)
- ✅ Settings look right
- ✅ You're mentally ready
- ✅ Laptop plugged in / good internet

**Then type:** `YES` and press Enter

**Checkpoint:** ✅ System confirmed and starting

---

### **7. Wait for Buffer to Fill** ⏰ (8:00 AM - 8:15 AM)

**What you'll see:**
```
Initializing data buffer...
Fetching last 100 bars for feature calculation...

Progress: [===========>           ] 60/100 bars

Buffer initialized with 100 bars
Buffer range: 2025-12-30 07:00:00 to 2025-12-30 08:15:00

✓ Buffer ready for trading
```

**This takes ~15 minutes** - be patient!

**While waiting:**
- ✅ Grab water
- ✅ Review today's economic calendar
- ✅ Check market news (any major events?)
- ✅ Mental prep - deep breath!

**Checkpoint:** ✅ Buffer shows 100+ bars

---

### **8. Verify Dashboard is Updating** ✅

**You should see (updates every 10 seconds):**
```
================================================================================
                          LIVE TRADING DASHBOARD
                   Last Update: 2025-12-30 08:15:43
================================================================================

┌─ ACCOUNT STATUS ────────────────────────────────────────────────────────┐
│ Equity:      $149,925.51   │   Starting:      $149,925.51   │
│ Return:           +0.00%    │   Peak:          $149,925.51   │
│ Daily P&L:          $0.00   │   Open Positions:            0   │
└─────────────────────────────────────────────────────────────────────────┘

┌─ SIGNAL STATISTICS ─────────────────────────────────────────────────────┐
│ Signals Generated:        0   │   Executed:             0   │   Rejected:           0   │
│ Execution Rate:        0.0%   │
└─────────────────────────────────────────────────────────────────────────┘

Session Duration: 00h 00m
Press Ctrl+C to stop trading
================================================================================
```

**Checkpoint:**
- ✅ Dashboard updating every 10 seconds
- ✅ Equity shows correct starting balance
- ✅ No errors in terminal

---

## 🎯 READY TO TRADE! (8:30 AM)

### **9. Market Opens** 🔔

**What happens:**
```
[08:30:15] New bar: 2025-12-30 08:30:00, close=5012.50
[08:30:15] Generating features...
[08:30:15] Features generated: 32 features, 0 NaN
[08:30:16] Model prediction: p_target=0.12, p_stop=0.08, score=0.04
[08:30:16] Signal: BELOW threshold (0.04 < 0.10), skipping
```

**First few bars:** Likely no trades (most signals rejected - normal!)

**Checkpoint:** ✅ System processing bars

---

### **10. Monitor First Trade** 👀

**When first signal triggers:**
```
[08:47:23] Signal: LONG @ 5015.25, score=0.15
[08:47:23] Risk check: PASS
[08:47:24] Executing LONG 1 contract @ 5015.25
[08:47:24] Stop: 5000.00 (-15.25 pts), Target: 5029.75 (+14.50 pts)
[08:47:25] ✓ Order filled @ 5015.50 (0.25 slippage)
[08:47:25] Position opened: LONG 1 @ 5015.50
```

**CHECK IMMEDIATELY:**
- ✅ Fill price reasonable? (within 0.5-1.0 of expected)
- ✅ Stop/target set correctly?
- ✅ Dashboard updated with position?
- ✅ Alert logged?

**Checkpoint:** ✅ First trade executed successfully

---

## ⚠️ TROUBLESHOOTING GUIDE

### **Problem: "Connection failed" on startup**

```bash
# Check internet connection
ping google.com

# Check if .env loaded
echo $DATABENTO_API_KEY
# If empty, reload:
source ../.env

# Restart system
PYTHONPATH=".." python live_trading/live_runner.py
```

---

### **Problem: "Model bundle not found"**

```bash
# Check if file exists
ls -lh models/saved/

# If missing, check other locations:
find .. -name "model_bundle.pkl" 2>/dev/null

# If you need to retrain:
# Open notebook and run training section
```

---

### **Problem: "Buffer health check failed"**

```
Warning: Buffer only has 50 bars (need 100)
```

**Solution:** Just wait! Buffer fills over time.
- Takes ~10-15 minutes to get 100 bars
- System won't trade until buffer is full
- This is normal on startup

---

### **Problem: Dashboard not updating**

```bash
# Check if script still running
# Look for cursor blinking in terminal

# If frozen:
# 1. Ctrl+C to stop
# 2. Check logs for errors:
cat logs/alerts_*.log | tail -20

# 3. Restart
PYTHONPATH=".." python live_trading/live_runner.py
```

---

### **Problem: All signals rejected**

```
[08:45] Signal generated: score=0.12
[08:45] Rejected: risk_below_threshold
```

**Possible causes:**
1. **Threshold too high** - Check `configs/live_trading.yaml` line 75
   - Current: 0.10
   - If no trades after 30 min, consider lowering to 0.08

2. **Risk limits too tight** - Check if daily loss near limit

3. **Volatility filter active** - Check `configs/backtest.yaml`
   - Should be disabled: `volatility_filter.enabled: false`

---

### **Problem: Features look wrong**

```
[08:35] WARNING: Feature vol_20 = 15.32 (seems high)
```

**Check:**
```bash
# Stop system (Ctrl+C)

# Run feature test
python tests/test_infrastructure_fixes.py

# Look for:
# "✅ vol_20 not annualized: 0.001492"
# If shows large number (>1.0), features are wrong!
```

---

### **Problem: Fills way off expected price**

```
Expected: LONG @ 5015.25
Filled:   LONG @ 5017.00  (1.75 slippage!)
```

**This is bad if consistent. Check:**
- Market orders vs limit orders
- Spread at time of execution
- If this happens 3+ times, might need to adjust execution

**Acceptable slippage:** 0.25-1.0 points
**Concerning slippage:** >2.0 points

---

## 🛑 WHEN TO STOP TRADING

**Stop immediately if:**

❌ **Daily loss > $1,500** (75% of limit)
```
Current P&L: -$1,520
Action: Ctrl+C, stop for today
```

❌ **Trailing drawdown > $1,875** (75% of limit)
```
Max DD: $1,900
Action: Ctrl+C, stop for today
```

❌ **System errors repeatedly**
```
Multiple "Connection failed" errors
Action: Stop, debug offline, try tomorrow
```

❌ **Features clearly wrong**
```
vol_20 showing 15.0 instead of 0.0015
Action: STOP - features broken, need fix
```

❌ **Every signal rejected for 2+ hours**
```
0 trades after 30 signals
Action: Stop, check configs
```

---

## ✅ END OF DAY CHECKLIST (3:00 PM)

### **After Market Close:**

```bash
# 1. System will auto-stop or Ctrl+C
# Dashboard shows final summary

# 2. Review the day
cat logs/trades_*.csv | tail -10

# 3. Check metrics
python analyze_backtest.py logs/trades_*.csv

# 4. Review alerts
cat logs/alerts_*.log | grep -E "WARNING|CRITICAL"
```

### **Questions to Ask:**

- ✅ How many trades? (Expected: 3-8/day)
- ✅ Win rate? (Expected: 50-60%)
- ✅ Any technical issues?
- ✅ Fills close to expected?
- ✅ Max drawdown staying safe?

### **Adjustments for Tomorrow:**

- If too many trades: Raise threshold
- If too few trades: Lower threshold
- If bad fills: Check execution timing
- If features off: Debug and fix

---

## 📊 MONDAY SUCCESS CRITERIA

**Minimum (Still Success!):**
- ✅ System ran without crashes
- ✅ Got at least 1 trade
- ✅ No major technical issues
- ✅ P&L: -$200 to +$300

**Good Day:**
- ✅ 3-5 trades
- ✅ Minor issues only
- ✅ P&L: +$100 to +$400

**Great Day:**
- ✅ 5-8 trades
- ✅ No issues
- ✅ P&L: +$300 to +$600

**Remember:** Day 1 is about LEARNING, not maximizing profit!

---

## 🎯 FINAL PRE-LAUNCH CHECKLIST

**Print this and check off Monday morning:**

```
7:45 AM
□ In correct directory
□ Model bundle exists
□ .env file loaded
□ Infrastructure tests pass
□ Monitoring tests pass

8:00 AM
□ Live runner started
□ All startup checks PASS
□ Manual confirmation given
□ Buffer filling

8:15 AM
□ Buffer has 100+ bars
□ Dashboard updating
□ No errors in logs

8:30 AM
□ Market open
□ Bars processing
□ Ready for signals

DURING DAY
□ First trade executed OK
□ Fills reasonable
□ Dashboard working
□ Staying under limits

3:00 PM
□ Day complete
□ Results reviewed
□ Logs checked
□ Plan for tomorrow
```

---

## 💪 FINAL WORDS

You've prepared well:
- ✅ Excellent backtest (Grade A, 90.9%)
- ✅ Fixed features
- ✅ Monitoring ready
- ✅ Comprehensive checklist

**Monday might be bumpy, but you've got this!**

See you at 8:30 AM! 🚀

---

**Last updated:** 2025-12-27
**Your backtest:** 1,000 trades, $35,806 profit, Sharpe 3.54
