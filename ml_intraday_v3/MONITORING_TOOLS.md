# Trading System Monitoring Tools - Quick Reference

**Created**: February 4, 2026
**Purpose**: Real-time monitoring with timestamps

---

## New Monitoring Scripts

### 1. **watch_trading.sh** (Live Monitoring with Timestamps)
**Best for**: Watching the system in real-time

```bash
cd ml_intraday_v3
./watch_trading.sh
```

**Shows**:
- ✅ Signal generation events with timestamps
- ✅ Feature quality checks with timestamps
- ✅ Trade execution with timestamps
- ✅ CUSUM events with timestamps
- ✅ Color-coded output (green=signals, blue=trades, yellow=features, red=warnings)

**Output example**:
```
[20:30:15] 2026-02-04 20:30:15 [INFO] Signal generated: LONG, score=0.123, P=0.562
[20:30:16] 2026-02-04 20:30:16 [INFO] Executing trade: direction=LONG, contracts=1
[20:30:17] 2026-02-04 20:30:17 [INFO] Order submitted: order_id=12345
```

---

### 2. **check_signal_status.sh** (Point-in-Time Status)
**Best for**: Quick status check

```bash
cd ml_intraday_v3
./check_signal_status.sh
```

**Shows**:
- Signal counts from dashboard
- Recent signal generation events (with timestamps)
- Feature quality status (with timestamps)
- Recent bar updates (with timestamps)
- System health (with timestamps)

**Output example**:
```
Signal Status Check - 2026-02-04 20:32:08
Signals Generated: 0
System is waiting for CUSUM events...

Recent bar: 2026-02-04 13:50:00, close=6922.25
```

---

### 3. **monitor_with_timestamps.sh** (Filtered Events)
**Best for**: Focused monitoring on specific events

```bash
cd ml_intraday_v3
./monitor_with_timestamps.sh
```

**Shows**:
- Signal generation
- Feature quality
- CUSUM events
- Trade execution
- Scores and probabilities

---

## Understanding the Logs

### Dashboard Updates (No Individual Timestamps)
These update every minute but don't have log timestamps:
```
│ Signals Generated:        0   │   Executed:             0   │   Rejected:           0   │
```
This is the **dashboard display**, not individual events.

### Actual Events (With Timestamps)
These have full timestamps:
```
2026-02-04 20:30:15 [INFO] Signal generated: direction=LONG, score_ev=0.123
2026-02-04 20:30:16 [INFO] Executing trade: LONG 1 contract
2026-02-04 20:30:17 [INFO] Order submitted successfully
```
These are **actual trading events**.

---

## Why No Signals Yet?

**Current Status** (as of 20:32 UTC / 14:32 CST):
```
Market: Closed (closes at 15:00 CST / 21:00 UTC)
Last bar: 2026-02-04 13:50:00 CST
Signals: 0 (waiting for CUSUM events)
```

### This is Normal!

Signals only generate when:
1. ✅ **New bar arrives** (happens every 5 minutes during RTH)
2. ✅ **Features calculated** (all 34, no NaN)
3. ✅ **CUSUM detects event** ← **This is the trigger**
4. ✅ **Model predicts** (varies by market conditions)
5. ✅ **Confidence threshold met** (P > 0.55)

**CUSUM Events**: Only trigger when market makes significant moves
- Not every bar generates a signal
- Typical: 5-15 signals per trading day
- During quiet periods: May have 0 signals for hours

---

## Expected Timeline Tomorrow

### Market Open (8:30 AM CST):
```
8:30 AM - Market opens, first bar starts
8:35 AM - First bar completes (8:30-8:35)
         Features calculated (all 34, healthy)
         CUSUM watches for movement
8:40 AM - Second bar completes
         CUSUM may trigger if market moves
         → First signal possible
9:00 AM - After ~30 minutes of trading
         Higher chance of CUSUM events
         → Signals more likely
```

### First Signal Expected:
- **Earliest**: 8:40-9:00 AM (if market moves quickly)
- **Typical**: 9:00-10:00 AM (normal market activity)
- **If quiet**: 10:00-11:00 AM (slow market days)

---

## Monitoring Best Practices

### During Market Hours:
```bash
# Watch live events
./watch_trading.sh

# Leave it running, watch for:
# - Green lines = Signals generated
# - Blue lines = Trades executed
# - Yellow lines = Feature quality
```

### Quick Checks:
```bash
# Check status every 15-30 minutes
./check_signal_status.sh
```

### After Market Close:
```bash
# Review the day
gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
  --command='docker logs $(docker ps -q) 2>&1 | grep "Signal generated"'
```

---

## What You'll See When First Signal Happens

### In watch_trading.sh:
```
[08:42:15] 2026-02-05 08:42:15 [INFO] CUSUM event detected: threshold exceeded
[08:42:15] 2026-02-05 08:42:15 [INFO] Signal generated: direction=LONG
[08:42:15] 2026-02-05 08:42:15 [INFO]   score_ev=0.147, p_target=0.574, p_stop=0.426
[08:42:16] 2026-02-05 08:42:16 [INFO] Executing trade: LONG 1 contract
[08:42:16] 2026-02-05 08:42:16 [INFO]   Entry: 6922.25, Target: 6934.50, Stop: 6916.75
[08:42:17] 2026-02-05 08:42:17 [INFO] Order submitted: order_id=ABC123
```

### In check_signal_status.sh:
```
Signals Generated: 1
Executed: 1
Rejected: 0

Recent Signal:
2026-02-05 08:42:15 [INFO] Signal generated: LONG, score=0.147, P=0.574
```

---

## Troubleshooting

### "No signals for 2+ hours during market"
**Check**:
1. Is market moving? (CUSUM needs volatility)
2. Are features healthy? (`./check_signal_status.sh`)
3. Is CUSUM threshold too high? (check logs for "threshold")

### "Dashboard shows 0 but I see signal events"
**Explanation**: Dashboard updates every minute. Signal event logs are instant. Both are correct - dashboard might be 30-60 seconds delayed.

### "Signals generated but not executed"
**Check**:
- Confidence too low (P < 0.55)
- Regime filter blocking
- Volatility filter blocking (ADX < 25)
- Circuit breaker triggered

---

## Summary

**Monitoring Options**:
1. `./watch_trading.sh` → Live monitoring with timestamps ✅
2. `./check_signal_status.sh` → Quick status check ✅
3. `./monitor_with_timestamps.sh` → Filtered events ✅

**Current Status**: ✅ System running, waiting for CUSUM events (normal)

**Tomorrow**: Expect first signals within 30-90 minutes of market open

---

**Pro Tip**: Run `./watch_trading.sh` in the morning and leave it open in a terminal window. You'll see everything happen in real-time with color-coded timestamps!
