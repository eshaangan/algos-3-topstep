# Performance Monitoring & Alerts Guide

## 🎯 Overview

Your live trading system now includes comprehensive monitoring and alerting features:
- **Real-time Dashboard** - Terminal-based performance metrics
- **Alert System** - Notifications for important events
- **Performance Tracking** - Detailed metrics storage
- **Auto-saved Reports** - CSV files for post-session analysis

---

## 📊 What You'll See

### Terminal Dashboard (Updates Every 10 Seconds)

```
================================================================================
                          LIVE TRADING DASHBOARD
                   Last Update: 2025-12-30 09:45:23
================================================================================

┌─ ACCOUNT STATUS ──────────────────────────────────────────────────────────┐
│ Equity:      $149,925.51   │   Starting:      $149,925.51   │
│ Return:          +0.00%     │   Peak:          $149,925.51   │
│ Daily P&L:         $0.00    │   Open Positions:            0   │
└────────────────────────────────────────────────────────────────────────────┘

┌─ TRADING PERFORMANCE ─────────────────────────────────────────────────────┐
│ Total Trades:         15   │   Winning:            9   │   Losing:      6   │
│ Win Rate:           60.0%   │   Profit Factor:     2.15   │
│ Total P&L:      $1,245.50   │   Avg Trade:        $83.03   │
│ Max Win:          $425.00   │   Max Loss:       -$215.00   │
└────────────────────────────────────────────────────────────────────────────┘

┌─ RISK METRICS ────────────────────────────────────────────────────────────┐
│ Max Drawdown:    $125.00 ( 0.1%)   │   Daily Loss Limit: $2,000   │
│ Daily Limit Used:      0.0%   │   DD Limit: $2,500             │
│ Current Streak:       3 W   │   Max Win Streak:          4   │
└────────────────────────────────────────────────────────────────────────────┘

┌─ SIGNAL STATISTICS ───────────────────────────────────────────────────────┐
│ Signals Generated:       45   │   Executed:        15   │   Rejected:    30   │
│ Execution Rate:        33.3%   │
└────────────────────────────────────────────────────────────────────────────┘

┌─ ALERTS ──────────────────────────────────────────────────────────────────┐
│ Alerts: 20 total (INFO: 15, WARNING: 3, CRITICAL: 2)                      │
└────────────────────────────────────────────────────────────────────────────┘

Session Duration: 01h 15m

Press Ctrl+C to stop trading
================================================================================
```

---

## 🔔 Alert System

### Alert Levels

**INFO** ⓘ - Normal events
- Trade executed
- Trade closed
- Session started/ended
- Connection restored

**WARNING** ⚠️ - Important notifications
- Approaching risk limits (75%+)
- Losing trades
- Multiple consecutive losses

**CRITICAL** 🚨 - Urgent issues
- Risk limit breached
- Connection lost
- Errors occurred
- Account liquidation

### Alert Delivery

**1. Terminal Logging**
- Shows in real-time during trading
- Color-coded by severity

**2. Alert Log File**
```
logs/alerts_YYYYMMDD_HHMMSS.log
```

**3. Sound Alerts** (macOS)
- Critical alerts play sound
- Helps catch urgent issues

### Example Alerts

```
2025-12-30 09:31:00 [INFO] Trade Executed: LONG 1 @ 5012.50
2025-12-30 09:45:00 [INFO] Trade Closed: LONG closed, PnL=$45.25 (target)
2025-12-30 10:15:00 [WARNING] Daily Loss Warning: Daily P&L $-1,625.00 is 81.3% of limit ($2,000.00)
2025-12-30 10:30:00 [CRITICAL] RISK BREACH: daily_loss: $2,050.00 exceeds limit $2,000.00
```

---

## 📈 Performance Metrics Tracked

### Trading Statistics
- Total trades (winning/losing)
- Win rate %
- Profit factor
- Average trade size
- Max win/loss
- Current & max streaks

### P&L Metrics
- Total P&L
- Gross profit/loss
- Daily P&L
- Return %

### Risk Metrics
- Current equity
- Peak equity (HWM)
- Max drawdown ($ and %)
- Daily loss limit usage %
- Drawdown limit usage %

### Execution Stats
- Signals generated
- Signals executed
- Signals rejected
- Execution rate %

---

## 💾 Saved Files

### During Trading Session

**1. Live Metrics** (updated every 10 seconds)
```
logs/metrics_YYYYMMDD_HHMMSS.csv
```
Contains snapshots of all metrics over time.

**2. Alert Log**
```
logs/alerts_YYYYMMDD_HHMMSS.log
```
All alerts with timestamps and details.

**3. Session Log**
```
logs/live_trading_YYYYMMDD_HHMMSS.log
```
Complete session log (all terminal output).

### End of Session

**4. Trade History**
```
logs/trades_YYYYMMDD_HHMMSS.csv
```
Every trade with entry/exit details, P&L, duration.

**5. Final Metrics Snapshot**
```
logs/metrics_YYYYMMDD_HHMMSS.csv
```
Updated with final session stats.

---

## 📊 Post-Session Analysis

### View Trade History
```bash
# Latest trades
ls -lt logs/trades_*.csv | head -1 | awk '{print $9}' | xargs cat

# Open in Excel/Numbers
open logs/trades_20251230_150000.csv
```

### View Session Metrics
```bash
# Latest metrics
ls -lt logs/metrics_*.csv | head -1 | awk '{print $9}' | xargs cat
```

### Analyze Performance
```python
import pandas as pd

# Load trades
trades = pd.read_csv('logs/trades_20251230_150000.csv')

# Win rate by direction
print(trades.groupby('direction')['pnl'].agg(['count', 'mean', lambda x: (x > 0).mean()]))

# Best/worst hours
trades['hour'] = pd.to_datetime(trades['entry_time']).dt.hour
print(trades.groupby('hour')['pnl'].agg(['count', 'sum', 'mean']))

# Exit reason analysis
print(trades.groupby('exit_reason')['pnl'].agg(['count', 'mean']))
```

---

## ⚙️ Customization

### Update Frequency

**Dashboard** (default: 10 seconds)

Edit `ml_intraday_v3/live_trading/live_runner.py`:
```python
self.dashboard_update_interval = 10  # Change to desired seconds
```

### Alert Thresholds

**Risk Warnings** (default: 75% of limits)

Edit `ml_intraday_v3/live_trading/live_runner.py`:
```python
# Daily loss warning at 75% ($1,500 of $2,000)
if status['daily_pnl'] < 0 and abs(status['daily_pnl']) > 1500:
    self.alert_manager.daily_loss_warning(status['daily_pnl'], 2000)

# Change to 80% ($1,600)
if status['daily_pnl'] < 0 and abs(status['daily_pnl']) > 1600:
    self.alert_manager.daily_loss_warning(status['daily_pnl'], 2000)
```

### Disable Sound Alerts

Edit `ml_intraday_v3/live_trading/live_runner.py`:
```python
self.alert_manager = AlertManager(logs_dir, enable_sound=False)
```

---

## 🎯 Key Features

### ✅ Real-time Monitoring
- Live dashboard updates every 10 seconds
- No need to check logs constantly
- See performance at a glance

### ✅ Proactive Alerts
- Warns before hitting risk limits (at 75%)
- Catch issues early
- Take action before forced halt

### ✅ Complete Audit Trail
- Every trade logged with full details
- All alerts saved to file
- Full session replay from logs

### ✅ Performance Analytics
- Track win rate, profit factor, streaks
- Identify best/worst trading hours
- Analyze exit reasons (stop vs target)

---

## 🚀 Quick Start

**Run with monitoring (automatic):**
```bash
cd ml_intraday_v3
PYTHONPATH=".." python live_trading/live_runner.py
```

**What you'll see:**
1. Startup checks display
2. Manual confirmation prompt
3. Live dashboard (updates every 10s)
4. Trade alerts in real-time
5. Risk warnings if approaching limits
6. End-of-session summary

**Stop trading:**
- Press `Ctrl+C`
- Dashboard shows final summary
- All files auto-saved

---

## 📁 File Organization

```
logs/
├── live_trading_20251230_090000.log   # Full session log
├── alerts_20251230_090000.log          # Alert history
├── trades_20251230_153000.csv          # Trade details
└── metrics_20251230_090000.csv         # Performance metrics
```

**Retention:** Keep important sessions, delete others to save space.

---

## ✅ You're Ready!

All monitoring is **automatic** - just run the system and watch the dashboard!

When markets open Monday:
```bash
cd ml_intraday_v3
PYTHONPATH=".." python live_trading/live_runner.py
```

You'll see the live dashboard updating every 10 seconds with all your performance metrics! 📊🚀
