# Quick Start Guide: Paper Trading Setup

## ✅ Setup Complete!

Your paper trading system is ready. Here's what's been configured:

### Account Status
- **Paper Account**: $49,625.11 equity
- **Configured as**: $50,000 starting balance (Topstep 50K Combine rules)
- **Risk Limits**:
  - Daily loss: $2,000
  - Trailing drawdown: $2,500
  - Max concurrent positions: 5

### System Components

All live trading modules have been created and tested:

```
ml_intraday_v3/
├── configs/
│   ├── live_trading.yaml     ← Trading configuration
│   ├── risk.yaml              ← Risk limits (updated to $2K/$2.5K)
│   └── execution_spec.yaml    ← Execution settings
├── live_trading/
│   ├── data_fetcher.py        ← Real-time data from Databento
│   ├── feature_generator.py   ← Live feature calculation
│   ├── model_predictor.py     ← Model loading & predictions
│   ├── execution_engine.py    ← Trade execution via Topstep API
│   └── live_runner.py         ← Main orchestrator
├── LIVE_TRADING_CHECKLIST.md  ← Detailed safety checklist
└── START_TRADING.md           ← This file
```

## 🚀 How to Start Trading

### Step 1: Dry Run (Recommended First)

Test signal generation without executing trades:

```bash
cd ml_intraday_v3
PYTHONPATH="." python live_trading/live_runner.py --dry-run
```

**What happens**:
- Connects to data feed
- Loads trained model
- Generates features and signals
- Shows what trades WOULD be executed
- Does NOT place actual orders

**Duration**: Run for 1-2 hours to verify everything works

### Step 2: Paper Trading (Execute on Paper Account)

Start actual paper trading:

```bash
cd ml_intraday_v3
PYTHONPATH="." python live_trading/live_runner.py
```

**What happens**:
1. Startup checks run (data, API, model, risk)
2. Manual confirmation required (type 'yes')
3. Starts trading loop:
   - Fetches 1m bars from Databento
   - Calculates features
   - Generates predictions
   - Executes trades via Topstep API
   - Monitors positions
   - Enforces risk limits

**Duration**: Run for at least 1 full trading day initially

### Step 3: Stop Trading

Press **Ctrl+C** to stop gracefully. The system will:
- Stop generating new signals
- Flatten all open positions
- Save trade log to `logs/trades_YYYYMMDD_HHMMSS.csv`
- Display final statistics

## 📊 Monitoring

### Real-time Logs

**Terminal**: Watch live output for:
- ✓ New bars received
- ✓ Signals generated
- ✓ Trades executed
- ⚠️ Risk warnings
- ✗ Errors

**Log Files**: All terminal output is also saved to:
```
logs/live_trading_YYYYMMDD_HHMMSS.log
```

### Trade Log

After each session, review:
```bash
# Trade history (CSV)
ls -lt ml_intraday_v3/logs/trades_*.csv | head -1

# Full session log
ls -lt ml_intraday_v3/logs/live_trading_*.log | head -1
```

### Topstep Dashboard

Monitor your account:
1. Log into https://www.topstepx.com
2. Go to "Positions" tab
3. Verify trades match system logs

## ⚠️ Safety Features

### Automated Risk Gates

Trading will HALT automatically if:
- Daily loss ≥ $2,000
- Trailing drawdown ≥ $2,500
- 5 consecutive losses
- Max 20 trades per day
- Equity < $40,000 (80% floor)
- API connection lost
- Feature quality issues

### Manual Stop

Three ways to stop trading:

1. **Keyboard**: `Ctrl+C`
2. **Config**: Set `trading.enabled: false` in `live_trading.yaml`
3. **Dashboard**: Manually close positions in Topstep

## 📋 Pre-Flight Checklist

Before each trading session:

- [ ] Verify `.env` has all credentials
- [ ] Check `live_trading.yaml` settings
- [ ] Confirm model is latest (`runs/mid2022_*/walkforward/...`)
- [ ] Review risk limits in `risk.yaml`
- [ ] Create `logs/` directory: `mkdir -p ml_intraday_v3/logs`
- [ ] Check market hours (RTH: 8:30 AM - 3:00 PM CT)

## 🎯 Expected Performance

Based on backtest results:

| Metric | Value |
|--------|-------|
| Walk-forward PnL | $26,311 (14 windows) |
| Holdout PnL | $18,816 (Oct-Dec 2025) |
| Topstep pass rate | 100% (10,000 simulations) |
| Average daily trades | ~30 |
| Win rate | ~60% |
| Max drawdown | <$648 |

**Note**: Live performance may vary due to:
- Execution slippage
- Market conditions
- Data quality
- Fill rates

## 🔧 Troubleshooting

### "No model bundle found"
```bash
# Verify model exists
ls runs/mid2022_*/walkforward/bar_size=1m/window_*/model_bundle.pkl
```

### "Data connection failed"
```bash
# Check Databento API key
echo $DATABENTO_API_KEY
# Reload .env
source .env
```

### "API connection failed"
```bash
# Test connection
cd "ml_intraday_v3"
python -c "from dotenv import load_dotenv; load_dotenv('../.env'); import sys; sys.path.insert(0, '..'); from core.projectx_client import ProjectXClient; client = ProjectXClient(); print(client.get_account_state())"
```

### Session token expired
1. Log into Topstep dashboard
2. Navigate to API settings
3. Regenerate session token
4. Update `TOPSTEPX_SESSION_TOKEN` in `.env`

## 📖 Additional Resources

- **Full checklist**: `ml_intraday_v3/LIVE_TRADING_CHECKLIST.md`
- **Risk config**: `ml_intraday_v3/configs/risk.yaml`
- **Live config**: `ml_intraday_v3/configs/live_trading.yaml`
- **Topstep docs**: https://www.topstepx.com/support

## ⚡ Quick Commands

```bash
# Dry run (no execution)
cd ml_intraday_v3 && PYTHONPATH="." python live_trading/live_runner.py --dry-run

# Paper trading (with execution)
cd ml_intraday_v3 && PYTHONPATH="." python live_trading/live_runner.py

# Test API connection
python -c "from dotenv import load_dotenv; load_dotenv('.env'); import sys; sys.path.insert(0, '.'); from core.projectx_client import ProjectXClient; c = ProjectXClient(); print('✓ Connected:', c.get_account_state())"

# View latest trade log
ls -lt ml_intraday_v3/logs/trades_*.csv | head -1 | awk '{print $9}' | xargs cat

# Create logs directory
mkdir -p ml_intraday_v3/logs
```

## 🎓 Learning Path

**Week 1**: Dry run mode
- Understand signal generation
- Verify model predictions
- Watch feature calculation

**Week 2**: Paper trading
- Execute on paper account
- Monitor P&L
- Test risk limits

**Week 3+**: Extended paper trading
- Build confidence
- Verify consistency
- Compare to backtest

**When ready**: Real money (start with 1 contract)

---

## ✅ Ready to Trade!

Your system is configured and tested. When you're ready:

```bash
cd "ml_intraday_v3"
PYTHONPATH="." python live_trading/live_runner.py --dry-run
```

Good luck! 🚀
