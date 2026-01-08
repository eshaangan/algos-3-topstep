# Live Trading Startup Checklist

## Pre-Trading Checklist

**CRITICAL: Always start with paper trading before going live!**

### 1. Environment Setup

- [ ] Verify `.env` file contains all required credentials:
  - [ ] `DATABENTO_API_KEY` - Data feed
  - [ ] `TOPSTEPX_USERNAME` - Topstep account email
  - [ ] `TOPSTEPX_PROJECTX_API_KEY` - API key
  - [ ] `TOPSTEPX_ACCOUNT_ID` - Account ID (150k paper account: 15390514)
  - [ ] `TOPSTEPX_SESSION_TOKEN` - Valid session token
  - [ ] `TOPSTEPX_CONTRACT_ID` - Current MES contract (e.g., CON.F.US.EP.Z25)
  - [ ] `TOPSTEPX_PROJECTX_BASE_URL` - https://api.topstepx.com

- [ ] Verify Python environment:
  ```bash
  cd "ml_intraday_v3"
  python --version  # Should be 3.11+
  pip install -r requirements-mlv3.txt
  ```

### 2. Configuration Validation

- [ ] Review `configs/live_trading.yaml`:
  - [ ] `trading.environment` = `"paper"` (for testing)
  - [ ] `trading.dry_run` = `false` (to actually execute on paper)
  - [ ] `signals.primary_threshold` = `0.10` (matches backtest)
  - [ ] `positions.contracts_per_trade` = `1`
  - [ ] `positions.max_concurrent` = `5`
  - [ ] `startup.require_manual_confirmation` = `true`

- [ ] Review `configs/risk.yaml`:
  - [ ] `topstep.starting_balance` = `50000` (treat 150k as 50k)
  - [ ] `daily_loss_limit.max_daily_loss` = `2000`
  - [ ] `trailing_drawdown.max_drawdown` = `2500`
  - [ ] `position_limits.max_concurrent_positions` = `5`
  - [ ] `intraday_controls.max_trades_per_day` = `20`

- [ ] Review `configs/execution_spec.yaml`:
  - [ ] `instrument.symbol` = `"MES"`
  - [ ] `costs.commission_per_contract` = `0.62` (your broker's commission)
  - [ ] `session_rules.allowed_sessions` - Verify RTH hours

### 3. Model Validation

- [ ] Verify trained model exists:
  ```bash
  ls -la runs/mid2022_*/walkforward/bar_size=1m/window_*/model_bundle.pkl
  ```

- [ ] Check latest model performance:
  ```bash
  cat runs/mid2022_*/walkforward/bar_size=1m/summary.json
  cat runs/mid2022_*/holdout_results.json
  ```

- [ ] Expected results:
  - [ ] Walk-forward total PnL: ~$26,311 (14 windows)
  - [ ] Holdout PnL: ~$18,816 (Oct-Dec 2025)
  - [ ] Topstep 50K pass rate: 100%

### 4. API Connection Tests

- [ ] Test Databento connection:
  ```python
  import databento as db
  import os
  client = db.Historical(key=os.getenv("DATABENTO_API_KEY"))
  # Should not raise error
  ```

- [ ] Test Topstep connection (dry run first):
  ```bash
  cd ml_intraday_v3
  PYTHONPATH="." python -c "
  import os
  import sys
  sys.path.insert(0, '..')
  from core.projectx_client import ProjectXClient

  client = ProjectXClient(
      base_url=os.getenv('TOPSTEPX_PROJECTX_BASE_URL'),
      username=os.getenv('TOPSTEPX_USERNAME'),
      api_key=os.getenv('TOPSTEPX_PROJECTX_API_KEY'),
      account_id=os.getenv('TOPSTEPX_ACCOUNT_ID'),
      contract_id=os.getenv('TOPSTEPX_CONTRACT_ID'),
  )

  account = client.get_account()
  print(f'Account: {account.account_id}, Equity: {account.equity}')
  "
  ```

### 5. Directory Structure

- [ ] Create logs directory:
  ```bash
  mkdir -p ml_intraday_v3/logs
  ```

- [ ] Verify directory structure:
  ```
  ml_intraday_v3/
  ├── configs/
  │   ├── live_trading.yaml
  │   ├── risk.yaml
  │   ├── execution_spec.yaml
  │   └── ...
  ├── live_trading/
  │   ├── __init__.py
  │   ├── data_fetcher.py
  │   ├── feature_generator.py
  │   ├── model_predictor.py
  │   ├── execution_engine.py
  │   └── live_runner.py
  ├── logs/          # Create this
  └── runs/
      └── mid2022_*/  # Latest trained model
  ```

## Starting Live Trading

### Phase 1: Dry Run (No Execution)

**Purpose**: Verify signal generation without executing trades

```bash
cd ml_intraday_v3
PYTHONPATH="." python live_trading/live_runner.py --dry-run
```

**What to watch**:
- [ ] Data connection successful
- [ ] Model loads without errors
- [ ] Features calculate correctly (no NaN/inf warnings)
- [ ] Signals generate with reasonable frequency
- [ ] No crashes or exceptions

**Expected**: Should generate signals but NOT execute any trades.

**Duration**: Run for 1-2 hours during RTH

### Phase 2: Paper Trading (Actual Execution on Paper Account)

**Purpose**: Execute real trades on paper account to test execution logic

1. [ ] Update `configs/live_trading.yaml`:
   ```yaml
   trading:
     environment: "paper"
     dry_run: false  # ENABLE execution
   ```

2. [ ] Run with manual confirmation:
   ```bash
   cd ml_intraday_v3
   PYTHONPATH="." python live_trading/live_runner.py
   ```

3. [ ] Startup checks should all pass:
   ```
   ✓ data_connection: PASS
   ✓ api_connection: PASS
   ✓ model_loaded: PASS
   ✓ risk_config_valid: PASS
   ```

4. [ ] Review confirmation prompt:
   ```
   Environment: paper
   Dry Run: False
   Account: 15390514

   Start trading? (type 'yes' to confirm):
   ```

5. [ ] Type `yes` to start

**What to monitor**:
- [ ] Trades execute successfully
- [ ] Orders appear in Topstep dashboard
- [ ] Risk limits enforced correctly
- [ ] Position tracking accurate
- [ ] Trade log saved to `logs/`

**Safety checks every 30 minutes**:
- [ ] Check equity in Topstep dashboard
- [ ] Verify daily loss < $2,000
- [ ] Verify total drawdown < $2,500
- [ ] Review `logs/trades_*.csv` for anomalies

**Duration**: Run for 1 full trading day (6.5 hours RTH)

### Phase 3: Extended Paper Trading

Before going live with real money:

- [ ] Run paper trading for at least 1 week
- [ ] Verify all days are profitable or within acceptable loss limits
- [ ] No risk breaches
- [ ] No execution errors
- [ ] Position limits working correctly

**Track these metrics**:
- Total P&L
- Win rate
- Max drawdown
- Number of trades
- Average trade size
- Largest win/loss

## Emergency Stop Procedures

### Method 1: Keyboard Interrupt

Press `Ctrl+C` in terminal. The system will:
- Stop generating new signals
- Flatten all open positions
- Save trade log
- Shutdown gracefully

### Method 2: Kill Switch

Update `configs/live_trading.yaml`:
```yaml
trading:
  enabled: false
```

System will halt on next update cycle.

### Method 3: Manual Flatten via Topstep

1. Log into Topstep dashboard
2. Go to Positions
3. Manually close all open positions

## Risk Monitoring

### Automated Risk Gates

The system will automatically halt trading if:
- Daily loss ≥ $2,000
- Trailing drawdown ≥ $2,500
- 5 consecutive losses
- Max 20 trades per day reached
- Equity drops below $40,000 (80% floor)
- API connection lost
- Feature quality issues detected

### Manual Monitoring

Check every 30 minutes:
- [ ] Current equity
- [ ] Daily P&L
- [ ] Open positions
- [ ] Trade log for errors

## Troubleshooting

### Issue: "No model bundle found"
**Solution**: Verify latest walk-forward run exists:
```bash
ls runs/mid2022_*/walkforward/bar_size=1m/window_*/model_bundle.pkl
```

### Issue: "Data connection failed"
**Solution**: Check Databento API key:
```bash
echo $DATABENTO_API_KEY
```

### Issue: "API connection failed"
**Solution**: Check Topstep credentials and session token:
```bash
echo $TOPSTEPX_SESSION_TOKEN
# Token may have expired - re-authenticate via Topstep dashboard
```

### Issue: "Feature quality issues"
**Solution**:
- Check if there's sufficient historical data
- Verify market is open (not pre/post market)
- Check for data feed issues

### Issue: "Trade rejected: max_concurrent_positions"
**Expected behavior**: System limits to 5 concurrent positions as configured.

### Issue: "Trade rejected: risk_daily_loss"
**Expected behavior**: Daily loss limit reached. Trading halted for the day.

## Post-Trading Review

After each trading day:

1. [ ] Review trade log: `logs/trades_YYYYMMDD_HHMMSS.csv`
2. [ ] Calculate daily metrics:
   - Total P&L
   - Win rate
   - Number of trades
   - Largest win/loss
3. [ ] Check for any errors or warnings in logs
4. [ ] Verify all positions were closed by session end
5. [ ] Compare actual execution vs expected (backtest results)

## When to Go Live with Real Money

**DO NOT go live until**:
- [ ] Minimum 1 week successful paper trading
- [ ] All emergency procedures tested
- [ ] Comfortable with system operation
- [ ] Risk limits thoroughly tested
- [ ] No execution errors observed
- [ ] Performance matches backtest expectations

**Additional considerations**:
- Start with minimum position size (1 contract)
- Run during low-volatility periods first
- Have manual override ready
- Monitor closely for first week

## Contact Information

**For technical issues**:
- Review logs in `ml_intraday_v3/logs/`
- Check Topstep API status
- Verify Databento data feed status

**For trading issues**:
- Topstep Support: https://www.topstepx.com/support
- Review risk.yaml settings
- Check execution_spec.yaml parameters

---

**REMEMBER**:
- Always treat the 150k paper account as a 50k account (mentally)
- Risk management is paramount
- When in doubt, STOP TRADING and review
- Paper test any configuration changes before going live
