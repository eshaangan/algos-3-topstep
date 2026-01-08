# Topstep 50K Paper Trading Preparation Plan

**Date**: January 7, 2026
**Goal**: Prepare and validate the existing live trading infrastructure for Topstep 50K Combine paper trading

---

## Current Situation

### What Exists
- ✅ Complete live trading infrastructure in `ml_intraday_v3/live_trading/`
- ✅ ProjectX API client integration (`core/projectx_client.py`)
- ✅ Monitoring and alerting systems
- ✅ Trained models in `runs/v3_2022_5m/`
- ✅ User has API keys and environment configured
- ✅ Documentation and checklists

### Critical Issues Found
- ❌ **Daily loss limit**: Set to $2,000, should be $1,000 (Topstep rule)
- ❌ **Max contracts per position**: Set to 2, should be 1 (Topstep rule)
- ⚠️ **Flatten time mismatch**: Backtest uses 15:55, live uses 14:45
- ⚠️ **Contract ID**: CON.F.US.EP.Z25 (Dec 2025) - may need updating

---

## Implementation Plan

### Phase 1: Fix Critical Configuration Issues

#### 1.1 Fix Daily Loss Limit
**File**: `ml_intraday_v3/configs/risk.yaml`
**Line**: 20
**Change**:
```yaml
# Before:
max_daily_loss: 2000

# After:
max_daily_loss: 1000
```

#### 1.2 Fix Max Contracts Per Position
**File**: `ml_intraday_v3/configs/execution_spec.yaml`
**Line**: 38
**Change**:
```yaml
# Before:
max_contracts_per_position: 2

# After:
max_contracts_per_position: 1
```

#### 1.3 Align Flatten Times
**Decision**: ✅ Use **15:55 CT** (standard, matches backtest)

**File to Update**: `ml_intraday_v3/configs/live_trading.yaml`
**Line**: ~107
**Change**:
```yaml
# Before:
session_close_time: "15:00"
flatten_buffer_minutes: 15  # Flattens at 14:45

# After:
session_close_time: "16:00"  # Session ends at 16:00 CT
flatten_buffer_minutes: 5    # Flattens at 15:55 CT
```

**Rationale**: Matches backtest behavior (15:55), allows full RTH trading

#### 1.4 Update Contract ID (EXPIRED)
**Files**: `.env` and `ml_intraday_v3/configs/live_trading.yaml`
**Current**: `CON.F.US.EP.Z25` (MES December 2025 - EXPIRED)
**New**: `CON.F.US.EP.H26` (MES March 2026)

**Changes**:
1. `.env` line 10:
```bash
# Before:
TOPSTEPX_CONTRACT_ID='CON.F.US.EP.Z25'

# After:
TOPSTEPX_CONTRACT_ID='CON.F.US.EP.H26'
```

2. `live_trading.yaml` line ~48:
```yaml
# Before:
contract_id: "CON.F.US.EP.Z25"

# After:
contract_id: "CON.F.US.EP.H26"
```

**Also Update**: Line 23 in `.env` (duplicate)
```bash
CONTRACT_ID='CON.F.US.EP.H26'
```

---

### Phase 2: Environment Variable Verification

#### 2.1 Required Environment Variables

**Data Feed**:
```bash
DATABENTO_API_KEY=<your_databento_api_key>
```

**Topstep ProjectX API**:
```bash
TOPSTEPX_USERNAME=<your_topstep_email>
TOPSTEPX_PROJECTX_API_KEY=<your_projectx_api_key>
TOPSTEPX_PROJECTX_BASE_URL=https://api.topstepx.com
TOPSTEPX_ACCOUNT_ID=<your_paper_account_id>
TOPSTEPX_CONTRACT_ID=<current_contract_id>
TOPSTEPX_SESSION_TOKEN=<jwt_session_token>
```

#### 2.2 Verification Steps
1. Check if `.env` file exists in project root
2. Verify all required variables are present
3. Test API connection with ProjectX client
4. Test Databento data feed connection

---

### Phase 3: Model Bundle Preparation

#### 3.1 Verify Trained Models Exist
**Location**: `runs/v3_2022_5m/bar_size=5m/`

**Check for**:
- `training/purged_kfold/fold_0/` through `fold_5/`
- Model files (`.pkl` or `.joblib`)
- Preprocessor/scaler artifacts
- Feature schema

#### 3.2 Create Model Bundle (if needed)
**Script**: `ml_intraday_v3/create_live_model_bundle.py`

**Command**:
```bash
python3 -m ml_intraday_v3.create_live_model_bundle \
    --run-dir runs/v3_2022_5m \
    --bar-size 5m \
    --output-dir live_models/v3_2022_5m_prod
```

**Output**:
- `model_bundle.pkl` with model, preprocessor, thresholds, feature schema

#### 3.3 Update Bar Size to Match Training
**Decision**: ✅ Use **5m bars** in live trading (matches training)

**File**: `ml_intraday_v3/configs/live_trading.yaml`
**Line**: ~30
**Change**:
```yaml
# Before:
bar_size: "1m"

# After:
bar_size: "5m"
```

**Impact**:
- Matches training data (99.8% feature completeness vs 5.9% for 1m)
- More reliable predictions (no train/test mismatch)
- Fewer signals (expect 10-20 trades/day, not 50-100)
- Better feature quality (less NaN issues)

---

### Phase 4: Pre-Trading Testing

#### 4.1 API Connection Test & Account Verification
**Purpose**: Test both account IDs to determine which is active

**Script**: Create quick test script

```python
import os
from core.projectx_client import ProjectXClient

# Test Account A (15266746)
print("Testing Account A: 15266746")
os.environ['TOPSTEPX_ACCOUNT_ID'] = '15266746'
try:
    client_a = ProjectXClient()
    account_a = client_a.get_account_state()
    print(f"✓ Account A ACTIVE: ${account_a.equity:,.2f} equity")
except Exception as e:
    print(f"✗ Account A FAILED: {e}")

# Test Account B (15390514)
print("\nTesting Account B: 15390514")
os.environ['TOPSTEPX_ACCOUNT_ID'] = '15390514'
try:
    client_b = ProjectXClient()
    account_b = client_b.get_account_state()
    print(f"✓ Account B ACTIVE: ${account_b.equity:,.2f} equity")
except Exception as e:
    print(f"✗ Account B FAILED: {e}")

# Use whichever account works for live trading
```

**Expected Outcome**: One account will connect successfully, use that ID in .env

#### 4.2 Dry Run Test (No Execution)
**Duration**: 1-2 hours
**Purpose**: Validate data feed, features, predictions WITHOUT executing trades

**Command**:
```bash
cd ml_intraday_v3
PYTHONPATH="." python live_trading/live_runner.py --dry-run
```

**Monitor**:
- Data fetching works
- Features calculate without NaNs
- Model predictions generate
- Signals logged but not executed

#### 4.3 Paper Trading Test (With Execution)
**Duration**: 1 full trading day
**Purpose**: Execute real trades in paper account

**Command**:
```bash
cd ml_intraday_v3
PYTHONPATH="." python live_trading/live_runner.py
```

**Monitor**:
- Orders execute successfully
- Risk gates enforce correctly
- Positions tracked accurately
- Flatten at session close works
- Metrics logged properly

---

### Phase 5: Go-Live Checklist

#### 5.1 Pre-Session Checks
- [ ] All config files corrected (daily loss, position size, flatten time)
- [ ] Environment variables loaded and verified
- [ ] API connection test passed
- [ ] Model bundle exists and loads successfully
- [ ] Dry run completed without errors
- [ ] Paper trading test completed (1+ days)
- [ ] Contract ID is current (not expired)
- [ ] Databento API key active and has credits
- [ ] Topstep account funded and active

#### 5.2 Session Startup
- [ ] Review `ml_intraday_v3/START_TRADING.md`
- [ ] Review `ml_intraday_v3/LIVE_TRADING_CHECKLIST.md`
- [ ] Check market hours (RTH: 08:30-15:00 CT)
- [ ] Start with manual confirmation enabled
- [ ] Monitor first 30 minutes closely

#### 5.3 During Session Monitoring
- [ ] Dashboard updates every 10 seconds
- [ ] Alert system active (logs to `logs/alerts_*.log`)
- [ ] Equity tracked (starting: $50,000 paper balance)
- [ ] Daily loss monitored (limit: $1,000)
- [ ] Drawdown from HWM monitored (limit: $2,500)
- [ ] Max 1 contract per position enforced
- [ ] Max 5 concurrent positions enforced

#### 5.4 End-of-Session Review
- [ ] All positions flattened (auto at 14:45 or 15:55)
- [ ] Review trade log: `logs/trades_*.csv`
- [ ] Review metrics: `logs/metrics_*.csv`
- [ ] Check P&L reconciliation with Topstep account
- [ ] Review any alerts or warnings

---

## Critical Files

### Configuration Files
| File | Purpose | Changes Needed |
|------|---------|----------------|
| `configs/risk.yaml` | Risk limits | Fix daily loss to $1,000 |
| `configs/execution_spec.yaml` | Position sizing | Fix max contracts to 1 |
| `configs/backtest.yaml` | Backtest settings | Align flatten time |
| `configs/live_trading.yaml` | Live trading settings | Align flatten time, verify contract ID |

### Code Files (No Changes Needed)
| File | Purpose |
|------|---------|
| `live_trading/live_runner.py` | Main orchestrator |
| `live_trading/execution_engine.py` | Order execution + risk enforcement |
| `live_trading/data_fetcher.py` | Real-time data from Databento |
| `live_trading/feature_generator.py` | Live feature calculation |
| `live_trading/model_predictor.py` | Model inference |
| `core/projectx_client.py` | Topstep API client |
| `backtesting_v3/risk.py` | Risk manager (used in live) |

### Documentation
| File | Purpose |
|------|---------|
| `START_TRADING.md` | Step-by-step startup guide |
| `LIVE_TRADING_CHECKLIST.md` | Pre-flight checklist |
| `MONITORING_GUIDE.md` | Dashboard and alerts guide |
| `PAPER_TRADING_READINESS.md` | Validation roadmap |

---

## User Decisions (Answered)

1. **Flatten Time**: ✅ **15:55 CT** (standard, 5 min before session end)
   - Matches backtest configuration
   - Allows full RTH trading
   - **Action**: Update `live_trading.yaml` line 107 to match `backtest.yaml`

2. **Bar Size**: ✅ **Use 5m bars in live trading**
   - Matches training data (99.8% feature completeness)
   - More reliable predictions
   - **Action**: Update `live_trading.yaml` to use 5m instead of 1m

3. **Environment Variables**: ✅ **All present in .env file**
   - Databento API key ✓
   - Topstep credentials ✓
   - **Issue Found**: Contract ID is `CON.F.US.EP.Z25` (Dec 2025 - EXPIRED)
   - **Fix**: Update to `CON.F.US.EP.H26` (March 2026)

4. **Account ID**: ⚠️ **CLARIFICATION NEEDED**
   - Two account IDs found in .env:
     - `TOPSTEPX_ACCOUNT_ID=15266746`
     - `ACCOUNT_ID_TO_WATCH=15390514`
   - **Question**: Which is your active Topstep paper account?

---

## Risk Considerations

### Configuration Risks (Being Fixed)
- ✅ Daily loss limit being corrected to $1,000
- ✅ Position size being corrected to 1 contract
- ✅ Flatten time being aligned

### Operational Risks
- **Data Feed Failure**: Databento outage → Use redundant source or manual halt
- **API Connection Loss**: ProjectX disconnect → Auto-flatten positions
- **Feature NaNs**: Data quality issues → Skip trading that bar
- **Model Prediction Errors**: Exception handling → Log error, skip signal

### Market Risks
- **Slippage**: Market orders may have worse fills than backtest → Monitor actual vs expected
- **Low Liquidity**: Near close or overnight → Already enforced via session hours
- **Gap Risk**: Overnight positions → Already enforced (flatten at close)

---

## Success Metrics (First Week)

### Daily Targets
- **Win Rate**: 50-70% (backtest: ~59%)
- **Average Trade P&L**: $25-50 (backtest varies by fold)
- **Trade Frequency**: 10-20 trades/day (depends on threshold)
- **Risk Breaches**: 0 (must be zero)
- **Technical Failures**: <5% (signal generation issues)

### Weekly Target (Topstep Combine Progress)
- **Profit Target**: +$3,000 (to pass combine)
- **Daily Loss**: Never exceed $1,000
- **Trailing DD**: Never exceed $2,500
- **Consistency**: No single day > 50% of total profit
- **Time**: Ideally within 30 days (20 trading days)

---

## Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Config Fixes | 10 min | Fix risk.yaml, execution_spec.yaml, align flatten times |
| Env Verification | 15 min | Check .env, test API connections |
| Model Bundle | 5 min | Verify or create bundle |
| Dry Run Test | 1-2 hours | Test signal generation without execution |
| Paper Test | 1 day | First full trading day with execution |
| Go Live | Ongoing | Monitor and iterate |

**Total Time to Start Paper Trading**: ~2-3 hours (if no model retraining needed)

---

## Emergency Procedures

### Stop Trading Immediately If:
1. Daily loss approaches $800 (80% of $1,000 limit)
2. Trailing DD approaches $2,000 (80% of $2,500 limit)
3. 3 consecutive technical failures (order rejections, data feed issues)
4. Risk gates not enforcing properly

### Emergency Stop Methods
1. **Ctrl+C** in terminal → Triggers graceful shutdown with position flattening
2. **Manual flatten** via Topstep dashboard
3. **Kill process**: Find PID and kill

### Post-Emergency Actions
1. Flatten all positions immediately
2. Review logs for root cause
3. Fix issue before restarting
4. Document incident

---

## Configuration Changes Summary

### Files to Modify:

#### 1. `ml_intraday_v3/configs/risk.yaml` (line 20)
```yaml
max_daily_loss: 1000  # Change from 2000
```

#### 2. `ml_intraday_v3/configs/execution_spec.yaml` (line 38)
```yaml
max_contracts_per_position: 1  # Change from 2
```

#### 3. `ml_intraday_v3/configs/live_trading.yaml`
**Three changes:**
- Line ~30: `bar_size: "5m"` (change from 1m)
- Line ~48: `contract_id: "CON.F.US.EP.H26"` (change from Z25)
- Line ~107: `session_close_time: "16:00"` and `flatten_buffer_minutes: 5` (to flatten at 15:55)

#### 4. `.env` file
**Two changes:**
- Line 10: `TOPSTEPX_CONTRACT_ID='CON.F.US.EP.H26'` (change from Z25)
- Line 23: `CONTRACT_ID='CON.F.US.EP.H26'` (change from Z25)

**Account ID Verification** (to be tested):
- Test both account IDs to determine which is active:
  - Account A: 15266746 (TOPSTEPX_ACCOUNT_ID)
  - Account B: 15390514 (ACCOUNT_ID_TO_WATCH)
- Will test API connection with both to confirm active account

---

## Next Immediate Actions

1. **Test account IDs** (5 min - verify which account is active: 15266746 or 15390514)
2. **Fix all configurations** (30 min - all changes listed above)
3. **Verify contract ID** (CON.F.US.EP.H26) works with active account
4. **Run dry run** (1-2 hours - no execution, validate signals)
5. **Paper trade** (1 day - first live test)

**Total time to first paper trade**: ~3-4 hours (if no issues)
