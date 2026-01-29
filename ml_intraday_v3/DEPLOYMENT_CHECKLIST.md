# Model Deployment Checklist

**Version**: 1.0  
**Date**: 2026-01-25  
**Model**: Balanced V3 Bidirectional Trading Model  
**Target**: Topstep 50k Combine

---

## Pre-Deployment Validation

### Model Capability Tests

- [ ] **Run capability validation script**
  ```bash
  python ml_intraday_v3/validate_model_capabilities.py \
      --model-path runs/balanced_v3_q1_2024_q4_2025/walkforward/bar_size=5m/window_0/model_bundle.pkl
  ```

- [ ] **Synthetic bullish features → predicts LONG (side=1)**
  - Model responds correctly to bullish market conditions
  - EV_long > EV_short for bullish features

- [ ] **Synthetic bearish features → predicts SHORT (side=-1)**
  - Model responds correctly to bearish market conditions
  - EV_short > EV_long for bearish features

- [ ] **EV_short > 0 for bearish scenarios**
  - SHORT predictions have positive expected value
  - Model is not structurally biased against SHORT trades

- [ ] **EV_long > 0 for bullish scenarios**
  - LONG predictions have positive expected value
  - Model recognizes profitable LONG opportunities

- [ ] **Real data: At least 30% SHORT predictions on balanced test set**
  - Model actively predicts both directions
  - Not stuck in single-direction mode

### Multi-Period Backtest Validation

- [ ] **Run multi-period backtest**
  ```bash
  python ml_intraday_v3/test_directional_fix.py
  ```

- [ ] **Q1 2024 (Bearish): Win rate > 50%**
  - Date range: 2024-02-01 to 2024-03-31
  - Market condition: Pullback/correction
  - Expected: More SHORT trades, consistent profitability

- [ ] **Q3 2024 (Volatile): Win rate > 50%**
  - Date range: 2024-08-01 to 2024-09-30
  - Market condition: High volatility, choppy
  - Expected: Balanced LONG/SHORT, robust performance

- [ ] **Dec 2025 (Bullish): Win rate > 50%**
  - Date range: 2025-12-01 to 2025-12-31
  - Market condition: Strong uptrend (held-out test)
  - Expected: More LONG trades, strong performance

- [ ] **LONG/SHORT distribution reasonable (30-70% each side)**
  - Not 100% biased to one direction
  - Distribution varies appropriately by regime

- [ ] **Sharpe ratio > 1.0 across all periods**
  - Risk-adjusted returns are acceptable
  - Consistent performance quality

- [ ] **Max drawdown < $2,000 on any period**
  - Within Topstep trailing drawdown tolerance
  - Risk management is effective

- [ ] **Average trade P&L > $25**
  - Sufficient edge per trade
  - Transaction costs are manageable

### Topstep Rules Compliance

- [ ] **Run compliance validation**
  ```bash
  # After running backtest, validate the trades CSV
  python ml_intraday_v3/validate_topstep_compliance.py \
      --trades-csv logs/trades_YYYYMMDD_HHMMSS.csv \
      --account-size 50000
  ```

- [ ] **Daily loss limit: No day with loss > $1,000**
  - Critical rule - violation = disqualification
  - Check worst day P&L

- [ ] **Trailing drawdown: Never exceeds $2,500**
  - Critical rule - violation = disqualification
  - Monitor from both starting equity and highest equity

- [ ] **Consistency rule: Best day < 50% of total profit**
  - Critical rule - violation = disqualification
  - Only applies if profitable overall

- [ ] **Profit target: Achieves $3,000 profit**
  - Required to pass Combine
  - Monitor progress toward target

---

## Model Bundle Verification

### File Structure

- [ ] **Model bundle exists at expected location**
  ```
  runs/balanced_v3_q1_2024_q4_2025/walkforward/bar_size=5m/window_0/model_bundle.pkl
  ```

- [ ] **Bundle size reasonable (< 5MB)**
  - Large bundles may indicate overfitting or data leakage
  - Typical size: 0.5-2MB

### Bundle Contents

- [ ] **has_side_feature = True**
  ```python
  import joblib
  bundle = joblib.load('path/to/model_bundle.pkl')
  assert bundle.get('has_side_feature') == True
  ```

- [ ] **'side' in primary_feature_columns**
  ```python
  assert 'side' in bundle['primary_feature_columns']
  ```

- [ ] **primary_preprocessor exists**
  ```python
  assert bundle['primary_preprocessor'] is not None
  ```

- [ ] **Metadata includes training statistics**
  - training_method
  - train_events count
  - train_long_pct, train_short_pct
  - test_auc, test_accuracy
  - training period dates

### Feature Verification

- [ ] **Feature count matches expected (34-35 features)**
  - 33-34 market features
  - 1 'side' feature

- [ ] **No data leakage features present**
  - No exit_price, ret_gross, ret_net
  - No label_type, exit_reason
  - No future information

- [ ] **All required features present**
  - Returns: log_return_1, log_return_2, etc.
  - Volatility: atr_14, vol_20, vol_regime
  - Trend: ema_13, ema_21, sma_20, sma_30
  - Time: minute_of_day_sin, minute_of_day_cos, day_of_week
  - Side: side

---

## Risk Management Checks

### Position Limits

- [ ] **Max position size = 1 contract (MES)**
  - Topstep requirement for 50k account
  - Verify in execution_spec.yaml

- [ ] **No simultaneous LONG and SHORT positions**
  - One direction at a time
  - Flat before reversing

### Trading Hours

- [ ] **RTH only (9:30 AM - 4:00 PM ET)**
  - No overnight positions
  - No pre-market or after-hours trading

- [ ] **No trading on market holidays**
  - Check calendar integration
  - Verify holiday detection logic

### Stop/Target Management

- [ ] **Stops and targets set for every trade**
  - No naked positions
  - Risk defined on entry

- [ ] **Stop distance appropriate (1.5x ATR)**
  - Not too tight (whipsaws)
  - Not too wide (excessive risk)

- [ ] **Target distance appropriate (2.5x ATR)**
  - Realistic profit expectation
  - Aligns with win rate

---

## Paper Trading Validation (1 Week Minimum)

### Setup

- [ ] **Configure dry_run mode**
  ```python
  # In live_runner.py or replay.py
  dry_run = True
  ```

- [ ] **Connect to paper trading account**
  - Use Topstep demo account
  - Or use broker paper trading

- [ ] **Start paper trading session**
  ```bash
  python ml_intraday_v3/live_trading/live_runner.py \
      --config configs/live_trading.yaml \
      --dry-run
  ```

### Daily Monitoring (5 Trading Days)

- [ ] **Day 1: Monitor LONG/SHORT distribution**
  - Record: LONG %, SHORT %
  - Expected: 30-70% each side (regime-dependent)

- [ ] **Day 2: Verify fills at expected prices**
  - Check slippage vs backtest assumptions
  - Validate execution model accuracy

- [ ] **Day 3: Check stop/target placement**
  - Stops triggered appropriately
  - Targets realistic and achievable

- [ ] **Day 4: Review daily P&L reports**
  - Within Topstep rules
  - Consistent with backtest expectations

- [ ] **Day 5: Validate feature quality**
  - No NaN/inf in live features
  - Features match backtest calculations

### Week-End Review

- [ ] **Overall paper trading results acceptable**
  - Win rate within 10% of backtest
  - P&L profile similar to backtest
  - No unexpected issues

- [ ] **LONG/SHORT balance maintained**
  - Not stuck in one direction
  - Responds to regime changes

- [ ] **Risk rules never violated**
  - No daily loss limit breaches
  - No trailing drawdown violations

---

## Technical Validation

### Feature Quality

- [ ] **No NaN/inf in predictions on test data**
  - Run validation on full test set
  - Check feature generation pipeline

- [ ] **Feature quality checks working**
  - Warmup period sufficient (50 bars)
  - Quality flags properly set

- [ ] **Feature calculations match training**
  - Live features = backtest features
  - No implementation drift

### Prediction Pipeline

- [ ] **Model loads successfully**
  - No version incompatibility
  - All dependencies available

- [ ] **Predictions are deterministic**
  - Same input → same output
  - No randomness in inference

- [ ] **Prediction latency acceptable (< 100ms)**
  - Fast enough for live trading
  - No timeouts or delays

### Execution Pipeline

- [ ] **Orders execute correctly**
  - Market orders fill properly
  - Stop/target orders placed correctly

- [ ] **Position tracking accurate**
  - Open positions tracked
  - P&L calculated correctly

- [ ] **Error handling robust**
  - Graceful degradation
  - Logging and alerts working

---

## Documentation

### Model Documentation

- [ ] **Training summary document exists**
  - Location: `ml_intraday_v3/RETRAINING_COMPLETE_SUMMARY.md`
  - Contains: Training method, data periods, results

- [ ] **Validation reports saved**
  - Capability validation results
  - Backtest results by period
  - Compliance validation results

- [ ] **Feature importance documented**
  - Top features identified
  - Feature stability verified

### Deployment Documentation

- [ ] **Deployment instructions written**
  - How to deploy model
  - How to start live trading
  - How to monitor performance

- [ ] **Rollback plan documented**
  - How to revert to previous model
  - Backup model location
  - Emergency stop procedures

- [ ] **Monitoring dashboards configured**
  - Real-time P&L tracking
  - Trade log viewer
  - Alert system active

---

## Final Go-Live Approval

### Sign-Off Requirements

- [ ] **All critical validation checks PASS**
  - Model capabilities verified
  - Backtests successful
  - Compliance validated

- [ ] **Paper trading results acceptable**
  - 5+ days of consistent performance
  - No unexpected issues
  - Team review completed

- [ ] **Risk manager approval**
  - Topstep rules compliance verified
  - Risk management appropriate
  - Position sizing correct

- [ ] **Technical review approval**
  - Code quality acceptable
  - No known bugs
  - Monitoring in place

### Backup Plan

- [ ] **Previous model available for rollback**
  - Location documented
  - Tested and verified

- [ ] **Emergency stop procedures documented**
  - How to stop trading immediately
  - Who to contact
  - Incident response plan

- [ ] **Recovery procedures documented**
  - How to resume after stop
  - Data integrity checks
  - System health validation

---

## Go-Live Execution

### Pre-Launch

- [ ] **Final system health check**
  - All services running
  - Database connections active
  - API connections verified

- [ ] **Load production model**
  ```bash
  cp runs/balanced_v3_q1_2024_q4_2025/walkforward/bar_size=5m/window_0/model_bundle.pkl \
     ml_intraday_v3/models/saved/model_bundle.pkl
  ```

- [ ] **Disable dry_run mode**
  ```python
  dry_run = False
  ```

### Launch

- [ ] **Start live trading system**
  ```bash
  python ml_intraday_v3/live_trading/live_runner.py \
      --config configs/live_trading.yaml
  ```

- [ ] **Verify first trade**
  - Executed correctly
  - P&L tracking working
  - Stops/targets placed

### Post-Launch (First Week)

- [ ] **Daily monitoring**
  - Review all trades
  - Check rule compliance
  - Monitor LONG/SHORT balance

- [ ] **Weekly review**
  - Performance vs backtest
  - Any issues or concerns
  - Adjustments needed

---

## Success Criteria Summary

**Model must meet ALL of the following:**

✅ **Capability**: Can predict both LONG and SHORT profitably  
✅ **Performance**: Win rate > 50% across all market regimes  
✅ **Compliance**: Never violates Topstep rules (daily loss, drawdown, consistency)  
✅ **Stability**: Consistent results in paper trading for 5+ days  
✅ **Approval**: Sign-off from risk manager and technical team

**If ANY criterion fails**: DO NOT deploy. Fix issues and re-validate.

---

## Contact Information

**Risk Manager**: [Name]  
**Technical Lead**: [Name]  
**Topstep Support**: support@topstep.com  
**Emergency Stop**: [Procedure/Contact]

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-01-25 | Initial checklist for balanced V3 model | System |

---

**IMPORTANT**: This checklist must be completed in full before deploying any model to live trading. Skipping steps may result in financial loss or Topstep disqualification.
