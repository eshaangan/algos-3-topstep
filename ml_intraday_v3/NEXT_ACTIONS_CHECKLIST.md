# Next Actions Checklist

## ✅ Phase 1 Complete (Days 1-6)
- [x] Implement confidence filter (threshold 0.55)
- [x] Implement adaptive circuit breaker
- [x] Implement regime detector (KS test)
- [x] Enable volatility filter
- [x] Create validation tests
- [x] Validate model on Dec 2025 data
- [x] Analyze Jan 2026 failure
- [x] Calculate filter impact

**Status**: ✅ **COMPLETE** - All filters implemented and validated

---

## ⏳ Phase 2a: Filter Integration (Days 7-8)

### Day 7: Integrate Adaptive Circuit Breaker

**File to modify**: `ml_intraday_v3/live_trading/live_runner.py`

**Tasks**:
- [ ] Import `AdaptiveCircuitBreaker` from `monitoring.adaptive_circuit_breaker`
- [ ] Initialize circuit breaker at session start with parameters:
  - `consecutive_losses_limit=3`
  - `cooling_off_minutes=30`
  - `daily_loss_limit=-500.0`
  - `base_confidence_threshold=0.55`
- [ ] After each trade, call `check_and_adapt(trade_result, daily_pnl, current_time)`
- [ ] Handle circuit breaker actions:
  - `'continue'` → Trade normally
  - `'cooling_off'` → Skip trade, wait for cooling-off period to end
  - `'adapted'` → Log adaptation (higher threshold, smaller size)
  - `'stop_today'` → Stop all trading for the day
- [ ] Add logging for circuit breaker events
- [ ] Test in paper trading for 1 day

**Validation**:
```bash
# Test circuit breaker integration
python ml_intraday_v3/live_trading/live_runner.py --paper-trading --test-mode
# Should see circuit breaker trigger on consecutive losses
```

### Day 8: Integrate Regime Detector

**File to modify**: `ml_intraday_v3/live_trading/live_runner.py`

**Tasks**:
- [ ] Import `RegimeDetector` from `filters.regime_filter`
- [ ] Initialize regime detector at session start:
  - Load feature columns from model config
  - Fit on last 90 days of training data
  - Set `max_shifted_features_pct=0.30`
- [ ] Before generating each signal, call `detect_shift(current_features)`
- [ ] If regime shift detected (`is_safe=False`):
  - Log warning with shift percentage and shifted features
  - Skip signal generation
  - Continue monitoring (check every 100 bars)
- [ ] Add daily regime check summary
- [ ] Test in paper trading for 1 day

**Validation**:
```bash
# Test regime detector on Jan 2026 data
python ml_intraday_v3/experiments/test_regime_detector.py \
    --train-end 2025-12-31 \
    --test-start 2026-01-01 \
    --test-end 2026-01-31
# Should flag regime shift by Jan 5-7
```

**Integration reference**: See `ml_intraday_v3/FILTER_INTEGRATION_GUIDE.md` for detailed code examples

---

## ⏳ Phase 2b: Signal Quality Improvements (Days 9-18)

### Days 9-11: Entry Timing Optimization

**Goal**: Improve avg trade by $8-10

**Tasks**:
- [ ] Create `ml_intraday_v3/execution/entry_optimizer.py`
- [ ] Implement pullback detection:
  - Don't enter immediately when signal fires
  - Wait for 2-3 tick retracement
  - Set limit order at better price
- [ ] Add "entry zone" logic:
  - Calculate support/resistance levels
  - Only enter within 5 ticks of S/R
  - Don't chase runaway prices
- [ ] Add timeout (max 15 min wait for entry)
- [ ] Backtest on Dec 2025 data
- [ ] Validate improvement: avg trade should increase by $5-10

**Success criteria**: Avg trade $14.72 → $22-25

### Days 12-13: Dynamic Stop/Target Adjustment

**Goal**: Reduce avg loss by $3-5, improve risk/reward

**Tasks**:
- [ ] Create `ml_intraday_v3/execution/dynamic_stops.py`
- [ ] Calculate current volatility (20-bar ATR)
- [ ] Adjust stops based on volatility percentile:
  - Low vol (0-30%): 1.5x ATR stop
  - Normal vol (30-70%): 2.0x ATR stop
  - High vol (70-100%): 2.5x ATR stop
- [ ] Set targets at minimum 2:1 risk/reward
- [ ] Backtest on Dec 2025 data
- [ ] Validate improvement: avg loss should reduce by $3-5

**Success criteria**: Avg loss -$20 → -$15, better win rate

### Days 14-15: Tiered Position Sizing

**Goal**: +$200-300 total P&L

**Tasks**:
- [ ] Create `ml_intraday_v3/execution/tiered_sizing.py`
- [ ] Implement confidence-based sizing:
  - P > 0.65 (high confidence): 1.5x base size
  - P = 0.55-0.65 (medium): 1.0x base size
  - P = 0.50-0.55 (medium-low): 0.5x base size
  - P < 0.50 (low): 0x (don't trade)
- [ ] Add Topstep position limits check
- [ ] Add max risk per trade limit ($100)
- [ ] Backtest on Dec 2025 data
- [ ] Validate improvement: total P&L should increase $200-300

**Success criteria**: Total P&L +$200-300 from amplifying good trades

### Days 16-17: Feature Engineering

**Goal**: +5-10% win rate

**Tasks**:
- [ ] Create `ml_intraday_v3/features/volume_features.py`
- [ ] Add volume profile features:
  - Point of Control (POC)
  - Value Area High (VAH)
  - Value Area Low (VAL)
  - Volume-weighted price
- [ ] Add order flow features:
  - Bid/ask delta
  - Cumulative delta
  - Order flow imbalance
- [ ] Add time-of-day interactions:
  - `vol_20 * minute_of_day_sin`
  - `log_return_1 * hour_of_day`
- [ ] Test for feature leakage (purge/embargo)
- [ ] Retrain model with new features
- [ ] Validate on Dec 2025 out-of-sample data

**Success criteria**: Win rate 54.7% → 59-60%

### Day 18: Model Ensemble

**Goal**: +3-5% win rate, better regime adaptation

**Tasks**:
- [ ] Create `ml_intraday_v3/models/ensemble.py`
- [ ] Train 3 models:
  - Model A: 12-month lookback (long-term, stable)
  - Model B: 6-month lookback (medium-term)
  - Model C: 3-month lookback (short-term, adaptive)
- [ ] Implement weighted voting:
  - Weight C: 50% (most recent, most adaptive)
  - Weight B: 30% (medium-term)
  - Weight A: 20% (long-term stability)
- [ ] Test ensemble on Dec 2025 and Jan 2026
- [ ] Validate improvement: ensemble should work better in regime shifts

**Success criteria**:
- Dec 2025: Win rate 59-60% → 62-65%
- Jan 2026: Win rate 35.5% → 45-50% (better regime handling)

---

## ⏳ Phase 3: Paper Trading Validation (Days 19-25)

### Setup

**Tasks**:
- [ ] Enable all improvements in paper trading mode
- [ ] Set up daily monitoring dashboard
- [ ] Configure alerts (email/SMS for circuit breaker trips)
- [ ] Create daily P&L tracking spreadsheet

### Daily Monitoring (5-7 trading days)

**Track each day**:
- [ ] Total trades (target: 5-7/day)
- [ ] Win rate (target: ≥50%)
- [ ] Daily P&L (target: +$100-200)
- [ ] Circuit breaker triggers (max: 2/week)
- [ ] Regime detector status (should be stable)
- [ ] Max drawdown (max: -$500)

### GO/NO-GO Decision (After Day 5)

**GO Criteria** (ALL must pass):
- [ ] 5/7 days profitable (71% positive days)
- [ ] Overall win rate ≥ 50%
- [ ] Avg daily P&L ≥ +$100
- [ ] No day worse than -$500
- [ ] Circuit breaker trips ≤ 2 times
- [ ] Regime detector stable (no shift flags)

**If GO**: ✅ Proceed to Topstep combine
**If PARTIAL**: Extend to 10 days, adjust thresholds
**If NO-GO**: ❌ Revisit improvements, possible retrain

---

## ⏳ Phase 4: Topstep 50k Combine (Days 26-45)

### Pre-Flight Checklist

**Before starting combine**:
- [ ] All improvements implemented and validated
- [ ] Paper trading passed GO criteria
- [ ] Circuit breaker + regime detector active
- [ ] Daily P&L tracking ready
- [ ] Alerts configured (email/SMS)
- [ ] Risk limits verified:
  - Max daily loss: -$1,000 (Topstep rule)
  - Max trailing drawdown: -$2,500 (Topstep rule)
  - Circuit breaker at -$500 (internal rule)
- [ ] Position sizing confirmed (start 0.5 contracts)

### Week 1 (Days 1-5)

**Targets**:
- [ ] Daily target: $120-200
- [ ] Cumulative target: +$600-1,000
- [ ] Win rate: Maintain 50%+
- [ ] Position size: 0.5 contracts
- [ ] Trades/day: 5-7

**Daily checklist**:
- [ ] Pre-market: Check regime detector, review previous day
- [ ] During market: Take only high-confidence trades (P>0.55)
- [ ] Post-market: Log trades, calculate cumulative P&L
- [ ] Decision: If 4/5 days positive, scale to 0.75 contracts

### Week 2 (Days 6-10)

**Targets**:
- [ ] Daily target: $150-200
- [ ] Cumulative target: +$1,500-2,000 (halfway to $3,000)
- [ ] Win rate: Maintain 50%+
- [ ] Position size: 0.75 contracts
- [ ] Trades/day: 6-7

**Critical check**:
- [ ] If cumulative < $1,000 after 10 days → Re-evaluate strategy
- [ ] If on pace → Scale to 1.0 contracts

### Week 3 (Days 11-15)

**Targets**:
- [ ] Daily target: $150-200
- [ ] Cumulative target: +$2,500-3,000+ (finish strong)
- [ ] Win rate: Maintain 50%+
- [ ] Position size: 1.0 contracts
- [ ] Trades/day: 5-7

**Final push**:
- [ ] Once hit $2,500 → Reduce size to 0.5 (conservative finish)
- [ ] Target $50-100/day for final $500
- [ ] Avoid overtrading

### Week 4 (Days 16-20, if needed)

**Targets**:
- [ ] Conservative finish
- [ ] Position size: 0.5 contracts
- [ ] Daily target: $50-100
- [ ] Trades/day: 3-5 (quality only)

**Final check**:
- [ ] Reach $3,000 cumulative ✅
- [ ] No daily loss violations
- [ ] No trailing drawdown violations
- [ ] Minimum trade count met (typically 5-10 trades)

---

## ❌ Circuit Breaker Rules (NEVER BREAK)

1. [ ] **3 consecutive losses** → Pause 30 min, raise threshold
2. [ ] **-$500 daily loss** → STOP trading for the day
3. [ ] **Regime shift detected** → STOP until shift resolves
4. [ ] **Win rate < 30%** after 10 trades → STOP, diagnose

**If circuit breaker trips**:
- Do NOT override or disable
- Do NOT increase position size to "recover"
- Do NOT chase losses
- Accept the loss, stop for the day, review tomorrow

---

## 📊 Success Metrics

### Phase 2a (Filter Integration)
- [ ] Circuit breaker triggers correctly on test data
- [ ] Regime detector identifies Jan 2026 shift
- [ ] 1-2 days paper trading without issues

### Phase 2b (Signal Improvements)
- [ ] Entry timing: Avg trade +$8-10
- [ ] Dynamic stops: Avg loss -$3-5
- [ ] Tiered sizing: Total P&L +$200-300
- [ ] Features: Win rate +5-10%
- [ ] Ensemble: Win rate +3-5%

### Phase 3 (Paper Trading)
- [ ] 5/7 days profitable
- [ ] Win rate ≥ 50%
- [ ] Avg daily P&L ≥ +$100
- [ ] Circuit breaker trips ≤ 2

### Phase 4 (Topstep Combine)
- [ ] Reach $3,000 cumulative profit
- [ ] No daily loss violations
- [ ] No trailing drawdown violations
- [ ] 60%+ positive days
- [ ] ✅ **FUNDED ACCOUNT**

---

## 📁 Quick Reference Files

**Documentation**:
- `ml_intraday_v3/EXECUTIVE_SUMMARY.md` - Quick overview
- `ml_intraday_v3/PHASE_1_VALIDATION_COMPLETE.md` - Full Phase 1 results
- `ml_intraday_v3/FILTER_INTEGRATION_GUIDE.md` - Integration instructions

**Implementation**:
- `ml_intraday_v3/monitoring/adaptive_circuit_breaker.py` - Circuit breaker
- `ml_intraday_v3/filters/regime_filter.py` - Regime detector
- `ml_intraday_v3/live_trading/live_runner.py` - Main live trading loop

**Validation**:
- `ml_intraday_v3/experiments/validate_dec2025.py` - Dec 2025 validation
- `ml_intraday_v3/experiments/jan2026_exact_metrics.py` - Jan 2026 analysis

---

## 🚨 Emergency Contacts

**If circuit breaker keeps tripping** (>3 times/day):
- STOP trading immediately
- Review last 20 trades for patterns
- Check if regime shift occurred
- Consider extending paper trading

**If daily loss approaches -$1,000** (Topstep limit):
- Circuit breaker should have stopped you at -$500
- If not, immediately disable live trading
- Review circuit breaker configuration
- Contact support/review logs

**If combine fails**:
- Save all trade logs for analysis
- Run post-mortem analysis
- Identify failure pattern
- Return to paper trading for 14 days minimum
- Consider model retrain or different approach

---

**Current Status**: ✅ Phase 1 Complete, ready for Phase 2

**Next Action**: Choose integration approach and start Day 7

**Timeline**: 4-6 weeks to funded account

**Confidence**: 60-70% success probability with full implementation
