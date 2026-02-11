# Implementation Plan: Live Trading Critical Bug Fixes

**Objective**: Fix 6 critical bugs preventing the $650/day model from trading safely on GCP while respecting Topstep risk limits.

**Model**: `model_bundle_retrained_oct2024_nov2025.pkl` (validated, ready for deployment)

**Priority**: HIGH - System is deployed but not trading due to bugs

---

## Executive Summary

The live trading system has 6 critical bugs that prevent safe execution:

1. **Circuit breaker hardcoded to disabled** - No daily loss protection
2. **Risk manager state not synced** - Daily loss limits won't trigger  
3. **Regime detector using wrong data** - 100 bars instead of 90 days
4. **Model class mapping not validated** - Predictions could be inverted
5. **CUSUM state not reset daily** - Stale state affects first bars
6. **Direction change doesn't place new trade** - Delayed by 1 bar

**Impact**: Without these fixes, the system risks:
- Breaching Topstep's $1,000 daily loss limit
- False regime shift detection blocking valid trades
- Inverted predictions (buying when it should sell)
- Missed trading opportunities

---

## Bug Prioritization by Risk

### Priority 1: CRITICAL SAFETY (Must fix before any trading)
1. **Circuit Breaker Disabled** - Highest risk to capital
2. **Risk Manager State Sync** - Required for daily loss limits
3. **Model Class Validation** - Could invert all predictions

### Priority 2: OPERATIONAL (Affects trade execution)
4. **Regime Detector Data** - Blocks valid trades unnecessarily
5. **Direction Change Logic** - Misses reversal entries

### Priority 3: SESSION HYGIENE (Edge cases)
6. **CUSUM State Reset** - Only affects first bar of new session

---

## Detailed Fix Plans

### Bug 1: Circuit Breaker Hardcoded to False

**File**: `ml_intraday_v3/live_trading/execution_engine.py`  
**Lines**: 108-113

**Current Code**:
```python
# Circuit Breaker configuration (NEW)
cb_cfg = self.config.get("circuit_breaker", {})
# Operator override: disable circuit breaker when requested.
self.circuit_breaker_enabled = False  # ❌ HARDCODED
self.max_drawdown_limit = cb_cfg.get("max_drawdown_limit", 1500.0)
```

**Fix**:
```python
# Circuit Breaker configuration (NEW)
cb_cfg = self.config.get("circuit_breaker", {})
# Read from config, respect operator override
self.circuit_breaker_enabled = cb_cfg.get("enabled", False)  # ✅ FROM CONFIG
self.max_drawdown_limit = cb_cfg.get("max_drawdown_limit", 1500.0)
logger.info(f"Circuit Breaker Config: enabled={self.circuit_breaker_enabled}, limit=${self.max_drawdown_limit}")
```

**Current Config** (`live_trading.yaml` line 79):
```yaml
circuit_breaker:
  enabled: true  # Already enabled in config!
```

**Verification**:
- Check logs for "Circuit Breaker Config: enabled=True"
- Verify circuit breaker triggers at -$500 daily loss

---

### Bug 2: Risk Manager State Not Synced After Trades

**File**: `ml_intraday_v3/live_trading/live_runner.py`  
**Lines**: 355-356, 695-715

**Current Code**:
```python
# __init__ (line 355-356) - Only synced once at startup
if self.live_risk_manager:
    self.live_risk_manager.sync_equity(self.execution_engine.get_equity())

# _process_bar (line 695-715) - PnL never updated!
closed = self.execution_engine.update_positions(...)
if closed:
    for pos in closed:
        # ❌ Missing: self.live_risk_manager.update_pnl(pos['pnl_usd'])
```

**Fix** (line 715 in live_runner.py):
```python
if closed:
    logger.info(f"Closed {len(closed)} positions")

    for pos in closed:
        self.metrics_tracker.record_trade(...)
        
        # ✅ FIX: Update risk manager with realized PnL
        if self.live_risk_manager:
            self.live_risk_manager.update_pnl(pos['pnl_usd'])
            logger.debug(f"Risk manager updated: pnl={pos['pnl_usd']:.2f}, "
                        f"daily_pnl={self.live_risk_manager.daily_pnl:.2f}")
```

**Note**: The `live_risk_manager` has an `update_pnl()` method that accumulates daily PnL. Without calling it, the risk manager thinks daily PnL is always $0.

**Verification**:
- Check logs for "Risk manager updated: pnl=..." after each trade
- Verify daily_pnl accumulates correctly in risk checks

---

### Bug 3: Regime Detector Using Wrong Reference Data

**File**: `ml_intraday_v3/live_trading/live_runner.py`  
**Lines**: 619-642

**Current Code**:
```python
if self.regime_detector_enabled:
    logger.info("Initializing regime detector with training data...")
    try:
        # Get buffer for feature calculation
        bars_df = self.data_fetcher.get_buffer()  # ❌ Only 100 bars (~8 hours)
        
        # Generate features on historical data
        historical_features = self.feature_generator.generate_features(bars_df)
        
        # Initialize regime detector
        self.regime_detector = RegimeDetector(
            feature_cols=self.predictor.feature_columns,
            reference_window_days=90,  # ❌ Says 90 days but uses 100 bars!
            current_window_bars=100,
            max_shifted_features_pct=0.30
        )
        
        # Fit on historical features (buffer contains last N days)
        self.regime_detector.fit(historical_features)  # ❌ Wrong data!
```

**Problem**: 
- Config says `reference_window_days=90` but we're fitting on 100 bars (~8 hours)
- Should fetch 90 days of historical data for reference distribution
- Current buffer only has last 100 bars for feature calculation

**Fix Strategy**:
1. **Option A (Recommended)**: Disable regime detector until we can fetch 90 days
   - Set `regime_detector.enabled: false` in `live_trading.yaml`
   - Add TODO to implement proper historical data fetch
   
2. **Option B**: Fetch 90 days of historical bars during initialization
   - Requires TopstepX API call to fetch last 90 days
   - More complex but correct implementation

**Fix (Option A - Safe for immediate deployment)**:
```yaml
# live_trading.yaml (line 87-91)
regime_detector:
  enabled: false  # ✅ Disabled until we fetch proper historical data
  reference_window_days: 90
  current_window_bars: 100
  max_shifted_features_pct: 0.30
```

**Fix (Option B - Proper implementation)** in `live_runner.py`:
```python
if self.regime_detector_enabled:
    logger.info("Initializing regime detector with 90 days of training data...")
    try:
        # ✅ Fetch 90 days of historical bars (not just buffer)
        reference_start = pd.Timestamp.now() - pd.Timedelta(days=90)
        historical_bars = self.data_fetcher.fetch_historical_range(
            start=reference_start,
            end=pd.Timestamp.now()
        )
        
        if len(historical_bars) < 100:
            logger.error("Insufficient historical data for regime detector")
            self.regime_detector_enabled = False
            raise ValueError("Need 90 days of data for regime detector")
        
        # Generate features on 90 days of historical data
        historical_features = self.feature_generator.generate_features(historical_bars)
        
        # Initialize regime detector
        self.regime_detector = RegimeDetector(
            feature_cols=self.predictor.feature_columns,
            reference_window_days=90,
            current_window_bars=100,
            max_shifted_features_pct=0.30
        )
        
        # Fit on proper 90-day historical features
        self.regime_detector.fit(historical_features)
        logger.info(f"✅ Regime detector fitted on {len(historical_bars)} bars "
                   f"({(len(historical_bars)*5/60/24):.1f} days)")
```

**Recommendation**: Use Option A for immediate deployment, implement Option B later.

**Verification**:
- Check logs for "Regime detector fitted on N bars (X days)"
- Verify days is close to 90, not 0.3

---

### Bug 4: Model Class Mapping Not Validated

**File**: `ml_intraday_v3/live_trading/model_predictor.py`  
**Lines**: 130-145

**Current Code**:
```python
# Get class ordering from the underlying model
if hasattr(self.model.long_model, 'classes_') and self.model.long_model.classes_ is not None:
    classes = list(self.model.long_model.classes_)
elif hasattr(self.model, 'classes_') and self.model.classes_ is not None:
    classes = list(self.model.classes_)
else:
    classes = [0, 1, 2]  # ❌ Default assumption

# Map outcome labels to indices
if classes == [0, 1, 2]:  # ❌ Assumes [stop, vertical, target]
    stop_idx, vertical_idx, target_idx = 0, 1, 2
else:
    # ❌ Tries to find -1, 0, 1 in classes but model uses 0, 1, 2!
    target_idx = classes.index(1) if 1 in classes else 2
    stop_idx = classes.index(-1) if -1 in classes else 0
    vertical_idx = classes.index(0) if 0 in classes else 1
```

**Problem**:
- Assumes `[0, 1, 2]` maps to `[stop, vertical, target]`
- Model might use different encoding: `[0, 1, 2]` could be `[target, stop, vertical]`
- No validation that indices are correct
- **Critical**: Inverted mapping means buying when we should sell!

**Fix**:
```python
# Get class ordering from the underlying model
if hasattr(self.model.long_model, 'classes_') and self.model.long_model.classes_ is not None:
    classes = list(self.model.long_model.classes_)
elif hasattr(self.model, 'classes_') and self.model.classes_ is not None:
    classes = list(self.model.classes_)
else:
    # ✅ Fail hard if classes not available - don't assume!
    raise ValueError("Model must have 'classes_' attribute for safe prediction")

# ✅ Validate and map outcome labels to indices
# Model training uses: -1=stop, 0=vertical, 1=target
# Model classes_ should be: [-1, 0, 1] or [0, 1, 2]
if classes == [-1, 0, 1]:
    # Direct mapping: class values = outcome values
    stop_idx = classes.index(-1)
    vertical_idx = classes.index(0)
    target_idx = classes.index(1)
    logger.info(f"Class mapping (direct): stop={stop_idx}, vertical={vertical_idx}, target={target_idx}")
elif classes == [0, 1, 2]:
    # Encoded mapping: check bundle metadata for encoding
    # Most sklearn classifiers use sorted class order
    # Assumption: [0, 1, 2] maps to sorted [-1, 0, 1] = [stop, vertical, target]
    stop_idx, vertical_idx, target_idx = 0, 1, 2
    logger.info(f"Class mapping (encoded): stop={stop_idx}, vertical={vertical_idx}, target={target_idx}")
    logger.warning("⚠️ Using assumed class encoding [0,1,2]=[stop,vertical,target]. Verify in training logs!")
else:
    # ✅ Unknown encoding - fail hard!
    raise ValueError(f"Unknown class encoding: {classes}. Expected [-1,0,1] or [0,1,2]")

# ✅ Sanity check: Verify probabilities sum to 1.0
# Later in predict(), after getting proba:
if not np.isclose(proba.sum(axis=1), 1.0, atol=0.01):
    logger.error(f"Invalid probability distribution: {proba}")
    raise ValueError("Model probabilities don't sum to 1.0")
```

**Additional Verification** (add to `__init__`):
```python
# After loading model, validate with a dummy prediction
logger.info("Validating model class mapping...")
X_test = np.zeros((1, len(self.feature_columns)))
if self.has_dual_model:
    proba_long, proba_short = self.model.predict_proba_dual(X_test)
    logger.info(f"Test prediction shape: long={proba_long.shape}, short={proba_short.shape}")
    logger.info(f"Test proba sums: long={proba_long.sum():.3f}, short={proba_short.sum():.3f}")
else:
    proba = self.model.predict_proba(X_test)
    logger.info(f"Test prediction shape: {proba.shape}")
    logger.info(f"Test proba sum: {proba.sum():.3f}")

# Check that we can access all required indices
try:
    _ = proba[0, stop_idx]
    _ = proba[0, vertical_idx]
    _ = proba[0, target_idx]
    logger.info(f"✅ Class indices validated: stop={stop_idx}, vertical={vertical_idx}, target={target_idx}")
except IndexError as e:
    raise ValueError(f"Invalid class indices: {e}")
```

**Verification Plan**:
1. Check model bundle for class encoding in training logs
2. Add test prediction at startup to validate indices
3. Monitor first few live predictions for sensible probabilities
4. **CRITICAL**: Manually verify first trade direction matches expected signal

---

### Bug 5: CUSUM State Not Reset Daily

**File**: `ml_intraday_v3/live_trading/live_runner.py` + `event_detector.py`  
**Issue**: Event detector accumulates CUSUM state across sessions

**Current Code** (no daily reset):
```python
# In live_runner.py __init__ (line 316-343)
self.event_detector = LiveEventDetector(
    atr_period=atr_period,
    cusum_threshold_atr_mult=cusum_mult,
    min_cusum_threshold=min_cusum_threshold,
)
# ❌ Event detector state persists forever!

# In _process_bar (line 940-952)
if self.event_detector is not None:
    is_event, event_info = self.event_detector.is_event(
        bars_df=bars_df,
        current_bar_close=float(latest_bar['close'])
    )
    # ❌ No state reset at session boundary
```

**Problem**:
- CUSUM accumulator (`s_pos`, `s_neg`) persists across trading sessions
- First bar of new session inherits stale state from yesterday's last bar
- Can cause false events or missed events at session start

**Fix** (add to `live_runner.py`):
```python
def _process_bar(self, bar_time: pd.Timestamp, latest_bar: pd.Series):
    """
    Process a new bar: generate features, predict, and potentially execute.
    """
    # ✅ Check for new trading day and reset event detector
    if self.event_detector is not None:
        current_day = bar_time.date()
        if not hasattr(self, '_last_event_detector_reset_day'):
            self._last_event_detector_reset_day = current_day
        
        if current_day != self._last_event_detector_reset_day:
            logger.info(f"New trading day detected: {current_day}. Resetting CUSUM event detector.")
            self.event_detector.reset_state()
            self._last_event_detector_reset_day = current_day
    
    # Get rolling buffer
    bars_df = self.data_fetcher.get_buffer()
    # ... rest of function
```

**Also need to add reset method to event detector** (`live_trading/event_detector.py`):
```python
class LiveEventDetector:
    def reset_state(self):
        """Reset CUSUM accumulators for new trading session."""
        self.s_pos = 0.0
        self.s_neg = 0.0
        logger.info("CUSUM state reset: s_pos=0.0, s_neg=0.0")
```

**Verification**:
- Check logs for "New trading day detected" at session start
- Verify "CUSUM state reset" message appears
- Monitor first bar of day has fresh CUSUM state

---

### Bug 6: Direction Change Doesn't Place New Trade

**File**: `ml_intraday_v3/live_trading/execution_engine.py`  
**Lines**: 234-261

**Current Code**:
```python
if self.direction_change_enabled and self.open_positions:
    current_direction = self.get_net_position_direction()
    if current_direction != "FLAT" and current_direction != direction:
        score_ev = abs(prediction.get('score_ev', 0.0))
        
        if score_ev >= self.direction_change_threshold:
            # ❌ Flattens opposing positions but returns immediately
            current_price = bars_df.iloc[-1]['close']
            self.flatten_all_positions(timestamp, current_price, 
                                      f"direction_change_{current_direction}_to_{direction}")
            return False, "direction_changed_awaiting_confirmation"  # ❌ STOPS HERE
        else:
            return False, "opposing_signal_too_weak"
```

**Problem**:
- When direction changes (LONG→SHORT or SHORT→LONG), flattens old positions
- But returns `False` immediately, so new direction trade never executes
- Misses entry by 1 bar (next bar might not trigger same signal)

**Fix Options**:

**Option A (Conservative)**: Keep current behavior but document it
- Rationale: Safer to wait 1 bar for confirmation after direction change
- Prevents whipsaw trades
- Document as intended behavior in code comments

**Option B (Aggressive)**: Place new trade after flattening
```python
if self.direction_change_enabled and self.open_positions:
    current_direction = self.get_net_position_direction()
    if current_direction != "FLAT" and current_direction != direction:
        score_ev = abs(prediction.get('score_ev', 0.0))
        
        if score_ev >= self.direction_change_threshold:
            logger.info(
                f"STRONG opposing signal: {current_direction} -> {direction} "
                f"(|score_ev|={score_ev:.3f} >= {self.direction_change_threshold:.3f})"
            )
            # Flatten opposing positions
            current_price = bars_df.iloc[-1]['close']
            self.flatten_all_positions(timestamp, current_price, 
                                      f"direction_change_{current_direction}_to_{direction}")
            
            # ✅ FIX: Continue to place new trade in opposite direction
            logger.info(f"Proceeding to enter new {direction} position after direction change")
            # Don't return - fall through to normal trade execution below
            
        else:
            logger.info(
                f"WEAK opposing signal rejected: {current_direction} -> {direction} "
                f"(|score_ev|={score_ev:.3f} < {self.direction_change_threshold:.3f}) "
            )
            return False, "opposing_signal_too_weak"

# Continue with normal trade execution...
# (existing code for risk checks, etc.)
```

**Recommendation**: 
- Use **Option A** for initial deployment (safer)
- Monitor in dry-run to see if missed entries are significant
- Switch to Option B if we're leaving money on the table

**Verification**:
- Check logs for "Proceeding to enter new ... position after direction change"
- Verify new position opens on same bar as direction change
- Monitor P&L impact vs waiting 1 bar

---

## Implementation Sequence

### Phase 1: Critical Safety Fixes (Deploy Immediately)

**Must fix before any live trading:**

1. **Circuit Breaker** (5 min)
   - Change line 111 in `execution_engine.py`
   - Deploy and verify in logs

2. **Risk Manager Sync** (10 min)
   - Add `update_pnl()` call in `live_runner.py` line 715
   - Deploy and verify daily_pnl updates

3. **Model Class Validation** (30 min)
   - Add validation in `model_predictor.py` lines 130-145
   - Add test prediction at startup
   - **CRITICAL**: Manually verify first trade direction

**Total Phase 1**: ~45 minutes

### Phase 2: Operational Improvements (Can deploy incrementally)

4. **Regime Detector Fix** (Choose one)
   - Quick: Disable in config (2 min)
   - Proper: Implement 90-day fetch (2 hours)
   
5. **CUSUM State Reset** (20 min)
   - Add daily reset logic
   - Add reset method to event detector

6. **Direction Change** (10 min)
   - Document current behavior OR
   - Implement immediate entry

**Total Phase 2**: 30 min (quick) or 2.5 hours (proper)

---

## Deployment Procedure

### Pre-Deployment Checklist

1. **Code Changes**:
   - [ ] Circuit breaker reads from config
   - [ ] Risk manager PnL updates after trades
   - [ ] Model class validation with hard failures
   - [ ] (Optional) Regime detector disabled or fixed
   - [ ] (Optional) CUSUM daily reset
   - [ ] (Optional) Direction change fix

2. **Config Validation**:
   - [ ] `circuit_breaker.enabled: true` in `live_trading.yaml`
   - [ ] `regime_detector.enabled: false` (if not fixed)
   - [ ] Contract ID matches current month: `CON.F.US.MES.H26`

3. **Environment Variables** (GCP VM):
   ```bash
   TOPSTEPX_USERNAME=<username>
   TOPSTEPX_PROJECTX_API_KEY=<api_key>
   TOPSTEPX_ACCOUNT_ID=<account_id>
   TOPSTEPX_CONTRACT_ID=CON.F.US.MES.H26
   ```

### Deployment Steps

```bash
# 1. Build Docker image with fixes
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep/ml_intraday_v3
docker build -t gcr.io/trading-algo-3/topstep-trader:latest .

# 2. Push to Google Container Registry
docker push gcr.io/trading-algo-3/topstep-trader:latest

# 3. Stop running VM
gcloud compute instances stop topstep-trader-vm --zone=us-central1-a

# 4. Update container image
gcloud compute instances update-container topstep-trader-vm \
    --zone=us-central1-a \
    --container-image=gcr.io/trading-algo-3/topstep-trader:latest

# 5. Start VM
gcloud compute instances start topstep-trader-vm --zone=us-central1-a

# 6. Wait for startup (10 seconds)
sleep 10

# 7. Tail logs to verify startup
gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
    --command="docker logs -f \$(docker ps -q) 2>&1"
```

### Post-Deployment Verification

**Check logs for these messages**:

```
✅ PASS: Circuit Breaker Config: enabled=True, limit=$1500.0
✅ PASS: Class indices validated: stop=0, vertical=1, target=2
✅ PASS: Startup checks passed
✅ PASS: Contract sanity check passed: CON.F.US.MES.H26 is MES
✅ PASS: Live trading started
```

**Monitor first bar**:
```
INFO: New bar: 2026-02-03 09:35:00, close=5842.75
INFO: ✓ CUSUM event detected (positive): threshold=8.50, price_diff=12.30
INFO: Signal generated: score=0.245, p_target=0.623, p_stop=0.378
INFO: ✓ Trade executed: LONG 1 contracts, score=0.245
```

**Verify risk manager updates after trade**:
```
INFO: Position closed: DRY_..., reason=target, pnl=$62.50
DEBUG: Risk manager updated: pnl=62.50, daily_pnl=62.50
```

---

## Rollback Plan

**If issues arise during deployment:**

### Emergency Stop
```bash
# Immediate halt - stop VM
gcloud compute instances stop topstep-trader-vm --zone=us-central1-a
```

### Rollback to Previous Image
```bash
# Find previous image tag
gcloud container images list-tags gcr.io/trading-algo-3/topstep-trader

# Rollback to previous version (e.g., sha256:abc123...)
gcloud compute instances update-container topstep-trader-vm \
    --zone=us-central1-a \
    --container-image=gcr.io/trading-algo-3/topstep-trader@sha256:abc123...

# Restart VM
gcloud compute instances start topstep-trader-vm --zone=us-central1-a
```

### Manual Position Flatten
```bash
# SSH to VM and run manual flatten script
gcloud compute ssh topstep-trader-vm --zone=us-central1-a

# Inside VM:
python3 ml_intraday_v3/test_manual_trade.py --flatten-all
```

---

## Testing Strategy

### Phase 1: Dry-Run Testing (Paper Trading)

**Duration**: 1-2 days  
**Config**: `trading.dry_run: true`

**Test Cases**:
1. Circuit breaker triggers at -$500 daily loss
2. Risk manager daily_pnl accumulates correctly
3. Model predictions have correct probabilities (sum to 1.0)
4. CUSUM state resets at session start
5. Direction change behavior (flatten + optional new entry)

**Success Criteria**:
- No errors in logs
- Circuit breaker triggers correctly
- Daily PnL matches trade log
- Trade directions match expected signals

### Phase 2: Live Testing (Small Size)

**Duration**: 1 week  
**Config**: 
- `trading.dry_run: false`
- `trading.environment: paper`
- `positions.contracts_per_trade: 1`

**Monitor**:
- Daily P&L stays within $150/day range
- Circuit breaker prevents runaway losses
- Win rate ~60% (from backtest)
- Average trade P&L ~$50 (from backtest)

**Red Flags**:
- Daily loss exceeds -$200 (investigate immediately)
- Win rate < 50% (model may be inverted!)
- Trades executing in wrong direction

### Phase 3: Full Deployment

**After 1 week of successful paper trading:**
- Switch to live environment: `trading.environment: live`
- Start with 1 contract
- Gradually increase to 2 contracts if profitable

---

## Monitoring and Alerts

### Critical Metrics

**Log Monitoring** (every 5 minutes):
```bash
gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
    --command="docker logs \$(docker ps -q) 2>&1 | tail -50"
```

**Key Phrases to Watch**:
- `✗ FAIL` - Startup check failure
- `Circuit Breaker Triggered` - Daily loss limit hit
- `⚠️ REGIME SHIFT DETECTED` - Trading paused
- `Error in main loop` - Crash or bug

**Daily Health Check**:
```bash
# Check VM is running
gcloud compute instances list | grep topstep-trader-vm

# Check container is running
gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
    --command="docker ps"

# Check recent trades
gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
    --command="tail -20 /app/logs/trades_*.csv"
```

### Performance Targets

**Daily Metrics** (from backtesting):
- Trades per day: 10-20
- Win rate: 60%
- Average winner: $75
- Average loser: -$40
- Daily P&L target: $150-200
- Max daily loss: -$500 (circuit breaker)

**Weekly Targets**:
- Week 1: $500-700 profit
- Week 2+: $1,000+ profit

---

## Risk Management

### Topstep Limits (50k Combine)

**Hard Limits**:
- Daily loss limit: -$1,000 (breach = fail combine)
- Trailing max drawdown: -$2,500 (breach = fail combine)

**Our Circuit Breakers**:
- Daily loss limit: -$500 (stop before Topstep limit)
- Consecutive losses: 3 (pause 30 min)
- Max drawdown: -$1,500 (stop before Topstep limit)

**Safety Margin**:
- Topstep daily: $1,000
- Our limit: $500
- Buffer: $500 (50%)

### Position Sizing

**Current Configuration**:
- Contracts per trade: 1
- Max concurrent positions: 1
- Kelly sizing: Disabled

**Rationale**:
- Start conservative with 1 contract
- Prove profitability before scaling
- Avoid Topstep consistency rule violations

---

## Success Criteria

### Phase 1 Complete (Critical Fixes)
- [ ] Circuit breaker reads from config
- [ ] Risk manager updates after every trade
- [ ] Model class mapping validated
- [ ] All startup checks pass
- [ ] First trade executes correctly

### Phase 2 Complete (Paper Trading)
- [ ] 2 days of paper trading with no errors
- [ ] Win rate ≥ 55%
- [ ] Daily P&L positive on average
- [ ] Circuit breaker never triggered (no -$500 days)

### Phase 3 Complete (Live Trading)
- [ ] 1 week of live trading
- [ ] Cumulative profit ≥ $500
- [ ] Max daily loss ≤ $200
- [ ] No Topstep rule violations

---

## Critical Files for Implementation

### Files to Modify (Priority Order):

1. **`/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/live_trading/execution_engine.py`**
   - Line 111: Circuit breaker hardcoded to False → read from config
   - Lines 234-261: Direction change logic (optional fix)

2. **`/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/live_trading/live_runner.py`**
   - Line 715: Add `self.live_risk_manager.update_pnl(pos['pnl_usd'])` after trade close
   - Lines 619-642: Fix regime detector initialization (disable or fetch 90 days)
   - New method: Add `_process_bar` daily CUSUM reset check

3. **`/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/live_trading/model_predictor.py`**
   - Lines 130-145: Add model class validation with hard failures
   - Lines 60-80: Add test prediction at startup

### Configuration Files:

4. **`/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/configs/live_trading.yaml`**
   - Line 79: Already has `circuit_breaker.enabled: true` ✓
   - Line 88: Disable regime detector if not fixing: `enabled: false`
   - Line 55: Verify contract ID: `CON.F.US.MES.H26` (March 2026)

### Deployment Files:

5. **`/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/Dockerfile`**
   - Verify model path correct: Line 23
   - Verify entry point correct: Lines 32-34

---

## Estimated Timeline

### Minimal Safe Deployment (Phase 1 Only)
- **Code changes**: 45 minutes
- **Testing**: 30 minutes
- **Deployment**: 15 minutes
- **Verification**: 30 minutes
- **Total**: 2 hours

### Full Implementation (All Fixes)
- **Phase 1 (critical)**: 2 hours
- **Phase 2 (operational)**: 2-4 hours
- **Testing**: 2 days (paper trading)
- **Total**: 2-3 days to live trading

---

## Notes and Assumptions

### Assumptions Made:
1. Model bundle `model_bundle_retrained_oct2024_nov2025.pkl` is validated and correct
2. TopstepX API credentials are configured in GCP VM environment
3. Contract ID `CON.F.US.MES.H26` is correct for current trading period
4. Backtest results ($650/day, 60% win rate) are reliable estimates

### Known Limitations:
1. Regime detector needs 90 days of data (currently disabled/broken)
2. CUSUM state reset only happens once per day (not per session)
3. Direction change may miss immediate entry (1 bar delay)

### Future Enhancements:
1. Implement proper 90-day historical data fetch for regime detector
2. Add intraday CUSUM reset at session boundaries
3. Optimize direction change to enter immediately after flatten
4. Add Kelly sizing after 30+ profitable trades
5. Implement position scaling strategy (1→2→3 contracts)

---

## Contact and Escalation

### If Issues Arise:

**Immediate Actions**:
1. Stop VM: `gcloud compute instances stop topstep-trader-vm --zone=us-central1-a`
2. Check logs for errors
3. Verify no open positions on Topstep account

**Debug Logs Location**:
- Container logs: `docker logs $(docker ps -q)`
- Trade logs: `/app/logs/trades_*.csv`
- Session logs: `/app/logs/live_trading_*.log`

**Emergency Contacts**:
- Check Topstep account dashboard for live positions
- Review trade history in Topstep web interface

---

## Conclusion

This plan prioritizes **capital safety** while fixing critical bugs that prevent trading. The phased approach allows for:

1. **Immediate deployment** of critical safety fixes (2 hours)
2. **Incremental testing** in paper trading (2 days)
3. **Gradual transition** to live trading (1 week)

**Key Success Factor**: Validate model class mapping on first live trade - this is the highest risk bug that could invert all predictions.

**Expected Outcome**: After fixes, system should safely trade 10-20 times per day with 60% win rate and $150+ daily profit while respecting Topstep limits.
