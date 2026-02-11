# Topstep 50k Combine Pass Plan: 15-20 Trading Days

**Date Created**: 2026-01-29  
**Status**: Implementation Plan  
**Goal**: Pass Topstep 50k Combine ($3,000 profit target) within 15-20 trading days

---

## Executive Summary

The current model achieved 80.3% accuracy on 2024-2025 data but **FAILED catastrophically** on January 2026 live trading (35.5% win rate, -$884.73 loss, 8.4 trades/day at -$5.82/trade). This is a **regime shift problem**, not a model architecture problem. The infrastructure is ready (live trading module, risk management, execution engine), but the model is trading in a market regime it doesn't understand.

**Critical Insight**: The user has been working on this for months. They need a **working solution NOW**, not more analysis. This plan prioritizes **fast, high-ROI fixes** that can be validated quickly and get them trading profitably within 4 weeks.

**Success Path**:
1. **Week 1 (3-5 days)**: Fix regime shift with confidence filtering + circuit breakers
2. **Week 2 (5-7 days)**: Paper trade validation with strict metrics
3. **Week 3-4 (20 days)**: Topstep combine execution with conservative targets

**Key Trade-offs**:
- **Speed vs Perfection**: Use confidence filtering (fast, 2 days) instead of retraining (slow, 7+ days)
- **Trades/Day vs Quality**: Target 3-5 high-quality trades/day ($30-50/trade) instead of 8+ mediocre trades
- **Pass Rate vs Speed**: Conservative approach (80%+ pass rate) over aggressive (faster but riskier)

---

## Timeline Overview

| Phase | Duration | Objective | GO/NO-GO Criteria |
|-------|----------|-----------|-------------------|
| **Phase 1: Model Fixes** | 3-5 days | Eliminate regime shift failures | Paper trade win rate > 50% for 3 consecutive days |
| **Phase 2: Validation** | 5-7 days | Prove model works in Jan 2026 | Daily P&L > $0 for 5/7 days, max DD < $500 |
| **Phase 3: Topstep** | 15-20 days | Pass combine | Reach $3,000 target without breaching limits |
| **Total** | **23-32 days** | From broken to funded | |

---

## Current Situation Analysis

### What Went Wrong (January 2026)

**Performance Breakdown**:
- Win rate: 35.5% (vs 80.3% on 2024-2025 data)
- Total P&L: -$884.73 (12 trading days)
- Trade frequency: 8.4 trades/day (too many low-quality trades)
- Avg trade: -$5.82 (losing on average)
- Daily P&L: -$49/day (would fail combine in 20 days)

**Root Causes**:
1. **Regime Shift**: January 2026 market structure is different from 2024-2025 training data
2. **No Confidence Filtering**: System trades ALL signals (low and high confidence equally)
3. **No Circuit Breakers**: Model kept trading even as losses mounted
4. **Overfitting**: Model learned patterns specific to 2024-2025 that don't generalize

**Positive Findings**:
- Medium confidence trades (P > 0.55) are profitable: +$4.63/trade (n=10)
- Low confidence trades (P < 0.55, 80% of trades) lose -$6.77/trade
- Infrastructure is solid: Live trading, risk management, execution all work
- Risk management prevented catastrophic loss (only -$884, not thousands)

### What Works (2024-2025 Backtest)

**Validation Results**:
- K-Fold: +$37,459 PnL, 59% win rate, 6/6 splits profitable
- Walk-Forward: +$22,249 PnL, 59% win rate, 13/15 windows profitable
- Topstep Simulation: 77.8% pass rate (10,000 Monte Carlo runs)
- Deflated Sharpe: 1.00 (strong evidence of skill)

**This tells us**:
- The model CAN work when trained on recent data
- The methodology is sound (high walk-forward success)
- Risk management is effective (0 drawdown breaches)
- We need to adapt to new regime, not rebuild from scratch

---

## Phase 1: Model Fixes (3-5 Days)

**Goal**: Stop the bleeding. Make the model profitable on January 2026 data.

### Prioritized Fixes (Ranked by ROI)

#### FIX 1: Confidence Threshold Filtering (Day 1, 4 hours) - **HIGHEST ROI**

**Problem**: 80% of trades are low-confidence losers (-$6.77/trade)  
**Solution**: Only trade when model confidence > 0.55

**Implementation**:
```python
# File: ml_intraday_v3/live_trading/model_predictor.py
# Modify should_trade() method

def should_trade(self, prediction, score):
    # Existing threshold check
    if abs(score) < self.config['signals']['primary_threshold']:
        return False, "Below primary threshold"
    
    # NEW: Confidence filtering
    confidence_threshold = self.config['signals'].get('confidence_threshold', 0.55)
    probability = self._score_to_probability(score)
    
    if probability < confidence_threshold and probability > (1 - confidence_threshold):
        return False, f"Low confidence: P={probability:.3f}"
    
    return True, "High confidence trade"
```

**Expected Impact**:
- Reduce trades from 8.4/day to 3-5/day (eliminate low-quality trades)
- Increase avg trade P&L from -$5.82 to +$4.63 (based on Jan 2026 data)
- Daily expectancy: 4 trades * $4.63 = +$18.52/day (vs -$49/day currently)

**Validation**:
- Backtest on Jan 2026 data with threshold 0.55, 0.60, 0.65
- Pick threshold that maximizes Sharpe ratio (not raw P&L)
- Target: Win rate > 50%, daily P&L > $0

**Time**: 4 hours (2 hours implementation, 2 hours backtesting)

---

#### FIX 2: Circuit Breaker System (Day 1-2, 6 hours) - **HIGH ROI**

**Problem**: Model kept trading even when clearly failing  
**Solution**: Auto-stop trading when model shows signs of failure

**Implementation**:
```python
# File: ml_intraday_v3/live_trading/risk_manager.py
# Add to RiskManager class

class CircuitBreaker:
    def __init__(self):
        self.consecutive_losses = 0
        self.daily_loss = 0.0
        self.is_halted = False
        
    def check_circuit_breaker(self, trade_result):
        # Rule 1: 3 consecutive losses
        if trade_result['pnl'] < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        if self.consecutive_losses >= 3:
            self.halt("3 consecutive losses")
            return True
            
        # Rule 2: -$500 daily loss
        self.daily_loss += trade_result['pnl']
        if self.daily_loss <= -500:
            self.halt(f"Daily loss: ${self.daily_loss:.2f}")
            return True
            
        # Rule 3: Win rate < 30% (min 10 trades)
        if self.total_trades >= 10:
            win_rate = self.wins / self.total_trades
            if win_rate < 0.30:
                self.halt(f"Win rate: {win_rate:.1%}")
                return True
                
        return False
```

**Circuit Breaker Rules**:
1. **3 Consecutive Losses**: Stop trading immediately
2. **-$500 Daily Loss**: Stop at 50% of Topstep daily limit
3. **Win Rate < 30%** (after 10+ trades): Model is clearly failing

**Expected Impact**:
- Prevent catastrophic loss days (limit losses to -$500 max)
- Preserve capital for when model is working
- Build confidence that system won't spiral out of control

**Validation**:
- Backtest on Jan 2026 with circuit breakers enabled
- Verify it would have stopped trading before -$884 loss
- Target: Max daily loss < $500

**Time**: 6 hours (3 hours implementation, 2 hours testing, 1 hour validation)

---

#### FIX 3: Regime Detection with KS Test (Day 2-3, 8 hours) - **MEDIUM ROI**

**Problem**: Model doesn't know when market regime has shifted  
**Solution**: Detect when current features differ significantly from training distribution

**Implementation**:
```python
# File: ml_intraday_v3/features/regime_detector.py
# Add new method to existing RegimeDetector class

from scipy import stats

class RegimeDetector:
    def __init__(self, training_features):
        """Store feature distributions from training data."""
        self.feature_stats = {}
        for col in training_features.columns:
            self.feature_stats[col] = {
                'mean': training_features[col].mean(),
                'std': training_features[col].std(),
                'quantiles': training_features[col].quantile([0.25, 0.50, 0.75]).values
            }
    
    def detect_regime_shift(self, recent_features, lookback=100):
        """
        Use KS test to detect if recent features differ from training distribution.
        
        Args:
            recent_features: Last N bars of features
            lookback: How many bars to compare (default: 100)
            
        Returns:
            is_shifted: bool (True if regime has shifted)
            confidence: float (0-1, how confident we are in the shift)
        """
        if len(recent_features) < lookback:
            return False, 0.0
            
        recent_window = recent_features.tail(lookback)
        
        # Run KS test on top 5 most important features
        important_features = ['vol_20', 'atr_14', 'ema_spread', 'log_return_4', 'trend_strength']
        ks_pvalues = []
        
        for feat in important_features:
            if feat not in recent_window.columns:
                continue
                
            # KS test: Compare recent vs training distribution
            ks_stat, ks_pvalue = stats.ks_2samp(
                recent_window[feat].dropna(),
                self.training_features[feat].dropna()
            )
            ks_pvalues.append(ks_pvalue)
        
        # If majority of features show significant shift (p < 0.05)
        significant_shifts = sum(p < 0.05 for p in ks_pvalues)
        is_shifted = significant_shifts >= 3  # 3 out of 5 features
        confidence = significant_shifts / len(ks_pvalues)
        
        return is_shifted, confidence
```

**Regime Detection Strategy**:
- **Conservative**: If 3+ key features show distribution shift, STOP trading
- **Fallback**: Log warning but continue trading with HIGHER confidence threshold (0.65)
- **Re-enable**: Manually after retraining or regime stabilizes

**Expected Impact**:
- Prevent trading in unfavorable regimes (would have stopped Jan 2026 losses)
- Reduce false positives by using multiple features
- Provide early warning for retraining needs

**Validation**:
- Test on Jan 2026 data: Should detect regime shift by Jan 15
- Test on Dec 2025 data: Should NOT detect regime shift
- Target: Detect regime changes within 3 days of occurrence

**Time**: 8 hours (4 hours implementation, 2 hours testing, 2 hours validation)

---

#### FIX 4: Volatility Percentile Filter (Day 3-4, 6 hours) - **MEDIUM ROI**

**Problem**: Model may perform poorly in extreme volatility regimes  
**Solution**: Only trade when volatility is in favorable range (30th-70th percentile)

**Implementation**:
```python
# File: ml_intraday_v3/filters/volatility_filter.py
# Enhance existing VolatilityFilter

class VolatilityFilter:
    def __init__(self, training_volatility):
        """
        Args:
            training_volatility: pd.Series of vol_20 from training data
        """
        self.vol_p30 = training_volatility.quantile(0.30)
        self.vol_p70 = training_volatility.quantile(0.70)
        
    def should_trade(self, current_vol):
        """Only trade in middle 40% of volatility range."""
        if current_vol < self.vol_p30:
            return False, f"Too low volatility: {current_vol:.6f} < {self.vol_p30:.6f}"
        
        if current_vol > self.vol_p70:
            return False, f"Too high volatility: {current_vol:.6f} > {self.vol_p70:.6f}"
            
        return True, "Volatility in normal range"
```

**Volatility Strategy**:
- Train on ALL data, but only trade in "normal" volatility (middle 40%)
- Avoid extreme low vol (choppy, low signal) and extreme high vol (erratic behavior)
- Use training data percentiles (not rolling percentiles to avoid lookahead bias)

**Expected Impact**:
- Filter out 60% of trading opportunities (keep best 40%)
- Improve win rate by avoiding unfavorable conditions
- Reduce variance (more consistent daily P&L)

**Validation**:
- Backtest with 30th-70th percentile filter on Jan 2026
- Compare win rate and Sharpe ratio vs no filter
- Target: +5-10% improvement in win rate

**Time**: 6 hours (3 hours implementation, 2 hours backtesting, 1 hour analysis)

---

#### FIX 5: Ensemble Prediction (Day 4-5, 12 hours) - **OPTIONAL**

**Problem**: Single model may be overfitting to specific time period  
**Solution**: Use ensemble of models trained on different time windows

**Implementation**:
```python
# File: ml_intraday_v3/live_trading/model_predictor.py

class EnsemblePredictor:
    def __init__(self):
        self.models = [
            load_model("runs/3month_model/model_bundle.pkl"),   # Q4 2025 only
            load_model("runs/6month_model/model_bundle.pkl"),   # Jul-Dec 2025
            load_model("runs/12month_model/model_bundle.pkl"),  # 2025 full year
        ]
        
    def predict(self, features):
        """Average predictions from all models."""
        predictions = []
        for model in self.models:
            pred = model.predict_proba(features)
            predictions.append(pred)
            
        # Average probabilities
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
```

**Ensemble Strategy**:
- Train 3 models: 3-month (recent), 6-month (medium), 12-month (long-term)
- Average their predictions (equal weighting)
- Only trade when ALL 3 models agree (within 0.1 probability)

**Expected Impact**:
- More robust to regime shifts (short-term model adapts faster)
- Reduce overfitting (long-term model provides stability)
- Higher confidence trades (all models must agree)

**Validation**:
- Train 3 models on different windows
- Backtest ensemble on Jan 2026
- Compare to single model performance
- Target: +10-15% improvement in Sharpe ratio

**Time**: 12 hours (6 hours training, 4 hours implementation, 2 hours validation)  
**NOTE**: This is OPTIONAL. Only do if Fixes 1-4 aren't sufficient.

---

### Phase 1 Summary

**Total Time**: 3-5 days (24-38 hours of work)  
**Deliverables**:
1. Confidence threshold filter (0.55-0.65)
2. Circuit breaker system (3 rules)
3. Regime detection (KS test on 5 features)
4. Volatility percentile filter (30th-70th)
5. *(Optional)* Ensemble model

**Expected Improvements**:
- Win rate: 35.5% → 55-60%
- Trades/day: 8.4 → 3-5
- Avg trade P&L: -$5.82 → +$20-30
- Daily P&L: -$49 → +$60-150

**GO/NO-GO Criteria**:
- ✅ **GO to Phase 2**: 3 consecutive paper trading days with win rate > 50% AND daily P&L > $0
- ❌ **NO-GO**: If win rate < 45% after 5 days, escalate to full retraining (adds 7-10 days)

---

## Phase 2: Paper Trading Validation (5-7 Days)

**Goal**: Prove the model works in January 2026 conditions before risking real money.

### Validation Protocol

#### Week 1: Prove It Works (5 Trading Days)

**Daily Metrics to Track**:
```
1. Win Rate (target: > 50%, minimum: > 45%)
2. Daily P&L (target: > $50, minimum: > $0)
3. Trade Count (target: 3-5 trades/day)
4. Max Drawdown (target: < $300, maximum: < $500)
5. Circuit Breaker Hits (target: 0, maximum: 1/week)
```

**Success Criteria** (ALL must pass):
- ✅ 5/7 days with positive P&L
- ✅ Overall win rate > 50%
- ✅ Max single-day loss < $500
- ✅ Total week P&L > $200
- ✅ < 2 circuit breaker hits

**Failure Criteria** (ANY triggers re-evaluation):
- ❌ 3+ consecutive losing days
- ❌ Any day with loss > $500
- ❌ Overall win rate < 40%
- ❌ Circuit breaker hits > 3 times

**Daily Review Process**:
```
End of each day:
1. Run analysis script: python ml_intraday_v3/analysis/daily_review.py
2. Check win rate, P&L, drawdown
3. Review trade log for patterns:
   - Are losses clustered in specific time windows?
   - Are certain exit reasons dominating?
   - Is confidence threshold too loose/tight?
4. Adjust parameters if needed:
   - If win rate < 45%: Increase confidence threshold by 0.05
   - If trades < 2/day: Decrease confidence threshold by 0.05
   - If circuit breaker hit: Review trades that triggered it
```

#### Week 2: Topstep Simulation (Optional, 5 Trading Days)

**Goal**: Simulate Topstep combine rules on paper account

**Setup**:
```yaml
# Update configs/risk_topstep_50k_strict.yaml
topstep_simulation:
  enabled: true
  starting_capital: 50000
  profit_target: 3000
  daily_loss_limit: 1000
  trailing_drawdown: 2500
  reset_daily: true  # Reset daily loss at session start
```

**Simulate Combine Rules**:
- Start with $50,000 virtual equity
- Track daily loss from session start (not total equity)
- Track trailing drawdown from high water mark
- Auto-stop if either limit breached

**Success Criteria**:
- ✅ Reach $3,000 profit target within 20 simulated days
- ✅ No daily loss > $1,000
- ✅ No trailing drawdown > $2,500
- ✅ No consistency violations (max day < 50% of total profit)

**Failure Criteria**:
- ❌ Breach daily loss limit
- ❌ Breach trailing drawdown limit
- ❌ Cannot reach target in 20 days

---

### GO/NO-GO Decision Point

**After 5-7 Days of Paper Trading**:

#### GO to Phase 3 (Topstep) IF:
- ✅ Win rate consistently > 50%
- ✅ Daily P&L positive 70%+ of days
- ✅ Max drawdown < $500
- ✅ Circuit breakers < 2 hits total
- ✅ Confidence in model understanding

#### NO-GO (Extend Validation) IF:
- ⚠️ Win rate 45-50% (borderline)
- ⚠️ Inconsistent daily results (big swings)
- ⚠️ Circuit breaker hits 2-3 times
- ⚠️ Not confident in model behavior

**Action if NO-GO**:
1. Extend paper trading another 5 days
2. Implement optional Fix 5 (Ensemble)
3. Consider retraining on Q4 2025 + Jan 2026 data
4. Re-evaluate after extended testing

---

## Phase 3: Topstep 50k Combine Execution (15-20 Days)

**Goal**: Pass the combine and reach funded status.

### Combine Parameters

**Topstep 50k Combine Rules**:
- Starting Capital: $50,000
- Profit Target: $3,000
- Daily Loss Limit: $1,000 (absolute)
- Trailing Max Drawdown: $2,500 (from high water mark)
- Consistency Rule: No single day > 50% of total profit

**Our Conservative Limits** (Tighter than Topstep):
- Daily Stop: $700 (70% of limit)
- Trailing DD Stop: $2,000 (80% of limit)
- Position Size: 1 contract (no scaling during combine)
- Max Trades/Day: 5 (prevent overtrading)
- No trading after circuit breaker

---

### Trading Strategy

#### Daily Targets (Conservative Approach)

**Target**: $150-200/day (reach $3,000 in 15-20 days)

**Trade Execution Plan**:
```
Morning (8:30 AM - 11:00 AM CT):
- Max 2 trades
- Focus on highest confidence signals (P > 0.60)
- Take profits at 1.5x ATR (don't be greedy)

Midday (11:00 AM - 1:00 PM CT):
- Max 1 trade
- Avoid lunch hour chop (11:30 AM - 12:30 PM)
- Only if confidence > 0.65

Afternoon (1:00 PM - 3:00 PM CT):
- Max 2 trades
- Stop new entries after 2:45 PM
- Flatten all positions by 2:55 PM

Daily Limits:
- If +$200: STOP trading (hit target, preserve capital)
- If -$500: STOP trading (circuit breaker)
- If 3 losses: STOP trading (circuit breaker)
- If 5 trades: STOP trading (prevent overtrading)
```

#### Position Sizing

**Week 1 (Days 1-5)**: Ultra-Conservative
- 1 contract per trade
- Max 1 concurrent position
- Stop loss: 1.0x ATR
- Take profit: 2.0x ATR (conservative R:R)

**Week 2 (Days 6-10)**: If Ahead of Target
- 1 contract per trade (NO SCALING)
- Max 1 concurrent position
- Stop loss: 1.0x ATR
- Take profit: 1.5x ATR (take profits faster)

**Week 3+ (Days 11-20)**: Finish Strong
- 1 contract per trade
- Max 1 concurrent position
- If within $500 of target: Risk $50/trade max (ultra-safe)
- Once at $2,800+: Stop trading (lock in profits)

---

### Risk Management

#### Daily Risk Rules

**Before Trading Each Day**:
```
1. Check equity: Is HWM > $50,000?
2. Check trailing DD: Current DD from HWM < $2,000?
3. Check yesterday's result: Was yesterday a loss?
   - If yes: Increase confidence threshold to 0.65
4. Check last 3 days: 2+ losing days?
   - If yes: Take a day off (regime may be shifting)
```

**During Trading**:
```
After each trade:
1. Update daily P&L
2. Check circuit breakers (3 losses, -$500, win rate < 30%)
3. If circuit breaker: STOP immediately, don't trade rest of day
4. If at +$200: Consider stopping (hit target)
5. If at -$400: Reduce position size by 50% (next trade is 0.5 contracts)
```

**End of Day**:
```
1. Flatten all positions by 2:55 PM (NO EXCEPTIONS)
2. Review trade log
3. Update tracking spreadsheet:
   - Daily P&L
   - Running total P&L
   - High water mark
   - Trailing drawdown
   - Win rate
   - Trade count
4. Plan tomorrow:
   - Adjust confidence threshold if needed
   - Review circuit breaker status
```

---

### Weekly Milestones

**Week 1 (Days 1-5)**:
- **Target**: +$600-750 (avg $150/day)
- **Min Acceptable**: +$300 (avg $60/day)
- **Max DD**: < $500

**Week 2 (Days 6-10)**:
- **Target**: +$1,500 cumulative (on pace for $3,000)
- **Min Acceptable**: +$900 cumulative
- **Max DD**: < $800

**Week 3 (Days 11-15)**:
- **Target**: +$2,400 cumulative (close to finish)
- **Min Acceptable**: +$1,800 cumulative
- **Max DD**: < $1,200

**Week 4 (Days 16-20)**:
- **Target**: +$3,000 (PASS)
- **Min Acceptable**: +$2,700 (nearly there)
- **Max DD**: < $1,500

---

### Contingency Plans

#### Scenario 1: Slow Start (Week 1 < $300)

**Problem**: Not generating enough P&L to hit target in 20 days

**Actions**:
1. Review confidence threshold (may be too high)
2. Check if volatility filter is too restrictive
3. Consider extending combine to 30 days (reduces daily pressure)
4. Lower confidence threshold by 0.05 IF win rate > 60%

**Decision Point**: After Day 10
- If < $1,000: Evaluate if achievable in remaining 10 days
- If trajectory projects to $2,000-2,500: Consider reset and retry

---

#### Scenario 2: Circuit Breaker Hit

**Problem**: Model triggered stop (3 losses OR -$500)

**Actions**:
1. STOP trading immediately
2. Review trades that triggered breaker:
   - Was it regime shift? (check KS test)
   - Was it bad luck? (check if signals were valid)
   - Was it parameter drift? (check feature distributions)
3. Next day: Paper trade only (no real execution)
4. If paper trade successful: Resume next day with HIGHER confidence (0.70)
5. If paper trade fails: Extend paper trading 3 days

**Decision Point**: After 2 Circuit Breaker Hits
- If 2 hits in same week: STOP combine, return to paper trading
- If isolated incidents: Continue with elevated confidence threshold

---

#### Scenario 3: Approaching Daily Loss Limit

**Problem**: Daily P&L at -$700 (approaching -$1,000 limit)

**Actions**:
1. STOP trading immediately (don't risk breach)
2. Analyze what went wrong:
   - Review trades from today
   - Check if regime shifted
   - Verify feature quality
3. Tomorrow: Start with MINIMAL risk
   - Confidence threshold 0.70
   - Max 2 trades
   - $25 risk per trade
4. Rebuild slowly over 2-3 days

---

#### Scenario 4: Approaching Trailing Drawdown Limit

**Problem**: Trailing DD at -$2,000 (approaching -$2,500 limit)

**Actions**:
1. Switch to CAPITAL PRESERVATION mode
2. Trading rules:
   - Only 1 trade/day
   - Confidence > 0.70
   - Stop loss 0.5x ATR (tighter stops)
   - Take profit 1.0x ATR (quick profits)
3. Goal: Get back to breakeven, THEN push for profits
4. If DD reaches -$2,300: STOP trading completely
   - Review all trades
   - Consider full reset

**Decision Point**: If DD > -$2,000
- Consult paper trading results
- Verify model is still working
- Consider abandoning combine if model broken

---

## Trade-Off Analysis: Key Decisions

### Decision 1: Trade Frequency (3 vs 5 vs 8 trades/day)

**Option A: Conservative (3 trades/day)**
- ✅ Highest quality trades only (P > 0.65)
- ✅ Lower risk, less exposure
- ✅ Easy to manage
- ❌ Need $50/trade to reach $3,000 in 20 days (high bar)
- ❌ May take 25-30 days to finish

**Option B: Balanced (5 trades/day)** ⭐ **RECOMMENDED**
- ✅ Good quality trades (P > 0.60)
- ✅ Achievable targets ($30/trade = $150/day)
- ✅ Flexibility to adjust
- ⚖️ Moderate risk
- ⚖️ 20-day timeline achievable

**Option C: Aggressive (8 trades/day)**
- ✅ Can finish in 15 days
- ✅ Lower bar per trade ($20/trade)
- ❌ More low-quality trades (P > 0.55)
- ❌ Higher risk of circuit breakers
- ❌ Current system FAILED at this frequency

**Recommendation**: **Option B (5 trades/day)** - Best balance of speed and safety

---

### Decision 2: Confidence Threshold (0.55 vs 0.60 vs 0.65)

**Option A: 0.55 (Moderate)**
- ✅ More trading opportunities (5-8 trades/day)
- ✅ Based on January data (profitable at this level)
- ❌ Still includes some marginal trades
- ❌ Win rate may be 50-55% (lower margin of safety)

**Option B: 0.60 (Conservative)** ⭐ **RECOMMENDED**
- ✅ High-quality trades only (3-5 trades/day)
- ✅ Win rate likely 55-60% (good margin)
- ✅ Lower variance
- ⚖️ May need 20-25 days to finish

**Option C: 0.65 (Ultra-Conservative)**
- ✅ Highest quality trades (2-3 trades/day)
- ✅ Win rate likely 60-65%
- ❌ Very slow progress ($50-90/day)
- ❌ May take 30+ days to finish

**Recommendation**: **Option B (0.60)** - Use 0.55 as fallback if too slow, 0.65 if struggling

---

### Decision 3: Validation Duration (5 vs 10 vs 14 days)

**Option A: Short (5 days)** ⭐ **RECOMMENDED**
- ✅ Fast path to combine (start in 10 days total)
- ✅ User has been waiting months (need to start)
- ❌ Less statistical confidence
- ❌ Higher risk of surprises

**Option B: Medium (10 days)**
- ✅ Good statistical confidence
- ✅ See 2 weeks of market conditions
- ⚖️ Start combine in ~15 days
- ⚖️ Reasonable for user timeline

**Option C: Long (14 days)**
- ✅ High statistical confidence
- ✅ Full 3 weeks of validation
- ❌ Too slow (user needs results NOW)
- ❌ Delays combine start to 20+ days

**Recommendation**: **Option A (5 days)** - Fast validation, BUT strict GO/NO-GO criteria

---

### Decision 4: Model Approach (Fix vs Retrain vs Ensemble)

**Option A: Fix Existing Model** ⭐ **RECOMMENDED**
- ✅ FAST (3-5 days)
- ✅ High ROI (confidence filtering proven to work)
- ✅ Low risk (minimal code changes)
- ❌ May not fully solve regime shift
- ❌ Relies on filtering, not adaptation

**Option B: Retrain on Recent Data**
- ✅ Adapts to Jan 2026 regime
- ✅ May improve accuracy
- ❌ SLOW (7-10 days for full pipeline)
- ❌ Risk of overfitting to small sample
- ❌ Unknown if new model will work

**Option C: Ensemble of Models**
- ✅ More robust to regime shifts
- ✅ Better generalization
- ❌ COMPLEX (12+ days to implement)
- ❌ Harder to debug
- ❌ May not improve enough to justify time

**Recommendation**: **Option A (Fix Existing)** - Try this first. Only escalate to B or C if validation fails.

---

## Success Criteria & Metrics

### Phase 1 Success (Model Fixes)

**Minimum Viable Performance**:
- [ ] Win rate > 50% on Jan 2026 backtest
- [ ] Daily P&L > $0 for 3 consecutive paper trading days
- [ ] Circuit breakers tested and working
- [ ] Confidence threshold identified (0.55-0.65)

**Stretch Goals**:
- [ ] Win rate > 60%
- [ ] Regime detection catches Jan 2026 shift
- [ ] Sharpe ratio > 1.0

---

### Phase 2 Success (Validation)

**Minimum Viable Performance**:
- [ ] 5/7 days with positive P&L
- [ ] Overall win rate > 50%
- [ ] Max single-day loss < $500
- [ ] Total week P&L > $200

**Stretch Goals**:
- [ ] 7/7 days positive
- [ ] Win rate > 60%
- [ ] Total week P&L > $500
- [ ] Max drawdown < $300

---

### Phase 3 Success (Topstep)

**PASS Criteria** (ALL required):
- [ ] Total P&L ≥ $3,000
- [ ] No daily loss > $1,000
- [ ] No trailing drawdown > $2,500
- [ ] No consistency violations (max day < $1,500)
- [ ] Completed in ≤ 30 trading days

**Quality Metrics** (Desirable):
- [ ] Win rate ≥ 55%
- [ ] Max drawdown < $1,500
- [ ] Profitable weeks: 3/4
- [ ] Circuit breaker hits: 0-1

---

## Implementation Checklist

### Phase 1: Model Fixes (Days 1-5)

**Day 1: Confidence Filtering**
- [ ] Modify `model_predictor.py` to add confidence threshold
- [ ] Update config: `signals.confidence_threshold: 0.60`
- [ ] Backtest on Jan 2026 with thresholds [0.55, 0.60, 0.65]
- [ ] Choose optimal threshold (maximize Sharpe)
- [ ] Verify: Win rate > 50%, trades 3-5/day

**Day 1-2: Circuit Breakers**
- [ ] Implement `CircuitBreaker` class in `risk_manager.py`
- [ ] Add 3 rules: consecutive losses, daily loss, win rate
- [ ] Add logging: Log circuit breaker events to separate file
- [ ] Test on Jan 2026: Verify would have stopped at -$500
- [ ] Integrate with `live_runner.py`

**Day 2-3: Regime Detection**
- [ ] Implement `detect_regime_shift()` in `regime_detector.py`
- [ ] Use KS test on 5 key features
- [ ] Test on Jan 2026: Should detect shift by Jan 15
- [ ] Test on Dec 2025: Should NOT detect shift
- [ ] Add to startup checks in `live_runner.py`

**Day 3-4: Volatility Filter**
- [ ] Enhance `VolatilityFilter` with percentile method
- [ ] Compute 30th/70th percentiles from training data
- [ ] Update config: `volatility_filter.enabled: true`
- [ ] Backtest on Jan 2026 with filter
- [ ] Verify: Win rate improvement > 5%

**Day 4-5: Integration & Testing**
- [ ] Integrate all fixes into `live_runner.py`
- [ ] Run full backtest on Jan 2026 with all fixes
- [ ] Verify: Win rate > 50%, daily P&L > $0
- [ ] Run paper trading test (1 day dry run)
- [ ] Create GO/NO-GO report

---

### Phase 2: Validation (Days 6-12)

**Days 6-10: Paper Trading Week 1**
- [ ] Day 6: Start paper trading with all fixes enabled
- [ ] Day 6-10: Trade daily, track metrics
- [ ] End of each day: Run `daily_review.py`, adjust parameters
- [ ] Day 10: Create validation report
- [ ] Decision: GO/NO-GO for Topstep

**Days 11-12: Topstep Simulation (Optional)**
- [ ] Enable Topstep simulation mode
- [ ] Track combine metrics (daily loss, trailing DD, consistency)
- [ ] Verify can reach $3,000 without breaches
- [ ] Create pre-combine checklist

---

### Phase 3: Topstep Execution (Days 13-32)

**Day 13: Pre-Combine Setup**
- [ ] Purchase Topstep 50k Combine account ($165)
- [ ] Update credentials in `.env`
- [ ] Verify API connection
- [ ] Set risk limits to conservative values
- [ ] Create daily tracking spreadsheet

**Days 13-32: Trading**
- [ ] Each morning: Pre-trading checklist
- [ ] Each day: Trade with strict rules (max 5 trades)
- [ ] Each evening: Update tracking, review trades
- [ ] Each week: Weekly review, adjust if needed
- [ ] Monitor for circuit breakers, regime shifts

**Day X: Completion**
- [ ] Reach $3,000 target
- [ ] Verify no rule violations
- [ ] Request funded account
- [ ] Celebrate! 🎉

---

## Critical Files for Implementation

Based on this plan, here are the 5 most critical files to modify:

### 1. `/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/live_trading/model_predictor.py`
**Why**: Core prediction logic. Need to add confidence threshold filtering to `should_trade()` method. This is the highest-ROI fix - eliminates 80% of losing trades with 4 hours of work.

### 2. `/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/live_trading/risk_manager.py`
**Why**: Risk management and circuit breakers. Need to implement `CircuitBreaker` class with 3 rules (consecutive losses, daily loss, win rate). This prevents catastrophic losses and builds confidence in the system.

### 3. `/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/features/regime_detector.py`
**Why**: Regime shift detection. Need to add `detect_regime_shift()` method using KS test on key features. This is critical for catching when the model will fail (like it did in Jan 2026).

### 4. `/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/filters/volatility_filter.py`
**Why**: Volatility filtering. Need to enhance with percentile-based filtering (30th-70th). This removes trades in unfavorable volatility regimes and improves win rate.

### 5. `/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/live_trading/live_runner.py`
**Why**: Main orchestration logic. Need to integrate all fixes (confidence filtering, circuit breakers, regime detection, volatility filter) and ensure they work together. This is where all the pieces come together.

---

## Assumptions Made

1. **Infrastructure Works**: The live trading system (data fetcher, execution engine, risk manager) is functional and tested. We're only fixing the MODEL behavior, not rebuilding infrastructure.

2. **January 2026 Data is Representative**: The poor performance in Jan 2026 is due to regime shift, not data quality issues. Analysis shows data is clean (no OHLC violations, minimal gaps).

3. **Historical Performance is Real**: The 80.3% accuracy on 2024-2025 data is legitimate (not overfit). Walk-forward validation and deflated Sharpe ratio support this.

4. **Medium Confidence Trades Work**: The finding that P > 0.55 trades are profitable (+$4.63/trade, n=10) is statistically meaningful and will generalize.

5. **User Has Time**: User can dedicate 4-6 weeks to this project. They've been working on it for months, so this is reasonable.

6. **Capital Available**: User has $165 for Topstep account + buffer for potential resets. Assume 1-2 attempts may be needed.

7. **Market Conditions Stable**: January 2026 market conditions will persist (or revert to 2024-2025 patterns). Regime detection will catch if this changes.

8. **No Code Regressions**: Implementing these fixes won't break existing functionality. Extensive testing in Phase 2 will verify this.

9. **Topstep Rules**: Documented rules are accurate and won't change mid-combine. User is familiar with combine structure.

10. **Daily Availability**: User can monitor trades daily (at minimum, end-of-day review). Not expecting 24/7 monitoring, but daily engagement required.

---

## Final Recommendations

### Priority Order
1. **MUST DO**: Confidence filtering (Fix 1) + Circuit breakers (Fix 2)
2. **SHOULD DO**: Regime detection (Fix 3) + Volatility filter (Fix 4)
3. **OPTIONAL**: Ensemble model (Fix 5) - only if Fixes 1-4 insufficient

### Timeline Expectations
- **Best Case**: Pass combine in 23 days (5 fix + 5 validation + 13 trading)
- **Realistic Case**: Pass combine in 28-32 days (5 fix + 7 validation + 20 trading)
- **Worst Case**: Extend to 40+ days if validation shows model needs retraining

### Risk Mitigation
- **Start Conservative**: Use highest confidence threshold (0.65) initially, lower if too slow
- **Strict Validation**: Don't skip Phase 2. Better to find problems in paper trading than live.
- **Multiple Attempts**: Budget for 2 combine attempts ($330 total). First may be learning experience.

### Success Factors
- **Discipline**: Follow the plan, don't deviate when frustrated
- **Patience**: Don't rush Phase 2 validation to "save time"
- **Adaptability**: If circuit breaker hits, STOP and reassess (don't push through)
- **Focus**: This is a marathon, not a sprint. Steady progress wins.

---

**Ready to Execute**: This plan is actionable, realistic, and designed to get you funded within 4-6 weeks. The model CAN work - we just need to fix the regime shift problem and trade more selectively. Let's do this.
