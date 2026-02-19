# Final Model Selection & Validation Plan

## Current Status
- **Total experiments**: 412/610 (67% complete, Batch 5 running)
- **Best AUC found**: 0.593
- **Models with AUC > 0.50**: 308 (75%)
- **Original problem**: SOLVED (models now produce diverse signals vs identical -$1,212.56 PnL)

## Phase 1: Model Selection Criteria (When Batch 5 Completes)

### Primary Ranking Metric
```
Final Score = 0.40 * AUC
            + 0.30 * Signal_Quality
            + 0.20 * Calibration
            + 0.10 * Stability
```

**Components:**
1. **AUC (40%)**: Out-of-sample predictive power
2. **Signal Quality (30%)**: % predictions >0.55 confidence (actionability)
3. **Calibration (20%)**: Brier score (probability accuracy)
4. **Stability (10%)**: 1 / std(test_auc) across folds (consistency)

### Selection Process
1. Filter to models with AUC > 0.52 (top ~25%)
2. Rank by Final Score
3. Select top 10 candidates
4. Check for diversity (different approaches)

### Minimum Requirements
- ✅ AUC > 0.52 (meaningful edge)
- ✅ Signal rate >0.15 (at least 15% actionable)
- ✅ Trades/day > 2 (sufficient opportunities)
- ✅ 5/5 successful CV folds (no failures)
- ✅ Train-test gap < 0.40 (not overfitting)

## Phase 2: True Out-of-Sample Validation

### Data Splits
**Training Data** (used by all experiments):
- Oct 2024 - Dec 2025 (14 months)
- Used for model training and CV

**Validation Data** (partially seen):
- Jan - Feb 2026 (2 months)
- Used in Batch 4 forward test
- **Status**: Models tested, but results are CV metrics only (no backtest PnL)

**True OOS Test Data** (NEVER SEEN):
- **Feb 11 - Mar 31, 2026** (7 weeks)
- Will be used for final validation
- **Critical**: Must show positive PnL to deploy

### Validation Steps
1. **Download/prepare truly unseen data** (Feb 11 - Mar 31, 2026)
2. **Retrain top 10 models** on full training data (Oct 2024 - Dec 2025)
3. **Generate predictions** on true OOS period
4. **Run full backtest** with:
   - Entry at model signal >0.55
   - Exit: PT=2.0x ATR, SL=1.5x ATR, Time=24 bars
   - Risk: $100 max per trade
   - Topstep rules: Daily loss limit, trailing DD

### Success Criteria for True OOS
Must achieve **ALL** of these:
- ✅ **Positive PnL** (any amount > $0)
- ✅ **Win rate > 45%** (better than random)
- ✅ **Max drawdown < $1,000** (Topstep limit)
- ✅ **Sharpe ratio > 0.5** (risk-adjusted return)
- ✅ **Different signals than baseline** (proves model matters)

### If OOS Validation Fails
**Plan B**: Fall back to rule-based system in `rule_based_v1/`
- Already built and tested
- 106 tests passing
- Uses EMA trend + confirmations
- No ML dependency

## Phase 3: Model Comparison & Selection

### Compare Top 10 Models
For each of the 10 candidates, calculate:

| Metric | Weight | Description |
|--------|--------|-------------|
| True OOS PnL | 40% | Actual $ on unseen data |
| True OOS Sharpe | 25% | Risk-adjusted performance |
| True OOS Win Rate | 15% | Consistency |
| CV AUC | 10% | Training signal |
| Max Drawdown | 10% | Risk control |

**Selection**: Highest weighted score wins

### Ensemble Option (if close race)
If top 3 models are within 5% of each other:
- Consider ensemble (average predictions)
- Test ensemble on true OOS data
- Select ensemble if it outperforms individuals

## Phase 4: Final Model Preparation

### Model Artifacts to Save
1. **Trained model file** (`.pkl`)
   - LightGBM model
   - Calibration wrapper (if used)
   - Feature columns list
   - Preprocessing parameters

2. **Configuration**
   - Labeling method & parameters
   - Feature engineering steps
   - CV method used
   - Sample weighting approach

3. **Performance Report**
   - CV metrics (all folds)
   - True OOS metrics
   - Signal distribution
   - Trade statistics

4. **Validation Results**
   - Backtest equity curve
   - Drawdown chart
   - Win/loss distribution
   - Signal timing analysis

### Integration Checklist
- [ ] Model loads correctly in `live_trading/model_predictor.py`
- [ ] Features match between training and live
- [ ] Prediction outputs are calibrated probabilities
- [ ] Confidence threshold (0.55) is appropriate
- [ ] Risk management parameters set
- [ ] Topstep rules enforced

## Phase 5: Pre-Live Testing

### Paper Trading Validation (1-2 weeks)
**Before going live**, run paper trading:
1. Generate signals in real-time (no live orders)
2. Log all signals, features, predictions
3. Track "would-be" PnL
4. Monitor for:
   - Feature drift
   - Signal frequency (should be ~10-12/day)
   - Prediction distribution shift
   - Regime changes

### Red Flags (Stop if any occur)
- ❌ Signal frequency drops below 5/day or exceeds 20/day
- ❌ Prediction distribution shifts significantly (>20% change in mean)
- ❌ Win rate in paper trading drops below 40%
- ❌ Max drawdown exceeds $500 in first week
- ❌ Model predictions cluster at extremes (0.0 or 1.0)

### Green Lights (Proceed to live)
- ✅ Signal frequency 8-15/day (stable)
- ✅ Win rate 45-55% (consistent)
- ✅ Drawdowns controlled (<$300)
- ✅ Predictions well-distributed (0.4-0.7 range)
- ✅ No major errors or bugs in live pipeline

## Phase 6: Live Deployment (Topstep Combine)

### Initial Parameters (Conservative)
```yaml
risk:
  max_position_size: $100  # 1 MES contract
  daily_loss_limit: $400   # Topstep 50k combine limit
  max_drawdown: $1000      # Trailing drawdown limit

confidence:
  min_probability: 0.55    # Entry threshold
  min_edge: 0.05           # p_target - p_stop > 5%

position:
  profit_target: 2.0       # ATR multiple
  stop_loss: 1.5           # ATR multiple
  time_exit: 24            # bars (2 hours)
```

### Monitoring & Adaptation
**Daily checks**:
- PnL vs expected (based on backtest)
- Win rate tracking
- Drawdown monitoring
- Signal quality (prediction distribution)

**Weekly reviews**:
- Feature importance (any drift?)
- Regime detection (is market different?)
- Correlation with S&P500/VIX
- Adjust thresholds if needed

### Circuit Breakers
Auto-stop trading if:
- Daily loss hits $300 (75% of limit)
- 5 consecutive losses
- 3 consecutive days negative
- Drawdown exceeds $800
- Signal frequency anomaly (>50% change)

## Timeline

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| 1. Model Selection | 1 hour | After Batch 5 | Immediate |
| 2. True OOS Validation | 4 hours | After selection | Same day |
| 3. Model Comparison | 1 hour | After validation | Same day |
| 4. Model Prep | 2 hours | After comparison | Same day |
| 5. Paper Trading | 1-2 weeks | After prep | Before live |
| 6. Live Deployment | Ongoing | After paper success | TBD |

**Total time to live**: 2-3 weeks (including paper trading)

## Files to Create

### Immediate (while Batch 5 runs)
1. ✅ `final_model_selection_plan.md` (this file)
2. `select_top_models.py` - Implements Phase 1 selection
3. `validate_true_oos.py` - Implements Phase 2 validation
4. `prepare_final_model.py` - Implements Phase 4 preparation

### After Selection
5. `final_model_report.md` - Complete performance documentation
6. `deployment_checklist.md` - Pre-live verification steps
7. `monitoring_dashboard.py` - Real-time performance tracking

## Key Decision Points

### Decision 1: Single Model vs Ensemble
**When**: After Phase 2 (true OOS validation)
**Criteria**: If top 3 models within 5% score, test ensemble
**Default**: Single model (simpler, more maintainable)

### Decision 2: Deploy or Rule-Based Fallback
**When**: After Phase 2 (true OOS validation)
**Criteria**: Must pass ALL success criteria
**Threshold**: If ANY criterion fails, use rule-based system

### Decision 3: Paper Trading Duration
**When**: During Phase 5
**Criteria**:
- Minimum: 1 week (if all green lights)
- Extended: 2 weeks (if any yellow flags)
- Abort: If any red flags

## Risk Management Philosophy

**Priority Order**:
1. **Capital Preservation** (don't blow up account)
2. **Consistency** (steady wins > big swings)
3. **Growth** (compound returns over time)

**Topstep-Specific**:
- Daily loss limit is HARD STOP
- Trailing DD is monitored constantly
- Consistency rule: no single trade >40% of daily profit
- Must hit profit target ($3,000) before funded

## Success Definition

**Minimum Success** (Topstep Combine Pass):
- Pass 5-day evaluation without violating rules
- Hit $3,000 profit target
- No daily loss limit violations
- No consistency rule violations

**Target Success** (Funded Account):
- Maintain funded account for 90 days
- Achieve 10% account growth
- Keep max DD below 50%
- Generate consistent monthly profit

**Stretch Goal** (Scale Up):
- Prove profitability over 6 months
- Scale to 2-3 contracts
- Achieve $10k+ monthly profit
- Transition to full-time trading
