# Prompt for Claude to Fix Overfitting in ML Trading Model

Copy and paste this entire prompt to Claude Code to fix the overfitting issues:

---

# Task: Fix Severe Overfitting in Bidirectional Trading Model

## Context

I have a machine learning trading model for ES futures (Topstep 50k combine) that shows **severe overfitting**:

**Current Model Location**: `ml_intraday_v3/models/saved/model_bundle.pkl`

**Training Run**: `runs/bidirectional_24h_20260114/bar_size=5m/`

**Problem Summary**:
- **Cross-validation results show 124% coefficient of variation** (anything >50% is concerning)
- **2 out of 6 folds lose money** (fold_0: -$1,474, fold_1: -$1,080)
- **Fold 4 drives 58% of profit** ($50,048 out of $86,199 total)
- **8 days exceeded $1,000 daily loss limit** (worst: -$1,865 on 2023-04-25)
- **Max drawdown: $4,082** (Topstep limit: $2,500) - **WOULD FAIL COMBINE**

**K-Fold Backtest Results**:
```
Fold 0: -$1,474 (47.9% win rate, 359 trades)
Fold 1: -$1,080 (53.2% win rate, 453 trades)
Fold 2: +$7,716 (53.2% win rate, 832 trades)
Fold 3: +$22,067 (55.3% win rate, 929 trades)
Fold 4: +$50,048 (55.0% win rate, 1,362 trades) ← PROBLEM
Fold 5: +$8,922 (60.1% win rate, 213 trades)
```

## Your Mission

**Implement ALL of the following fixes and provide a comprehensive analysis**:

### 1. Update Topstep Combine Notebook

**File**: `analysis/topstep_50k_combine_test.ipynb`

**Tasks**:
- Update notebook to point to new run: `runs/bidirectional_24h_20260114/bar_size=5m/backtests/purged_kfold`
- Load executed trades from all 6 folds
- Run Monte Carlo simulation (10,000 iterations) with Topstep 50K rules:
  - Starting balance: $50,000
  - Profit target: $3,000
  - Daily loss limit: $1,000
  - Trailing max drawdown: $2,500
  - Consistency: No single day > 50% of total profit
- Generate:
  - Pass rate percentage
  - Median days to pass (if successful)
  - Failure breakdown (daily loss vs trailing DD)
  - Equity curve visualization
  - Drawdown analysis
- Save results to: `analysis/topstep_50k_results_bidirectional_24h.json`

**Expected Output**:
- Updated notebook that runs successfully
- JSON results file
- Pass rate estimate (likely 20-40% based on current metrics)

---

### 2. Implement Ensemble Model (Short-term Fix)

**Goal**: Reduce overfitting by ensembling across all 6 fold models instead of using a single model.

**Tasks**:
- Load all 6 fold models from: `runs/bidirectional_24h_20260114/bar_size=5m/training/purged_kfold/fold_{0-5}/bundle.pkl`
- Create ensemble predictor that:
  - Runs prediction through all 6 models
  - Averages probabilities across models
  - For bidirectional: evaluates both LONG and SHORT using averaged probs
  - Chooses side with highest expected value
- Save ensemble bundle to: `ml_intraday_v3/models/saved/model_bundle_ensemble.pkl`
- Update bundle format to match LiveModelPredictor expectations:
  ```python
  {
    'primary_model': ensemble_model,  # your ensemble wrapper
    'primary_preprocessor': preprocessor_state,
    'primary_feature_columns': feature_columns,
    'thresholds': {'primary_threshold': 0.03},
    'model_type': 'LGBMEnsemble',
    'n_base_models': 6,
  }
  ```
- Backtest the ensemble on the same data
- Compare performance vs single model

**Expected Impact**:
- Smoother performance across time periods
- Reduced variance (coefficient of variation < 80%)
- More stable daily PnL distribution

---

### 3. Retrain with Better Regularization (Long-term Fix)

**Goal**: Train a new model with stronger regularization to prevent overfitting.

**Current Training Config**: `ml_intraday_v3/configs/training.yaml`

**Tasks**:

a) **Create new training config**: `ml_intraday_v3/configs/training_regularized.yaml`

Update LightGBM parameters:
```yaml
model:
  type: lightgbm
  params:
    n_estimators: 200  # Reduce from 500
    max_depth: 6  # Add depth limit (was unlimited)
    learning_rate: 0.03  # Reduce from 0.05
    num_leaves: 31  # Reduce from 63
    min_child_samples: 100  # Increase from 20
    min_split_gain: 0.01  # Add minimum gain
    subsample: 0.8  # Add row sampling
    colsample_bytree: 0.8  # Add column sampling
    reg_alpha: 0.1  # Add L1 regularization
    reg_lambda: 1.0  # Add L2 regularization
    class_weight: balanced
    random_state: 42
```

b) **Retrain model**:
- Use notebook: `ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb`
- Set RUN_ID: `"regularized_24h_20260114"`
- Use BAR_SIZE: `"5m"`
- Use CV_KIND: `"purged_kfold"`
- Copy training_regularized.yaml to training.yaml (or modify notebook to load it)
- Run full pipeline (takes ~2-3 hours)

c) **Validate regularization worked**:
- Check K-fold performance consistency
- Calculate new coefficient of variation (target: <60%)
- Verify no fold loses money
- Check max drawdown across folds
- Verify daily loss violations reduced

**Expected Results**:
- More consistent fold performance
- Lower max drawdown
- Fewer daily loss violations
- Slightly lower overall PnL but more stable
- Better Topstep pass rate (target: >60%)

---

### 4. Add Risk Management to Live Trading

**Goal**: Add circuit breakers to prevent Topstep violations even if model is imperfect.

**Files to Modify**:
- `ml_intraday_v3/configs/live_trading.yaml`
- `ml_intraday_v3/configs/risk.yaml` (if exists, else create it)
- `ml_intraday_v3/live_trading/live_runner.py`

**Tasks**:

a) **Create/update risk config**: `ml_intraday_v3/configs/risk.yaml`
```yaml
risk_management:
  enabled: true

  # Daily loss limits
  daily_loss_limit_usd: 1000  # Hard stop (Topstep rule)
  daily_loss_warning_usd: 500  # Soft stop (reduce size)
  daily_loss_critical_usd: 750  # Stop new trades

  # Position sizing
  base_position_size: 1  # contracts
  reduce_size_on_loss: true
  size_after_warning: 0.5  # 50% size after -$500
  size_after_critical: 0  # No new trades after -$750

  # Trailing drawdown
  max_drawdown_from_hwm: 2500  # Topstep limit
  drawdown_warning: 1500  # Warning at 60% of limit

  # Time filters
  avoid_first_hour: true  # Skip 17:00-18:00 ET
  avoid_last_hour: true  # Skip 15:00-16:00 ET

  # Threshold adjustments
  threshold_increase_on_loss: 0.02  # Raise threshold after losses
```

b) **Implement risk manager class**:

Create: `ml_intraday_v3/live_trading/risk_manager.py`

```python
class RiskManager:
    def __init__(self, config_path: Path):
        # Load risk config
        # Track daily P&L
        # Track high water mark
        pass

    def update_pnl(self, trade_pnl: float):
        # Update daily P&L and equity
        pass

    def check_trading_allowed(self) -> tuple[bool, str]:
        # Returns (allowed, reason)
        # Check daily loss limit
        # Check trailing drawdown
        # Check time filters
        pass

    def get_position_size(self, base_size: int) -> int:
        # Return adjusted position size based on daily P&L
        pass

    def get_threshold_adjustment(self) -> float:
        # Return threshold increase if in loss
        pass

    def reset_daily(self):
        # Reset daily counters at session end
        pass
```

c) **Integrate into live_runner.py**:
- Add RiskManager initialization
- Check `risk_manager.check_trading_allowed()` before each trade
- Use `risk_manager.get_position_size()` for sizing
- Add threshold adjustment to signal threshold
- Update P&L after each trade execution
- Add daily reset at session rollover

**Expected Impact**:
- No daily loss over $1,000 (guaranteed)
- Drawdown capped closer to $2,000-$2,500
- Position sizing reduces risk during losing periods
- Fewer trades during high-risk time periods

---

### 5. Comparative Analysis & Report

**Tasks**:

a) **Compare all approaches**:

Run backtests for:
1. Original single model (fold_0)
2. Ensemble model (6 folds averaged)
3. Regularized model (new training)

For each, calculate:
- Total PnL
- Sharpe ratio
- Max drawdown
- Daily loss violations (>$1,000)
- K-Fold coefficient of variation
- Win rate and profit factor
- Topstep Monte Carlo pass rate

b) **Generate comparison report**: `analysis/overfitting_fixes_comparison.md`

Include:
- Side-by-side performance metrics table
- Equity curves comparison
- Drawdown charts
- Risk metrics table
- Recommendation: which model to use

c) **Update CURRENT_SETUP_STATUS.md**:
- Replace current model info
- Add risk management status
- Update expected performance
- Add warning about which model NOT to use

---

## Deliverables

When you're done, provide:

1. **Updated Topstep notebook** with Monte Carlo results for current model
2. **Ensemble model** at `ml_intraday_v3/models/saved/model_bundle_ensemble.pkl`
3. **Regularized model** trained and saved (if training completes)
4. **Risk manager** implemented and integrated
5. **Comprehensive comparison report** showing which approach is best
6. **Updated documentation** with recommendations

## Critical Requirements

- ✅ All code must be compatible with existing `LiveModelPredictor` class
- ✅ Ensemble model must support bidirectional evaluation (LONG/SHORT)
- ✅ Risk manager must integrate cleanly with existing live_runner.py
- ✅ All changes must be backwards compatible (don't break existing code)
- ✅ Follow existing code style and patterns in ml_intraday_v3/
- ✅ All configs must be in YAML format matching existing patterns
- ✅ Test each component before declaring complete

## Files You'll Need to Read

Key files for context:
- `OVERFITTING_ANALYSIS.md` - Detailed problem analysis
- `CURRENT_SETUP_STATUS.md` - Current model setup
- `ml_intraday_v3/live_trading/model_predictor.py` - How predictions work
- `ml_intraday_v3/live_trading/live_runner.py` - Main trading loop
- `ml_intraday_v3/configs/training.yaml` - Current training config
- `ml_intraday_v3/configs/live_trading.yaml` - Current live config
- `analysis/topstep_50k_combine_test.ipynb` - Topstep simulator

## Success Criteria

A successful fix means:
- ✅ Topstep Monte Carlo pass rate > 60% (currently ~20-40%)
- ✅ K-Fold CV < 60% (currently 124%)
- ✅ No fold loses money
- ✅ Daily loss violations < 2% of days (currently 1.7%)
- ✅ Max drawdown < $2,500 (currently $4,082)
- ✅ Risk manager prevents any Topstep rule violation

## Priority Order

If you can't complete everything:

**Priority 1 (MUST DO)**:
1. Update Topstep notebook and get current pass rate
2. Implement risk manager with circuit breakers
3. Generate comparison report

**Priority 2 (SHOULD DO)**:
4. Create ensemble model
5. Backtest ensemble

**Priority 3 (NICE TO HAVE)**:
6. Retrain with regularization (takes 2-3 hours)
7. Full comparative analysis

## Questions to Ask Me

If anything is unclear:
- Should I use a specific model selection criteria for ensemble?
- What should I do if training takes longer than expected?
- Do you want me to paper trade test any of the fixes?
- Should I create a separate config file or modify existing ones?

---

## Start Here

Begin by:
1. Reading `OVERFITTING_ANALYSIS.md` to understand the full problem
2. Checking the current model performance in the backtest results
3. Updating the Topstep notebook to quantify the risk
4. Then proceed with fixes in priority order

Let me know when you've completed each section and I'll review before you move to the next!

---

**IMPORTANT**: This is for a TOPSTEP COMBINE where I have:
- $1,000 daily loss limit (HARD STOP)
- $2,500 trailing drawdown limit (HARD STOP)
- $3,000 profit target
- Must maintain consistency (no single day > 50% of profit)

Any model that violates these limits would immediately fail the combine. The goal is to pass the combine within 30-40 days with high confidence (>60% Monte Carlo pass rate).

Good luck! 🚀
