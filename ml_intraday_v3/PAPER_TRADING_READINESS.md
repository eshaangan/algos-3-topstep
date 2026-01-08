# Paper Trading Readiness Checklist
**Date**: January 7, 2026
**Run**: v3_2022_5m (2022+ data, 5m bars)

## Executive Summary

### Model Performance
| Metric | K-Fold | Walk-Forward | Target |
|--------|--------|--------------|--------|
| **Total PnL** | +$37,459 | +$22,249 | > $0 ✓ |
| **Profitable Periods** | 6/6 (100%) | 13/15 (87%) | > 60% ✓ |
| **Win Rate** | ~59% | ~59% | > 50% ✓ |
| **DD Breaches** | 0 | 0 | 0 ✓ |

### Topstep 50K Combine Simulation
- **Monte Carlo Pass Rate**: 77.8% (10,000 simulations)
- **Sequential Test**: PASSED in 37 days
- **Daily Loss Risk**: 2 days out of 283 exceeded $1,000 limit (0.7%)

### Overfitting Analysis
✓ **VERDICT: LOW OVERFITTING RISK**
- Walk-forward retains 59% of K-fold performance (good generalization)
- DSR = 1.00 (strong evidence of skill after selection bias correction)
- No time decay (2nd half performs better than 1st)
- Stable CV metrics (CoV = 0.13)

---

## Phase 1: Pre-Paper Trading (COMPLETED ✓)

### 1.1 Model Training ✓
- [x] 2022+ data filtering implemented
- [x] 5m bars (99.8% feature completeness vs 5.9% for 1m)
- [x] HMM regime disabled (too noisy for intraday)
- [x] Walk-forward validation added
- [x] All 6 K-fold splits profitable
- [x] 13/15 walk-forward windows profitable

### 1.2 Validation ✓
- [x] Purged K-fold cross-validation
- [x] Walk-forward validation (expanding window)
- [x] Deflated Sharpe Ratio computed
- [x] Topstep combine simulation (Monte Carlo + sequential)
- [x] Overfitting diagnostics passed

### 1.3 Risk Management ✓
- [x] Topstep 50K rules integrated
- [x] Daily loss limit: $1,000
- [x] Trailing max drawdown: $2,500
- [x] Position limit: 1 contract
- [x] Max concurrent positions: 5
- [x] Flatten time: 15:55 Chicago

---

## Phase 2: Paper Trading Setup (NEXT STEPS)

### 2.1 Infrastructure Setup

#### A. Model Bundle Creation
```bash
# Create production model bundle
python3 -m ml_intraday_v3.create_live_model_bundle \
    --run-dir runs/v3_2022_5m \
    --bar-size 5m \
    --output-dir live_models/v3_2022_5m_prod
```

**Bundle should include:**
- [ ] Trained models (fold_0 through fold_5)
- [ ] Calibrators (isotonic)
- [ ] Scalers and preprocessors
- [ ] Feature schema
- [ ] Config snapshots

#### B. Data Pipeline
- [ ] Live data feed connection (ProjectX or similar)
- [ ] 5m bar aggregation from tick/1m data
- [ ] Feature computation pipeline
- [ ] Data quality checks (NaN detection)

#### C. Execution Infrastructure
- [ ] Broker connection (paper trading account)
- [ ] Order management system
- [ ] Position tracking
- [ ] Fill simulation (slippage model)

### 2.2 Monitoring Dashboard

#### Required Metrics (Real-time)
- [ ] Daily PnL
- [ ] High water mark
- [ ] Current drawdown from HWM
- [ ] Daily loss (from session start)
- [ ] Open positions
- [ ] Win rate (rolling 20 trades)
- [ ] Trade count today
- [ ] Model score distribution

#### Alerts
- [ ] Daily loss approaching $800 (80% of $1,000 limit)
- [ ] Trailing DD approaching $2,000 (80% of $2,500 limit)
- [ ] Feature NaN detected
- [ ] Model score outside expected range
- [ ] No trades for 2+ hours during market hours
- [ ] Position stuck (> 30 min)

### 2.3 Testing Protocol

#### Dry Run (No Real Orders)
**Duration**: 5 trading days
**Purpose**: Validate infrastructure without placing orders

- [ ] Day 1: Feature computation only
  - Verify features match backtest
  - Check for NaN patterns
  - Validate bar timing (exactly on 5m close)

- [ ] Day 2-3: Model scoring
  - Generate predictions
  - Compare scores to backtest ranges
  - Log decision logic (trade/skip)

- [ ] Day 4-5: Full simulation
  - Simulate orders (but don't send)
  - Track simulated PnL
  - Verify risk gates work

#### Paper Trading Phase 1
**Duration**: 20 trading days
**Goal**: Match backtest statistics

**Success Criteria:**
- Win rate within ±10% of backtest (target: 50-70%)
- Average trade PnL within ±30% of backtest (target: $25-50)
- Trade frequency: 10-20 trades/day
- Zero missed signals (technical failures)
- Zero risk breaches

**Failure Criteria (STOP IMMEDIATELY):**
- 3+ consecutive days of losses > $500
- Daily loss limit breached
- Trailing DD limit breached
- Win rate < 40% for 10+ consecutive days
- Technical failure rate > 5%

#### Paper Trading Phase 2
**Duration**: 30 trading days
**Goal**: Pass simulated Topstep combine

**Success Criteria:**
- Reach +$3,000 profit target
- No daily loss > $1,000
- No trailing DD > $2,500
- Consistency rule met (no day > 50% of profit)

---

## Phase 3: Live Trading Preparation

### 3.1 Final Validation
- [ ] Paper trading Phase 1 completed successfully
- [ ] Paper trading Phase 2 completed successfully
- [ ] All monitoring alerts tested
- [ ] Manual kill switch tested
- [ ] Disaster recovery plan documented

### 3.2 Capital Allocation
- [ ] Topstep 50K Combine account funded ($165)
- [ ] Emergency stop-loss procedures established
- [ ] Performance review schedule (daily for first week)

### 3.3 Risk Limits (Conservative Start)
**First Week:**
- Max position size: 1 contract (same as backtest)
- Max concurrent positions: 3 (reduced from 5)
- Stop trading if daily loss > $700 (70% of limit)
- Stop trading if trailing DD > $2,000 (80% of limit)

**After First Week (if stable):**
- Gradually increase to backtest limits
- Monitor for regime changes

---

## Known Risks & Mitigation

### Model Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Regime change** | Medium | High | Monitor performance weekly, retrain monthly |
| **Overfitting (despite tests)** | Low | High | Paper trading validation |
| **Slippage worse than backtest** | Medium | Medium | Start with limit orders, measure actual fills |

### Operational Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Data feed failure** | Low | High | Redundant data sources, heartbeat monitoring |
| **Broker connection loss** | Low | High | Auto-flatten on disconnect |
| **Feature computation error** | Low | High | Pre-trade validation, NaN checks |
| **Position limit exceeded** | Very Low | Medium | Hard limit in code |

### Market Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Flash crash** | Very Low | High | Wide stop losses, position limits |
| **Low liquidity (after hours)** | Medium | Medium | Flatten before 15:55 |
| **Gap opening** | Low | Medium | Overnight flatten (already enforced) |

---

## Critical Files & Paths

### Model Artifacts
```
runs/v3_2022_5m/
├── bar_size=5m/
│   ├── training/purged_kfold/
│   │   ├── fold_0/ ... fold_5/  # Trained models
│   ├── backtests/purged_kfold/
│   │   └── summary.json          # Backtest results
├── walkforward/bar_size=5m/
│   └── summary.json               # Walk-forward results
```

### Configuration
```
ml_intraday_v3/configs/
├── data.yaml           # Data filtering, bar sizes
├── risk.yaml           # Topstep rules, position limits
├── execution_spec.yaml # Slippage, costs
├── backtest.yaml       # Decision thresholds, regime filters
└── live_trading.yaml   # Live trading config (to be created)
```

### Live Trading Code
```
ml_intraday_v3/live_trading/
├── execution_engine.py    # Order execution
├── risk_manager.py        # Real-time risk checks
├── data_pipeline.py       # Live data feed
├── model_server.py        # Model inference
└── monitoring.py          # Dashboard & alerts
```

---

## Appendix: Quick Commands

### Check Current Run Status
```bash
# View summary
python3 -c "
import json
kf = json.load(open('runs/v3_2022_5m/bar_size=5m/backtests/purged_kfold/summary.json'))
wf = json.load(open('runs/v3_2022_5m/walkforward/bar_size=5m/summary.json'))
print(f'K-Fold: \${sum(f[\"total_pnl_usd\"] for f in kf[\"metrics_by_split\"]):,.0f}')
print(f'Walk-Forward: \${sum(w[\"total_pnl_usd\"] for w in wf[\"metrics_by_window\"]):,.0f}')
"
```

### Run Topstep Simulation
```bash
cd analysis
jupyter notebook topstep_50k_combine_test.ipynb
```

### Check for NaN Features
```bash
python3 -c "
import pandas as pd
feat = pd.read_parquet('runs/v3_2022_5m/bar_size=5m/features.parquet')
print(f'Usable: {feat[\"usable_for_training\"].mean():.1%}')
"
```

---

## Decision: Ready for Paper Trading?

### Checklist
- [x] Model profitable in K-fold validation
- [x] Model profitable in walk-forward validation
- [x] Topstep simulation shows >50% pass rate
- [x] Overfitting tests passed
- [x] Infrastructure code exists
- [ ] **Data feed integrated** ← NEXT STEP
- [ ] **Paper trading infrastructure tested** ← NEXT STEP
- [ ] **Monitoring dashboard built** ← NEXT STEP

### Recommendation
**Status**: Ready for Phase 2 (Paper Trading Setup)

**Next Immediate Steps:**
1. Test live data feed connection (ProjectX or broker API)
2. Build 5m bar aggregation from live ticks
3. Create model bundle for production
4. Set up monitoring dashboard (even if simple)
5. Run 5-day dry run (no orders)

**Timeline Estimate:**
- Infrastructure setup: 2-3 days
- Dry run testing: 5 trading days
- Paper trading Phase 1: 20 trading days
- Paper trading Phase 2: 30 trading days

**Total to live trading**: ~60 trading days (12 weeks)

---

**Last Updated**: January 7, 2026
**Author**: ML Pipeline V3 Team
