# Model Retraining Guide: Fix Distribution Shift & Directional Bias

## Problem Statement

The current model trained on Dec 2024 data has severe performance degradation on Jan 2026:

| Metric | Dec 2024 (Train) | Jan 2026 (Test) | Change |
|--------|------------------|-----------------|--------|
| Win Rate | 58.0% | 13.7% | **-44.3pp** |
| Profit Factor | 1.78 | 0.19 | **-89%** |
| Total P&L | +$1,580 | -$9,220 | **-$10,800** |
| LONG % | ~50% | 100.0% | **+50pp** |
| SHORT % | ~50% | 0.0% | **-50pp** |

### Root Causes

1. **Distribution Shift**: 13-month gap between training and test data (concept drift)
2. **Directional Bias**: Bug in `replay.py:345` causes 100% LONG signals despite model predicting both sides
3. **Poor Risk-Reward**: 86% of trades hit stop-loss (symptom of bad direction predictions)

## Solution: Walk-Forward Retraining

Train model on **recent data** (Q4 2024 + Jan 2026) to:
1. Adapt to current market regime
2. Validate bidirectional prediction works correctly
3. Improve risk-reward through better direction forecasting

### Research Backing

- **Walk-forward optimization**: Standard institutional practice ([QuantInsti](https://blog.quantinsti.com/walk-forward-optimization-introduction/))
- **Concept drift detection**: Common in financial markets ([Evidently AI](https://www.evidentlyai.com/blog/machine-learning-monitoring-data-and-concept-drift))
- **Purged k-fold CV**: Prevents leakage in time series ([QuantBeckman](https://www.quantbeckman.com/p/with-code-combinatorial-purged-cross))

## Quick Start

### Prerequisites

1. **Databento API key** in `.env`:
   ```bash
   DATABENTO_API_KEY=your_key_here
   ```

2. **Python environment** (from project root):
   ```bash
   cd ml_intraday_v3
   pip install -r requirements-mlv3.txt
   ```

### Run Retraining (Single Command)

```bash
cd ml_intraday_v3
python retrain_q4_jan26.py
```

This will:
1. ✅ Fetch Q4 2024 + Jan 2026 data from Databento (5 min)
2. ✅ Filter to RTH and generate features (2 min)
3. ✅ Train LightGBM model with same architecture as baseline (2 min)
4. ✅ Create production model bundle with `has_side_feature=True`
5. ✅ Generate deployment guide and monitoring checklist

**Total time**: ~10 minutes

### Using Cached Data

If you've already fetched data once:

```bash
python retrain_q4_jan26.py --use-cached-data
```

This skips Databento API call and uses cached parquet file.

### Custom Date Range

For monthly retraining (rolling 4-month window):

```bash
# February 2026 retraining
python retrain_q4_jan26.py \\
    --start-date 2024-11-01 \\
    --end-date 2026-02-28

# March 2026 retraining  
python retrain_q4_jan26.py \\
    --start-date 2024-12-01 \\
    --end-date 2026-03-31
```

## Validation Workflow

### Step 1: Run Backtest on Jan 2026

After retraining, validate on same Jan 2026 data that failed:

```bash
python backtest_databento_recent.py \\
    --model-bundle models/saved/model_bundle_retrained_q4_jan26.pkl \\
    --start-date 2026-01-04 \\
    --end-date 2026-01-23 \\
    --output-dir backtest_results/retrained_validation_jan26/
```

### Step 2: Check Success Criteria

Model **MUST** achieve on Jan 2026 backtest:

- [ ] **Win rate > 40%** (vs 13.7% baseline)
- [ ] **Profit factor > 1.0** (vs 0.19 baseline)
- [ ] **LONG % = 40-60%** (vs 100% baseline)
- [ ] **SHORT % = 40-60%** (vs 0% baseline)
- [ ] **Total P&L positive** (vs -$9,220 baseline)

### Step 3: Paper Trading (MANDATORY)

Before live deployment, run paper trading for 1 week:

```bash
python live_trading/paper_trade.py --duration 7d
```

Monitor daily:
- Win rate (must stay >45%)
- Profit factor (must stay >1.2)
- Direction balance (both sides 40-60%)
- Max drawdown (must be <$1,500)

## Production Deployment

**ONLY AFTER** paper trading validation passes:

### 1. Backup Old Model

```bash
cp models/saved/model_bundle.pkl \\
   models/saved/model_bundle_dec2024_baseline_backup.pkl
```

### 2. Deploy New Model

```bash
cp models/saved/model_bundle_retrained_q4_jan26.pkl \\
   models/saved/model_bundle.pkl
```

### 3. Verify Deployment

```bash
python -c "
import pickle
with open('models/saved/model_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)

print(f'has_side_feature: {bundle.get(\"has_side_feature\")}')
print(f'n_features: {len(bundle.get(\"primary_feature_columns\", []))}')
print(f'Training method: {bundle[\"metadata\"][\"training_method\"]}')

# CRITICAL CHECK
assert bundle.get('has_side_feature') is True, 'FATAL: has_side_feature must be True!'
print('✅ Deployment verified')
"
```

### 4. Restart Trading System

```bash
bash monday_startup.sh
```

### 5. Gradual Rollout

- **Week 1**: 1 micro contract ($200 risk) - monitor closely
- **Week 2**: 2 micros if profitable
- **Week 3+**: Full size (4-5 micros) if consistent

## Monthly Retraining Schedule

To keep model fresh and avoid future distribution shift:

| Month | Training Window | Command |
|-------|----------------|---------|
| Feb 2026 | Nov 2024 - Feb 2026 | `--start-date 2024-11-01 --end-date 2026-02-28` |
| Mar 2026 | Dec 2024 - Mar 2026 | `--start-date 2024-12-01 --end-date 2026-03-31` |
| Apr 2026 | Jan 2025 - Apr 2026 | `--start-date 2025-01-01 --end-date 2026-04-30` |

**Frequency**: Monthly on first weekend of month  
**Window**: Rolling 4 months (captures recent regime, not too noisy)

## Trigger-Based Retraining

Retrain **immediately** if live trading shows:

1. ⚠️ Win rate drops below 40% for 1 week
2. ⚠️ Profit factor drops below 1.0 for 1 week
3. ⚠️ Direction bias becomes >80% one direction
4. ⚠️ Max drawdown approaches Topstep limits ($2,500 for 50k)

## Troubleshooting

### Issue: Still Getting 100% LONG After Retraining

**Check 1: Feature Generation**
```bash
# Verify 'side' feature is in features.yaml
grep -A5 "has_side_feature" configs/features.yaml
```

Should see `has_side_feature: true` in labeling config.

**Check 2: Model Bundle**
```python
import pickle
with open('models/saved/model_bundle_retrained_q4_jan26.pkl', 'rb') as f:
    bundle = pickle.load(f)

print('side' in bundle['primary_feature_columns'])  # Should be True
print(bundle['has_side_feature'])  # Should be True
```

**Check 3: Model Predictor**
```bash
# Check if predictor uses prediction['side'] field
grep -n "prediction.get" live_trading/model_predictor.py
```

Should see code that extracts `prediction.get('side')` and uses it for direction.

### Issue: Databento API Fails

**Solution 1: Use Cached Data**
```bash
python retrain_q4_jan26.py --use-cached-data
```

**Solution 2: Check API Key**
```bash
grep DATABENTO_API_KEY ../.env
```

**Solution 3: Check Databento Credits**
Visit https://databento.com/portal and verify credits available.

### Issue: Training Takes Too Long

Expected times:
- Data fetch: 5 minutes
- Feature generation: 2 minutes  
- Model training: 2 minutes
- **Total: ~10 minutes**

If much longer:
1. Check bar count (should be ~300k-400k bars)
2. Check feature config (reduce vol_regime_lookback if >30)
3. Run on machine with >4GB RAM

### Issue: Low Test AUC

If test AUC < 0.55:
1. Check if training period has mixed regimes (bullish + bearish)
2. Verify features don't have excessive NaN
3. Consider expanding training window to 6 months

## Emergency Rollback

If retrained model fails in production:

```bash
# Stop trading
pkill -f "live_trading"

# Restore backup
cp models/saved/model_bundle_dec2024_baseline_backup.pkl \\
   models/saved/model_bundle.pkl

# Restart with old model
bash monday_startup.sh
```

## Files Created by Retraining

```
ml_intraday_v3/
├── data/
│   └── databento_q4_2024_jan_2026.parquet     # Cached raw data
├── models/saved/
│   ├── model_bundle_retrained_q4_jan26.pkl    # NEW MODEL
│   └── model_bundle.pkl                        # Current production model
├── backtest_results/
│   └── retrained_validation_jan26/             # Validation results
├── retrain_q4_jan26.py                         # THIS SCRIPT
└── RETRAINED_MODEL_DEPLOYMENT.md               # Auto-generated guide
```

## Success Metrics (Expected Results)

Based on walk-forward analysis best practices:

### Minimal Success (Deploy with caution)
- Win rate: 40-45%
- Profit factor: 1.0-1.5
- Direction balance: 35-65% / 65-35%

### Good Success (Deploy with confidence)
- Win rate: 45-50%
- Profit factor: 1.5-2.0
- Direction balance: 40-60% / 60-40%

### Excellent Success (Deploy immediately)
- Win rate: >50%
- Profit factor: >2.0
- Direction balance: 45-55% / 55-45%

## References

### Research Papers
- Walk-forward optimization: `research papers/walk_forward_*.pdf`
- Purged k-fold CV: `research papers/lopez_de_prado_*.pdf`

### Codebase
- Feature generation: `ml_intraday_v3/features/build.py`
- Labeling: `ml_intraday_v3/labels/triple_barrier.py`
- Training: `ml_intraday_v3/training/train.py`
- Backtesting: `ml_intraday_v3/live_trading/replay.py`

### Configuration
- Features: `ml_intraday_v3/configs/features.yaml`
- Labeling: `ml_intraday_v3/configs/labeling.yaml`
- Training: `ml_intraday_v3/configs/training.yaml`

### Baseline Analysis
- Jan 2026 failure: `ml_intraday_v3/backtest_results/databento_validation_20260125_000415/`
- Directional bias investigation: `ml_intraday_v3/SHORT_SIGNAL_INVESTIGATION_SUMMARY.md`
- Config fixes: `ml_intraday_v3/CONFIG_FIX_REPORT.md`

## Questions?

Review these docs:
1. `ML_PIPELINE_V3_BLUEPRINT.md` - Pipeline architecture
2. `TRAINING_GUIDE.md` - Detailed training documentation
3. `MONDAY_MORNING_CHECKLIST.md` - Live trading checklist
4. `.claude/CLAUDE.md` - Project rules and standards

## Next Steps After Retraining

1. ✅ Retrain model (`python retrain_q4_jan26.py`)
2. ✅ Validate on Jan 2026 backtest
3. ✅ Paper trade for 1 week
4. ✅ Deploy to production if successful
5. ✅ Monitor daily for 1 week
6. ✅ Scale up gradually
7. ✅ Schedule monthly retraining

**Remember**: Topstep compliance depends on consistent, profitable trading. Better to be cautious with gradual rollout than aggressive and risk account failure.
