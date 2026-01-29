# Implementation Workflow - Model Degradation Fixes

## Quick Reference: What to Run Next

### Step 1: Fix vol_regime (5 minutes)
```bash
# Edit config file
vim ml_intraday_v3/configs/features.yaml

# Change line:
volatility:
  vol_regime_lookback: 30  # was 50

# Regenerate features and backtest
python ml_intraday_v3/fetch_and_backtest_recent.py
```

### Step 2: Test Wider Stops (10 minutes)
```bash
# Edit execution config
vim ml_intraday_v3/configs/execution_spec.yaml

# Change lines:
risk:
  stop_multiple: 1.5    # was 1.0
  target_multiple: 2.5  # was 2.0

# Re-run backtest with new stops
python ml_intraday_v3/backtest_databento_recent.py
```

### Step 3: Compare Results (5 minutes)
```bash
# View new backtest results
python ml_intraday_v3/analyze_jan22_with_threshold.py

# Expected improvements:
# - SHORT signals: 0% → 10-30%
# - Win rate: 13.7% → 25-35%
# - Profit factor: 0.19 → 0.8-1.2
```

---

## Detailed Implementation Steps

### Fix #1: vol_regime Warmup Issue

**Problem**: Feature requires 70 bars but only 69 available

**Solution**: Reduce rolling window from 50 to 30 bars

**Files Changed**:
- `ml_intraday_v3/configs/features.yaml`

**Changes**:
```yaml
volatility:
  enable_regime_features: true
  atr_period: 14
  vol_regime_lookback: 30  # CHANGE: was 50
```

**Validation**:
```bash
# After regenerating features, check NaN count
python -c "
import pandas as pd
features = pd.read_parquet('ml_intraday_v3/runs/databento_backtest_*/bar_size=5m/features.parquet')
print(f'vol_regime NaN count: {features[\"vol_regime\"].isna().sum()} / {len(features)}')
print(f'Expected: ~50 NaN (warmup), Actual: {features[\"vol_regime\"].isna().sum()}')
"
```

**Expected Outcome**:
- vol_regime NaN count: 50 (instead of 69)
- Model can now use vol_regime for ~19 additional bars
- May restore SHORT signal generation

---

### Fix #2: Optimize Stop-Loss Width

**Problem**: 86% of trades hit stop-loss, 0% win rate

**Solution**: Widen stop from 1x ATR to 1.5x ATR

**Files Changed**:
- `ml_intraday_v3/configs/execution_spec.yaml`

**Changes**:
```yaml
risk:
  stop_multiple: 1.5    # CHANGE: was 1.0
  target_multiple: 2.5  # CHANGE: was 2.0 (maintain 1:1.67 ratio)
```

**Alternative to Test**:
```yaml
# Option A: Tighter ratio (1:2)
risk:
  stop_multiple: 1.5
  target_multiple: 3.0

# Option B: Maintain original ratio (1:2)
risk:
  stop_multiple: 1.2
  target_multiple: 2.4
```

**Validation**:
```bash
# After backtest, check exit distribution
python -c "
import pandas as pd
trades = pd.read_csv('ml_intraday_v3/backtest_results/*/baseline/trades_*.csv')
print(trades['exit_reason'].value_counts(normalize=True))
# Expected: stop ~60%, target ~40% (was 86%/14%)
"
```

**Expected Outcome**:
- Stop-hit rate: 86% → 60-70%
- Win rate: 13.7% → 25-35%
- Profit factor: 0.19 → 0.8-1.2

---

### Fix #3: Training Data Distribution Analysis (Task #4)

**Problem**: Unknown if Dec 2024 training data has LONG bias

**Solution**: Load training data and compare distributions

**Script to Create**:
```python
# ml_intraday_v3/analyze_training_distribution.py

import pandas as pd
import numpy as np
from pathlib import Path

# Load training data
train_path = Path("runs/run_20251224_*/bar_size=1m/")
train_features = pd.read_parquet(train_path / "features.parquet")
train_labels = pd.read_parquet(train_path / "labels.parquet")

# Load test data
test_path = Path("ml_intraday_v3/runs/databento_backtest_20260124_182118/bar_size=5m/")
test_features = pd.read_parquet(test_path / "features.parquet")

print("="*80)
print("TRAINING DATA DISTRIBUTION")
print("="*80)

# Label distribution
print("\nLabel Distribution (Training):")
print(train_labels['label'].value_counts(normalize=True))

# Feature statistics
print("\nFeature Statistics Comparison:")
for col in train_features.columns:
    if col in test_features.columns:
        train_mean = train_features[col].mean()
        test_mean = test_features[col].mean()
        diff_pct = ((test_mean - train_mean) / train_mean * 100) if train_mean != 0 else 0
        print(f"{col:30s}: Train={train_mean:>8.4f}, Test={test_mean:>8.4f}, Drift={diff_pct:>6.1f}%")

# KS test for distribution shift
from scipy.stats import ks_2samp
print("\nKolmogorov-Smirnov Test (p < 0.05 = significant drift):")
for col in train_features.columns:
    if col in test_features.columns and train_features[col].notna().sum() > 0:
        stat, p_value = ks_2samp(
            train_features[col].dropna(),
            test_features[col].dropna()
        )
        drift_status = "⚠️ DRIFT" if p_value < 0.05 else "✓ OK"
        print(f"{col:30s}: p={p_value:.4f} {drift_status}")
```

**Run**:
```bash
python ml_intraday_v3/analyze_training_distribution.py
```

**Expected Findings**:
- Training LONG/SHORT label ratio (hypothesis: 70/30 or worse)
- Features with significant drift (vol_20, ema_spread, etc.)
- Confirmation of distribution shift hypothesis

---

## Testing & Validation Checklist

### Before Deploying Fixes
- [ ] Backup current model and configs
- [ ] Create new branch: `fix/model-degradation`
- [ ] Document baseline metrics (13.7% WR, 0.19 PF)

### After vol_regime Fix
- [ ] Verify vol_regime NaN count reduced to ~50
- [ ] Re-run Jan 2026 backtest
- [ ] Check if SHORT signals appear (target: >10%)
- [ ] Compare win rate to baseline

### After Stop-Loss Fix
- [ ] Verify stop-hit rate reduced to <70%
- [ ] Check win rate improvement
- [ ] Ensure profit factor >0.5 minimum
- [ ] Measure drawdown impact

### Combined Fixes
- [ ] Run full backtest with both fixes
- [ ] Target metrics:
  - Win rate: >35%
  - Profit factor: >1.0
  - LONG/SHORT ratio: 60/40 to 40/60
  - Max drawdown: <$1,500

### Validation on Fresh Data
- [ ] If Feb 2026 data available, test there
- [ ] Maintain >40% win rate out-of-sample
- [ ] No new failure modes introduced

---

## Rollback Plan

If fixes make things worse:

```bash
# Restore original configs
git checkout main -- ml_intraday_v3/configs/features.yaml
git checkout main -- ml_intraday_v3/configs/execution_spec.yaml

# Revert to baseline model
cp ml_intraday_v3/models/saved/model_bundle_baseline.pkl \
   ml_intraday_v3/models/saved/model_bundle.pkl

# Re-run baseline backtest to confirm
python ml_intraday_v3/backtest_databento_recent.py --config baseline
```

---

## Success Criteria

### Minimum Viable (Ready for Paper Trading)
- ✅ Win rate: >35%
- ✅ Profit factor: >1.0
- ✅ LONG/SHORT balance: 60/40 to 40/60
- ✅ Max drawdown: <$2,000

### Target (Ready for Live)
- ✅ Win rate: >45%
- ✅ Profit factor: >1.3
- ✅ LONG/SHORT balance: 55/45 to 45/55
- ✅ Max drawdown: <$1,500
- ✅ Consecutive losses: <5

### Ideal (Combine-Ready)
- ✅ Win rate: >50%
- ✅ Profit factor: >1.5
- ✅ LONG/SHORT balance: 52/48 to 48/52
- ✅ Max drawdown: <$1,000
- ✅ Sharpe ratio: >1.5

---

## Timeline Estimate

| Task | Duration | Blocker |
|------|----------|---------|
| Fix vol_regime config | 5 min | None |
| Regenerate features | 2 min | None |
| Backtest with vol_regime fix | 3 min | None |
| Fix stop-loss config | 5 min | None |
| Backtest with stop fix | 3 min | None |
| Backtest with both fixes | 3 min | None |
| Analyze results | 15 min | None |
| **TOTAL (Quick Wins)** | **~35 min** | **None** |
| | | |
| Write distribution analysis script | 30 min | Need baseline run path |
| Run distribution analysis | 5 min | None |
| Interpret drift results | 15 min | None |
| **TOTAL (Task #4)** | **~50 min** | **Baseline data** |

**Grand Total**: ~90 minutes to complete all immediate fixes and analysis

---

## Commands Ready to Copy-Paste

```bash
# Navigate to project
cd "/Users/eshaanganguly/Documents/projects/algos 3 topstep"

# Fix 1: vol_regime
sed -i '' 's/vol_regime_lookback: 50/vol_regime_lookback: 30/' ml_intraday_v3/configs/features.yaml

# Fix 2: Stop-loss
sed -i '' 's/stop_multiple: 1\.0/stop_multiple: 1.5/' ml_intraday_v3/configs/execution_spec.yaml
sed -i '' 's/target_multiple: 2\.0/target_multiple: 2.5/' ml_intraday_v3/configs/execution_spec.yaml

# Regenerate and backtest
python ml_intraday_v3/fetch_and_backtest_recent.py

# Analyze results
python ml_intraday_v3/analyze_jan22_with_threshold.py
```

---

**Status**: Ready to implement
**Risk**: Low (config changes only, easily reversible)
**Expected Impact**: Significant improvement in key metrics
**Time Required**: ~90 minutes total
