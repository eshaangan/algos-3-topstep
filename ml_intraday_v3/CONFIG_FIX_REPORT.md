# Config Fix Implementation Report
**Date**: 2026-01-24  
**Status**: PARTIAL - Configs updated, full pipeline regeneration required

## Summary
Applied quick fixes for model degradation identified in diagnostic analysis:
1. **vol_regime warmup issue** - Fixed in features.yaml
2. **Stop-loss too tight** - Fixed in labeling.yaml

However, backtest results show NO improvement because:
- Labels were already generated with OLD config values
- Features are computed on-the-fly and will use NEW config
- Need full pipeline regeneration to apply label changes

## Changes Applied

### 1. features.yaml (✓ APPLIED)
**File**: `ml_intraday_v3/configs/features.yaml`  
**Backup**: `features.yaml.backup`

```yaml
# BEFORE
vol_regime_lookback: 50

# AFTER
vol_regime_lookback: 30
```

**Impact**:
- Reduces warmup requirement from 70 bars to 50 bars
- Should reduce vol_regime NaN count from 70 → ~49 (30% improvement)
- Features are computed on-the-fly, so this takes effect immediately

### 2. labeling.yaml (✓ APPLIED, ⚠️ REGENERATION REQUIRED)
**File**: `ml_intraday_v3/configs/labeling.yaml`  
**Backup**: `labeling.yaml.backup`

```yaml
# BEFORE
pt_multipliers:
  - 2.9    # Profit target
sl_multipliers:
  - 3.4    # Stop loss

# AFTER
pt_multipliers:
  - 4.35    # Widened 1.5x (2.9 * 1.5)
sl_multipliers:
  - 5.1     # Widened 1.5x (3.4 * 1.5)
```

**Impact**:
- Wider stops should reduce stop-hit rate from 86% → 60-70%
- Wider targets maintain reward/risk ratio of ~0.85
- **CRITICAL**: Labels must be regenerated for this to take effect

## Backtest Results (Using OLD Labels)

```
Metric                               baseline
----------------------------------------
Total Trades                              168
Win Rate                                13.7%  ← Target: >25% (FAIL)
Profit Factor                            0.19  ← Target: >0.8 (FAIL)
Stop Hit Rate                           86.3%  ← Target: <70% (FAIL)
Target Hit Rate                         13.7%
LONG %                                 100.0%
SHORT %                                  0.0%  ← Target: >10% (FAIL)
```

**Why NO improvement?**:
The backtest script (`backtest_databento_recent.py`) hardcodes old label schema:
```python
label_schema = {
    "stop_multiple": 1.0,    # Should be 1.5
    "target_multiple": 2.0,  # Should be 2.5
    ...
}
```

And uses pre-generated labels from the run directory, which were created with old config.

## Root Cause Analysis

### Issue 1: Hardcoded Label Schema in Backtest Script
**File**: `ml_intraday_v3/backtest_databento_recent.py:103-108`

The script creates a label_schema.json with hardcoded values instead of reading from labeling.yaml:
```python
label_schema = {
    "stop_multiple": 1.0,     # ← HARDCODED OLD VALUE
    "target_multiple": 2.0,   # ← HARDCODED OLD VALUE
    "atr_period": 14,
    "atr_bar_size": bar_size,
}
```

### Issue 2: Pre-Generated Labels
The replay system uses labels from `runs/{run_id}/bar_size={bar_size}/` which were:
1. Generated when data was first fetched
2. Computed using OLD config values
3. Stored as immutable parquet files

### Issue 3: Feature vs Label Config Separation
- **Features**: Computed on-the-fly in live_trading/replay → New config DOES take effect
- **Labels**: Pre-computed and stored → New config DOES NOT take effect

## Next Steps

### Required: Full Pipeline Regeneration

To apply the label changes, need to:

1. **Delete old run directories** (or create new ones):
   ```bash
   rm -rf ml_intraday_v3/runs/databento_backtest_*
   rm -rf ml_intraday_v3/runs/recent_data_*
   ```

2. **Regenerate labels from scratch**:
   - Run full labeling pipeline on Jan 2026 data
   - Use updated labeling.yaml config
   - Generate new events.parquet with wider barriers

3. **Update backtest script**:
   - Remove hardcoded label_schema
   - Read from labeling.yaml instead:
     ```python
     from ml_intraday_v3.labels.schema import LabelSchema
     schema = LabelSchema.from_config(config_path)
     ```

4. **Retrain model** (optional but recommended):
   - If we change training labels, model should be retrained
   - Current model was trained with 2.9/3.4 barriers
   - New model with 4.35/5.1 barriers may have different predictions

### Alternative: Quick Test with Modified Execution

To test the barrier changes WITHOUT full regeneration:

1. Modify `backtest_databento_recent.py` lines 103-108:
   ```python
   label_schema = {
       "stop_multiple": 1.5,    # NEW VALUE
       "target_multiple": 2.5,  # NEW VALUE
       "atr_period": 14,
       "atr_bar_size": bar_size,
   }
   ```

2. Modify `replay_session()` to ignore label barriers and use execution config instead

This is a hack but would validate whether the barrier changes help.

## Files Modified

1. ✓ `ml_intraday_v3/configs/features.yaml` (vol_regime_lookback: 50 → 30)
2. ✓ `ml_intraday_v3/configs/labeling.yaml` (pt: 2.9 → 4.35, sl: 3.4 → 5.1)
3. ✓ Created `ml_intraday_v3/analyze_jan22_with_threshold.py` (analysis script)
4. ✓ Created backups: `features.yaml.backup`, `labeling.yaml.backup`

## Files That Need Updates

1. ⚠️ `ml_intraday_v3/backtest_databento_recent.py` (remove hardcoded schema)
2. ⚠️ Label generation pipeline (regenerate with new config)
3. ⚠️ Model retraining (optional, for consistency)

## Expected Outcomes (After Full Regeneration)

Based on diagnostic analysis, fixing these issues should yield:

| Metric | Baseline (Before) | Expected (After) | Status |
|--------|-------------------|------------------|---------|
| Win Rate | 13.7% | 25-35% | Pending regen |
| Profit Factor | 0.19 | 0.8-1.2 | Pending regen |
| Stop Hit Rate | 86% | 60-70% | Pending regen |
| SHORT Signals | 0% | 10-30% | Pending regen |
| vol_regime NaN | 70/100 | ~49/100 | ✓ Config ready |

## Conclusion

**Config changes applied**: ✓  
**Pipeline regeneration required**: ⚠️  
**Current backtest results**: No improvement (using old labels)  
**Next action**: Either regenerate labels OR quick-test with modified backtest script
