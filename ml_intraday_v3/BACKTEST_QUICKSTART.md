# Direction Change Validation Backtests - Quick Start Guide

## Overview

This guide will help you run three backtests to validate the direction change fix:

1. **Baseline** - No direction change logic (disabled)
2. **High-Confidence** - Direction change threshold = 0.20 (RECOMMENDED)
3. **Aggressive** - Direction change threshold = 0.10 (old behavior that caused Jan 22 issues)

## Prerequisites

- Trained model exists at: `ml_intraday_v3/runs/run_20251224_123456`
- Configuration files exist in: `ml_intraday_v3/configs/`
- All code changes from direction change fix have been applied

## Quick Start (Easiest Method)

### Option 1: Run All Three Backtests Automatically

```bash
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep/ml_intraday_v3

# Run all three backtests with one command
./run_backtests.sh
```

This will automatically:
- Run baseline (no direction change)
- Run high-confidence (threshold=0.20)
- Run aggressive (threshold=0.10)
- Save results to `ml_intraday_v3/runs/direction_change_validation_YYYYMMDD_HHMMSS/`

## Manual Method (More Control)

### Option 2: Run Individual Backtests

```bash
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep

# Run baseline (no direction change)
python ml_intraday_v3/run_direction_change_backtests.py \
  --run-dir ml_intraday_v3/runs/run_20251224_123456 \
  --config-dir ml_intraday_v3/configs \
  --bar-size 1m \
  --start-date 2025-12-01 \
  --end-date 2026-01-20 \
  --skip-high-confidence \
  --skip-aggressive

# Run high-confidence only (RECOMMENDED)
python ml_intraday_v3/run_direction_change_backtests.py \
  --run-dir ml_intraday_v3/runs/run_20251224_123456 \
  --config-dir ml_intraday_v3/configs \
  --bar-size 1m \
  --start-date 2025-12-01 \
  --end-date 2026-01-20 \
  --skip-baseline \
  --skip-aggressive

# Run aggressive only (for comparison)
python ml_intraday_v3/run_direction_change_backtests.py \
  --run-dir ml_intraday_v3/runs/run_20251224_123456 \
  --config-dir ml_intraday_v3/configs \
  --bar-size 1m \
  --start-date 2025-12-01 \
  --end-date 2026-01-20 \
  --skip-baseline \
  --skip-high-confidence
```

## Expected Output

During execution, you'll see:

```
========================================================================
Running backtest: high_confidence_dc_0.20
  Direction change enabled: True
  High-confidence threshold: 0.20
========================================================================
Starting replay: 2025-12-01 to 2026-01-20
...
========================================================================
RESULTS: high_confidence_dc_0.20
  Total trades: 150
  Win rate: 58.2%
  Total P&L: $12,450.00
  Exit reasons:
    target: 65 (43.3%)
    stop: 45 (30.0%)
    direction_change_LONG_to_SHORT: 25 (16.7%)
    direction_change_SHORT_to_LONG: 15 (10.0%)
========================================================================
```

## Output Files

Results are saved to: `ml_intraday_v3/runs/direction_change_validation_YYYYMMDD_HHMMSS/`

Each backtest creates a subdirectory:
```
direction_change_validation_20260124_150000/
├── baseline_no_dc/
│   ├── metrics.csv              # Performance metrics over time
│   ├── trade_log.csv            # All trade entries/exits
│   ├── closed_positions.csv     # Completed trades with P&L
│   └── configs/                 # Config files used for this run
│       ├── live_trading.yaml
│       ├── risk.yaml
│       └── execution_spec.yaml
├── high_confidence_dc_0.20/
│   └── (same structure)
└── aggressive_dc_0.10/
    └── (same structure)
```

## Analyzing Results

### 1. Quick Summary

```bash
# View results summary for all backtests
OUTPUT_DIR="ml_intraday_v3/runs/direction_change_validation_YYYYMMDD_HHMMSS"

for dir in $OUTPUT_DIR/*/; do
    echo "========================================"
    echo "Results: $(basename $dir)"
    echo "========================================"
    tail -1 $dir/metrics.csv
    echo ""
done
```

### 2. Exit Reason Distribution

This is the KEY metric for validating the fix:

```bash
# Analyze exit reasons
OUTPUT_DIR="ml_intraday_v3/runs/direction_change_validation_YYYYMMDD_HHMMSS"

for dir in $OUTPUT_DIR/*/; do
    echo "========================================"
    echo "$(basename $dir)"
    echo "========================================"
    cut -d, -f7 $dir/closed_positions.csv | tail -n+2 | sort | uniq -c | sort -rn
    echo ""
done
```

Expected output:
```
========================================
baseline_no_dc
========================================
     85 target
     65 stop

========================================
high_confidence_dc_0.20
========================================
     75 target
     50 stop
     20 direction_change_LONG_to_SHORT
      5 direction_change_SHORT_to_LONG

========================================
aggressive_dc_0.10
========================================
     30 target
     25 stop
     65 direction_change_LONG_to_SHORT
     30 direction_change_SHORT_to_LONG
```

**Target**: `high_confidence_dc_0.20` should have:
- Natural exits (target + stop) > 70%
- Direction change exits < 20%

### 3. Performance Metrics

```bash
# Compare win rates and P&L
OUTPUT_DIR="ml_intraday_v3/runs/direction_change_validation_YYYYMMDD_HHMMSS"

echo "Backtest | Total Trades | Win Rate | Total P&L | Sharpe"
echo "---------|--------------|----------|-----------|-------"

for dir in $OUTPUT_DIR/*/; do
    name=$(basename $dir)
    trades=$(wc -l < $dir/closed_positions.csv)
    trades=$((trades - 1))  # Subtract header

    # Use Python to calculate metrics
    python3 -c "
import pandas as pd
df = pd.read_csv('$dir/closed_positions.csv')
wins = (df['pnl_usd'] > 0).sum()
win_rate = wins / len(df) * 100 if len(df) > 0 else 0
total_pnl = df['pnl_usd'].sum()
print(f'$name | {len(df)} | {win_rate:.1f}% | \${total_pnl:,.2f}')
"
done
```

### 4. Detailed Analysis (Python)

```python
import pandas as pd
from pathlib import Path

# Load results
output_dir = Path("ml_intraday_v3/runs/direction_change_validation_YYYYMMDD_HHMMSS")

baseline = pd.read_csv(output_dir / "baseline_no_dc" / "closed_positions.csv")
high_conf = pd.read_csv(output_dir / "high_confidence_dc_0.20" / "closed_positions.csv")
aggressive = pd.read_csv(output_dir / "aggressive_dc_0.10" / "closed_positions.csv")

# Compare exit reasons
print("Exit Reason Distribution:")
print("\nBaseline:")
print(baseline['exit_reason'].value_counts(normalize=True) * 100)

print("\nHigh-Confidence (0.20):")
print(high_conf['exit_reason'].value_counts(normalize=True) * 100)

print("\nAggressive (0.10):")
print(aggressive['exit_reason'].value_counts(normalize=True) * 100)

# Compare performance
print("\n\nPerformance Comparison:")
for name, df in [("Baseline", baseline), ("High-Confidence", high_conf), ("Aggressive", aggressive)]:
    print(f"\n{name}:")
    print(f"  Total Trades: {len(df)}")
    print(f"  Win Rate: {(df['pnl_usd'] > 0).mean() * 100:.1f}%")
    print(f"  Total P&L: ${df['pnl_usd'].sum():,.2f}")
    print(f"  Avg Win: ${df[df['pnl_usd'] > 0]['pnl_usd'].mean():,.2f}")
    print(f"  Avg Loss: ${df[df['pnl_usd'] < 0]['pnl_usd'].mean():,.2f}")
```

## Success Criteria

The **High-Confidence (0.20)** backtest should show:

✅ **Exit Reason Distribution**:
- Target exits: > 40%
- Stop exits: 20-40%
- Direction change exits: < 20%

✅ **Performance Metrics**:
- Win rate: ≥ 55% (baseline: 58%)
- Profit factor: ≥ 1.5 (baseline: 1.78)
- Sharpe ratio: ≥ 3.0 (baseline: 11.36)
- Max drawdown: < $1,800 (baseline: -$1,204)

✅ **Comparison**:
- High-confidence should outperform or match aggressive
- Natural exits should be significantly higher than aggressive

## Troubleshooting

### Error: "Run directory not found"

```bash
# Check if run directory exists
ls -la ml_intraday_v3/runs/run_20251224_123456

# If not found, use a different run directory
python ml_intraday_v3/run_direction_change_backtests.py \
  --run-dir ml_intraday_v3/runs/YOUR_RUN_DIR \
  ...
```

### Error: "Model bundle not found"

The script looks for the latest walkforward model. If not found:

```bash
# Check for walkforward models
ls -la ml_intraday_v3/runs/run_20251224_123456/walkforward/

# If walkforward doesn't exist, you may need to train models first
```

### Error: "No bars available after slicing"

Adjust date range:

```bash
# Check available data range
python3 -c "
import pandas as pd
bars = pd.read_parquet('ml_intraday_v3/runs/run_20251224_123456/bar_size=1m/bars.parquet')
print('Data range:', bars.index.min(), 'to', bars.index.max())
"

# Use available date range in backtest
```

### Backtests Running Too Slow

Use a shorter date range for quick testing:

```bash
python ml_intraday_v3/run_direction_change_backtests.py \
  --start-date 2025-12-15 \
  --end-date 2025-12-31 \
  ...
```

## Next Steps

After running backtests:

1. **Review Results**: Check exit reason distribution and performance metrics
2. **Compare to Baseline**: Ensure high-confidence meets or exceeds Dec 2024 baseline
3. **Deploy to GCP**: If validation passes, commit changes and deploy
4. **Monitor Live**: Watch first 5-10 trading days closely

## Additional Commands

### Check Script Version

```bash
# View script help
python ml_intraday_v3/run_direction_change_backtests.py --help
```

### Run with Custom Output Directory

```bash
python ml_intraday_v3/run_direction_change_backtests.py \
  --output-dir ml_intraday_v3/runs/my_custom_validation_$(date +%Y%m%d) \
  ...
```

### Run Only High-Confidence (Fastest)

```bash
./run_backtests.sh --skip-baseline --skip-aggressive
```

Wait, that doesn't work with the shell script. Use Python directly:

```bash
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep

python ml_intraday_v3/run_direction_change_backtests.py \
  --skip-baseline \
  --skip-aggressive
```

## Documentation References

- **Implementation Details**: `ml_intraday_v3/DIRECTION_CHANGE_FIX_IMPLEMENTATION.md`
- **Verification Script**: `ml_intraday_v3/verify_direction_change_fix.sh`
- **This Guide**: `ml_intraday_v3/BACKTEST_QUICKSTART.md`
