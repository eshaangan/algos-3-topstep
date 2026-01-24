#!/bin/bash
#
# Quick Start: Multi-Market Trading Setup
# Downloads free historical data and validates diversification benefit
#

set -e  # Exit on error

echo "=================================================="
echo "MULTI-MARKET TRADING SETUP"
echo "Markets: MES (US) + NKD (Asia)"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements-mlv3.txt" ]; then
    echo "❌ Error: Must run from ml_intraday_v3/ directory"
    exit 1
fi

# Step 1: Install dependencies
echo "Step 1: Installing dependencies..."
pip install yfinance >/dev/null 2>&1 || {
    echo "⚠️  Warning: yfinance install failed. Trying again..."
    pip install yfinance
}
echo "✓ Dependencies installed"
echo ""

# Step 2: Create output directories
echo "Step 2: Creating output directories..."
mkdir -p data/raw_futures
mkdir -p analysis
echo "✓ Directories created"
echo ""

# Step 3: Download historical data
echo "Step 3: Downloading FREE historical data from Yahoo Finance..."
echo "   This will take 30-60 seconds..."
python data/download_free_futures.py \
    --start 2020-01-01 \
    --end 2025-12-31 \
    --output-dir data/raw_futures
echo ""

# Step 4: Quick validation
echo "Step 4: Validating data..."
python -c "
import pandas as pd
from pathlib import Path

# Check MES
mes_file = Path('data/raw_futures').glob('mes_daily_*.parquet')
mes_files = list(mes_file)
if mes_files:
    mes = pd.read_parquet(mes_files[0])
    print(f'✓ MES: {len(mes):,} days ({mes.index.min().date()} to {mes.index.max().date()})')
else:
    print('❌ MES data not found')

# Check NKD
nkd_file = Path('data/raw_futures').glob('nkd_daily_*.parquet')
nkd_files = list(nkd_file)
if nkd_files:
    nkd = pd.read_parquet(nkd_files[0])
    print(f'✓ NKD: {len(nkd):,} days ({nkd.index.min().date()} to {nkd.index.max().date()})')
else:
    print('❌ NKD data not found')
" || {
    echo "⚠️  Warning: Validation failed. Check data files manually."
}
echo ""

# Step 5: Calculate correlation
echo "Step 5: Calculating MES-NKD correlation..."
python -c "
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
mes_file = list(Path('data/raw_futures').glob('mes_daily_*.parquet'))[0]
nkd_file = list(Path('data/raw_futures').glob('nkd_daily_*.parquet'))[0]

mes = pd.read_parquet(mes_file)
nkd = pd.read_parquet(nkd_file)

# Calculate returns
mes['returns'] = mes['Close'].pct_change()
nkd['returns'] = nkd['Close'].pct_change()

# Merge
merged = pd.merge(
    mes[['returns']].rename(columns={'returns': 'mes_returns'}),
    nkd[['returns']].rename(columns={'returns': 'nkd_returns'}),
    left_index=True,
    right_index=True,
    how='inner'
)

# Calculate correlation
corr = merged.corr().loc['mes_returns', 'nkd_returns']
print(f'')
print(f'MES-NKD Correlation: {corr:.3f}')
print(f'')
if corr < 0.8:
    print('✓ GOOD: Correlation < 0.8 (genuine diversification)')
    print('  Expected: Multi-market should improve risk-adjusted returns')
else:
    print('⚠️  WARNING: Correlation > 0.8 (less diversification than expected)')
    print('  May still be worth it for extended trading hours')
print(f'')
print(f'For comparison:')
print(f'  ES-NQ correlation: ~0.90 (highly correlated)')
print(f'  ES-MES correlation: 1.00 (identical)')
print(f'  MES-NKD correlation: {corr:.2f}')
" || {
    echo "⚠️  Could not calculate correlation. Check data manually."
}
echo ""

echo "=================================================="
echo "✓ SETUP COMPLETE!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Review QUICK_START_MULTI_MARKET.md for detailed guide"
echo "2. Create configs/training_nkd.yaml (copy from training_mes.yaml)"
echo "3. Train NKD model: python -m ml_intraday_v3.cli build-train --config configs/training_nkd.yaml"
echo "4. Backtest multi-market strategy"
echo ""
echo "Files created:"
echo "  - data/raw_futures/mes_daily_*.parquet"
echo "  - data/raw_futures/nkd_daily_*.parquet"
echo ""
echo "Trading hours:"
echo "  NKD (Asia):  18:00-03:00 CT"
echo "  MES (US):    08:30-15:00 CT"
echo "  Total:       ~16 hours/day"
echo ""
