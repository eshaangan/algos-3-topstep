#!/usr/bin/env bash
set -euo pipefail

python3 data/prepare_dataset.py
python3 models/train.py --data-path data/processed/mes_bars.h5
python3 backtesting/backtest.py --data-path data/processed/mes_bars.h5 --model-dir models/saved --save-trades analysis/backtest_trades.csv
