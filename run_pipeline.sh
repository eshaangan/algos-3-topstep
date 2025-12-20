#!/usr/bin/env bash
set -euo pipefail

PY_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
fi

PYTHONPATH=. "$PY_BIN" data/prepare_dataset.py
PYTHONPATH=. "$PY_BIN" models/train.py --data-path data/processed/mes_bars.h5
# Default to the OOS test split saved by the trainer (can override with --split full).
PYTHONPATH=. "$PY_BIN" backtesting/backtest.py --data-path data/processed/mes_bars.h5 --model-dir models/saved --split test --save-trades analysis/backtest_trades.csv
