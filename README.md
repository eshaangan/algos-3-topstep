# Topstep ML v2

End-to-end ML trading system for TopstepX (MES) with strict risk guardrails, no lookahead in features, and trade-outcome labels.

## What this does

- Prepares 1-minute MES data into RTH-only 5-minute bars.
- Builds leakage-safe features and trade-outcome labels.
- Trains separate long/short classifiers with time-aware splits and purge gaps.
- Backtests with Topstep-style daily loss and trailing drawdown rules.
- Provides a live runner that can paper trade or send orders via ProjectX.

## Quick start

1) Create a venv and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Prepare data

```bash
python3 data/prepare_dataset.py
```

3) Train models

```bash
python3 models/train.py --data-path data/processed/mes_bars.h5
```

4) Backtest

```bash
python3 backtesting/backtest.py --data-path data/processed/mes_bars.h5 --model-dir models/saved --save-trades analysis/backtest_trades.csv
```

5) Quick end-to-end check (subset)

```bash
python3 analysis/test_pipeline.py --bars 5000 --save-models
```

## Live runner (paper by default)

The live runner pulls bars from ProjectX and evaluates the ML signal on each new 5-minute bar.

```bash
python3 live/runner.py --model-dir models/saved --symbol MES
```

To send live orders (use with caution):

```bash
python3 live/runner.py --model-dir models/saved --symbol MES --live
```

Required environment variables in `.env`:

- `TOPSTEPX_USERNAME`
- `TOPSTEPX_PROJECTX_API_KEY`
- `TOPSTEPX_ACCOUNT_ID`
- `TOPSTEPX_CONTRACT_ID`

## Notes

- The data prep script defaults to `data/raw/MES_1min_bars.csv`. If that file is missing, it will fall back to another CSV in `data/raw`.
- Training uses time-ordered splits with a purge gap equal to the max hold period to reduce label leakage.
- The live runner uses the existing `core/risk_management.py` guardrails for sizing and daily loss enforcement.

## Project structure

```
core/               Core system, configs, and risk engine
features/           Feature engineering and label creation
models/             Training code and saved models
backtesting/        Backtest engine
analysis/           Pipeline tests and analysis
live/               Live strategy and runner
```

## Safety

This repository is for research. Always paper trade before enabling live orders and ensure TopstepX rules are respected.
