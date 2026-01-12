# Suggested Commands

## Environment Setup

### Activate Virtual Environment
```bash
source .venv/bin/activate
```

### Install Dependencies
```bash
# Main project dependencies
pip install -r requirements.txt

# ML Intraday V3 specific dependencies
pip install -r ml_intraday_v3/requirements-mlv3.txt
```

## ML Intraday V3 Pipeline Commands

### 1. Data Preparation (Phase 1 - COMPLETE)
```bash
# Build data for both 1m and 5m bar sizes
python -m ml_intraday_v3.cli build-data \
  --config ml_intraday_v3/configs/data.yaml \
  --run-id <run_id> \
  --seed 42
```

### 2. Feature Engineering (Phase 2 - COMPLETE)
```bash
# Build features for a completed data run
python -m ml_intraday_v3.cli build-features \
  --run-dir runs/<run_id> \
  --features-config ml_intraday_v3/configs/features.yaml
```

### 3. Labeling (Phase 3)
```bash
# Generate triple-barrier labels
python -m ml_intraday_v3.cli build-labels \
  --run-dir runs/<run_id> \
  --labeling-config ml_intraday_v3/configs/labeling.yaml \
  --execution-spec ml_intraday_v3/configs/execution_spec.yaml
```

### 4. Sample Weights (Phase 4)
```bash
# Generate sample weights (uniqueness + magnitude)
python -m ml_intraday_v3.cli build-weights \
  --run-dir runs/<run_id> \
  --labeling-config ml_intraday_v3/configs/labeling.yaml
```

### 5. Cross-Validation
```bash
# Generate CV splits (purged + embargoed) and CPCV paths
python -m ml_intraday_v3.cli build-cv \
  --run-dir runs/<run_id> \
  --validation-config ml_intraday_v3/configs/validation.yaml
```

### 6. Model Training
```bash
# Train baseline model on CV splits
python -m ml_intraday_v3.cli build-train \
  --run-dir runs/<run_id> \
  --training-config ml_intraday_v3/configs/training.yaml \
  --cv-kind purged_kfold
```

### 7. Backtesting
```bash
# Run offline backtest on CV test splits
python -m ml_intraday_v3.cli build-backtest \
  --run-dir runs/<run_id> \
  --training-dir runs/<run_id> \
  --backtest-config ml_intraday_v3/configs/backtest.yaml \
  --cv-kind purged_kfold
```

### 8. Walk-Forward Evaluation
```bash
# Run walk-forward evaluation
python -m ml_intraday_v3.cli run-walkforward \
  --run-dir runs/<run_id> \
  --walkforward-config ml_intraday_v3/configs/walkforward.yaml
```

### 9. Experiments & Diagnostics
```bash
# Run experiment grid + diagnostics (PBO, DSR)
python -m ml_intraday_v3.cli run-experiments \
  --run-dir runs/<run_id> \
  --grid-config ml_intraday_v3/configs/experiment_grid.yaml
```

### 10. Audit
```bash
# Run end-to-end audit for a run
python -m ml_intraday_v3.cli run-audit \
  --run-dir runs/<run_id> \
  --strict false
```

## Testing Commands

### Run All Tests
```bash
cd ml_intraday_v3
pytest tests/ -v
```

### Run Specific Test Module
```bash
pytest tests/test_features.py -v
```

### Run Leakage Tests
```bash
pytest tests/test_features.py::TestFuturePerturbation -v
```

### Run Label Tests
```bash
pytest tests/test_labels.py -v
```

### Run Backtest Tests
```bash
pytest tests/test_backtest.py -v
```

## Live Trading (V3)

### Paper Trading
```bash
python -m ml_intraday_v3.live_trading.live_runner \
  --model-dir runs/<run_id>/bar_size=1m/training/purged_kfold/fold_0 \
  --symbol MES \
  --dry-run
```

### Live Trading (CAUTION)
```bash
python -m ml_intraday_v3.live_trading.live_runner \
  --model-dir runs/<run_id>/bar_size=1m/training/purged_kfold/fold_0 \
  --symbol MES \
  --live
```

## Git Commands (macOS/Darwin specific)

### Standard Git Operations
```bash
git status
git add <file>
git commit -m "message"
git push
git pull
git log
git diff
```

## macOS/Darwin Specific Commands

### File Operations
```bash
# List files
ls -la

# Find files
find . -name "*.py"

# Search content (use grep or ripgrep)
grep -r "pattern" ml_intraday_v3/

# Change directory
cd ml_intraday_v3/

# Print working directory
pwd
```

### Process Management
```bash
# Check running processes
ps aux | grep python

# Kill process
kill -9 <pid>
```

## Jupyter Notebook

### Primary Testbed
```bash
# Open main pipeline notebook
jupyter notebook ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb
```

## Utility Scripts

### Get Topstep Account IDs
```bash
python get_topstep_accounts.py
```

### Check Account Status
```bash
python check_account_status.py
```

### Test API Trade
```bash
python test_api_trade.py
```

## Viewing Artifacts

### Check Run Manifest
```bash
cat runs/<run_id>/run_manifest.json
```

### View QA Report
```bash
cat runs/<run_id>/bar_size=1m/qa_report.json
```

### View Data Metadata
```bash
cat runs/<run_id>/bar_size=1m/data_metadata.json
```

### Load Parquet Files in Python
```python
import pandas as pd
df = pd.read_parquet("runs/<run_id>/bar_size=1m/bars.parquet")
print(df.head())
```