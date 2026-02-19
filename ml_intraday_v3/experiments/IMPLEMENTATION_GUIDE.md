# Model Generalization Fix - Implementation Guide

## Overview

This guide walks through implementing and launching the comprehensive experiment batches designed to fix model generalization issues.

**Problem**: All 100 models from previous exhaustive search produced identical negative PnL (-$1,212.56) on Jan-Feb 2026 out-of-sample data, proving labels are deterministic and model predictions are irrelevant.

**Solution**: Test 600 experiments across 3 batches exploring trend scanning labels, fractional differentiation, CPCV, sample uniqueness weighting, and meta-labeling.

**Estimated Cost**: ~$1.20 for 600 GCP experiments

## Prerequisites

### 1. Local Environment
```bash
# Python dependencies (already in requirements.txt)
pip install numpy pandas scikit-learn lightgbm joblib pyyaml statsmodels scipy hmmlearn

# GCP SDK
# Install from: https://cloud.google.com/sdk/install
gcloud init
gcloud auth login
```

### 2. GCP Setup
```bash
# Set project
export PROJECT_ID="your-gcp-project-id"
gcloud config set project $PROJECT_ID

# Create GCS bucket (if not exists)
gsutil mb gs://trading-algo-3

# Create directories in bucket
gsutil mkdir gs://trading-algo-3/code-snapshots
gsutil mkdir gs://trading-algo-3/experiment-data
gsutil mkdir gs://trading-algo-3/experiments
gsutil mkdir gs://trading-algo-3/experiment-results
gsutil mkdir gs://trading-algo-3/experiment-logs

# Enable Compute Engine API
gcloud services enable compute.googleapis.com
```

### 3. Data Preparation
```bash
# Upload training data to GCS
gsutil cp data/processed/*.parquet gs://trading-algo-3/experiment-data/

# Verify upload
gsutil ls gs://trading-algo-3/experiment-data/
```

## Architecture

### Phase 1 - Core Implementations (Already Complete)

All Phase 1 files have been implemented:

1. **Trend Scanning Labels** (`ml_intraday_v3/labels/trend_scanning.py`)
   - Adaptive horizon selection based on t-values
   - Labels: {-1: downtrend, 0: no trend, 1: uptrend}
   - Sample weights from t-values

2. **Fractional Differentiation** (`ml_intraday_v3/features/fractional_diff.py`)
   - Makes features stationary while preserving memory
   - ADF test for stationarity validation
   - Configurable differentiation order (d)

3. **Combinatorial Purged CV** (`ml_intraday_v3/cross_validation/cpcv.py`)
   - Prevents temporal leakage
   - Purging + embargo for label overlap
   - Combinatorial test splits

4. **Sample Uniqueness Weighting** (`ml_intraday_v3/sampling/uniqueness.py`)
   - Weights by temporal overlap
   - Optional time decay
   - Reduces overfitting on concurrent labels

5. **Meta-Labeling** (`ml_intraday_v3/models/meta_labeling.py`)
   - Two-stage architecture
   - Primary model: direction
   - Secondary model: filter false positives

### Experiment Batches

#### Batch 1: Labeling Methods (200 configs)
**Focus**: Test labeling methods and sample weighting

**Axes**:
- Labeling: triple_barrier, trend_scanning, trend_scanning_adaptive
- Sample weights: uniform, uniqueness, uniqueness_decay
- Features: baseline, baseline_fracdiff
- Triple barrier params: PT (1.5-2.5), SL (1.0-2.0), Time (20-60 bars)
- Trend scanning params: Lookahead (15-30 bars), min_t_value (1.5-2.5)

**Generated**: `ml_intraday_v3/experiments/batch1_configs/` (200 configs)

#### Batch 2: CV Methods & Features (200 configs)
**Focus**: Test cross-validation and enhanced features

**Axes**:
- CV method: kfold, cpcv, cpcv_embargo
- Features: baseline, baseline_hmm, baseline_multiresolution, baseline_fracdiff_hmm
- Calibration: none, isotonic, sigmoid
- Model params: num_leaves (31-127), max_depth (-1 to 20), min_data_in_leaf (20-100)

**Generated**: `ml_intraday_v3/experiments/batch2_configs/` (200 configs)

#### Batch 3: Meta-Labeling (200 configs)
**Focus**: Test two-stage meta-labeling architecture

**Axes**:
- Architecture: single_model, meta_labeling
- Primary model: recall_threshold (0.60-0.80), t_value_threshold (1.0-2.0)
- Secondary model: num_leaves (15-63), max_depth (5-15)
- Final threshold: 0.50, 0.55, 0.60

**Generated**: `ml_intraday_v3/experiments/batch3_configs/` (200 configs)

## Local Testing

### Step 1: Verify Configs Generated
```bash
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep

# Check config counts (should be 200 each + 1 manifest)
ls ml_intraday_v3/experiments/batch1_configs/*.json | wc -l  # 201
ls ml_intraday_v3/experiments/batch2_configs/*.json | wc -l  # 201
ls ml_intraday_v3/experiments/batch3_configs/*.json | wc -l  # 201
```

### Step 2: Test Single Experiment Locally
```bash
# Make test script executable
chmod +x ml_intraday_v3/experiments/test_single_experiment.py

# Test a Batch 1 config (trend scanning)
python ml_intraday_v3/experiments/test_single_experiment.py \
    --config ml_intraday_v3/experiments/batch1_configs/batch1_exp_00001.json \
    --data-dir data/processed \
    --config-dir ml_intraday_v3/configs \
    --output /tmp/test_result.json

# View result
cat /tmp/test_result.json | python -m json.tool
```

Expected output:
```json
{
  "exp_id": "batch1_exp_00001",
  "status": "success",
  "metrics": {
    "train_auc": 0.65,
    "val_auc": 0.58,
    "test_auc": 0.56,
    "cv_n_splits": 5,
    "n_train_events": 1234,
    "n_test_events": 456
  },
  "config": { ... }
}
```

### Step 3: Test Different Labeling Methods
```bash
# Test triple barrier baseline
grep -l '"labeling_method": "triple_barrier"' ml_intraday_v3/experiments/batch1_configs/*.json | head -1 | \
xargs python ml_intraday_v3/experiments/test_single_experiment.py --config

# Test trend scanning
grep -l '"labeling_method": "trend_scanning"' ml_intraday_v3/experiments/batch1_configs/*.json | head -1 | \
xargs python ml_intraday_v3/experiments/test_single_experiment.py --config

# Test with fractional differentiation
grep -l '"feature_set_name": "baseline_fracdiff"' ml_intraday_v3/experiments/batch1_configs/*.json | head -1 | \
xargs python ml_intraday_v3/experiments/test_single_experiment.py --config
```

## GCP Launch

### Step 1: Update Configuration
Edit `ml_intraday_v3/experiments/launch_gcp_experiments.sh`:
```bash
# Line 8-10: Update with your GCP details
PROJECT_ID="your-gcp-project-id"
BUCKET="gs://your-bucket-name"
ZONE="us-central1-a"  # Or your preferred zone
```

### Step 2: Make Scripts Executable
```bash
chmod +x ml_intraday_v3/experiments/launch_gcp_experiments.sh
chmod +x ml_intraday_v3/experiments/gcp_startup_v2.sh
```

### Step 3: Launch Experiments

**Option A: Launch All 3 Batches in Parallel** (~$1.20, 30-45 min)
```bash
./ml_intraday_v3/experiments/launch_gcp_experiments.sh all
```

**Option B: Launch Single Batch** (~$0.40, 10-15 min)
```bash
# Just Batch 1 (labeling methods)
./ml_intraday_v3/experiments/launch_gcp_experiments.sh batch1

# Just Batch 2 (CV & features)
./ml_intraday_v3/experiments/launch_gcp_experiments.sh batch2

# Just Batch 3 (meta-labeling)
./ml_intraday_v3/experiments/launch_gcp_experiments.sh batch3
```

The script will:
1. Upload configs to GCS
2. Create code snapshot and upload
3. Verify data files exist
4. Launch preemptible VMs
5. VMs auto-shutdown when complete

### Step 4: Monitor Progress

**Check completion rate:**
```bash
# Count completed experiments
gsutil ls gs://trading-algo-3/experiment-results/batch1/*.json | wc -l
gsutil ls gs://trading-algo-3/experiment-results/batch2/*.json | wc -l
gsutil ls gs://trading-algo-3/experiment-results/batch3/*.json | wc -l
```

**View live logs:**
```bash
# SSH into VM and tail logs
gcloud compute ssh ml-batch1-vm --zone=us-central1-a \
    --command='tail -f /var/log/syslog'
```

**Check VM status:**
```bash
gcloud compute instances list
```

### Step 5: Handle Issues

**If VM gets preempted:**
```bash
# Check summary
gsutil cat gs://trading-algo-3/experiment-results/batch1_summary.json

# Relaunch with remaining configs
# (script will skip already-completed experiments)
./ml_intraday_v3/experiments/launch_gcp_experiments.sh batch1
```

**If experiment fails:**
```bash
# View error logs
gsutil ls gs://trading-algo-3/experiment-logs/batch1/

# Download specific log
gsutil cp gs://trading-algo-3/experiment-logs/batch1/exp_batch1_exp_00042.log /tmp/

# View log
cat /tmp/exp_batch1_exp_00042.log
```

**Stop experiments early:**
```bash
# Stop VMs (they'll resume on restart)
gcloud compute instances stop ml-batch1-vm ml-batch2-vm ml-batch3-vm \
    --zone=us-central1-a

# Or delete VMs entirely
gcloud compute instances delete ml-batch1-vm ml-batch2-vm ml-batch3-vm \
    --zone=us-central1-a --quiet
```

## Results Analysis

### Step 1: Download Results
```bash
# Download all results
mkdir -p ml_intraday_v3/experiments/results
gsutil -m cp -r gs://trading-algo-3/experiment-results/* \
    ml_intraday_v3/experiments/results/

# Download logs
mkdir -p ml_intraday_v3/experiments/logs
gsutil -m cp -r gs://trading-algo-3/experiment-logs/* \
    ml_intraday_v3/experiments/logs/
```

### Step 2: Analyze Results
```bash
# Analyze each batch
python ml_intraday_v3/experiments/analyze_batch_results.py \
    --batch batch1 \
    --results-dir ml_intraday_v3/experiments/results/batch1 \
    --output ml_intraday_v3/experiments/results/batch1_analysis.json

python ml_intraday_v3/experiments/analyze_batch_results.py \
    --batch batch2 \
    --results-dir ml_intraday_v3/experiments/results/batch2 \
    --output ml_intraday_v3/experiments/results/batch2_analysis.json

python ml_intraday_v3/experiments/analyze_batch_results.py \
    --batch batch3 \
    --results-dir ml_intraday_v3/experiments/results/batch3 \
    --output ml_intraday_v3/experiments/results/batch3_analysis.json
```

### Step 3: Identify Top Models
```bash
# Find top 10 models by Sharpe ratio
python ml_intraday_v3/experiments/rank_models.py \
    --results-dir ml_intraday_v3/experiments/results \
    --metric sharpe_ratio \
    --top-n 10 \
    --output ml_intraday_v3/experiments/results/top_10_models.json
```

### Step 4: Validate on Forward Test Data
```bash
# Test top 10 models on Feb 11 - Mar 31, 2026 (truly unseen)
python ml_intraday_v3/experiments/validate_top10_forward_test.py \
    --configs ml_intraday_v3/experiments/results/top_10_models.json \
    --data data/processed/feb_mar_2026_forward_test.parquet \
    --output ml_intraday_v3/experiments/results/forward_test_results.json
```

## Success Criteria

### Minimum Success
ANY model that produces:
- ✅ Positive PnL on Jan-Feb 2026 test set (currently all -$1,212.56)
- ✅ Win rate > 45% (currently 41.3%)
- ✅ DIFFERENT signals than other models (proves labels aren't deterministic)

### Target Success
Top model achieves:
- ✅ Sharpe Ratio > 1.5
- ✅ Win Rate > 52%
- ✅ Max Drawdown < $800 (Topstep constraint)
- ✅ Passes forward test on Feb-Mar 2026 with positive PnL

### Stretch Goal
Meta-labeling system:
- ✅ Primary model: 70% recall, 48% precision
- ✅ Secondary model: filters to 65% precision
- ✅ Combined Sharpe > 2.0
- ✅ Ready for Topstep combine simulation

## Cost Estimate

| Batch | Configs | VM Type | Est. Time | Est. Cost |
|-------|---------|---------|-----------|-----------|
| Batch 1 | 200 | n1-highmem-4 | ~15 min | ~$0.40 |
| Batch 2 | 200 | n1-highmem-4 | ~15 min | ~$0.40 |
| Batch 3 | 200 | n1-highmem-4 | ~15 min | ~$0.40 |
| **Total** | **600** | | **~45 min** | **~$1.20** |

*Preemptible pricing in us-central1-a: ~$0.096/hour for n1-highmem-4*

## Troubleshooting

### Issue: "No data files found"
```bash
# Upload data to GCS
gsutil cp data/processed/*.parquet gs://trading-algo-3/experiment-data/

# Verify
gsutil ls gs://trading-algo-3/experiment-data/
```

### Issue: "Config file not found"
```bash
# Regenerate configs
python ml_intraday_v3/experiments/generate_batch1_configs.py \
    --n-configs 200 \
    --output-dir ml_intraday_v3/experiments/batch1_configs
```

### Issue: "Import error: module not found"
```bash
# The GCP startup script installs all dependencies
# If running locally, install from requirements.txt
pip install -r requirements.txt
```

### Issue: "VM quota exceeded"
```bash
# Request quota increase in GCP Console
# Or use lower-spec VMs by editing launch script:
# MACHINE_TYPE="n1-standard-4"  # Instead of n1-highmem-4
```

### Issue: "Experiments taking too long"
```bash
# Check VM logs to see where it's stuck
gcloud compute ssh ml-batch1-vm --zone=us-central1-a \
    --command='tail -100 /var/log/syslog'

# Common causes:
# 1. Data download slow -> upload to GCS beforehand
# 2. Python deps install slow -> snapshot includes compiled wheels
# 3. Model training slow -> reduce n_estimators in configs
```

## Next Steps After Results

1. **Analyze Top Performers**
   - Which labeling method worked best?
   - Did fractional diff help?
   - Did CPCV improve generalization?

2. **Retrain Production Model**
   - Use best config from experiments
   - Train on full dataset (Oct 2024 - Feb 2026)
   - Validate on Mar-Apr 2026

3. **Live Trading Preparation**
   - Update live_trading configs with best params
   - Run paper trading simulation
   - Monitor for 1-2 weeks before Topstep combine

## References

- Plan document: `ml_intraday_v3/experiments/PLAN.md`
- Experiment runner: `ml_intraday_v3/experiments/comprehensive_grid_search_v2.py`
- Config generators: `ml_intraday_v3/experiments/generate_batch*.py`
- Analysis script: `ml_intraday_v3/experiments/analyze_batch_results.py`

## Contact & Support

For issues or questions:
1. Check logs in `ml_intraday_v3/experiments/logs/`
2. Review error messages in GCS: `gs://trading-algo-3/experiment-logs/`
3. Test locally first before GCP launch
4. Start with Batch 1 only (~$0.40) to validate setup
