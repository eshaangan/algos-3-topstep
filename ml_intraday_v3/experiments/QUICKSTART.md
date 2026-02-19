# Quick Start - GCP Experiments

## TL;DR - 5 Minutes to Launch

```bash
# 1. Update GCP config (REQUIRED)
vim ml_intraday_v3/experiments/launch_gcp_experiments.sh
# Change PROJECT_ID and BUCKET to your values

# 2. Upload data to GCS (one-time)
gsutil cp data/processed/*.parquet gs://your-bucket/experiment-data/

# 3. Launch experiments
cd /Users/eshaanganguly/Documents/projects/algos\ 3\ topstep
./ml_intraday_v3/experiments/launch_gcp_experiments.sh all

# 4. Monitor progress (optional)
watch -n 30 'gsutil ls gs://your-bucket/experiment-results/batch*/*.json | wc -l'

# 5. Download results when complete (~45 min)
gsutil -m cp -r gs://your-bucket/experiment-results/* ml_intraday_v3/experiments/results/

# 6. Analyze results
python ml_intraday_v3/experiments/rank_models.py \
    --results-dir ml_intraday_v3/experiments/results \
    --metric sharpe_ratio \
    --top-n 10
```

## What Gets Launched?

**3 GCP VMs** running in parallel:
- **Batch 1**: 200 configs testing labeling methods (triple barrier vs trend scanning)
- **Batch 2**: 200 configs testing CV methods (KFold vs CPCV) and features
- **Batch 3**: 200 configs testing meta-labeling architectures

**Cost**: ~$1.20 total
**Time**: ~30-45 minutes
**VMs**: Preemptible n1-highmem-4 (auto-shutdown when done)

## What Problem Does This Solve?

**Current Issue**: All 100 models from previous exhaustive search produced:
- Identical negative PnL: -$1,212.56
- Identical win rate: 41.3%
- Identical signal count: 624

This proves model predictions are **irrelevant** - labels alone drive outcomes.

**Root Cause**: Deterministic triple barrier labels that don't use model probabilities.

**Solution**: Test 600 experiments with:
1. **Trend scanning labels** - adaptive horizons based on t-values
2. **Fractional differentiation** - stationary features that preserve memory
3. **CPCV** - purged cross-validation to prevent temporal leakage
4. **Sample uniqueness weighting** - reduce overfitting on overlapping labels
5. **Meta-labeling** - two-stage filtering for higher precision

## Before You Launch

### ✅ Checklist

1. **GCP Account Ready**
   ```bash
   gcloud auth list  # Should show your account
   gcloud config get-value project  # Should show project ID
   ```

2. **Data Uploaded to GCS**
   ```bash
   gsutil ls gs://your-bucket/experiment-data/*.parquet
   # Should show at least one .parquet file
   ```

3. **Configs Generated**
   ```bash
   ls ml_intraday_v3/experiments/batch1_configs/*.json | wc -l  # Should be 201
   ls ml_intraday_v3/experiments/batch2_configs/*.json | wc -l  # Should be 201
   ls ml_intraday_v3/experiments/batch3_configs/*.json | wc -l  # Should be 201
   ```

4. **Test Locally First** (RECOMMENDED)
   ```bash
   python ml_intraday_v3/experiments/test_single_experiment.py \
       --config ml_intraday_v3/experiments/batch1_configs/batch1_exp_00001.json \
       --data-dir data/processed \
       --output /tmp/test_result.json

   # Should complete in 2-5 min and save result to /tmp/test_result.json
   ```

### 🚨 Cost Warning

Estimated costs:
- **Batch 1 only**: ~$0.40 (~15 min)
- **All 3 batches**: ~$1.20 (~45 min)

Preemptible VMs can be terminated by GCP. If this happens:
- Results saved so far are preserved in GCS
- Relaunch script will skip completed experiments
- Total cost may be slightly higher due to restart overhead

Set budget alert:
```bash
# In GCP Console > Billing > Budgets & alerts
# Set alert at $5, hard stop at $10
```

## Launch Options

### Option 1: All Batches (Recommended)
```bash
./ml_intraday_v3/experiments/launch_gcp_experiments.sh all
```

Launches 3 VMs in parallel. Results in ~45 minutes.

### Option 2: Single Batch (Conservative)
```bash
# Test Batch 1 first (~$0.40)
./ml_intraday_v3/experiments/launch_gcp_experiments.sh batch1

# If Batch 1 results look good, launch others
./ml_intraday_v3/experiments/launch_gcp_experiments.sh batch2
./ml_intraday_v3/experiments/launch_gcp_experiments.sh batch3
```

### Option 3: Manual Control
```bash
# 1. Upload configs
gsutil -m cp ml_intraday_v3/experiments/batch1_configs/*.json \
    gs://your-bucket/experiments/batch1_configs/

# 2. Upload code snapshot
tar -czf /tmp/snapshot.tar.gz ml_intraday_v3/
gsutil cp /tmp/snapshot.tar.gz gs://your-bucket/code-snapshots/

# 3. Launch VM manually
gcloud compute instances create ml-batch1-vm \
    --zone=us-central1-a \
    --machine-type=n1-highmem-4 \
    --preemptible \
    --scopes=cloud-platform \
    --metadata=batch=batch1,bucket=gs://your-bucket \
    --metadata-from-file=startup-script=ml_intraday_v3/experiments/gcp_startup_v2.sh
```

## Monitoring

### Check Progress
```bash
# Count completed experiments
gsutil ls gs://your-bucket/experiment-results/batch1/*.json | wc -l
gsutil ls gs://your-bucket/experiment-results/batch2/*.json | wc -l
gsutil ls gs://your-bucket/experiment-results/batch3/*.json | wc -l

# Auto-refresh every 30 seconds
watch -n 30 'gsutil ls gs://your-bucket/experiment-results/batch*/*.json | wc -l'
```

### View Logs
```bash
# SSH into VM
gcloud compute ssh ml-batch1-vm --zone=us-central1-a

# Tail syslog
tail -f /var/log/syslog

# View experiment logs
ls /tmp/*.log

# Exit SSH
exit
```

### Check VM Status
```bash
# List all VMs
gcloud compute instances list

# Should show 3 VMs running (or terminated if complete)
```

## When Complete

### Download Results
```bash
# Download all results
mkdir -p ml_intraday_v3/experiments/results
gsutil -m cp -r gs://your-bucket/experiment-results/* \
    ml_intraday_v3/experiments/results/

# Check file count
ls ml_intraday_v3/experiments/results/batch*/*.json | wc -l
# Should be ~600 (200 per batch)
```

### Analyze Results
```bash
# Rank by Sharpe ratio
python ml_intraday_v3/experiments/rank_models.py \
    --results-dir ml_intraday_v3/experiments/results \
    --metric sharpe_ratio \
    --top-n 10 \
    --output ml_intraday_v3/experiments/results/top_10_sharpe.json

# Rank by win rate
python ml_intraday_v3/experiments/rank_models.py \
    --results-dir ml_intraday_v3/experiments/results \
    --metric win_rate \
    --top-n 10 \
    --output ml_intraday_v3/experiments/results/top_10_winrate.json

# Rank by PnL
python ml_intraday_v3/experiments/rank_models.py \
    --results-dir ml_intraday_v3/experiments/results \
    --metric test_pnl \
    --top-n 10 \
    --output ml_intraday_v3/experiments/results/top_10_pnl.json
```

### View Top Models
```bash
# View Sharpe top 10
cat ml_intraday_v3/experiments/results/top_10_sharpe.json | python -m json.tool | less

# Extract configs
jq '.[].config' ml_intraday_v3/experiments/results/top_10_sharpe.json
```

## Success Criteria

**Minimum Success**: Find ANY model with:
- ✅ Positive PnL on Jan-Feb 2026 (vs current -$1,212.56)
- ✅ Win rate > 45% (vs current 41.3%)
- ✅ Different signals across models (proves labels matter)

**Target Success**: Top model with:
- ✅ Sharpe > 1.5
- ✅ Win rate > 52%
- ✅ Max drawdown < $800

**Stretch Goal**: Meta-labeling with:
- ✅ Combined Sharpe > 2.0
- ✅ Precision > 65%
- ✅ Ready for Topstep combine

## Troubleshooting

### "No data files found in GCS"
```bash
# Upload data
gsutil cp data/processed/*.parquet gs://your-bucket/experiment-data/

# Verify
gsutil ls gs://your-bucket/experiment-data/
```

### "Config not found"
```bash
# Regenerate configs (already done, but in case of corruption)
python ml_intraday_v3/experiments/generate_batch1_configs.py \
    --n-configs 200 \
    --output-dir ml_intraday_v3/experiments/batch1_configs
```

### "VM won't start"
```bash
# Check quota
gcloud compute project-info describe --project=your-project-id

# If quota exceeded, use smaller VMs
# Edit launch script: MACHINE_TYPE="n1-standard-4"
```

### "Experiments taking forever"
```bash
# Check if VM is stuck
gcloud compute ssh ml-batch1-vm --zone=us-central1-a \
    --command='ps aux | grep python'

# If stuck, delete and relaunch
gcloud compute instances delete ml-batch1-vm --zone=us-central1-a
./ml_intraday_v3/experiments/launch_gcp_experiments.sh batch1
```

### "VM preempted"
```bash
# Check summary
gsutil cat gs://your-bucket/experiment-results/batch1_summary.json

# Relaunch (will skip completed experiments)
./ml_intraday_v3/experiments/launch_gcp_experiments.sh batch1
```

## Next Steps

After analyzing results:

1. **Identify what worked**
   - Which labeling method performed best?
   - Did fractional differentiation help?
   - Did CPCV improve generalization?

2. **Retrain production model**
   - Use best config from top 10
   - Train on full dataset
   - Validate on truly unseen data (Mar-Apr 2026)

3. **Update live trading**
   - Apply best config to live_trading.yaml
   - Run paper trading simulation
   - Monitor for 1-2 weeks before Topstep combine

## Support

Full documentation: `ml_intraday_v3/experiments/IMPLEMENTATION_GUIDE.md`

For issues:
1. Test locally first: `test_single_experiment.py`
2. Check GCP logs: `gsutil ls gs://your-bucket/experiment-logs/`
3. Start with Batch 1 only to validate setup
