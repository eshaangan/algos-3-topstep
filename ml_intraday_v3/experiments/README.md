# Comprehensive Grid Search for Model Edge Discovery

This directory contains infrastructure for running an exhaustive 3-phase grid search on GCP to find a model configuration with genuine predictive edge.

## Problem Statement

The current live trading system generates signals but no trades because all predicted probabilities cluster around 0.50 (random). Walk-forward validation confirmed AUC ~0.509 with massive overfitting gap (0.19-0.25). The system needs either:
1. A model configuration with genuine edge (AUC > 0.52, signals > 0.55), OR
2. Conclusive proof that ML lacks edge on this data (pivot to rule-based system)

## Solution: Three-Phase Hierarchical Search

### Phase 1: Coarse Grid (200-300 experiments, ~3 hours, ~$45)
- **Strategy**: Latin hypercube sampling across all major dimensions
- **Dimensions**:
  - 6 model complexity levels (minimal → aggressive)
  - 9 feature sets (full, top20, top15, top10, top5, structural, volatility, time, momentum)
  - 5 training windows (3, 6, 12, 18, 24 months)
  - 4 labeling barrier configs (PT/SL/Hz variations)
  - 3 sample weighting strategies (uniform, time_decay)
  - 3 calibration methods (isotonic, sigmoid, none)
- **Output**: Top 20 configurations ranked by composite score
- **Metrics**: Test AUC, train-test gap, % signals > 0.55, stability

### Phase 2: Fine Grid (150-250 experiments, ~2.5 hours, ~$37)
- **Strategy**: Create hyperparameter neighborhoods around top 20 from Phase 1
- **Fine-tuning**:
  - n_estimators: ±50
  - max_depth: ±1
  - min_child_samples: ×0.5, ×1.0, ×2.0
  - learning_rate: 0.01, 0.05, 0.1
  - feature_fraction: 0.8, 0.9, 1.0
  - bagging_fraction: 0.8, 0.9, 1.0
- **Output**: Top 5 configurations

### Phase 3: Validation & Ensemble (20-50 experiments, ~1 hour, ~$3)
- **Holdout validation**: Test top 5 on fresh Jul-Dec 2025 data
- **Ensemble experiments**: Test simple/weighted averaging and stacking
- **Final selection**: Choose model meeting all success criteria

**Total cost: ~$85 | Total time: ~7 hours | Remaining budget: $200**

## Success Criteria

For a configuration to be considered viable:
1. **Holdout AUC** > 0.52 (meaningful predictive edge)
2. **Signal strength**: ≥30% of predictions > 0.55
3. **Overfitting gap**: Train-test AUC difference < 0.15
4. **Stability**: AUC std across folds < 0.05
5. **Live viability**: Estimated 8-15 signals/day with P > 0.55

## File Structure

```
experiments/
├── README.md                          # This file
├── grid_config.yaml                   # Master configuration
├── comprehensive_grid_search.py       # Single experiment runner
├── gcp_orchestrator.py               # VM spawning and job distribution
├── analyze_results.py                # Results aggregation and ranking
├── gcp_startup.sh                    # VM initialization script
├── requirements_experiments.txt      # Python dependencies
└── results/                          # Local results cache
    ├── phase1/
    ├── phase2/
    └── phase3/
```

## Prerequisites

### 1. GCP Setup
```bash
# Authenticate
gcloud auth login
gcloud config set project trading-algo-3

# Create GCS bucket (if not exists)
gsutil mb -l us-central1 gs://trading-algo-3/
```

### 2. Upload Data to GCS
```bash
cd ml_intraday_v3

# Upload data
gsutil -m cp data/MES_5min_Oct2024_Dec2025.parquet \
    gs://trading-algo-3/experiment-data/

# Upload configs
gsutil -m cp configs/*.yaml gs://trading-algo-3/experiment-configs/

# Package and upload code
tar -czf /tmp/ml_intraday_v3.tar.gz \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    .
gsutil cp /tmp/ml_intraday_v3.tar.gz gs://trading-algo-3/code/
```

### 3. Install Local Dependencies
```bash
pip install -r experiments/requirements_experiments.txt
```

## Running the Grid Search

### Phase 1: Coarse Grid

```bash
cd ml_intraday_v3/experiments

# Run Phase 1 (spawns 10 VMs)
python gcp_orchestrator.py \
    --phase 1 \
    --num-vms 10 \
    --config grid_config.yaml

# Monitor progress (in another terminal)
watch -n 30 'python analyze_results.py --phase 1 --quick-stats --total-experiments 250'

# After completion, analyze results
python analyze_results.py --phase 1 --top-k 20

# View results
cat results/phase1_all_results.csv
```

**Output**: `results/phase1_top20.json` (input for Phase 2)

### Phase 2: Fine Grid

```bash
# Run Phase 2 using top 20 from Phase 1
python gcp_orchestrator.py \
    --phase 2 \
    --num-vms 10 \
    --config grid_config.yaml \
    --base-configs results/phase1_top20.json

# Monitor
watch -n 30 'python analyze_results.py --phase 2 --quick-stats --total-experiments 200'

# Analyze
python analyze_results.py --phase 2 --top-k 5
```

**Output**: `results/phase2_top5.json` (input for Phase 3)

### Phase 3: Validation & Ensemble

```bash
# Run Phase 3 using top 5 from Phase 2
python gcp_orchestrator.py \
    --phase 3 \
    --num-vms 2 \
    --config grid_config.yaml \
    --base-configs results/phase2_top5.json

# Analyze
python analyze_results.py --phase 3 --select-final
```

**Output**: Final production model selection

## Local Testing (Before GCP)

Test the single experiment runner locally to verify it works:

```bash
# Create a test config
cat > /tmp/test_config.json << 'EOFJSON'
{
  "exp_id": "local_test_001",
  "phase": 1,
  "model_name": "conservative",
  "model_params": {
    "n_estimators": 100,
    "max_depth": 4,
    "num_leaves": 15,
    "min_child_samples": 300,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "learning_rate": 0.05
  },
  "feature_set_name": "top10",
  "feature_set": ["side", "autocorr_5", "vol_regime", "vol_20", "ema_ratio", 
                  "relative_volume", "lower_wick", "vol_forecast", "parkinson_vol", "ema_spread"],
  "training_window_months": 6,
  "labeling": {"pt": 3.0, "sl": 2.5, "hz": 12},
  "sample_weight": "uniform",
  "calibration": "isotonic"
}
EOFJSON

# Run locally
python comprehensive_grid_search.py \
    --config /tmp/test_config.json \
    --data-dir ../data \
    --output /tmp/test_result.json

# Check result
cat /tmp/test_result.json | python -m json.tool
```

Expected runtime: 2-5 minutes locally

## Monitoring Live Progress

### Option 1: Quick Stats (No Download)
```bash
python analyze_results.py --phase 1 --quick-stats --total-experiments 250
```

### Option 2: GCS Web Console
Visit: https://console.cloud.google.com/storage/browser/trading-algo-3/experiment-results/

### Option 3: VM Logs (Debug Individual VM)
```bash
# SSH into a worker VM
gcloud compute ssh grid-worker-phase1-0 --zone us-central1-a

# View logs
tail -f /var/log/syslog | grep experiment
```

## Cost Control

### Budget Tracking
- Phase 1: ~$45 (10 VMs × 3 hours × $1.50/hour)
- Phase 2: ~$37 (10 VMs × 2.5 hours × $1.50/hour)
- Phase 3: ~$3 (2 VMs × 1 hour × $1.50/hour)
- **Total: ~$85** (well within $200 budget)

### Early Stop
If Phase 1 shows no promise (max AUC < 0.51 after 100 experiments):
```bash
# Stop running VMs
gcloud compute instances delete grid-worker-phase1-* --zone us-central1-a --quiet

# Pivot to rule-based system immediately
```

### Automatic Cleanup
VMs automatically shut down after completing their batch (see `gcp_startup.sh`). Manual cleanup:
```bash
# List all running VMs
gcloud compute instances list --filter="name~grid-worker"

# Delete all
gcloud compute instances delete $(gcloud compute instances list --filter="name~grid-worker" --format="value(name)") --zone us-central1-a --quiet
```

## Interpreting Results

### Good Result (Edge Found)
```
Rank  Exp ID                Test AUC  Sig>0.55  Gap     Model              Features  Window
1     phase1_exp_0089       0.547     42%       0.14    conservative       top10     6mo
2     phase1_exp_0156       0.541     38%       0.16    very_conservative  top15     12mo
```
- **Action**: Deploy best model to live trading
- **Expected**: 8-15 trades/day, 52-60% win rate, $300-700/day

### Bad Result (No Edge)
```
Rank  Exp ID                Test AUC  Sig>0.55  Gap     Model        Features  Window
1     phase1_exp_0023       0.514     12%       0.18    aggressive   full      24mo
2     phase1_exp_0067       0.511     8%        0.21    current      top20     18mo
```
- **Action**: ML approach lacks edge, switch to `rule_based_v1/`
- **Rationale**: Exhaustive search found nothing viable

## Troubleshooting

### VM Creation Fails
```bash
# Check quotas
gcloud compute project-info describe --project=trading-algo-3

# Request quota increase if needed
# https://console.cloud.google.com/iam-admin/quotas
```

### Experiments Timeout
- Increase `timeout_per_experiment_minutes` in `grid_config.yaml`
- Reduce `num_samples` in Phase 1
- Use smaller training windows

### Out of Memory
- Increase VM `machine_type` to `n2-highmem-8` (more RAM)
- Reduce feature set size
- Reduce training window length

### Data Not Found
```bash
# Verify data uploaded
gsutil ls gs://trading-algo-3/experiment-data/

# Re-upload if missing
gsutil cp ../data/MES_5min_Oct2024_Dec2025.parquet \
    gs://trading-algo-3/experiment-data/
```

## Next Steps After Grid Search

### If Edge Found (≥1 config meets criteria)
1. Save best model bundle:
   ```bash
   # Train final model with best config
   python train_best_model.py --config results/phase3_best.json
   ```

2. Update live trading config:
   ```yaml
   # configs/live_trading.yaml
   model:
     path: "models/saved/model_bundle_grid_search_best_v1.pkl"
     confidence_threshold: 0.55
   ```

3. Backtest on Jan 2026:
   ```bash
   python run_backtest.py --start 2026-01-01 --end 2026-01-31
   ```

4. Deploy to GCP and paper trade for 2-3 days

5. Go live if paper trading successful

### If No Edge Found (0 configs meet criteria)
1. Document findings:
   - "Exhaustive search of 480+ configurations found no viable ML model"
   - "Best AUC: X.XXX (below 0.52 threshold)"
   - "Conclusion: MES 5-min intraday lacks ML-exploitable patterns"

2. Switch to rule-based system:
   ```bash
   cd ../../rule_based_v1
   python scripts/run_backtest.py --start 2026-01-01 --end 2026-01-31
   ```

3. Deploy `rule_based_v1/` to GCP

4. Monitor rule-based performance

## Configuration Reference

### Model Complexity Levels
- **minimal**: 20 trees, depth 2 → prevents overfitting, may underfit
- **very_conservative**: 50 trees, depth 3 → low capacity
- **conservative**: 100 trees, depth 4 → **recommended starting point**
- **current**: 150 trees, depth 6 → existing (overfits badly)
- **moderate**: 200 trees, depth 5 → balanced
- **aggressive**: 300 trees, depth 7 → high capacity, overfitting risk

### Feature Sets
- **full**: All 34 features → may have noise
- **top20**: Top 20 by gain → balanced
- **top10**: Top 10 by gain → **recommended**, reduces noise
- **top5**: Minimal set → prevents overfitting but may miss signals
- **structural**: Candle patterns only → pure price action
- **volatility**: Vol-based features → regime-aware
- **time**: Time features only → session effects
- **momentum**: Trend/momentum → directional bias

### Training Windows
- **3 months**: Recent data only → adaptive but unstable
- **6 months**: **Recommended** → balances recency and stability
- **12 months**: Full year → stable but may be stale
- **18-24 months**: Very stable → risk of concept drift

## Support

For issues or questions:
1. Check logs: `gcloud compute instances get-serial-port-output <vm-name>`
2. Review this README thoroughly
3. Test locally before running on GCP
4. Start with Phase 1, ensure it works before Phase 2/3

---

**Remember**: The goal is to either find edge OR conclusively prove it doesn't exist. Both outcomes are valuable and will inform the deployment decision (ML vs rule-based).
