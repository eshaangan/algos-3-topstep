# Monitoring Quick Reference - Feature Fix Verification

**Date**: February 4, 2026
**Purpose**: Quick commands to verify identical predictions fix

---

## Quick Health Check (30 seconds)

```bash
cd ml_intraday_v3
./verify_feature_fix.sh
```

**Look for**:
- ✅ Buffer: ~300 bars
- ✅ Feature quality: `healthy: True`
- ✅ Predictions: VARYING scores/probabilities
- ✅ Trades: Executing when P > 0.55

---

## Real-Time Monitoring (Live)

```bash
cd ml_intraday_v3
./monitor_signals.sh
```

**What to watch**:
- Signal generation every 5-15 minutes
- Different scores each time (NOT identical)
- Trades execute on high-probability signals

---

## Direct Log Access

### Stream Live Logs
```bash
gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
  --command='docker logs -f $(docker ps -q)'
```

### Check Last 50 Lines
```bash
gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
  --command='docker logs --tail=50 $(docker ps -q)'
```

---

## Specific Checks

### Buffer Size
```bash
gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
  --command='docker logs $(docker ps -q) 2>&1 | grep "Buffer initialized"'
```
**Expected**: `Buffer initialized with ~300 bars`

### Feature Quality
```bash
gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
  --command='docker logs $(docker ps -q) 2>&1 | grep "Feature quality"'
```
**Expected**: `healthy: True` (after warmup)

### Signal Diversity
```bash
gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
  --command='docker logs $(docker ps -q) 2>&1 | grep -E "score_ev|p_target" | tail -20'
```
**Expected**: VARYING values (not identical)

### Recent Trades
```bash
gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
  --command='docker logs $(docker ps -q) 2>&1 | grep -i "trade executed\|order submitted" | tail -10'
```
**Expected**: Trades when P > 0.55

---

## Success/Failure Indicators

### ✅ System Healthy
```
16:40 - score=0.123, p_target=0.562  ← Different
16:45 - score=0.087, p_target=0.544  ← Different
17:00 - score=0.151, p_target=0.576  ← Different
```

### ❌ System Broken
```
16:40 - score=0.085, p_target=0.543  ← Identical
16:45 - score=0.085, p_target=0.543  ← Identical
17:00 - score=0.085, p_target=0.543  ← Identical
```

---

## Troubleshooting

### No Signals Generated
**Check**:
1. Is CUSUM event detector working?
2. Is market open (RTH: 8:30 AM - 3:00 PM CT)?
3. Are features still warming up? (wait ~50 bars)

**Command**:
```bash
gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
  --command='docker logs $(docker ps -q) 2>&1 | grep -i "CUSUM\|regime change"'
```

### All Signals Rejected
**Check**:
1. Confidence threshold too high? (should be 0.55)
2. Feature quality check blocking? (should pass after warmup)
3. All predictions below threshold? (check diversity first)

**Command**:
```bash
gcloud compute ssh topstep-trader-vm --zone=us-central1-a \
  --command='docker logs $(docker ps -q) 2>&1 | grep -i "rejected\|below threshold"'
```

### Still Identical Predictions
**Action**: Rollback immediately
1. Reduce buffer to 200 bars (test)
2. If still broken, investigate feature calculation
3. Check preprocessing logic

---

## Emergency Rollback

```bash
cd "/Users/eshaanganguly/Documents/projects/algos 3 topstep"

# Edit config
vim ml_intraday_v3/configs/live_trading.yaml
# Change: lookback_bars: 200 (or 100 if emergency)

# Rebuild
docker buildx build --platform linux/amd64 \
  -t gcr.io/trading-algo-3/topstep-trader:latest \
  -f ml_intraday_v3/Dockerfile.production .

# Deploy
docker push gcr.io/trading-algo-3/topstep-trader:latest
gcloud compute instances stop topstep-trader-vm --zone=us-central1-a
gcloud compute instances start topstep-trader-vm --zone=us-central1-a
```

---

## Key Metrics to Track

### Daily Monitoring
- [ ] Buffer size: ~300 bars ✓
- [ ] Feature quality: `healthy: True` ✓
- [ ] Signal count: 5-15 per hour
- [ ] Signal diversity: >80% unique
- [ ] Trade execution: >30% of signals
- [ ] Win rate: >55% (backtest baseline)

### Weekly Review
- [ ] Prediction variance: Non-zero
- [ ] Feature distribution: Matches training
- [ ] No NaN warnings: All clear
- [ ] Trade frequency: Consistent with backtest
- [ ] P&L trending: Positive expectancy

---

## Contact

**Issues**: Run `./verify_feature_fix.sh` and review `FIX_IDENTICAL_PREDICTIONS_COMPLETE.md`

**Next Review**: February 4, 2026 21:00 UTC (2 hours post-deployment)
