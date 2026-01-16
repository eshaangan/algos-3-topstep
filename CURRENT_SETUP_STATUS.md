# Current Setup Status - Topstep 50K (5m) Candidate

**Date**: 2026-01-15
**Status**: ✅ TOPSTEP-CANDIDATE (ACCEL + CIRCUIT BREAKERS)

---

## 🚫 Legacy Model (DO NOT USE FOR COMBINE)

**Model Bundle**: `ml_intraday_v3/models/saved/model_bundle.pkl`

**Why it fails Topstep**:
- **Max drawdown**: ~$4,082 (limit $2,500)
- **Daily loss limit breaches**: 8 days > $1,000
- **K-Fold CV**: ~124% (severe instability)
- **2/6 folds lose money** (fold_0, fold_1)

---

## ✅ Recommended Combine Setup (CUSUM / Dual / 5m)

**Live Config (accelerated)**: `ml_intraday_v3/configs/live_trading_topstep_50k_accel.yaml`

**Risk Config (mid, allows 3 contracts + sigma sizing)**: `ml_intraday_v3/configs/risk_topstep_50k_mid_c3.yaml`

**Model Bundle**: `ml_intraday_v3/models/saved/model_bundle_topstep_candidate.pkl`

**Backtest Config (matches sizing logic)**: `ml_intraday_v3/configs/backtest_topstep_accel_t06_rth_dynsigma_c3.yaml`

**Key Backtest Result Directory (dual, RTH, dynamic sizing)**:
- `runs/cusum_dual_24h_20260114/bar_size=5m/backtests/purged_kfold__topstep_accel_dual_regularized_t06_rth_dynsigma_c3_riskmid`

**Topstep Monte Carlo (trade-resampling, consistency = best day ≤ 50% total profit)**:
- `pass_rate`: ~0.843
- `pass_within_15d`: ~0.622
- `days_to_pass_p50`: 13
- `daily_loss_violations` (stitched): 0

**Why this is the current best fit**
- Dynamic sigma sizing prevents the single 5m bar loss that caused -$1,000+ days at 2 contracts.
- 3/2/1/0 contract ladder improves speed while keeping high-sigma exposure low.

---

## ✅ Other Artifacts Created (not the primary recommendation)

### 1) Ensemble Bundle (Short-Term Fix)
- **Path**: `ml_intraday_v3/models/saved/model_bundle_ensemble.pkl`
- **Base Models**: 6 folds from `runs/bidirectional_24h_20260114`
- **Status**: Built, **backtest pending**

### 2) Regularized Training (Long-Term Fix)
- **Config**: `ml_intraday_v3/configs/training_regularized.yaml`
- **Run**: `runs/regularized_24h_20260114/bar_size=5m`
- **Notebook**: `ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb`
- **Status**: Completed, **still fails Topstep limits**
  - CV ~116.8% (target < 60%)
  - Max drawdown ~$4,143 (limit $2,500)
  - Daily loss violations: 9

### 3) Regularized + CUSUM (No Lookahead)
- **Config**: `ml_intraday_v3/configs/labeling_cusum.yaml`
- **Run**: `runs/regularized_cusum_24h_20260114/bar_size=5m`
- **Status**: Completed, **worse stability**
  - CV ~230.5% (target < 60%)
  - Max drawdown ~$5,233 (limit $2,500)
  - Daily loss violations: 9

---

## ✅ Risk Management (Live Trading)

Circuit breakers added to reduce Topstep breach probability:
- **Configs**: `ml_intraday_v3/configs/risk_topstep_50k_*.yaml`
- **Live Runner Integration**: `ml_intraday_v3/live_trading/live_runner.py`
- **Risk Manager Class**: `ml_intraday_v3/live_trading/risk_manager.py`

Key protections:
- **Daily loss hard stop**: $1,000
- **Critical stop**: configurable (default accel: $650)
- **Sigma sizing**: 3/2/1/0 contracts by sigma bucket
- **Trailing drawdown cap**: $2,500
- **Time filters**: avoid 17:00-18:00 ET and 15:00-16:00 ET
- **Threshold bump on losses**: configurable (default accel: +0.03)

---

## 📊 Latest Monte Carlo (Legacy Bidirectional Model)

From `analysis/topstep_50k_results_bidirectional_24h.json` (legacy):
- **Sequential Pass**: ❌ Failed
- **Failure Drivers**: Daily loss + trailing drawdown breaches

⚠️ **Note**: Monte Carlo sampling across folds is optimistic; sequential run still fails Topstep limits.

---

## 🚀 Next Steps

1. **Paper trade the accelerated config for 2–5 sessions**
   - Config: `ml_intraday_v3/configs/live_trading_topstep_50k_accel.yaml`
2. **If volatility regime changes**
   - Tighten sigma thresholds (e.g., move 2->1 cut from 30 to 28) before increasing thresholds or size.

---

## 📁 Key Files

**Models**
- `ml_intraday_v3/models/saved/model_bundle.pkl` (current, overfit)
- `ml_intraday_v3/models/saved/model_bundle_ensemble.pkl` (new)

**Runs**
- `runs/bidirectional_24h_20260114/bar_size=5m/`

**Configs**
- `ml_intraday_v3/configs/training_regularized.yaml`
- `ml_intraday_v3/configs/live_trading.yaml`
- `ml_intraday_v3/configs/live_trading_topstep_50k_accel.yaml`
- `ml_intraday_v3/configs/risk_topstep_50k_mid_c3.yaml`

**Reports**
- `analysis/topstep_50k_results_bidirectional_24h.json`
- `analysis/topstep_50k_equity_curve_bidirectional_24h.png`
- `analysis/overfitting_fixes_comparison.md`
