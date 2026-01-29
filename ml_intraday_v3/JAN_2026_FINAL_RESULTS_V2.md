# Jan 2026 Final Results (V2) — `BALANCED_V3`
**Generated**: 2026-01-27  
**Model bundle**: `ml_intraday_v3/models/saved/model_bundle_balanced_v3.pkl`  
**Bar size**: 5m  
**Jan 2026 data**: `ml_intraday_v3/backtest_results/jan_2026_test/bar_size=5m/bars.parquet`  

## Executive summary
- **Circuit breaker disabled** (operator override) for now.
- **Live risk**: fixed so entries use **server-side bracket orders** (stop-loss + take-profit) instead of “naked” entries.
- **Performance**: `BALANCED_V3` is **still negative expectancy** on Jan 2026 and most longer slices tested. It is **not production-ready** for a Topstep Combine as-is.

## Key Jan 2026 results (out-of-sample)
From `ml_intraday_v3/test_multi_period.py` (Jan 2026 slice):
- **Trades**: 319  
- **Direction mix**: 79.6% LONG / 20.4% SHORT  
- **Win rate**: 44.8%  
- **Avg trade**: -$5.30  
- **Total P&L**: **-$1,692**  

## Multi-period validation (5m, regime/vol filters as currently enforced in replay)
From `ml_intraday_v3/test_multi_period.py` run on 2026-01-27:

| Period | Trades | LONG% | SHORT% | Win% | Avg trade | Total P&L |
|---|---:|---:|---:|---:|---:|---:|
| Q1 2024 (Bear/Correction) | 128 | 100.0% | 0.0% | 36.7% | -$7.75 | -$992 |
| Q3 2024 (Volatile) | 123 | 100.0% | 0.0% | 36.6% | +$11.79 | +$1,450 |
| Full Year 2024 | 287 | 100.0% | 0.0% | 34.1% | -$7.02 | -$2,016 |
| Jan–Nov 2025 (In-sample sanity) | 46 | 100.0% | 0.0% | 19.6% | -$43.58 | -$2,005 |
| Dec 2025 (Holdout) | 71 | 100.0% | 0.0% | 14.1% | -$42.79 | -$3,038 |
| **Jan 2026 (Out-of-sample)** | **319** | **79.6%** | **20.4%** | **44.8%** | **-$5.30** | **-$1,692** |

## Overfitting / robustness notes (what the tests suggest)
- **Strong time-instability**: performance flips materially across regimes/years (one slice positive, most negative). This is a classic symptom of **non-stationarity + regime dependence**.
- **Directional imbalance**: most historical slices are effectively **LONG-only** (100/0), while Jan 2026 shows some short participation. This mismatch needs to be understood (model behavior vs filters).
- **Conclusion**: even with some improvements vs the broken baseline, the strategy **does not yet show robust profitability across multiple time windows**, which is the minimum bar for Combine readiness.

## Research takeaways translated into low-overfitting action items
### From “Artificial Intelligence for Trading Strategies” (Jevtic et al., arXiv:2208.07168)
- **Evaluate by market phase, not just overall averages**: explicitly track performance in **high-vol vs low-vol** and **bull/bear** phases; this paper shows models can look “good on average” but fail badly in specific regimes.
- **Keep complexity proportional**: add only what improves *out-of-sample* performance; avoid feature bloat.

### From “A review of machine learning experiments in equity investment…” (Buczynski et al., 2021)
- **Avoid “cherry-picking”**: don’t run many variants and report the best; treat each change as a “single configuration” test to reduce accidental overfitting.
- **Backtest realism**: always include execution realism (costs/slippage assumptions), and prefer **robust, out-of-sample** comparisons over in-sample “hit-rate” style wins.

## Recommended next steps (keep it simple)
- **Tighten entry selectivity**: raise `primary_threshold` and/or add a lightweight volatility/market-quality gate (e.g., skip low-vol chop).
- **Feature pruning**: reduce to a small, stable feature set using permutation importance across multiple periods; retrain and re-test the same slices.
- **Directional sanity checks**: verify LONG/SHORT mix *per regime*; if shorts are being blocked or never chosen, fix that first before further tuning.

## Repro commands
- Multi-period validation:
  - `python3 ml_intraday_v3/test_multi_period.py`

# January 2026 Final Validation Report (V2)

**Date**: January 25, 2026  
**Status**: STRUCTURAL FIX CONFIRMED, PERFORMANCE OPTIMIZATION REQUIRED

---

## Executive Summary

We successfully fixed the **structural directional bias** in the trading model. The new `BALANCED_V3` model, trained with the 'side' feature included, is capable of predicting both LONG and SHORT trades. However, profitability remains a challenge in the difficult January 2026 market regime.

### Key Achievements
1. **✅ Directional Bias Fixed**: Model went from 100% LONG to **79% SHORT / 21% LONG** in Jan 2026.
2. **✅ Technical Implementation**: Successfully integrated 'side' feature into training and prediction pipeline.
3. **✅ Research Integration**: Analyzed academic literature to identify next steps for performance improvement.

### Critical Issues
1. **❌ Low Win Rate**: The new model achieved only **11.3% win rate** in Jan 2026.
2. **❌ Losses**: Net loss of **-$2,188** (similar to baseline models).
3. **Insight**: The model correctly identified the bearish/choppy regime (hence mostly SHORT), but likely suffered from timing issues or "whipsaw" price action typical of volatile transitions.

---

## Comparative Results: January 2026

| Model | Training Data | Directional Bias | Win Rate | Total P&L | Status |
|-------|---------------|------------------|----------|-----------|--------|
| **OLD_BASELINE** | Bull Market Only | 100% LONG | 19.0% | -$2,277 | ❌ Structurally Broken |
| **RETRAINED_CLEAN**| Bull Market Only | 100% LONG | 17.6% | -$2,434 | ❌ Structurally Broken |
| **BALANCED_V3** | **Balanced (Bull/Bear)** | **79% SHORT / 21% LONG** | **11.3%** | **-$2,189** | ✅ **Structurally Fixed** |

### Why is the Win Rate Lower?
The `BALANCED_V3` model correctly shifted to SHORTing, but January 2026 appears to be a **mean-reverting / choppy** market rather than a clean trend.
- **Trend Following models** (like ours) get killed in chop.
- Predicting SHORT in a choppy market often leads to selling at the bottom of a range (just before a bounce).
- The 100% LONG models lost slightly less simply because the market might have had a slight upward drift or fewer triggers.

---

## Research Insights: "AI for Trading Strategies" (arXiv:2208.07168)

We analyzed the paper *Artificial Intelligence for Trading Strategies* (Jevtic et al., 2022) to understand how to improve performance in volatile regimes.

### Key Takeaways
1. **Random Forest (Tree Ensembles) Superiority**: 
   - Tree-based models (like our LightGBM) outperformed LSTM and SVM in high-volatility periods (e.g., COVID-19).
   - **Insight**: We are on the right track with LightGBM, but may need deeper trees or different hyperparameters.

2. **Regime Sensitivity**:
   - Models like kNN failed completely in crises.
   - Robustness comes from training on diverse regimes (which we started with 2024-2025 data).

3. **Critical Features**:
   - Top performing features were **MACD** and **Stochastic Oscillator (K-Percent)**.
   - **Action Item**: Verify our feature set includes these momentum oscillators to help in mean-reverting markets.

4. **Extreme Value Analysis**:
   - Models perform best when predicting "extreme" returns.
   - **Action Item**: We might need to increase our `primary_threshold` (currently 0.10) to only take high-confidence trades, filtering out the "noise" that causes the 89% loss rate.

---

## Recommendations & Next Steps

### 1. Optimize Thresholds (Immediate)
The current win rate (11%) suggests we are taking too many low-quality trades.
- **Action**: Run a parameter sweep on `primary_threshold` (0.10 -> 0.15, 0.20).
- **Hypothesis**: Higher threshold = fewer trades, higher win rate.

### 2. Feature Engineering (Short Term)
- **Action**: Explicitly add **MACD** and **Stochastic Oscillator** to the feature set if missing.
- **Reason**: These are mean-reversion indicators that help prevent selling at bottoms or buying at tops in choppy markets.

### 3. Regime Detection (Medium Term)
- **Action**: Implement a "Regime Filter".
- If `Volatility > High` AND `TrendStrength < Low` (Choppy), **DISABLE TRADING** or switch to a Mean Reversion model.
- Our current model is likely a Trend Follower trying to trade a Choppy market.

### 4. Stop Loss / Take Profit Tuning
- **Action**: Re-evaluate the 1.5x / 2.5x ATR multiples.
- In volatile chop, stops are hit frequently before the move develops. Widening stops or reducing position size might be necessary.

---

## Conclusion

We have **solved the critical engineering flaw** (directional bias). The model is now a valid bidirectional trader. The remaining challenge is **financial performance optimization**, which requires standard quant workflow (feature selection, threshold tuning, regime filtering) rather than bug fixing.

**DO NOT DEPLOY YET.** The model needs tuning to achieve >50% win rate.
