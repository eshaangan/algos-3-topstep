## Appendix B) Quant-Guild-Library ideas to integrate into V3 (ES/MES intraday)

This appendix documents **concrete, reusable ideas** extracted from the Quant-Guild-Library repo and maps them into V3 modules. This is additive and does not replace existing V3 design.

### B.1 Volatility Forecast module (EWMA / ARCH / GARCH) for triple-barrier + sizing
**Motivation:** triple-barrier labels and Topstep-safe sizing both require a volatility estimate \(\sigma\). Using only ATR/RV is acceptable, but a forecast volatility model (EWMA/GARCH) can adapt faster to volatility clustering.

**Core equations**
- ARCH(q): \(\sigma_t^2 = \alpha_0 + \alpha_1\epsilon_{t-1}^2 + \dots + \alpha_q\epsilon_{t-q}^2\)
- GARCH(p,q): \(\sigma_t^2 = \alpha_0 + \sum_{i=1}^{q} \alpha_i\epsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j\sigma_{t-j}^2\)
- Common GARCH(1,1): \(\sigma_t^2 = \omega + \alpha\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2\)

**Implementation**
- `ml_intraday_v3/features/volatility_forecast.py`
  - `ewma_vol(returns, lambda_)`
  - optional: `garch11_vol(returns)` after baseline stability
- `ml_intraday_v3/labels/triple_barrier.py`
  - `sigma_t0 = vol_forecast_t0`
  - `pt = m_pt * sigma_t0`, `sl = m_sl * sigma_t0`
- `ml_intraday_v3/core/risk_sizing.py`
  - size contracts inversely with volatility (bounded + Topstep caps)

### B.2 Markov regime features (3-state posterior probabilities)
**Motivation:** intraday markets switch regimes (low/med/high vol, trend/chop). A small Markov filter provides stable regime probabilities that the **meta-model** can learn from.

**Implementation (ported from repo)**
- `ml_intraday_v3/features/regime_markov.py`
  - maintain `transition_matrix`, emission `means/stds`, and `state_probs`
  - each bar: compute observed vol; posterior \(\propto\) prior × likelihood; normalize
  - output:
    - `regime_p0, regime_p1, regime_p2`
    - `regime_state = argmax(p)`
- Uses:
  - meta-labeling selectivity
  - barrier multiplier adjustments by regime
  - volatility-based sizing adjustments

### B.3 “Violated assumptions” hardening (fat tails, dependence, nonstationarity)
**Motivation:** intraday returns are heavy-tailed, correlated, and nonstationary; naive metrics and iid assumptions exaggerate edge.

**Implementation**
- `ml_intraday_v3/analysis/assumptions.py`
  - rolling kurtosis / tail metrics
  - autocorrelation diagnostics (e.g., Ljung-Box)
  - regime stability summaries
- `ml_intraday_v3/analysis/bootstrap.py`
  - block bootstrap CIs for metrics (not iid bootstrap)
- Add acceptance gates:
  - min trades per fold
  - report uncertainty intervals

### B.4 PCA (fold-safe) and component drift monitoring
**Motivation:** reduces collinearity and stabilizes models with many correlated features. Must be fold-safe and monitored for drift.

**Implementation**
- `ml_intraday_v3/features/transforms/pca.py`
  - fit PCA on train fold only; transform val/test
  - persist components + explained variance
- optional monitoring:
  - compare live loadings to train loadings (drift score)

### B.5 Kelly sizing (optional, fractional + capped)
**Motivation:** tie size to edge, but keep Topstep constraints primary.

**Implementation**
- `ml_intraday_v3/core/sizing_kelly.py`
  - compute fractional Kelly from calibrated probabilities + payoff ratio
  - apply strong caps (contracts, $ risk, daily loss, trailing drawdown)

