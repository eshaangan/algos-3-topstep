# Quick Start: Multi-Market Trading (MES + NKD)

**Goal**: Pass Topstep Combine faster by trading both US (MES) and Asian (NKD) sessions using free historical data.

## Why This Works

✅ **16 hours of trading** vs. 6.5 hours (MES-only)
✅ **Lower correlation** (~0.65 vs. 0.9 for ES/NQ)
✅ **Same Topstep account** - live data for both
✅ **FREE historical data** - Yahoo Finance
✅ **More setups = faster profit target**

## Markets

| Market | RTH Hours (CT) | Advantage |
|--------|---------------|-----------|
| **NKD** (Nikkei) | 18:00-03:00 | Trade Asian session |
| **MES** (S&P 500) | 08:30-15:00 | Trade US session |

**No overlap** = lower correlation, cleaner signals

---

## Phase 1: Get Historical Data (TODAY)

### Step 1: Install yfinance
```bash
pip install yfinance
```

### Step 2: Download Free Historical Data
```bash
cd ml_intraday_v3/data

# Download 5 years of daily data (FREE from Yahoo Finance)
python download_free_futures.py --start 2020-01-01 --end 2025-12-31

# Output:
# data/raw_futures/mes_daily_2020-01-01_2025-12-31.parquet
# data/raw_futures/nkd_daily_2020-01-01_2025-12-31.parquet
```

### Step 3: Verify Data Quality
```python
import pandas as pd

# Load MES data
mes = pd.read_parquet("data/raw_futures/mes_daily_2020-01-01_2025-12-31.parquet")
print(f"MES: {len(mes)} days, {mes.index.min()} to {mes.index.max()}")

# Load NKD data
nkd = pd.read_parquet("data/raw_futures/nkd_daily_2020-01-01_2025-12-31.parquet")
print(f"NKD: {len(nkd)} days, {nkd.index.min()} to {nkd.index.max()}")

# Check for gaps (missing days)
# TODO: Add gap detection logic
```

---

## Phase 2: Calculate Correlations (1 HOUR)

### Verify Diversification Benefit

Create `ml_intraday_v3/analysis/analyze_correlations.py`:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
mes = pd.read_parquet("data/raw_futures/mes_daily_2020-01-01_2025-12-31.parquet")
nkd = pd.read_parquet("data/raw_futures/nkd_daily_2020-01-01_2025-12-31.parquet")

# Calculate daily returns
mes['returns'] = mes['Close'].pct_change()
nkd['returns'] = nkd['Close'].pct_change()

# Merge on date
merged = pd.merge(
    mes[['returns']].rename(columns={'returns': 'mes_returns'}),
    nkd[['returns']].rename(columns={'returns': 'nkd_returns'}),
    left_index=True,
    right_index=True,
    how='inner'
)

# Overall correlation
overall_corr = merged.corr().loc['mes_returns', 'nkd_returns']
print(f"Overall Correlation: {overall_corr:.3f}")

# Rolling 60-day correlation
rolling_corr = merged['mes_returns'].rolling(60).corr(merged['nkd_returns'])
print(f"Rolling Correlation (60d): Mean={rolling_corr.mean():.3f}, Std={rolling_corr.std():.3f}")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Returns scatter
ax1.scatter(merged['mes_returns'], merged['nkd_returns'], alpha=0.3)
ax1.set_xlabel('MES Daily Returns')
ax1.set_ylabel('NKD Daily Returns')
ax1.set_title(f'MES vs NKD Returns (Corr={overall_corr:.3f})')
ax1.grid(True, alpha=0.3)

# Rolling correlation
ax2.plot(rolling_corr)
ax2.axhline(overall_corr, color='r', linestyle='--', label=f'Mean={overall_corr:.3f}')
ax2.set_xlabel('Date')
ax2.set_ylabel('60-day Rolling Correlation')
ax2.set_title('MES-NKD Rolling Correlation')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/mes_nkd_correlation.png', dpi=150)
print("✓ Chart saved to analysis/mes_nkd_correlation.png")
```

**Expected Result**: Correlation ~0.6-0.7 (much better than ES/NQ at ~0.9)

---

## Phase 3: Adapt Your Pipeline (2-3 DAYS)

### Option A: Quick Validation (Fastest)

**Train separate models, combine signals manually**

1. Train MES model (you already have this):
   ```bash
   python -m ml_intraday_v3.cli build-train --config configs/training_mes.yaml
   ```

2. Create NKD config by copying MES config:
   ```bash
   cp configs/training_mes.yaml configs/training_nkd.yaml
   # Edit: Change data paths, model output dir
   ```

3. Train NKD model:
   ```bash
   python -m ml_intraday_v3.cli build-train --config configs/training_nkd.yaml
   ```

4. Run both models in parallel during live trading:
   - NKD model active 18:00-03:00 CT
   - MES model active 08:30-15:00 CT

### Option B: Unified System (Better, Slower)

**Build proper multi-market orchestrator**

1. **Portfolio Risk Manager**
   - Track aggregate P&L across both markets
   - Enforce Topstep daily loss limit globally
   - Dynamic position sizing based on correlation

2. **Timezone-Aware Features**
   - Hour-of-day relative to market's local open
   - Session transition features

3. **24/7 Orchestrator**
   - Auto-switch between models based on time
   - Handle session transitions
   - Unified logging

---

## Phase 4: Backtest Multi-Market (1 DAY)

### Simple Backtest Approach

```python
# Pseudo-code for multi-market backtest
portfolio = Portfolio(initial_capital=50_000)

for date in date_range(start, end):
    # Asian session (NKD)
    if is_nkd_rth(date):
        nkd_signal = nkd_model.predict(date)
        if portfolio.can_trade(nkd_signal):
            portfolio.execute(nkd_signal, market='NKD')

    # US session (MES)
    if is_mes_rth(date):
        mes_signal = mes_model.predict(date)
        if portfolio.can_trade(mes_signal):
            portfolio.execute(mes_signal, market='MES')

    # Update portfolio
    portfolio.update_pnl(date)

    # Check Topstep limits
    if portfolio.daily_loss > 2000:
        portfolio.stop_trading_for_day()

# Calculate metrics
metrics = portfolio.calculate_metrics()
print(f"Sharpe: {metrics['sharpe']:.2f}")
print(f"Max DD: {metrics['max_drawdown']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
```

**Compare**:
- MES-only strategy vs. MES+NKD strategy
- Expected: Higher Sharpe, lower drawdown, faster to profit

---

## Phase 5: Live Trading Setup (WHEN READY)

### Topstep Live Data Integration

```python
# Your live runner needs to:
# 1. Connect to Topstep data feed (already working for MES)
# 2. Load NKD model in addition to MES model
# 3. Track current time in CT
# 4. Activate appropriate model based on time

from datetime import datetime
import pytz

class MultiMarketLiveRunner:
    def __init__(self):
        self.mes_model = load_model("models/mes_production")
        self.nkd_model = load_model("models/nkd_production")
        self.portfolio_risk = PortfolioRiskManager()

    def run(self):
        while True:
            current_time = datetime.now(pytz.timezone('America/Chicago'))
            hour = current_time.hour

            # Asian session (NKD)
            if 18 <= hour or hour < 3:
                if self.portfolio_risk.can_trade('NKD'):
                    self.trade_market('NKD', self.nkd_model)

            # US session (MES)
            elif 8 <= hour < 15:
                if self.portfolio_risk.can_trade('MES'):
                    self.trade_market('MES', self.mes_model)

            time.sleep(60)  # Check every minute
```

---

## Quick Win Timeline

**Week 1: Validation**
- Day 1: Download data, verify quality ✓
- Day 2: Calculate correlations, confirm diversification ✓
- Day 3: Backtest MES-only (baseline) ✓

**Week 2: Build**
- Day 4-5: Train NKD model ✓
- Day 6: Simple multi-market backtest ✓
- Day 7: Compare results, decide GO/NO-GO ✓

**Week 3: Live**
- Day 8-9: Integrate NKD into live runner ✓
- Day 10: Paper trade both markets ✓
- Day 11-14: Start Topstep Combine with multi-market ✓

---

## Key Risk Controls

### Portfolio-Level Limits (Critical!)

```yaml
risk_limits:
  max_daily_loss: $2000          # Topstep limit (TOTAL across both)
  max_positions: 2                # Never both at once if highly correlated
  max_exposure: 0.02 * account   # 2% total risk

  rules:
    - "If daily_loss > $1500, stop all trading"
    - "If correlation > 0.8, only trade one market"
    - "If losing in both markets, reduce size by 50%"
```

### Why This Matters
Topstep doesn't care if you lose $1000 in MES and $1000 in NKD separately. They see **$2000 total loss** = rule violation.

Must track aggregate P&L in real-time.

---

## Expected Benefits

### More Trading Opportunities
- **MES-only**: ~6.5 hours/day = maybe 2-3 quality setups
- **MES+NKD**: ~16 hours/day = maybe 5-7 quality setups
- **2-3x more chances** to hit profit target

### Better Risk-Adjusted Returns
- Lower correlation = smoother equity curve
- When US chops, Asia might trend
- Diversification without false diversification (unlike ES+NQ)

### Faster Combine Pass
- More setups = faster to $3000 profit target
- Better consistency (less dependent on single session)
- Could pass in 2-3 weeks vs. 4-6 weeks

---

## Free Resources Used

✅ **Yahoo Finance**: Historical futures data (MES=F, NKD=F)
✅ **Topstep**: Live data feeds for both markets
✅ **Your existing ml_intraday_v3 pipeline**: Just duplicate for NKD
✅ **No additional data costs**

---

## Next Action Items

### TODAY:
1. Run `download_free_futures.py` to get historical data
2. Verify data quality (check for gaps)
3. Calculate MES-NKD correlation to confirm diversification

### THIS WEEK:
4. Create `configs/training_nkd.yaml` (copy from MES config)
5. Train NKD model on historical data
6. Simple backtest: MES-only vs. MES+NKD

### NEXT WEEK:
7. Integrate NKD into live_runner.py
8. Paper trade for 2-3 days
9. Start Topstep Combine with multi-market

---

## Questions?

**Q: Can I use intraday data instead of daily?**
A: Yes! Yahoo Finance offers 1h, 5m intervals. Just change `--interval 5m` in the download script. BUT: More data = slower training. Start with daily to validate approach.

**Q: What if NKD model underperforms?**
A: Trade MES-only. No harm done. You're just expanding opportunity, not replacing what works.

**Q: How do I handle overnight risk?**
A: NKD trades during Asian hours (your evening/night). Use same stops as MES. Set alerts for large moves.

**Q: Commission costs with more markets?**
A: Topstep pricing is per-contract, not per-market. MES and NKD likely same commission rate. Model this in backtest.

---

**Created**: 2026-01-23
**Status**: Ready to implement
**Estimated Time to Production**: 2-3 weeks
