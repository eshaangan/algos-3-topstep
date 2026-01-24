# Multi-Market Trading System Architecture

**Goal**: Trade across US, European, and Asian sessions with a single unified Topstep account, leveraging true geographic diversification for 24-hour coverage.

## Executive Summary

### Why Multi-Market?
- **True Diversification**: Different macro drivers (Fed vs ECB vs BOJ)
- **24/7 Opportunity**: Always trading during peak liquidity hours for THAT market
- **Smoother Equity**: Spread P&L across time zones reduces concentrated risk
- **Topstep-Friendly**: Better consistency metrics, less "sit and wait" during chop

### Critical Decision: Data & Instruments

**Alpha Vantage Limitation**: Provides STOCK data, not FUTURES data
- Topstep trades **futures** (MES, FDAX, NKD)
- Alpha Vantage provides **stocks** (IBM, TSCO.LON, Toyota)

**Recommended Path**: CME-listed international futures
- All traded on same exchange (CME)
- Consistent data infrastructure
- Direct Topstep compatibility

## Market Selection

### Primary Markets (24-Hour Coverage)

| Market | Symbol | Exchange | Session (CT) | Correlation w/ MES |
|--------|--------|----------|--------------|-------------------|
| **S&P Micro** | MES | CME | 08:30-15:00 | 1.00 (baseline) |
| **Nikkei 225** | NKD | CME | 18:00-03:00 | ~0.65 |
| **DAX** | FDAX | EUREX | 02:00-11:00 | ~0.75 |

**Alternative European**: Euro Stoxx 50 (FESX) if DAX not available

### Why These Three?
1. **Lower correlation** than ES/NQ/YM (false diversification)
2. **Different trading hours** for true 24/7 coverage
3. **CME-listed** (NKD, MES) or major exchange (FDAX) with good data
4. **Liquid enough** for algorithmic execution

## System Architecture

### Directory Structure
```
ml_intraday_v3/
├── configs/
│   ├── training_mes.yaml          # US session config
│   ├── training_nkd.yaml          # Asian session config
│   ├── training_fdax.yaml         # European session config
│   └── portfolio_risk.yaml        # Unified risk limits
├── models/
│   ├── mes_production/            # Trained MES model
│   ├── nkd_production/            # Trained NKD model
│   └── fdax_production/           # Trained FDAX model
├── data/
│   ├── mes_features/              # US market features
│   ├── nkd_features/              # Asian market features
│   └── fdax_features/             # European market features
├── live_trading/
│   ├── market_orchestrator.py     # 24/7 session manager
│   ├── portfolio_risk_manager.py  # Aggregate risk tracking
│   └── timezone_monitor.py        # Market hours detection
└── backtesting/
    └── multi_market_backtest.py   # Portfolio-level backtest
```

### Core Components

#### 1. Market-Specific Configurations
Each market gets its own training config with:
- **Timezone**: Local market hours in UTC
- **Feature Engineering**: Market-specific tick size, ATR scaling
- **Position Sizing**: Per-market Kelly fraction (constrained by portfolio limit)
- **Session Features**: Hour-of-day relative to THAT market's open

#### 2. Portfolio Risk Manager (CRITICAL)
Topstep limits apply to TOTAL P&L, not per-market:
```yaml
portfolio_risk:
  max_total_exposure: 0.02          # 2% account risk across ALL positions
  max_concurrent_positions: 2        # Never trade all 3 simultaneously
  max_daily_loss: $2000             # Topstep limit
  trailing_drawdown: $2500          # Topstep limit

  correlation_matrix:
    update_frequency: "daily"
    lookback_window: 60             # days

  position_sizing:
    method: "hierarchical_risk_parity"  # Accounts for correlations
    # Alternative: "portfolio_kelly"

  rules:
    - if: "correlation(mes, nkd) > 0.8"
      then: "reduce_position_size * 0.5"
    - if: "daily_loss > $1500"
      then: "stop_all_trading_for_day"
```

#### 3. Timezone-Aware Feature Engineering
Features must be local to the market:
```python
# BAD: Uses US Eastern time for all markets
hour_of_day = timestamp.hour

# GOOD: Uses local market time
market_tz = pytz.timezone(market.timezone)
local_time = timestamp.astimezone(market_tz)
hour_since_open = (local_time - market.open_time).total_seconds() / 3600

# CROSS-MARKET: How did Asian session close?
asian_close_pnl = get_prior_session_pnl("NKD")  # Use during US open
```

#### 4. 24/7 Live Trading Orchestrator
```python
class MarketOrchestrator:
    """Manages trading across multiple sessions"""

    def run_24_7(self):
        while True:
            current_time = datetime.now(pytz.utc)

            # Determine active markets
            active_markets = self.get_active_markets(current_time)

            # Check portfolio risk FIRST
            if self.portfolio_risk.daily_limit_reached():
                self.shutdown_all_markets()
                sleep_until_next_day()
                continue

            # Run active market models
            for market in active_markets:
                if self.portfolio_risk.can_trade(market):
                    self.run_market_model(market)

            time.sleep(60)  # Check every minute
```

#### 5. Separate Model Training Per Market
Each market has different microstructure:
```bash
# Train MES model (US)
python -m ml_intraday_v3.cli build-train --config configs/training_mes.yaml

# Train NKD model (Asia)
python -m ml_intraday_v3.cli build-train --config configs/training_nkd.yaml

# Train FDAX model (Europe)
python -m ml_intraday_v3.cli build-train --config configs/training_fdax.yaml
```

**Why separate?**
- Different volatility regimes
- Different spreads (DAX wider than MES)
- Different optimal stop loss / take profit
- Different feature distributions

#### 6. Multi-Market Backtest Framework
```python
class MultiMarketBacktest:
    """Simulates 24-hour trading across markets"""

    def run(self, start_date, end_date):
        portfolio = Portfolio(initial_capital=50_000)

        # Iterate through time chronologically
        for timestamp in date_range(start_date, end_date):
            # Determine which markets are open
            active_markets = get_open_markets(timestamp)

            for market in active_markets:
                # Generate signal from market-specific model
                signal = self.models[market].predict(timestamp)

                # Check portfolio-level risk
                if portfolio.can_enter_trade(market, signal):
                    portfolio.execute_trade(market, signal)

            # Update portfolio P&L
            portfolio.update(timestamp)

        return portfolio.get_metrics()
```

## Implementation Phases

### Phase 1: Validation (Week 1)
**Objective**: Verify feasibility before building anything

1. **Check Topstep Instrument Support**
   - Does 50k Combine support FDAX, NKD?
   - Or only CME instruments (MES, MNQ, M2K)?

2. **Data Provider Research**
   - Cost for international futures data
   - Quality/latency of feeds
   - Historical depth for training

3. **Correlation Analysis**
   - Pull historical data for MES, NKD, FDAX
   - Calculate rolling correlations
   - Verify diversification benefit exists

**Deliverable**: GO/NO-GO decision document

### Phase 2: Single International Market (Weeks 2-4)
**Objective**: Add ONE international market to validate pipeline

1. **Choose**: NKD (Asian session) or FDAX (European)
2. **Build**:
   - Market-specific config
   - Feature engineering with timezone awareness
   - Model training pipeline
   - Basic portfolio risk manager
3. **Backtest**:
   - Compare MES-only vs. MES+International
   - Measure Sharpe improvement, drawdown reduction

**Deliverable**: Working 2-market system with backtest proof

### Phase 3: Full Multi-Market System (Weeks 5-8)
**Objective**: Complete 24/7 trading system

1. **Add Third Market**: Complete US/Europe/Asia coverage
2. **Unified Risk Manager**: Portfolio-level position sizing
3. **Live Trading Infrastructure**: 24/7 orchestrator
4. **Cloud Deployment**: Auto-restart, monitoring, alerts

**Deliverable**: Production-ready multi-market system

### Phase 4: Optimization & Live Testing (Ongoing)
1. Walk-forward validation per market
2. Cross-market signal combination experiments
3. Adaptive correlation monitoring
4. Live paper trading before Topstep

## Key Technical Challenges

### 1. Data Costs
- **Problem**: CME futures data expensive ($500+/month per market)
- **Solution**: Start with delayed data for development, upgrade for live

### 2. Correlation Regime Changes
- **Problem**: Markets uncorrelated normally, spike to 1.0 during crashes
- **Solution**: Dynamic position sizing based on realized correlation

### 3. 24/7 Operational Complexity
- **Problem**: Must handle 3 AM failures, data feed drops
- **Solution**: Robust error handling, alerts, auto-restart

### 4. Leakage Safety Across Markets
- **Problem**: Cross-market features could introduce look-ahead bias
- **Solution**: Strict temporal alignment, separate purging per market

### 5. Commission Accumulation
- **Problem**: More markets = more trades = more commissions
- **Solution**: Model commission impact in backtest, ensure positive expectancy

## Risk Management Framework

### Portfolio-Level Constraints
```python
class PortfolioRiskManager:
    def __init__(self):
        self.max_daily_loss = 2000  # Topstep limit
        self.max_positions = 2      # Max concurrent
        self.correlation_threshold = 0.7  # Don't trade if corr > this

    def can_enter_trade(self, market: str, signal: Signal) -> bool:
        # Check 1: Daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            return False

        # Check 2: Max concurrent positions
        if len(self.active_positions) >= self.max_positions:
            return False

        # Check 3: Correlation with existing positions
        for position in self.active_positions:
            corr = self.get_correlation(market, position.market)
            if corr > self.correlation_threshold:
                return False  # Too correlated, skip

        # Check 4: Aggregate exposure
        proposed_risk = self.calculate_risk(market, signal)
        if self.total_risk + proposed_risk > 0.02 * self.account_size:
            return False

        return True
```

### Per-Market Position Sizing
```python
# Each market has its own Kelly fraction
market_kelly = {
    "MES": 0.018,   # 1.8% per trade
    "NKD": 0.015,   # 1.5% per trade
    "FDAX": 0.012,  # 1.2% per trade (wider spreads)
}

# But constrained by portfolio limit
def get_position_size(market: str) -> float:
    base_size = market_kelly[market]

    # Adjust for existing positions
    correlation_adjustment = calculate_correlation_penalty()

    return base_size * correlation_adjustment
```

## Success Metrics

### Portfolio-Level Goals
- **Sharpe Ratio**: > 2.0 (vs. 1.5 for MES-only)
- **Max Drawdown**: < 3% (vs. 5% for MES-only)
- **Consistency**: 70%+ winning days
- **Topstep Compliance**: Zero rule violations

### Per-Market Requirements
Each market must be individually profitable:
- Positive expected value after commissions
- Sharpe > 1.5 standalone
- Passes walk-forward validation

## Alternative: Alpha Vantage Research Path

If Topstep doesn't support international futures, use Alpha Vantage for RESEARCH:

1. **Study International Indices**:
   - Fetch FTSE 100 stocks, DAX stocks via Alpha Vantage
   - Analyze correlation patterns, volatility regimes
   - Understand session dynamics

2. **Adapt to US-Listed Proxies**:
   - Trade SPY (S&P 500 ETF) instead of MES
   - Trade EWJ (Japan ETF) instead of NKD
   - Trade EWG (Germany ETF) instead of FDAX

3. **Limitations**:
   - ETFs have lower leverage than futures
   - May not be Topstep-compatible (check!)
   - Different microstructure than futures

## Next Steps

1. **[IMMEDIATE]** Check Topstep documentation for supported instruments
2. **[IMMEDIATE]** Research CME data costs for NKD, FDAX
3. **[WEEK 1]** Download sample data, calculate correlations
4. **[WEEK 1]** Create GO/NO-GO decision document
5. **[IF GO]** Start Phase 2 implementation

## Resources & References

- **Topstep Rules**: https://topstep.com/combine-rules/
- **CME Data**: https://www.cmegroup.com/market-data.html
- **Alpha Vantage API**: https://www.alphavantage.co/documentation/
- **Project Blueprint**: `ml_intraday_v3/ML_PIPELINE_V3_BLUEPRINT.md`

---

**Author**: Multi-Market Trading System Design
**Date**: 2026-01-23
**Status**: DESIGN PHASE - Awaiting Topstep Instrument Verification
