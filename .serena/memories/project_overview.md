# Project Overview: Topstep ML Trading System

## Primary Goal
Build an algorithmic trading system capable of **passing the Topstep 50k Combine** and maintaining long-term profitability in the funded phase.

## Key Objectives
1. **Pass Topstep Combine**: Meet all Topstep evaluation criteria
2. **Risk-Adjusted Returns**: Optimize for Sharpe/Sortino ratios, not just raw PnL
3. **Strict Risk Management**: Enforce daily loss limits and trailing max drawdown
4. **Consistency**: Avoid outlier trades that violate consistency rules
5. **Leakage-Free ML**: No lookahead in features, proper validation with purging/embargo

## Trading Context
- **Instrument**: MES (Micro E-mini S&P 500 futures)
- **Timeframes**: 1-minute and 5-minute bars (RTH-only)
- **Strategy Type**: Intraday ML-based long/short signals
- **Risk Framework**: Topstep-compliant (daily loss limit, trailing drawdown)

## Topstep Risk Constraints
- **Daily Loss Limit**: Strict enforcement (typically $1,000 for 50k combine)
- **Trailing Max Drawdown**: Capital preservation prioritized ($2,500 for 50k combine)
- **Consistency**: Avoid lucky outlier trades that violate consistency rules
- **Position Limits**: Max contracts and concurrent positions enforced

## Development Philosophy
- **Research-Grade**: Prioritize correctness, leakage safety, and reproducibility over speed
- **Senior Quant Researcher mindset**: All implementations must be academically rigorous
- **Literature-Driven**: Reference papers in `ml_intraday_v3/research papers/` for complex algorithms
- **No Guessing**: Ask user for papers if specific algorithms are mentioned but not available

## Project Maturity
This is a research and live trading system with:
- Full data pipeline (Phase 1 COMPLETE)
- Feature engineering (Phase 2 COMPLETE)
- Triple-barrier labeling with sample weighting
- Purged K-Fold CV and CPCV
- Walk-forward validation
- Live trading capabilities (paper + real)
- Full reproducibility via run manifests