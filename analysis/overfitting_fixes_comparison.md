# Overfitting Fixes Comparison - Bidirectional 24H

**Date**: 2026-01-14
**Run**: `runs/bidirectional_24h_20260114/bar_size=5m`
**Backtests**: `backtests/purged_kfold`

## Summary
- Current purged_kfold aggregate still breaches Topstep drawdown and daily loss limits.
- Ensemble bundle created; backtest pending.
- Regularized run completed; still breaches drawdown and daily loss limits with high CV.
- Regularized + CUSUM (no lookahead) run completed; performance degrades with higher CV and drawdown.

## Side-by-Side Metrics (Required Models)

| Metric | Single Model (fold_0) | Ensemble (6-fold avg) | Regularized Model |
| --- | --- | --- | --- |
| Total PnL | -$1,474 | Pending | $124,891 |
| Sharpe (daily) | -2.79 | Pending | 6.08 |
| Max Drawdown | $2,699 | Pending | $4,143 |
| Daily Loss Violations (>$1,000) | 0 | Pending | 9 |
| K-Fold CV | N/A | Pending | 116.8% |
| Win Rate | 47.9% | Pending | 54.1% |
| Profit Factor | 0.84 | Pending | 1.64 |
| Topstep MC Pass Rate | 7.0% | Pending | 80.0% |

## Regularized + CUSUM (No Lookahead) Results
- Run: `runs/regularized_cusum_24h_20260114/bar_size=5m`
- Total PnL: $53,454
- Win Rate: 50.2%
- Profit Factor: 2.06
- K-Fold CV: 230.5%
- Max Drawdown: $5,233
- Daily Loss Violations: 9
- Topstep MC Pass Rate: 36.3%

## Current Aggregate (All Folds, for Context)
- Total PnL: $86,198
- Win Rate: 54.2%
- Profit Factor: 1.58
- Daily Loss Violations: 8
- Max Drawdown: $4,082 (Topstep limit $2,500)
- K-Fold CV: 124% (population std/mean)
- Topstep MC Pass Rate: 75.3% (sequential run fails due to drawdown)

## Equity Curves & Drawdowns
- Aggregate equity curve: `analysis/topstep_50k_equity_curve_bidirectional_24h.png`
- Ensemble and regularized curves: Pending (requires backtests)

## Recommendation (Current State)
- Do **not** trade the current single-model bundle in a Topstep combine.
- Next best step is to backtest the ensemble bundle; if CV drops and drawdown improves, use that for paper trading.
- Regularized run still violates drawdown/daily loss; needs a different mitigation (ensemble + stricter risk).

## Next Steps to Complete Comparison
1. Backtest ensemble bundle on purged_kfold splits and write metrics.
2. Run the regularized training pipeline (`regularized_24h_20260114`) and backtest.
3. Update this report with final metrics, equity curves, and Topstep MC pass rates.
