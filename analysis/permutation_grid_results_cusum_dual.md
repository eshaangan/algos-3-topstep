# CUSUM Dual Label Grid Results

| scenario | total_pnl_usd | max_drawdown_usd | win_rate | profit_factor | trades | daily_loss_violations | daily_loss_violation_pct | sharpe | cv | mc_pass_rate | backtest_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cusum_dual_long_baseline | 43070.089631 | 2840.977805 | 0.475075 | 1.930162 | 1665 | 9 | 0.098901 | 4.499741 | 2.632652 | 0.697700 | runs/cusum_dual_24h_20260114/bar_size=5m/backtests/purged_kfold_long_baseline |
| cusum_dual_short_baseline | -979.296722 | 2513.576760 | 0.453376 | 0.923927 | 311 | 1 | 0.008621 | -0.555302 | -6.714534 | 0.118100 | runs/cusum_dual_24h_20260114/bar_size=5m/backtests/purged_kfold_short_baseline |
| cusum_dual_long_regularized | 43429.621205 | 2895.957805 | 0.476868 | 1.933629 | 1686 | 10 | 0.109890 | 4.498804 | 2.634599 | 0.704900 | runs/cusum_dual_24h_20260114/bar_size=5m/backtests/purged_kfold_long_regularized |
| cusum_dual_short_regularized | 91.828015 | 2693.128387 | 0.488636 | 1.011388 | 176 | 1 | 0.013333 | 0.071623 | 92.962357 | 0.289300 | runs/cusum_dual_24h_20260114/bar_size=5m/backtests/purged_kfold_short_regularized |
| cusum_dual_dual_baseline | 46969.076918 | 2840.977805 | 0.494208 | 1.954243 | 1813 | 10 | 0.106383 | 4.715429 | 2.388820 | 0.692900 | runs/cusum_dual_24h_20260114/bar_size=5m/backtests/purged_kfold_dual_baseline |
| cusum_dual_dual_regularized | 43580.009274 | 2895.957805 | 0.478982 | 1.936856 | 1689 | 10 | 0.109890 | 4.516327 | 2.624183 | 0.705100 | runs/cusum_dual_24h_20260114/bar_size=5m/backtests/purged_kfold_dual_regularized |

