# Standalone ML Candidate Report

- Generated: `2026-04-12T23:58:02.195634Z`
- Overall pass: `False`
- Passing windows: `2/5`
- Pass ratio: `0.40`
- Overall total PnL: `$-1202.82`

## Overall Direction

- Long trades: `75`
- Short trades: `78`
- Long share: `0.49019607843137253`
- Short share: `0.5098039215686274`

## Overall Risk

- Worst day: `$-487.74`
- Best day: `$887.06`
- Daily Sharpe: `-1.3026`
- Longest loss streak: `8`
- Recovery factor: `-0.5331`

## Overall Failures

- `all_windows` actual=`2` expected=`5 passing windows`
- `pass_ratio` actual=`0.4` expected=`>= 1.0`
- `overall_total_pnl_usd` actual=`-1202.8172898935904` expected=`>= 500.0`

## Windows

### dec_2025_bull

- Passed: `True`
- Train events: `1152`
- Test events: `171`
- Test AUC: `0.6127`
- Test accuracy: `0.6374`
- Total PnL: `$160.88`
- Profit factor: `1.1191`
- Max drawdown: `$533.16`
- Rank IC vs ret_net: `0.2592`
- ECE: `0.0578`
- Decile spread: `8.5767`
- Long trades: `8`
- Short trades: `15`
- Worst day: `$-308.28`
- Longest loss streak: `4`
- Worst short bucket: `2025-12` / `med_vol_downtrend` avg_ret_net=`-5.9210` rank_ic=`0.5874`

### jan_2026_regime_shift

- Passed: `False`
- Train events: `1040`
- Test events: `261`
- Test AUC: `0.5582`
- Test accuracy: `0.5632`
- Total PnL: `$13.78`
- Profit factor: `1.0069`
- Max drawdown: `$712.22`
- Rank IC vs ret_net: `0.0801`
- ECE: `0.0935`
- Decile spread: `1.9719`
- Long trades: `5`
- Short trades: `43`
- Worst day: `$-281.94`
- Longest loss streak: `3`
- Worst short bucket: `2026-01` / `low_vol_uptrend` avg_ret_net=`-15.2597` rank_ic=`-0.8000`
- Failures:
  - `avg_trade_usd` actual=`0.287124014736784` expected=`>= 3.0`
  - `profit_factor` actual=`1.0069475838229576` expected=`>= 1.05`
  - `long_share` actual=`0.10416666666666667` expected=`>= 0.2`
  - `short_share` actual=`0.8958333333333334` expected=`<= 0.8`

### feb_2026_followthrough

- Passed: `False`
- Train events: `1010`
- Test events: `289`
- Test AUC: `0.5708`
- Test accuracy: `0.5571`
- Total PnL: `$-497.45`
- Profit factor: `0.8144`
- Max drawdown: `$978.33`
- Rank IC vs ret_net: `0.1535`
- ECE: `0.0730`
- Decile spread: `22.4152`
- Long trades: `34`
- Short trades: `11`
- Worst day: `$-283.79`
- Longest loss streak: `5`
- Worst short bucket: `2026-02` / `low_vol_downtrend` avg_ret_net=`-4.5244` rank_ic=`0.2219`
- Failures:
  - `total_pnl_usd` actual=`-497.4546923482386` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-11.054548718849746` expected=`>= 3.0`
  - `profit_factor` actual=`0.8144374223154764` expected=`>= 1.05`

### mar_2026_extension

- Passed: `False`
- Train events: `1030`
- Test events: `260`
- Test AUC: `0.4830`
- Test accuracy: `0.4962`
- Total PnL: `$-1787.25`
- Profit factor: `0.3245`
- Max drawdown: `$1700.90`
- Rank IC vs ret_net: `0.0365`
- ECE: `0.1155`
- Decile spread: `-6.7690`
- Long trades: `17`
- Short trades: `2`
- Worst day: `$-487.74`
- Longest loss streak: `7`
- Worst short bucket: `2026-03` / `high_vol_downtrend` avg_ret_net=`-36.7336` rank_ic=`0.4048`
- Failures:
  - `test_auc` actual=`0.4830498027613412` expected=`>= 0.52`
  - `test_accuracy` actual=`0.49615384615384617` expected=`>= 0.5`
  - `total_pnl_usd` actual=`-1787.2469131352116` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-94.0656270071164` expected=`>= 3.0`
  - `profit_factor` actual=`0.32448770951653394` expected=`>= 1.05`
  - `win_rate` actual=`0.2631578947368421` expected=`>= 0.45`
  - `max_drawdown_usd` actual=`1700.904936342471` expected=`<= 1200.0`
  - `long_share` actual=`0.8947368421052632` expected=`<= 0.8`
  - `short_share` actual=`0.10526315789473684` expected=`>= 0.2`

### apr_2026_partial

- Passed: `True`
- Train events: `1025`
- Test events: `84`
- Test AUC: `0.5394`
- Test accuracy: `0.5595`
- Total PnL: `$907.23`
- Profit factor: `1.7588`
- Max drawdown: `$279.02`
- Rank IC vs ret_net: `0.0093`
- ECE: `0.1100`
- Decile spread: `-9.4965`
- Long trades: `11`
- Short trades: `7`
- Worst day: `$-256.33`
- Longest loss streak: `2`
- Worst short bucket: `2026-04` / `med_vol_sideways` avg_ret_net=`-18.4090` rank_ic=`0.5000`

