# Standalone ML Candidate Report

- Generated: `2026-04-13T04:10:40.822269Z`
- Overall pass: `False`
- Passing windows: `3/5`
- Pass ratio: `0.60`
- Overall total PnL: `$3012.31`

## Overall Direction

- Long trades: `90`
- Short trades: `55`
- Long share: `0.6206896551724138`
- Short share: `0.3793103448275862`

## Overall Risk

- Worst day: `$-554.94`
- Best day: `$931.04`
- Daily Sharpe: `3.2742`
- Longest loss streak: `7`
- Recovery factor: `1.7450`

## Overall Failures

- `all_windows` actual=`3` expected=`5 passing windows`
- `pass_ratio` actual=`0.6` expected=`>= 1.0`

## Windows

### dec_2025_bull

- Passed: `True`
- Train events: `1679`
- Test events: `162`
- Test AUC: `0.5903`
- Test accuracy: `0.6111`
- Total PnL: `$599.87`
- Profit factor: `1.5952`
- Max drawdown: `$554.94`
- Rank IC vs ret_net: `0.2229`
- ECE: `0.0870`
- Decile spread: `6.6902`
- Long trades: `10`
- Short trades: `11`
- Worst day: `$-554.94`
- Longest loss streak: `3`
- Worst short bucket: `2025-12` / `high_vol_uptrend` avg_ret_net=`-5.7383` rank_ic=`0.4780`

### jan_2026_regime_shift

- Passed: `True`
- Train events: `1590`
- Test events: `238`
- Test AUC: `0.6102`
- Test accuracy: `0.5882`
- Total PnL: `$1002.56`
- Profit factor: `1.8994`
- Max drawdown: `$231.74`
- Rank IC vs ret_net: `0.1563`
- ECE: `0.0782`
- Decile spread: `12.3791`
- Long trades: `14`
- Short trades: `25`
- Worst day: `$-99.55`
- Longest loss streak: `3`
- Worst short bucket: `2026-01` / `med_vol_uptrend` avg_ret_net=`-11.0304` rank_ic=`0.6000`

### feb_2026_followthrough

- Passed: `False`
- Train events: `1562`
- Test events: `253`
- Test AUC: `0.5850`
- Test accuracy: `0.5652`
- Total PnL: `$118.55`
- Profit factor: `1.0556`
- Max drawdown: `$707.24`
- Rank IC vs ret_net: `0.1556`
- ECE: `0.0586`
- Decile spread: `22.1227`
- Long trades: `30`
- Short trades: `10`
- Worst day: `$-284.77`
- Longest loss streak: `5`
- Worst short bucket: `2026-02` / `high_vol_downtrend` avg_ret_net=`-6.9028` rank_ic=`-0.1000`
- Failures:
  - `avg_trade_usd` actual=`2.9636287781489754` expected=`>= 3.0`

### mar_2026_extension

- Passed: `False`
- Train events: `1609`
- Test events: `248`
- Test AUC: `0.4908`
- Test accuracy: `0.5363`
- Total PnL: `$-45.21`
- Profit factor: `0.9825`
- Max drawdown: `$966.39`
- Rank IC vs ret_net: `0.0521`
- ECE: `0.1197`
- Decile spread: `-0.8178`
- Long trades: `30`
- Short trades: `3`
- Worst day: `$-491.90`
- Longest loss streak: `7`
- Worst short bucket: `2026-03` / `high_vol_downtrend` avg_ret_net=`-35.6447` rank_ic=`-0.0857`
- Failures:
  - `test_auc` actual=`0.4908108108108108` expected=`>= 0.52`
  - `total_pnl_usd` actual=`-45.21467515435617` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-1.3701416713441263` expected=`>= 3.0`
  - `profit_factor` actual=`0.9824881411883837` expected=`>= 1.05`
  - `long_share` actual=`0.9090909090909091` expected=`<= 0.8`
  - `short_share` actual=`0.09090909090909091` expected=`>= 0.2`

### apr_2026_partial

- Passed: `True`
- Train events: `1575`
- Test events: `64`
- Test AUC: `0.6266`
- Test accuracy: `0.5781`
- Total PnL: `$1336.54`
- Profit factor: `4.3745`
- Max drawdown: `$196.40`
- Rank IC vs ret_net: `0.1830`
- ECE: `0.1359`
- Decile spread: `-2.0538`
- Long trades: `6`
- Short trades: `6`
- Worst day: `$37.75`
- Longest loss streak: `2`
- Worst short bucket: `2026-04` / `med_vol_downtrend` avg_ret_net=`-21.7237` rank_ic=`-0.6000`

