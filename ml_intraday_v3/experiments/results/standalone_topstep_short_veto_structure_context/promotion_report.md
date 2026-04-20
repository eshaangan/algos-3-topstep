# Standalone ML Candidate Report

- Generated: `2026-04-12T23:58:05.317442Z`
- Overall pass: `False`
- Passing windows: `2/5`
- Pass ratio: `0.40`
- Overall total PnL: `$-1313.63`

## Overall Direction

- Long trades: `73`
- Short trades: `84`
- Long share: `0.46496815286624205`
- Short share: `0.535031847133758`

## Overall Risk

- Worst day: `$-566.24`
- Best day: `$931.04`
- Daily Sharpe: `-1.2848`
- Longest loss streak: `7`
- Recovery factor: `-0.4812`

## Overall Failures

- `all_windows` actual=`2` expected=`5 passing windows`
- `pass_ratio` actual=`0.4` expected=`>= 1.0`
- `overall_total_pnl_usd` actual=`-1313.6317951909657` expected=`>= 500.0`

## Windows

### dec_2025_bull

- Passed: `False`
- Train events: `1679`
- Test events: `162`
- Test AUC: `0.5903`
- Test accuracy: `0.6111`
- Total PnL: `$-919.83`
- Profit factor: `0.5448`
- Max drawdown: `$1103.08`
- Rank IC vs ret_net: `0.2229`
- ECE: `0.0870`
- Decile spread: `6.6902`
- Long trades: `4`
- Short trades: `20`
- Worst day: `$-566.24`
- Longest loss streak: `4`
- Worst short bucket: `2025-12` / `high_vol_uptrend` avg_ret_net=`-5.7383` rank_ic=`0.4780`
- Failures:
  - `total_pnl_usd` actual=`-919.8312143898229` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-38.32630059957595` expected=`>= 3.0`
  - `profit_factor` actual=`0.5448353267570922` expected=`>= 1.05`
  - `win_rate` actual=`0.4166666666666667` expected=`>= 0.45`
  - `long_share` actual=`0.16666666666666666` expected=`>= 0.2`
  - `short_share` actual=`0.8333333333333334` expected=`<= 0.8`

### jan_2026_regime_shift

- Passed: `False`
- Train events: `1590`
- Test events: `238`
- Test AUC: `0.6102`
- Test accuracy: `0.5882`
- Total PnL: `$-228.38`
- Profit factor: `0.8950`
- Max drawdown: `$610.43`
- Rank IC vs ret_net: `0.1563`
- ECE: `0.0782`
- Decile spread: `12.3791`
- Long trades: `7`
- Short trades: `38`
- Worst day: `$-306.65`
- Longest loss streak: `3`
- Worst short bucket: `2026-01` / `med_vol_uptrend` avg_ret_net=`-11.0304` rank_ic=`0.6000`
- Failures:
  - `total_pnl_usd` actual=`-228.37554193531645` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-5.075012043007032` expected=`>= 3.0`
  - `profit_factor` actual=`0.8949761693999626` expected=`>= 1.05`
  - `long_share` actual=`0.15555555555555556` expected=`>= 0.2`
  - `short_share` actual=`0.8444444444444444` expected=`<= 0.8`

### feb_2026_followthrough

- Passed: `True`
- Train events: `1562`
- Test events: `253`
- Test AUC: `0.5850`
- Test accuracy: `0.5652`
- Total PnL: `$369.34`
- Profit factor: `1.1726`
- Max drawdown: `$674.68`
- Rank IC vs ret_net: `0.1556`
- ECE: `0.0586`
- Decile spread: `22.1227`
- Long trades: `30`
- Short trades: `12`
- Worst day: `$-336.48`
- Longest loss streak: `4`
- Worst short bucket: `2026-02` / `high_vol_downtrend` avg_ret_net=`-6.9028` rank_ic=`-0.1000`

### mar_2026_extension

- Passed: `False`
- Train events: `1609`
- Test events: `248`
- Test AUC: `0.4908`
- Test accuracy: `0.5363`
- Total PnL: `$-1780.77`
- Profit factor: `0.5310`
- Max drawdown: `$1843.37`
- Rank IC vs ret_net: `0.0521`
- ECE: `0.1197`
- Decile spread: `-0.8178`
- Long trades: `28`
- Short trades: `7`
- Worst day: `$-560.24`
- Longest loss streak: `7`
- Worst short bucket: `2026-03` / `high_vol_downtrend` avg_ret_net=`-35.6447` rank_ic=`-0.0857`
- Failures:
  - `test_auc` actual=`0.4908108108108108` expected=`>= 0.52`
  - `total_pnl_usd` actual=`-1780.7742374613542` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-50.879263927467264` expected=`>= 3.0`
  - `profit_factor` actual=`0.5309569953919723` expected=`>= 1.05`
  - `win_rate` actual=`0.34285714285714286` expected=`>= 0.45`
  - `max_drawdown_usd` actual=`1843.3707652069206` expected=`<= 1200.0`

### apr_2026_partial

- Passed: `True`
- Train events: `1575`
- Test events: `64`
- Test AUC: `0.6266`
- Test accuracy: `0.5781`
- Total PnL: `$1246.01`
- Profit factor: `4.1107`
- Max drawdown: `$196.40`
- Rank IC vs ret_net: `0.1830`
- ECE: `0.1359`
- Decile spread: `-2.0538`
- Long trades: `4`
- Short trades: `7`
- Worst day: `$-56.44`
- Longest loss streak: `2`
- Worst short bucket: `2026-04` / `med_vol_downtrend` avg_ret_net=`-21.7237` rank_ic=`-0.6000`

