# Standalone ML Candidate Report

- Generated: `2026-04-17T00:27:10.637499Z`
- Overall pass: `False`
- Passing windows: `2/5`
- Pass ratio: `0.40`
- Overall total PnL: `$3977.33`

## Overall Direction

- Long trades: `107`
- Short trades: `33`
- Long share: `0.7642857142857142`
- Short share: `0.2357142857142857`

## Overall Risk

- Worst day: `$-1216.73`
- Best day: `$2085.70`
- Daily Sharpe: `2.1060`
- Longest loss streak: `6`
- Recovery factor: `1.3090`

## Overall Failures

- `all_windows` actual=`2` expected=`5 passing windows`
- `pass_ratio` actual=`0.4` expected=`>= 1.0`
- `overall_long_share` actual=`0.7642857142857142` expected=`<= 0.75`
- `overall_short_share` actual=`0.2357142857142857` expected=`>= 0.25`

## Windows

### dec_2025_bull

- Passed: `False`
- Train events: `1679`
- Test events: `162`
- Test AUC: `0.5903`
- Test accuracy: `0.6111`
- Total PnL: `$548.54`
- Profit factor: `1.2405`
- Max drawdown: `$1216.73`
- Rank IC vs ret_net: `0.2229`
- ECE: `0.0870`
- Decile spread: `6.6902`
- Long trades: `16`
- Short trades: `6`
- Worst day: `$-1216.73`
- Longest loss streak: `4`
- Worst short bucket: `2025-12` / `high_vol_uptrend` avg_ret_net=`-5.7383` rank_ic=`0.4780`
- Failures:
  - `max_drawdown_usd` actual=`1216.7330808967672` expected=`<= 1200.0`

### jan_2026_regime_shift

- Passed: `True`
- Train events: `1590`
- Test events: `238`
- Test AUC: `0.6102`
- Test accuracy: `0.5882`
- Total PnL: `$590.29`
- Profit factor: `1.1861`
- Max drawdown: `$1137.83`
- Rank IC vs ret_net: `0.1563`
- ECE: `0.0782`
- Decile spread: `12.3791`
- Long trades: `23`
- Short trades: `18`
- Worst day: `$-729.66`
- Longest loss streak: `6`
- Worst short bucket: `2026-01` / `med_vol_uptrend` avg_ret_net=`-11.0304` rank_ic=`0.6000`

### feb_2026_followthrough

- Passed: `False`
- Train events: `1562`
- Test events: `253`
- Test AUC: `0.5850`
- Test accuracy: `0.5652`
- Total PnL: `$703.13`
- Profit factor: `1.1716`
- Max drawdown: `$1355.80`
- Rank IC vs ret_net: `0.1556`
- ECE: `0.0586`
- Decile spread: `22.1227`
- Long trades: `34`
- Short trades: `5`
- Worst day: `$-569.55`
- Longest loss streak: `5`
- Worst short bucket: `2026-02` / `high_vol_downtrend` avg_ret_net=`-6.9028` rank_ic=`-0.1000`
- Failures:
  - `max_drawdown_usd` actual=`1355.8002855017548` expected=`<= 1200.0`
  - `long_share` actual=`0.8717948717948718` expected=`<= 0.8`
  - `short_share` actual=`0.1282051282051282` expected=`>= 0.2`

### mar_2026_extension

- Passed: `False`
- Train events: `1609`
- Test events: `248`
- Test AUC: `0.4908`
- Test accuracy: `0.5363`
- Total PnL: `$-992.14`
- Profit factor: `0.7912`
- Max drawdown: `$1578.33`
- Rank IC vs ret_net: `0.0521`
- ECE: `0.1197`
- Decile spread: `-0.8178`
- Long trades: `26`
- Short trades: `0`
- Worst day: `$-894.98`
- Longest loss streak: `5`
- Worst short bucket: `2026-03` / `high_vol_downtrend` avg_ret_net=`-35.6447` rank_ic=`-0.0857`
- Failures:
  - `test_auc` actual=`0.4908108108108108` expected=`>= 0.52`
  - `total_pnl_usd` actual=`-992.1367562928816` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-38.159106011264676` expected=`>= 3.0`
  - `profit_factor` actual=`0.7912110611586737` expected=`>= 1.05`
  - `win_rate` actual=`0.4230769230769231` expected=`>= 0.45`
  - `max_drawdown_usd` actual=`1578.3321138938772` expected=`<= 1200.0`
  - `long_share` actual=`1.0` expected=`<= 0.8`
  - `short_share` actual=`0.0` expected=`>= 0.2`

### apr_2026_partial

- Passed: `True`
- Train events: `1575`
- Test events: `64`
- Test AUC: `0.6266`
- Test accuracy: `0.5781`
- Total PnL: `$3127.51`
- Profit factor: `6.2222`
- Max drawdown: `$209.91`
- Rank IC vs ret_net: `0.1830`
- ECE: `0.1359`
- Decile spread: `-2.0538`
- Long trades: `8`
- Short trades: `4`
- Worst day: `$268.77`
- Longest loss streak: `1`
- Worst short bucket: `2026-04` / `med_vol_downtrend` avg_ret_net=`-21.7237` rank_ic=`-0.6000`

