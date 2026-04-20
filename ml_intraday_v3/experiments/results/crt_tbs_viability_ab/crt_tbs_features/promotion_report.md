# Standalone ML Candidate Report

- Generated: `2026-04-17T00:27:19.556396Z`
- Overall pass: `False`
- Passing windows: `3/5`
- Pass ratio: `0.60`
- Overall total PnL: `$6713.68`

## Overall Direction

- Long trades: `109`
- Short trades: `33`
- Long share: `0.7676056338028169`
- Short share: `0.2323943661971831`

## Overall Risk

- Worst day: `$-983.81`
- Best day: `$1862.08`
- Daily Sharpe: `3.8153`
- Longest loss streak: `5`
- Recovery factor: `1.8996`

## Overall Failures

- `all_windows` actual=`3` expected=`5 passing windows`
- `pass_ratio` actual=`0.6` expected=`>= 1.0`
- `overall_long_share` actual=`0.7676056338028169` expected=`<= 0.75`
- `overall_short_share` actual=`0.2323943661971831` expected=`>= 0.25`

## Windows

### dec_2025_bull

- Passed: `True`
- Train events: `1679`
- Test events: `162`
- Test AUC: `0.5862`
- Test accuracy: `0.5926`
- Total PnL: `$1869.14`
- Profit factor: `2.0565`
- Max drawdown: `$651.75`
- Rank IC vs ret_net: `0.2145`
- ECE: `0.0743`
- Decile spread: `10.9341`
- Long trades: `19`
- Short trades: `5`
- Worst day: `$-181.49`
- Longest loss streak: `3`
- Worst short bucket: `2025-12` / `high_vol_uptrend` avg_ret_net=`-5.7383` rank_ic=`0.4066`

### jan_2026_regime_shift

- Passed: `True`
- Train events: `1590`
- Test events: `238`
- Test AUC: `0.6110`
- Test accuracy: `0.5924`
- Total PnL: `$1172.38`
- Profit factor: `1.4880`
- Max drawdown: `$729.84`
- Rank IC vs ret_net: `0.1537`
- ECE: `0.0959`
- Decile spread: `16.7359`
- Long trades: `19`
- Short trades: `16`
- Worst day: `$-504.51`
- Longest loss streak: `4`
- Worst short bucket: `2026-01` / `med_vol_uptrend` avg_ret_net=`-11.0304` rank_ic=`0.6167`

### feb_2026_followthrough

- Passed: `False`
- Train events: `1562`
- Test events: `253`
- Test AUC: `0.5988`
- Test accuracy: `0.5810`
- Total PnL: `$605.36`
- Profit factor: `1.1449`
- Max drawdown: `$1252.08`
- Rank IC vs ret_net: `0.1922`
- ECE: `0.0573`
- Decile spread: `14.3967`
- Long trades: `33`
- Short trades: `7`
- Worst day: `$-569.55`
- Longest loss streak: `5`
- Worst short bucket: `2026-02` / `high_vol_downtrend` avg_ret_net=`-6.9028` rank_ic=`0.1333`
- Failures:
  - `max_drawdown_usd` actual=`1252.0802627984158` expected=`<= 1200.0`
  - `long_share` actual=`0.825` expected=`<= 0.8`
  - `short_share` actual=`0.175` expected=`>= 0.2`

### mar_2026_extension

- Passed: `False`
- Train events: `1609`
- Test events: `248`
- Test AUC: `0.5030`
- Test accuracy: `0.5403`
- Total PnL: `$226.63`
- Profit factor: `1.0494`
- Max drawdown: `$1350.70`
- Rank IC vs ret_net: `0.0573`
- ECE: `0.0771`
- Decile spread: `1.8623`
- Long trades: `29`
- Short trades: `1`
- Worst day: `$-983.81`
- Longest loss streak: `5`
- Worst short bucket: `2026-03` / `high_vol_downtrend` avg_ret_net=`-35.6447` rank_ic=`-0.2571`
- Failures:
  - `test_auc` actual=`0.5030405405405405` expected=`>= 0.52`
  - `profit_factor` actual=`1.049395509287439` expected=`>= 1.05`
  - `max_drawdown_usd` actual=`1350.6992218131272` expected=`<= 1200.0`
  - `long_share` actual=`0.9666666666666667` expected=`<= 0.8`
  - `short_share` actual=`0.03333333333333333` expected=`>= 0.2`

### apr_2026_partial

- Passed: `True`
- Train events: `1575`
- Test events: `64`
- Test AUC: `0.6006`
- Test accuracy: `0.5781`
- Total PnL: `$2840.16`
- Profit factor: `4.6608`
- Max drawdown: `$392.80`
- Rank IC vs ret_net: `0.1808`
- ECE: `0.1713`
- Decile spread: `11.8574`
- Long trades: `9`
- Short trades: `4`
- Worst day: `$-96.56`
- Longest loss streak: `2`
- Worst short bucket: `2026-04` / `med_vol_downtrend` avg_ret_net=`-21.7237` rank_ic=`0.0000`

