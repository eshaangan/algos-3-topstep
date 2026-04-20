# Standalone ML Candidate Report

- Generated: `2026-04-13T04:19:15.582352Z`
- Overall pass: `False`
- Passing windows: `4/5`
- Pass ratio: `0.80`
- Overall total PnL: `$1864.77`

## Overall Direction

- Long trades: `91`
- Short trades: `50`
- Long share: `0.6453900709219859`
- Short share: `0.3546099290780142`

## Overall Risk

- Worst day: `$-647.55`
- Best day: `$931.04`
- Daily Sharpe: `1.9516`
- Longest loss streak: `7`
- Recovery factor: `0.8741`

## Overall Failures

- `all_windows` actual=`4` expected=`5 passing windows`
- `pass_ratio` actual=`0.8` expected=`>= 1.0`

## Windows

### dec_2025_bull

- Passed: `True`
- Train events: `1679`
- Test events: `162`
- Test AUC: `0.6116`
- Test accuracy: `0.6235`
- Total PnL: `$552.16`
- Profit factor: `1.5334`
- Max drawdown: `$647.55`
- Rank IC vs ret_net: `0.2542`
- ECE: `0.0903`
- Decile spread: `8.4425`
- Long trades: `12`
- Short trades: `9`
- Worst day: `$-647.55`
- Longest loss streak: `4`
- Worst short bucket: `2025-12` / `high_vol_uptrend` avg_ret_net=`-5.7383` rank_ic=`0.4890`

### jan_2026_regime_shift

- Passed: `True`
- Train events: `1590`
- Test events: `238`
- Test AUC: `0.6093`
- Test accuracy: `0.5924`
- Total PnL: `$138.27`
- Profit factor: `1.1101`
- Max drawdown: `$472.93`
- Rank IC vs ret_net: `0.1661`
- ECE: `0.0674`
- Decile spread: `12.5402`
- Long trades: `11`
- Short trades: `24`
- Worst day: `$-182.20`
- Longest loss streak: `3`
- Worst short bucket: `2026-01` / `med_vol_uptrend` avg_ret_net=`-11.0304` rank_ic=`0.5000`

### feb_2026_followthrough

- Passed: `True`
- Train events: `1562`
- Test events: `253`
- Test AUC: `0.5973`
- Test accuracy: `0.5810`
- Total PnL: `$199.75`
- Profit factor: `1.0956`
- Max drawdown: `$626.04`
- Rank IC vs ret_net: `0.1718`
- ECE: `0.0570`
- Decile spread: `12.1593`
- Long trades: `31`
- Short trades: `9`
- Worst day: `$-284.77`
- Longest loss streak: `5`
- Worst short bucket: `2026-02` / `high_vol_downtrend` avg_ret_net=`-6.9028` rank_ic=`-0.1167`

### mar_2026_extension

- Passed: `False`
- Train events: `1609`
- Test events: `248`
- Test AUC: `0.4663`
- Test accuracy: `0.5040`
- Total PnL: `$-407.57`
- Profit factor: `0.8610`
- Max drawdown: `$996.34`
- Rank IC vs ret_net: `0.0042`
- ECE: `0.0943`
- Decile spread: `-3.9859`
- Long trades: `31`
- Short trades: `3`
- Worst day: `$-491.90`
- Longest loss streak: `7`
- Worst short bucket: `2026-03` / `high_vol_downtrend` avg_ret_net=`-35.6447` rank_ic=`0.2571`
- Failures:
  - `test_auc` actual=`0.4662837837837837` expected=`>= 0.52`
  - `total_pnl_usd` actual=`-407.5671068094574` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-11.987267847336982` expected=`>= 3.0`
  - `profit_factor` actual=`0.8610335531635586` expected=`>= 1.05`
  - `long_share` actual=`0.9117647058823529` expected=`<= 0.8`
  - `short_share` actual=`0.08823529411764706` expected=`>= 0.2`

### apr_2026_partial

- Passed: `True`
- Train events: `1575`
- Test events: `64`
- Test AUC: `0.5936`
- Test accuracy: `0.5938`
- Total PnL: `$1382.16`
- Profit factor: `5.6158`
- Max drawdown: `$196.40`
- Rank IC vs ret_net: `0.1482`
- ECE: `0.1210`
- Decile spread: `16.6871`
- Long trades: `6`
- Short trades: `5`
- Worst day: `$134.38`
- Longest loss streak: `2`
- Worst short bucket: `2026-04` / `med_vol_downtrend` avg_ret_net=`-21.7237` rank_ic=`0.1000`

