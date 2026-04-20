# Standalone ML Candidate Report

- Generated: `2026-04-13T04:32:40.122692Z`
- Overall pass: `False`
- Passing windows: `3/5`
- Pass ratio: `0.60`
- Overall total PnL: `$3384.26`

## Overall Direction

- Long trades: `87`
- Short trades: `69`
- Long share: `0.5576923076923077`
- Short share: `0.4423076923076923`

## Overall Risk

- Worst day: `$-491.90`
- Best day: `$1042.85`
- Daily Sharpe: `3.5003`
- Longest loss streak: `6`
- Recovery factor: `1.5069`

## Overall Failures

- `all_windows` actual=`3` expected=`5 passing windows`
- `pass_ratio` actual=`0.6` expected=`>= 1.0`

## Windows

### dec_2025_bull

- Passed: `False`
- Train events: `5707`
- Test events: `576`
- Test AUC: `0.6282`
- Test accuracy: `0.5990`
- Total PnL: `$-41.69`
- Profit factor: `0.9688`
- Max drawdown: `$462.94`
- Rank IC vs ret_net: `0.2471`
- ECE: `0.0638`
- Decile spread: `9.8009`
- Long trades: `18`
- Short trades: `11`
- Worst day: `$-202.41`
- Longest loss streak: `6`
- Worst short bucket: `2025-12` / `low_vol_downtrend` avg_ret_net=`-5.6667` rank_ic=`-0.4191`
- Failures:
  - `total_pnl_usd` actual=`-41.68668314594021` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-1.4374718326186278` expected=`>= 3.0`
  - `profit_factor` actual=`0.9688097829204391` expected=`>= 1.05`

### jan_2026_regime_shift

- Passed: `True`
- Train events: `5355`
- Test events: `238`
- Test AUC: `0.5782`
- Test accuracy: `0.5798`
- Total PnL: `$749.78`
- Profit factor: `1.5961`
- Max drawdown: `$333.65`
- Rank IC vs ret_net: `0.0982`
- ECE: `0.0602`
- Decile spread: `7.7377`
- Long trades: `8`
- Short trades: `27`
- Worst day: `$-104.85`
- Longest loss streak: `4`
- Worst short bucket: `2026-01` / `med_vol_uptrend` avg_ret_net=`-11.0304` rank_ic=`0.1167`

### feb_2026_followthrough

- Passed: `True`
- Train events: `4732`
- Test events: `253`
- Test AUC: `0.6033`
- Test accuracy: `0.5692`
- Total PnL: `$1221.65`
- Profit factor: `1.7174`
- Max drawdown: `$273.74`
- Rank IC vs ret_net: `0.1659`
- ECE: `0.0589`
- Decile spread: `23.6593`
- Long trades: `26`
- Short trades: `13`
- Worst day: `$-178.85`
- Longest loss streak: `2`
- Worst short bucket: `2026-02` / `high_vol_downtrend` avg_ret_net=`-6.9028` rank_ic=`0.4500`

### mar_2026_extension

- Passed: `False`
- Train events: `4145`
- Test events: `248`
- Test AUC: `0.5380`
- Test accuracy: `0.5806`
- Total PnL: `$71.77`
- Profit factor: `1.0245`
- Max drawdown: `$907.97`
- Rank IC vs ret_net: `0.0773`
- ECE: `0.1143`
- Decile spread: `2.6583`
- Long trades: `26`
- Short trades: `13`
- Worst day: `$-491.90`
- Longest loss streak: `5`
- Worst short bucket: `2026-03` / `high_vol_downtrend` avg_ret_net=`-35.6447` rank_ic=`0.4286`
- Failures:
  - `avg_trade_usd` actual=`1.8403462013591259` expected=`>= 3.0`
  - `profit_factor` actual=`1.0244973732248794` expected=`>= 1.05`

### apr_2026_partial

- Passed: `True`
- Train events: `3405`
- Test events: `64`
- Test AUC: `0.6396`
- Test accuracy: `0.5938`
- Total PnL: `$1382.75`
- Profit factor: `3.8537`
- Max drawdown: `$185.11`
- Rank IC vs ret_net: `0.2429`
- ECE: `0.0737`
- Decile spread: `30.4530`
- Long trades: `9`
- Short trades: `5`
- Worst day: `$-144.91`
- Longest loss streak: `2`
- Worst short bucket: `2026-04` / `med_vol_downtrend` avg_ret_net=`-21.7237` rank_ic=`0.3000`

