# Standalone ML Candidate Report

- Generated: `2026-04-13T04:32:55.509105Z`
- Overall pass: `False`
- Passing windows: `0/3`
- Pass ratio: `0.00`
- Overall total PnL: `$2209.00`

## Overall Direction

- Long trades: `121`
- Short trades: `17`
- Long share: `0.8768115942028986`
- Short share: `0.12318840579710146`

## Overall Risk

- Worst day: `$-414.42`
- Best day: `$359.72`
- Daily Sharpe: `3.9087`
- Longest loss streak: `8`
- Recovery factor: `1.9855`

## Overall Failures

- `all_windows` actual=`0` expected=`3 passing windows`
- `pass_ratio` actual=`0.0` expected=`>= 1.0`
- `overall_long_share` actual=`0.8768115942028986` expected=`<= 0.75`
- `overall_short_share` actual=`0.12318840579710146` expected=`>= 0.25`

## Windows

### jul_2024

- Passed: `False`
- Train events: `1649`
- Test events: `314`
- Test AUC: `0.5130`
- Test accuracy: `0.5127`
- Total PnL: `$78.92`
- Profit factor: `1.0427`
- Max drawdown: `$872.55`
- Rank IC vs ret_net: `0.1035`
- ECE: `0.0801`
- Decile spread: `4.7608`
- Long trades: `39`
- Short trades: `7`
- Worst day: `$-323.51`
- Longest loss streak: `8`
- Worst short bucket: `2024-07` / `med_vol_downtrend` avg_ret_net=`-10.5430` rank_ic=`0.1187`
- Failures:
  - `test_auc` actual=`0.5129696069941029` expected=`>= 0.52`
  - `avg_trade_usd` actual=`1.7156090557614707` expected=`>= 3.0`
  - `profit_factor` actual=`1.04265644362381` expected=`>= 1.05`
  - `long_share` actual=`0.8478260869565217` expected=`<= 0.8`
  - `short_share` actual=`0.15217391304347827` expected=`>= 0.2`

### aug_2024

- Passed: `False`
- Train events: `1696`
- Test events: `291`
- Test AUC: `0.5025`
- Test accuracy: `0.5155`
- Total PnL: `$610.12`
- Profit factor: `1.2560`
- Max drawdown: `$976.40`
- Rank IC vs ret_net: `0.0312`
- ECE: `0.0884`
- Decile spread: `0.3091`
- Long trades: `47`
- Short trades: `6`
- Worst day: `$-414.42`
- Longest loss streak: `5`
- Worst short bucket: `2024-08` / `high_vol_downtrend` avg_ret_net=`-26.3845` rank_ic=`0.0000`
- Failures:
  - `test_auc` actual=`0.5025405511041625` expected=`>= 0.52`
  - `long_share` actual=`0.8867924528301887` expected=`<= 0.8`
  - `short_share` actual=`0.11320754716981132` expected=`>= 0.2`

### sep_2024

- Passed: `False`
- Train events: `1772`
- Test events: `246`
- Test AUC: `0.5212`
- Test accuracy: `0.5285`
- Total PnL: `$1519.96`
- Profit factor: `2.0555`
- Max drawdown: `$378.68`
- Rank IC vs ret_net: `0.0885`
- ECE: `0.1079`
- Decile spread: `9.6242`
- Long trades: `35`
- Short trades: `4`
- Worst day: `$-230.00`
- Longest loss streak: `4`
- Worst short bucket: `2024-09` / `med_vol_uptrend` avg_ret_net=`-13.0343` rank_ic=`0.3007`
- Failures:
  - `long_share` actual=`0.8974358974358975` expected=`<= 0.8`
  - `short_share` actual=`0.10256410256410256` expected=`>= 0.2`

