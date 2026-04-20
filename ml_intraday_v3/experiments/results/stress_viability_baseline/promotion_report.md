# Standalone ML Candidate Report

- Generated: `2026-04-13T04:19:37.278880Z`
- Overall pass: `False`
- Passing windows: `3/5`
- Pass ratio: `0.60`
- Overall total PnL: `$3435.91`

## Overall Direction

- Long trades: `92`
- Short trades: `47`
- Long share: `0.6618705035971223`
- Short share: `0.3381294964028777`

## Overall Risk

- Worst day: `$-524.60`
- Best day: `$1030.99`
- Daily Sharpe: `3.2766`
- Longest loss streak: `4`
- Recovery factor: `1.6351`

## Overall Failures

- `all_windows` actual=`3` expected=`5 passing windows`
- `pass_ratio` actual=`0.6` expected=`>= 1.0`

## Windows

### dec_2025_bull

- Passed: `True`
- Train events: `1515`
- Test events: `147`
- Test AUC: `0.6047`
- Test accuracy: `0.5782`
- Total PnL: `$594.58`
- Profit factor: `1.6441`
- Max drawdown: `$524.60`
- Rank IC vs ret_net: `0.2695`
- ECE: `0.0926`
- Decile spread: `22.7132`
- Long trades: `13`
- Short trades: `7`
- Worst day: `$-524.60`
- Longest loss streak: `3`
- Worst short bucket: `2025-12` / `low_vol_downtrend` avg_ret_net=`-8.1677` rank_ic=`0.2000`

### jan_2026_regime_shift

- Passed: `True`
- Train events: `1440`
- Test events: `224`
- Test AUC: `0.6342`
- Test accuracy: `0.6027`
- Total PnL: `$380.54`
- Profit factor: `1.2392`
- Max drawdown: `$546.18`
- Rank IC vs ret_net: `0.1663`
- ECE: `0.0699`
- Decile spread: `17.9116`
- Long trades: `13`
- Short trades: `24`
- Worst day: `$-476.25`
- Longest loss streak: `3`
- Worst short bucket: `2026-01` / `med_vol_uptrend` avg_ret_net=`-12.5891` rank_ic=`0.1167`

### feb_2026_followthrough

- Passed: `False`
- Train events: `1428`
- Test events: `238`
- Test AUC: `0.5568`
- Test accuracy: `0.5336`
- Total PnL: `$799.92`
- Profit factor: `1.4667`
- Max drawdown: `$327.36`
- Rank IC vs ret_net: `0.1020`
- ECE: `0.0837`
- Decile spread: `8.9947`
- Long trades: `30`
- Short trades: `7`
- Worst day: `$-183.86`
- Longest loss streak: `3`
- Worst short bucket: `2026-02` / `high_vol_downtrend` avg_ret_net=`-8.0162` rank_ic=`0.2167`
- Failures:
  - `long_share` actual=`0.8108108108108109` expected=`<= 0.8`
  - `short_share` actual=`0.1891891891891892` expected=`>= 0.2`

### mar_2026_extension

- Passed: `False`
- Train events: `1496`
- Test events: `234`
- Test AUC: `0.5206`
- Test accuracy: `0.4957`
- Total PnL: `$103.22`
- Profit factor: `1.0371`
- Max drawdown: `$1044.79`
- Rank IC vs ret_net: `0.0535`
- ECE: `0.1009`
- Decile spread: `5.6827`
- Long trades: `30`
- Short trades: `3`
- Worst day: `$-516.95`
- Longest loss streak: `4`
- Worst short bucket: `2026-03` / `high_vol_downtrend` avg_ret_net=`-39.8130` rank_ic=`-1.0000`
- Failures:
  - `test_accuracy` actual=`0.49572649572649574` expected=`>= 0.5`
  - `profit_factor` actual=`1.0370624162507123` expected=`>= 1.05`
  - `long_share` actual=`0.9090909090909091` expected=`<= 0.8`
  - `short_share` actual=`0.09090909090909091` expected=`>= 0.2`

### apr_2026_partial

- Passed: `True`
- Train events: `1478`
- Test events: `61`
- Test AUC: `0.6198`
- Test accuracy: `0.6230`
- Total PnL: `$1557.63`
- Profit factor: `6.5230`
- Max drawdown: `$113.06`
- Rank IC vs ret_net: `0.1558`
- ECE: `0.1156`
- Decile spread: `1.9556`
- Long trades: `6`
- Short trades: `6`
- Worst day: `$30.79`
- Longest loss streak: `1`
- Worst short bucket: `2026-04` / `med_vol_downtrend` avg_ret_net=`-23.8233` rank_ic=`-0.4000`

