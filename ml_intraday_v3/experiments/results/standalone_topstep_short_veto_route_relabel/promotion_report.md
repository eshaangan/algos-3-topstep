# Standalone ML Candidate Report

- Generated: `2026-04-13T00:07:39.620903Z`
- Overall pass: `False`
- Passing windows: `3/5`
- Pass ratio: `0.60`
- Overall total PnL: `$2445.18`

## Overall Direction

- Long trades: `94`
- Short trades: `80`
- Long share: `0.5402298850574713`
- Short share: `0.45977011494252873`

## Overall Risk

- Worst day: `$-485.41`
- Best day: `$931.04`
- Daily Sharpe: `2.4850`
- Longest loss streak: `8`
- Recovery factor: `1.2286`

## Overall Failures

- `all_windows` actual=`3` expected=`5 passing windows`
- `pass_ratio` actual=`0.6` expected=`>= 1.0`

## Windows

### dec_2025_bull

- Passed: `True`
- Train events: `1716`
- Test events: `171`
- Test AUC: `0.5997`
- Test accuracy: `0.6667`
- Total PnL: `$658.95`
- Profit factor: `1.5188`
- Max drawdown: `$337.19`
- Rank IC vs ret_net: `0.2505`
- ECE: `0.1002`
- Decile spread: `7.6585`
- Long trades: `10`
- Short trades: `17`
- Worst day: `$-337.19`
- Longest loss streak: `4`
- Worst short bucket: `2025-12` / `med_vol_downtrend` avg_ret_net=`-5.9210` rank_ic=`0.3077`

### jan_2026_regime_shift

- Passed: `False`
- Train events: `1603`
- Test events: `261`
- Test AUC: `0.5664`
- Test accuracy: `0.5632`
- Total PnL: `$487.47`
- Profit factor: `1.2890`
- Max drawdown: `$580.40`
- Rank IC vs ret_net: `0.0849`
- ECE: `0.1171`
- Decile spread: `7.7373`
- Long trades: `4`
- Short trades: `40`
- Worst day: `$-182.20`
- Longest loss streak: `6`
- Worst short bucket: `2026-01` / `low_vol_uptrend` avg_ret_net=`-15.2597` rank_ic=`-0.8000`
- Failures:
  - `long_share` actual=`0.09090909090909091` expected=`>= 0.2`
  - `short_share` actual=`0.9090909090909091` expected=`<= 0.8`

### feb_2026_followthrough

- Passed: `True`
- Train events: `1589`
- Test events: `289`
- Test AUC: `0.5771`
- Test accuracy: `0.5536`
- Total PnL: `$826.70`
- Profit factor: `1.4063`
- Max drawdown: `$409.14`
- Rank IC vs ret_net: `0.1679`
- ECE: `0.0480`
- Decile spread: `15.5853`
- Long trades: `34`
- Short trades: `9`
- Worst day: `$-278.46`
- Longest loss streak: `3`
- Worst short bucket: `2026-02` / `low_vol_downtrend` avg_ret_net=`-4.5244` rank_ic=`0.0588`

### mar_2026_extension

- Passed: `False`
- Train events: `1630`
- Test events: `260`
- Test AUC: `0.5413`
- Test accuracy: `0.5577`
- Total PnL: `$-613.41`
- Profit factor: `0.8451`
- Max drawdown: `$1119.61`
- Rank IC vs ret_net: `0.1355`
- ECE: `0.0899`
- Decile spread: `1.3721`
- Long trades: `36`
- Short trades: `8`
- Worst day: `$-485.41`
- Longest loss streak: `8`
- Worst short bucket: `2026-03` / `high_vol_downtrend` avg_ret_net=`-36.7336` rank_ic=`0.6190`
- Failures:
  - `total_pnl_usd` actual=`-613.4109010316879` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-13.94115684162927` expected=`>= 3.0`
  - `profit_factor` actual=`0.8451417710277089` expected=`>= 1.05`
  - `win_rate` actual=`0.4318181818181818` expected=`>= 0.45`
  - `long_share` actual=`0.8181818181818182` expected=`<= 0.8`
  - `short_share` actual=`0.18181818181818182` expected=`>= 0.2`

### apr_2026_partial

- Passed: `True`
- Train events: `1602`
- Test events: `84`
- Test AUC: `0.6047`
- Test accuracy: `0.5595`
- Total PnL: `$1085.47`
- Profit factor: `2.0816`
- Max drawdown: `$196.40`
- Rank IC vs ret_net: `0.1630`
- ECE: `0.1732`
- Decile spread: `5.6883`
- Long trades: `10`
- Short trades: `6`
- Worst day: `$-256.33`
- Longest loss streak: `2`
- Worst short bucket: `2026-04` / `med_vol_sideways` avg_ret_net=`-18.4090` rank_ic=`0.5000`

