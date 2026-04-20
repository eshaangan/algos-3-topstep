# Standalone ML Candidate Report

- Generated: `2026-04-12T23:29:15.762564Z`
- Overall pass: `False`
- Passing windows: `1/5`
- Pass ratio: `0.20`
- Overall total PnL: `$-822.84`

## Overall Direction

- Long trades: `81`
- Short trades: `92`
- Long share: `0.4682080924855491`
- Short share: `0.5317919075144508`

## Overall Failures

- `all_windows` actual=`1` expected=`5 passing windows`
- `pass_ratio` actual=`0.2` expected=`>= 1.0`
- `overall_total_pnl_usd` actual=`-822.8376400494096` expected=`>= 500.0`

## Windows

### dec_2025_bull

- Passed: `False`
- Train events: `1716`
- Test events: `171`
- Test AUC: `0.6116`
- Test accuracy: `0.6550`
- Total PnL: `$-11.45`
- Profit factor: `0.9930`
- Max drawdown: `$566.64`
- Long trades: `11`
- Short trades: `18`
- Failures:
  - `total_pnl_usd` actual=`-11.454571837025696` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-0.3949852357595068` expected=`>= 3.0`
  - `profit_factor` actual=`0.9929793564976342` expected=`>= 1.05`
  - `win_rate` actual=`0.41379310344827586` expected=`>= 0.45`

### jan_2026_regime_shift

- Passed: `False`
- Train events: `1603`
- Test events: `261`
- Test AUC: `0.5703`
- Test accuracy: `0.5670`
- Total PnL: `$-734.35`
- Profit factor: `0.7090`
- Max drawdown: `$969.00`
- Long trades: `4`
- Short trades: `47`
- Failures:
  - `total_pnl_usd` actual=`-734.3486143012428` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-14.39899243727927` expected=`>= 3.0`
  - `profit_factor` actual=`0.7089797241568221` expected=`>= 1.05`
  - `long_share` actual=`0.0784313725490196` expected=`>= 0.2`
  - `short_share` actual=`0.9215686274509803` expected=`<= 0.8`

### feb_2026_followthrough

- Passed: `False`
- Train events: `1589`
- Test events: `289`
- Test AUC: `0.5840`
- Test accuracy: `0.5640`
- Total PnL: `$-269.61`
- Profit factor: `0.9010`
- Max drawdown: `$663.94`
- Long trades: `34`
- Short trades: `12`
- Failures:
  - `total_pnl_usd` actual=`-269.6112368586394` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-5.861113844753031` expected=`>= 3.0`
  - `profit_factor` actual=`0.9009549094908003` expected=`>= 1.05`

### mar_2026_extension

- Passed: `False`
- Train events: `1630`
- Test events: `260`
- Test AUC: `0.5268`
- Test accuracy: `0.5269`
- Total PnL: `$-1768.04`
- Profit factor: `0.4652`
- Max drawdown: `$1681.69`
- Long trades: `24`
- Short trades: `6`
- Failures:
  - `total_pnl_usd` actual=`-1768.0357826361455` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-58.93452608787152` expected=`>= 3.0`
  - `profit_factor` actual=`0.4651907984068216` expected=`>= 1.05`
  - `win_rate` actual=`0.36666666666666664` expected=`>= 0.45`
  - `max_drawdown_usd` actual=`1681.693805843388` expected=`<= 1200.0`

### apr_2026_partial

- Passed: `True`
- Train events: `1602`
- Test events: `84`
- Test AUC: `0.5775`
- Test accuracy: `0.5833`
- Total PnL: `$1960.61`
- Profit factor: `4.3140`
- Max drawdown: `$268.70`
- Long trades: `8`
- Short trades: `9`

