# Standalone ML Candidate Report

- Generated: `2026-04-12T22:06:42.591519Z`
- Overall pass: `False`
- Passing windows: `2/5`
- Pass ratio: `0.40`
- Overall total PnL: `$-525.95`

## Overall Direction

- Long trades: `85`
- Short trades: `87`
- Long share: `0.4941860465116279`
- Short share: `0.5058139534883721`

## Overall Failures

- `all_windows` actual=`2` expected=`5 passing windows`
- `pass_ratio` actual=`0.4` expected=`>= 1.0`
- `overall_total_pnl_usd` actual=`-525.9517045294763` expected=`>= 500.0`

## Windows

### dec_2025_bull

- Passed: `False`
- Train events: `1716`
- Test events: `171`
- Test AUC: `0.5777`
- Test accuracy: `0.5906`
- Total PnL: `$-1151.01`
- Profit factor: `0.5373`
- Max drawdown: `$1414.82`
- Long trades: `8`
- Short trades: `23`
- Failures:
  - `total_pnl_usd` actual=`-1151.0126316943429` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-37.129439732075575` expected=`>= 3.0`
  - `profit_factor` actual=`0.5372644949922294` expected=`>= 1.05`
  - `win_rate` actual=`0.3548387096774194` expected=`>= 0.45`
  - `max_drawdown_usd` actual=`1414.8174128611572` expected=`<= 1200.0`

### jan_2026_regime_shift

- Passed: `False`
- Train events: `1603`
- Test events: `261`
- Test AUC: `0.5432`
- Test accuracy: `0.5096`
- Total PnL: `$-237.31`
- Profit factor: `0.9031`
- Max drawdown: `$840.75`
- Long trades: `18`
- Short trades: `32`
- Failures:
  - `total_pnl_usd` actual=`-237.3092923691656` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-4.7461858473833125` expected=`>= 3.0`
  - `profit_factor` actual=`0.9031309669958927` expected=`>= 1.05`
  - `win_rate` actual=`0.44` expected=`>= 0.45`

### feb_2026_followthrough

- Passed: `True`
- Train events: `1589`
- Test events: `289`
- Test AUC: `0.5931`
- Test accuracy: `0.5398`
- Total PnL: `$1505.79`
- Profit factor: `1.7311`
- Max drawdown: `$451.28`
- Long trades: `30`
- Short trades: `17`

### mar_2026_extension

- Passed: `False`
- Train events: `1630`
- Test events: `260`
- Test AUC: `0.5371`
- Test accuracy: `0.5231`
- Total PnL: `$-1760.61`
- Profit factor: `0.4019`
- Max drawdown: `$1674.27`
- Long trades: `18`
- Short trades: `8`
- Failures:
  - `total_pnl_usd` actual=`-1760.6128085989426` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-67.71587725380549` expected=`>= 3.0`
  - `profit_factor` actual=`0.4019076100456313` expected=`>= 1.05`
  - `win_rate` actual=`0.3076923076923077` expected=`>= 0.45`
  - `max_drawdown_usd` actual=`1674.2708318062068` expected=`<= 1200.0`

### apr_2026_partial

- Passed: `True`
- Train events: `1602`
- Test events: `84`
- Test AUC: `0.5891`
- Test accuracy: `0.5476`
- Total PnL: `$1117.19`
- Profit factor: `2.0158`
- Max drawdown: `$199.30`
- Long trades: `11`
- Short trades: `7`

