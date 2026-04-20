# Standalone ML Candidate Report

- Generated: `2026-04-12T21:50:10.888977Z`
- Overall pass: `False`
- Passing windows: `2/5`
- Pass ratio: `0.40`
- Overall total PnL: `$1359.44`

## Overall Direction

- Long trades: `27`
- Short trades: `71`
- Long share: `0.2755102040816326`
- Short share: `0.7244897959183674`

## Overall Failures

- `all_windows` actual=`2` expected=`5 passing windows`
- `pass_ratio` actual=`0.4` expected=`>= 1.0`

## Windows

### dec_2025_bull

- Passed: `False`
- Train events: `1716`
- Test events: `171`
- Test AUC: `0.6434`
- Test accuracy: `0.6550`
- Total PnL: `$119.08`
- Profit factor: `1.2373`
- Max drawdown: `$282.49`
- Long trades: `0`
- Short trades: `12`
- Failures:
  - `long_share` actual=`0.0` expected=`>= 0.2`
  - `short_share` actual=`1.0` expected=`<= 0.8`

### jan_2026_regime_shift

- Passed: `False`
- Train events: `1603`
- Test events: `261`
- Test AUC: `0.5491`
- Test accuracy: `0.5479`
- Total PnL: `$-71.26`
- Profit factor: `0.9562`
- Max drawdown: `$541.45`
- Long trades: `0`
- Short trades: `35`
- Failures:
  - `total_pnl_usd` actual=`-71.25873605199877` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-2.035963887199965` expected=`>= 3.0`
  - `profit_factor` actual=`0.956173892054583` expected=`>= 1.05`
  - `long_share` actual=`0.0` expected=`>= 0.2`
  - `short_share` actual=`1.0` expected=`<= 0.8`

### feb_2026_followthrough

- Passed: `True`
- Train events: `1589`
- Test events: `289`
- Test AUC: `0.5810`
- Test accuracy: `0.5640`
- Total PnL: `$565.61`
- Profit factor: `1.9799`
- Max drawdown: `$263.43`
- Long trades: `10`
- Short trades: `10`

### mar_2026_extension

- Passed: `False`
- Train events: `1630`
- Test events: `260`
- Test AUC: `0.5447`
- Test accuracy: `0.5577`
- Total PnL: `$-729.08`
- Profit factor: `0.7182`
- Max drawdown: `$1535.78`
- Long trades: `15`
- Short trades: `6`
- Failures:
  - `total_pnl_usd` actual=`-729.080105237724` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-34.71810024941543` expected=`>= 3.0`
  - `profit_factor` actual=`0.7182454789033835` expected=`>= 1.05`
  - `win_rate` actual=`0.3333333333333333` expected=`>= 0.45`
  - `max_drawdown_usd` actual=`1535.7798799249213` expected=`<= 1200.0`

### apr_2026_partial

- Passed: `True`
- Train events: `1602`
- Test events: `84`
- Test AUC: `0.6238`
- Test accuracy: `0.5952`
- Total PnL: `$1475.10`
- Profit factor: `9.4476`
- Max drawdown: `$91.44`
- Long trades: `2`
- Short trades: `8`

