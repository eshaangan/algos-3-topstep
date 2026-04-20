# Standalone ML Candidate Report

- Generated: `2026-04-12T21:45:34.110090Z`
- Overall pass: `False`
- Passing windows: `2/5`
- Pass ratio: `0.40`
- Overall total PnL: `$-471.38`

## Overall Direction

- Long trades: `79`
- Short trades: `89`
- Long share: `0.47023809523809523`
- Short share: `0.5297619047619048`

## Overall Failures

- `all_windows` actual=`2` expected=`5 passing windows`
- `pass_ratio` actual=`0.4` expected=`>= 1.0`
- `overall_total_pnl_usd` actual=`-471.3760855134724` expected=`>= 500.0`

## Windows

### dec_2025_bull

- Passed: `True`
- Train events: `1716`
- Test events: `171`
- Test AUC: `0.6434`
- Test accuracy: `0.6550`
- Total PnL: `$294.24`
- Profit factor: `1.1959`
- Max drawdown: `$586.91`
- Long trades: `13`
- Short trades: `17`

### jan_2026_regime_shift

- Passed: `False`
- Train events: `1603`
- Test events: `261`
- Test AUC: `0.5491`
- Test accuracy: `0.5479`
- Total PnL: `$-860.48`
- Profit factor: `0.6938`
- Max drawdown: `$930.63`
- Long trades: `14`
- Short trades: `39`
- Failures:
  - `total_pnl_usd` actual=`-860.4781908986558` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-16.235437564125583` expected=`>= 3.0`
  - `profit_factor` actual=`0.6938257070990062` expected=`>= 1.05`
  - `win_rate` actual=`0.37735849056603776` expected=`>= 0.45`

### feb_2026_followthrough

- Passed: `False`
- Train events: `1589`
- Test events: `289`
- Test AUC: `0.5810`
- Test accuracy: `0.5640`
- Total PnL: `$1.50`
- Profit factor: `1.0006`
- Max drawdown: `$798.80`
- Long trades: `32`
- Short trades: `17`
- Failures:
  - `avg_trade_usd` actual=`0.03058518749083314` expected=`>= 3.0`
  - `profit_factor` actual=`1.0005519405847882` expected=`>= 1.05`

### mar_2026_extension

- Passed: `False`
- Train events: `1630`
- Test events: `260`
- Test AUC: `0.5447`
- Test accuracy: `0.5577`
- Total PnL: `$-1761.88`
- Profit factor: `0.2957`
- Max drawdown: `$1479.92`
- Long trades: `12`
- Short trades: `6`
- Failures:
  - `total_pnl_usd` actual=`-1761.8842209924837` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-97.88245672180466` expected=`>= 3.0`
  - `profit_factor` actual=`0.29570068574908015` expected=`>= 1.05`
  - `win_rate` actual=`0.2777777777777778` expected=`>= 0.45`
  - `max_drawdown_usd` actual=`1479.9164657099245` expected=`<= 1200.0`

### apr_2026_partial

- Passed: `True`
- Train events: `1602`
- Test events: `84`
- Test AUC: `0.6238`
- Test accuracy: `0.5952`
- Total PnL: `$1855.25`
- Profit factor: `3.7398`
- Max drawdown: `$276.58`
- Long trades: `8`
- Short trades: `10`

