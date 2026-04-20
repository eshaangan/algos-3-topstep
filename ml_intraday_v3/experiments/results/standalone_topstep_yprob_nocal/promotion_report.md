# Standalone ML Candidate Report

- Generated: `2026-04-12T21:30:19.528660Z`
- Overall pass: `False`
- Passing windows: `1/3`
- Pass ratio: `0.33`
- Overall total PnL: `$-698.70`

## Overall Direction

- Long trades: `19`
- Short trades: `85`
- Long share: `0.18269230769230768`
- Short share: `0.8173076923076923`

## Overall Failures

- `all_windows` actual=`1` expected=`3 passing windows`
- `pass_ratio` actual=`0.3333333333333333` expected=`>= 1.0`
- `overall_total_pnl_usd` actual=`-698.7007880943323` expected=`>= 500.0`
- `overall_long_share` actual=`0.18269230769230768` expected=`>= 0.25`
- `overall_short_share` actual=`0.8173076923076923` expected=`<= 0.75`

## Windows

### dec_2025_bull

- Passed: `False`
- Train events: `1716`
- Test events: `171`
- Test AUC: `0.6434`
- Test accuracy: `0.6550`
- Total PnL: `$-800.10`
- Profit factor: `0.6422`
- Max drawdown: `$952.38`
- Long trades: `6`
- Short trades: `25`
- Failures:
  - `total_pnl_usd` actual=`-800.1004223445032` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-25.80969104337107` expected=`>= 3.0`
  - `profit_factor` actual=`0.6422127269778002` expected=`>= 1.05`
  - `win_rate` actual=`0.41935483870967744` expected=`>= 0.45`
  - `long_share` actual=`0.1935483870967742` expected=`>= 0.2`
  - `short_share` actual=`0.8064516129032258` expected=`<= 0.8`

### jan_2026_regime_shift

- Passed: `False`
- Train events: `1603`
- Test events: `928`
- Test AUC: `0.5448`
- Test accuracy: `0.5550`
- Total PnL: `$-38.56`
- Profit factor: `0.9803`
- Max drawdown: `$523.04`
- Long trades: `4`
- Short trades: `49`
- Failures:
  - `total_pnl_usd` actual=`-38.56395370584204` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-0.7276217680347554` expected=`>= 3.0`
  - `profit_factor` actual=`0.9803179171519936` expected=`>= 1.05`
  - `long_share` actual=`0.07547169811320754` expected=`>= 0.2`
  - `short_share` actual=`0.9245283018867925` expected=`<= 0.8`

### feb_2026_followthrough

- Passed: `True`
- Train events: `2250`
- Test events: `273`
- Test AUC: `0.5627`
- Test accuracy: `0.5751`
- Total PnL: `$139.96`
- Profit factor: `1.1419`
- Max drawdown: `$575.72`
- Long trades: `9`
- Short trades: `11`

