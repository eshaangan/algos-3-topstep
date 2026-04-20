# Standalone ML Candidate Report

- Generated: `2026-04-12T21:28:39.337330Z`
- Overall pass: `False`
- Passing windows: `0/3`
- Pass ratio: `0.00`
- Overall total PnL: `$-177.50`

## Overall Direction

- Long trades: `0`
- Short trades: `1`
- Long share: `0.0`
- Short share: `1.0`

## Overall Failures

- `all_windows` actual=`0` expected=`3 passing windows`
- `pass_ratio` actual=`0.0` expected=`>= 1.0`
- `overall_total_pnl_usd` actual=`-177.49844071308618` expected=`>= 500.0`
- `overall_long_share` actual=`0.0` expected=`>= 0.25`
- `overall_short_share` actual=`1.0` expected=`<= 0.75`

## Windows

### dec_2025_bull

- Passed: `False`
- Train events: `1716`
- Test events: `171`
- Test AUC: `0.6434`
- Test accuracy: `0.6550`
- Total PnL: `$0.00`
- Profit factor: `None`
- Max drawdown: `None`
- Long trades: `0`
- Short trades: `0`
- Failures:
  - `trades_count` actual=`0.0` expected=`>= 10.0`

### jan_2026_regime_shift

- Passed: `False`
- Train events: `1603`
- Test events: `928`
- Test AUC: `0.5448`
- Test accuracy: `0.5550`
- Total PnL: `$0.00`
- Profit factor: `None`
- Max drawdown: `None`
- Long trades: `0`
- Short trades: `0`
- Failures:
  - `trades_count` actual=`0.0` expected=`>= 10.0`

### feb_2026_followthrough

- Passed: `False`
- Train events: `2250`
- Test events: `273`
- Test AUC: `0.5627`
- Test accuracy: `0.5751`
- Total PnL: `$-177.50`
- Profit factor: `None`
- Max drawdown: `$0.00`
- Long trades: `0`
- Short trades: `1`
- Failures:
  - `total_pnl_usd` actual=`-177.49844071308618` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-177.49844071308618` expected=`>= 3.0`
  - `win_rate` actual=`0.0` expected=`>= 0.45`
  - `trades_count` actual=`1.0` expected=`>= 10.0`
  - `long_share` actual=`0.0` expected=`>= 0.2`
  - `short_share` actual=`1.0` expected=`<= 0.8`

