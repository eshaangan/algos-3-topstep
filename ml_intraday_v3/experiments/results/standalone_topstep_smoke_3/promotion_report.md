# Standalone ML Candidate Report

- Generated: `2026-04-12T21:25:19.290259Z`
- Overall pass: `False`
- Passing windows: `0/3`
- Pass ratio: `0.00`
- Overall total PnL: `$0.00`

## Overall Direction

- Long trades: `0`
- Short trades: `0`
- Long share: `None`
- Short share: `None`

## Overall Failures

- `all_windows` actual=`0` expected=`3 passing windows`
- `pass_ratio` actual=`0.0` expected=`>= 1.0`
- `overall_total_pnl_usd` actual=`0.0` expected=`>= 500.0`

## Windows

### dec_2025_bull

- Passed: `False`
- Train events: `1716`
- Test events: `171`
- Test AUC: `0.5000`
- Test accuracy: `0.5906`
- Total PnL: `$0.00`
- Profit factor: `None`
- Max drawdown: `None`
- Long trades: `0`
- Short trades: `0`
- Failures:
  - `test_auc` actual=`0.5` expected=`>= 0.52`
  - `trades_count` actual=`0.0` expected=`>= 10.0`
  - `positive_prediction_rate` actual=`0.0` expected=`>= 0.02`

### jan_2026_regime_shift

- Passed: `False`
- Train events: `1603`
- Test events: `928`
- Test AUC: `0.5472`
- Test accuracy: `0.5830`
- Total PnL: `$0.00`
- Profit factor: `None`
- Max drawdown: `None`
- Long trades: `0`
- Short trades: `0`
- Failures:
  - `trades_count` actual=`0.0` expected=`>= 10.0`
  - `positive_prediction_rate` actual=`0.0` expected=`>= 0.02`

### feb_2026_followthrough

- Passed: `False`
- Train events: `2250`
- Test events: `273`
- Test AUC: `0.4987`
- Test accuracy: `0.5897`
- Total PnL: `$0.00`
- Profit factor: `None`
- Max drawdown: `None`
- Long trades: `0`
- Short trades: `0`
- Failures:
  - `test_auc` actual=`0.49866903283052355` expected=`>= 0.52`
  - `trades_count` actual=`0.0` expected=`>= 10.0`
  - `positive_prediction_rate` actual=`0.0` expected=`>= 0.02`

