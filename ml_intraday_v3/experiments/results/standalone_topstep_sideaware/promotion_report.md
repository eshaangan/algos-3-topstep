# Standalone ML Candidate Report

- Generated: `2026-04-12T21:35:46.408674Z`
- Overall pass: `False`
- Passing windows: `2/3`
- Pass ratio: `0.67`
- Overall total PnL: `$82.41`

## Overall Direction

- Long trades: `40`
- Short trades: `67`
- Long share: `0.37383177570093457`
- Short share: `0.6261682242990654`

## Overall Failures

- `all_windows` actual=`2` expected=`3 passing windows`
- `pass_ratio` actual=`0.6666666666666666` expected=`>= 1.0`
- `overall_total_pnl_usd` actual=`82.4114934144381` expected=`>= 500.0`

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
- Test events: `928`
- Test AUC: `0.5448`
- Test accuracy: `0.5550`
- Total PnL: `$-420.51`
- Profit factor: `0.8075`
- Max drawdown: `$520.91`
- Long trades: `16`
- Short trades: `40`
- Failures:
  - `total_pnl_usd` actual=`-420.5139650072357` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-7.50917794655778` expected=`>= 3.0`
  - `profit_factor` actual=`0.8075248853405624` expected=`>= 1.05`
  - `win_rate` actual=`0.42857142857142855` expected=`>= 0.45`

### feb_2026_followthrough

- Passed: `True`
- Train events: `2250`
- Test events: `273`
- Test AUC: `0.5627`
- Test accuracy: `0.5751`
- Total PnL: `$208.69`
- Profit factor: `1.2013`
- Max drawdown: `$507.45`
- Long trades: `11`
- Short trades: `10`

