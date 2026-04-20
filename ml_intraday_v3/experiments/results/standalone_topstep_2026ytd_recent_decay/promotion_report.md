# Standalone ML Candidate Report

- Generated: `2026-04-12T22:11:17.013506Z`
- Overall pass: `False`
- Passing windows: `3/5`
- Pass ratio: `0.60`
- Overall total PnL: `$3193.92`

## Overall Direction

- Long trades: `87`
- Short trades: `105`
- Long share: `0.453125`
- Short share: `0.546875`

## Overall Failures

- `all_windows` actual=`3` expected=`5 passing windows`
- `pass_ratio` actual=`0.6` expected=`>= 1.0`

## Windows

### dec_2025_bull

- Passed: `True`
- Train events: `1716`
- Test events: `171`
- Test AUC: `0.6088`
- Test accuracy: `0.6491`
- Total PnL: `$658.70`
- Profit factor: `1.5583`
- Max drawdown: `$337.19`
- Long trades: `7`
- Short trades: `18`

### jan_2026_regime_shift

- Passed: `False`
- Train events: `1603`
- Test events: `261`
- Test AUC: `0.5695`
- Test accuracy: `0.5670`
- Total PnL: `$-756.90`
- Profit factor: `0.7029`
- Max drawdown: `$833.29`
- Long trades: `4`
- Short trades: `47`
- Failures:
  - `total_pnl_usd` actual=`-756.8977477032281` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-14.841132307906433` expected=`>= 3.0`
  - `profit_factor` actual=`0.70285631142442` expected=`>= 1.05`
  - `long_share` actual=`0.0784313725490196` expected=`>= 0.2`
  - `short_share` actual=`0.9215686274509803` expected=`<= 0.8`

### feb_2026_followthrough

- Passed: `True`
- Train events: `1589`
- Test events: `289`
- Test AUC: `0.5665`
- Test accuracy: `0.5536`
- Total PnL: `$1123.60`
- Profit factor: `1.5047`
- Max drawdown: `$401.24`
- Long trades: `30`
- Short trades: `19`

### mar_2026_extension

- Passed: `False`
- Train events: `1630`
- Test events: `260`
- Test AUC: `0.5391`
- Test accuracy: `0.5500`
- Total PnL: `$422.62`
- Profit factor: `1.1185`
- Max drawdown: `$1358.31`
- Long trades: `37`
- Short trades: `13`
- Failures:
  - `win_rate` actual=`0.44` expected=`>= 0.45`
  - `max_drawdown_usd` actual=`1358.3115453637874` expected=`<= 1200.0`

### apr_2026_partial

- Passed: `True`
- Train events: `1602`
- Test events: `84`
- Test AUC: `0.6059`
- Test accuracy: `0.5595`
- Total PnL: `$1745.90`
- Profit factor: `3.5133`
- Max drawdown: `$268.70`
- Long trades: `9`
- Short trades: `8`

