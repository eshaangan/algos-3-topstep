# Standalone ML Candidate Report

- Generated: `2026-04-12T23:30:02.583307Z`
- Overall pass: `False`
- Passing windows: `3/5`
- Pass ratio: `0.60`
- Overall total PnL: `$509.10`

## Overall Direction

- Long trades: `94`
- Short trades: `89`
- Long share: `0.5136612021857924`
- Short share: `0.48633879781420764`

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
- Total PnL: `$587.11`
- Profit factor: `1.4375`
- Max drawdown: `$337.19`
- Long trades: `10`
- Short trades: `18`

### jan_2026_regime_shift

- Passed: `False`
- Train events: `1603`
- Test events: `261`
- Test AUC: `0.5664`
- Test accuracy: `0.5632`
- Total PnL: `$-616.23`
- Profit factor: `0.7407`
- Max drawdown: `$795.79`
- Long trades: `4`
- Short trades: `46`
- Failures:
  - `total_pnl_usd` actual=`-616.228858980505` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-12.3245771796101` expected=`>= 3.0`
  - `profit_factor` actual=`0.7407072805785595` expected=`>= 1.05`
  - `long_share` actual=`0.08` expected=`>= 0.2`
  - `short_share` actual=`0.92` expected=`<= 0.8`

### feb_2026_followthrough

- Passed: `True`
- Train events: `1589`
- Test events: `289`
- Test AUC: `0.5771`
- Test accuracy: `0.5536`
- Total PnL: `$173.70`
- Profit factor: `1.0703`
- Max drawdown: `$613.15`
- Long trades: `34`
- Short trades: `10`

### mar_2026_extension

- Passed: `False`
- Train events: `1630`
- Test events: `260`
- Test AUC: `0.5413`
- Test accuracy: `0.5577`
- Total PnL: `$-613.41`
- Profit factor: `0.8451`
- Max drawdown: `$1119.61`
- Long trades: `36`
- Short trades: `8`
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
- Total PnL: `$977.94`
- Profit factor: `1.8802`
- Max drawdown: `$210.57`
- Long trades: `10`
- Short trades: `7`

