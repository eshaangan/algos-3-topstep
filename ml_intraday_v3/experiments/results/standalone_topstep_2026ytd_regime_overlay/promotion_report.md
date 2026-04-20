# Standalone ML Candidate Report

- Generated: `2026-04-12T21:59:22.560114Z`
- Overall pass: `False`
- Passing windows: `3/5`
- Pass ratio: `0.60`
- Overall total PnL: `$2238.02`

## Overall Direction

- Long trades: `50`
- Short trades: `101`
- Long share: `0.33112582781456956`
- Short share: `0.6688741721854304`

## Overall Failures

- `all_windows` actual=`3` expected=`5 passing windows`
- `pass_ratio` actual=`0.6` expected=`>= 1.0`

## Windows

### dec_2025_bull

- Passed: `True`
- Train events: `1716`
- Test events: `171`
- Test AUC: `0.6434`
- Test accuracy: `0.6550`
- Total PnL: `$511.70`
- Profit factor: `1.3636`
- Max drawdown: `$341.51`
- Long trades: `8`
- Short trades: `20`

### jan_2026_regime_shift

- Passed: `False`
- Train events: `1603`
- Test events: `261`
- Test AUC: `0.5491`
- Test accuracy: `0.5479`
- Total PnL: `$-941.77`
- Profit factor: `0.6289`
- Max drawdown: `$1066.09`
- Long trades: `4`
- Short trades: `45`
- Failures:
  - `total_pnl_usd` actual=`-941.774305367594` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-19.219883783012122` expected=`>= 3.0`
  - `profit_factor` actual=`0.6288773414120641` expected=`>= 1.05`
  - `win_rate` actual=`0.42857142857142855` expected=`>= 0.45`
  - `long_share` actual=`0.08163265306122448` expected=`>= 0.2`
  - `short_share` actual=`0.9183673469387755` expected=`<= 0.8`

### feb_2026_followthrough

- Passed: `True`
- Train events: `1589`
- Test events: `289`
- Test AUC: `0.5810`
- Test accuracy: `0.5640`
- Total PnL: `$1530.09`
- Profit factor: `2.1358`
- Max drawdown: `$491.21`
- Long trades: `18`
- Short trades: `20`

### mar_2026_extension

- Passed: `False`
- Train events: `1630`
- Test events: `260`
- Test AUC: `0.5447`
- Test accuracy: `0.5577`
- Total PnL: `$-875.46`
- Profit factor: `0.6026`
- Max drawdown: `$1785.99`
- Long trades: `12`
- Short trades: `7`
- Failures:
  - `total_pnl_usd` actual=`-875.457886084371` expected=`>= 0.0`
  - `avg_trade_usd` actual=`-46.07673084654584` expected=`>= 3.0`
  - `profit_factor` actual=`0.6025517466531495` expected=`>= 1.05`
  - `win_rate` actual=`0.3157894736842105` expected=`>= 0.45`
  - `max_drawdown_usd` actual=`1785.9901308018088` expected=`<= 1200.0`

### apr_2026_partial

- Passed: `True`
- Train events: `1602`
- Test events: `84`
- Test AUC: `0.6238`
- Test accuracy: `0.5952`
- Total PnL: `$2013.47`
- Profit factor: `4.8914`
- Max drawdown: `$177.26`
- Long trades: `8`
- Short trades: `9`

