# Regime Watch — When to Deploy Next Filters

## Current live config (as of Mar 23, 2026)
- PrevVWAP filter: DEPLOYED ✅
- VXN/VIX filter: READY but blocked by current regime
- GEX proxy filter: READY but needs more trade sample (N=10, need N≥20)
- BTC filter: Not deploying (no proven edge)

---

## Trigger 1: VXN/VIX Filter — Deploy when regime normalizes

**Check this daily before market open:**
```bash
python -c "
import yfinance as yf
vix = yf.Ticker('^VIX').history(period='2d')['Close'].iloc[-1]
vxn = yf.Ticker('^VXN').history(period='2d')['Close'].iloc[-1]
spread = vxn / vix - 1.0
status = 'DEPLOY READY ✅' if spread < 0.08 else f'blocked ({spread:.3f} > 0.08)'
print(f'VXN/VIX spread: {spread:.3f}  →  {status}')
"
```

**When spread < 0.08 for 3 consecutive days:** Add to `rules.yaml`:
```yaml
require_vxn_normal: true
vxn_vix_threshold: 0.08
```
Then add the check to `live/runner.py::_process_bar()` using `make_vxn_filter(0.08)` from `novel_filter_sweep.py`.

**Expected improvement when deployed:** WR jumps ~15pp, MaxDD cuts in half.
**Backtest reference:** `novel_filter_results.json` → key `vxn_0.08`; ES 2024-2025: 72% WR on 18 trades.

---

## Trigger 2: GEX Proxy Filter — Deploy after combine passes OR N≥20

**Condition:** Combine passed → funded account. OR MNQ YTD accumulates 20+ more trades.

**Config to deploy:** PrevVWAP + GEX proxy (combined)
- Backtest result: 10 trades, 80% WR, Sharpe 10.08, MaxDD -$796
- Implementation: `compute_gex.py::run_gex_backtest()` has full backtest + filter logic

**How to add live:**
1. At session start (9:20 ET), call `compute_gex_today("QQQ")` → get `result["explosive"]`
2. Store as `session_gex_explosive: bool`
3. In `_process_bar()`: `if not self.session_gex_explosive: return`

**GEX proxy data cached:** `data/processed/qqq_gex_daily.csv`

---

## Trigger 3: True Delta ORB — After Databento refill (~$38)

- Refill Databento account
- Run `fetch_trades_l2.py` to pull Feb-Mar tick data
- Build `delta_orb_backtest.py`: require OR cumulative buy_vol > sell_vol for LONG entries
- This is the highest-ceiling filter — requires true tick data, most retailers can't do this

---

## Current regime (Mar 23, 2026)
- VXN/VIX spread: ~0.10 (elevated, tariff uncertainty)
- GEX proxy: mixed (some explosive days)
- Market tone: risk-off for tech, ORB breakouts more likely to reverse on high-VXN days
- PrevVWAP blocking: prevents trading on bearish-close days — correct behavior
