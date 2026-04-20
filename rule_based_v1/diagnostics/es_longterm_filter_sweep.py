"""
15-Year ES ORB Filter Sweep — GCP Overnight Run
================================================
Backtests VXN/VIX and BTC regime filters on 15 years of ES 5-min data
(June 2010 – Dec 2025). Validates whether the novel filters identified on
MNQ 2026 YTD generalize across market regimes.

Why ES instead of MNQ for long-term backtest:
  - We have 15 years of ES data (306,933 bars) — MNQ only launched 2019
  - ES and MNQ track NASDAQ futures with high correlation (>0.97)
  - Results on ES validate the principle; live trading stays on MNQ

Run locally (~5 min):
    python rule_based_v1/diagnostics/es_longterm_filter_sweep.py

Run on GCP VM (background, log to file):
    nohup python3 rule_based_v1/diagnostics/es_longterm_filter_sweep.py \
        > ~/logs/es_sweep.log 2>&1 &

Output:
    rule_based_v1/diagnostics/es_longterm_results.json  — full results
    (Console output: year-by-year WR, Sharpe, max DD per config)
"""
from __future__ import annotations

import argparse
import json
import sys
import datetime
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.indicators import atr as compute_atr

DATA_PATH    = ROOT / "data" / "processed" / "es_bars_2010_2025.h5"
RESULTS_PATH = ROOT / "rule_based_v1" / "diagnostics" / "es_longterm_results.json"

# ── Execution config (ES-adapted) ─────────────────────────────────────────────
# ES point value = $50/point; MNQ = $2/point
# Calibrate to similar dollar exposure: 1 ES contract ≈ 25 MNQ contracts
# Use 1 ES contract for consistency; results will be in ES P&L
N_CONTRACTS    = 1
POINT_VALUE    = 50.0         # ES: $50/point
TICK_SIZE      = 0.25
COMMISSION     = 2.02         # ES commission per side
SLIPPAGE_TICKS = 1
PT_MULT        = 3.0
SL_MULT        = 1.5
ATR_PERIOD     = 14
TIME_STOP_BARS = 24
MAX_DAILY_LOSS = -2_000.0     # ES equivalent of MNQ -$950 (×2.1 scale)
MAX_TRADES_DAY = 1
STARTING_EQUITY  = 50_000.0
DRAWDOWN_BUFFER  = 4_000.0    # ES equivalent of MNQ $1,950


@dataclass
class Pos:
    entry_price:   float
    entry_bar_idx: int
    stop_loss:     float
    profit_target: float
    time_stop_bar: int
    n_contracts:   int
    atr:           float


def _slip(price, direction, is_entry):
    s = SLIPPAGE_TICKS * TICK_SIZE
    return price + s * direction if is_entry else price - s * direction


def _calc_pnl(entry, exit_, n):
    return (exit_ - entry) * n * POINT_VALUE - 2 * COMMISSION * n


def _check_exit(pos, bar, idx, sess_close):
    h, l, c = bar["high"], bar["low"], bar["close"]
    if sess_close:
        return True, _slip(c, 1, False), "session_close"
    if idx >= pos.time_stop_bar:
        return True, _slip(c, 1, False), "time_stop"
    if l <= pos.stop_loss:
        return True, _slip(pos.stop_loss, 1, False), "stop_loss"
    if h >= pos.profit_target:
        return True, _slip(pos.profit_target, 1, False), "profit_target"
    return False, 0.0, ""


def _build_day_meta_vectorized(bars: pd.DataFrame) -> dict:
    """Fast vectorized day meta: ATR and gap only (no ATR percentile for speed)."""
    daily = bars.resample("1D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),   close=("close", "last"),
    ).dropna()
    if daily.index.tz is None:
        daily.index = daily.index.tz_localize("US/Eastern")

    daily["atr14"]   = compute_atr(daily["high"], daily["low"], daily["close"], 14)
    daily["gap_pct"] = (daily["open"] - daily["close"].shift(1)) / daily["close"].shift(1)

    meta = {}
    for ts, row in daily.iterrows():
        d = ts.tz_convert("US/Eastern").date() if hasattr(ts, "tz_convert") else ts.date()
        meta[d] = {
            "atr14":   float(row["atr14"])   if not np.isnan(row["atr14"])   else 10.0,
            "gap_pct": float(row["gap_pct"]) if not np.isnan(row["gap_pct"]) else 0.0,
        }
    return meta


def _enrich_vix_vxn(meta: dict, start: str, end: str, verbose: bool = True) -> dict:
    """Add tech_fear_spread = VXN/VIX - 1 (previous day's) to each date in meta."""
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed — VIX/VXN enrichment skipped")
        for d in meta:
            meta[d]["tech_fear_spread"] = None
        return meta

    if verbose:
        print(f"  Fetching ^VIX and ^VXN ({start} → {end}) ...")

    try:
        vix_df = yf.download("^VIX", start=start, end=end, interval="1d", progress=False, auto_adjust=True)
        vxn_df = yf.download("^VXN", start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"  yfinance error: {e}")
        for d in meta:
            meta[d]["tech_fear_spread"] = None
        return meta

    def to_series(df, label):
        c = df["Close"] if hasattr(df, "columns") else df
        if hasattr(c, "squeeze"):
            c = c.squeeze()
        idx = c.index
        if hasattr(idx, "tz") and idx.tz is not None:
            dates = [pd.to_datetime(t).tz_convert("US/Eastern").date() for t in idx]
        else:
            dates = [pd.to_datetime(t).date() for t in idx]
        return pd.Series(c.values.flatten().astype(float), index=dates, name=label)

    vix = to_series(vix_df, "VIX")
    vxn = to_series(vxn_df, "VXN")
    spread = (vxn / vix - 1.0)

    # VXN only available from around 2001; before that spread=None
    dates_sorted = sorted(meta.keys())
    hits = 0
    for i, d in enumerate(dates_sorted):
        if i == 0:
            meta[d]["tech_fear_spread"] = None
        else:
            prev_d = dates_sorted[i - 1]
            val = spread.get(prev_d)
            if val is not None and not np.isnan(float(val)):
                meta[d]["tech_fear_spread"] = float(val)
                hits += 1
            else:
                meta[d]["tech_fear_spread"] = None

    if verbose:
        print(f"    VIX/VXN spread populated for {hits}/{len(dates_sorted)} days")
    return meta


def _enrich_btc(meta: dict, start: str, end: str, verbose: bool = True) -> dict:
    """Add btc_prev_return (previous day BTC daily return) to meta."""
    try:
        import yfinance as yf
    except ImportError:
        for d in meta:
            meta[d]["btc_prev_return"] = None
        return meta

    # BTC-USD data only available from ~2014 on yfinance
    btc_start = max(start, "2014-01-01")
    if verbose:
        print(f"  Fetching BTC-USD daily ({btc_start} → {end}) ...")

    try:
        btc_df = yf.download("BTC-USD", start=btc_start, end=end, interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"  BTC fetch error: {e}")
        for d in meta:
            meta[d]["btc_prev_return"] = None
        return meta

    def to_series_btc(df):
        c = df["Close"] if hasattr(df, "columns") else df
        if hasattr(c, "squeeze"):
            c = c.squeeze()
        idx = c.index
        if hasattr(idx, "tz") and idx.tz is not None:
            dates = [pd.to_datetime(t).tz_convert("US/Eastern").date() for t in idx]
        else:
            dates = [pd.to_datetime(t).date() for t in idx]
        return pd.Series(c.values.flatten().astype(float), index=dates)

    btc = to_series_btc(btc_df)
    btc_ret = btc.pct_change()

    dates_sorted = sorted(meta.keys())
    hits = 0
    for i, d in enumerate(dates_sorted):
        if i == 0:
            meta[d]["btc_prev_return"] = None
        else:
            prev_d = dates_sorted[i - 1]
            val = btc_ret.get(prev_d)
            if val is not None and not np.isnan(float(val)):
                meta[d]["btc_prev_return"] = float(val)
                hits += 1
            else:
                meta[d]["btc_prev_return"] = None

    if verbose:
        print(f"    BTC return populated for {hits}/{len(dates_sorted)} days")
    return meta


def _run_es_orb(
    bars: pd.DataFrame,
    day_meta: dict,
    filter_fn=None,
    label: str = "",
) -> dict:
    """Run ORB backtest on ES bars — simplified (no ORB rule object, pattern-based)."""
    # On ES we use a simplified ORB: track OR high/low from 9:30–10:04 ET
    # Signal: close above OR high → enter LONG on next bar
    atr_s    = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars = ATR_PERIOD + 1

    pos = None; trades = []; equity = STARTING_EQUITY; peak = STARTING_EQUITY
    max_dd = 0.0; cur_date = None; daily_pnl: dict = {}
    trades_today = 0; daily_loss = 0.0

    # Per-day OR high/low tracking
    or_high: dict[datetime.date, float] = {}
    or_low:  dict[datetime.date, float] = {}

    for i in range(min_bars, len(bars)):
        bar   = bars.iloc[i]
        bt_et = bars.index[i].tz_convert("US/Eastern")
        bdate = bt_et.date()
        bh, bm = bt_et.hour, bt_et.minute
        btime = bh * 60 + bm

        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = daily_loss
            daily_loss = 0.0; trades_today = 0
        cur_date = bdate

        meta    = day_meta.get(bdate, {})
        atr_now = atr_s.iloc[i]
        if np.isnan(atr_now) or atr_now <= 0:
            continue

        # Build OR (09:30 to 10:04 ET): include all bars whose bar-start <= 10:00
        # (5-min bars: 9:30, 9:35, ..., 10:00 — 7 bars total)
        OR_START = 9 * 60 + 30
        OR_END   = 10 * 60 + 0   # last bar start time included in OR window
        ENTRY_CUTOFF = 12 * 60

        if OR_START <= btime <= OR_END:
            if bdate not in or_high:
                or_high[bdate] = bar["high"]
                or_low[bdate]  = bar["low"]
            else:
                or_high[bdate] = max(or_high[bdate], bar["high"])
                or_low[bdate]  = min(or_low[bdate],  bar["low"])

        is_last    = (i + 1 >= len(bars)) or bars.index[i + 1].tz_convert("US/Eastern").date() != bdate
        sess_close = is_last or (bh == 15 and bm >= 55)

        if pos is not None:
            exited, exit_p, reason = _check_exit(pos, bar, i, sess_close)
            if exited:
                p = _calc_pnl(pos.entry_price, exit_p, pos.n_contracts)
                trades.append({"date": bdate, "entry": pos.entry_price, "exit": exit_p,
                               "pnl": p, "reason": reason})
                equity += p; daily_loss += p
                peak    = max(peak, equity)
                max_dd  = min(max_dd, equity - peak)
                pos = None

        can_enter = (
            pos is None and not sess_close
            and trades_today < MAX_TRADES_DAY
            and daily_loss > MAX_DAILY_LOSS
            and (equity - peak) > -DRAWDOWN_BUFFER
            and bdate in or_high           # OR was built
            and btime > OR_END             # past OR window
            and btime <= ENTRY_CUTOFF
        )

        if can_enter:
            should_trade = True
            if filter_fn is not None:
                should_trade = filter_fn(meta)

            if should_trade and bdate in or_high:
                orh = or_high[bdate]
                orl = or_low[bdate]
                or_range = orh - orl
                # Require minimum OR range (0.3x ATR)
                if or_range >= 0.3 * atr_now:
                    # Signal: bar close > OR high (confirmed breakout)
                    if bar["close"] > orh:
                        ep = _slip(bar["close"], 1, True)
                        sl = ep - SL_MULT * atr_now
                        pt = ep + PT_MULT * atr_now
                        pos = Pos(ep, i, sl, pt, i + TIME_STOP_BARS, N_CONTRACTS, atr_now)
                        trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = daily_loss

    wins  = [t for t in trades if t["pnl"] > 0]
    total = sum(t["pnl"] for t in trades)
    gp    = sum(t["pnl"] for t in wins)
    gl    = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    daily = pd.Series(daily_pnl)
    daily = daily[daily != 0]
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0.0

    return {
        "label":  label,
        "n":      len(trades),
        "wr":     len(wins) / max(len(trades), 1),
        "pnl":    total,
        "sharpe": sharpe,
        "dd":     max_dd,
        "trades": trades,
    }


def yearly_stats(r: dict) -> dict:
    """Aggregate results by year."""
    by_year: dict[int, list] = defaultdict(list)
    for t in r["trades"]:
        by_year[t["date"].year].append(t)
    out = {}
    for yr, ts in sorted(by_year.items()):
        wins = [t for t in ts if t["pnl"] > 0]
        out[yr] = {
            "n":  len(ts),
            "wr": len(wins) / max(len(ts), 1),
            "pnl": sum(t["pnl"] for t in ts),
        }
    return out


def print_row(label: str, r: dict, width: int = 46) -> None:
    print(f"  {label:<{width}} {r['n']:>5}  {r['wr']:>5.1%}  ${r['pnl']:>10,.0f}  {r['sharpe']:>7.2f}  ${r['dd']:>8,.0f}")


def print_yearly(label: str, r: dict) -> None:
    ys = yearly_stats(r)
    print(f"\n  [{label}]  Sharpe={r['sharpe']:.2f}  TotalPnL=${r['pnl']:,.0f}  N={r['n']}")
    print(f"  {'Year':>6}  {'N':>4}  {'WR':>5}  {'PnL':>12}  {'Cumul':>14}")
    print(f"  {'-'*50}")
    cum = 0.0
    for yr, s in sorted(ys.items()):
        cum += s["pnl"]
        print(f"  {yr:>6}  {s['n']:>4}  {s['wr']:>5.1%}  ${s['pnl']:>10,.0f}  ${cum:>12,.0f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2010-06-01", help="Start date")
    parser.add_argument("--end",   default="2025-12-31", help="End date")
    parser.add_argument("--years", default=None, type=int, help="Limit to last N years (e.g. --years 5)")
    parser.add_argument("--save",  action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"  ES data not found: {DATA_PATH}")
        print(f"  Expected: {DATA_PATH}")
        sys.exit(1)

    print(f"\n{'='*72}")
    print(f"  15-YEAR ES ORB FILTER SWEEP")
    print(f"{'='*72}")
    print(f"\nLoading ES 5-min bars ...")
    bars = pd.read_hdf(str(DATA_PATH), key="bars_5min")

    # ES data has timestamp as a column, not the index
    if "timestamp" in bars.columns:
        bars.index = pd.to_datetime(bars["timestamp"])
        bars = bars.drop(columns=["timestamp"])

    if not isinstance(bars.index, pd.DatetimeIndex):
        bars.index = pd.to_datetime(bars.index)
    if getattr(bars.index, "tz", None) is None:
        bars.index = bars.index.tz_localize("UTC")
    bars.index = bars.index.tz_convert("US/Eastern")

    # RTH only
    rth = (
        ((bars.index.hour > 9) | ((bars.index.hour == 9) & (bars.index.minute >= 30)))
        & (bars.index.hour < 16)
    )
    bars = bars.loc[rth]

    # Date range filter
    if args.years:
        cutoff = bars.index[-1] - pd.DateOffset(years=args.years)
        bars   = bars[bars.index >= cutoff]
        args.start = str(bars.index[0].date())

    print(f"  {len(bars):,} bars  {bars.index[0].date()} → {bars.index[-1].date()}")

    # Metadata
    print("\nBuilding day metadata ...")
    meta_start = (pd.Timestamp(args.start) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    meta_end   = (pd.Timestamp(args.end)   + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    meta = _build_day_meta_vectorized(bars)
    meta = _enrich_vix_vxn(meta, meta_start, meta_end)
    meta = _enrich_btc(meta, meta_start, meta_end)

    # Filters
    def vxn_filter_008(m):
        s = m.get("tech_fear_spread")
        return True if s is None else s < 0.08

    def vxn_filter_010(m):
        s = m.get("tech_fear_spread")
        return True if s is None else s < 0.10

    def btc_filter_neg002(m):
        r = m.get("btc_prev_return")
        return True if r is None else r > -0.02

    def btc_filter_0(m):
        r = m.get("btc_prev_return")
        return True if r is None else r > 0.0

    def combined_filter(m):
        return vxn_filter_008(m) and btc_filter_neg002(m)

    configs = [
        ("Baseline (no filter)",       None),
        ("VXN/VIX spread < 0.08",      vxn_filter_008),
        ("VXN/VIX spread < 0.10",      vxn_filter_010),
        ("BTC prev return > -0.02",    btc_filter_neg002),
        ("BTC prev return > 0.0",      btc_filter_0),
        ("VXN<0.08 + BTC>-0.02",       combined_filter),
    ]

    print(f"\n{'='*72}")
    print(f"  {'Config':<46} {'N':>5}  {'WR':>5}  {'PnL':>11}  {'Sharpe':>7}  {'MaxDD':>9}")
    print(f"  {'-'*72}")

    results = {}
    for lbl, fn in configs:
        print(f"  Running: {lbl} ...", end=" ", flush=True)
        r = _run_es_orb(bars, meta, filter_fn=fn, label=lbl)
        results[lbl] = r
        print_row(lbl, r)

    # Year-by-year breakdown for top configs
    print(f"\n{'='*72}")
    print(f"  YEARLY BREAKDOWN — Baseline vs Best Filter")
    print(f"{'='*72}")
    print_yearly("Baseline", results["Baseline (no filter)"])
    if "VXN/VIX spread < 0.08" in results:
        print_yearly("VXN/VIX spread < 0.08", results["VXN/VIX spread < 0.08"])
    if "VXN<0.08 + BTC>-0.02" in results:
        print_yearly("VXN<0.08 + BTC>-0.02", results["VXN<0.08 + BTC>-0.02"])

    # Summary
    print(f"\n{'='*72}")
    print(f"  INTERPRETATION")
    print(f"{'='*72}")
    base_wr = results["Baseline (no filter)"]["wr"]
    for lbl in ["VXN/VIX spread < 0.08", "BTC prev return > -0.02", "VXN<0.08 + BTC>-0.02"]:
        if lbl in results:
            r = results[lbl]
            wr_delta = r["wr"] - base_wr
            n_blocked = results["Baseline (no filter)"]["n"] - r["n"]
            print(f"  {lbl}")
            print(f"    WR delta: {wr_delta:+.1%}  |  Trades blocked: {n_blocked}  |  "
                  f"Sharpe: {r['sharpe']:.2f}  |  DD: ${r['dd']:,.0f}")

    # Save
    if args.save:
        serializable = {}
        for lbl, r in results.items():
            sv = {k: v for k, v in r.items() if k != "trades"}
            sv["yearly"] = {str(yr): s for yr, s in yearly_stats(r).items()}
            serializable[lbl] = sv
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(serializable, indent=2, default=str))
        print(f"\n  Results saved → {RESULTS_PATH}")
    else:
        print(f"\n  Tip: run with --save to write results to {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
