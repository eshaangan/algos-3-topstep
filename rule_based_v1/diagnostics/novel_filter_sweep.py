"""
Novel Cross-Asset Regime Filter Sweep — MNQ ORB 2026 YTD
=========================================================
Tests 3 novel regime filters that most retail ORB algos never use:

1. VXN/VIX Divergence — "Tech Fear Spread"
   - When VXN >> VIX, NQ is in tech-specific risk-off → skip longs
   - Standard retail ORB algos use VIX alone; VXN-VIX spread is niche

2. BTC Previous Day Return — "Global Risk Thermometer"
   - Yesterday BTC daily return as risk-on/off proxy
   - Negative BTC day → risk-off overnight → ORB longs may fade

3. Combined stacking with existing PrevVWAP filter
   - PrevVWAP + VXN/VIX + BTC for maximum drawdown reduction

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/novel_filter_sweep.py
    python rule_based_v1/diagnostics/novel_filter_sweep.py --no-vwap  # skip PrevVWAP combos

Data cost: FREE — VIX, VXN, BTC-USD via yfinance
"""
from __future__ import annotations

import argparse
import sys
import json
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

from engine.signal_aggregator import SignalAggregator
from rules.opening_range import OpeningRangeBreakoutRule
from utils.indicators import atr as compute_atr

DATA_PATH    = ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5"
RESULTS_PATH = ROOT / "rule_based_v1" / "diagnostics" / "novel_filter_results.json"

# ── Execution config (mirrors live config) ────────────────────────────────────
N_CONTRACTS    = 3
POINT_VALUE    = 2.0
TICK_SIZE      = 0.25
COMMISSION     = 0.62
SLIPPAGE_TICKS = 1
PT_MULT        = 3.0
SL_MULT        = 1.5
ATR_PERIOD     = 14
TIME_STOP_BARS = 24
MAX_DAILY_LOSS = -950.0
MAX_TRADES_DAY = 1
STARTING_EQUITY  = 50_000.0
DRAWDOWN_BUFFER  = 1_950.0


# ── Position dataclass ────────────────────────────────────────────────────────
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


# ── Day metadata builders ─────────────────────────────────────────────────────
def build_day_meta(bars: pd.DataFrame) -> dict:
    """Base per-day metadata: gap%, ATR, ATR percentile."""
    daily = bars.resample("1D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),   close=("close", "last"),
    ).dropna()
    if daily.index.tz is None:
        daily.index = daily.index.tz_localize("US/Eastern")

    daily["atr14"]   = compute_atr(daily["high"], daily["low"], daily["close"], 14)
    daily["gap_pct"] = (daily["open"] - daily["close"].shift(1)) / daily["close"].shift(1)

    atr_pct = pd.Series(np.nan, index=daily.index)
    for i in range(len(daily)):
        window = daily["atr14"].iloc[max(0, i - 59): i + 1].dropna()
        if len(window) >= 5:
            atr_pct.iloc[i] = (window < daily["atr14"].iloc[i]).mean() * 100
    daily["atr_pct"] = atr_pct

    meta = {}
    for ts, row in daily.iterrows():
        d = ts.tz_convert("US/Eastern").date()
        meta[d] = {
            "gap_pct": float(row["gap_pct"]) if not np.isnan(row["gap_pct"]) else 0.0,
            "atr14":   float(row["atr14"])   if not np.isnan(row["atr14"])   else 300.0,
            "atr_pct": float(row["atr_pct"]) if not np.isnan(row["atr_pct"]) else 50.0,
        }
    return meta


def enrich_prev_vwap(bars: pd.DataFrame, meta: dict) -> dict:
    """Add prev_vwap_bullish: True/False/None to each day in meta."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    cum_tp  = (typical * bars["volume"]).groupby(bars.index.date).cumsum()
    cum_vol = bars["volume"].groupby(bars.index.date).cumsum().replace(0, float("nan"))
    vwap    = cum_tp / cum_vol

    # Per-date: last vwap vs last close
    vwap_daily   = vwap.groupby(vwap.index.map(lambda t: t.date())).last()
    close_daily  = bars["close"].groupby(bars.index.map(lambda t: t.date())).last()

    dates_sorted = sorted(meta.keys())
    for i, d in enumerate(dates_sorted):
        if i == 0:
            meta[d]["prev_vwap_bullish"] = None
        else:
            prev_d = dates_sorted[i - 1]
            if prev_d in vwap_daily.index and prev_d in close_daily.index:
                meta[d]["prev_vwap_bullish"] = bool(
                    close_daily[prev_d] > vwap_daily[prev_d]
                )
            else:
                meta[d]["prev_vwap_bullish"] = None
    return meta


def enrich_vix_vxn(meta: dict, start: str, end: str) -> dict:
    """
    Add tech_fear_spread = (vxn/vix) - 1.0 for each day (uses previous session close).
    Requires: pip install yfinance
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  [WARN] yfinance not installed — skipping VIX/VXN enrichment")
        for d in meta:
            meta[d]["tech_fear_spread"] = None
        return meta

    print(f"  Fetching ^VIX and ^VXN ({start} → {end}) ...")
    try:
        vix_df = yf.download("^VIX",  start=start, end=end, interval="1d", progress=False, auto_adjust=True)
        vxn_df = yf.download("^VXN",  start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"  [WARN] yfinance download failed: {e}")
        for d in meta:
            meta[d]["tech_fear_spread"] = None
        return meta

    # Normalize to Series (yfinance may return DataFrame with MultiIndex columns)
    def _to_series(df_or_s, label):
        c = df_or_s["Close"] if hasattr(df_or_s, "columns") else df_or_s
        if hasattr(c, "squeeze"):
            c = c.squeeze()   # (N,1) DataFrame → Series
        idx = c.index
        if hasattr(idx, "tz") and idx.tz is not None:
            dates = pd.to_datetime(idx).tz_convert("US/Eastern")
            dates = [t.date() for t in dates]
        else:
            dates = [pd.to_datetime(t).date() for t in idx]
        return pd.Series(c.values.flatten(), index=dates, name=label)

    vix_s = _to_series(vix_df, "VIX")
    vxn_s = _to_series(vxn_df, "VXN")
    spread = (vxn_s / vix_s - 1.0)

    dates_sorted = sorted(meta.keys())
    hits = 0
    for i, d in enumerate(dates_sorted):
        if i == 0:
            meta[d]["tech_fear_spread"] = None
        else:
            prev_d = dates_sorted[i - 1]
            val = spread.get(prev_d)
            meta[d]["tech_fear_spread"] = float(val) if val is not None and not np.isnan(float(val)) else None
            if meta[d]["tech_fear_spread"] is not None:
                hits += 1

    print(f"    VIX/VXN spread populated for {hits}/{len(dates_sorted)} days")
    return meta


def enrich_btc(meta: dict, start: str, end: str) -> dict:
    """
    Add btc_prev_return = previous day BTC-USD close-to-close return.
    Uses yfinance daily data — covers full YTD range.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  [WARN] yfinance not installed — skipping BTC enrichment")
        for d in meta:
            meta[d]["btc_prev_return"] = None
        return meta

    print(f"  Fetching BTC-USD daily ({start} → {end}) ...")
    try:
        btc_df = yf.download("BTC-USD", start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"  [WARN] yfinance BTC download failed: {e}")
        for d in meta:
            meta[d]["btc_prev_return"] = None
        return meta

    def _to_series_btc(df_or_s):
        c = df_or_s["Close"] if hasattr(df_or_s, "columns") else df_or_s
        if hasattr(c, "squeeze"):
            c = c.squeeze()
        idx = c.index
        if hasattr(idx, "tz") and idx.tz is not None:
            dates = [pd.to_datetime(t).tz_convert("US/Eastern").date() for t in idx]
        else:
            dates = [pd.to_datetime(t).date() for t in idx]
        return pd.Series(c.values.flatten().astype(float), index=dates)

    btc_s   = _to_series_btc(btc_df)
    btc_ret = btc_s.pct_change()

    dates_sorted = sorted(meta.keys())
    hits = 0
    for i, d in enumerate(dates_sorted):
        if i == 0:
            meta[d]["btc_prev_return"] = None
        else:
            prev_d = dates_sorted[i - 1]
            val = btc_ret.get(prev_d)
            meta[d]["btc_prev_return"] = float(val) if val is not None and not np.isnan(float(val)) else None
            if meta[d]["btc_prev_return"] is not None:
                hits += 1

    print(f"    BTC prev-day return populated for {hits}/{len(dates_sorted)} days")
    return meta


# ── Core backtest loop ────────────────────────────────────────────────────────
def run_orb(bars: pd.DataFrame, day_meta: dict, filter_fn=None, label: str = "") -> dict:
    """Run ORB backtest. filter_fn(meta_dict) -> bool: return False to skip trade."""
    orb = OpeningRangeBreakoutRule(
        or_end_time="10:04", min_or_bars=7, min_range_atr=0.3,
        entry_cutoff_time="12:00", atr_period=ATR_PERIOD, long_only=True,
    )
    agg      = SignalAggregator(primary_rule=orb, filter_rules=[], confirmation_rules=[], min_confirmations=0)
    atr_s    = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars = agg.required_bars()

    pos = None; trades = []; equity = STARTING_EQUITY; peak = STARTING_EQUITY
    max_dd = 0.0; cur_date = None; daily_pnl: dict[datetime.date, float] = {}
    trades_today = 0; daily_loss = 0.0

    for i in range(min_bars, len(bars)):
        bar   = bars.iloc[i]
        bt_et = bars.index[i].tz_convert("US/Eastern")
        bdate = bt_et.date()
        bh, bm = bt_et.hour, bt_et.minute

        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = daily_loss
            daily_loss = 0.0; trades_today = 0
        cur_date = bdate

        meta    = day_meta.get(bdate, {})
        atr_now = atr_s.iloc[i]
        if np.isnan(atr_now) or atr_now <= 0:
            continue

        is_last    = (i + 1 >= len(bars)) or bars.index[i + 1].tz_convert("US/Eastern").date() != bdate
        sess_close = is_last or (bh == 15 and bm >= 55)

        if pos is not None:
            exited, exit_p, reason = _check_exit(pos, bar, i, sess_close)
            if exited:
                p = _calc_pnl(pos.entry_price, exit_p, pos.n_contracts)
                trades.append({
                    "date": bdate, "entry": pos.entry_price, "exit": exit_p,
                    "pnl": p, "reason": reason, "n": pos.n_contracts,
                    "tech_fear_spread":  meta.get("tech_fear_spread"),
                    "btc_prev_return":   meta.get("btc_prev_return"),
                    "prev_vwap_bullish": meta.get("prev_vwap_bullish"),
                })
                equity += p; daily_loss += p
                peak = max(peak, equity)
                max_dd = min(max_dd, equity - peak)
                pos = None

        can_enter = (
            pos is None and not sess_close
            and trades_today < MAX_TRADES_DAY
            and daily_loss > MAX_DAILY_LOSS
            and (equity - peak) > -DRAWDOWN_BUFFER
        )

        if can_enter:
            should_trade = True
            if filter_fn is not None:
                should_trade = filter_fn(meta)

            if should_trade:
                lookback = bars.iloc[max(0, i - min_bars + 1): i + 1]
                dec      = agg.evaluate(lookback)
                if dec.should_trade:
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
        "label":    label,
        "n":        len(trades),
        "wr":       len(wins) / max(len(trades), 1),
        "pnl":      total,
        "sharpe":   sharpe,
        "dd":       max_dd,
        "mll":      max_dd > -2000,
        "pf":       gp / gl if gl > 0 else float("inf"),
        "trades":   trades,
        "daily_pnl": dict(daily_pnl),
    }


# ── Output helpers ────────────────────────────────────────────────────────────
def print_row(label: str, r: dict, width: int = 48) -> None:
    mll = "✅" if r["mll"] else "❌"
    print(f"  {label:<{width}} {r['n']:>4}  {r['wr']:>5.1%} ${r['pnl']:>8,.0f} "
          f"{r['sharpe']:>7.2f} ${r['dd']:>7,.0f}  {mll}")


def monthly_wr(r: dict) -> dict:
    m = defaultdict(list)
    for t in r["trades"]:
        m[str(t["date"])[:7]].append(t)
    return {
        ym: (len(ts), sum(1 for t in ts if t["pnl"] > 0) / len(ts), sum(t["pnl"] for t in ts))
        for ym, ts in m.items()
    }


def print_monthly(label: str, r: dict) -> None:
    ms = monthly_wr(r)
    print(f"\n  [{label}]  Sharpe={r['sharpe']:.2f}  DD=${r['dd']:,.0f}  "
          f"PF={r['pf']:.2f}  MLL={'✅' if r['mll'] else '❌'}")
    print(f"  {'Month':<10} {'N':>4} {'WR':>6} {'PnL':>10} {'Cumul':>12}")
    print(f"  {'-'*46}")
    cum = 0.0
    for ym in sorted(ms):
        n, wr, pnl = ms[ym]; cum += pnl
        print(f"  {ym:<10} {n:>4}  {wr:>5.1%} ${pnl:>8,.0f}  ${cum:>10,.0f}")


# ── Filter definitions ────────────────────────────────────────────────────────
def make_vxn_filter(threshold: float):
    """Allow trade only when tech_fear_spread < threshold (VXN/VIX ratio)."""
    def fn(meta):
        spread = meta.get("tech_fear_spread")
        if spread is None:
            return True   # no data = don't filter
        return spread < threshold
    return fn


def make_btc_filter(threshold: float):
    """Allow trade only when btc_prev_return > threshold."""
    def fn(meta):
        ret = meta.get("btc_prev_return")
        if ret is None:
            return True   # no data = don't filter
        return ret > threshold
    return fn


def make_prev_vwap_filter():
    """Allow trade only when previous session close > VWAP."""
    def fn(meta):
        pv = meta.get("prev_vwap_bullish")
        if pv is False:
            return False
        return True
    return fn


def make_combined_filter(*filters):
    """All filters must pass (AND logic)."""
    def fn(meta):
        return all(f(meta) for f in filters)
    return fn


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-vwap",   action="store_true", help="Skip PrevVWAP combo configs")
    parser.add_argument("--save",      action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    print(f"\nLoading MNQ 2026 YTD bars from {DATA_PATH} ...")
    bars = pd.read_hdf(str(DATA_PATH), key="bars_5min")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    print(f"  {len(bars):,} 5-min bars  {bars.index[0].date()} → {bars.index[-1].date()}")

    start_str = str(bars.index[0].date() - datetime.timedelta(days=5))
    end_str   = str(bars.index[-1].date() + datetime.timedelta(days=1))

    # Build metadata
    print("\nBuilding day metadata ...")
    meta = build_day_meta(bars)
    meta = enrich_prev_vwap(bars, meta)
    meta = enrich_vix_vxn(meta, start_str, end_str)
    meta = enrich_btc(meta, start_str, end_str)

    # Print metadata sample
    print("\n  Sample day_meta (last 5 trading days):")
    for d in sorted(meta.keys())[-5:]:
        m = meta[d]
        vxn_s  = f"{m.get('tech_fear_spread', 'N/A'):.3f}" if isinstance(m.get("tech_fear_spread"), float) else "N/A"
        btc_r  = f"{m.get('btc_prev_return', 'N/A'):.3f}"  if isinstance(m.get("btc_prev_return"), float) else "N/A"
        pvwap  = m.get("prev_vwap_bullish", "N/A")
        print(f"    {d}  vxn_spread={vxn_s}  btc_prev={btc_r}  prev_vwap={pvwap}")

    # ── Run sweeps ────────────────────────────────────────────────────────────
    results = {}

    print(f"\n{'='*88}")
    print(f"  NOVEL CROSS-ASSET FILTER SWEEP — MNQ 2026 YTD  |  3c  |  LONG-only  |  PT=3x SL=1.5x")
    print(f"{'='*88}")
    print(f"  {'Config':<48} {'N':>4}  {'WR':>5}  {'PnL':>9}  {'Sharpe':>7} {'MaxDD':>8}  MLL")
    print(f"  {'-'*86}")

    # Baseline
    r = run_orb(bars, meta, label="Baseline (no filter)")
    results["baseline"] = r
    print_row("Baseline (no filter)", r)

    # PrevVWAP only (existing deployed filter)
    r = run_orb(bars, meta, filter_fn=make_prev_vwap_filter(), label="PrevVWAP only")
    results["prev_vwap"] = r
    print_row("PrevVWAP only  [DEPLOYED]", r)

    print(f"\n  --- Filter 1: VXN/VIX Tech Fear Spread ---")

    for thresh in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
        lbl = f"VXN/VIX spread < {thresh:.2f}"
        r = run_orb(bars, meta, filter_fn=make_vxn_filter(thresh), label=lbl)
        results[f"vxn_{thresh:.2f}"] = r
        print_row(lbl, r)

    print(f"\n  --- Filter 2: BTC Previous-Day Return ---")

    for thresh in [-0.05, -0.03, -0.02, -0.01, 0.0, 0.01]:
        lbl = f"BTC prev return > {thresh:.2f}"
        r = run_orb(bars, meta, filter_fn=make_btc_filter(thresh), label=lbl)
        results[f"btc_{thresh:.2f}"] = r
        print_row(lbl, r)

    if not args.no_vwap:
        print(f"\n  --- Filter 3: Combined PrevVWAP + Novel Filters ---")

        # VXN thresholds with PrevVWAP
        for thresh in [0.06, 0.08, 0.10, 0.12]:
            lbl = f"PrevVWAP + VXN < {thresh:.2f}"
            r = run_orb(
                bars, meta,
                filter_fn=make_combined_filter(make_prev_vwap_filter(), make_vxn_filter(thresh)),
                label=lbl,
            )
            results[f"pvwap_vxn_{thresh:.2f}"] = r
            print_row(lbl, r)

        # BTC thresholds with PrevVWAP
        for thresh in [-0.03, -0.01, 0.0]:
            lbl = f"PrevVWAP + BTC > {thresh:.2f}"
            r = run_orb(
                bars, meta,
                filter_fn=make_combined_filter(make_prev_vwap_filter(), make_btc_filter(thresh)),
                label=lbl,
            )
            results[f"pvwap_btc_{thresh:.2f}"] = r
            print_row(lbl, r)

        # Triple combo
        for vxn_t, btc_t in [(0.08, -0.02), (0.10, -0.01), (0.08, 0.0)]:
            lbl = f"PrevVWAP + VXN<{vxn_t:.2f} + BTC>{btc_t:.2f}"
            r = run_orb(
                bars, meta,
                filter_fn=make_combined_filter(
                    make_prev_vwap_filter(),
                    make_vxn_filter(vxn_t),
                    make_btc_filter(btc_t),
                ),
                label=lbl,
            )
            results[f"triple_{vxn_t:.2f}_{btc_t:.2f}"] = r
            print_row(lbl, r)

    # ── Summary: top MLL-safe configs by Sharpe ──────────────────────────────
    mll_safe = [(k, v) for k, v in results.items() if v["mll"] and v["n"] >= 10]
    mll_safe.sort(key=lambda x: x[1]["sharpe"], reverse=True)

    print(f"\n{'='*88}")
    print(f"  TOP 5 MLL-SAFE CONFIGS (N≥10, MaxDD > -$2,000) by Sharpe")
    print(f"{'='*88}")
    for k, r in mll_safe[:5]:
        print_row(r["label"], r)

    print(f"\n{'='*88}")
    print(f"  MONTHLY BREAKDOWN — Top 3 MLL-safe configs")
    print(f"{'='*88}")
    for k, r in mll_safe[:3]:
        print_monthly(r["label"], r)

    # ── Filter day count analysis ─────────────────────────────────────────────
    print(f"\n{'='*88}")
    print(f"  FILTER BLOCKING ANALYSIS (days blocked vs baseline trade days)")
    print(f"{'='*88}")

    baseline_dates = {t["date"] for t in results["baseline"]["trades"]}
    for key, fn, label in [
        ("prev_vwap",        make_prev_vwap_filter(),    "PrevVWAP"),
        ("vxn_0.08",         make_vxn_filter(0.08),      "VXN<0.08"),
        ("btc_-0.02",        make_btc_filter(-0.02),     "BTC>-0.02"),
    ]:
        blocked = sum(1 for d in baseline_dates if not fn(meta.get(d, {})))
        print(f"  {label:<20}  blocked {blocked}/{len(baseline_dates)} trade days  "
              f"({blocked/max(len(baseline_dates),1):.0%} reduction)")

    # ── Save results ──────────────────────────────────────────────────────────
    if args.save:
        serializable = {}
        for k, v in results.items():
            sv = {kk: vv for kk, vv in v.items() if kk not in ("trades", "daily_pnl")}
            sv["monthly"] = monthly_wr(v)
            sv["monthly"] = {m: list(vals) for m, vals in sv["monthly"].items()}
            serializable[k] = sv
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(serializable, indent=2, default=str))
        print(f"\n  Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
