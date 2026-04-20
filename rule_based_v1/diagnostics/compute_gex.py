"""
QQQ Gamma Exposure (GEX) Computation — "The Jackpot Filter"
============================================================
Computes net dealer Gamma Exposure from QQQ's options chain.

Theory:
  - Dealers are typically SHORT calls (sold to retail buyers) and LONG puts.
  - GEX = sum(gamma × OI × 100 × spot) for calls  —  sum(gamma × OI × 100 × spot) for puts
  - GEX > 0: Dealers are net LONG gamma → they SELL into strength, BUY weakness → market PIN
  - GEX < 0: Dealers are net SHORT gamma → they BUY into strength, SELL weakness → EXPLOSIVE moves
  - Negative GEX = ideal regime for ORB breakouts hitting 3x ATR targets

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/compute_gex.py               # today's GEX
    python rule_based_v1/diagnostics/compute_gex.py --hist        # build historical GEX from cached data
    python rule_based_v1/diagnostics/compute_gex.py --backtest    # run GEX-gated ORB backtest

Note on historical GEX:
  yfinance options OI is *current snapshot only* — it cannot reconstruct past OI.
  For historical analysis we use a proxy: QQQ front-month IV level as GEX approximation.
  High IV + rising market → dealers typically short gamma (negative GEX regime).
  This is an approximation; true historical GEX requires paid data (OptionMetrics, CBOE).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

GEX_CACHE_PATH = ROOT / "data" / "processed" / "qqq_gex_daily.csv"


# ── Black-Scholes gamma ───────────────────────────────────────────────────────
def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma for a European option."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    from math import log, sqrt, exp, pi
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return exp(-0.5 * d1 ** 2) / (S * sigma * sqrt(T) * sqrt(2 * pi))


def compute_gex_today(ticker: str = "QQQ", verbose: bool = True) -> dict:
    """
    Compute net dealer GEX from today's live options chain.
    Returns dict with gex, regime, spot, and per-expiry breakdown.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed. Run: pip install yfinance")
        return {}

    tkr  = yf.Ticker(ticker)
    spot = tkr.history(period="1d")["Close"].iloc[-1]

    if verbose:
        print(f"\n  {ticker} spot: ${spot:.2f}")
        print(f"  Computing GEX from options chain ...")

    exps = tkr.options
    if not exps:
        print("  No options expirations found.")
        return {}

    # Use front 3 expirations (most gamma is here)
    use_exps = exps[:3]
    total_gex = 0.0
    breakdown = {}
    r = 0.045   # risk-free rate approximation

    for exp in use_exps:
        try:
            chain = tkr.option_chain(exp)
        except Exception as e:
            if verbose:
                print(f"    {exp}: fetch error ({e})")
            continue

        exp_ts = pd.Timestamp(exp)
        today  = pd.Timestamp.now().normalize()
        T = (exp_ts - today).days / 365.0
        if T <= 0:
            T = 0.5 / 365.0  # same-day expiry

        calls = chain.calls[["strike", "openInterest", "impliedVolatility"]].copy()
        puts  = chain.puts [["strike", "openInterest", "impliedVolatility"]].copy()
        calls["oi"] = calls["openInterest"].fillna(0).astype(int)
        puts["oi"]  = puts["openInterest"].fillna(0).astype(int)

        # Focus on near-the-money strikes (±15%)
        calls = calls[(calls["strike"] >= spot * 0.85) & (calls["strike"] <= spot * 1.15)]
        puts  = puts [(puts ["strike"] >= spot * 0.85) & (puts ["strike"] <= spot * 1.15)]

        call_gex = 0.0
        for _, row in calls.iterrows():
            g = bs_gamma(spot, row["strike"], T, r, max(row["impliedVolatility"], 0.01))
            call_gex += g * row["oi"] * 100 * spot

        put_gex = 0.0
        for _, row in puts.iterrows():
            g = bs_gamma(spot, row["strike"], T, r, max(row["impliedVolatility"], 0.01))
            put_gex += g * row["oi"] * 100 * spot

        # Dealer convention: short calls (negative call GEX for dealer) + long puts (positive put GEX for dealer)
        # Net dealer GEX = call_gex - put_gex  (positive = dealer long gamma = pinned)
        net = call_gex - put_gex
        total_gex += net
        breakdown[exp] = {"call_gex": call_gex, "put_gex": put_gex, "net": net}

        if verbose:
            print(f"    {exp}  T={T*365:.1f}d  call_gex=${call_gex/1e6:.1f}M  "
                  f"put_gex=${put_gex/1e6:.1f}M  net=${net/1e6:.1f}M")

    regime = "NEGATIVE (explosive ✓)" if total_gex < 0 else "POSITIVE (pinned ✗)"
    result = {
        "ticker":    ticker,
        "spot":      float(spot),
        "gex":       total_gex,
        "regime":    regime,
        "explosive": total_gex < 0,
        "breakdown": breakdown,
    }

    if verbose:
        print(f"\n  ┌─ GEX Summary ──────────────────────────────────┐")
        print(f"  │  {ticker} spot:    ${spot:.2f}")
        print(f"  │  Net dealer GEX: ${total_gex/1e9:.3f}B")
        print(f"  │  Regime:         {regime}")
        print(f"  │  ORB signal:     {'TRADE (explosive regime)' if total_gex < 0 else 'SKIP  (pinned regime)'}")
        print(f"  └────────────────────────────────────────────────┘")

    return result


def compute_gex_proxy_historical(start: str, end: str, verbose: bool = True) -> pd.Series:
    """
    Build a historical GEX PROXY using QQQ IV levels (VXN as IV proxy).
    Since yfinance can't give past OI, we use:
      - IV spike + QQQ down day → dealers likely short gamma (GEX < 0 → explosive)
      - Low IV + calm market    → dealers likely long gamma  (GEX > 0 → pinned)

    Proxy formula: gex_proxy = -(vxn_z_score) where high VXN z-score → negative GEX
    This is a rough approximation for backtesting only.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed.")
        return pd.Series(dtype=float)

    if verbose:
        print(f"  Fetching QQQ and ^VXN for GEX proxy ({start} → {end}) ...")

    qqq_df = yf.download("QQQ", start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    vxn_df = yf.download("^VXN", start=start, end=end, interval="1d", progress=False, auto_adjust=True)

    if qqq_df.empty or vxn_df.empty:
        print("  Failed to fetch proxy data.")
        return pd.Series(dtype=float)

    def _squeeze(df, col="Close"):
        s = df[col] if hasattr(df, "columns") else df
        if hasattr(s, "squeeze"):
            s = s.squeeze()
        return pd.Series(s.values.flatten().astype(float), index=pd.to_datetime(s.index).normalize())

    qqq_ret = _squeeze(qqq_df).pct_change()
    vxn_lvl = _squeeze(vxn_df)

    # Align on common dates
    common = qqq_ret.index.intersection(vxn_lvl.index)
    qqq_ret = qqq_ret.loc[common]
    vxn_lvl = vxn_lvl.loc[common]

    # Rolling 60-day VXN z-score
    vxn_roll_mean = vxn_lvl.rolling(60, min_periods=20).mean()
    vxn_roll_std  = vxn_lvl.rolling(60, min_periods=20).std()
    vxn_z         = (vxn_lvl - vxn_roll_mean) / vxn_roll_std.replace(0, float("nan"))

    # GEX proxy: negative when VXN elevated (IV spike → dealers short gamma → explosive)
    # Shift by 1: use previous day's reading for today's trade decision
    gex_proxy = -vxn_z.shift(1)

    # Normalize dates
    if hasattr(gex_proxy.index, "tz") and gex_proxy.index.tz is not None:
        gex_proxy.index = pd.to_datetime(gex_proxy.index).tz_convert("US/Eastern").normalize()
    else:
        gex_proxy.index = pd.to_datetime(gex_proxy.index).normalize()

    # Drop NaN and convert index to date objects
    gex_proxy = gex_proxy.dropna()
    gex_by_date = {}
    for ts, val in gex_proxy.items():
        d = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
        gex_by_date[d] = float(val)
    gex_proxy_dated = pd.Series(gex_by_date)

    if verbose:
        neg_days = int((gex_proxy_dated < 0).sum())
        pos_days = int((gex_proxy_dated >= 0).sum())
        mn = float(gex_proxy_dated.min())
        mx = float(gex_proxy_dated.max())
        print(f"    GEX proxy: {neg_days} negative days (explosive), {pos_days} positive days (pinned)")
        print(f"    Range: {mn:.2f} to {mx:.2f}")

    return gex_proxy_dated


def run_gex_backtest(verbose: bool = True) -> None:
    """
    Run GEX-gated ORB backtest using historical VXN-based GEX proxy.
    Requires mnq_2026ytd_5min.h5 for MNQ data.
    """
    DATA_PATH = ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5"
    if not DATA_PATH.exists():
        print(f"  Data not found: {DATA_PATH}")
        return

    bars = pd.read_hdf(str(DATA_PATH), key="bars_5min")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")

    start_str = str(bars.index[0].date() - pd.Timedelta(days=90))
    end_str   = str(bars.index[-1].date() + pd.Timedelta(days=1))

    gex_proxy = compute_gex_proxy_historical(start_str, end_str, verbose=verbose)
    if gex_proxy is None or len(gex_proxy) == 0:
        return

    # gex_proxy index is already date objects (from compute_gex_proxy_historical)
    gex_by_date: dict = {d: float(v) for d, v in gex_proxy.items() if pd.notna(v)}

    # Import run_orb from novel_filter_sweep
    DIAG = ROOT / "rule_based_v1" / "diagnostics"
    if str(DIAG) not in sys.path:
        sys.path.insert(0, str(DIAG))
    from novel_filter_sweep import run_orb, build_day_meta, enrich_prev_vwap, print_row, print_monthly

    meta = build_day_meta(bars)
    meta = enrich_prev_vwap(bars, meta)
    # Inject GEX proxy into meta
    for d in meta:
        meta[d]["gex_proxy"] = gex_by_date.get(d)

    def gex_filter(m):
        """Allow trade when GEX proxy is negative (explosive regime)."""
        gp = m.get("gex_proxy")
        if gp is None:
            return True
        return gp < 0   # negative proxy = negative GEX = explosive

    def gex_pos_filter(m):
        """Control: only trade positive GEX (pinned — should be worse)."""
        gp = m.get("gex_proxy")
        if gp is None:
            return True
        return gp >= 0

    def pvwap_filter(m):
        pv = m.get("prev_vwap_bullish")
        return pv is not False

    print(f"\n{'='*80}")
    print(f"  GEX PROXY BACKTEST — MNQ 2026 YTD  |  3c  |  LONG-only")
    print(f"{'='*80}")
    print(f"  GEX proxy: -VXN_zscore (high IV → negative GEX → explosive regime)")
    print(f"  {'Config':<46} {'N':>4}  {'WR':>5}  {'PnL':>9}  {'Sharpe':>7} {'MaxDD':>8}  MLL")
    print(f"  {'-'*80}")

    configs = [
        ("Baseline (no filter)", None),
        ("PrevVWAP only [DEPLOYED]", pvwap_filter),
        ("GEX proxy < 0 (explosive only)", gex_filter),
        ("GEX proxy >= 0 (pinned only — control)", gex_pos_filter),
        ("PrevVWAP + GEX < 0", lambda m: pvwap_filter(m) and gex_filter(m)),
    ]

    results = []
    for lbl, fn in configs:
        r = run_orb(bars, meta, filter_fn=fn, label=lbl)
        results.append(r)
        print_row(lbl, r)

    print(f"\n  GEX negative days = explosive regime (dealers short gamma → amplified moves)")
    print(f"  If GEX<0 config beats baseline: consider adding to live filter pipeline")

    # Monthly breakdown of best GEX config
    gex_r = results[2]
    if gex_r["n"] >= 8:
        print_monthly("GEX proxy < 0", gex_r)

    # Save GEX proxy values
    proxy_df = pd.DataFrame([
        {"date": str(d), "gex_proxy": v}
        for d, v in sorted(gex_by_date.items())
    ])
    GEX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    proxy_df.to_csv(str(GEX_CACHE_PATH), index=False)
    print(f"\n  GEX proxy saved → {GEX_CACHE_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker",   default="QQQ",    help="Options ticker (default: QQQ)")
    parser.add_argument("--hist",     action="store_true", help="Build historical GEX proxy only")
    parser.add_argument("--backtest", action="store_true", help="Run GEX-gated ORB backtest")
    args = parser.parse_args()

    if args.backtest:
        run_gex_backtest()
        return

    if args.hist:
        proxy = compute_gex_proxy_historical("2025-10-01", "2026-03-25")
        print(proxy.tail(20).to_string())
        return

    # Default: today's live GEX
    compute_gex_today(ticker=args.ticker)


if __name__ == "__main__":
    main()
