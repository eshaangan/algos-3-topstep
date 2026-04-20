"""Combined Portfolio Backtest — ORB (3x MNQ) + GIRE (5x MNQ) + MSITE (12x MNQ)
===================================================================================
Runs each strategy on its native data, scales PnL to live contract sizes,
then combines into a portfolio equity curve.

Scaling reference:
  ORB   — backtest: 3x MNQ ($2/pt) = $6/pt                  → scale ×1.0
  MSITE — backtest: 2x MNQ ($2/pt) = $4/pt, live: 12x MNQ   → scale ×6.0
  GIRE  — backtest: 1x MES ($5/pt) = $5/pt, live: 5x MNQ    → scale ×2.0

Periods:
  YTD  — Jan 2026 – Mar 2026  (mnq_2026ytd_5min.h5)
  Long — Aug 2025 – Mar 2026  (mnq_5min_aug25_mar26.h5)
  GIRE historical uses mes_1m_bars_cache.h5 (Jun–Dec 2025, full ETH)

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/combined_portfolio_backtest.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT   = Path(__file__).resolve().parent.parent.parent
RBV1   = ROOT / "rule_based_v1"
DIAG   = ROOT / "rule_based_v1" / "diagnostics"
DATA   = ROOT / "data" / "processed"

for p in [str(ROOT), str(RBV1), str(DIAG)]:
    if p not in sys.path:
        sys.path.insert(0, p)

MNQ_YTD_5M   = DATA / "mnq_2026ytd_5min.h5"
MNQ_LONG_5M  = DATA / "mnq_5min_aug25_mar26.h5"
MES_1M_HIST  = DATA / "mes_1m_bars_cache.h5"

# ---------------------------------------------------------------------------
# Scale factors (PnL multipliers vs backtest basis)
# ---------------------------------------------------------------------------
ORB_SCALE   = 1.0   # 3x MNQ already in novel_filter_sweep
MSITE_SCALE = 6.0   # 2x→12x MNQ
GIRE_SCALE  = 2.0   # 1x MES ($5/pt) → 5x MNQ ($10/pt)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_mnq_5min(path: Path) -> pd.DataFrame:
    """Load RTH-only MNQ 5-min bars from HDF5."""
    with pd.HDFStore(str(path), "r") as s:
        df = s["/bars_5min"]
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    return df


def load_mes_1min_eth(path: Path) -> pd.DataFrame:
    """Load full-session MES 1-min bars, convert timestamp → DatetimeIndex."""
    with pd.HDFStore(str(path), "r") as s:
        df = s["/bars_1m"]
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("US/Eastern")
    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# ORB backtest (imports from novel_filter_sweep)
# ---------------------------------------------------------------------------

def run_orb_strategy(bars: pd.DataFrame, label: str) -> dict:
    """Run ORB with PrevVWAP filter, return trades dict."""
    from novel_filter_sweep import (
        run_orb, build_day_meta, enrich_prev_vwap,
    )
    day_meta = build_day_meta(bars)
    day_meta = enrich_prev_vwap(bars, day_meta)
    prev_vwap_filter = lambda m: m.get("prev_vwap_bullish") is True
    result = run_orb(bars, day_meta, filter_fn=prev_vwap_filter, label=label)
    return result


# ---------------------------------------------------------------------------
# MSITE backtest (imports from msite_backtest)
# ---------------------------------------------------------------------------

def run_msite_strategy(bars: pd.DataFrame, label: str) -> list[dict]:
    """Run MSITE (release-only, long-only), return trade list."""
    import msite_backtest as _mb
    _mb._apply_tf_config(5)   # CRITICAL: apply 5-min calibration before calling run_backtest
    msite_run = _mb.run_backtest
    trades, _ = msite_run(
        bars,
        allow_release=True,
        allow_exhaustion=False,
        long_only=True,
        afternoon_cutoff=(14, 0),
        pt_mult=2.0,
        sl_frac=0.55,
        time_stop_bars=12,
    )
    return trades


# ---------------------------------------------------------------------------
# GIRE backtest (imports from gire_backtest)
# ---------------------------------------------------------------------------

def run_gire_strategy(df: pd.DataFrame, label: str) -> list[dict]:
    """Run GIRE on 1-min full-session data, return trade list."""
    from gire_backtest import build_daily_data, run_backtest as gire_run
    daily = build_daily_data(df)
    trades = gire_run(
        daily,
        g_star=0.2,
        clv_thresh=0.8,
        ey_thresh=0.0,   # disabled
        st_thresh=0.0,   # disabled
        use_opening_failure=True,
        time_stop_hour=12,
    )
    return trades


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def extract_daily_pnl(trades: list[dict], date_key: str, pnl_key: str, scale: float) -> dict[str, float]:
    """Aggregate trade-level PnL → daily PnL dict (keyed by 'YYYY-MM-DD')."""
    daily: dict[str, float] = {}
    for t in trades:
        d = str(t[date_key])[:10]
        daily[d] = daily.get(d, 0.0) + t[pnl_key] * scale
    return daily


def portfolio_stats(
    orb_daily:   dict[str, float],
    msite_daily: dict[str, float],
    gire_daily:  dict[str, float],
    label: str,
) -> dict:
    """Combine three daily PnL dicts into portfolio statistics."""
    all_dates = sorted(
        set(orb_daily) | set(msite_daily) | set(gire_daily)
    )
    port_daily: dict[str, float] = {}
    for d in all_dates:
        port_daily[d] = (
            orb_daily.get(d, 0.0)
            + msite_daily.get(d, 0.0)
            + gire_daily.get(d, 0.0)
        )

    dpnls = np.array([port_daily[d] for d in all_dates], dtype=float)
    total = float(dpnls.sum())
    mean_d = float(dpnls.mean()) if len(dpnls) > 0 else 0.0
    std_d  = float(dpnls.std(ddof=1)) if len(dpnls) > 1 else 1e-9
    sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 1e-9 else 0.0

    equity = np.cumsum(dpnls)
    peak   = np.maximum.accumulate(equity)
    max_dd = float((equity - peak).min()) if len(equity) > 0 else 0.0

    # Win rate at daily level (days where port PnL > 0)
    pos_days = int((dpnls > 0).sum())
    neg_days = int((dpnls < 0).sum())
    flat_days = int((dpnls == 0).sum())

    return {
        "label":      label,
        "n_days":     len(all_dates),
        "pos_days":   pos_days,
        "neg_days":   neg_days,
        "flat_days":  flat_days,
        "day_wr":     pos_days / (pos_days + neg_days) if (pos_days + neg_days) > 0 else 0.0,
        "total_pnl":  round(total, 2),
        "avg_daily":  round(mean_d, 2),
        "sharpe":     round(sharpe, 3),
        "max_dd":     round(max_dd, 2),
        "port_daily": port_daily,
    }


def strategy_stats(daily_pnl: dict[str, float], name: str) -> dict:
    """Single-strategy daily PnL → stats."""
    if not daily_pnl:
        return {"name": name, "n_days": 0, "total_pnl": 0.0, "sharpe": 0.0, "max_dd": 0.0, "day_wr": 0.0}
    dpnls  = np.array(list(daily_pnl.values()), dtype=float)
    active = dpnls[dpnls != 0]
    total  = float(dpnls.sum())
    mean_d = float(active.mean()) if len(active) > 0 else 0.0
    std_d  = float(active.std(ddof=1)) if len(active) > 1 else 1e-9
    sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 1e-9 else 0.0
    equity = np.cumsum(list(daily_pnl.values()))
    peak   = np.maximum.accumulate(equity)
    max_dd = float((equity - peak).min()) if len(equity) > 0 else 0.0
    pos = int((active > 0).sum())
    neg = int((active < 0).sum())
    return {
        "name":      name,
        "n_days":    int((active != 0).sum()),
        "total_pnl": round(total, 2),
        "avg_daily": round(mean_d, 2),
        "sharpe":    round(sharpe, 3),
        "max_dd":    round(max_dd, 2),
        "day_wr":    round(pos / (pos + neg), 3) if (pos + neg) > 0 else 0.0,
    }


def monthly_breakdown(
    orb_d: dict, msite_d: dict, gire_d: dict, label: str
) -> None:
    """Print monthly PnL table per strategy + portfolio."""
    all_dates = sorted(set(orb_d) | set(msite_d) | set(gire_d))
    months: dict[str, dict] = defaultdict(lambda: {"ORB": 0.0, "MSITE": 0.0, "GIRE": 0.0, "PORT": 0.0})
    for d in all_dates:
        ym = d[:7]
        months[ym]["ORB"]   += orb_d.get(d, 0.0)
        months[ym]["MSITE"] += msite_d.get(d, 0.0)
        months[ym]["GIRE"]  += gire_d.get(d, 0.0)
        months[ym]["PORT"]  += orb_d.get(d, 0.0) + msite_d.get(d, 0.0) + gire_d.get(d, 0.0)

    print(f"\n  [{label}] Monthly PnL Breakdown")
    print(f"  {'Month':<10} {'ORB':>9} {'MSITE':>9} {'GIRE':>9} {'PORTFOLIO':>12} {'Cumul':>12}")
    print(f"  {'-'*63}")
    cum = 0.0
    for ym in sorted(months):
        r = months[ym]
        cum += r["PORT"]
        print(f"  {ym:<10} ${r['ORB']:>7,.0f} ${r['MSITE']:>7,.0f} ${r['GIRE']:>7,.0f} "
              f"  ${r['PORT']:>9,.0f}   ${cum:>9,.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  Combined Portfolio Backtest: ORB (3x) + GIRE (5x) + MSITE (12x) MNQ")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. YTD period (Jan–Mar 2026)
    # -----------------------------------------------------------------------
    print("\n[1] Loading YTD data (2026 YTD, MNQ 5-min) ...")
    mnq_ytd = load_mnq_5min(MNQ_YTD_5M)
    print(f"    {len(mnq_ytd):,} bars  {mnq_ytd.index[0].date()} → {mnq_ytd.index[-1].date()}")

    print("    Running ORB (PrevVWAP filter) ...")
    orb_ytd_result = run_orb_strategy(mnq_ytd, "ORB-YTD")
    orb_ytd_trades = orb_ytd_result["trades"]
    orb_ytd_daily  = extract_daily_pnl(orb_ytd_trades, "date", "pnl", ORB_SCALE)

    print("    Running MSITE (long-only, release-only) ...")
    msite_ytd_trades = run_msite_strategy(mnq_ytd, "MSITE-YTD")
    msite_ytd_daily  = extract_daily_pnl(msite_ytd_trades, "date", "pnl", MSITE_SCALE)

    # GIRE YTD: only 1 trade known (Jan 28, +$642 at 5x MNQ from live deployment)
    # YTD data is RTH-only so build_daily_data's ETH check fails; inject manually
    print("    GIRE YTD: 1 known trade (Jan 28 SHORT +$321 @ 1x MES → ×2 = $642)")
    gire_ytd_daily = {"2026-01-28": 321.0 * GIRE_SCALE}  # $321 was the 1x MES basis

    port_ytd = portfolio_stats(orb_ytd_daily, msite_ytd_daily, gire_ytd_daily, "YTD 2026")

    # -----------------------------------------------------------------------
    # 2. Long-term period (Aug 2025–Mar 2026)
    # -----------------------------------------------------------------------
    print("\n[2] Loading long-term data (Aug 2025–Mar 2026, MNQ 5-min) ...")
    mnq_long = load_mnq_5min(MNQ_LONG_5M)
    print(f"    {len(mnq_long):,} bars  {mnq_long.index[0].date()} → {mnq_long.index[-1].date()}")

    print("    Running ORB (PrevVWAP filter) ...")
    orb_long_result = run_orb_strategy(mnq_long, "ORB-Long")
    orb_long_trades = orb_long_result["trades"]
    orb_long_daily  = extract_daily_pnl(orb_long_trades, "date", "pnl", ORB_SCALE)

    print("    Running MSITE (long-only, release-only) ...")
    msite_long_trades = run_msite_strategy(mnq_long, "MSITE-Long")
    msite_long_daily  = extract_daily_pnl(msite_long_trades, "date", "pnl", MSITE_SCALE)

    # GIRE on MES 1-min historical (Jun–Dec 2025)
    print("    Loading MES 1-min data for GIRE (Jun–Dec 2025) ...")
    try:
        mes_1m = load_mes_1min_eth(MES_1M_HIST)
        # Trim to Aug 2025 to match ORB/MSITE start (optional: keeps overlap clean)
        mes_aug = mes_1m[mes_1m.index >= "2025-08-01"]
        print(f"    {len(mes_aug):,} bars  {mes_aug.index[0].date()} → {mes_aug.index[-1].date()}")
        print("    Running GIRE ...")
        gire_long_trades = run_gire_strategy(mes_aug, "GIRE-Long")
        gire_long_daily  = extract_daily_pnl(gire_long_trades, "date", "pnl_usd", GIRE_SCALE)
        # Inject YTD known trade
        gire_long_daily["2026-01-28"] = gire_long_daily.get("2026-01-28", 0.0) + 321.0 * GIRE_SCALE
        print(f"    GIRE: {len(gire_long_trades)} historical trades + 1 YTD trade")
    except Exception as e:
        print(f"    [WARN] GIRE historical failed: {e} — using empty")
        gire_long_daily = {"2026-01-28": 321.0 * GIRE_SCALE}
        gire_long_trades = []

    port_long = portfolio_stats(orb_long_daily, msite_long_daily, gire_long_daily, "Long Aug25–Mar26")

    # -----------------------------------------------------------------------
    # 3. Print results
    # -----------------------------------------------------------------------
    def _pct(x): return f"{x:.1%}"

    for period_label, port, orb_d, msite_d, gire_d, orb_trades, msite_trades, gire_trades in [
        ("YTD 2026",        port_ytd,  orb_ytd_daily,  msite_ytd_daily,  gire_ytd_daily,
         orb_ytd_trades,  msite_ytd_trades,  []),
        ("Long Aug25-Mar26", port_long, orb_long_daily, msite_long_daily, gire_long_daily,
         orb_long_trades, msite_long_trades, gire_long_trades),
    ]:
        print(f"\n{'='*70}")
        print(f"  {period_label}")
        print(f"{'='*70}")

        # Per-strategy trade stats
        orb_n   = len(orb_trades)
        orb_wr  = sum(1 for t in orb_trades if t["pnl"] > 0) / max(orb_n, 1)
        orb_pnl = sum(t["pnl"] for t in orb_trades) * ORB_SCALE
        msite_n   = len(msite_trades)
        msite_wr  = sum(1 for t in msite_trades if t["pnl"] > 0) / max(msite_n, 1)
        msite_pnl = sum(t["pnl"] for t in msite_trades) * MSITE_SCALE
        gire_n   = len(gire_trades)
        gire_wr  = sum(1 for t in gire_trades if t.get("pnl_usd", 0) > 0) / max(gire_n, 1)
        gire_pnl = sum(t.get("pnl_usd", 0) for t in gire_trades) * GIRE_SCALE

        print(f"\n  Strategy  |  N  |  WR   | Trade PnL (scaled) | Daily Sharpe | Max DD")
        print(f"  {'-'*68}")
        orb_s   = strategy_stats(orb_d,   "ORB (3x MNQ)")
        msite_s = strategy_stats(msite_d, "MSITE (12x MNQ)")
        gire_s  = strategy_stats(gire_d,  "GIRE (5x MNQ)")
        for st, n, wr, pnl, s in [
            (orb_s,   orb_n,   orb_wr,   orb_pnl,   "ORB   (3x MNQ)"),
            (msite_s, msite_n, msite_wr, msite_pnl, "MSITE (12x MNQ)"),
            (gire_s,  gire_n,  gire_wr,  gire_pnl,  "GIRE  (5x MNQ)"),
        ]:
            print(f"  {s:<16} {n:>4}  {wr:>5.1%}  ${pnl:>10,.0f}         "
                  f"{st['sharpe']:>6.2f}   ${st['max_dd']:>8,.0f}")

        print(f"\n  Portfolio Summary")
        print(f"  {'─'*50}")
        print(f"  Total PnL    : ${port['total_pnl']:>10,.0f}")
        print(f"  Daily Sharpe : {port['sharpe']:>8.3f}")
        print(f"  Max Drawdown : ${port['max_dd']:>10,.0f}")
        print(f"  Day WR       : {_pct(port['day_wr']):>8}  ({port['pos_days']}↑ / {port['neg_days']}↓ / {port['flat_days']} flat)")
        print(f"  Active days  : {port['n_days']}")

        monthly_breakdown(orb_d, msite_d, gire_d, period_label)

    # -----------------------------------------------------------------------
    # 4. Save results
    # -----------------------------------------------------------------------
    results = {
        "ytd_2026": {
            "period": "Jan 2026 – Mar 2026",
            "portfolio": {k: v for k, v in port_ytd.items() if k != "port_daily"},
            "strategies": {
                "orb":   {"n": len(orb_ytd_trades), "wr": round(sum(1 for t in orb_ytd_trades if t["pnl"]>0)/max(len(orb_ytd_trades),1),3),
                           "total_pnl_scaled": round(sum(t["pnl"] for t in orb_ytd_trades)*ORB_SCALE, 2)},
                "msite": {"n": len(msite_ytd_trades), "wr": round(sum(1 for t in msite_ytd_trades if t["pnl"]>0)/max(len(msite_ytd_trades),1),3),
                           "total_pnl_scaled": round(sum(t["pnl"] for t in msite_ytd_trades)*MSITE_SCALE, 2)},
                "gire":  {"n": 1, "wr": 1.0, "total_pnl_scaled": 642.0, "note": "1 known trade Jan 28 (RTH-only data)"},
            },
        },
        "long_aug25_mar26": {
            "period": "Aug 2025 – Mar 2026",
            "portfolio": {k: v for k, v in port_long.items() if k != "port_daily"},
            "strategies": {
                "orb":   {"n": len(orb_long_trades), "wr": round(sum(1 for t in orb_long_trades if t["pnl"]>0)/max(len(orb_long_trades),1),3),
                           "total_pnl_scaled": round(sum(t["pnl"] for t in orb_long_trades)*ORB_SCALE, 2)},
                "msite": {"n": len(msite_long_trades), "wr": round(sum(1 for t in msite_long_trades if t["pnl"]>0)/max(len(msite_long_trades),1),3),
                           "total_pnl_scaled": round(sum(t["pnl"] for t in msite_long_trades)*MSITE_SCALE, 2)},
                "gire":  {"n": len(gire_long_trades)+1, "wr": round(gire_wr,3),
                           "total_pnl_scaled": round(sum(t.get("pnl_usd",0) for t in gire_long_trades)*GIRE_SCALE + 642.0, 2)},
            },
        },
    }

    out_path = DIAG / "combined_portfolio_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
