"""Scaling Optimization Sweep — ORB + MSITE + GIRE Portfolio
=============================================================
Analyzes contract scaling safety and parameter optimization across three
live strategies on the Topstep 50k Combine.

Sections:
  1. Contract Scaling Analysis (per-strategy sweep)
  2. Portfolio Joint Scaling (grid search with Monte Carlo P(pass))
  3. Parameter Optimization (ORB PT/SL, MSITE pt/sl_frac, GIRE clv/g_star)

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/scaling_optimization_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
DIAG = ROOT / "rule_based_v1" / "diagnostics"

for p in [str(ROOT), str(RBV1), str(DIAG)]:
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS_PATH = DIAG / "scaling_optimization_results.json"

# ---------------------------------------------------------------------------
# Topstep 50k constraints
# ---------------------------------------------------------------------------
TRAIL_DD_LIMIT  = 2000.0   # trailing max drawdown
DAILY_LOSS_LIMIT = 1000.0  # per day
PROFIT_TARGET   = 3000.0   # to pass combine
CONSISTENCY_FLAG = 1200.0  # single trade > this = flag
DD_BUFFER       = 1800.0   # safe threshold (buffer below $2000)
STARTING_EQUITY = 50_000.0

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    print("Loading data ...")

    bars_long = pd.read_hdf(
        str(ROOT / "data" / "processed" / "mnq_5min_aug25_mar26.h5"),
        "/bars_5min",
    )
    if bars_long.index.tz is None:
        bars_long.index = bars_long.index.tz_localize("US/Eastern")

    bars_ytd = pd.read_hdf(
        str(ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5"),
        "/bars_5min",
    )
    if bars_ytd.index.tz is None:
        bars_ytd.index = bars_ytd.index.tz_localize("US/Eastern")

    with pd.HDFStore(str(ROOT / "data" / "processed" / "mes_1m_bars_cache.h5"), "r") as s:
        mes1m = s["/bars_1m"].set_index("timestamp")
    mes1m.index = pd.to_datetime(mes1m.index, utc=True).tz_convert("US/Eastern")
    mes_aug = mes1m[mes1m.index >= "2025-08-01"].sort_index()

    print(f"  bars_long : {len(bars_long):,} bars  "
          f"{bars_long.index[0].date()} -> {bars_long.index[-1].date()}")
    print(f"  bars_ytd  : {len(bars_ytd):,} bars  "
          f"{bars_ytd.index[0].date()} -> {bars_ytd.index[-1].date()}")
    print(f"  mes_aug   : {len(mes_aug):,} 1-min bars  "
          f"{mes_aug.index[0].date()} -> {mes_aug.index[-1].date()}")

    return bars_long, bars_ytd, mes_aug


# ---------------------------------------------------------------------------
# Base strategy runners
# ---------------------------------------------------------------------------

def run_base_strategies(bars_long: pd.DataFrame, mes_aug: pd.DataFrame):
    """Run each strategy once at base scale and return raw trade lists."""
    from novel_filter_sweep import run_orb, build_day_meta, enrich_prev_vwap
    import msite_backtest as mb
    from gire_backtest import build_daily_data, run_backtest as gire_run

    print("\nRunning ORB (3x MNQ, PrevVWAP filter) ...")
    orb_meta = enrich_prev_vwap(bars_long, build_day_meta(bars_long))
    orb_result = run_orb(
        bars_long, orb_meta,
        filter_fn=lambda m: m.get("prev_vwap_bullish") is True,
    )
    orb_trades = orb_result["trades"]
    print(f"  ORB base: {len(orb_trades)} trades, "
          f"PnL=${sum(t['pnl'] for t in orb_trades):,.0f}")

    print("\nRunning MSITE (2x MNQ base) ...")
    mb._apply_tf_config(5)
    msite_trades, _ = mb.run_backtest(
        bars_long,
        allow_release=True,
        allow_exhaustion=False,
        long_only=True,
        afternoon_cutoff=(14, 0),
        pt_mult=2.0,
        sl_frac=0.55,
        time_stop_bars=12,
    )
    print(f"  MSITE base: {len(msite_trades)} trades, "
          f"PnL=${sum(t['pnl'] for t in msite_trades):,.0f}")

    print("\nRunning GIRE (1x MES base) ...")
    gire_daily_data = build_daily_data(mes_aug)
    gire_trades = gire_run(
        gire_daily_data,
        g_star=0.2,
        clv_thresh=0.8,
        ey_thresh=0.0,
        st_thresh=0.0,
        use_opening_failure=True,
        time_stop_hour=12,
    )
    # Add known YTD live trade at 1x MES basis
    gire_trades.append({"date": "2026-01-28", "pnl_usd": 321.0})
    print(f"  GIRE base: {len(gire_trades)} trades, "
          f"PnL=${sum(t['pnl_usd'] for t in gire_trades):,.0f}")

    return orb_trades, msite_trades, gire_trades


# ---------------------------------------------------------------------------
# Scaling utilities
# ---------------------------------------------------------------------------

def orb_daily_pnl(trades: list[dict], n_contracts: int) -> dict[str, float]:
    """ORB: base is 3x MNQ. Scale to n_contracts MNQ."""
    daily: dict[str, float] = {}
    for t in trades:
        d = str(t["date"])[:10]
        scaled_pnl = t["pnl"] / 3.0 * n_contracts
        daily[d] = daily.get(d, 0.0) + scaled_pnl
    return daily


def msite_daily_pnl(trades: list[dict], n_live: int) -> dict[str, float]:
    """MSITE: base is 2x MNQ. Scale to n_live MNQ."""
    daily: dict[str, float] = {}
    for t in trades:
        d = str(t["date"])[:10]
        scaled_pnl = t["pnl"] / 2.0 * n_live
        daily[d] = daily.get(d, 0.0) + scaled_pnl
    return daily


def gire_daily_pnl(trades: list[dict], n_mnq: int) -> dict[str, float]:
    """GIRE: base is 1x MES ($5/pt). Convert to n_mnq MNQ ($2/pt).
    Scale factor = n_mnq * (2.0 / 5.0)
    """
    scale = n_mnq * (2.0 / 5.0)
    daily: dict[str, float] = {}
    for t in trades:
        d = str(t["date"])[:10]
        scaled_pnl = t["pnl_usd"] * scale
        daily[d] = daily.get(d, 0.0) + scaled_pnl
    return daily


def combine_daily(
    orb_d: dict[str, float],
    msite_d: dict[str, float],
    gire_d: dict[str, float],
) -> dict[str, float]:
    all_dates = sorted(set(orb_d) | set(msite_d) | set(gire_d))
    return {
        d: orb_d.get(d, 0.0) + msite_d.get(d, 0.0) + gire_d.get(d, 0.0)
        for d in all_dates
    }


def portfolio_metrics(daily: dict[str, float]) -> dict:
    """Compute max daily loss, max portfolio DD, total PnL, Sharpe."""
    if not daily:
        return {
            "max_single_day_loss": 0.0,
            "max_dd": 0.0,
            "total_pnl": 0.0,
            "sharpe": 0.0,
        }
    vals = np.array([daily[d] for d in sorted(daily)], dtype=float)
    total = float(vals.sum())
    non_zero = vals[vals != 0.0]
    if len(non_zero) > 1 and non_zero.std(ddof=1) > 0:
        sharpe = float(non_zero.mean() / non_zero.std(ddof=1) * np.sqrt(252))
    else:
        sharpe = 0.0

    equity = np.cumsum(vals)
    peak = np.maximum.accumulate(equity)
    dd_series = equity - peak
    max_dd = float(dd_series.min())
    max_day_loss = float(vals.min())

    return {
        "max_single_day_loss": round(max_day_loss, 2),
        "max_dd": round(max_dd, 2),
        "total_pnl": round(total, 2),
        "sharpe": round(sharpe, 3),
    }


def safety_flag(max_day_loss: float, max_dd: float) -> str:
    daily_ok = max_day_loss > -DAILY_LOSS_LIMIT
    dd_ok = max_dd > -DD_BUFFER
    if daily_ok and dd_ok:
        return "SAFE"
    elif dd_ok:
        return "RISK"
    else:
        return "BUST"


def max_single_trade_loss(
    orb_trades: list[dict],
    msite_trades: list[dict],
    gire_trades: list[dict],
    orb_n: int,
    msite_n: int,
    gire_n: int,
) -> float:
    """Worst single-trade PnL across all strategies at given scales."""
    worst = 0.0
    for t in orb_trades:
        p = t["pnl"] / 3.0 * orb_n
        if p < worst:
            worst = p
    for t in msite_trades:
        p = t["pnl"] / 2.0 * msite_n
        if p < worst:
            worst = p
    gire_scale = gire_n * (2.0 / 5.0)
    for t in gire_trades:
        p = t["pnl_usd"] * gire_scale
        if p < worst:
            worst = p
    return round(worst, 2)


# ---------------------------------------------------------------------------
# Monte Carlo P(pass)
# ---------------------------------------------------------------------------

def monte_carlo_pass_prob(
    daily_pnl_dict: dict[str, float],
    n_paths: int = 5000,
    window_days: int = 60,
    trail_dd: float = TRAIL_DD_LIMIT,
    daily_limit: float = DAILY_LOSS_LIMIT,
    target: float = PROFIT_TARGET,
    seed: int = 42,
) -> float:
    """Bootstrap daily PnL into n_paths × window_days paths, apply rules."""
    rng = np.random.default_rng(seed)
    vals = np.array([v for v in daily_pnl_dict.values() if v != 0.0], dtype=float)
    if len(vals) < 5:
        return 0.0

    # Bootstrap rows
    idx = rng.integers(0, len(vals), size=(n_paths, window_days))
    paths = vals[idx]  # shape (n_paths, window_days)

    passes = 0
    for path in paths:
        equity = 0.0
        peak = 0.0
        busted = False
        for pnl in path:
            # Daily loss check
            if pnl < -daily_limit:
                busted = True
                break
            equity += pnl
            if equity > peak:
                peak = equity
            # Trailing drawdown check
            if peak - equity > trail_dd:
                busted = True
                break
        if not busted and equity >= target:
            passes += 1

    return round(passes / n_paths, 4)


# ---------------------------------------------------------------------------
# Section 1: Contract Scaling Analysis
# ---------------------------------------------------------------------------

def section1_scaling(
    orb_trades: list[dict],
    msite_trades: list[dict],
    gire_trades: list[dict],
) -> dict:
    print("\n" + "=" * 72)
    print("  SECTION 1: Contract Scaling Analysis")
    print("=" * 72)

    results = {}

    # Fixed base daily PnL for non-varied strategies
    orb_base_daily   = orb_daily_pnl(orb_trades,   n_contracts=3)
    msite_base_daily = msite_daily_pnl(msite_trades, n_live=12)
    gire_base_daily  = gire_daily_pnl(gire_trades,   n_mnq=5)

    # ---- ORB sweep ----
    print("\n  ORB Contract Sweep (MSITE=12x, GIRE=5x fixed)")
    hdr = f"  {'Contracts':>10} {'N':>4} {'MaxTrdLoss':>12} {'MaxDayLoss':>12} {'MaxDD':>10} {'TotPnL':>10} {'Sharpe':>7} {'Flag':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    orb_sweep = []
    for n in [1, 2, 3, 4, 5]:
        od = orb_daily_pnl(orb_trades, n)
        port = combine_daily(od, msite_base_daily, gire_base_daily)
        m = portfolio_metrics(port)
        mtl = max_single_trade_loss(orb_trades, msite_trades, gire_trades, n, 12, 5)
        flag = safety_flag(m["max_single_day_loss"], m["max_dd"])
        row = {
            "n_contracts": n,
            "n_trades": len(orb_trades),
            "max_trade_loss": mtl,
            "max_day_loss": m["max_single_day_loss"],
            "max_dd": m["max_dd"],
            "total_pnl": m["total_pnl"],
            "sharpe": m["sharpe"],
            "flag": flag,
        }
        orb_sweep.append(row)
        marker = " <-- current" if n == 3 else ""
        print(f"  {n:>10}x {'':>4} {mtl:>+12,.0f} {m['max_single_day_loss']:>+12,.0f} "
              f"{m['max_dd']:>+10,.0f} {m['total_pnl']:>+10,.0f} "
              f"{m['sharpe']:>7.2f} {flag:>6}{marker}")

    results["orb_sweep"] = orb_sweep

    # ---- MSITE sweep ----
    print("\n  MSITE Live-Contract Sweep (ORB=3x, GIRE=5x fixed)")
    hdr = f"  {'LiveContracts':>13} {'MaxTrdLoss':>12} {'MaxDayLoss':>12} {'MaxDD':>10} {'TotPnL':>10} {'Sharpe':>7} {'Flag':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    msite_sweep = []
    for n in [6, 9, 12, 15, 18]:
        md = msite_daily_pnl(msite_trades, n)
        port = combine_daily(orb_base_daily, md, gire_base_daily)
        m = portfolio_metrics(port)
        mtl = max_single_trade_loss(orb_trades, msite_trades, gire_trades, 3, n, 5)
        flag = safety_flag(m["max_single_day_loss"], m["max_dd"])
        row = {
            "n_live": n,
            "max_trade_loss": mtl,
            "max_day_loss": m["max_single_day_loss"],
            "max_dd": m["max_dd"],
            "total_pnl": m["total_pnl"],
            "sharpe": m["sharpe"],
            "flag": flag,
        }
        msite_sweep.append(row)
        marker = " <-- current" if n == 12 else ""
        print(f"  {n:>13}x {mtl:>+12,.0f} {m['max_single_day_loss']:>+12,.0f} "
              f"{m['max_dd']:>+10,.0f} {m['total_pnl']:>+10,.0f} "
              f"{m['sharpe']:>7.2f} {flag:>6}{marker}")

    results["msite_sweep"] = msite_sweep

    # ---- GIRE sweep ----
    print("\n  GIRE Live-MNQ Sweep (ORB=3x, MSITE=12x fixed)")
    hdr = f"  {'LiveMNQ':>9} {'MaxTrdLoss':>12} {'MaxDayLoss':>12} {'MaxDD':>10} {'TotPnL':>10} {'Sharpe':>7} {'Flag':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    gire_sweep = []
    for n in [3, 4, 5, 6, 7]:
        gd = gire_daily_pnl(gire_trades, n)
        port = combine_daily(orb_base_daily, msite_base_daily, gd)
        m = portfolio_metrics(port)
        mtl = max_single_trade_loss(orb_trades, msite_trades, gire_trades, 3, 12, n)
        flag = safety_flag(m["max_single_day_loss"], m["max_dd"])
        row = {
            "n_mnq": n,
            "max_trade_loss": mtl,
            "max_day_loss": m["max_single_day_loss"],
            "max_dd": m["max_dd"],
            "total_pnl": m["total_pnl"],
            "sharpe": m["sharpe"],
            "flag": flag,
        }
        gire_sweep.append(row)
        marker = " <-- current" if n == 5 else ""
        print(f"  {n:>9}x {mtl:>+12,.0f} {m['max_single_day_loss']:>+12,.0f} "
              f"{m['max_dd']:>+10,.0f} {m['total_pnl']:>+10,.0f} "
              f"{m['sharpe']:>7.2f} {flag:>6}{marker}")

    results["gire_sweep"] = gire_sweep
    return results


# ---------------------------------------------------------------------------
# Section 2: Portfolio Joint Scaling
# ---------------------------------------------------------------------------

def section2_joint(
    orb_trades: list[dict],
    msite_trades: list[dict],
    gire_trades: list[dict],
) -> list[dict]:
    print("\n" + "=" * 72)
    print("  SECTION 2: Portfolio Joint Scaling (safe configs only)")
    print("=" * 72)

    orb_grid   = [2, 3, 4]
    msite_grid = [9, 12, 15]
    gire_grid  = [4, 5, 6]

    safe_rows = []
    total_combos = len(orb_grid) * len(msite_grid) * len(gire_grid)
    print(f"\n  Testing {total_combos} combinations ...")

    for orb_n, msite_n, gire_n in product(orb_grid, msite_grid, gire_grid):
        od = orb_daily_pnl(orb_trades, orb_n)
        md = msite_daily_pnl(msite_trades, msite_n)
        gd = gire_daily_pnl(gire_trades, gire_n)
        port = combine_daily(od, md, gd)
        m = portfolio_metrics(port)

        daily_ok = m["max_single_day_loss"] > -DAILY_LOSS_LIMIT
        dd_ok = m["max_dd"] > -DD_BUFFER

        if not (daily_ok and dd_ok):
            continue

        p_pass = monte_carlo_pass_prob(port, n_paths=5000, window_days=60)

        safe_rows.append({
            "orb_contracts":  orb_n,
            "msite_live":     msite_n,
            "gire_live":      gire_n,
            "max_day_loss":   m["max_single_day_loss"],
            "max_dd":         m["max_dd"],
            "total_pnl":      m["total_pnl"],
            "sharpe":         m["sharpe"],
            "p_pass":         p_pass,
        })

    safe_rows.sort(key=lambda r: r["sharpe"], reverse=True)
    top10 = safe_rows[:10]

    print(f"\n  Safe configs: {len(safe_rows)} / {total_combos}")
    print(f"\n  Top 10 by Sharpe:")
    hdr = (f"  {'ORB':>4} {'MSITE':>6} {'GIRE':>5} "
           f"{'MaxDayLoss':>12} {'MaxDD':>10} {'TotPnL':>10} "
           f"{'Sharpe':>7} {'P(pass)':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in top10:
        print(f"  {r['orb_contracts']:>4}  {r['msite_live']:>5}  {r['gire_live']:>4}  "
              f"{r['max_day_loss']:>+11,.0f}  {r['max_dd']:>+9,.0f}  "
              f"{r['total_pnl']:>+9,.0f}  {r['sharpe']:>7.2f}  {r['p_pass']:>8.1%}")

    return safe_rows


# ---------------------------------------------------------------------------
# Section 3a: ORB parameter sweep
# ---------------------------------------------------------------------------

def section3a_orb_params(bars_long: pd.DataFrame) -> list[dict]:
    print("\n" + "=" * 72)
    print("  SECTION 3a: ORB PT/SL Sweep (3x MNQ, long period)")
    print("=" * 72)

    import novel_filter_sweep as nf
    from novel_filter_sweep import run_orb, build_day_meta, enrich_prev_vwap

    day_meta = enrich_prev_vwap(bars_long, build_day_meta(bars_long))
    prev_vwap_filter = lambda m: m.get("prev_vwap_bullish") is True

    pt_grid = [1.5, 2.0, 2.5, 3.0]
    sl_grid = [1.0, 1.25, 1.5]

    rows = []
    hdr = (f"  {'PT':>5} {'SL':>6} {'N':>4} {'WR':>6} "
           f"{'TotPnL':>10} {'Sharpe':>7} {'MaxDD':>10} {'MaxTrdLoss':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for pt, sl in product(pt_grid, sl_grid):
        nf.PT_MULT = pt
        nf.SL_MULT = sl
        result = run_orb(bars_long, day_meta, filter_fn=prev_vwap_filter)
        trades = result["trades"]
        total_pnl = sum(t["pnl"] for t in trades)
        n = len(trades)
        wr = sum(1 for t in trades if t["pnl"] > 0) / max(n, 1)
        sharpe = result.get("sharpe", 0.0)
        max_dd = result.get("dd", 0.0)
        mtl = min((t["pnl"] for t in trades), default=0.0)

        marker = " <-- base" if (abs(pt - 3.0) < 0.01 and abs(sl - 1.5) < 0.01) else ""
        print(f"  {pt:>5.2f} {sl:>6.2f} {n:>4}  {wr:>5.1%} "
              f"{total_pnl:>+10,.0f} {sharpe:>7.2f} {max_dd:>+10,.0f} {mtl:>+12,.0f}{marker}")

        rows.append({
            "pt_mult": pt,
            "sl_mult": sl,
            "n": n,
            "wr": round(wr, 4),
            "total_pnl": round(total_pnl, 2),
            "sharpe": round(sharpe, 3),
            "max_dd": round(max_dd, 2),
            "max_trade_loss": round(mtl, 2),
        })

    # Restore defaults
    nf.PT_MULT = 3.0
    nf.SL_MULT = 1.5

    return rows


# ---------------------------------------------------------------------------
# Section 3b: MSITE parameter sweep
# ---------------------------------------------------------------------------

def section3b_msite_params(bars_long: pd.DataFrame) -> list[dict]:
    print("\n" + "=" * 72)
    print("  SECTION 3b: MSITE pt_mult x sl_frac Sweep (at 12x scale)")
    print("=" * 72)

    import msite_backtest as mb

    pt_grid  = [1.5, 2.0, 2.5]
    sl_grid  = [0.40, 0.55, 0.70]
    live_scale = 12.0 / 2.0   # base is 2x MNQ, live is 12x

    rows = []
    hdr = (f"  {'pt_mult':>8} {'sl_frac':>8} {'N':>4} {'WR':>6} "
           f"{'TotPnL@12x':>12} {'Sharpe':>7} {'MaxTrdLoss@12x':>16}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for pt, sl in product(pt_grid, sl_grid):
        mb._apply_tf_config(5)
        trades, _ = mb.run_backtest(
            bars_long,
            allow_release=True,
            allow_exhaustion=False,
            long_only=True,
            afternoon_cutoff=(14, 0),
            pt_mult=pt,
            sl_frac=sl,
            time_stop_bars=12,
        )
        if not trades:
            continue

        pnls = np.array([t["pnl"] for t in trades])
        total_pnl_scaled = float(pnls.sum() * live_scale)
        n = len(trades)
        wr = float((pnls > 0).mean())

        dates_d: dict[str, float] = {}
        for t in trades:
            dates_d[t["date"]] = dates_d.get(t["date"], 0.0) + t["pnl"]
        dv = np.array(list(dates_d.values()))
        if len(dv) > 1 and dv.std(ddof=1) > 0:
            sharpe = float(dv.mean() / dv.std(ddof=1) * np.sqrt(252))
        else:
            sharpe = 0.0

        mtl_scaled = float(pnls.min() * live_scale)

        marker = " <-- base" if (abs(pt - 2.0) < 0.01 and abs(sl - 0.55) < 0.01) else ""
        print(f"  {pt:>8.2f} {sl:>8.3f} {n:>4}  {wr:>5.1%} "
              f"{total_pnl_scaled:>+12,.0f} {sharpe:>7.2f} {mtl_scaled:>+16,.0f}{marker}")

        rows.append({
            "pt_mult": pt,
            "sl_frac": sl,
            "n": n,
            "wr": round(wr, 4),
            "total_pnl_12x": round(total_pnl_scaled, 2),
            "sharpe": round(sharpe, 3),
            "max_trade_loss_12x": round(mtl_scaled, 2),
        })

    return rows


# ---------------------------------------------------------------------------
# Section 3c: GIRE parameter sweep
# ---------------------------------------------------------------------------

def section3c_gire_params(mes_aug: pd.DataFrame) -> list[dict]:
    print("\n" + "=" * 72)
    print("  SECTION 3c: GIRE clv_thresh x g_star Sweep (at 5x MNQ scale)")
    print("=" * 72)

    from gire_backtest import build_daily_data, run_backtest as gire_run

    gire_daily_data = build_daily_data(mes_aug)
    live_scale = 5.0 * (2.0 / 5.0)   # 5x MNQ at $2/pt from 1x MES $5/pt basis

    clv_grid = [0.70, 0.75, 0.80]
    gstar_grid = [0.15, 0.20, 0.25]

    rows = []
    hdr = (f"  {'clv_thresh':>10} {'g_star':>7} {'N':>4} {'WR':>6} "
           f"{'TotPnL@5xMNQ':>14} {'Sharpe':>7} {'MaxTrdLoss':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for clv, g in product(clv_grid, gstar_grid):
        trades = gire_run(
            gire_daily_data,
            g_star=g,
            clv_thresh=clv,
            ey_thresh=0.0,
            st_thresh=0.0,
            use_opening_failure=True,
            time_stop_hour=12,
        )
        if not trades:
            continue

        pnls_base = np.array([t["pnl_usd"] for t in trades])
        pnls_scaled = pnls_base * live_scale
        n = len(trades)
        wr = float((pnls_base > 0).mean())
        total_pnl_scaled = float(pnls_scaled.sum())

        dates_d: dict[str, float] = {}
        for t in trades:
            dates_d[t["date"]] = dates_d.get(t["date"], 0.0) + t["pnl_usd"]
        dv = np.array(list(dates_d.values()))
        if len(dv) > 1 and dv.std(ddof=1) > 0:
            sharpe = float(dv.mean() / dv.std(ddof=1) * np.sqrt(252))
        else:
            sharpe = 0.0

        mtl_scaled = float(pnls_scaled.min())

        marker = " <-- base" if (abs(clv - 0.80) < 0.01 and abs(g - 0.20) < 0.01) else ""
        print(f"  {clv:>10.2f} {g:>7.2f} {n:>4}  {wr:>5.1%} "
              f"{total_pnl_scaled:>+14,.0f} {sharpe:>7.2f} {mtl_scaled:>+12,.0f}{marker}")

        rows.append({
            "clv_thresh": clv,
            "g_star": g,
            "n": n,
            "wr": round(wr, 4),
            "total_pnl_5x_mnq": round(total_pnl_scaled, 2),
            "sharpe": round(sharpe, 3),
            "max_trade_loss_5x_mnq": round(mtl_scaled, 2),
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("  Scaling Optimization Sweep — ORB + MSITE + GIRE")
    print("  Topstep 50k Combine constraints:")
    print(f"    Trail DD limit : ${TRAIL_DD_LIMIT:,.0f}")
    print(f"    Daily loss cap : ${DAILY_LOSS_LIMIT:,.0f}")
    print(f"    Profit target  : ${PROFIT_TARGET:,.0f}")
    print(f"    Safety buffer  : max_dd > ${-DD_BUFFER:,.0f}")
    print("=" * 72)

    bars_long, bars_ytd, mes_aug = load_data()
    orb_trades, msite_trades, gire_trades = run_base_strategies(bars_long, mes_aug)

    # Section 1
    s1 = section1_scaling(orb_trades, msite_trades, gire_trades)

    # Section 2
    s2 = section2_joint(orb_trades, msite_trades, gire_trades)

    # Section 3a
    s3a = section3a_orb_params(bars_long)

    # Section 3b
    s3b = section3b_msite_params(bars_long)

    # Section 3c
    s3c = section3c_gire_params(mes_aug)

    # Save results
    output = {
        "section1_orb_sweep":   s1["orb_sweep"],
        "section1_msite_sweep": s1["msite_sweep"],
        "section1_gire_sweep":  s1["gire_sweep"],
        "section2_joint_safe":  s2,
        "section3a_orb_params": s3a,
        "section3b_msite_params": s3b,
        "section3c_gire_params":  s3c,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {RESULTS_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
