"""
MCL (Micro Crude Oil) ORB Backtest
====================================
1. Fetches MCL.c.0 1-min bars from Databento
2. Resamples to 5-min, filters to RTH (9:30–14:30 ET)
3. Runs ORB backtest on OOS period (Jan 1 – Feb 10, 2026)
4. Sweeps or_end × entry_cutoff to find MCL-optimal params
5. Prints comparison vs MES results

Contract specs (MCL):
  point_value : $100  (1 point = $100)
  tick_size   : $0.01 per point
  tick_value  : $1.00 per tick

Usage:
    cd "algos 3 topstep"
    python ml_intraday_v3/diagnostics/fetch_backtest_mcl.py [--fetch-only] [--backtest-only]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
_RBV1_DIR = _PROJECT_ROOT / "rule_based_v1"
_DIAG_DIR = _HERE.parent

for _p in [str(_PROJECT_ROOT), str(_RBV1_DIR), str(_DIAG_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MCL_H5_PATH = _PROJECT_ROOT / "data" / "processed" / "mcl_bars_5min.h5"
MCL_RESULTS_PATH = _DIAG_DIR / "mcl_orb_results.json"
MES_RESULTS_PATH = _DIAG_DIR / "orb_oos_results.json"  # for comparison

# ---------------------------------------------------------------------------
# Contract specs
# ---------------------------------------------------------------------------
MCL_SPECS = dict(
    point_value=100.0,
    tick_size=0.01,
    tick_value=1.00,
)

# ---------------------------------------------------------------------------
# Backtest params — same logic as MES best config
# ---------------------------------------------------------------------------
BASE = dict(
    min_range_atr=0.3,
    pt_atr_mult=2.0,
    sl_atr_mult=1.5,
    n_contracts=2,
    trailing_activation_atr=999.0,
    trailing_distance_atr=0.75,
    atr_period=14,
    commission_per_side=0.35,   # MCL NinjaTrader typical commission
    slippage_ticks=1,
    time_stop_bars=24,
)

RISK = dict(
    point_value=MCL_SPECS["point_value"],
    tick_size=MCL_SPECS["tick_size"],
    tick_value=MCL_SPECS["tick_value"],
    max_daily_loss=-950.0,
    per_trade_max_loss=1000.0,
    max_consecutive_losses=3,
    cooldown_bars=3,
    flatten_minutes_before_close=5,
    drawdown_buffer=1800.0,
)

COMBINE = dict(
    account_size=50_000,
    profit_target=3_000,
    max_trailing_drawdown=2_000,
    max_daily_loss=1_000,
    consistency_pct=0.30,
    min_trading_days=5,
)

# Sweep grids — same as MES sweep
ENTRY_CUTOFFS = ["11:00", "12:00", "13:00", "14:00"]
OR_WINDOWS    = ["09:44", "09:55", "10:14"]


# ===========================================================================
# STEP 1: FETCH
# ===========================================================================

def fetch_mcl_bars(start_date: str = "2025-08-01",
                   end_date: str = "2026-02-10") -> pd.DataFrame:
    """Fetch MCL 1-min bars from Databento, resample to 5-min, filter RTH."""
    from dotenv import load_dotenv
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not set in environment / .env")

    import databento as db
    client = db.Historical(key=api_key)

    logger.info(f"Fetching MCL.c.0  {start_date} → {end_date} ...")
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["MCL.c.0"],
        schema="ohlcv-1m",
        start=start_date,
        end=end_date,
        stype_in="continuous",
    )

    df = data.to_df()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]]
    logger.info(f"Fetched {len(df):,} 1-min bars  "
                f"{df.index[0].date()} → {df.index[-1].date()}")

    # Convert to US/Eastern for filtering
    df_et = df.copy()
    df_et.index = df_et.index.tz_convert("US/Eastern")

    # RTH filter: 9:30 AM – 2:30 PM ET  (match ORB session_open hardcode)
    # Crude RTH is 9:00–14:30 but ORB window starts at 9:30; keep 9:30-14:30
    rth = ((df_et.index.hour > 9) |
           ((df_et.index.hour == 9) & (df_et.index.minute >= 30)))
    rth &= ((df_et.index.hour < 14) |
            ((df_et.index.hour == 14) & (df_et.index.minute <= 30)))
    df_et = df_et.loc[rth]
    logger.info(f"After RTH filter (9:30–14:30 ET): {len(df_et):,} 1-min bars")

    # Resample to 5-min
    bars_5m = df_et.resample("5min").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["open", "close"])

    # Drop bars where market was closed (e.g. holidays produce NaN volume)
    bars_5m = bars_5m[bars_5m["volume"] > 0]

    logger.info(f"After resample: {len(bars_5m):,} 5-min bars  "
                f"{bars_5m.index[0]} → {bars_5m.index[-1]}")
    return bars_5m


# ===========================================================================
# STEP 2: BACKTEST
# ===========================================================================

def run_config(bars, or_end, cutoff, simulator):
    """Run one ORB configuration, return result dict."""
    from engine.backtest_engine import BacktestEngine
    from engine.signal_aggregator import SignalAggregator
    from engine.risk_manager import RiskManager
    from rules.opening_range import OpeningRangeBreakoutRule
    from rules.time_of_day import TimeOfDayRule

    primary = OpeningRangeBreakoutRule(
        or_end_time=or_end,
        min_or_bars=2,
        min_range_atr=BASE["min_range_atr"],
        entry_cutoff_time=cutoff,
        atr_period=BASE["atr_period"],
        use_close_for_signal=True,
    )
    agg = SignalAggregator(
        primary_rule=primary,
        filter_rules=[TimeOfDayRule(
            session_start="09:35",
            session_end="14:30",   # MCL session ends 2:30 PM ET
            lunch_filter_enabled=False,
        )],
        confirmation_rules=[],
        min_confirmations=0,
    )
    rm = RiskManager(
        contracts=BASE["n_contracts"],
        **{k: RISK[k] for k in RISK},
    )
    engine = BacktestEngine(
        aggregator=agg,
        risk_manager=rm,
        commission_per_side=BASE["commission_per_side"],
        slippage_ticks=BASE["slippage_ticks"],
        profit_target_atr=BASE["pt_atr_mult"],
        stop_loss_atr=BASE["sl_atr_mult"],
        time_stop_bars=BASE["time_stop_bars"],
        trailing_activation_atr=BASE["trailing_activation_atr"],
        trailing_distance_atr=BASE["trailing_distance_atr"],
        atr_period=BASE["atr_period"],
    )

    result = engine.run(bars, starting_equity=50_000.0)
    s = result.summary()

    trade_pnls = [t.pnl for t in result.trades]
    if len(trade_pnls) >= 5:
        mc = simulator.monte_carlo(
            trade_pnl_list=trade_pnls,
            n_paths=10_000,
            trades_per_day_range=(1, 4),
            max_days=40,
            seed=42,
        )
    else:
        mc = {"p_pass": 0.0, "median_days": None,
              "p95_max_drawdown": None, "p_fail_trailing": None}

    exit_counts = {}
    for t in result.trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    return {
        "or_end":    or_end,
        "cutoff":    cutoff,
        "n_trades":  s["num_trades"],
        "win_rate":  s.get("win_rate", 0.0),
        "total_pnl": s.get("total_pnl", 0.0),
        "avg_pnl":   s.get("avg_trade_pnl", 0.0),
        "sharpe":    s.get("sharpe_ratio", 0.0),
        "max_dd":    s.get("max_drawdown", 0.0),
        "p_pass":    mc["p_pass"],
        "med_days":  mc.get("median_days"),
        "p95_dd":    mc.get("p95_max_drawdown"),
        "p_fail_dd": mc.get("p_fail_trailing"),
        "exits":     exit_counts,
    }


def run_backtest(bars: pd.DataFrame) -> list[dict]:
    """Sweep all or_end × cutoff combos, return ranked results."""
    from topstep_combine_simulator import TopstepCombineSimulator

    simulator = TopstepCombineSimulator(**COMBINE)
    results = []

    total = len(OR_WINDOWS) * len(ENTRY_CUTOFFS)
    for i, (or_end, cutoff) in enumerate(product(OR_WINDOWS, ENTRY_CUTOFFS), 1):
        print(f"  [{i:2d}/{total}] or_end={or_end}  cutoff={cutoff}",
              end="  ", flush=True)
        r = run_config(bars, or_end, cutoff, simulator)
        results.append(r)
        print(f"trades={r['n_trades']:2d}  wr={r['win_rate']:.1%}  "
              f"pnl=${r['total_pnl']:+.0f}  P(pass)={r['p_pass']:.1%}")

    return results


# ===========================================================================
# STEP 3: PRINT COMPARISON TABLES
# ===========================================================================

def print_tables(results: list[dict]) -> None:
    print("\n" + "=" * 95)
    print("MCL ORB — ENTRY CUTOFF SWEEP  (or_end fixed = 09:44 ET, best OR window)")
    print("=" * 95)
    print(f"{'Cutoff':>8}  {'Trades':>6}  {'WR':>6}  {'PnL':>8}  {'Avg':>7}  "
          f"{'Sharpe':>7}  {'MaxDD':>8}  {'P(pass)':>8}  {'MedDays':>8}  {'p95DD':>8}")
    print("-" * 95)
    for r in results:
        if r["or_end"] != "09:44":
            continue
        marker = " ◄ BEST OR" if r["or_end"] == "09:44" and r["cutoff"] == "12:00" else ""
        print(f"  {r['cutoff']:>6}  {r['n_trades']:>6}  {r['win_rate']:>5.1%}  "
              f"${r['total_pnl']:>+7.0f}  ${r['avg_pnl']:>+6.0f}  "
              f"{r['sharpe']:>7.2f}  ${r['max_dd']:>+7.0f}  "
              f"{r['p_pass']:>7.1%}  {str(r['med_days'] or '—'):>8}  "
              f"${r['p95_dd'] or 0:>7.0f}{marker}")

    print("\n" + "=" * 95)
    print("MCL ORB — OR WINDOW SWEEP  (cutoff fixed = 12:00 ET)")
    print("=" * 95)
    print(f"{'OR end':>8}  {'Trades':>6}  {'WR':>6}  {'PnL':>8}  {'Avg':>7}  "
          f"{'Sharpe':>7}  {'P(pass)':>8}  {'MedDays':>8}")
    print("-" * 95)
    for r in results:
        if r["cutoff"] != "12:00":
            continue
        print(f"  {r['or_end']:>6}  {r['n_trades']:>6}  {r['win_rate']:>5.1%}  "
              f"${r['total_pnl']:>+7.0f}  ${r['avg_pnl']:>+6.0f}  "
              f"{r['sharpe']:>7.2f}  {r['p_pass']:>7.1%}  "
              f"{str(r['med_days'] or '—'):>8}")

    print("\n" + "=" * 95)
    print("MCL ORB — TOP 10  (ranked by P(pass))")
    print("=" * 95)
    ranked = sorted(results, key=lambda x: (-x["p_pass"], -x["sharpe"]))
    print(f"{'OR end':>8}  {'Cutoff':>8}  {'Trades':>6}  {'WR':>6}  {'PnL':>8}  "
          f"{'Sharpe':>7}  {'P(pass)':>8}  {'MedDays':>8}  {'p95DD':>8}")
    print("-" * 95)
    for r in ranked[:10]:
        print(f"  {r['or_end']:>6}  {r['cutoff']:>8}  {r['n_trades']:>6}  "
              f"{r['win_rate']:>5.1%}  ${r['total_pnl']:>+7.0f}  "
              f"{r['sharpe']:>7.2f}  {r['p_pass']:>7.1%}  "
              f"{str(r['med_days'] or '—'):>8}  ${r['p95_dd'] or 0:>7.0f}")

    # --- vs MES comparison ---
    if MES_RESULTS_PATH.exists():
        with open(MES_RESULTS_PATH) as f:
            mes = json.load(f)
        # MES results file format may vary — try to get summary row
        if isinstance(mes, dict):
            mes_row = mes.get("summary", mes)
        elif isinstance(mes, list):
            mes_row = mes[0]  # first result
        else:
            mes_row = {}

        best_mcl = ranked[0] if ranked else {}
        print("\n" + "=" * 60)
        print("INSTRUMENT COMPARISON (best config each)")
        print("=" * 60)
        print(f"{'':10}  {'Trades':>6}  {'WR':>6}  {'PnL':>8}  {'P(pass)':>8}  {'p95DD':>8}")
        print("-" * 60)
        if best_mcl:
            print(f"  {'MCL':8}  {best_mcl['n_trades']:>6}  {best_mcl['win_rate']:>5.1%}  "
                  f"${best_mcl['total_pnl']:>+7.0f}  {best_mcl['p_pass']:>7.1%}  "
                  f"${best_mcl['p95_dd'] or 0:>7.0f}")
        if isinstance(mes_row, dict) and "win_rate" in mes_row:
            print(f"  {'MES':8}  {mes_row.get('num_trades', mes_row.get('n_trades', '?')):>6}  "
                  f"{mes_row.get('win_rate', 0):>5.1%}  "
                  f"${mes_row.get('total_pnl', 0):>+7.0f}  "
                  f"{mes_row.get('p_pass', 0):>7.1%}  "
                  f"${mes_row.get('p95_dd', mes_row.get('p95_max_drawdown', 0)) or 0:>7.0f}")
        print("\nNote: running BOTH instruments on the same account doubles drawdown exposure.")
        print("If trading MCL + MES simultaneously, use n=1 contract each.")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Fetch MCL bars and run ORB backtest")
    parser.add_argument("--fetch-only",   action="store_true",
                        help="Only fetch and save bars, skip backtest")
    parser.add_argument("--backtest-only", action="store_true",
                        help="Skip fetch, use cached bars (MCL_H5_PATH must exist)")
    parser.add_argument("--start",  default="2025-08-01",
                        help="Fetch start date (default: 2025-08-01)")
    parser.add_argument("--end",    default="2026-02-10",
                        help="Fetch end date  (default: 2026-02-10)")
    parser.add_argument("--oos-start", default="2026-01-01",
                        help="OOS backtest start (default: 2026-01-01)")
    parser.add_argument("--oos-end",   default="2026-02-10",
                        help="OOS backtest end   (default: 2026-02-10)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # FETCH
    # ------------------------------------------------------------------
    if not args.backtest_only:
        bars_full = fetch_mcl_bars(args.start, args.end)

        output_dir = _PROJECT_ROOT / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        bars_full.to_hdf(MCL_H5_PATH, key="bars_5min", mode="w")
        logger.info(f"Saved {len(bars_full):,} bars → {MCL_H5_PATH}")
    else:
        if not MCL_H5_PATH.exists():
            logger.error(f"Cached bars not found: {MCL_H5_PATH}. Run without --backtest-only first.")
            sys.exit(1)
        bars_full = pd.read_hdf(MCL_H5_PATH, key="bars_5min")
        logger.info(f"Loaded {len(bars_full):,} cached bars from {MCL_H5_PATH}")

    if args.fetch_only:
        logger.info("--fetch-only: skipping backtest.")
        return

    # ------------------------------------------------------------------
    # BACKTEST on OOS slice
    # ------------------------------------------------------------------
    oos_start = pd.Timestamp(args.oos_start, tz="US/Eastern")
    oos_end   = pd.Timestamp(args.oos_end,   tz="US/Eastern") + pd.Timedelta(days=1)

    # Ensure index is tz-aware
    if bars_full.index.tz is None:
        bars_full.index = bars_full.index.tz_localize("US/Eastern")
    else:
        bars_full.index = bars_full.index.tz_convert("US/Eastern")

    bars_oos = bars_full.loc[oos_start:oos_end]
    logger.info(f"OOS slice: {len(bars_oos):,} bars  "
                f"{bars_oos.index[0]} → {bars_oos.index[-1]}")

    if len(bars_oos) < 100:
        logger.error("Too few OOS bars for a meaningful backtest. "
                     "Check date range or re-fetch.")
        sys.exit(1)

    # Quick ATR sanity check — print median ATR to gauge dollar risk
    from utils.indicators import atr as _atr
    atr_vals = _atr(bars_oos["high"], bars_oos["low"], bars_oos["close"], period=14).dropna()
    med_atr = atr_vals.median()
    logger.info(
        f"MCL median 5-min ATR = {med_atr:.3f} pts  "
        f"= ${med_atr * MCL_SPECS['point_value']:.1f}/contract  "
        f"(2x ATR PT = ${med_atr * MCL_SPECS['point_value'] * 2 * BASE['n_contracts']:.0f} for "
        f"{BASE['n_contracts']} contracts)"
    )

    print(f"\n{'='*60}")
    print(f"MCL ORB SWEEP — OOS {args.oos_start} → {args.oos_end}")
    print(f"Contract: MCL  point_value=${MCL_SPECS['point_value']:.0f}  "
          f"tick=${MCL_SPECS['tick_value']:.2f}")
    print(f"Bars: {len(bars_oos):,}  Median ATR: {med_atr:.3f} pts "
          f"(${med_atr * MCL_SPECS['point_value']:.1f}/contract)")
    print(f"{'='*60}\n")

    results = run_backtest(bars_oos)

    print_tables(results)

    # Save
    with open(MCL_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nFull results → {MCL_RESULTS_PATH}")

    # Verdict
    best = max(results, key=lambda x: x["p_pass"])
    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")
    if best["p_pass"] >= 0.50:
        print(f"✅ MCL ORB has REAL EDGE: best P(pass)={best['p_pass']:.1%}  "
              f"(or_end={best['or_end']}, cutoff={best['cutoff']})")
        print(f"   WR={best['win_rate']:.1%}  trades={best['n_trades']}  "
              f"PnL=${best['total_pnl']:+.0f}")
        print(f"   CONSIDER: Add MCL alongside MES with n=1 each to diversify.")
        print(f"   CAUTION: Shared $2,000 trailing drawdown — monitor combined drawdown.")
    elif best["p_pass"] >= 0.30:
        print(f"⚠️  MCL ORB shows MARGINAL EDGE: best P(pass)={best['p_pass']:.1%}")
        print(f"   Win rate {best['win_rate']:.1%} with {best['n_trades']} trades on OOS")
        print(f"   NOT RECOMMENDED for combine without further optimization.")
    else:
        print(f"❌ MCL ORB has NO SIGNIFICANT EDGE: best P(pass)={best['p_pass']:.1%}")
        print(f"   Crude oil ORB does not generalize to this period.")
        print(f"   Stick with MES-only strategy.")


if __name__ == "__main__":
    main()
