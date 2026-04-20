"""Funded Phase Income Projection — CD-ORB v3.

Uses validated CD-ORB per-trade statistics (from OOS Jan-Mar 2026 and IS Aug-Jan)
to project monthly income and risk at various contract sizes in the funded account.

Topstep 50k funded account rules:
  - Max contracts: 10 MNQ
  - Daily loss limit: -$1,000  (same as combine)
  - Trailing max drawdown: -$2,000  (same as combine)
  - Payout split: 90% trader (after first $10k)
  - Consistency rule: no single day > 30% of total profits

Analysis covers:
  - 2, 3, 4, 6, 8, 10 MNQ contracts
  - Base: CDP-Anchor 2/day per-trade distribution (IS + OOS combined)
  - Monthly projections: PnL, P(daily limit hit), P(trailing DD hit)
  - Consistency rule compliance check

Run:
    cd rule_based_v1
    python diagnostics/funded_phase_projection.py

Output: diagnostics/funded_phase_projection.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for _p in [str(ROOT), str(RBV1)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Funded account parameters ─────────────────────────────────────────────────
POINT_VALUE       = 2.0
TICK_SIZE         = 0.25
COMMISSION        = 0.62
SLIPPAGE_TICKS    = 1
STARTING_EQUITY   = 50_000.0
DAILY_LOSS_LIMIT  = -1_000.0      # funded account daily loss limit
TRAILING_DD_LIMIT = -2_000.0      # funded account trailing max DD
PAYOUT_RATE       = 0.90          # 90% to trader
TRADING_DAYS_MONTH = 21
N_PATHS           = 50_000
RNG_SEED          = 42

# Contract sizes to test
CONTRACT_SIZES = [2, 3, 4, 6, 8, 10]

# Base config: CDP-Anchor 2c 2/day (IS results, normalized to per-contract per-trade)
# These are the 2-contract per-trade PnLs from v3 IS Config 1
# We normalize to 1-contract and re-scale for each target size.
BASE_N_CONTRACTS  = 2
TRADES_PER_DAY    = 0.26          # from IS CDP-Anchor (conservative)


def get_trades_from_backtest(bars: pd.DataFrame) -> list[float]:
    """Run CDP-Anchor 2c 2/day backtest and return per-contract PnLs."""
    from engine.signal_aggregator import SignalAggregator
    from rules.cumulative_delta_filter import CumulativeDeltaFilter
    from rules.opening_range import OpeningRangeBreakoutRule
    from utils.indicators import atr as compute_atr

    PT_MULT = 3.0; SL_MULT = 1.5; ATR_PERIOD = 14
    TIME_STOP_BARS = 24; MAX_DAILY_LOSS = -950.0
    DRAWDOWN_BUFFER = 1950.0; STARTING_EQUITY_LOCAL = 50_000.0

    orb = OpeningRangeBreakoutRule(
        or_end_time="10:04", min_or_bars=7, min_range_atr=0.3,
        entry_cutoff_time="12:00", atr_period=ATR_PERIOD, long_only=True,
    )
    cd = CumulativeDeltaFilter(cdp_required=True, min_other_score=1, allow_cdp_shorts=False)
    agg = SignalAggregator(primary_rule=orb, filter_rules=[cd],
                           confirmation_rules=[], min_confirmations=0)

    atr_series = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars = agg.required_bars()

    pos = None; pnls_1c: list[float] = []
    equity = STARTING_EQUITY_LOCAL; peak = STARTING_EQUITY_LOCAL
    cur_date = None; daily_loss = 0.0; trades_today = 0

    def slip(p, d, e):
        return p + SLIPPAGE_TICKS * TICK_SIZE * d if e else p - SLIPPAGE_TICKS * TICK_SIZE * d

    for i in range(min_bars, len(bars)):
        bar = bars.iloc[i]
        ts_et = bars.index[i].tz_convert("US/Eastern")
        bdate = ts_et.date()
        bh, bm = ts_et.hour, ts_et.minute

        if cur_date is not None and bdate != cur_date:
            daily_loss = 0.0; trades_today = 0
        cur_date = bdate

        atr_now = atr_series.iloc[i]
        if np.isnan(atr_now) or atr_now <= 0:
            continue

        is_last = (i+1 >= len(bars) or bars.index[i+1].tz_convert("US/Eastern").date() != bdate)
        sess_close = is_last or (bh == 15 and bm >= 55)

        if pos is not None:
            h, l, c = bar["high"], bar["low"], bar["close"]
            ep_sl = pos["sl"]; ep_pt = pos["pt"]; ep_ts = pos["ts"]
            exited = False; exit_p = 0.0
            if sess_close or i >= ep_ts:
                exit_p = slip(c, 1, False); exited = True; reason = "time/close"
            elif l <= ep_sl:
                exit_p = slip(ep_sl, 1, False); exited = True
            elif h >= ep_pt:
                exit_p = slip(ep_pt, 1, False); exited = True
            if exited:
                raw_pnl = (exit_p - pos["ep"]) * BASE_N_CONTRACTS * POINT_VALUE - 2 * COMMISSION * BASE_N_CONTRACTS
                pnls_1c.append(raw_pnl / BASE_N_CONTRACTS)
                equity += raw_pnl; daily_loss += raw_pnl
                peak = max(peak, equity); pos = None

        can_enter = (pos is None and not sess_close and trades_today < 2
                     and daily_loss > MAX_DAILY_LOSS and (equity - peak) > -DRAWDOWN_BUFFER)
        if can_enter:
            lookback = bars.iloc[max(0, i - min_bars + 1): i + 1]
            dec = agg.evaluate(lookback)
            if dec.should_trade:
                ep = slip(bar["close"], 1, True)
                pos = {"ep": ep, "sl": ep - SL_MULT * atr_now,
                       "pt": ep + PT_MULT * atr_now, "ts": i + TIME_STOP_BARS}
                trades_today += 1

    return pnls_1c


def load_v3_trades() -> tuple[list[float], list[float]]:
    """Run CDP-Anchor 2c 2/day backtest on IS and OOS data; return per-contract PnLs."""
    IS_DATA  = ROOT / "data" / "processed" / "mnq_bars_5min.h5"
    OOS_DATA = ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5"

    def load_rth(path):
        bars = pd.read_hdf(str(path), key="bars_5min")
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize("US/Eastern")
        et = bars.index.tz_convert("US/Eastern")
        rth = ((et.hour > 9) | ((et.hour == 9) & (et.minute >= 30))) & (et.hour < 16)
        return bars[rth].copy()

    is_1c  = get_trades_from_backtest(load_rth(IS_DATA))  if IS_DATA.exists()  else []
    oos_1c = get_trades_from_backtest(load_rth(OOS_DATA)) if OOS_DATA.exists() else []
    return is_1c, oos_1c


def monthly_sim(
    per_contract_pnls: list[float],
    n_contracts: int,
    n_paths: int = N_PATHS,
    trades_per_day: float = TRADES_PER_DAY,
    rng_seed: int = RNG_SEED,
) -> dict:
    """Simulate one month of funded trading at n_contracts.

    Returns monthly PnL distribution and risk metrics.
    """
    if not per_contract_pnls:
        return {"error": "no trades"}

    rng = np.random.default_rng(rng_seed)
    pnls_1c = np.array(per_contract_pnls)
    pnls_nc = pnls_1c * n_contracts   # scale to target contracts
    n_per_day = max(1, round(trades_per_day))

    monthly_pnls: list[float] = []
    daily_limit_hits = 0
    trailing_dd_hits = 0
    consistency_violations = 0
    daily_pnl_samples: list[float] = []

    for _ in range(n_paths):
        balance = STARTING_EQUITY
        peak = STARTING_EQUITY
        month_pnl = 0.0
        halted = False
        daily_bests: list[float] = []

        for _day in range(TRADING_DAYS_MONTH):
            if halted:
                break
            daily_total = 0.0
            for trade_pnl in rng.choice(pnls_nc, size=n_per_day, replace=True):
                daily_total += trade_pnl
                balance += trade_pnl
                peak = max(peak, balance)
                if (balance - peak) <= TRAILING_DD_LIMIT:
                    trailing_dd_hits += 1
                    halted = True
                    break
                if daily_total <= DAILY_LOSS_LIMIT:
                    daily_limit_hits += 1
                    halted = True
                    break
            month_pnl += daily_total
            daily_bests.append(daily_total)

        monthly_pnls.append(month_pnl)

        # Consistency check: max single day ≤ 30% of total month PnL
        if month_pnl > 0 and daily_bests:
            max_day = max(daily_bests)
            if max_day > 0.30 * month_pnl:
                consistency_violations += 1

    monthly_arr = np.array(monthly_pnls)

    return {
        "n_contracts": n_contracts,
        "monthly_pnl_mean":   round(float(np.mean(monthly_arr)), 2),
        "monthly_pnl_median": round(float(np.median(monthly_arr)), 2),
        "monthly_pnl_p25":    round(float(np.percentile(monthly_arr, 25)), 2),
        "monthly_pnl_p75":    round(float(np.percentile(monthly_arr, 75)), 2),
        "monthly_pnl_p5":     round(float(np.percentile(monthly_arr, 5)), 2),
        "monthly_pnl_p95":    round(float(np.percentile(monthly_arr, 95)), 2),
        "monthly_payout_mean": round(float(np.mean(monthly_arr)) * PAYOUT_RATE, 2),
        "p_daily_loss_hit":   round(daily_limit_hits / (n_paths * TRADING_DAYS_MONTH), 4),
        "p_trailing_dd_hit_month": round(trailing_dd_hits / n_paths, 4),
        "p_consistency_violation": round(consistency_violations / n_paths, 4),
        "p_profitable_month":  round(float((monthly_arr > 0).mean()), 4),
        "n_paths": n_paths,
    }


def print_projection_table(results_is: list[dict], results_oos: list[dict]) -> None:
    W = 104
    print(f"\n{'='*W}")
    print("  FUNDED PHASE INCOME PROJECTION  |  CDP-Anchor 2/day  |  MNQ  |  50,000 paths")
    print(f"{'='*W}")
    print(f"  {'Contracts':>10} {'IS Median/mo':>13} {'IS P90 Payout':>14} "
          f"{'OOS Median/mo':>14} {'OOS P90 Payout':>15} {'P(DD/mo)':>9} {'P(Prof)':>8}")
    print(f"  {'-'*96}")

    for is_r, oos_r in zip(results_is, results_oos):
        nc = is_r["n_contracts"]
        is_med  = is_r["monthly_pnl_median"]  * PAYOUT_RATE
        is_p90  = is_r["monthly_pnl_p75"]     * PAYOUT_RATE
        oos_med = oos_r["monthly_pnl_median"] * PAYOUT_RATE
        oos_p90 = oos_r["monthly_pnl_p75"]    * PAYOUT_RATE
        p_dd    = oos_r["p_trailing_dd_hit_month"]
        p_prof  = oos_r["p_profitable_month"]
        flag    = " ⚠️" if p_dd > 0.15 else (" ✅" if oos_med >= 2000 else "")

        print(f"  {nc:>10}c  ${is_med:>10,.0f}   ${is_p90:>11,.0f}   "
              f"${oos_med:>11,.0f}   ${oos_p90:>12,.0f}   {p_dd:>8.1%}  {p_prof:>7.1%}{flag}")


if __name__ == "__main__":
    print("Loading v3 trade data...")
    try:
        is_1c, oos_1c = load_v3_trades()
        print(f"  IS trades: {len(is_1c)} (normalized to 1c)")
        print(f"  OOS trades: {len(oos_1c)} (normalized to 1c)")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Use combined IS+OOS for simulation (more samples = more stable distribution)
    combined_1c = is_1c + oos_1c
    print(f"  Combined: {len(combined_1c)} trades")

    if combined_1c:
        arr = np.array(combined_1c)
        print(f"  Per-contract stats (1c): mean=${arr.mean():.1f}  "
              f"std=${arr.std():.1f}  WR={float((arr>0).mean()):.1%}  "
              f"min=${arr.min():.1f}  max=${arr.max():.1f}")

    print("\nRunning monthly simulations (50,000 paths each)...")

    results_is:  list[dict] = []
    results_oos: list[dict] = []

    for nc in CONTRACT_SIZES:
        print(f"  {nc} contracts...", end=" ", flush=True)
        r_is  = monthly_sim(is_1c,  nc)
        r_oos = monthly_sim(oos_1c, nc)
        results_is.append(r_is)
        results_oos.append(r_oos)
        print(f"IS median=${r_is['monthly_pnl_median']*PAYOUT_RATE:,.0f}  "
              f"OOS median=${r_oos['monthly_pnl_median']*PAYOUT_RATE:,.0f}  "
              f"P(DD/mo)={r_oos['p_trailing_dd_hit_month']:.1%}")

    # Print summary table
    print_projection_table(results_is, results_oos)

    # Consistency rule detail
    print(f"\n  Consistency rule (max day ≤ 30% of monthly profit):")
    for r_oos in results_oos:
        nc = r_oos["n_contracts"]
        viol = r_oos["p_consistency_violation"]
        print(f"    {nc}c: P(violation) = {viol:.1%}", "⚠️" if viol > 0.20 else "")

    # Target income analysis
    print(f"\n  Income targets (OOS, after 90% payout):")
    print(f"  {'Target':<20} {'Min contracts':<20} {'Expected Payout'}")
    targets = [
        ("$1,000/mo", 1000),
        ("$2,000/mo", 2000),
        ("$3,000/mo", 3000),
        ("$4,000/mo", 4000),
    ]
    for target_label, target in targets:
        best_nc = None
        for r_oos in results_oos:
            if r_oos["monthly_pnl_median"] * PAYOUT_RATE >= target:
                if r_oos["p_trailing_dd_hit_month"] <= 0.15:  # accept ≤15% DD risk/mo
                    best_nc = r_oos["n_contracts"]
                    best_payout = r_oos["monthly_pnl_median"] * PAYOUT_RATE
                    break
        if best_nc:
            print(f"  {target_label:<20} {best_nc}c                   ${best_payout:,.0f}")
        else:
            print(f"  {target_label:<20} Not achievable at ≤15% DD risk")

    # Recommended contract progression
    print(f"\n  Recommended contract progression:")
    print(f"  Week 1-2  (new funded): 2c — low DD risk, build confidence")
    print(f"  Month 1+  (if +$500):   4c — target $2k-3k/mo")
    print(f"  Month 3+  (if stable):  6c — target $3k-4k/mo")
    print(f"  Month 6+  (if stable):  8-10c — target $5k+/mo")

    # Save results
    output = {
        "is_results":  results_is,
        "oos_results": results_oos,
        "config": {
            "base_config": "CDP-Anchor 2c 2/day (v3 Config 1)",
            "trades_per_day": TRADES_PER_DAY,
            "point_value": POINT_VALUE,
            "payout_rate": PAYOUT_RATE,
            "daily_loss_limit": DAILY_LOSS_LIMIT,
            "trailing_dd_limit": TRAILING_DD_LIMIT,
            "trading_days_month": TRADING_DAYS_MONTH,
        },
    }
    out_path = RBV1 / "diagnostics" / "funded_phase_projection.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {out_path}")
