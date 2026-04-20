"""
Regime Quality Sweep — Three High-Value Edges for 2026 Tariff-Fear Regime
==========================================================================
Tests three data-validated patterns against MNQ 2026 YTD:

1. PT Optimization — PT=3x is too far; most winners exit on time stops.
   Sweep PT=[1.5, 2.0, 2.5, 3.0] to find the Sharpe-optimal multiplier.

2. Gap+PrevVWAP "Double Momentum" — trade only when gap direction AND
   PrevVWAP both confirm bullish flow.

3. Bear Streak Reversal — trade only on "first bull day" after ≥N consecutive
   bearish PrevVWAP closes (institutions reload after extended selling).

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/regime_quality_sweep.py
    python rule_based_v1/diagnostics/regime_quality_sweep.py --save
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

# Re-use helpers from novel_filter_sweep without re-implementing them
from rule_based_v1.diagnostics.novel_filter_sweep import (
    build_day_meta,
    enrich_prev_vwap,
    make_prev_vwap_filter,
    make_combined_filter,
    _calc_pnl,
    _slip,
    print_row,
    monthly_wr,
    print_monthly,
    Pos,
    TICK_SIZE,
    SLIPPAGE_TICKS,
    COMMISSION,
    ATR_PERIOD,
    SL_MULT,
    TIME_STOP_BARS,
    MAX_DAILY_LOSS,
    MAX_TRADES_DAY,
    STARTING_EQUITY,
    DRAWDOWN_BUFFER,
    N_CONTRACTS,
)

DATA_PATH    = ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5"
RESULTS_PATH = ROOT / "rule_based_v1" / "diagnostics" / "regime_quality_results.json"


# ── Exit check supporting both LONG ───────────────────────────────────────────
def _check_exit_long(pos: Pos, bar, idx: int, sess_close: bool):
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


# ── Streak enrichment ─────────────────────────────────────────────────────────
def enrich_streak(meta: dict) -> dict:
    """
    Add bearish_streak_ending_yesterday: count of consecutive bearish closes
    immediately before today. e.g., if yesterday and the day before were bearish
    but the day before that was bullish, value = 2.
    """
    dates_sorted = sorted(meta.keys())

    # Build a streak counter that increments for each consecutive bearish day
    streak = 0
    streak_by_date = {}
    for d in dates_sorted:
        pv = meta[d].get("prev_vwap_bullish")
        if pv is False:
            streak += 1
        elif pv is True:
            streak = 0
        # pv=None means no data, treat as not bearish (don't extend streak)
        streak_by_date[d] = streak

    # bearish_streak_ending_yesterday[d] = streak as of yesterday
    for i, d in enumerate(dates_sorted):
        if i == 0:
            meta[d]["bearish_streak_ending_yesterday"] = 0
        else:
            prev_d = dates_sorted[i - 1]
            meta[d]["bearish_streak_ending_yesterday"] = streak_by_date[prev_d]

    return meta


# ── Parameterized ORB backtest ─────────────────────────────────────────────────
def run_orb_parameterized(
    bars: pd.DataFrame,
    day_meta: dict,
    pt_mult: float = 3.0,
    filter_fn=None,
    label: str = "",
) -> dict:
    """
    Run ORB backtest with configurable PT multiplier.
    Records exit reasons for exit distribution analysis.
    """
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
            exited, exit_p, reason = _check_exit_long(pos, bar, i, sess_close)
            if exited:
                p = _calc_pnl(pos.entry_price, exit_p, pos.n_contracts)
                trades.append({
                    "date":    bdate,
                    "entry":   pos.entry_price,
                    "exit":    exit_p,
                    "pnl":     p,
                    "reason":  reason,
                    "n":       pos.n_contracts,
                    "atr":     pos.atr,
                    "prev_vwap_bullish":             meta.get("prev_vwap_bullish"),
                    "gap_pct":                       meta.get("gap_pct"),
                    "bearish_streak_ending_yesterday": meta.get("bearish_streak_ending_yesterday", 0),
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
                    pt = ep + pt_mult * atr_now
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

    exit_dist = defaultdict(int)
    for t in trades:
        exit_dist[t["reason"]] += 1

    return {
        "label":     label,
        "n":         len(trades),
        "wr":        len(wins) / max(len(trades), 1),
        "pnl":       total,
        "sharpe":    sharpe,
        "dd":        max_dd,
        "mll":       max_dd > -2000,
        "pf":        gp / gl if gl > 0 else float("inf"),
        "avg_win":   sum(t["pnl"] for t in wins) / max(len(wins), 1),
        "avg_loss":  sum(t["pnl"] for t in trades if t["pnl"] <= 0) / max(len([t for t in trades if t["pnl"] <= 0]), 1),
        "expectancy": total / max(len(trades), 1),
        "exit_dist": dict(exit_dist),
        "trades":    trades,
        "daily_pnl": dict(daily_pnl),
    }


# ── New filters ───────────────────────────────────────────────────────────────
def make_gap_prevvwap_filter(min_gap_pct: float = 0.001):
    """
    Allow trade only when gap is up >= min_gap_pct AND previous close > VWAP.
    Double momentum confirmation: institutional flow aligned at session start.
    """
    def fn(meta):
        pv = meta.get("prev_vwap_bullish")
        if pv is False or pv is None:
            return False
        gap = meta.get("gap_pct", 0.0)
        return gap >= min_gap_pct
    return fn


def make_streak_reversal_filter(min_streak: int = 2):
    """
    Allow trade only when today is bullish PrevVWAP AND at least min_streak
    consecutive bearish days just ended (mean-reversion buy-the-dip signal).
    """
    def fn(meta):
        pv = meta.get("prev_vwap_bullish")
        if pv is not True:
            return False
        streak = meta.get("bearish_streak_ending_yesterday", 0)
        return streak >= min_streak
    return fn


# ── Print helpers ─────────────────────────────────────────────────────────────
def print_exit_dist(r: dict, label: str) -> None:
    total = r["n"]
    if total == 0:
        return
    ed = r["exit_dist"]
    parts = []
    for reason in ["profit_target", "time_stop", "stop_loss", "session_close"]:
        n = ed.get(reason, 0)
        parts.append(f"{reason.replace('_', ' ')}: {n} ({n/total:.0%})")
    print(f"    {label}: " + " | ".join(parts))


def print_result_row(label: str, r: dict, width: int = 52) -> None:
    mll = "✅" if r["mll"] else "❌"
    exp = r.get("expectancy", 0)
    print(f"  {label:<{width}} {r['n']:>4}  {r['wr']:>5.1%} ${r['pnl']:>8,.0f} "
          f"{r['sharpe']:>6.2f} ${r['dd']:>7,.0f}  exp${exp:>6,.0f}  {mll}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    print(f"\nLoading MNQ 2026 YTD bars from {DATA_PATH} ...")
    bars = pd.read_hdf(str(DATA_PATH), key="bars_5min")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    print(f"  {len(bars):,} 5-min bars  {bars.index[0].date()} → {bars.index[-1].date()}")

    # Build metadata (no yfinance needed — uses only price data)
    print("\nBuilding day metadata ...")
    meta = build_day_meta(bars)
    meta = enrich_prev_vwap(bars, meta)
    meta = enrich_streak(meta)

    # Quick metadata diagnostics
    dates_sorted = sorted(meta.keys())
    pv_bull = sum(1 for d in dates_sorted if meta[d].get("prev_vwap_bullish") is True)
    pv_bear = sum(1 for d in dates_sorted if meta[d].get("prev_vwap_bullish") is False)
    gap_up  = sum(1 for d in dates_sorted if meta[d].get("gap_pct", 0) > 0)
    print(f"  {len(dates_sorted)} trading days | PrevVWAP bull={pv_bull} bear={pv_bear}")
    print(f"  Gap-up days: {gap_up}/{len(dates_sorted)} ({gap_up/max(len(dates_sorted),1):.0%})")

    # Show streak distribution
    streaks = [meta[d]["bearish_streak_ending_yesterday"] for d in dates_sorted]
    print(f"  Streak distribution (bearish days before today):")
    for s in range(0, 11):
        n = sum(1 for x in streaks if x == s)
        if n > 0:
            first_bull_after = sum(
                1 for d in dates_sorted
                if meta[d]["bearish_streak_ending_yesterday"] == s
                and meta[d].get("prev_vwap_bullish") is True
            )
            print(f"    streak={s}: {n} days, {first_bull_after} are 'first bull after'")

    results = {}
    header_w = 52

    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*92}")
    print(f"  GROUP 1: PT MULTIPLIER OPTIMIZATION")
    print(f"  Hypothesis: PT=3x is too far — time stops dominate (79% WR) but PT hits are rare")
    print(f"{'='*92}")
    print(f"  {'Config':<{header_w}} {'N':>4}  {'WR':>5}  {'PnL':>9}  {'Sharpe':>6} {'MaxDD':>8}  {'Exp$/t':>7}  MLL")
    print(f"  {'-'*90}")

    for pt_mult in [1.5, 2.0, 2.5, 3.0]:
        lbl = f"PT={pt_mult:.1f}x SL=1.5x  (baseline PrevVWAP)"
        r = run_orb_parameterized(bars, meta, pt_mult=pt_mult, filter_fn=make_prev_vwap_filter(), label=lbl)
        results[f"pt_{pt_mult:.1f}"] = r
        print_result_row(lbl, r, header_w)

    print(f"\n  Exit distribution by PT multiplier:")
    for pt_mult in [1.5, 2.0, 2.5, 3.0]:
        print_exit_dist(results[f"pt_{pt_mult:.1f}"], f"PT={pt_mult:.1f}x")

    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*92}")
    print(f"  GROUP 2: GAP + PREVVWAP DOUBLE MOMENTUM FILTER")
    print(f"  Hypothesis: gap-up + bullish VWAP = highest-conviction LONG setup")
    print(f"{'='*92}")
    print(f"  {'Config':<{header_w}} {'N':>4}  {'WR':>5}  {'PnL':>9}  {'Sharpe':>6} {'MaxDD':>8}  {'Exp$/t':>7}  MLL")
    print(f"  {'-'*90}")

    # Baseline: PrevVWAP only (PT=3x, for comparison)
    r_pv = run_orb_parameterized(bars, meta, pt_mult=3.0, filter_fn=make_prev_vwap_filter(), label="PrevVWAP only [reference]")
    results["prevvwap_ref"] = r_pv
    print_result_row("PrevVWAP only [reference, PT=3x]", r_pv, header_w)

    for min_gap in [0.0, 0.001, 0.002, 0.003, 0.005]:
        lbl = f"Gap≥{min_gap*100:.1f}% + PrevVWAP  (PT=3x)"
        r = run_orb_parameterized(
            bars, meta, pt_mult=3.0,
            filter_fn=make_gap_prevvwap_filter(min_gap),
            label=lbl,
        )
        results[f"gap_{min_gap:.3f}"] = r
        print_result_row(lbl, r, header_w)

    # Also test best gap threshold with PT=2x
    print(f"\n  Gap filter variants with PT=2x (potentially better exit timing):")
    for min_gap in [0.001, 0.002, 0.003]:
        lbl = f"Gap≥{min_gap*100:.1f}% + PrevVWAP  (PT=2x)"
        r = run_orb_parameterized(
            bars, meta, pt_mult=2.0,
            filter_fn=make_gap_prevvwap_filter(min_gap),
            label=lbl,
        )
        results[f"gap_{min_gap:.3f}_pt2"] = r
        print_result_row(lbl, r, header_w)

    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*92}")
    print(f"  GROUP 3: BEAR STREAK REVERSAL FILTER")
    print(f"  Hypothesis: 'First bull after N bearish' = institutional reload signal")
    print(f"{'='*92}")
    print(f"  {'Config':<{header_w}} {'N':>4}  {'WR':>5}  {'PnL':>9}  {'Sharpe':>6} {'MaxDD':>8}  {'Exp$/t':>7}  MLL")
    print(f"  {'-'*90}")

    for min_streak in [1, 2, 3, 4, 5]:
        lbl = f"First bull after ≥{min_streak} bear streak  (PT=3x)"
        r = run_orb_parameterized(
            bars, meta, pt_mult=3.0,
            filter_fn=make_streak_reversal_filter(min_streak),
            label=lbl,
        )
        results[f"streak_{min_streak}"] = r
        print_result_row(lbl, r, header_w)

    # Best streak + PT=2x combo
    print(f"\n  Streak variants with PT=2x:")
    for min_streak in [1, 2, 3]:
        lbl = f"First bull after ≥{min_streak} bear streak  (PT=2x)"
        r = run_orb_parameterized(
            bars, meta, pt_mult=2.0,
            filter_fn=make_streak_reversal_filter(min_streak),
            label=lbl,
        )
        results[f"streak_{min_streak}_pt2"] = r
        print_result_row(lbl, r, header_w)

    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*92}")
    print(f"  TOP CONFIGS BY SHARPE (N≥10)")
    print(f"{'='*92}")
    print(f"  {'Config':<{header_w}} {'N':>4}  {'WR':>5}  {'PnL':>9}  {'Sharpe':>6} {'MaxDD':>8}  {'Exp$/t':>7}  MLL")
    print(f"  {'-'*90}")
    ranked = sorted(
        [(k, v) for k, v in results.items() if v["n"] >= 10],
        key=lambda x: x[1]["sharpe"], reverse=True
    )
    for k, r in ranked[:8]:
        print_result_row(r["label"], r, header_w)

    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*92}")
    print(f"  DECISION SUMMARY")
    print(f"{'='*92}")

    ref_sharpe = results["prevvwap_ref"]["sharpe"]
    print(f"  Reference (PrevVWAP only, PT=3x): Sharpe={ref_sharpe:.2f}, "
          f"N={results['prevvwap_ref']['n']}, WR={results['prevvwap_ref']['wr']:.1%}")

    print(f"\n  PT Optimization:")
    for pt in [1.5, 2.0, 2.5, 3.0]:
        r = results[f"pt_{pt:.1f}"]
        delta = r["sharpe"] - ref_sharpe
        marker = " ← BETTER" if delta > 0.2 else ""
        print(f"    PT={pt:.1f}x: Sharpe={r['sharpe']:.2f} (Δ{delta:+.2f}){marker}")

    print(f"\n  Gap+PrevVWAP filter:")
    for min_gap in [0.0, 0.001, 0.002, 0.003, 0.005]:
        r = results[f"gap_{min_gap:.3f}"]
        delta = r["sharpe"] - ref_sharpe
        n_ratio = r["n"] / max(results["prevvwap_ref"]["n"], 1)
        marker = " ← DEPLOY CANDIDATE" if delta > 0.3 and r["wr"] >= 0.65 and r["n"] >= 15 else ""
        print(f"    Gap≥{min_gap*100:.1f}%: Sharpe={r['sharpe']:.2f} (Δ{delta:+.2f}), "
              f"N={r['n']} ({n_ratio:.0%} of baseline){marker}")

    print(f"\n  Bear Streak Reversal:")
    for min_streak in [1, 2, 3, 4, 5]:
        r = results[f"streak_{min_streak}"]
        if r["n"] == 0:
            print(f"    Streak≥{min_streak}: N=0 (no setups in dataset)")
            continue
        delta = r["sharpe"] - ref_sharpe
        marker = " ← SIZE UP CANDIDATE" if r["wr"] >= 0.70 and r["n"] >= 5 else ""
        print(f"    Streak≥{min_streak}: Sharpe={r['sharpe']:.2f} (Δ{delta:+.2f}), "
              f"N={r['n']}, WR={r['wr']:.1%}{marker}")

    # ── Save results ─────────────────────────────────────────────────────────
    if args.save:
        serializable = {}
        for k, v in results.items():
            sv = {kk: vv for kk, vv in v.items() if kk not in ("trades", "daily_pnl")}
            sv["exit_dist"] = v.get("exit_dist", {})
            sv["monthly"]   = {m: list(vals) for m, vals in monthly_wr(v).items()}
            serializable[k] = sv
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(serializable, indent=2, default=str))
        print(f"\n  Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
