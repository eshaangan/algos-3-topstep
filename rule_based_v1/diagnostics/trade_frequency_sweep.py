"""
Trade Frequency Sweep — how to get more trades without killing WR
=================================================================
Options tested:
1. Extend entry cutoff (12:00 → 13:00, 14:00)
2. Second trade only after first is a winner (momentum continuation)
3. Afternoon session ORB — independent 13:00 range → breakout
4. Shorter OR window (5-min, 15-min) — Tsai et al. found 1-5 min optimal for US futures
5. Lower min_range_atr filter (0.3 → 0.1)
6. Multiple OR windows in sequence (30-min morning + 5-min re-entry)
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from engine.signal_aggregator import SignalAggregator
from rules.opening_range import OpeningRangeBreakoutRule
from utils.indicators import atr as compute_atr

DATA_PATH = ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5"

N_CONTRACTS = 3; POINT_VALUE = 2.0; TICK_SIZE = 0.25
COMMISSION = 0.62; SLIPPAGE_TICKS = 1
PT_MULT = 3.0; SL_MULT = 1.5; ATR_PERIOD = 14; TIME_STOP_BARS = 24
MAX_DAILY_LOSS = -950.0; STARTING_EQUITY = 50_000.0; DRAWDOWN_BUFFER = 1_950.0


@dataclass
class Pos:
    entry_price: float; entry_bar_idx: int; stop_loss: float
    profit_target: float; time_stop_bar: int; n_contracts: int; atr: float


def slip(p, d, e):
    s = SLIPPAGE_TICKS * TICK_SIZE
    return p + s * d if e else p - s * d


def calc_pnl(entry, exit_, n):
    return (exit_ - entry) * n * POINT_VALUE - 2 * COMMISSION * n


def check_exit(pos, bar, idx, sc):
    h, l, c = bar["high"], bar["low"], bar["close"]
    if sc:                       return True, slip(c, pos.n_contracts, False),            "session_close"
    if idx >= pos.time_stop_bar: return True, slip(c, 1, False),                          "time_stop"
    if l <= pos.stop_loss:       return True, slip(pos.stop_loss, 1, False),              "stop_loss"
    if h >= pos.profit_target:   return True, slip(pos.profit_target, 1, False),          "profit_target"
    return False, 0.0, ""


def run_orb_variant(
    bars,
    or_end_time="10:04",
    min_or_bars=7,
    min_range_atr=0.3,
    entry_cutoff="12:00",
    max_trades_day=1,
    require_winner_for_second=False,
    afternoon_or_start="13:00",
    afternoon_or_end="13:30",
    afternoon_cutoff="14:30",
    use_afternoon=False,
    n_base=3,
):
    """
    Main ORB with optional afternoon session.
    If use_afternoon=True, also evaluates an afternoon OR independently.
    If require_winner_for_second=True, second trade only allowed after a winning first trade.
    """
    # Morning ORB
    orb_am = OpeningRangeBreakoutRule(
        or_end_time=or_end_time, min_or_bars=min_or_bars,
        min_range_atr=min_range_atr, entry_cutoff_time=entry_cutoff,
        atr_period=ATR_PERIOD, long_only=True,
    )
    agg_am = SignalAggregator(primary_rule=orb_am, filter_rules=[], confirmation_rules=[], min_confirmations=0)

    # Afternoon ORB (if enabled)
    agg_pm = None
    if use_afternoon:
        orb_pm = OpeningRangeBreakoutRule(
            or_end_time=afternoon_or_end, min_or_bars=3,
            min_range_atr=min_range_atr, entry_cutoff_time=afternoon_cutoff,
            atr_period=ATR_PERIOD, long_only=True,
        )
        agg_pm = SignalAggregator(primary_rule=orb_pm, filter_rules=[], confirmation_rules=[], min_confirmations=0)

    atr_s    = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars = agg_am.required_bars()

    pos = None; trades = []; equity = STARTING_EQUITY; peak = STARTING_EQUITY
    max_dd = 0.0; cur_date = None; daily_pnl = {}
    trades_today = 0; daily_loss = 0.0
    first_trade_won = False  # for require_winner_for_second

    for i in range(min_bars, len(bars)):
        bar   = bars.iloc[i]
        bt    = bars.index[i]
        bt_et = bt.tz_convert("US/Eastern")
        bdate = bt_et.date()
        bh, bm = bt_et.hour, bt_et.minute

        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = daily_loss
            daily_loss = 0.0; trades_today = 0; first_trade_won = False
        cur_date = bdate

        atr_now = atr_s.iloc[i]
        if np.isnan(atr_now) or atr_now <= 0:
            continue

        is_last    = (i + 1 >= len(bars)) or bars.index[i+1].tz_convert("US/Eastern").date() != bdate
        sess_close = is_last or (bh == 15 and bm >= 55)

        if pos is not None:
            exited, exit_p, reason = check_exit(pos, bar, i, sess_close)
            if exited:
                p = calc_pnl(pos.entry_price, exit_p, pos.n_contracts)
                trades.append({
                    "date": bdate, "entry": pos.entry_price, "exit": exit_p,
                    "pnl": p, "reason": reason, "n": pos.n_contracts,
                    "trade_num": trades_today,
                })
                if trades_today == 1:
                    first_trade_won = p > 0
                equity += p; daily_loss += p
                peak    = max(peak, equity)
                max_dd  = min(max_dd, equity - peak)
                pos = None

        can_enter = (
            pos is None and not sess_close
            and trades_today < max_trades_day
            and daily_loss > MAX_DAILY_LOSS
            and (equity - peak) > -DRAWDOWN_BUFFER
        )

        # Second-trade gate
        if can_enter and trades_today >= 1 and require_winner_for_second:
            if not first_trade_won:
                can_enter = False

        if can_enter:
            # Determine which aggregator to use
            # Afternoon session: only evaluate PM agg after afternoon_or_end
            agg_to_use = None
            pm_cutoff_h = int(afternoon_cutoff.split(":")[0])
            pm_cutoff_m = int(afternoon_cutoff.split(":")[1])
            pm_end_h    = int(afternoon_or_end.split(":")[0])
            pm_end_m    = int(afternoon_or_end.split(":")[1])
            am_cutoff_h = int(entry_cutoff.split(":")[0])
            am_cutoff_m = int(entry_cutoff.split(":")[1])

            current_mins = bh * 60 + bm

            if use_afternoon and trades_today >= 1 and agg_pm is not None:
                pm_end_mins    = pm_end_h * 60 + pm_end_m
                pm_cutoff_mins = pm_cutoff_h * 60 + pm_cutoff_m
                if pm_end_mins <= current_mins <= pm_cutoff_mins:
                    agg_to_use = agg_pm
            elif trades_today == 0:
                am_cutoff_mins = am_cutoff_h * 60 + am_cutoff_m
                if current_mins <= am_cutoff_mins:
                    agg_to_use = agg_am
            elif trades_today >= 1 and not use_afternoon:
                # Regular 2nd trade from AM agg
                am_cutoff_mins = am_cutoff_h * 60 + am_cutoff_m
                if current_mins <= am_cutoff_mins:
                    agg_to_use = agg_am

            if agg_to_use is not None:
                mb = agg_to_use.required_bars()
                lookback = bars.iloc[max(0, i - mb + 1): i + 1]
                dec = agg_to_use.evaluate(lookback)
                if dec.should_trade:
                    ep = slip(bar["close"], 1, True)
                    sl = ep - SL_MULT * atr_now
                    pt = ep + PT_MULT * atr_now
                    pos = Pos(ep, i, sl, pt, i + TIME_STOP_BARS, n_base, atr_now)
                    trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = daily_loss

    wins  = [t for t in trades if t["pnl"] > 0]
    total = sum(t["pnl"] for t in trades)
    gp    = sum(t["pnl"] for t in wins)
    gl    = abs(sum(t["pnl"] for t in [t for t in trades if t["pnl"] <= 0]))
    daily = pd.Series(daily_pnl)
    daily = daily[daily != 0]
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0

    t1 = [t for t in trades if t["trade_num"] == 1]
    t2 = [t for t in trades if t["trade_num"] == 2]
    t1_wr = sum(1 for t in t1 if t["pnl"] > 0) / max(len(t1), 1)
    t2_wr = sum(1 for t in t2 if t["pnl"] > 0) / max(len(t2), 1)

    return {
        "n": len(trades), "wr": len(wins) / max(len(trades), 1),
        "pnl": total, "sharpe": sharpe, "dd": max_dd,
        "mll": max_dd > -2000, "pf": gp / gl if gl > 0 else float("inf"),
        "t1_n": len(t1), "t1_wr": t1_wr,
        "t2_n": len(t2), "t2_wr": t2_wr,
        "trades": trades, "daily_pnl": daily_pnl,
    }


def print_row(label, r, W=42):
    mll = "✅" if r["mll"] else "❌"
    t2_info = f"  [2nd: {r['t2_n']}t {r['t2_wr']:.0%}]" if r["t2_n"] > 0 else ""
    print(f"  {label:<{W}} {r['n']:>4}  {r['wr']:>5.1%} ${r['pnl']:>8,.0f} "
          f"{r['sharpe']:>7.2f} ${r['dd']:>7,.0f}  {mll}{t2_info}")


def monthly(r):
    m = defaultdict(list)
    for t in r["trades"]: m[str(t["date"])[:7]].append(t)
    return {ym: (len(ts), sum(1 for t in ts if t["pnl"] > 0) / len(ts), sum(t["pnl"] for t in ts))
            for ym, ts in m.items()}


if __name__ == "__main__":
    bars = pd.read_hdf(str(DATA_PATH), key="bars_5min")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")

    W = 44
    print(f"\n{'='*84}")
    print(f"  TRADE FREQUENCY SWEEP — MNQ 2026 YTD  |  3c  |  LONG-only  |  PT=3.0x SL=1.5x")
    print(f"{'='*84}")
    print(f"  {'Config':<{W}} {'N':>4} {'WR':>6} {'PnL':>10} {'Sharpe':>7} {'MaxDD':>9}  MLL?")
    print(f"  {'-'*78}")

    results = {}

    # Baseline
    r = run_orb_variant(bars)
    results["Baseline 1t/day"] = r
    print_row("Baseline 1t/day (live)", r, W)

    # ── Group 1: Extend entry cutoff ────────────────────────────────────────
    print(f"\n  -- Extend morning entry cutoff --")
    for cutoff in ["13:00", "14:00", "15:00"]:
        label = f"Entry cutoff {cutoff}"
        r = run_orb_variant(bars, entry_cutoff=cutoff, max_trades_day=1)
        results[label] = r
        print_row(label, r, W)

    # ── Group 2: Allow 2 trades/day ─────────────────────────────────────────
    print(f"\n  -- Allow 2 trades/day --")

    r = run_orb_variant(bars, max_trades_day=2)
    results["2 trades/day (any)"] = r
    print_row("2 trades/day (any)", r, W)

    r = run_orb_variant(bars, max_trades_day=2, require_winner_for_second=True)
    results["2 trades/day (2nd only after win)"] = r
    print_row("2 trades/day (2nd only after win)", r, W)

    r = run_orb_variant(bars, entry_cutoff="13:00", max_trades_day=2)
    results["2t/day + cutoff 13:00"] = r
    print_row("2t/day + cutoff 13:00", r, W)

    r = run_orb_variant(bars, entry_cutoff="13:00", max_trades_day=2, require_winner_for_second=True)
    results["2t/day + winner gate + 13:00"] = r
    print_row("2t/day + winner gate + 13:00", r, W)

    # ── Group 3: Shorter OR windows (Tsai et al. — shorter = more signals) ─
    print(f"\n  -- Shorter OR windows (Tsai et al. 2019) --")
    for end_time, min_bars_or, label in [
        ("09:44",  3, "5-min OR  (09:30-09:44, 3 bars)"),
        ("09:59",  6, "15-min OR (09:30-09:59, 6 bars)"),
        ("10:04",  7, "30-min OR (09:30-10:04, 7 bars) [live]"),
        ("10:29", 12, "45-min OR (09:30-10:29, 12 bars)"),
    ]:
        r = run_orb_variant(bars, or_end_time=end_time, min_or_bars=min_bars_or,
                             entry_cutoff="12:00", max_trades_day=1)
        results[label] = r
        print_row(label, r, W)

    # ── Group 4: Lower ATR filter ───────────────────────────────────────────
    print(f"\n  -- Lower min_range_atr filter --")
    for atr_thresh in [0.1, 0.2, 0.3, 0.5]:
        label = f"min_range_atr={atr_thresh}"
        r = run_orb_variant(bars, min_range_atr=atr_thresh, max_trades_day=1)
        results[label] = r
        print_row(label, r, W)

    # ── Group 5: Afternoon session ──────────────────────────────────────────
    print(f"\n  -- Afternoon OR session (independent post-lunch range) --")

    r = run_orb_variant(bars, max_trades_day=2, use_afternoon=True,
                         afternoon_or_start="13:00", afternoon_or_end="13:30",
                         afternoon_cutoff="14:30")
    results["AM + PM session (13:00-13:30 OR)"] = r
    print_row("AM + PM session (13:00-13:30 OR)", r, W)

    r = run_orb_variant(bars, max_trades_day=2, use_afternoon=True,
                         afternoon_or_start="13:30", afternoon_or_end="14:00",
                         afternoon_cutoff="15:00")
    results["AM + PM session (13:30-14:00 OR)"] = r
    print_row("AM + PM session (13:30-14:00 OR)", r, W)

    # ── Group 6: Short OR + allow 2 trades ─────────────────────────────────
    print(f"\n  -- Short OR (15-min) + 2 trades/day --")
    r = run_orb_variant(bars, or_end_time="09:59", min_or_bars=6,
                         entry_cutoff="13:00", max_trades_day=2)
    results["15-min OR + 2t/day"] = r
    print_row("15-min OR + 2t/day", r, W)

    r = run_orb_variant(bars, or_end_time="09:59", min_or_bars=6,
                         entry_cutoff="13:00", max_trades_day=2,
                         require_winner_for_second=True)
    results["15-min OR + 2t/day + winner gate"] = r
    print_row("15-min OR + 2t/day + winner gate", r, W)

    # ── Top performers ──────────────────────────────────────────────────────
    print(f"\n{'='*84}")
    print(f"  TOP RESULTS (MLL-safe, ≥30 trades, sorted by Sharpe)")
    print(f"{'='*84}")

    top = sorted(
        [(k, v) for k, v in results.items() if v["mll"] and v["n"] >= 30],
        key=lambda x: x[1]["sharpe"], reverse=True
    )[:6]

    for label, r in top:
        ms = monthly(r)
        t2_info = f"  2nd-trade: {r['t2_n']}t WR={r['t2_wr']:.0%}" if r["t2_n"] > 0 else ""
        print(f"\n  [{label}]  N={r['n']} WR={r['wr']:.1%} PnL=${r['pnl']:,.0f} "
              f"Sharpe={r['sharpe']:.2f} DD=${r['dd']:,.0f} {'✅' if r['mll'] else '❌'}{t2_info}")
        print(f"  {'Month':<10} {'N':>4} {'WR':>6} {'PnL':>10} {'Cumul':>12}")
        print(f"  {'-'*44}")
        cum = 0.0
        for ym in sorted(ms):
            n, wr, pnl = ms[ym]; cum += pnl
            print(f"  {ym:<10} {n:>4}  {wr:>5.1%} ${pnl:>8,.0f}  ${cum:>10,.0f}")

    if not top:
        print(f"\n  No config with ≥30 trades passes MLL. Lowering threshold to ≥25:")
        top = sorted(
            [(k, v) for k, v in results.items() if v["mll"] and v["n"] >= 25],
            key=lambda x: x[1]["sharpe"], reverse=True
        )[:6]
        for label, r in top:
            ms = monthly(r)
            t2 = f"  2nd: {r['t2_n']}t/{r['t2_wr']:.0%}" if r["t2_n"] > 0 else ""
            print(f"\n  [{label}]  N={r['n']} WR={r['wr']:.1%} PnL=${r['pnl']:,.0f} "
                  f"Sharpe={r['sharpe']:.2f} DD=${r['dd']:,.0f} {'✅' if r['mll'] else '❌'}{t2}")
            cum = 0.0
            for ym in sorted(ms):
                n, wr, pnl = ms[ym]; cum += pnl
                print(f"  {ym:<10} {n:>4}  {wr:>5.1%} ${pnl:>8,.0f}  ${cum:>10,.0f}")
