"""
ORB Regime Filter Sweep — what works in bearish/volatile markets?
Tests filters on top of the live ORB LONG-only strategy.
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

POINT_VALUE = 2.0; TICK_SIZE = 0.25; COMMISSION = 0.62; SLIPPAGE_TICKS = 1
PT_MULT = 3.0; SL_MULT = 1.5; ATR_PERIOD = 14; TIME_STOP_BARS = 24
MAX_DAILY_LOSS = -950.0; MAX_TRADES_DAY = 1
STARTING_EQUITY = 50_000.0; DRAWDOWN_BUFFER = 1_950.0


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
    if sc:              return True, slip(c, 1, False), "session_close"
    if idx >= pos.time_stop_bar: return True, slip(c, 1, False), "time_stop"
    if l <= pos.stop_loss:       return True, slip(pos.stop_loss, 1, False), "stop_loss"
    if h >= pos.profit_target:   return True, slip(pos.profit_target, 1, False), "profit_target"
    return False, 0.0, ""


def build_daily_regime(bars):
    daily = bars.resample("1D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),   close=("close", "last")
    ).dropna()
    if daily.index.tz is None:
        daily.index = daily.index.tz_localize("US/Eastern")

    daily["ema20"]    = daily["close"].ewm(span=20, adjust=False).mean()
    daily["atr14"]    = compute_atr(daily["high"], daily["low"], daily["close"], 14)
    daily["atr_avg"]  = daily["atr14"].rolling(20).mean()
    daily["atr_ratio"] = daily["atr14"] / daily["atr_avg"]
    daily["gap_pct"]  = (daily["open"] - daily["close"].shift(1)) / daily["close"].shift(1)

    regime_by_date = {}
    for ts, row in daily.iterrows():
        d = ts.tz_convert("US/Eastern").date()
        regime_by_date[d] = {
            "above_ema20": bool(row["close"] > row["ema20"]),
            "atr_ratio":   float(row["atr_ratio"]) if not np.isnan(row["atr_ratio"]) else 1.0,
            "gap_pct":     float(row["gap_pct"])   if not np.isnan(row["gap_pct"]) else 0.0,
        }
    return regime_by_date


def run_orb(bars, regime_by_date, n_base=3, filter_fn=None, scale_fn=None):
    orb = OpeningRangeBreakoutRule(
        or_end_time="10:04", min_or_bars=7, min_range_atr=0.3,
        entry_cutoff_time="12:00", atr_period=ATR_PERIOD, long_only=True,
    )
    agg = SignalAggregator(primary_rule=orb, filter_rules=[], confirmation_rules=[], min_confirmations=0)
    atr_s     = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars  = agg.required_bars()

    pos = None; trades = []; equity = STARTING_EQUITY; peak = STARTING_EQUITY
    max_dd = 0.0; cur_date = None; daily_pnl = {}
    trades_today = 0; daily_loss = 0.0

    for i in range(min_bars, len(bars)):
        bar   = bars.iloc[i]
        bt    = bars.index[i]
        bt_et = bt.tz_convert("US/Eastern")
        bdate = bt_et.date()
        bh, bm = bt_et.hour, bt_et.minute

        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = daily_loss
            daily_loss = 0.0
            trades_today = 0
        cur_date = bdate

        regime  = regime_by_date.get(bdate, {})
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
                    "pnl": p, "reason": reason, "n": pos.n_contracts, "atr": pos.atr,
                    "above_ema": regime.get("above_ema20", True),
                })
                equity += p; daily_loss += p
                peak    = max(peak, equity)
                max_dd  = min(max_dd, equity - peak)
                pos = None

        can_enter = (
            pos is None and not sess_close
            and trades_today < MAX_TRADES_DAY
            and daily_loss > MAX_DAILY_LOSS
            and (equity - peak) > -DRAWDOWN_BUFFER
        )

        if can_enter:
            should_trade = True
            if filter_fn:
                should_trade = filter_fn(regime)

            if should_trade:
                lookback = bars.iloc[max(0, i - min_bars + 1): i + 1]
                dec = agg.evaluate(lookback)
                if dec.should_trade:
                    n  = scale_fn(regime, n_base) if scale_fn else n_base
                    ep = slip(bar["close"], 1, True)
                    sl = ep - SL_MULT * atr_now
                    pt = ep + PT_MULT * atr_now
                    pos = Pos(ep, i, sl, pt, i + TIME_STOP_BARS, n, atr_now)
                    trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = daily_loss

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total  = sum(t["pnl"] for t in trades)
    gp     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))
    daily  = pd.Series(daily_pnl)
    daily  = daily[daily != 0]
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0

    return {
        "n": len(trades), "wr": len(wins) / max(len(trades), 1),
        "pnl": total, "sharpe": sharpe, "dd": max_dd,
        "mll": max_dd > -2000, "trades": trades, "daily_pnl": daily_pnl,
    }


def monthly_stats(r):
    m = defaultdict(list)
    for t in r["trades"]:
        m[str(t["date"])[:7]].append(t)
    return {
        ym: (len(ts), sum(1 for t in ts if t["pnl"] > 0) / len(ts), sum(t["pnl"] for t in ts))
        for ym, ts in m.items()
    }


if __name__ == "__main__":
    bars = pd.read_hdf(str(DATA_PATH), key="bars_5min")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")

    regime_by_date = build_daily_regime(bars)

    # Print current regime state
    print("\nCurrent regime (last 10 trading days):")
    print(f"  {'Date':<12} {'AboveEMA20':>11} {'ATR-ratio':>10} {'Gap%':>7}")
    print("  " + "-" * 44)
    sorted_dates = sorted(regime_by_date.keys())
    for d in sorted_dates[-10:]:
        r = regime_by_date[d]
        em = "YES" if r["above_ema20"] else "NO "
        print(f"  {str(d):<12} {em:>11} {r['atr_ratio']:>10.2f} {r['gap_pct']*100:>6.2f}%")

    # Filters
    def f_above_ema(r):       return r.get("above_ema20", True)
    def f_no_gap_down(r):     return r.get("gap_pct", 0) >= -0.005
    def f_no_big_gap(r):      return r.get("gap_pct", 0) >= -0.003
    def f_low_atr(r):         return r.get("atr_ratio", 1.0) <= 1.15
    def f_ema_or_gap(r):      return f_above_ema(r) or f_no_gap_down(r)
    def f_ema_and_gap(r):     return f_above_ema(r) and f_no_gap_down(r)

    def s_2c_below_ema(r, n): return n if r.get("above_ema20", True) else max(1, n - 1)
    def s_2c_high_atr(r, n):  return n if r.get("atr_ratio", 1.0) <= 1.15 else max(1, n - 1)
    def s_2c_bearish(r, n):   return n if r.get("above_ema20", True) else max(1, n - 1)

    configs = [
        ("Baseline 3c (live)",          3, None,          None),
        ("Skip if below EMA20",         3, f_above_ema,   None),
        ("Skip if gap down >0.3%",      3, f_no_big_gap,  None),
        ("Skip if ATR ratio >1.15",     3, f_low_atr,     None),
        ("Skip if below EMA AND gap↓",  3, f_ema_and_gap, None),
        ("2c when below EMA20",         3, None,          s_2c_below_ema),
        ("2c when ATR ratio >1.15",     3, None,          s_2c_high_atr),
        ("2c below EMA + skip gap↓",    3, f_no_gap_down, s_2c_bearish),
    ]

    print(f"\n{'='*78}")
    print(f"  ORB LONG-only — Regime Filter Sweep  |  MNQ 2026 YTD  |  PT=3.0x SL=1.5x")
    print(f"{'='*78}")
    print(f"  {'Config':<32} {'N':>4} {'WR':>6} {'PnL':>10} {'Sharpe':>7} {'MaxDD':>9}  MLL?")
    print(f"  {'-'*72}")

    results = {}
    for label, n, filt, scale in configs:
        r = run_orb(bars, regime_by_date, n, filt, scale)
        mll = "✅" if r["mll"] else "❌"
        print(f"  {label:<32} {r['n']:>4}  {r['wr']:>5.1%} ${r['pnl']:>8,.0f} {r['sharpe']:>7.2f} ${r['dd']:>7,.0f}  {mll}")
        results[label] = r

    # Monthly breakdown comparison
    print(f"\n  Monthly breakdown: Baseline vs best filters")
    print(f"  {'Month':<10} {'Baseline':>18}  {'2c-below-EMA':>18}  {'Skip-below-EMA':>18}")
    print(f"  {'-'*70}")

    bl  = monthly_stats(results["Baseline 3c (live)"])
    sc  = monthly_stats(results["2c when below EMA20"])
    sk  = monthly_stats(results["Skip if below EMA20"])

    for ym in sorted(set(bl) | set(sc) | set(sk)):
        def fmt(d, k):
            if k not in d: return "       —          "
            n, wr, p = d[k]
            return f"{n:>2}t {wr:>5.1%} ${p:>+7,.0f}"
        print(f"  {ym:<10} {fmt(bl, ym)}  {fmt(sc, ym)}  {fmt(sk, ym)}")

    # Which days did EMA filter skip? (below EMA days)
    below_ema_dates = [d for d, r in regime_by_date.items() if not r.get("above_ema20", True)]
    print(f"\n  Days below EMA20: {len(below_ema_dates)}")
    print(f"  Below-EMA trades in baseline:")
    for t in results["Baseline 3c (live)"]["trades"]:
        if not t.get("above_ema", True):
            print(f"    {t['date']}  ${t['pnl']:>+7,.0f}  [{t['reason']}]")
