"""
Research-Backed Filter Sweep — MNQ ORB 2026 YTD
================================================
Tests 3 academically-grounded filters on top of live ORB LONG-only config:

1. Economic Calendar Filter (Andersen et al. 2002)
   - NFP, CPI, FOMC announcement days have stronger/weaker ORB signals
   - Test: skip these days | only trade these days | boost/reduce size

2. Overnight Gap vs ORB Direction (Lou et al. 2019, Stübinger & Schneider 2019)
   - Gap same direction as ORB → retail-driven, weaker → skip or reduce size
   - Gap opposing ORB direction → institutional confirmation → keep or boost
   - Test: skip gap-confirms, skip gap-large-confirms, size-reduce on confirms

3. ATR Volatility Percentile Filter (Baltussen et al. 2021, Lundström 2017)
   - ORB returns scale with volatility
   - Only trade when ATR > Nth percentile of trailing 60-day ATR distribution
   - Test: 30th, 40th, 50th percentile thresholds

Also tests all combinations that pass MLL.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from engine.signal_aggregator import SignalAggregator
from rules.opening_range import OpeningRangeBreakoutRule
from utils.indicators import atr as compute_atr

DATA_PATH = ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5"

# ── Execution config (mirrors live) ─────────────────────────────────────────
N_CONTRACTS = 3; POINT_VALUE = 2.0; TICK_SIZE = 0.25
COMMISSION = 0.62; SLIPPAGE_TICKS = 1
PT_MULT = 3.0; SL_MULT = 1.5; ATR_PERIOD = 14; TIME_STOP_BARS = 24
MAX_DAILY_LOSS = -950.0; MAX_TRADES_DAY = 1
STARTING_EQUITY = 50_000.0; DRAWDOWN_BUFFER = 1_950.0

# ── Known macro event dates (2026 YTD) ──────────────────────────────────────
# NFP: first Friday of each month
# CPI: mid-month Wednesday
# FOMC: meeting announcement days
MACRO_DATES = {
    # NFP
    datetime.date(2026, 1,  9): "NFP",
    datetime.date(2026, 2,  6): "NFP",
    datetime.date(2026, 3,  6): "NFP",
    # CPI
    datetime.date(2026, 1, 15): "CPI",
    datetime.date(2026, 2, 12): "CPI",
    datetime.date(2026, 3, 12): "CPI",
    # FOMC announcement days
    datetime.date(2026, 1, 29): "FOMC",
    datetime.date(2026, 3, 19): "FOMC",
}


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
    if sc:                       return True, slip(c, 1, False),              "session_close"
    if idx >= pos.time_stop_bar: return True, slip(c, 1, False),              "time_stop"
    if l <= pos.stop_loss:       return True, slip(pos.stop_loss, 1, False),  "stop_loss"
    if h >= pos.profit_target:   return True, slip(pos.profit_target, 1, False), "profit_target"
    return False, 0.0, ""


def build_day_meta(bars):
    """Pre-compute per-day metadata: gap%, daily ATR, ATR percentile, macro event."""
    daily = bars.resample("1D").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),   close=("close", "last"),
    ).dropna()
    if daily.index.tz is None:
        daily.index = daily.index.tz_localize("US/Eastern")

    daily["atr14"]    = compute_atr(daily["high"], daily["low"], daily["close"], 14)
    daily["gap_pct"]  = (daily["open"] - daily["close"].shift(1)) / daily["close"].shift(1)

    # Rolling 60-day ATR percentile rank
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
            "gap_pct":   float(row["gap_pct"])  if not np.isnan(row["gap_pct"])  else 0.0,
            "atr14":     float(row["atr14"])     if not np.isnan(row["atr14"])    else 300.0,
            "atr_pct":   float(row["atr_pct"])   if not np.isnan(row["atr_pct"]) else 50.0,
            "macro":     MACRO_DATES.get(d, None),
        }
    return meta


def run_orb(bars, day_meta, n_base=3, filter_fn=None, scale_fn=None, label=""):
    orb = OpeningRangeBreakoutRule(
        or_end_time="10:04", min_or_bars=7, min_range_atr=0.3,
        entry_cutoff_time="12:00", atr_period=ATR_PERIOD, long_only=True,
    )
    agg      = SignalAggregator(primary_rule=orb, filter_rules=[], confirmation_rules=[], min_confirmations=0)
    atr_s    = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars = agg.required_bars()

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
            daily_loss = 0.0; trades_today = 0
        cur_date = bdate

        meta    = day_meta.get(bdate, {})
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
                    "gap_pct": meta.get("gap_pct", 0),
                    "atr_pct": meta.get("atr_pct", 50),
                    "macro":   meta.get("macro"),
                })
                equity     += p; daily_loss += p
                peak        = max(peak, equity)
                max_dd      = min(max_dd, equity - peak)
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
                should_trade = filter_fn(meta)

            if should_trade:
                lookback = bars.iloc[max(0, i - min_bars + 1): i + 1]
                dec      = agg.evaluate(lookback)
                if dec.should_trade:
                    n  = scale_fn(meta, n_base) if scale_fn else n_base
                    ep = slip(bar["close"], 1, True)
                    sl = ep - SL_MULT * atr_now
                    pt = ep + PT_MULT * atr_now
                    pos = Pos(ep, i, sl, pt, i + TIME_STOP_BARS, n, atr_now)
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

    return {
        "n": len(trades), "wr": len(wins) / max(len(trades), 1),
        "pnl": total, "sharpe": sharpe, "dd": max_dd,
        "mll": max_dd > -2000, "trades": trades, "daily_pnl": daily_pnl,
        "pf":  gp / gl if gl > 0 else float("inf"),
    }


def monthly_wr(r):
    m = defaultdict(list)
    for t in r["trades"]:
        m[str(t["date"])[:7]].append(t)
    return {
        ym: (len(ts), sum(1 for t in ts if t["pnl"] > 0) / len(ts), sum(t["pnl"] for t in ts))
        for ym, ts in m.items()
    }


def print_row(label, r, width=40):
    mll = "✅" if r["mll"] else "❌"
    print(f"  {label:<{width}} {r['n']:>4}  {r['wr']:>5.1%} ${r['pnl']:>8,.0f} "
          f"{r['sharpe']:>7.2f} ${r['dd']:>7,.0f}  {mll}")


def print_monthly(label, r):
    ms = monthly_wr(r)
    print(f"\n  [{label}]  Sharpe={r['sharpe']:.2f}  DD=${r['dd']:,.0f}  "
          f"PF={r['pf']:.2f}  MLL={'✅' if r['mll'] else '❌'}")
    print(f"  {'Month':<10} {'N':>4} {'WR':>6} {'PnL':>10} {'Cumul':>12}")
    print(f"  {'-'*44}")
    cum = 0.0
    for ym in sorted(ms):
        n, wr, pnl = ms[ym]; cum += pnl
        print(f"  {ym:<10} {n:>4}  {wr:>5.1%} ${pnl:>8,.0f}  ${cum:>10,.0f}")


if __name__ == "__main__":
    bars = pd.read_hdf(str(DATA_PATH), key="bars_5min")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")

    day_meta = build_day_meta(bars)

    # ── Show macro event analysis on baseline ───────────────────────────────
    baseline = run_orb(bars, day_meta)
    print(f"\n{'='*82}")
    print(f"  RESEARCH FILTER SWEEP — MNQ 2026 YTD  |  3c  |  LONG-only  |  PT=3.0x SL=1.5x")
    print(f"{'='*82}")

    macro_trades = [t for t in baseline["trades"] if t["macro"]]
    non_macro    = [t for t in baseline["trades"] if not t["macro"]]
    print(f"\n  === Filter 1: Economic Calendar (Andersen et al. 2002) ===")
    print(f"  Macro event days in baseline:")
    for t in macro_trades:
        print(f"    {t['date']}  [{t['macro']}]  ${t['pnl']:>+7,.0f}  [{t['reason']}]")
    if macro_trades:
        mwr = sum(1 for t in macro_trades if t["pnl"] > 0) / len(macro_trades)
        nwr = sum(1 for t in non_macro if t["pnl"] > 0) / max(len(non_macro), 1)
        print(f"  Macro days: {len(macro_trades)} trades, WR={mwr:.1%}, avg=${np.mean([t['pnl'] for t in macro_trades]):,.0f}")
        print(f"  Non-macro:  {len(non_macro)} trades, WR={nwr:.1%}, avg=${np.mean([t['pnl'] for t in non_macro]):,.0f}")

    # ── Gap analysis ────────────────────────────────────────────────────────
    print(f"\n  === Filter 2: Overnight Gap vs ORB Direction (Lou et al. 2019) ===")
    gap_pos = [t for t in baseline["trades"] if t["gap_pct"] > 0]    # gap UP + LONG ORB = retail confirm
    gap_neg = [t for t in baseline["trades"] if t["gap_pct"] <= 0]   # gap DOWN + LONG ORB = institutional
    if gap_pos:
        print(f"  Gap UP (same dir as LONG ORB — retail signal):   {len(gap_pos):>2} trades  "
              f"WR={sum(1 for t in gap_pos if t['pnl']>0)/len(gap_pos):.1%}  "
              f"avg=${np.mean([t['pnl'] for t in gap_pos]):,.0f}")
    if gap_neg:
        print(f"  Gap DOWN (opposes LONG ORB — institutional):     {len(gap_neg):>2} trades  "
              f"WR={sum(1 for t in gap_neg if t['pnl']>0)/len(gap_neg):.1%}  "
              f"avg=${np.mean([t['pnl'] for t in gap_neg]):,.0f}")

    # ── ATR percentile analysis ─────────────────────────────────────────────
    print(f"\n  === Filter 3: ATR Volatility Percentile (Baltussen et al. 2021) ===")
    for pct_thresh in [30, 40, 50]:
        hi  = [t for t in baseline["trades"] if t["atr_pct"] >= pct_thresh]
        lo  = [t for t in baseline["trades"] if t["atr_pct"] <  pct_thresh]
        hi_wr = sum(1 for t in hi if t["pnl"] > 0) / max(len(hi), 1)
        lo_wr = sum(1 for t in lo if t["pnl"] > 0) / max(len(lo), 1)
        print(f"  ATR >{pct_thresh}th pct: {len(hi):>2} trades WR={hi_wr:.1%}  avg=${np.mean([t['pnl'] for t in hi]):>7,.0f}  |  "
              f"ATR <{pct_thresh}th: {len(lo):>2} trades WR={lo_wr:.1%}  avg=${np.mean([t['pnl'] for t in lo]) if lo else 0:>7,.0f}")

    # ── Full sweep ──────────────────────────────────────────────────────────
    W = 42
    print(f"\n  {'Config':<{W}} {'N':>4} {'WR':>6} {'PnL':>10} {'Sharpe':>7} {'MaxDD':>9}  MLL?")
    print(f"  {'-'*76}")

    results = {}

    # Baseline
    r = baseline
    results["Baseline"] = r
    print_row("Baseline 3c (live)", r, W)

    # ── FILTER 1: Economic Calendar ─────────────────────────────────────────
    print(f"\n  -- Economic Calendar (Andersen et al. 2002) --")

    def f_skip_macro(m):   return m.get("macro") is None
    def f_only_macro(m):   return m.get("macro") is not None
    def s_2c_macro(m, n):  return max(1, n - 1) if m.get("macro") else n
    def s_3c_macro(m, n):  return n + 0 if m.get("macro") else max(1, n - 0)  # no change (placeholder)

    for label, filt, scale in [
        ("Skip macro days",             f_skip_macro, None),
        ("Only macro days",             f_only_macro, None),
        ("2c on macro days",            None,         s_2c_macro),
    ]:
        r = run_orb(bars, day_meta, filter_fn=filt, scale_fn=scale)
        results[label] = r
        print_row(label, r, W)

    # ── FILTER 2: Overnight Gap ─────────────────────────────────────────────
    print(f"\n  -- Overnight Gap vs ORB Direction (Lou et al. 2019) --")

    # Gap UP + LONG = retail confirm = weaker signal → skip or reduce
    # Gap DOWN + LONG = institutional = stronger signal → keep
    def f_skip_gap_up(m):           return m.get("gap_pct", 0) <= 0.001      # skip when gap up >0.1%
    def f_skip_large_gap_up(m):     return m.get("gap_pct", 0) <= 0.003      # skip when gap up >0.3%
    def f_gap_down_only(m):         return m.get("gap_pct", 0) <= 0          # only trade gap-down days
    def s_2c_gap_up(m, n):          return max(1, n-1) if m.get("gap_pct", 0) > 0.001 else n
    def s_2c_large_gap_up(m, n):    return max(1, n-1) if m.get("gap_pct", 0) > 0.003 else n

    for label, filt, scale in [
        ("Skip gap-up >0.1%",           f_skip_gap_up,       None),
        ("Skip gap-up >0.3%",           f_skip_large_gap_up, None),
        ("Gap-down only",               f_gap_down_only,     None),
        ("2c when gap-up >0.1%",        None,                s_2c_gap_up),
        ("2c when gap-up >0.3%",        None,                s_2c_large_gap_up),
    ]:
        r = run_orb(bars, day_meta, filter_fn=filt, scale_fn=scale)
        results[label] = r
        print_row(label, r, W)

    # ── FILTER 3: ATR Percentile ────────────────────────────────────────────
    print(f"\n  -- ATR Volatility Percentile (Baltussen et al. 2021) --")

    for pct in [30, 40, 50]:
        def make_filt(p): return lambda m: m.get("atr_pct", 50) >= p
        def make_scale(p): return lambda m, n: n if m.get("atr_pct", 50) >= p else max(1, n-1)

        for label, filt, scale in [
            (f"Skip ATR <{pct}th pct",      make_filt(pct),  None),
            (f"2c when ATR <{pct}th pct",   None,            make_scale(pct)),
        ]:
            r = run_orb(bars, day_meta, filter_fn=filt, scale_fn=scale)
            results[label] = r
            print_row(label, r, W)

    # ── COMBINATIONS ────────────────────────────────────────────────────────
    print(f"\n  -- Combinations --")

    combos = [
        ("Skip macro + skip gap-up >0.3%",
         lambda m: f_skip_macro(m) and f_skip_large_gap_up(m), None),
        ("Skip gap-up + ATR >40th",
         lambda m: f_skip_gap_up(m) and m.get("atr_pct", 50) >= 40, None),
        ("2c gap-up + skip ATR <40th",
         lambda m: m.get("atr_pct", 50) >= 40,
         s_2c_gap_up),
        ("Skip macro + 2c gap-up + ATR >40th",
         lambda m: f_skip_macro(m) and m.get("atr_pct", 50) >= 40,
         s_2c_gap_up),
        ("Gap-down only + ATR >40th",
         lambda m: f_gap_down_only(m) and m.get("atr_pct", 50) >= 40, None),
    ]

    for label, filt, scale in combos:
        r = run_orb(bars, day_meta, filter_fn=filt, scale_fn=scale)
        results[label] = r
        print_row(label, r, W)

    # ── TOP PERFORMERS (MLL safe, Sharpe ranked) ────────────────────────────
    print(f"\n{'='*82}")
    print(f"  TOP RESULTS (MLL-safe, sorted by Sharpe)")
    print(f"{'='*82}")

    top = sorted(
        [(k, v) for k, v in results.items() if v["mll"] and v["n"] >= 8],
        key=lambda x: x[1]["sharpe"], reverse=True
    )[:5]

    for label, r in top:
        print_monthly(label, r)
