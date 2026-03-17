"""Variant sweep — LONG-only ORB, 2026 YTD.

Tests:
  Baseline : PT=3.0x, SL=1.5x, max_trades=2, daily_loss=-950
  V1       : max_trades=1
  V2       : daily_loss=-400  (tighter circuit breaker)
  V3       : PT=2.0x, SL=1.5x
  V4       : V1 + V2  (1 trade + tight daily limit)
  V5       : V2 + V3  (tight daily + lower PT)
  V6       : V1 + V3  (1 trade + lower PT)
  V7       : V1 + V2 + V3  (all three)
"""
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from engine.risk_manager import RiskManager, TradeRecord
from engine.signal_aggregator import SignalAggregator
from rules.opening_range import OpeningRangeBreakoutRule
from utils.indicators import atr

bars = pd.read_hdf(str(ROOT / "data/processed/mnq_2026ytd_5min.h5"), key="bars_5min")
if bars.index.tz is None:
    bars.index = bars.index.tz_localize("US/Eastern")

POINT, TICK, TICK_VAL, COMM, SLIP = 2.0, 0.25, 0.50, 0.62, 1
N = 2
OR_END, MIN_OR = "10:04", 7


@dataclass
class Pos:
    entry: float
    sl: float
    pt: float
    tbar: int


def slip(p, is_entry):
    s = SLIP * TICK
    return p + s if is_entry else p - s


def run(pt_mult, sl_mult, max_trades_day, daily_loss_limit):
    def calc_pnl(en, ex):
        return (ex - en) * N * POINT - 2 * COMM * N

    orb = OpeningRangeBreakoutRule(
        or_end_time=OR_END, min_or_bars=MIN_OR, min_range_atr=0.3,
        entry_cutoff_time="12:00", atr_period=14, long_only=True,
    )
    agg = SignalAggregator(primary_rule=orb, filter_rules=[], confirmation_rules=[], min_confirmations=0)
    rm = RiskManager(
        contracts=N, point_value=POINT, tick_size=TICK, tick_value=TICK_VAL,
        max_daily_loss=daily_loss_limit, per_trade_max_loss=1000.0,
        max_consecutive_losses=10, cooldown_bars=3, drawdown_buffer=1950.0,
    )
    rm.reset_all(50000.0)

    atr_s = atr(bars["high"], bars["low"], bars["close"], 14)
    mn = agg.required_bars()

    pos = None
    trades, eq_vals, eq_times = [], [50000.0], [bars.index[0]]
    equity = 50000.0
    cur_date, daily_pnl, trades_today = None, {}, 0

    for i in range(mn, len(bars)):
        bar = bars.iloc[i]
        bt_et = bars.index[i]
        bdate = bt_et.date()

        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = rm.daily_pnl
            rm.reset_daily()
            trades_today = 0
        cur_date = bdate
        rm.tick_bar()

        is_last = (i + 1 >= len(bars)) or (bars.index[i + 1].date() != bdate)
        sc = is_last or (bt_et.hour == 15 and bt_et.minute >= 55)

        if pos is not None:
            h, l, c = bar["high"], bar["low"], bar["close"]
            exited, ep, reason = False, 0.0, ""
            if sc:
                exited, ep, reason = True, slip(c, False), "session_close"
            elif i >= pos.tbar:
                exited, ep, reason = True, slip(c, False), "time_stop"
            elif l <= pos.sl:
                exited, ep, reason = True, slip(pos.sl, False), "stop_loss"
            elif h >= pos.pt:
                exited, ep, reason = True, slip(pos.pt, False), "profit_target"
            if exited:
                p = calc_pnl(pos.entry, ep)
                tr = TradeRecord(entry_bar=i, exit_bar=i, direction=1,
                                 entry_price=pos.entry, exit_price=ep, pnl=p, exit_reason=reason)
                trades.append(tr)
                rm.record_trade(tr)
                equity += p
                eq_vals.append(equity)
                eq_times.append(bt_et)
                pos = None

        if pos is None and not sc and trades_today < max_trades_day:
            ok, _ = rm.can_trade()
            if ok:
                lb = bars.iloc[max(0, i - mn + 1):i + 1]
                dec = agg.evaluate(lb)
                if dec.should_trade:
                    ca = atr_s.iloc[i]
                    if not (np.isnan(ca) or ca <= 0):
                        ep = slip(bar["close"], True)
                        pos = Pos(ep, ep - sl_mult * ca, ep + pt_mult * ca, i + 24)
                        trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = rm.daily_pnl

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    total = sum(t.pnl for t in trades)
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses))
    eq = pd.Series(eq_vals, index=eq_times)
    max_dd = (eq - eq.cummax()).min()
    daily = pd.Series(daily_pnl)
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0.0

    monthly = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0})
    for t in trades:
        m = str(bars.index[t.entry_bar].date())[:7]
        monthly[m]["n"] += 1
        monthly[m]["pnl"] += t.pnl
        if t.pnl > 0:
            monthly[m]["w"] += 1

    return {
        "trades": len(trades),
        "wr": len(wins) / len(trades) if trades else 0,
        "pnl": total,
        "pf": gp / gl if gl > 0 else 0,
        "sharpe": sharpe,
        "dd": max_dd,
        "monthly": dict(monthly),
    }


VARIANTS = [
    ("Baseline",      3.0, 1.5, 2, -950),
    ("V1 1trade/day", 3.0, 1.5, 1, -950),
    ("V2 DL=-400",    3.0, 1.5, 2, -400),
    ("V3 PT=2.0x",    2.0, 1.5, 2, -950),
    ("V4 V1+V2",      3.0, 1.5, 1, -400),
    ("V5 V2+V3",      2.0, 1.5, 2, -400),
    ("V6 V1+V3",      2.0, 1.5, 1, -950),
    ("V7 all three",  2.0, 1.5, 1, -400),
]

print("=" * 78)
print("  LONG-ONLY ORB VARIANT SWEEP  |  2026 YTD  |  2 contracts")
print("=" * 78)
print(f"  {'Variant':<18} {'Trades':>7} {'WR':>7} {'PnL':>9} {'PF':>6} {'Sharpe':>7} {'MaxDD':>10}  MLL?")
print("  " + "-" * 74)

results = {}
for name, pt, sl, mt, dl in VARIANTS:
    r = run(pt, sl, mt, dl)
    mll_ok = abs(r["dd"]) < 2000
    flag = "✅" if mll_ok else "❌"
    print(f"  {name:<18} {r['trades']:>7} {r['wr']:>7.1%} {r['pnl']:>+9,.0f} "
          f"{r['pf']:>6.2f} {r['sharpe']:>7.2f} {r['dd']:>10,.0f}  {flag}")
    results[name] = r

# Detailed monthly for top 3 by Sharpe that pass MLL
print()
passing = [(n, r) for n, r in results.items() if abs(r["dd"]) < 2000]
passing.sort(key=lambda x: x[1]["sharpe"], reverse=True)

print("  Monthly breakdown — variants that stay within $2,000 MLL:")
for name, r in passing[:4]:
    print(f"\n  [{name}]")
    print(f"  {'Month':<10} {'Trades':>7} {'WR':>7} {'PnL':>10} {'Cumul':>12}")
    print("  " + "-" * 46)
    cum = 0.0
    for m, s in sorted(r["monthly"].items()):
        wr = s["w"] / s["n"] if s["n"] else 0
        cum += s["pnl"]
        print(f"  {m:<10} {s['n']:>7} {wr:>7.1%} {s['pnl']:>+10,.0f} {cum:>+12,.0f}")
