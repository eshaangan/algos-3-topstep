"""Full LONG-only ORB backtest — 2026 YTD (Jan 2 - Mar 16), 2 contracts."""
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

OR_END, MIN_OR, PT, SL = "10:04", 7, 3.0, 1.5
POINT, TICK, TICK_VAL, COMM, SLIP = 2.0, 0.25, 0.50, 0.62, 1
N = 2


@dataclass
class Pos:
    entry: float
    sl: float
    pt: float
    tbar: int


def slip(p, is_entry):
    s = SLIP * TICK
    return p + s if is_entry else p - s


def calc_pnl(en, ex):
    return (ex - en) * N * POINT - 2 * COMM * N


orb = OpeningRangeBreakoutRule(
    or_end_time=OR_END, min_or_bars=MIN_OR, min_range_atr=0.3,
    entry_cutoff_time="12:00", atr_period=14, long_only=True,
)
agg = SignalAggregator(primary_rule=orb, filter_rules=[], confirmation_rules=[], min_confirmations=0)
rm = RiskManager(
    contracts=N, point_value=POINT, tick_size=TICK, tick_value=TICK_VAL,
    max_daily_loss=-950.0, per_trade_max_loss=1000.0,
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

    if pos is None and not sc and trades_today < 2:
        ok, _ = rm.can_trade()
        if ok:
            lb = bars.iloc[max(0, i - mn + 1):i + 1]
            dec = agg.evaluate(lb)
            if dec.should_trade:
                ca = atr_s.iloc[i]
                if not (np.isnan(ca) or ca <= 0):
                    ep = slip(bar["close"], True)
                    pos = Pos(ep, ep - SL * ca, ep + PT * ca, i + 24)
                    trades_today += 1

if cur_date and cur_date not in daily_pnl:
    daily_pnl[cur_date] = rm.daily_pnl

# ---------- Print results ----------
wins = [t for t in trades if t.pnl > 0]
losses = [t for t in trades if t.pnl < 0]
total = sum(t.pnl for t in trades)
gp = sum(t.pnl for t in wins)
gl = abs(sum(t.pnl for t in losses))
eq = pd.Series(eq_vals, index=eq_times)
max_dd = (eq - eq.cummax()).min()
daily = pd.Series(daily_pnl)
sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0.0
reasons = defaultdict(int)
for t in trades:
    reasons[t.exit_reason] += 1

print("=" * 62)
print("  LONG-ONLY ORB  |  2026 YTD  |  2c  |  Jan 2 - Mar 16")
print("=" * 62)
print(f"  Trades       : {len(trades)}  ({len(trades)/max(1,len(daily_pnl)):.2f}/day over {len(daily_pnl)} trading days)")
print(f"  Win Rate     : {len(wins)/len(trades):.1%}  ({len(wins)}W / {len(losses)}L)")
print(f"  Total PnL    : ${total:,.2f}")
print(f"  Avg Win      : ${gp/len(wins):,.2f}" if wins else "  Avg Win      : N/A")
print(f"  Avg Loss     : ${gl/len(losses):,.2f}" if losses else "  Avg Loss     : N/A")
print(f"  Profit Factor: {gp/gl:.2f}" if gl > 0 else "  Profit Factor: inf")
print(f"  Sharpe       : {sharpe:.2f}")
print(f"  Max Drawdown : ${max_dd:,.2f}")
print(f"  Exit reasons : {dict(reasons)}")

print()
print("  Monthly Breakdown:")
print(f"  {'Month':<10} {'Trades':>7} {'WR':>7} {'PnL':>10} {'Cumulative':>12}")
print("  " + "-" * 50)
monthly = defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0})
for t in trades:
    m = str(bars.index[t.entry_bar].date())[:7]
    monthly[m]["n"] += 1
    monthly[m]["pnl"] += t.pnl
    if t.pnl > 0:
        monthly[m]["w"] += 1
cum = 0.0
for m, s in sorted(monthly.items()):
    wr = s["w"] / s["n"] if s["n"] else 0
    cum += s["pnl"]
    print(f"  {m:<10} {s['n']:>7} {wr:>7.1%} {s['pnl']:>+10,.0f} {cum:>+12,.0f}")

print()
print("  Daily PnL (non-zero days only):")
print(f"  {'Date':<13} {'PnL':>8}  {'Cumul':>10}")
print("  " + "-" * 36)
cum = 0.0
for d, v in sorted(daily_pnl.items()):
    cum += v
    if v != 0:
        bar_c = "+" * min(20, int(abs(v) / 20)) if v > 0 else "-" * min(20, int(abs(v) / 20))
        print(f"  {str(d):<13} ${v:>7,.0f}  ${cum:>+9,.0f}  {bar_c}")

print()
print(f"  {'#':<4} {'Entry':>9} {'Exit':>9} {'PnL':>9}  Reason")
print("  " + "-" * 50)
for i, t in enumerate(trades, 1):
    print(f"  {i:<4} {t.entry_price:>9.2f} {t.exit_price:>9.2f} ${t.pnl:>8,.2f}  {t.exit_reason}")
