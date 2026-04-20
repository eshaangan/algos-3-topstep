"""VWAP Sentiment Filter Sweep — All Permutations.

Tests every combination of:
  - Signal direction: LONG-only | SHORT-only | Both (VWAP-gated)
  - Intraday VWAP  : price vs session VWAP at signal time
  - Prev-day VWAP  : did yesterday close above its VWAP?
  - Opening gap    : today open vs yesterday close

2026 YTD data, 3 contracts, 1 trade/day, or_end=10:04, PT=3.0x, SL=1.5x
"""
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product as iproduct

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

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
bars = pd.read_hdf(str(ROOT / "data/processed/mnq_2026ytd_5min.h5"), key="bars_5min")
if bars.index.tz is None:
    bars.index = bars.index.tz_localize("US/Eastern")

# ---------------------------------------------------------------------------
# Pre-compute daily sentiment signals
# ---------------------------------------------------------------------------

def compute_daily_signals(bars: pd.DataFrame) -> pd.DataFrame:
    """For each date, compute:
      - intraday_vwap[t]: cumulative VWAP from 9:30 AM up to bar t
      - prev_vwap_bullish[date]: True if previous day closed above its VWAP
      - gap_up[date]: True if today's 9:30 open > yesterday's last close
    """
    dates = sorted(set(bars.index.date))

    # Intraday VWAP — rolling cumulative per session
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    cum_tp_vol = (typical * bars["volume"]).groupby(bars.index.date).cumsum()
    cum_vol    = bars["volume"].groupby(bars.index.date).cumsum()
    vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)

    # Previous day VWAP close — last VWAP value of each session
    prev_vwap_bullish = {}
    session_vwap_close = {}
    for d in dates:
        day_mask = bars.index.date == d
        day_bars = bars[day_mask]
        day_vwap = vwap_series[day_mask]
        if len(day_bars) == 0:
            continue
        last_close = float(day_bars["close"].iloc[-1])
        last_vwap  = float(day_vwap.iloc[-1])
        session_vwap_close[d] = (last_close > last_vwap)  # True = closed above VWAP

    for i, d in enumerate(dates):
        if i == 0:
            prev_vwap_bullish[d] = None  # no prior day
        else:
            prev_vwap_bullish[d] = session_vwap_close.get(dates[i - 1])

    # Opening gap — today's first bar open vs yesterday's last bar close
    gap_up = {}
    for i, d in enumerate(dates):
        if i == 0:
            gap_up[d] = None
            continue
        prev_day_mask = bars.index.date == dates[i - 1]
        cur_day_mask  = bars.index.date == d
        prev_bars = bars[prev_day_mask]
        cur_bars  = bars[cur_day_mask]
        if len(prev_bars) == 0 or len(cur_bars) == 0:
            gap_up[d] = None
            continue
        prev_close = float(prev_bars["close"].iloc[-1])
        today_open = float(cur_bars["open"].iloc[0])
        gap_up[d] = today_open > prev_close

    return vwap_series, prev_vwap_bullish, gap_up


vwap_series, prev_vwap_bullish, gap_up = compute_daily_signals(bars)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OR_END, MIN_OR, PT, SL = "10:04", 7, 3.0, 1.5
POINT, TICK, TICK_VAL, COMM, SLIP = 2.0, 0.25, 0.50, 0.62, 1
N = 3


@dataclass
class Pos:
    direction: int
    entry: float
    sl: float
    pt: float
    tbar: int


def slip(p, direction, is_entry):
    s = SLIP * TICK
    return p + s * direction if is_entry else p - s * direction


def calc_pnl(en, ex, direction):
    return (ex - en) * direction * N * POINT - 2 * COMM * N


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------
def run(use_intraday_vwap: bool, use_prev_vwap: bool, use_gap: bool,
        allow_long: bool, allow_short: bool,
        require_agreement: int = 1) -> dict:
    """
    require_agreement: how many of the enabled sentiment signals must agree
    """
    if not allow_long and not allow_short:
        return None

    orb = OpeningRangeBreakoutRule(
        or_end_time=OR_END, min_or_bars=MIN_OR, min_range_atr=0.3,
        entry_cutoff_time="12:00", atr_period=14, long_only=False,
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
    trades = []
    eq_vals, eq_times = [50000.0], [bars.index[0]]
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
                exited, ep, reason = True, slip(c, pos.direction, False), "session_close"
            elif i >= pos.tbar:
                exited, ep, reason = True, slip(c, pos.direction, False), "time_stop"
            elif pos.direction == 1:
                if l <= pos.sl:
                    exited, ep, reason = True, slip(pos.sl, 1, False), "stop_loss"
                elif h >= pos.pt:
                    exited, ep, reason = True, slip(pos.pt, 1, False), "profit_target"
            else:
                if h >= pos.sl:
                    exited, ep, reason = True, slip(pos.sl, -1, False), "stop_loss"
                elif l <= pos.pt:
                    exited, ep, reason = True, slip(pos.pt, -1, False), "profit_target"
            if exited:
                p = calc_pnl(pos.entry, ep, pos.direction)
                tr = TradeRecord(entry_bar=i, exit_bar=i, direction=pos.direction,
                                 entry_price=pos.entry, exit_price=ep, pnl=p, exit_reason=reason)
                trades.append(tr)
                rm.record_trade(tr)
                equity += p
                eq_vals.append(equity)
                eq_times.append(bt_et)
                pos = None

        if pos is None and not sc and trades_today < 1:
            ok, _ = rm.can_trade()
            if ok:
                lb = bars.iloc[max(0, i - mn + 1):i + 1]
                dec = agg.evaluate(lb)
                if dec.should_trade:
                    direction = dec.direction
                    ca = atr_s.iloc[i]
                    if np.isnan(ca) or ca <= 0:
                        continue

                    # --- Sentiment scoring ---
                    cur_vwap = vwap_series.iloc[i] if i < len(vwap_series) else np.nan
                    bullish_scores, bearish_scores = 0, 0
                    n_signals = 0

                    if use_intraday_vwap and not np.isnan(cur_vwap):
                        n_signals += 1
                        if bar["close"] > cur_vwap:
                            bullish_scores += 1
                        else:
                            bearish_scores += 1

                    if use_prev_vwap:
                        pv = prev_vwap_bullish.get(bdate)
                        if pv is not None:
                            n_signals += 1
                            if pv:
                                bullish_scores += 1
                            else:
                                bearish_scores += 1

                    if use_gap:
                        gu = gap_up.get(bdate)
                        if gu is not None:
                            n_signals += 1
                            if gu:
                                bullish_scores += 1
                            else:
                                bearish_scores += 1

                    # Determine allowed direction from sentiment
                    if n_signals == 0:
                        # No filters active — use direction from signal
                        sentiment_long_ok = allow_long
                        sentiment_short_ok = allow_short
                    else:
                        # Need require_agreement signals pointing same way
                        sentiment_long_ok  = allow_long  and (bullish_scores >= require_agreement)
                        sentiment_short_ok = allow_short and (bearish_scores >= require_agreement)

                    # Gate the ORB signal by sentiment + allowed direction
                    if direction == 1 and not sentiment_long_ok:
                        continue
                    if direction == -1 and not sentiment_short_ok:
                        continue
                    # If signal direction doesn't match sentiment, try flipping
                    # (only when both sides are allowed)
                    if direction == 1 and not sentiment_long_ok and sentiment_short_ok:
                        direction = -1
                    if direction == -1 and not sentiment_short_ok and sentiment_long_ok:
                        direction = 1

                    ep = slip(bar["close"], direction, True)
                    sl_p = ep - direction * SL * ca
                    pt_p = ep + direction * PT * ca
                    pos = Pos(direction, ep, sl_p, pt_p, i + 24)
                    trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = rm.daily_pnl

    if not trades:
        return {"trades": 0, "wr": 0, "pnl": 0, "sharpe": 0, "dd": 0,
                "long_wr": 0, "short_wr": 0, "long_n": 0, "short_n": 0,
                "monthly": {}}

    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    longs  = [t for t in trades if t.direction == 1]
    shorts = [t for t in trades if t.direction == -1]
    total  = sum(t.pnl for t in trades)
    gp     = sum(t.pnl for t in wins)
    gl     = abs(sum(t.pnl for t in losses))

    eq  = pd.Series(eq_vals, index=eq_times)
    max_dd = (eq - eq.cummax()).min()
    daily  = pd.Series(daily_pnl)
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
        "wr": len(wins) / len(trades),
        "pnl": total,
        "pf": gp / gl if gl > 0 else 0,
        "sharpe": sharpe,
        "dd": max_dd,
        "long_n": len(longs),
        "short_n": len(shorts),
        "long_wr": sum(1 for t in longs if t.pnl > 0) / len(longs) if longs else 0,
        "short_wr": sum(1 for t in shorts if t.pnl > 0) / len(shorts) if shorts else 0,
        "monthly": dict(monthly),
    }


# ---------------------------------------------------------------------------
# Define all permutations
# ---------------------------------------------------------------------------
CONFIGS = []

# 1. Baselines
CONFIGS.append(("Baseline LONG-only",          False, False, False, True,  False, 1))
CONFIGS.append(("Baseline BOTH (no filter)",    False, False, False, True,  True,  1))

# 2. Intraday VWAP only
CONFIGS.append(("VWAP → LONG only",             True,  False, False, True,  False, 1))
CONFIGS.append(("VWAP → SHORT only",            True,  False, False, False, True,  1))
CONFIGS.append(("VWAP → L+S gated",             True,  False, False, True,  True,  1))

# 3. Prev-day VWAP only
CONFIGS.append(("PrevVWAP → LONG only",         False, True,  False, True,  False, 1))
CONFIGS.append(("PrevVWAP → SHORT only",        False, True,  False, False, True,  1))
CONFIGS.append(("PrevVWAP → L+S gated",         False, True,  False, True,  True,  1))

# 4. Gap only
CONFIGS.append(("Gap → LONG only",              False, False, True,  True,  False, 1))
CONFIGS.append(("Gap → SHORT only",             False, False, True,  False, True,  1))
CONFIGS.append(("Gap → L+S gated",              False, False, True,  True,  True,  1))

# 5. VWAP + PrevVWAP (need 1 of 2)
CONFIGS.append(("VWAP+Prev 1of2 → L+S",        True,  True,  False, True,  True,  1))
# 6. VWAP + PrevVWAP (need 2 of 2)
CONFIGS.append(("VWAP+Prev 2of2 → L+S",        True,  True,  False, True,  True,  2))

# 7. VWAP + Gap (need 1 of 2)
CONFIGS.append(("VWAP+Gap 1of2 → L+S",         True,  False, True,  True,  True,  1))
# 8. VWAP + Gap (need 2 of 2)
CONFIGS.append(("VWAP+Gap 2of2 → L+S",         True,  False, True,  True,  True,  2))

# 9. PrevVWAP + Gap
CONFIGS.append(("Prev+Gap 1of2 → L+S",         False, True,  True,  True,  True,  1))
CONFIGS.append(("Prev+Gap 2of2 → L+S",         False, True,  True,  True,  True,  2))

# 10. All three signals
CONFIGS.append(("All3 1of3 → L+S",             True,  True,  True,  True,  True,  1))
CONFIGS.append(("All3 2of3 → L+S",             True,  True,  True,  True,  True,  2))
CONFIGS.append(("All3 3of3 → L+S",             True,  True,  True,  True,  True,  3))

# 11. VWAP-gated LONG + unconditional LONG
CONFIGS.append(("VWAP+Prev 2of2 L-only",       True,  True,  False, True,  False, 2))
CONFIGS.append(("All3 2of3 L-only",            True,  True,  True,  True,  False, 2))


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
print("=" * 90)
print("  VWAP SENTIMENT SWEEP  |  2026 YTD  |  3c  |  1 trade/day  |  PT=3.0x SL=1.5x")
print("=" * 90)
print(f"  {'Config':<28} {'N':>4} {'WR':>6} {'PnL':>9} {'Sharpe':>7} {'MaxDD':>9}"
      f"  {'L_n':>4} {'L_wr':>6}  {'S_n':>4} {'S_wr':>6}  MLL?")
print("  " + "-" * 87)

results = []
for name, iv, pv, gp_flag, al, as_, req in CONFIGS:
    r = run(iv, pv, gp_flag, al, as_, req)
    if r is None:
        continue
    ok = abs(r["dd"]) < 2000
    flag = "✅" if ok else "❌"
    print(f"  {name:<28} {r['trades']:>4} {r['wr']:>6.1%} {r['pnl']:>+9,.0f}"
          f" {r['sharpe']:>7.2f} {r['dd']:>9,.0f}"
          f"  {r['long_n']:>4} {r['long_wr']:>6.1%}"
          f"  {r['short_n']:>4} {r['short_wr']:>6.1%}  {flag}")
    results.append((name, r, ok))

# ---------------------------------------------------------------------------
# Top 5 by Sharpe that pass MLL
# ---------------------------------------------------------------------------
passing = [(n, r) for n, r, ok in results if ok and r["trades"] >= 10]
passing.sort(key=lambda x: x[1]["sharpe"], reverse=True)

print()
print("  TOP 5 (Sharpe, MLL-safe, ≥10 trades):")
print()
for name, r in passing[:5]:
    print(f"  [{name}]")
    print(f"  {'Month':<10} {'N':>4} {'WR':>6} {'PnL':>10} {'Cumul':>12}")
    print("  " + "-" * 45)
    cum = 0.0
    for m, s in sorted(r["monthly"].items()):
        wr = s["w"] / s["n"] if s["n"] else 0
        cum += s["pnl"]
        print(f"  {m:<10} {s['n']:>4} {wr:>6.1%} {s['pnl']:>+10,.0f} {cum:>+12,.0f}")
    print(f"  Sharpe={r['sharpe']:.2f}  DD=${r['dd']:,.0f}  LONG {r['long_n']}t/{r['long_wr']:.0%}  SHORT {r['short_n']}t/{r['short_wr']:.0%}")
    print()
