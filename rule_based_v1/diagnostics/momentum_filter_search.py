"""Multi-day momentum filter search for $15k/5c/$2k-DD goal.

Baseline: no-filter LO ORB = 49 trades, 65.3% WR, $18,084, -$3,624 DD (5 contracts).
The max DD comes from 5 loss clusters:
  Feb 13 (-$1,305 SL), Feb 20 (-$1,327 SL)
  Mar 19 (-$905 SL), Mar 23 (-$977 SL), Mar 24 (-$1,020 SL)
All 5 are SL exits during sustained multi-day downtrend periods.

This script tests: require N-day return > threshold before entering LONG ORB signals.

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/momentum_filter_search.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

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

import logging
logging.basicConfig(level=logging.WARNING)

CACHE_PATH = ROOT / "data" / "processed" / "mnq_vc_backtest_5min.parquet"

OR_END_TIME      = "10:04"
MIN_OR_BARS      = 7
MIN_RANGE_ATR    = 0.3
ENTRY_CUTOFF     = "12:00"
ATR_PERIOD       = 14
TIME_STOP_BARS   = 24
POINT_VALUE      = 2.0
TICK_SIZE        = 0.25
COMMISSION       = 0.62
SLIPPAGE_TICKS   = 1

N = 5                    # 5 contracts
MAX_DAILY_LOSS   = -950.0
DRAWDOWN_BUFFER  = 1_950.0
COOLDOWN_BARS    = 3

TARGET_PNL = 15_000.0
TARGET_DD  = -2_000.0

BAD_TRADE_DATES = {"2026-02-13", "2026-02-20", "2026-03-19", "2026-03-23", "2026-03-24"}


def build_daily_closes(bars: pd.DataFrame) -> pd.Series:
    dates = sorted(set(bars.index.date))
    closes = {}
    for d in dates:
        day = bars[bars.index.date == d]
        if len(day) >= 5:
            closes[d] = float(day["close"].iloc[-1])
    return pd.Series(closes)


def build_mom_meta(bars: pd.DataFrame, closes: pd.Series, n_days: int) -> dict[object, float | None]:
    """For each trading day, compute the N-day return entering that day (using prior close prices)."""
    dates = sorted(set(bars.index.date))
    meta = {}
    for i, d in enumerate(dates):
        close_dates = closes.index.tolist()
        d_pos = close_dates.index(d) if d in close_dates else -1
        if d_pos < n_days:
            meta[d] = None
        else:
            cur = closes.iloc[d_pos - 1]        # prior day's close
            prev = closes.iloc[d_pos - 1 - n_days]  # N days before that
            meta[d] = (cur / prev - 1) if prev > 0 else None
    return meta


@dataclass
class Position:
    direction: int
    entry_price: float
    entry_bar_idx: int
    stop_loss: float
    profit_target: float
    time_stop_bar: int
    atr_at_entry: float


def _slip(price, direction, is_entry):
    s = SLIPPAGE_TICKS * TICK_SIZE
    return price + s * direction if is_entry else price - s * direction


def _pnl(entry, exit_, direction):
    return (exit_ - entry) * direction * N * POINT_VALUE - 2 * COMMISSION * N


def _check_exit(pos, bar, idx, sess_close, pt_mult, sl_mult):
    h, l, c = bar["high"], bar["low"], bar["close"]
    if sess_close or idx >= pos.time_stop_bar:
        return True, _slip(c, pos.direction, False), "time_stop"
    if pos.direction == 1:
        if l <= pos.stop_loss:
            return True, _slip(pos.stop_loss, 1, False), "stop_loss"
        if h >= pos.profit_target:
            return True, _slip(pos.profit_target, 1, False), "profit_target"
    else:
        if h >= pos.stop_loss:
            return True, _slip(pos.stop_loss, -1, False), "stop_loss"
        if l <= pos.profit_target:
            return True, _slip(pos.profit_target, -1, False), "profit_target"
    return False, 0.0, ""


def run_orb(
    bars: pd.DataFrame,
    mom_meta: dict,
    mom_threshold: float,
    pt_mult: float = 3.0,
    sl_mult: float = 1.5,
    max_trades_day: int = 1,
    track_trades: bool = False,
) -> dict:
    """LO ORB + N-day momentum filter. Only enter if N-day return > mom_threshold."""
    orb = OpeningRangeBreakoutRule(
        or_end_time=OR_END_TIME, min_or_bars=MIN_OR_BARS,
        min_range_atr=MIN_RANGE_ATR, entry_cutoff_time=ENTRY_CUTOFF,
        atr_period=ATR_PERIOD, long_only=True,
    )
    agg = SignalAggregator(primary_rule=orb, filter_rules=[], confirmation_rules=[], min_confirmations=0)
    rm = RiskManager(
        contracts=N, point_value=POINT_VALUE, tick_size=TICK_SIZE,
        tick_value=POINT_VALUE / 4, max_daily_loss=MAX_DAILY_LOSS,
        per_trade_max_loss=abs(MAX_DAILY_LOSS) * 1.5,
        max_consecutive_losses=10, cooldown_bars=COOLDOWN_BARS,
        drawdown_buffer=DRAWDOWN_BUFFER,
    )
    rm.reset_all(50_000.0)

    atr_s = atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    min_bars = agg.required_bars()

    pos = None
    trades: list = []
    equity = 50_000.0
    eq_vals = [equity]
    daily_pnl: dict = {}
    cur_date = None
    trades_today = 0

    for i in range(min_bars, len(bars)):
        bar = bars.iloc[i]
        bt_et = bars.index[i]
        if bt_et.tzinfo:
            bt_et = bt_et.tz_convert("US/Eastern")
        bdate = bt_et.date()

        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = rm.daily_pnl
            rm.reset_daily()
            trades_today = 0
        cur_date = bdate
        rm.tick_bar()

        is_last = (i + 1 >= len(bars)) or (
            bars.index[i+1].tz_convert("US/Eastern").date() != bdate
        )
        sess_close = is_last or (bt_et.hour == 15 and bt_et.minute >= 55)

        if pos is not None:
            exited, exit_p, reason = _check_exit(pos, bar, i, sess_close, pt_mult, sl_mult)
            if exited:
                p = _pnl(pos.entry_price, exit_p, pos.direction)
                t = {"pnl": p, "reason": reason, "dir": pos.direction,
                     "date": str(bdate), "entry": pos.entry_price, "exit": exit_p}
                trades.append(t)
                rm.record_trade(TradeRecord(0, 0, pos.direction, pos.entry_price, exit_p, p, reason))
                equity += p
                eq_vals.append(equity)
                pos = None

        if pos is None and not sess_close and trades_today < max_trades_day:
            can, _ = rm.can_trade()
            if not can:
                continue

            # Momentum gate
            mom = mom_meta.get(bdate)
            if mom is None:
                continue  # no history → skip (conservative)
            if mom <= mom_threshold:
                continue  # momentum not positive enough

            lookback = bars.iloc[max(0, i - min_bars + 1): i + 1]
            dec = agg.evaluate(lookback)
            if not dec.should_trade:
                continue

            cur_atr = float(atr_s.iloc[i])
            if np.isnan(cur_atr) or cur_atr <= 0:
                continue

            ep = _slip(bar["close"], dec.direction, True)
            sl = ep - sl_mult * cur_atr * dec.direction
            pt = ep + pt_mult * cur_atr * dec.direction
            pos = Position(
                direction=dec.direction, entry_price=ep, entry_bar_idx=i,
                stop_loss=sl, profit_target=pt,
                time_stop_bar=i + TIME_STOP_BARS, atr_at_entry=cur_atr,
            )
            trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = rm.daily_pnl

    if not trades:
        return {"n_trades": 0, "total_pnl": 0.0, "win_rate": 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0, "trades": []}

    wins = [t for t in trades if t["pnl"] > 0]
    total = sum(t["pnl"] for t in trades)
    eq = pd.Series(eq_vals)
    max_dd = float((eq - eq.cummax()).min())
    daily = pd.Series({k: v for k, v in daily_pnl.items() if v != 0})
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
              if len(daily) > 1 and daily.std() > 0 else 0.0)
    return {
        "n_trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3),
        "total_pnl": round(total, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "trades": trades if track_trades else [],
    }


def main():
    bars = pd.read_parquet(CACHE_PATH)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    else:
        bars.index = bars.index.tz_convert("US/Eastern")

    bars_2026 = bars[bars.index >= pd.Timestamp("2026-01-01", tz="US/Eastern")]
    print(f"2026 YTD: {len(bars_2026)} bars, {bars_2026.index[0].date()} → {bars_2026.index[-1].date()}\n")

    closes = build_daily_closes(bars_2026)

    # ── First: show momentum values on the 5 bad trade dates ─────────────────
    print("=== Momentum on bad trade dates ===")
    print(f"  {'Date':<12}  {'3d mom':>8}  {'5d mom':>8}  {'10d mom':>9}  {'15d mom':>9}")
    for n_days in [3, 5, 10, 15]:
        pass  # computed below

    bad_date_moms = {}
    for n_days in [3, 5, 10, 15]:
        meta = build_mom_meta(bars_2026, closes, n_days)
        for d in sorted(closes.index):
            ds = str(d)
            if ds in BAD_TRADE_DATES:
                bad_date_moms.setdefault(ds, {})[n_days] = meta.get(d)

    print(f"  {'Date':<12}  {'3d':>7}  {'5d':>7}  {'10d':>8}  {'15d':>8}")
    for ds in sorted(BAD_TRADE_DATES):
        m = bad_date_moms.get(ds, {})
        def fmt(v): return f"{v*100:+.2f}%" if v is not None else "  None"
        print(f"  {ds:<12}  {fmt(m.get(3)):>8}  {fmt(m.get(5)):>8}  {fmt(m.get(10)):>9}  {fmt(m.get(15)):>9}")

    # ── Grid search over lookback and threshold ───────────────────────────────
    print("\n=== Grid search: LO ORB + N-day momentum filter ===")
    print("  (Baseline no-filter LO ORB: 49 trades, 65.3% WR, $18,084, -$3,624 DD)\n")
    print(f"  {'N-days':>6}  {'Threshold':>10}  {'PT':>4}  {'SL':>4}  {'MT':>3}  |  {'Trades':>7}  {'WR':>6}  {'PnL':>10}  {'DD':>9}  {'Sharpe':>7}")
    print(f"  {'-'*85}")

    configs = list(product(
        [3, 5, 10],          # lookback days
        [-0.005, 0.0, 0.005, 0.01],  # momentum threshold
        [3.0],               # pt_mult (fixed at best)
        [1.5],               # sl_mult (fixed at best)
        [1, 2],              # max_trades_per_day
    ))

    hits = []
    for n_days, threshold, pt, sl, mt in configs:
        meta = build_mom_meta(bars_2026, closes, n_days)
        r = run_orb(bars_2026, meta, threshold, pt_mult=pt, sl_mult=sl, max_trades_day=mt)
        goal = (r["total_pnl"] >= TARGET_PNL and r["max_drawdown"] >= TARGET_DD)
        flag = " ✓GOAL" if goal else ""
        if goal:
            hits.append({**r, "n_days": n_days, "threshold": threshold, "pt": pt, "sl": sl, "mt": mt})
        print(f"  {n_days:>6}  {threshold:>+10.3f}  {pt:>4.1f}  {sl:>4.1f}  {mt:>3}  |  "
              f"{r['n_trades']:>7}  {r['win_rate']:>6.1%}  ${r['total_pnl']:>9,.0f}  ${r['max_drawdown']:>8,.0f}  {r['sharpe']:>7.2f}{flag}")

    print(f"\n  → {len(hits)} config(s) hit the goal!")

    # ── Extended PT/SL sweep for promising momentum params ────────────────────
    print("\n\n=== Extended PT/SL sweep on best momentum lookbacks ===")
    ext_configs = list(product(
        [3, 5, 10],
        [-0.005, 0.0, 0.005],
        [2.5, 3.0, 4.0],    # pt_mult
        [1.0, 1.5, 2.0],    # sl_mult
        [1, 2],
    ))
    ext_hits = []
    for n_days, threshold, pt, sl, mt in ext_configs:
        meta = build_mom_meta(bars_2026, closes, n_days)
        r = run_orb(bars_2026, meta, threshold, pt_mult=pt, sl_mult=sl, max_trades_day=mt)
        goal = (r["total_pnl"] >= TARGET_PNL and r["max_drawdown"] >= TARGET_DD)
        if goal:
            ext_hits.append({**r, "n_days": n_days, "threshold": threshold, "pt": pt, "sl": sl, "mt": mt})

    if ext_hits:
        print(f"\n  {len(ext_hits)} goal-hitting configs found!")
        print(f"  {'N':>4}  {'Thresh':>8}  {'PT':>4}  {'SL':>4}  {'MT':>3}  |  {'Trades':>7}  {'WR':>6}  {'PnL':>10}  {'DD':>9}  {'Sharpe':>7}")
        print(f"  {'-'*80}")
        for r in sorted(ext_hits, key=lambda x: -x["sharpe"])[:20]:
            print(f"  {r['n_days']:>4}  {r['threshold']:>+8.3f}  {r['pt']:>4.1f}  {r['sl']:>4.1f}  {r['mt']:>3}  |  "
                  f"{r['n_trades']:>7}  {r['win_rate']:>6.1%}  ${r['total_pnl']:>9,.0f}  ${r['max_drawdown']:>8,.0f}  {r['sharpe']:>7.2f}")
    else:
        print("  No goal hits in extended sweep. Showing top 10 by score:")
        top_ext = sorted(
            [{"n_days": n_days, "threshold": th, "pt": pt, "sl": sl, "mt": mt,
              **run_orb(bars_2026, build_mom_meta(bars_2026, closes, n_days), th, pt, sl, mt)}
             for n_days, th, pt, sl, mt in list(product([5], [0.0, 0.005], [3.0, 4.0], [1.5], [1]))],
            key=lambda x: x["total_pnl"] - max(0, -x["max_drawdown"] - 2000) * 5,
            reverse=True
        )[:10]
        print(f"  {'N':>4}  {'Thresh':>8}  {'PT':>4}  {'SL':>4}  {'MT':>3}  |  {'Trades':>7}  {'WR':>6}  {'PnL':>10}  {'DD':>9}  {'Sharpe':>7}")
        print(f"  {'-'*80}")
        for r in top_ext:
            print(f"  {r['n_days']:>4}  {r['threshold']:>+8.3f}  {r['pt']:>4.1f}  {r['sl']:>4.1f}  {r['mt']:>3}  |  "
                  f"{r['n_trades']:>7}  {r['win_rate']:>6.1%}  ${r['total_pnl']:>9,.0f}  ${r['max_drawdown']:>8,.0f}  {r['sharpe']:>7.2f}")

    # ── Trade-by-trade for best hit or closest config ─────────────────────────
    all_hits = hits + ext_hits
    if all_hits:
        best = sorted(all_hits, key=lambda x: x["sharpe"], reverse=True)[0]
        print(f"\n\n=== Trade-by-trade: best goal hit ===")
        print(f"  N={best['n_days']}d, thresh={best['threshold']:+.3f}, PT={best['pt']}, SL={best['sl']}, MT={best['mt']}")
        meta = build_mom_meta(bars_2026, closes, best["n_days"])
        detail = run_orb(bars_2026, meta, best["threshold"], pt_mult=best["pt"],
                         sl_mult=best["sl"], max_trades_day=best["mt"], track_trades=True)
        equity = 50_000.0
        peak = equity
        print(f"\n  {'#':>3}  {'Date':<12}  {'Dir':>4}  {'Entry':>8}  {'Exit':>8}  {'Reason':<14}  {'PnL':>9}  {'Equity':>10}  {'DrawDown':>9}")
        print(f"  {'-'*95}")
        for j, t in enumerate(detail["trades"]):
            equity += t["pnl"]
            peak = max(peak, equity)
            dd = equity - peak
            bad = " ← BAD" if t["date"] in BAD_TRADE_DATES else ""
            print(f"  {j+1:>3}  {t['date']:<12}  {'L' if t['dir']==1 else 'S':>4}  "
                  f"{t['entry']:>8.2f}  {t['exit']:>8.2f}  {t['reason']:<14}  "
                  f"${t['pnl']:>8,.0f}  ${equity:>9,.0f}  ${dd:>8,.0f}{bad}")
        print(f"\n  Final: {detail['n_trades']} trades, {detail['win_rate']:.1%} WR, "
              f"${detail['total_pnl']:,.0f} PnL, {detail['max_drawdown']:,.0f} max DD, "
              f"Sharpe {detail['sharpe']:.2f}")
    else:
        # Show which bad trades get filtered for different momentum settings
        print("\n\n=== Bad trade filter analysis (how many bad dates blocked per setting?) ===")
        print(f"  5d momentum on bad dates + how many get removed:\n")
        for n_days, threshold in product([3, 5, 10], [-0.005, 0.0, 0.005, 0.01]):
            meta = build_mom_meta(bars_2026, closes, n_days)
            removed = 0
            for ds in BAD_TRADE_DATES:
                # find the date object
                for d in closes.index:
                    if str(d) == ds:
                        mom = meta.get(d)
                        if mom is None or mom <= threshold:
                            removed += 1
                        break
            print(f"  N={n_days}d thresh={threshold:+.3f}: {removed}/5 bad dates removed")


if __name__ == "__main__":
    main()
