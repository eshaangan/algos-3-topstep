"""
VEB & Noon Reversal Backtest — MNQ Extended Dataset
====================================================

Strategy 1 — VEB (Volatility Expansion Breakout):
  Academic basis: Donaldson & Kamstra (1996), confirmed in crypto & futures.
  Edge: days with initial low-volatility compression (ATR below 14-day median)
  produce larger-than-average breakouts when the OR finally expands.

  Signal:
    - today_atr14 < atr14_rolling_median(20d) * compression_thresh
    - OR breaks out → trade in breakout direction
    - PT = 4.0x ATR (larger because compression releases more energy)
    - SL = 1.5x ATR

  This is an ALTERNATIVE to the baseline ORB — same entry mechanism but
  restricted to compressed-volatility days.

Strategy 2 — Noon Reversal (Mean Reversion):
  Academic basis: Lo, Mamaysky & Wang (2000) on intraday momentum reversals.
  Edge: after a strong morning trend (large first-3-hour move), the next
  3 hours often partially revert.

  Signal (computed at 12:30 PM ET):
    - morning_move = (close_12:30 - open_9:30) / open_9:30
    - |morning_move| >= morning_thresh (e.g. 0.5%)
    - Enter in OPPOSITE direction at 12:30 PM
    - PT = 0.5 * morning_move * reversion_frac (e.g. 50% reversion)
    - SL = 0.3 * ATR
    - Hard exit at 2:00 PM

Data: mnq_5min_aug25_mar26.h5 (Aug 2025 – Mar 2026, 162 days)
"""
from __future__ import annotations
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.indicators import atr as compute_atr

DATA_PATH    = ROOT / "data" / "processed" / "mnq_5min_aug25_mar26.h5"
RESULTS_PATH = RBV1 / "diagnostics" / "veb_noon_results.json"

POINT_VALUE = 2.0
TICK_SIZE   = 0.25
COMMISSION  = 0.62
SLIPPAGE    = 1
ATR_PERIOD  = 14
N_CONTRACTS = 3
MAX_DAILY_LOSS  = -950.0
DRAWDOWN_BUF    = 1_950.0
STARTING_EQ     = 50_000.0

OR_END_HOUR, OR_END_MIN = 10, 4
MIN_OR_BARS = 7


def slip(p, d, is_entry):
    return p + SLIPPAGE * TICK_SIZE * d if is_entry else p - SLIPPAGE * TICK_SIZE * d


def calc_pnl(entry, exit_, direction, n):
    return (exit_ - entry) * direction * n * POINT_VALUE - 2 * COMMISSION * n


def run_veb(
    bars: pd.DataFrame,
    n_contracts: int = 3,
    compression_thresh: float = 0.85,  # atr_today < median * this
    pt_mult: float = 4.0,
    sl_mult: float = 1.5,
    long_only: bool = True,
) -> dict:
    """Volatility-Expansion Breakout — trade only compressed ATR days."""
    atr_5m = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)

    # Build per-day ATR at close of OR (≈ 10:04)
    day_atrs = {}
    for date, grp in bars.groupby(bars.index.date):
        et = grp.index.tz_convert("US/Eastern")
        or_bars = grp[
            ((et.hour == 9) & (et.minute >= 30)) |
            ((et.hour == 10) & (et.minute <= OR_END_MIN))
        ]
        if len(or_bars) < MIN_OR_BARS:
            continue
        last_or = or_bars.index[-1]
        idx = bars.index.get_loc(last_or)
        a = float(atr_5m.iloc[idx])
        if not np.isnan(a):
            day_atrs[date] = a

    df_atr = pd.Series(day_atrs)
    atr_median = df_atr.rolling(20, min_periods=10).median()

    trades = []
    equity = STARTING_EQ; peak = STARTING_EQ; max_dd = 0.0
    daily_pnl = {}

    for date, grp in bars.groupby(bars.index.date):
        if date not in day_atrs:
            continue
        atr_today = day_atrs[date]
        med = float(atr_median.get(date, np.nan))
        if np.isnan(med):
            continue

        # Compression filter
        if atr_today >= med * compression_thresh:
            continue

        et     = grp.index.tz_convert("US/Eastern")
        or_bars = grp[
            ((et.hour == 9) & (et.minute >= 30)) |
            ((et.hour == 10) & (et.minute <= OR_END_MIN))
        ]
        if len(or_bars) < MIN_OR_BARS:
            continue

        or_high = float(or_bars["high"].max())
        or_low  = float(or_bars["low"].min())
        or_rng  = or_high - or_low
        if or_rng / atr_today < 0.2:
            continue

        post_or = grp[
            (et.hour > OR_END_HOUR) | ((et.hour == OR_END_HOUR) & (et.minute > OR_END_MIN))
        ]
        entry_bars = post_or[post_or.index.tz_convert("US/Eastern").hour < 12]

        day_loss = 0.0; pos = None; trades_today = 0

        for bar_ts in grp.index:
            bar    = grp.loc[bar_ts]
            bar_et = bar_ts.tz_convert("US/Eastern")
            is_close = bar_et.hour >= 16

            if pos is not None:
                bar_idx = list(grp.index).index(bar_ts)
                if is_close or bar_idx >= pos["ts_bar"]:
                    ep = slip(float(bar["close"]), pos["dir"], False)
                    pnl = calc_pnl(pos["entry"], ep, pos["dir"], n_contracts)
                    reason = "session_close" if is_close else "time_stop"
                    trades.append({"date": str(date), "pnl": round(pnl,2), "reason": reason})
                    day_loss += pnl; equity += pnl; peak = max(peak, equity)
                    max_dd = min(max_dd, equity - peak); pos = None
                    continue
                if pos["dir"] == 1:
                    if bar["low"] <= pos["stop"]:
                        ep = slip(pos["stop"], 1, False)
                        pnl = calc_pnl(pos["entry"], ep, 1, n_contracts)
                        trades.append({"date": str(date), "pnl": round(pnl,2), "reason": "stop_loss"})
                        day_loss += pnl; equity += pnl; peak = max(peak, equity)
                        max_dd = min(max_dd, equity - peak); pos = None; continue
                    if bar["high"] >= pos["tgt"]:
                        ep = slip(pos["tgt"], 1, False)
                        pnl = calc_pnl(pos["entry"], ep, 1, n_contracts)
                        trades.append({"date": str(date), "pnl": round(pnl,2), "reason": "profit_target"})
                        day_loss += pnl; equity += pnl; peak = max(peak, equity)
                        max_dd = min(max_dd, equity - peak); pos = None; continue
                else:
                    if bar["high"] >= pos["stop"]:
                        ep = slip(pos["stop"], -1, False)
                        pnl = calc_pnl(pos["entry"], ep, -1, n_contracts)
                        trades.append({"date": str(date), "pnl": round(pnl,2), "reason": "stop_loss"})
                        day_loss += pnl; equity += pnl; peak = max(peak, equity)
                        max_dd = min(max_dd, equity - peak); pos = None; continue
                    if bar["low"] <= pos["tgt"]:
                        ep = slip(pos["tgt"], -1, False)
                        pnl = calc_pnl(pos["entry"], ep, -1, n_contracts)
                        trades.append({"date": str(date), "pnl": round(pnl,2), "reason": "profit_target"})
                        day_loss += pnl; equity += pnl; peak = max(peak, equity)
                        max_dd = min(max_dd, equity - peak); pos = None; continue

            if pos is None and trades_today < 1 and day_loss > MAX_DAILY_LOSS and (equity-peak) > -DRAWDOWN_BUF:
                if bar_et.hour < 12 and (
                    (bar_et.hour > OR_END_HOUR) or
                    (bar_et.hour == OR_END_HOUR and bar_et.minute > OR_END_MIN)
                ):
                    h, l = float(bar["high"]), float(bar["low"])
                    if h > or_high:
                        entry = slip(or_high, 1, True)
                        bar_idx = list(grp.index).index(bar_ts)
                        pos = {"dir": 1, "entry": entry, "stop": entry - sl_mult*atr_today,
                               "tgt": entry + pt_mult*atr_today, "ts_bar": bar_idx + 24}
                        trades_today += 1
                    elif not long_only and l < or_low:
                        entry = slip(or_low, -1, True)
                        bar_idx = list(grp.index).index(bar_ts)
                        pos = {"dir": -1, "entry": entry, "stop": entry + sl_mult*atr_today,
                               "tgt": entry - pt_mult*atr_today, "ts_bar": bar_idx + 24}
                        trades_today += 1

        if pos is not None and grp.shape[0] > 0:
            last = grp.iloc[-1]
            ep = slip(float(last["close"]), pos["dir"], False)
            pnl = calc_pnl(pos["entry"], ep, pos["dir"], n_contracts)
            trades.append({"date": str(date), "pnl": round(pnl,2), "reason": "eod_close"})
            day_loss += pnl; equity += pnl; peak = max(peak, equity)
            max_dd = min(max_dd, equity - peak)

        if day_loss != 0:
            daily_pnl[str(date)] = round(day_loss, 2)

    wins  = [t for t in trades if t["pnl"] > 0]
    total = sum(t["pnl"] for t in trades)
    gp    = sum(t["pnl"] for t in wins)
    gl    = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    dp    = pd.Series(list(daily_pnl.values()))
    sharpe = dp.mean() / dp.std() * np.sqrt(252) if len(dp) > 1 and dp.std() > 0 else 0
    return {
        "n": len(trades), "wr": len(wins)/max(len(trades),1),
        "pnl": round(total,2), "sharpe": round(sharpe,3), "dd": round(max_dd,2),
        "mll": max_dd > -2000,
        "pf": round(gp/gl,3) if gl > 0 else float("inf"),
        "exit_reasons": {r: sum(1 for t in trades if t["reason"]==r)
                         for r in ["stop_loss","time_stop","profit_target","session_close","eod_close"]},
    }


def run_noon_reversal(
    bars: pd.DataFrame,
    n_contracts: int = 3,
    morning_thresh_pct: float = 0.005,  # need 0.5% morning move
    reversion_frac: float = 0.4,        # target 40% reversal
    sl_atr_mult: float = 0.8,
    exit_hour: int = 14,
) -> dict:
    """Noon Reversal — fade a strong morning trend at 12:30 PM."""
    atr_5m = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)

    trades = []
    equity = STARTING_EQ; peak = STARTING_EQ; max_dd = 0.0
    daily_pnl = {}

    for date, grp in bars.groupby(bars.index.date):
        et = grp.index.tz_convert("US/Eastern")
        open_bars = grp[(et.hour == 9) & (et.minute == 30)]
        sig_bars  = grp[(et.hour == 12) & (et.minute == 30)]
        if open_bars.empty or sig_bars.empty:
            continue

        open_930   = float(open_bars["open"].iloc[0])
        close_1230 = float(sig_bars["close"].iloc[-1])
        morning_mv = (close_1230 - open_930) / open_930

        if abs(morning_mv) < morning_thresh_pct:
            continue

        sig_idx = bars.index.get_loc(sig_bars.index[-1])
        atr_now = float(atr_5m.iloc[sig_idx])
        if np.isnan(atr_now) or atr_now <= 0:
            continue

        direction = -1 if morning_mv > 0 else 1  # fade the move
        target_reversion = abs(morning_mv) * open_930 * reversion_frac

        entry = slip(close_1230, direction, True)
        stop  = entry - direction * sl_atr_mult * atr_now
        tgt   = entry + direction * target_reversion

        after = grp[grp.index > sig_bars.index[-1]]
        exit_price = None; exit_reason = "session_close"

        for bar_ts, bar in after.iterrows():
            bar_et = bar_ts.tz_convert("US/Eastern")
            is_exit_time = (bar_et.hour >= exit_hour)
            is_close     = (bar_et.hour >= 16)

            if direction == 1:
                if bar["low"] <= stop:
                    exit_price = slip(stop, 1, False); exit_reason = "stop_loss"; break
                if bar["high"] >= tgt:
                    exit_price = slip(tgt, 1, False); exit_reason = "profit_target"; break
            else:
                if bar["high"] >= stop:
                    exit_price = slip(stop, -1, False); exit_reason = "stop_loss"; break
                if bar["low"] <= tgt:
                    exit_price = slip(tgt, -1, False); exit_reason = "profit_target"; break
            if is_exit_time or is_close:
                exit_price = slip(float(bar["close"]), direction, False)
                exit_reason = "time_exit"
                break

        if exit_price is None:
            last = grp.iloc[-1]
            exit_price = slip(float(last["close"]), direction, False)

        pnl = calc_pnl(entry, exit_price, direction, n_contracts)
        trades.append({"date": str(date), "dir": direction,
                       "morning_mv_pct": round(morning_mv*100, 3),
                       "pnl": round(pnl, 2), "reason": exit_reason})
        daily_pnl[str(date)] = round(pnl, 2)
        equity += pnl; peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    wins  = [t for t in trades if t["pnl"] > 0]
    total = sum(t["pnl"] for t in trades)
    gp    = sum(t["pnl"] for t in wins)
    gl    = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
    dp    = pd.Series(list(daily_pnl.values()))
    sharpe = dp.mean() / dp.std() * np.sqrt(252) if len(dp) > 1 and dp.std() > 0 else 0
    return {
        "n": len(trades), "wr": len(wins)/max(len(trades),1),
        "pnl": round(total,2), "sharpe": round(sharpe,3), "dd": round(max_dd,2),
        "mll": max_dd > -2000,
        "pf": round(gp/gl,3) if gl > 0 else float("inf"),
        "exit_reasons": {r: sum(1 for t in trades if t["reason"]==r)
                         for r in ["stop_loss","time_exit","profit_target","session_close"]},
    }


if __name__ == "__main__":
    print(f"Loading {DATA_PATH} ...")
    bars = pd.read_hdf(str(DATA_PATH), key="bars_5min")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    else:
        bars.index = bars.index.tz_convert("US/Eastern")
    bars = bars[
        ((bars.index.hour == 9) & (bars.index.minute >= 30)) |
        ((bars.index.hour > 9) & (bars.index.hour < 16))
    ].copy()
    print(f"Bars: {len(bars):,} | Days: {len(set(bars.index.date))}")

    W = 50
    print(f"\n{'='*80}")
    print("STRATEGY 1 — VEB (Volatility Expansion Breakout)")
    print(f"\n  {'Config':<{W}} {'N':>4} {'WR':>6} {'PnL':>9} {'Sharpe':>7} {'MaxDD':>9}")
    print(f"  {'-'*80}")
    veb_results = {}
    for comp in [0.70, 0.80, 0.85, 0.90]:
        for pt in [3.0, 4.0, 5.0]:
            r = run_veb(bars, N_CONTRACTS, comp, pt, 1.5, True)
            label = f"VEB comp<{comp:.0%} PT={pt}x LONG"
            veb_results[label] = r
            mll = "OK" if r["mll"] else "XX"
            print(f"  {label:<{W}} {r['n']:>4}  {r['wr']:>5.1%}  ${r['pnl']:>8,.0f}  {r['sharpe']:>6.2f}  ${r['dd']:>7,.0f}  {mll}")

    print(f"\n{'='*80}")
    print("STRATEGY 2 — Noon Reversal")
    print(f"\n  {'Config':<{W}} {'N':>4} {'WR':>6} {'PnL':>9} {'Sharpe':>7} {'MaxDD':>9}")
    print(f"  {'-'*80}")
    noon_results = {}
    for thresh in [0.003, 0.005, 0.007]:
        for rev_frac in [0.3, 0.4, 0.5]:
            for exit_h in [13, 14, 15]:
                r = run_noon_reversal(bars, N_CONTRACTS, thresh, rev_frac, 0.8, exit_h)
                label = f"Noon mv>{thresh*100:.1f}% rev={rev_frac:.0%} exit={exit_h}h"
                noon_results[label] = r
                if r["n"] >= 5:
                    mll = "OK" if r["mll"] else "XX"
                    print(f"  {label:<{W}} {r['n']:>4}  {r['wr']:>5.1%}  ${r['pnl']:>8,.0f}  {r['sharpe']:>6.2f}  ${r['dd']:>7,.0f}  {mll}")

    all_results = {"veb": veb_results, "noon_reversal": noon_results}
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved → {RESULTS_PATH}")
