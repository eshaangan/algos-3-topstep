"""
VWAP Strategy Backtest — MNQ 2026 YTD
======================================
Tests 4 VWAP-based long-only strategies:

1. VWAP Cross       : Enter LONG first time close crosses above VWAP after 10:00 AM
2. VWAP Pullback    : After price establishes above VWAP, enter LONG on dip back to VWAP
3. First30 + VWAP   : Inspired by Gao et al. (2018) "Market Intraday Momentum" —
                      LONG only if 9:30–10:00 return > 0 AND price currently above VWAP
4. VWAP Band Break  : Enter LONG when close breaks above VWAP + 1 rolling std dev band

Academic grounding:
  - Gao, Han, Li, Zhou (2018): First-30-min return predicts direction. JFE.
  - Lou, Polk, Skouras (2019): Overnight gap tends to reverse intraday.
  - Industry: VWAP pullback is standard institutional intraday entry technique.

Risk: 3 MNQ contracts, 1 trade/day, PT=3.0x ATR, SL=1.5x ATR (same as live config)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.indicators import atr as compute_atr

DATA_PATH = ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5"

# ── Risk / execution config (mirrors live) ──────────────────────────────────
N_CONTRACTS      = 3
POINT_VALUE      = 2.0
TICK_SIZE        = 0.25
COMMISSION       = 0.62          # per side per contract
SLIPPAGE_TICKS   = 1
PT_MULT          = 3.0
SL_MULT          = 1.5
ATR_PERIOD       = 14
TIME_STOP_BARS   = 24            # 2 hours
MAX_DAILY_LOSS   = -950.0
MAX_TRADES_DAY   = 1
STARTING_EQUITY  = 50_000.0
DRAWDOWN_BUFFER  = 1_950.0       # MLL $2,000 – $50 buffer

# ── Strategy parameters ──────────────────────────────────────────────────────
ENTRY_START_HOUR   = 10          # no entries before 10:00 AM
ENTRY_CUTOFF_HOUR  = 12          # no new entries after 12:00 PM
PULLBACK_THRESH    = 0.10        # VWAP Pullback: within X*ATR of VWAP counts as "at VWAP"
BAND_STDEV_MULT    = 1.0         # VWAP Band: enter when close > VWAP + N * rolling_std


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
@dataclass
class Position:
    direction: int
    entry_price: float
    entry_bar_idx: int
    stop_loss: float
    profit_target: float
    time_stop_bar: int
    atr_at_entry: float = 0.0


def slip(price: float, direction: int, is_entry: bool) -> float:
    s = SLIPPAGE_TICKS * TICK_SIZE
    return price + s * direction if is_entry else price - s * direction


def calc_pnl(entry: float, exit_: float, direction: int) -> float:
    raw = (exit_ - entry) * direction * N_CONTRACTS * POINT_VALUE
    return raw - 2 * COMMISSION * N_CONTRACTS


def check_exit(pos: Position, bar: pd.Series, idx: int, sess_close: bool):
    h, l, c = bar["high"], bar["low"], bar["close"]
    if sess_close:
        return True, slip(c, pos.direction, False), "session_close"
    if idx >= pos.time_stop_bar:
        return True, slip(c, pos.direction, False), "time_stop"
    if pos.direction == 1:
        if l <= pos.stop_loss:
            return True, slip(pos.stop_loss, 1, False), "stop_loss"
        if h >= pos.profit_target:
            return True, slip(pos.profit_target, 1, False), "profit_target"
    return False, 0.0, ""


def compute_session_vwap(bars: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP from 09:30 each day."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    pv = typical * bars["volume"]
    vwap = pd.Series(np.nan, index=bars.index)
    for date, grp in bars.groupby(bars.index.date):
        idx = grp.index
        cum_pv  = pv.loc[idx].cumsum()
        cum_vol = bars["volume"].loc[idx].cumsum()
        vwap.loc[idx] = (cum_pv / cum_vol.replace(0, np.nan)).values
    return vwap


def compute_vwap_std(bars: pd.DataFrame, vwap: pd.Series, window: int = 20) -> pd.Series:
    """Rolling std of (close - VWAP) within the session."""
    dev = bars["close"] - vwap
    return dev.rolling(window, min_periods=5).std()


def run_strategy(bars: pd.DataFrame, strategy: str) -> dict:
    """
    Run one of 4 VWAP strategies and return result dict.
    strategy: 'cross' | 'pullback' | 'first30' | 'band'
    """
    atr_s   = compute_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD)
    vwap_s  = compute_session_vwap(bars)
    vstd_s  = compute_vwap_std(bars, vwap_s, window=20)

    pos         = None
    trades      = []
    equity      = STARTING_EQUITY
    peak_equity = STARTING_EQUITY
    max_dd      = 0.0
    cur_date    = None
    daily_pnl   = {}
    trades_today = 0
    daily_loss  = 0.0

    # Per-day state for strategies
    day_state: dict = {}

    for i in range(ATR_PERIOD + 5, len(bars)):
        bar    = bars.iloc[i]
        bt     = bars.index[i]
        bt_et  = bt.tz_convert("US/Eastern") if bt.tzinfo else bt
        bdate  = bt_et.date()
        bh, bm = bt_et.hour, bt_et.minute

        # ── Day rollover ──────────────────────────────────────────────────
        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = daily_loss
            daily_loss   = 0.0
            trades_today = 0
            day_state    = {}
        cur_date = bdate

        vwap_now = vwap_s.iloc[i]
        vstd_now = vstd_s.iloc[i]
        atr_now  = atr_s.iloc[i]
        close    = bar["close"]
        if np.isnan(vwap_now) or np.isnan(atr_now) or atr_now <= 0:
            continue

        is_last    = (i + 1 >= len(bars)) or \
                     bars.index[i + 1].tz_convert("US/Eastern").date() != bdate
        sess_close = is_last or (bh == 15 and bm >= 55)

        # ── Exit check ────────────────────────────────────────────────────
        if pos is not None:
            exited, exit_p, reason = check_exit(pos, bar, i, sess_close)
            if exited:
                pnl = calc_pnl(pos.entry_price, exit_p, pos.direction)
                trades.append({
                    "date": bdate, "direction": "LONG",
                    "entry": pos.entry_price, "exit": exit_p,
                    "pnl": pnl, "reason": reason,
                    "atr": pos.atr_at_entry,
                })
                equity     += pnl
                daily_loss += pnl
                peak_equity = max(peak_equity, equity)
                max_dd      = min(max_dd, equity - peak_equity)
                pos = None

        # ── Entry logic ───────────────────────────────────────────────────
        can_enter = (
            pos is None
            and not sess_close
            and trades_today < MAX_TRADES_DAY
            and daily_loss > MAX_DAILY_LOSS
            and (equity - peak_equity) > -DRAWDOWN_BUFFER
            and bh >= ENTRY_START_HOUR
            and not (bh >= ENTRY_CUTOFF_HOUR)
        )

        signal = False

        if can_enter:
            if strategy == "cross":
                # Enter LONG first time close crosses above VWAP after 10 AM
                prev_close = bars["close"].iloc[i - 1]
                prev_vwap  = vwap_s.iloc[i - 1]
                already_traded = day_state.get("crossed", False)
                if not already_traded and prev_close <= prev_vwap and close > vwap_now:
                    signal = True
                    day_state["crossed"] = True

            elif strategy == "pullback":
                # Stage 1: establish above VWAP (close > VWAP for 3+ consecutive bars)
                # Stage 2: dip back to within PULLBACK_THRESH * ATR of VWAP
                # Stage 3: close back above VWAP → enter
                above_count = day_state.get("above_count", 0)
                dipped      = day_state.get("dipped", False)
                established = day_state.get("established", False)

                if close > vwap_now:
                    above_count += 1
                else:
                    above_count = 0
                day_state["above_count"] = above_count

                if above_count >= 3:
                    day_state["established"] = True
                    established = True

                if established and abs(close - vwap_now) <= PULLBACK_THRESH * atr_now:
                    day_state["dipped"] = True
                    dipped = True

                if established and dipped and close > vwap_now:
                    prev_close = bars["close"].iloc[i - 1]
                    prev_vwap  = vwap_s.iloc[i - 1]
                    # Confirm bounce: prev bar was at/below VWAP, now above
                    if prev_close <= prev_vwap + PULLBACK_THRESH * atr_now:
                        signal = True
                        # Reset dip state so we don't re-enter immediately
                        day_state["dipped"] = False

            elif strategy == "first30":
                # Gao et al. (2018): first-30-min return > 0 is the signal
                # Combine with current price > VWAP for confirmation
                if "first30_return" not in day_state:
                    # Compute first-30-min return: find 09:30 and 10:00 bars
                    today_bars = bars.iloc[:i + 1]
                    today_bars = today_bars[today_bars.index.tz_convert("US/Eastern").date == bdate]
                    open_bars  = today_bars[
                        (today_bars.index.tz_convert("US/Eastern").hour == 9) &
                        (today_bars.index.tz_convert("US/Eastern").minute == 30)
                    ]
                    close_bars = today_bars[
                        (today_bars.index.tz_convert("US/Eastern").hour == 10) &
                        (today_bars.index.tz_convert("US/Eastern").minute == 0)
                    ]
                    if len(open_bars) > 0 and len(close_bars) > 0:
                        open_price  = open_bars["open"].iloc[0]
                        close_price = close_bars["close"].iloc[-1]
                        day_state["first30_return"] = (close_price - open_price) / open_price

                f30 = day_state.get("first30_return", None)
                already_traded = day_state.get("traded_today", False)
                if (f30 is not None and f30 > 0 and close > vwap_now
                        and not already_traded):
                    # Only enter once per day on first valid bar
                    signal = True
                    day_state["traded_today"] = True

            elif strategy == "band":
                # Enter LONG when close breaks above VWAP + N * rolling_std
                if np.isnan(vstd_now) or vstd_now <= 0:
                    pass
                else:
                    upper_band  = vwap_now + BAND_STDEV_MULT * vstd_now
                    prev_close  = bars["close"].iloc[i - 1]
                    prev_vwap   = vwap_s.iloc[i - 1]
                    prev_vstd   = vstd_s.iloc[i - 1]
                    prev_upper  = (prev_vwap + BAND_STDEV_MULT * prev_vstd
                                   if not np.isnan(prev_vstd) else upper_band)
                    already_traded = day_state.get("band_traded", False)
                    if not already_traded and prev_close <= prev_upper and close > upper_band:
                        signal = True
                        day_state["band_traded"] = True

        if signal:
            ep = slip(close, 1, True)
            sl = ep - SL_MULT * atr_now
            pt = ep + PT_MULT * atr_now
            pos = Position(
                direction=1, entry_price=ep, entry_bar_idx=i,
                stop_loss=sl, profit_target=pt,
                time_stop_bar=i + TIME_STOP_BARS, atr_at_entry=atr_now,
            )
            trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = daily_loss

    return {"trades": trades, "daily_pnl": daily_pnl, "max_dd": max_dd}


def summarise(name: str, result: dict) -> dict:
    trades    = result["trades"]
    daily_pnl = result["daily_pnl"]
    if not trades:
        print(f"  {name:<22} — No trades")
        return {}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    total  = sum(t["pnl"] for t in trades)
    gp     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))
    pf     = gp / gl if gl > 0 else float("inf")

    daily  = pd.Series(daily_pnl)
    daily  = daily[daily != 0]
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0

    n      = len(trades)
    wr     = len(wins) / n
    mdd    = result["max_dd"]
    mll_ok = mdd > -2000

    return {
        "name": name, "n": n, "wr": wr, "pnl": total,
        "sharpe": sharpe, "max_dd": mdd, "pf": pf, "mll": mll_ok,
        "trades": trades, "daily_pnl": daily_pnl,
    }


def print_detail(r: dict):
    if not r:
        return
    print(f"\n  [{r['name']}]")
    print(f"  {'Month':<10} {'N':>4} {'WR':>6} {'PnL':>10} {'Cumul':>12}")
    print(f"  {'-'*46}")
    trades_by_month = defaultdict(list)
    for t in r["trades"]:
        ym = str(t["date"])[:7]
        trades_by_month[ym].append(t)
    cum = 0.0
    for ym in sorted(trades_by_month):
        ts   = trades_by_month[ym]
        pnl  = sum(t["pnl"] for t in ts)
        wr   = sum(1 for t in ts if t["pnl"] > 0) / len(ts)
        cum += pnl
        print(f"  {ym:<10} {len(ts):>4}  {wr:>5.1%}  ${pnl:>8,.0f}  ${cum:>10,.0f}")

    reasons = defaultdict(int)
    for t in r["trades"]:
        reasons[t["reason"]] += 1
    print(f"  Exits: {dict(reasons)}")
    print(f"  Sharpe={r['sharpe']:.2f}  DD=${r['max_dd']:,.0f}  PF={r['pf']:.2f}  MLL={'✅' if r['mll'] else '❌'}")

    # Recent trades
    print(f"\n  {'Date':<12} {'Entry':>9} {'Exit':>9} {'PnL':>8}  Reason")
    print(f"  {'-'*52}")
    for t in r["trades"][-10:]:
        print(f"  {str(t['date']):<12} {t['entry']:>9.2f} {t['exit']:>9.2f} ${t['pnl']:>7,.0f}  {t['reason']}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bars = pd.read_hdf(str(DATA_PATH), key="bars_5min")
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")

    print(f"\n{'='*80}")
    print(f"  VWAP STRATEGY BACKTEST  |  MNQ 2026 YTD  |  3c  |  1 trade/day  |  PT={PT_MULT}x SL={SL_MULT}x")
    print(f"{'='*80}")
    print(f"  {'Strategy':<24} {'N':>4} {'WR':>6} {'PnL':>10} {'Sharpe':>7} {'MaxDD':>8}  MLL?")
    print(f"  {'-'*70}")

    results = {}
    strategy_names = {
        "cross":    "VWAP Cross",
        "pullback": "VWAP Pullback",
        "first30":  "First-30min + VWAP",
        "band":     "VWAP Band Break",
    }

    for key, label in strategy_names.items():
        res = run_strategy(bars, key)
        r   = summarise(label, res)
        results[key] = r
        if r:
            mll = "✅" if r["mll"] else "❌"
            print(f"  {r['name']:<24} {r['n']:>4}  {r['wr']:>5.1%} ${r['pnl']:>9,.0f} {r['sharpe']:>7.2f} ${r['max_dd']:>7,.0f}  {mll}")

    # Baseline for reference
    print(f"\n  {'--- Baseline (ORB LONG-only) ---':<24}")
    print(f"  {'ORB LONG-only':<24}   27  70.4%    +$6,005    4.30  $-1,516  ✅")

    print(f"\n{'='*80}")
    print(f"  DETAILED BREAKDOWN — TOP VWAP STRATEGIES")
    print(f"{'='*80}")

    # Print detail for all non-empty strategies, sorted by Sharpe
    valid = [r for r in results.values() if r]
    valid.sort(key=lambda x: x["sharpe"], reverse=True)
    for r in valid:
        print_detail(r)
