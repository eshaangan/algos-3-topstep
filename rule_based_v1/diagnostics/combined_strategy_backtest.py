"""Combined ORB + VWAP MR backtest — 2026 YTD MNQ.

Tests ORB (with 3-day momentum filter) + VWAP Mean Reversion running
simultaneously. Sweeps contract sizes to find the $6k/week threshold
and the resulting max drawdown.

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/combined_strategy_backtest.py
    python rule_based_v1/diagnostics/combined_strategy_backtest.py --contracts 10
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT  = Path(__file__).resolve().parent.parent.parent
RBV1  = ROOT / "rule_based_v1"
RESULTS_PATH = RBV1 / "diagnostics" / "combined_strategy_results.json"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
_DATA_CANDIDATES = [
    ROOT / "data" / "processed" / "mnq_2026ytd_databento_5min_rth.h5",
    ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5",
]

def load_bars() -> pd.DataFrame:
    for p in _DATA_CANDIDATES:
        if p.exists():
            df = pd.read_hdf(str(p), key="bars_5min")
            if df.index.tz is None:
                df.index = df.index.tz_localize("US/Eastern")
            else:
                df.index = df.index.tz_convert("US/Eastern")
            print(f"Loaded {len(df):,} bars from {p.name}  "
                  f"[{df.index[0].date()} → {df.index[-1].date()}]")
            return df
    raise FileNotFoundError("No 2026 YTD data found — run fetch_backtest_2026ytd.py first")

# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
POINT_VALUE    = 2.0
TICK_SIZE      = 0.25
TICK_VALUE     = 0.50
COMMISSION     = 0.62   # per side per contract
SLIPPAGE_TICKS = 1      # 1 tick each way

def slip(price: float, direction: int, is_entry: bool) -> float:
    s = SLIPPAGE_TICKS * TICK_SIZE
    return price + s * direction if is_entry else price - s * direction

def pnl(entry: float, exit_: float, direction: int, n: int) -> float:
    raw = (exit_ - entry) * direction * n * POINT_VALUE
    return raw - 2 * COMMISSION * n

# ---------------------------------------------------------------------------
# ATR helper
# ---------------------------------------------------------------------------
def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

# ---------------------------------------------------------------------------
# Daily metadata helpers
# ---------------------------------------------------------------------------
def build_vwap_series(bars: pd.DataFrame) -> pd.Series:
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    tp_vol  = typical * bars["volume"]
    dates   = bars.index.map(lambda t: t.date())
    return tp_vol.groupby(dates).cumsum() / bars["volume"].groupby(dates).cumsum().replace(0, np.nan)

def build_daily_closes(bars: pd.DataFrame) -> dict:
    """Date -> session close price."""
    out = {}
    for d, grp in bars.groupby(bars.index.map(lambda t: t.date())):
        out[d] = float(grp["close"].iloc[-1])
    return out

def mom_filter_allows(closes_deque: deque, lookback: int = 3, threshold: float = -0.01,
                      pass_on_no_history: bool = True) -> bool:
    if len(closes_deque) < lookback + 1:
        return pass_on_no_history
    cl = list(closes_deque)
    ret = (cl[-1] / cl[-(lookback + 1)]) - 1.0
    return ret > threshold

# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------
@dataclass
class Position:
    strategy: str          # "orb" | "vwap"
    direction: int         # +1 LONG | -1 SHORT
    entry_price: float
    entry_bar: int
    stop_loss: float
    profit_target: float
    time_stop_bar: int
    atr_at_entry: float

@dataclass
class TradeRecord:
    strategy: str
    direction: int
    entry_price: float
    exit_price: float
    pnl: float
    exit_reason: str

# ---------------------------------------------------------------------------
# ORB signal
# ---------------------------------------------------------------------------
def orb_signal(
    bars: pd.DataFrame,
    i: int,
    atr_s: pd.Series,
    or_end_time: str = "10:04",
    min_or_bars: int = 7,
    min_range_atr: float = 0.3,
    entry_cutoff: str = "12:00",
) -> Optional[int]:
    """Return +1 (long), -1 (short), or None. Only generates LONG signals."""
    bar_et = bars.index[i]
    bt_time = bar_et.time()
    cutoff  = pd.Timestamp(f"2000-01-01 {entry_cutoff}").time()
    if bt_time > cutoff:
        return None

    today = bar_et.date()
    orb_end = pd.Timestamp(f"2000-01-01 {or_end_time}").time()

    today_bars = bars.iloc[:i+1]
    today_bars = today_bars[today_bars.index.map(lambda t: t.date()) == today]
    or_bars = today_bars[today_bars.index.map(lambda t: t.time()) <= orb_end]

    if len(or_bars) < min_or_bars:
        return None

    # Must be AFTER the OR ended
    if bt_time <= orb_end:
        return None

    or_high = float(or_bars["high"].max())
    or_low  = float(or_bars["low"].min())
    or_range = or_high - or_low

    atr_val = float(atr_s.iloc[i])
    if np.isnan(atr_val) or atr_val <= 0:
        return None
    if or_range < min_range_atr * atr_val:
        return None

    close = float(bars["close"].iloc[i])
    if close > or_high:
        return 1   # LONG breakout
    return None

# ---------------------------------------------------------------------------
# VWAP MR signal
# ---------------------------------------------------------------------------
def vwap_signal(
    bars: pd.DataFrame,
    i: int,
    atr_s: pd.Series,
    vwap_s: pd.Series,
    time_start: str = "10:30",
    time_end: str = "13:30",
    entry_dist_atr: float = 1.0,
    max_dist_atr: float = 3.0,
    max_move_from_open_atr: float = 2.0,
    long_only: bool = True,
) -> Optional[int]:
    bar_et = bars.index[i]
    bt_time = bar_et.time()
    tstart  = pd.Timestamp(f"2000-01-01 {time_start}").time()
    tend    = pd.Timestamp(f"2000-01-01 {time_end}").time()
    if not (tstart <= bt_time <= tend):
        return None

    atr_val = float(atr_s.iloc[i])
    if np.isnan(atr_val) or atr_val <= 0:
        return None

    vwap = float(vwap_s.iloc[i])
    if np.isnan(vwap):
        return None

    close  = float(bars["close"].iloc[i])
    prev_c = float(bars["close"].iloc[i - 1]) if i > 0 else close
    dev    = (close - vwap) / atr_val

    if abs(dev) > max_dist_atr:
        return None

    # Regime gate: skip if price has trended hard from open
    today = bar_et.date()
    today_bars = bars[bars.index.map(lambda t: t.date()) == today]
    if not today_bars.empty and max_move_from_open_atr > 0:
        day_open = float(today_bars["open"].iloc[0])
        move_atr = (close - day_open) / atr_val
        if dev < 0 and move_atr < -max_move_from_open_atr:
            return None
        if dev > 0 and move_atr > max_move_from_open_atr:
            return None

    if dev <= -entry_dist_atr and close > prev_c:
        return 1   # LONG: below VWAP, recovering
    if not long_only and dev >= entry_dist_atr and close < prev_c:
        return -1  # SHORT: above VWAP, declining
    return None

# ---------------------------------------------------------------------------
# Exit checker
# ---------------------------------------------------------------------------
def check_exit(pos: Position, bar: pd.Series, i: int, sess_close: bool):
    h, l, c = bar["high"], bar["low"], bar["close"]
    if sess_close or i >= pos.time_stop_bar:
        exit_p = slip(c, pos.direction, False)
        return True, exit_p, "time_stop" if not sess_close else "session_close"
    if pos.direction == 1:
        if l <= pos.stop_loss:
            return True, slip(pos.stop_loss, 1, False), "stop_loss"
        if h >= pos.profit_target:
            return True, slip(pos.profit_target, 1, False), "profit_target"
    else:
        if h >= pos.stop_loss:
            return True, slip(pos.stop_loss, -1, False), "stop_loss"
        if l <= pos.profit_target:
            return True, slip(pos.profit_target, -1, False), "profit_target"
    return False, 0.0, ""

# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def run_combined(
    bars: pd.DataFrame,
    n_contracts: int = 5,
    # ORB params
    orb_enabled: bool = True,
    or_end_time: str = "10:04",
    min_or_bars: int = 7,
    orb_pt_mult: float = 3.0,
    orb_sl_mult: float = 1.5,
    orb_time_stop_bars: int = 24,
    orb_entry_cutoff: str = "12:00",
    max_orb_per_day: int = 1,
    mom_lookback: int = 3,
    mom_threshold: float = -0.01,
    # VWAP MR params
    vwap_enabled: bool = True,
    vwap_entry_dist_atr: float = 1.0,
    vwap_max_dist_atr: float = 3.0,
    vwap_time_start: str = "10:30",
    vwap_time_end: str = "13:30",
    vwap_pt_mult: float = 2.0,
    vwap_sl_mult: float = 1.5,
    vwap_time_stop_bars: int = 12,
    max_vwap_per_day: int = 2,
    vwap_long_only: bool = True,
    vwap_max_move_from_open_atr: float = 2.0,
    # Risk
    max_daily_loss: float = -2000.0,
    max_total_dd: float = -6000.0,
    atr_period: int = 14,
) -> tuple[list[TradeRecord], list, list, dict]:

    atr_s  = compute_atr(bars["high"], bars["low"], bars["close"], atr_period)
    vwap_s = build_vwap_series(bars)
    daily_closes = build_daily_closes(bars)

    mom_deque: deque[float] = deque(maxlen=mom_lookback + 3)
    mom_last_date = None

    pos_orb:  Optional[Position] = None
    pos_vwap: Optional[Position] = None

    trades: list[TradeRecord] = []
    equity = 50_000.0
    peak_equity = equity
    eq_vals, eq_times = [equity], [bars.index[0]]

    cur_date  = None
    daily_pnl_map: dict = {}
    daily_pnl_running = 0.0
    orb_today  = 0
    vwap_today = 0
    orb_signaled_today = False  # only 1 ORB signal per day

    halted = False  # drawdown halt

    min_bars = atr_period + 3

    for i in range(min_bars, len(bars)):
        bar  = bars.iloc[i]
        bt   = bars.index[i]
        bdate = bt.date()

        # ---- Day boundary ----
        if cur_date is not None and bdate != cur_date:
            daily_pnl_map[cur_date] = daily_pnl_running

            # Seed momentum deque with yesterday's close
            if cur_date in daily_closes and cur_date != mom_last_date:
                mom_deque.append(daily_closes[cur_date])
                mom_last_date = cur_date

            daily_pnl_running = 0.0
            orb_today  = 0
            vwap_today = 0
            orb_signaled_today = False
            if halted and equity > 50_000.0 + max_total_dd * 0.5:
                halted = False  # partial recovery — not used strictly

        cur_date = bdate

        if halted:
            continue

        is_last = (i + 1 >= len(bars)) or (bars.index[i+1].date() != bdate)
        sess_close = is_last or (bt.hour == 15 and bt.minute >= 55)

        # ---- Check halts ----
        if daily_pnl_running <= max_daily_loss:
            # Close open positions, halt for day
            for p in [pos_orb, pos_vwap]:
                if p is not None:
                    ex_p = slip(float(bar["close"]), p.direction, False)
                    tr_pnl = pnl(p.entry_price, ex_p, p.direction, n_contracts)
                    trades.append(TradeRecord(p.strategy, p.direction, p.entry_price, ex_p, tr_pnl, "daily_limit"))
                    daily_pnl_running += tr_pnl
                    equity += tr_pnl
            pos_orb = pos_vwap = None
            continue

        dd = equity - peak_equity
        if dd <= max_total_dd:
            halted = True
            for p in [pos_orb, pos_vwap]:
                if p is not None:
                    ex_p = slip(float(bar["close"]), p.direction, False)
                    tr_pnl = pnl(p.entry_price, ex_p, p.direction, n_contracts)
                    trades.append(TradeRecord(p.strategy, p.direction, p.entry_price, ex_p, tr_pnl, "drawdown_halt"))
                    equity += tr_pnl
            pos_orb = pos_vwap = None
            continue

        # ---- Exit open positions ----
        for attr_name in ["pos_orb", "pos_vwap"]:
            p = pos_orb if attr_name == "pos_orb" else pos_vwap
            if p is None:
                continue
            exited, ex_p, reason = check_exit(p, bar, i, sess_close)
            if exited:
                tr_pnl = pnl(p.entry_price, ex_p, p.direction, n_contracts)
                trades.append(TradeRecord(p.strategy, p.direction, p.entry_price, ex_p, tr_pnl, reason))
                daily_pnl_running += tr_pnl
                equity += tr_pnl
                peak_equity = max(peak_equity, equity)
                eq_vals.append(equity)
                eq_times.append(bt)
                if attr_name == "pos_orb":
                    pos_orb = None
                else:
                    pos_vwap = None

        if sess_close:
            continue

        atr_val = float(atr_s.iloc[i])
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        # ---- ORB entry ----
        if orb_enabled and pos_orb is None and orb_today < max_orb_per_day and not orb_signaled_today:
            if mom_filter_allows(mom_deque, mom_lookback, mom_threshold):
                sig = orb_signal(bars, i, atr_s, or_end_time, min_or_bars, 0.3, orb_entry_cutoff)
                if sig is not None:
                    orb_signaled_today = True
                    ep = slip(float(bar["close"]), sig, True)
                    sl = ep - sig * orb_sl_mult * atr_val
                    pt = ep + sig * orb_pt_mult * atr_val
                    pos_orb = Position(
                        strategy="orb", direction=sig, entry_price=ep, entry_bar=i,
                        stop_loss=sl, profit_target=pt,
                        time_stop_bar=i + orb_time_stop_bars,
                        atr_at_entry=atr_val,
                    )
                    orb_today += 1

        # ---- VWAP MR entry ----
        if vwap_enabled and pos_vwap is None and vwap_today < max_vwap_per_day:
            # Don't enter VWAP if ORB is open (avoid overloading direction)
            sig = vwap_signal(
                bars, i, atr_s, vwap_s,
                vwap_time_start, vwap_time_end,
                vwap_entry_dist_atr, vwap_max_dist_atr,
                vwap_max_move_from_open_atr,
                vwap_long_only,
            )
            if sig is not None:
                ep = slip(float(bar["close"]), sig, True)
                sl = ep - sig * vwap_sl_mult * atr_val
                pt = ep + sig * vwap_pt_mult * atr_val
                pos_vwap = Position(
                    strategy="vwap", direction=sig, entry_price=ep, entry_bar=i,
                    stop_loss=sl, profit_target=pt,
                    time_stop_bar=i + vwap_time_stop_bars,
                    atr_at_entry=atr_val,
                )
                vwap_today += 1

    if cur_date and cur_date not in daily_pnl_map:
        daily_pnl_map[cur_date] = daily_pnl_running

    return trades, eq_vals, eq_times, daily_pnl_map

# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------
def summarize(trades, eq_vals, eq_times, daily_pnl_map, n_contracts, label=""):
    if not trades:
        print(f"  [{label}] No trades.")
        return {}

    orb_t  = [t for t in trades if t.strategy == "orb"]
    vwap_t = [t for t in trades if t.strategy == "vwap"]
    all_t  = trades

    def stats(tlist):
        if not tlist:
            return {}
        wins   = [t for t in tlist if t.pnl > 0]
        losses = [t for t in tlist if t.pnl <= 0]
        total  = sum(t.pnl for t in tlist)
        gp     = sum(t.pnl for t in wins)
        gl     = abs(sum(t.pnl for t in losses))
        return {
            "n": len(tlist), "wins": len(wins), "losses": len(losses),
            "wr": len(wins) / len(tlist),
            "total_pnl": total,
            "avg_win": gp / len(wins) if wins else 0,
            "avg_loss": -gl / len(losses) if losses else 0,
            "pf": gp / gl if gl > 0 else float("inf"),
        }

    eq  = pd.Series(eq_vals, index=eq_times)
    max_dd = float((eq - eq.cummax()).min())
    daily  = pd.Series(daily_pnl_map)
    active_days = daily[daily != 0]
    n_weeks = max(1, len(daily) / 5)
    pnl_total = sum(t.pnl for t in all_t)
    weekly_pnl = pnl_total / n_weeks

    sharpe = float(active_days.mean() / active_days.std() * np.sqrt(252)) if len(active_days) > 1 and active_days.std() > 0 else 0

    s_all  = stats(all_t)
    s_orb  = stats(orb_t)
    s_vwap = stats(vwap_t)

    print(f"\n{'='*65}")
    print(f"  {label}  |  {n_contracts} contracts")
    print(f"{'='*65}")
    print(f"  Period       : {daily.index[0]} → {daily.index[-1]}")
    print(f"  Trading days : {len(daily)}  ({n_weeks:.1f} weeks)")
    print(f"")
    print(f"  ALL TRADES   : {s_all['n']} trades, WR={s_all['wr']:.1%}, "
          f"Total=${pnl_total:,.0f}, $/wk=${weekly_pnl:,.0f}")
    print(f"  Max Drawdown : ${max_dd:,.0f}   Sharpe={sharpe:.2f}")
    print(f"  Profit Factor: {s_all['pf']:.2f}   Avg Win=${s_all['avg_win']:,.0f}  Avg Loss=${s_all['avg_loss']:,.0f}")
    print(f"")
    if orb_t:
        print(f"  ORB          : {s_orb['n']} trades, WR={s_orb['wr']:.1%}, "
              f"Total=${s_orb['total_pnl']:,.0f}  AvgW=${s_orb['avg_win']:,.0f}  AvgL=${s_orb['avg_loss']:,.0f}")
    if vwap_t:
        print(f"  VWAP MR      : {s_vwap['n']} trades, WR={s_vwap['wr']:.1%}, "
              f"Total=${s_vwap['total_pnl']:,.0f}  AvgW=${s_vwap['avg_win']:,.0f}  AvgL=${s_vwap['avg_loss']:,.0f}")

    # Exit reasons
    reasons = {}
    for t in all_t:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
    print(f"  Exit reasons : { {k: round(v/len(all_t)*100,1) for k,v in reasons.items()} }")

    # Weekly breakdown
    daily_s = pd.Series(daily_pnl_map)
    daily_s.index = pd.to_datetime(daily_s.index)
    weekly = daily_s.resample("W-FRI").sum()
    print(f"\n  Weekly PnL:")
    for wk, val in weekly.items():
        bar = "+" * min(30, int(max(0, val) / 200)) + "-" * min(30, int(max(0, -val) / 200))
        print(f"    {str(wk.date()):<12}  ${val:>7,.0f}  {bar}")

    return {
        "n_contracts": n_contracts,
        "n_trades": s_all["n"],
        "win_rate": round(s_all["wr"], 3),
        "total_pnl": round(pnl_total, 2),
        "weekly_pnl": round(weekly_pnl, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "orb_trades": s_orb.get("n", 0),
        "vwap_trades": s_vwap.get("n", 0),
        "orb_wr": round(s_orb.get("wr", 0), 3),
        "vwap_wr": round(s_vwap.get("wr", 0), 3),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", type=int, default=None,
                        help="Fix contract count (default: sweep 5,10,15,20,25,30)")
    parser.add_argument("--no-vwap", action="store_true", help="Disable VWAP MR (ORB only)")
    parser.add_argument("--no-orb",  action="store_true", help="Disable ORB (VWAP only)")
    parser.add_argument("--vwap-dist", type=float, default=1.0,
                        help="VWAP entry distance in ATR (default 1.0)")
    parser.add_argument("--vwap-both", action="store_true",
                        help="Allow SHORT VWAP signals (default: long only)")
    args = parser.parse_args()

    bars = load_bars()

    vwap_long_only = not args.vwap_both

    sweep_contracts = [args.contracts] if args.contracts else [5, 10, 15, 20, 25, 30]
    results = []

    vwap_dir = "long-only" if vwap_long_only else "long+short"
    vwap_desc = "disabled" if args.no_vwap else f"enabled (dist={args.vwap_dist}x ATR, {vwap_dir})"
    print(f"\nVWAP MR: {vwap_desc}")
    print(f"ORB:     {'disabled' if args.no_orb else 'enabled (3d momentum filter)'}")

    for nc in sweep_contracts:
        trades, eq_vals, eq_times, daily_pnl = run_combined(
            bars,
            n_contracts=nc,
            orb_enabled=not args.no_orb,
            vwap_enabled=not args.no_vwap,
            vwap_entry_dist_atr=args.vwap_dist,
            vwap_long_only=vwap_long_only,
            max_daily_loss=-nc * 200.0,      # scale daily loss limit with contracts
            max_total_dd=-nc * 700.0,        # rough drawdown ceiling
        )
        r = summarize(trades, eq_vals, eq_times, daily_pnl, nc,
                      label="ORB+VWAP" if not args.no_vwap and not args.no_orb
                            else "ORB-only" if args.no_vwap else "VWAP-only")
        results.append(r)

    # Summary table
    print(f"\n{'='*65}")
    print(f"  CONTRACT SIZE SWEEP SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Contracts':>10} {'$/week':>9} {'MaxDD':>9} {'WR':>7} {'Trades':>8} {'Sharpe':>8}")
    print(f"  {'-'*56}")
    for r in results:
        if r:
            target_marker = " <-- $6k" if r["weekly_pnl"] >= 6000 else (
                            " <-- $3k" if r["weekly_pnl"] >= 3000 else "")
            print(f"  {r['n_contracts']:>10} {r['weekly_pnl']:>9,.0f} "
                  f"{r['max_drawdown']:>9,.0f} {r['win_rate']:>7.1%} "
                  f"{r['n_trades']:>8} {r['sharpe']:>8.2f}{target_marker}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
