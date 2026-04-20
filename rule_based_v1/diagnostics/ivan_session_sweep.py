"""Ivan BOC — Session Window & Contract Size Scaling Study.

Tests:
  1. Three entry windows: Morning (09:35-12:00), Extended (09:35-14:00), Full Day (09:35-15:45)
  2. Contract size scaling: 2, 5, 10, 20 MNQ — monthly income projection

Best config from prior sweep (Sharpe=4.78):
  swing_days=5, tol=0.40, sl_buf=0.05, long_only=True, PT=3x ATR, SL=1.5x ATR

RUN:
    python rule_based_v1/diagnostics/ivan_session_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "processed"
MNQ_PATH_IS = DATA_DIR / "mnq_5min_aug25_mar26.h5"
MNQ_PATH_OOS = DATA_DIR / "mnq_2026ytd_databento_5min_rth.h5"

for p in [str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

POINT_VALUE = 2.0
SLIP = 0.25
COMMISSION = 0.62
KEY_LEVELS = [0.382, 0.500, 0.618]
ATR_PERIOD = 14
SWING_DAYS = 5
TOUCH_TOL_ATR = 0.40
SL_BUFFER_ATR = 0.05
PT_ATR = 3.0
SL_ATR = 1.5

# Sessions to test: (label, entry_start_min, entry_stop_min, hard_exit_min)
SESSIONS = [
    ("Morning only   09:35–12:00", 9*60+35, 12*60,    15*60+45),
    ("Extended       09:35–14:00", 9*60+35, 14*60,    15*60+45),
    ("Full Day       09:35–15:45", 9*60+35, 15*60+30, 15*60+45),
]

CONTRACT_SIZES = [2, 5, 10, 20]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_bars(path: Path) -> pd.DataFrame:
    with pd.HDFStore(str(path), mode="r") as store:
        df = store["/bars_5min"].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    return df.sort_index()


def _calc_atr(bars: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = bars["close"].shift(1)
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def build_daily_data(df: pd.DataFrame) -> dict:
    atr_series = _calc_atr(df)
    dates = sorted(df.index.normalize().unique())
    rth_by_date = {d: df[df.index.normalize() == d] for d in dates}
    date_list = sorted(rth_by_date.keys())
    daily = {}

    for i, d in enumerate(date_list):
        if i < SWING_DAYS:
            continue
        prior_dates = date_list[max(0, i - SWING_DAYS):i]
        prior_bars_list = [rth_by_date[pd_] for pd_ in prior_dates if pd_ in rth_by_date]
        if not prior_bars_list:
            continue
        prior_bars = pd.concat(prior_bars_list).sort_index()

        swing_high = float(prior_bars["high"].max())
        swing_low = float(prior_bars["low"].min())
        fib_range = swing_high - swing_low
        if fib_range < 1e-6:
            continue

        fib_prices = {lvl: swing_low + lvl * fib_range for lvl in KEY_LEVELS}
        cur_bars = rth_by_date.get(d)
        if cur_bars is None or len(cur_bars) < 2:
            continue

        today_open = float(cur_bars["open"].iloc[0])
        trend_price = swing_low + 0.500 * fib_range
        trend = 1 if today_open > trend_price else -1

        atr_slice = atr_series.loc[:cur_bars.index[0]]
        atr_val = float(atr_slice.iloc[-2]) if len(atr_slice) >= 2 else fib_range / 3.0
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = fib_range / 3.0

        daily[d] = {
            "swing_low": swing_low, "swing_high": swing_high,
            "fib_range": fib_range, "fib_prices": fib_prices,
            "trend": trend, "today_open": today_open,
            "atr_val": atr_val, "cur_bars": cur_bars,
        }
    return daily


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_day(
    day_data: dict,
    entry_start: int,
    entry_stop: int,
    hard_exit: int,
    n_contracts: int = 2,
) -> Optional[dict]:
    fib_prices = day_data["fib_prices"]
    trend = day_data["trend"]
    atr_val = day_data["atr_val"]
    cur_bars = day_data["cur_bars"]

    if trend == -1:  # long-only
        return None

    tolerance = TOUCH_TOL_ATR * atr_val
    sl_buffer = SL_BUFFER_ATR * atr_val
    max_sl_dist = SL_ATR * atr_val
    pt_dist = PT_ATR * atr_val

    signal_candle: Optional[dict] = None
    entry_price = entry_ts = fib_triggered = signal_level_price = None
    tp = sl = None

    bar_list = list(cur_bars.iterrows())

    for idx, (ts, bar) in enumerate(bar_list):
        bar_minute = ts.hour * 60 + ts.minute
        if bar_minute >= entry_stop:
            break
        if bar_minute < entry_start:
            continue

        # Phase 2: BOC entry check
        if signal_candle is not None:
            sc = signal_candle
            if float(bar["high"]) > sc["high"]:
                entry_price = sc["high"] + SLIP
                sl_dist = min(entry_price - (sc["low"] - sl_buffer), max_sl_dist)
                sl = entry_price - sl_dist
                tp = entry_price + pt_dist
                entry_ts = ts
                fib_triggered = sc["fib_level_pct"]
                signal_level_price = sc["fib_level_price"]
                break
            elif float(bar["low"]) < sc["low"] - sl_buffer:
                signal_candle = None  # invalidated

        # Phase 1: find signal candle
        if signal_candle is None:
            for level_pct, level_price in sorted(
                fib_prices.items(), key=lambda kv: abs(kv[1] - float(bar["close"]))
            ):
                touches = float(bar["low"]) <= level_price + tolerance
                near = float(bar["close"]) > level_price - tolerance
                if touches and near:
                    signal_candle = {
                        "high": float(bar["high"]),
                        "low": float(bar["low"]),
                        "fib_level_pct": level_pct,
                        "fib_level_price": level_price,
                    }
                    break

    if entry_ts is None:
        return None

    # Exit simulation
    post_entry = cur_bars[cur_bars.index > entry_ts]
    exit_price = None
    exit_reason = None

    for ts2, bar2 in post_entry.iterrows():
        bar_minute2 = ts2.hour * 60 + ts2.minute
        if bar_minute2 >= hard_exit:
            exit_price = float(bar2["open"]) - SLIP
            exit_reason = "eod"
            break
        if float(bar2["low"]) <= sl:
            exit_price = sl; exit_reason = "stop"; break
        if float(bar2["high"]) >= tp:
            exit_price = tp; exit_reason = "tp"; break

    if exit_price is None:
        last = post_entry.iloc[-1] if len(post_entry) > 0 else None
        exit_price = float(last["close"]) - SLIP if last is not None else entry_price
        exit_reason = "eod"

    pnl_pts = exit_price - entry_price
    pnl_usd = pnl_pts * POINT_VALUE * n_contracts - 2 * COMMISSION * n_contracts

    return {
        "date": str(entry_ts.date()),
        "fib_level_pct": round(fib_triggered, 3),
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2),
        "exit_reason": exit_reason,
        "pnl_usd": round(pnl_usd, 2),
        "entry_hour": entry_ts.hour,
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(trades: list[dict], label: str = "", n_months: float = 8.0) -> dict:
    if not trades:
        return {"N": 0, "WR": 0, "AvgPnL": 0, "TotalPnL": 0,
                "Sharpe": 0, "MaxDD": 0, "PerMonth": 0}
    pnls = [t["pnl_usd"] for t in trades]
    wins = [p for p in pnls if p > 0]

    daily_pnl: dict[str, float] = {}
    for t in trades:
        daily_pnl[t["date"]] = daily_pnl.get(t["date"], 0.0) + t["pnl_usd"]
    dpnl = list(daily_pnl.values())
    mean_d = float(np.mean(dpnl))
    std_d = float(np.std(dpnl, ddof=1)) if len(dpnl) > 1 else 1e-9
    sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 1e-9 else 0.0

    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity - peak).min())

    return {
        "N": len(trades), "WR": round(100*len(wins)/len(trades), 1),
        "AvgPnL": round(float(np.mean(pnls)), 2),
        "TotalPnL": round(float(sum(pnls)), 2),
        "Sharpe": round(sharpe, 2),
        "MaxDD": round(max_dd, 2),
        "PerMonth": round(float(sum(pnls)) / n_months, 0),
    }


def monthly_breakdown(trades: list[dict]) -> dict[str, float]:
    by_month: dict[str, float] = {}
    for t in trades:
        mo = t["date"][:7]
        by_month[mo] = by_month.get(mo, 0.0) + t["pnl_usd"]
    return dict(sorted(by_month.items()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading MNQ bars (in-sample Aug 2025 – Mar 2026)...")
    df_is = load_bars(MNQ_PATH_IS)

    print("Loading MNQ bars (OOS 2026 YTD)...")
    df_oos = load_bars(MNQ_PATH_OOS)

    df_full = pd.concat([df_is, df_oos]).sort_index()
    df_full = df_full[~df_full.index.duplicated(keep="last")]
    print(f"  Combined: {len(df_full):,} bars  {df_full.index[0].date()} → {df_full.index[-1].date()}")

    daily = build_daily_data(df_full)
    all_dates = sorted(daily.keys())

    # Split periods
    is_dates = [d for d in all_dates if str(d)[:10] <= "2025-12-31"]
    oos_dates = [d for d in all_dates if str(d)[:10] >= "2026-01-01"]

    is_months = len(is_dates) / 21.0   # trading days → months approx
    oos_months = len(oos_dates) / 21.0

    print(f"\n  In-sample:  {len(is_dates)} days ({is_months:.1f}mo)")
    print(f"  OOS:        {len(oos_dates)} days ({oos_months:.1f}mo)")

    # -----------------------------------------------------------------------
    # 1. Session window comparison (2 MNQ contracts)
    # -----------------------------------------------------------------------
    print(f"\n{'='*75}")
    print("  SESSION WINDOW COMPARISON  (2 MNQ contracts, long-only)")
    print(f"{'='*75}")
    print(f"\n  {'Session':<32}  {'N':>4}  {'WR':>6}  {'Sharpe':>7}  "
          f"{'MaxDD':>9}  {'TotalPnL':>10}  {'$/mo':>8}")
    print("  " + "-"*75)

    session_results = {}
    for label, e_start, e_stop, h_exit in SESSIONS:
        is_trades = [simulate_day(daily[d], e_start, e_stop, h_exit, 2)
                     for d in is_dates]
        is_trades = [t for t in is_trades if t]
        oos_trades = [simulate_day(daily[d], e_start, e_stop, h_exit, 2)
                      for d in oos_dates]
        oos_trades = [t for t in oos_trades if t]

        s_is = compute_stats(is_trades, n_months=is_months)
        s_oos = compute_stats(oos_trades, n_months=oos_months)
        session_results[label] = {"is": (is_trades, s_is), "oos": (oos_trades, s_oos)}

        tag = "(in-sample)"
        print(f"  {label:<32}  {s_is['N']:>4}  {s_is['WR']:>5.1f}%  {s_is['Sharpe']:>7.2f}  "
              f"${s_is['MaxDD']:>8.0f}  ${s_is['TotalPnL']:>9.0f}  ${s_is['PerMonth']:>7.0f}/mo  {tag}")
        tag = "(OOS 2026)"
        print(f"  {'':32}  {s_oos['N']:>4}  {s_oos['WR']:>5.1f}%  {s_oos['Sharpe']:>7.2f}  "
              f"${s_oos['MaxDD']:>8.0f}  ${s_oos['TotalPnL']:>9.0f}  ${s_oos['PerMonth']:>7.0f}/mo  {tag}")
        print()

    # -----------------------------------------------------------------------
    # 2. Per-session entry hour breakdown (where are the trades coming from?)
    # -----------------------------------------------------------------------
    label_full = SESSIONS[2][0]
    full_is_trades = session_results[label_full]["is"][0]
    print(f"\n{'='*75}")
    print("  ENTRY HOUR BREAKDOWN — Full Day in-sample (where are gains?)")
    print(f"{'='*75}")
    print(f"  {'Hour':>6}  {'N':>4}  {'WR':>6}  {'AvgPnL':>8}  {'TotalPnL':>10}")
    print("  " + "-"*40)
    by_hour: dict[int, list] = {}
    for t in full_is_trades:
        h = t["entry_hour"]
        by_hour.setdefault(h, []).append(t["pnl_usd"])
    for h in sorted(by_hour):
        pnls = by_hour[h]
        wins = [p for p in pnls if p > 0]
        print(f"  {h:>5}h  {len(pnls):>4}  {100*len(wins)/len(pnls):>5.0f}%"
              f"  ${np.mean(pnls):>7.2f}  ${sum(pnls):>9.0f}")

    # -----------------------------------------------------------------------
    # 3. Contract size scaling — best session (morning-only, proven edge)
    # -----------------------------------------------------------------------
    label_morning = SESSIONS[0][0]
    morning_is_trades_base = session_results[label_morning]["is"][0]

    print(f"\n{'='*75}")
    print("  CONTRACT SIZE SCALING — Morning session (proven Sharpe=4.78)")
    print(f"{'='*75}")
    print(f"\n  In-sample (Aug–Dec 2025, {is_months:.0f} months):")
    print(f"  {'Contracts':>10}  {'$/trade avg':>12}  {'$/month avg':>12}  "
          f"{'Total PnL':>11}  {'MaxDD':>9}  {'Combine risk?':>14}")
    print("  " + "-"*70)
    for nc in CONTRACT_SIZES:
        # Scale from 2-contract base by ratio nc/2
        ratio = nc / 2
        scaled_pnls = [t["pnl_usd"] * ratio for t in morning_is_trades_base]
        total = sum(scaled_pnls)
        avg_trade = float(np.mean(scaled_pnls))
        per_month = total / is_months
        equity = np.cumsum(scaled_pnls)
        peak = np.maximum.accumulate(equity)
        max_dd = float((equity - peak).min())
        combine_ok = "OK" if abs(max_dd) < 1800 else "EXCEEDS $1800 DD LIMIT"
        print(f"  {nc:>8} MNQ  ${avg_trade:>10.0f}  ${per_month:>10.0f}/mo  "
              f"${total:>10.0f}  ${max_dd:>8.0f}  {combine_ok}")

    print(f"\n  OOS 2026 YTD ({oos_months:.1f} months):")
    morning_oos_trades_base = session_results[label_morning]["oos"][0]
    print(f"  {'Contracts':>10}  {'$/trade avg':>12}  {'$/month avg':>12}  "
          f"{'Total PnL':>11}  {'MaxDD':>9}")
    print("  " + "-"*60)
    for nc in CONTRACT_SIZES:
        if not morning_oos_trades_base:
            print(f"  {nc:>8} MNQ  (no trades in OOS)")
            break
        ratio = nc / 2
        scaled_pnls = [t["pnl_usd"] * ratio for t in morning_oos_trades_base]
        total = sum(scaled_pnls)
        avg_trade = float(np.mean(scaled_pnls))
        per_month = total / oos_months
        equity = np.cumsum(scaled_pnls)
        peak = np.maximum.accumulate(equity)
        max_dd = float((equity - peak).min())
        print(f"  {nc:>8} MNQ  ${avg_trade:>10.0f}  ${per_month:>10.0f}/mo  "
              f"${total:>10.0f}  ${max_dd:>8.0f}")

    # -----------------------------------------------------------------------
    # 4. Monthly breakdown — morning session, 10 MNQ (funded account goal)
    # -----------------------------------------------------------------------
    print(f"\n{'='*75}")
    print("  MONTHLY PnL — 10 MNQ contracts (funded account target)")
    print(f"{'='*75}")
    ratio_10 = 10 / 2
    all_morning_trades = (session_results[label_morning]["is"][0] +
                          session_results[label_morning]["oos"][0])
    months = monthly_breakdown(all_morning_trades)
    print(f"\n  {'Month':>8}  {'Trades':>7}  {'PnL (10 MNQ)':>14}  {'Cumulative':>12}")
    cumulative = 0.0
    for mo, pnl_base in months.items():
        mo_trades = [t for t in all_morning_trades if t["date"].startswith(mo)]
        n_mo = len(mo_trades)
        scaled = pnl_base * ratio_10
        cumulative += scaled
        bar = "#" * max(0, int(scaled / 200))
        print(f"  {mo}  {n_mo:>7}  ${scaled:>12.0f}  ${cumulative:>11.0f}  {bar}")

    print(f"\n  Total (10 MNQ): ${sum(months.values()) * ratio_10:,.0f}")
    print(f"  Avg per month:  ${sum(months.values()) * ratio_10 / len(months):,.0f}/mo")

    # -----------------------------------------------------------------------
    # 5. Path to $10k/month — what it takes
    # -----------------------------------------------------------------------
    base_per_month = session_results[label_morning]["is"][1]["PerMonth"]
    print(f"\n{'='*75}")
    print("  PATH TO $10k/month")
    print(f"{'='*75}")
    print(f"\n  Baseline (2 MNQ, morning only): ${base_per_month:.0f}/mo")
    print()
    targets = [3000, 5000, 7500, 10000]
    for target in targets:
        mnq_needed = target / (base_per_month / 2)
        nq_equiv = mnq_needed / 10
        print(f"  ${target:>6}/mo  →  {mnq_needed:>5.0f} MNQ  or  {nq_equiv:>4.1f} NQ contracts")

    print(f"\n  Note: NQ is 10x MNQ ($20/point vs $2/point)")
    print(f"  Note: Funded Topstep accounts can scale to 15 contracts (MNQ)")
    print(f"        For $10k/mo you need either funded NQ account or multiple funded accounts")


if __name__ == "__main__":
    main()
