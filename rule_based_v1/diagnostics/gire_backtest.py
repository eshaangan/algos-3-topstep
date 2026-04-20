"""GIRE (Gap Inventory Repair Engine) Backtest — MES 1-min bars.

Strategy: fade extended overnight gaps that show opening rejection,
targeting inventory repair back to prior-day close.

Instrument: MES, $5/pt, 1 contract, 0.25 pt slippage per side.
Data: data/processed/mes_1m_bars_cache.h5, key /bars_1m
      2025-06-29 to 2025-12-26 ET

Run:
    python rule_based_v1/diagnostics/gire_backtest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "processed"
DIAG_DIR = ROOT / "rule_based_v1" / "diagnostics"
MES_PATH = DATA_DIR / "mes_1m_bars_cache.h5"
RESULTS_PATH = DIAG_DIR / "gire_results.json"

for p in [str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POINT_VALUE = 5.0       # $/pt MES
SLIP = 0.25             # slippage per side (pts)
N_CONTRACTS = 1

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bars() -> pd.DataFrame:
    with pd.HDFStore(str(MES_PATH), mode="r") as store:
        df = store["/bars_1m"].copy()
    df = df.set_index("timestamp")
    df.index = df.index.tz_convert("US/Eastern")
    df = df.sort_index()
    return df


def is_rth(idx: pd.DatetimeIndex) -> pd.Series:
    """True for bars whose time falls inside RTH 09:30–15:59."""
    minutes = idx.hour * 60 + idx.minute
    return pd.Series((minutes >= 570) & (minutes <= 959), index=idx)


# ---------------------------------------------------------------------------
# Build per-day structures
# ---------------------------------------------------------------------------

def build_daily_data(df: pd.DataFrame) -> dict:
    """Return dict keyed by trading date with all per-day structures."""
    rth_mask = is_rth(df.index)
    rth = df[rth_mask]

    rth_dates = sorted(rth.index.normalize().unique())
    ET = pd.Timestamp("2000-01-01", tz="US/Eastern").tzinfo

    # Build per-date RTH lookup
    rth_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    for d in rth_dates:
        day_rth = rth[rth.index.normalize() == d]
        if len(day_rth) > 0:
            rth_by_date[d] = day_rth

    date_list = sorted(rth_by_date.keys())

    # 20-day rolling ATR and median range
    ranges = {d: (rth_by_date[d]["high"].max() - rth_by_date[d]["low"].min())
              for d in date_list}
    range_series = pd.Series(ranges)
    atr_series = range_series.rolling(20, min_periods=5).mean()
    median_range_series = range_series.rolling(20, min_periods=5).median()

    daily: dict = {}

    for i, d in enumerate(date_list):
        if i == 0:
            continue  # need prior day

        prev_d = date_list[i - 1]
        if prev_d not in rth_by_date:
            continue

        prev_rth = rth_by_date[prev_d]
        C_y = prev_rth["close"].iloc[-1]
        H_y = prev_rth["high"].max()
        L_y = prev_rth["low"].min()

        ATR_d = float(atr_series.get(prev_d, np.nan))
        med_range = float(median_range_series.get(prev_d, np.nan))
        if np.isnan(ATR_d) or ATR_d <= 0:
            continue

        # ETH overnight: 16:00 prev_d to 09:29 current day d
        eth_start = prev_d.replace(hour=16, minute=0, second=0, tzinfo=ET)
        eth_end = d.replace(hour=9, minute=29, second=59, tzinfo=ET)
        eth_bars = df[(df.index >= eth_start) & (df.index <= eth_end)]
        if len(eth_bars) == 0:
            continue

        H_eth = eth_bars["high"].max()
        L_eth = eth_bars["low"].min()
        eth_range = H_eth - L_eth
        eth_mid = (H_eth + L_eth) / 2.0

        # Current RTH
        cur_rth = rth_by_date[d]
        O_t = cur_rth.iloc[0]["open"]

        # Signal components
        G_t = (O_t - C_y) / ATR_d
        CLV_y = (C_y - L_y) / (H_y - L_y + 1e-9)
        E_y = (H_y - L_y) / (med_range + 1e-9) if (not np.isnan(med_range) and med_range > 0) else np.nan
        S_t = (O_t - eth_mid) / (eth_range + 1e-9)

        # First two 5-min bars (grouped from 1-min bars)
        bar1_mask = (cur_rth.index.hour == 9) & (cur_rth.index.minute >= 30) & (cur_rth.index.minute <= 34)
        bar2_mask = (cur_rth.index.hour == 9) & (cur_rth.index.minute >= 35) & (cur_rth.index.minute <= 39)

        b1 = cur_rth[bar1_mask]
        b2 = cur_rth[bar2_mask]

        if len(b1) == 0:
            continue

        first_5min_open = b1["open"].iloc[0]
        first_5min_high = b1["high"].max()
        first_5min_low = b1["low"].min()
        first_5min_close = b1["close"].iloc[-1]

        # Opening failure for SHORT (up-gap fade)
        b1_range = first_5min_high - first_5min_open
        fail_short_a = first_5min_close < (first_5min_open + 0.3 * b1_range)
        fail_short_b = (len(b2) > 0) and (b2["low"].min() < first_5min_low)
        F_t_short = fail_short_a or fail_short_b

        # Opening failure for LONG (down-gap fade)
        b1_hl_range = first_5min_high - first_5min_low
        fail_long_a = first_5min_close > (first_5min_low + 0.7 * b1_hl_range)
        fail_long_b = (len(b2) > 0) and (b2["high"].max() > first_5min_high)
        F_t_long = fail_long_a or fail_long_b

        # Takeout check on first 2 RTH (1-min) bars
        first_2bars = cur_rth.iloc[:2]
        opening_high_taken = first_2bars["high"].max() > first_5min_high
        opening_low_taken = first_2bars["low"].min() < first_5min_low

        daily[d] = {
            "C_y": C_y,
            "H_y": H_y,
            "L_y": L_y,
            "ATR_d": ATR_d,
            "med_range": med_range,
            "H_eth": H_eth,
            "L_eth": L_eth,
            "eth_mid": eth_mid,
            "O_t": O_t,
            "G_t": G_t,
            "CLV_y": CLV_y,
            "E_y": E_y,
            "S_t": S_t,
            "F_t_short": F_t_short,
            "F_t_long": F_t_long,
            "first_5min_high": first_5min_high,
            "first_5min_low": first_5min_low,
            "opening_high_taken": opening_high_taken,
            "opening_low_taken": opening_low_taken,
            "cur_rth": cur_rth,
        }

    return daily


# ---------------------------------------------------------------------------
# Condition funnel diagnostic
# ---------------------------------------------------------------------------

def print_funnel(daily: dict, g_star: float):
    """Print step-by-step filter attrition for each direction."""
    for direction, sign in [("SHORT (up-gap)", 1), ("LONG (down-gap)", -1)]:
        c_g = c_clv = c_ext = c_st = c_ft = c_entry = 0
        for dd in daily.values():
            G = dd["G_t"]
            if (sign == 1 and G > g_star) or (sign == -1 and G < -g_star):
                c_g += 1
                clv_ok = (dd["CLV_y"] > 0.8) if sign == 1 else (dd["CLV_y"] < 0.2)
                if clv_ok:
                    c_clv += 1
                    ey_ok = (not np.isnan(dd["E_y"])) and (dd["E_y"] > 1.1)
                    if ey_ok:
                        c_ext += 1
                        st_ok = (dd["S_t"] > 0.4) if sign == 1 else (dd["S_t"] < -0.4)
                        if st_ok:
                            c_st += 1
                            ft_ok = dd["F_t_short"] if sign == 1 else dd["F_t_long"]
                            tak_ok = (not dd["opening_high_taken"]) if sign == 1 else (not dd["opening_low_taken"])
                            if ft_ok and tak_ok:
                                c_ft += 1
        print(f"  {direction}: G*={g_star}")
        print(f"    Gap filter    : {c_g:3d}")
        print(f"    + CLV filter  : {c_clv:3d}")
        print(f"    + E_y > 1.1   : {c_ext:3d}")
        print(f"    + S_t ±0.4    : {c_st:3d}")
        print(f"    + F_t + no-tak: {c_ft:3d}")


# ---------------------------------------------------------------------------
# Single-day trade simulation
# ---------------------------------------------------------------------------

def simulate_day(
    day_data: dict,
    g_star: float,
    clv_thresh: float = 0.8,      # CLV_y > thresh for shorts, < (1-thresh) for longs
    ey_thresh: float = 1.1,        # E_y > thresh (0 = disabled)
    st_thresh: float = 0.4,        # |S_t| > thresh (0 = disabled)
    use_opening_failure: bool = True,
    time_stop_hour: int = 12,
) -> Optional[dict]:
    """Return trade dict or None if no signal/entry on this day."""

    G_t = day_data["G_t"]
    CLV_y = day_data["CLV_y"]
    E_y = day_data["E_y"]
    S_t = day_data["S_t"]
    ATR_d = day_data["ATR_d"]
    C_y = day_data["C_y"]
    first_5min_high = day_data["first_5min_high"]
    first_5min_low = day_data["first_5min_low"]
    cur_rth = day_data["cur_rth"]

    if E_y is None or np.isnan(E_y):
        return None

    # Extension filter (E_y threshold; 0 = disabled)
    if ey_thresh > 0 and E_y <= ey_thresh:
        return None

    # Check direction conditions
    direction = None

    # SHORT: up-gap fade
    short_core = G_t > g_star and CLV_y > clv_thresh
    if st_thresh > 0:
        short_core = short_core and S_t > st_thresh
    if short_core:
        ft_ok = (not use_opening_failure) or day_data["F_t_short"]
        tak_ok = not day_data["opening_high_taken"]
        if ft_ok and tak_ok:
            direction = "SHORT"

    # LONG: down-gap fade
    if direction is None:
        long_core = G_t < -g_star and CLV_y < (1.0 - clv_thresh)
        if st_thresh > 0:
            long_core = long_core and S_t < -st_thresh
        if long_core:
            ft_ok = (not use_opening_failure) or day_data["F_t_long"]
            tak_ok = not day_data["opening_low_taken"]
            if ft_ok and tak_ok:
                direction = "LONG"

    if direction is None:
        return None

    # Entry: scan bars from 9:35 onward
    scan = cur_rth[
        (cur_rth.index.hour * 60 + cur_rth.index.minute) >= 575
    ]

    entry_price = None
    entry_time = None

    if direction == "SHORT":
        for ts, bar in scan.iterrows():
            if bar["low"] < first_5min_low:
                entry_price = first_5min_low - SLIP
                entry_time = ts
                break
    else:  # LONG
        for ts, bar in scan.iterrows():
            if bar["high"] > first_5min_high:
                entry_price = first_5min_high + SLIP
                entry_time = ts
                break

    if entry_price is None:
        return None

    # TP1 = prior close
    TP = C_y

    # Stop calculation
    if direction == "SHORT":
        stop_ref = max(first_5min_high, day_data["O_t"])
        raw_sl = stop_ref + 0.5 * ATR_d * 0.25
        sl_dist = raw_sl - entry_price
        sl_dist = min(sl_dist, 2.0 * ATR_d)
        stop_price = entry_price + sl_dist + SLIP
    else:
        stop_ref = min(first_5min_low, day_data["O_t"])
        raw_sl = stop_ref - 0.5 * ATR_d * 0.25
        sl_dist = entry_price - raw_sl
        sl_dist = min(sl_dist, 2.0 * ATR_d)
        stop_price = entry_price - sl_dist - SLIP

    # Time stop
    time_stop_minutes = time_stop_hour * 60

    # Scan bars after entry
    post_entry = cur_rth[cur_rth.index > entry_time]
    exit_price = None
    exit_reason = None

    for ts, bar in post_entry.iterrows():
        bar_minutes = ts.hour * 60 + ts.minute

        if bar_minutes >= time_stop_minutes:
            exit_price = bar["close"] - SLIP if direction == "SHORT" else bar["close"] + SLIP
            exit_reason = "time"
            break

        if direction == "SHORT":
            if bar["high"] >= stop_price:
                exit_price = stop_price
                exit_reason = "stop"
                break
            if bar["low"] <= TP:
                exit_price = TP - SLIP
                exit_reason = "tp"
                break
        else:
            if bar["low"] <= stop_price:
                exit_price = stop_price
                exit_reason = "stop"
                break
            if bar["high"] >= TP:
                exit_price = TP + SLIP
                exit_reason = "tp"
                break

    if exit_price is None:
        # End of RTH
        if len(post_entry) > 0:
            last_bar = post_entry.iloc[-1]
            exit_price = last_bar["close"] - SLIP if direction == "SHORT" else last_bar["close"] + SLIP
            exit_reason = "eod"
        else:
            exit_price = entry_price
            exit_reason = "eod"

    # PnL calculation
    if direction == "SHORT":
        pnl_pts = entry_price - exit_price
    else:
        pnl_pts = exit_price - entry_price

    pnl_usd = pnl_pts * POINT_VALUE * N_CONTRACTS

    return {
        "date": str(entry_time.date()),
        "direction": direction,
        "G_t": round(G_t, 4),
        "CLV_y": round(CLV_y, 4),
        "E_y": round(E_y, 4),
        "S_t": round(S_t, 4),
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "stop_price": round(stop_price, 4),
        "TP": round(TP, 4),
        "exit_reason": exit_reason,
        "pnl_pts": round(pnl_pts, 4),
        "pnl_usd": round(pnl_usd, 2),
        "abs_G": round(abs(G_t), 4),
    }


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    daily: dict,
    g_star: float,
    clv_thresh: float = 0.8,
    ey_thresh: float = 1.1,
    st_thresh: float = 0.4,
    use_opening_failure: bool = True,
    time_stop_hour: int = 12,
) -> list[dict]:
    trades = []
    for d in sorted(daily.keys()):
        result = simulate_day(
            daily[d],
            g_star=g_star,
            clv_thresh=clv_thresh,
            ey_thresh=ey_thresh,
            st_thresh=st_thresh,
            use_opening_failure=use_opening_failure,
            time_stop_hour=time_stop_hour,
        )
        if result is not None:
            trades.append(result)
    return trades


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"N": 0, "WR": 0.0, "AvgPnL": 0.0, "TotalPnL": 0.0, "Sharpe": 0.0, "MaxDD": 0.0}

    pnls = [t["pnl_usd"] for t in trades]
    wins = [p for p in pnls if p > 0]

    # Daily PnL for Sharpe
    daily_pnl: dict[str, float] = {}
    for t in trades:
        daily_pnl[t["date"]] = daily_pnl.get(t["date"], 0.0) + t["pnl_usd"]
    dpnl = list(daily_pnl.values())
    mean_d = np.mean(dpnl) if dpnl else 0.0
    std_d = np.std(dpnl, ddof=1) if len(dpnl) > 1 else 1e-9
    sharpe = (mean_d / std_d) * np.sqrt(252) if std_d > 1e-9 else 0.0

    # Max drawdown (trade-level equity curve)
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(dd.min())

    return {
        "N": len(trades),
        "WR": round(100.0 * len(wins) / len(trades), 1),
        "AvgPnL": round(np.mean(pnls), 2),
        "TotalPnL": round(sum(pnls), 2),
        "Sharpe": round(sharpe, 3),
        "MaxDD": round(max_dd, 2),
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def monotonicity_buckets(trades: list[dict]) -> list[dict]:
    buckets = [
        (0.3, 0.5, "[0.3,0.5)"),
        (0.5, 0.7, "[0.5,0.7)"),
        (0.7, 1.0, "[0.7,1.0)"),
        (1.0, 9.9, "[1.0,inf)"),
    ]
    results = []
    for lo, hi, label in buckets:
        bt = [t for t in trades if lo <= t["abs_G"] < hi]
        if bt:
            pnls = [t["pnl_usd"] for t in bt]
            wins = sum(1 for p in pnls if p > 0)
            results.append({
                "bucket": label,
                "N": len(bt),
                "WR": round(100.0 * wins / len(bt), 1),
                "AvgPnL": round(np.mean(pnls), 2),
            })
        else:
            results.append({"bucket": label, "N": 0, "WR": 0.0, "AvgPnL": 0.0})
    return results


def asymmetry_stats(trades: list[dict]) -> dict:
    up = [t for t in trades if t["direction"] == "SHORT"]
    dn = [t for t in trades if t["direction"] == "LONG"]

    def _s(ts):
        if not ts:
            return {"N": 0, "WR": 0.0, "AvgPnL": 0.0, "TotalPnL": 0.0}
        pnls = [t["pnl_usd"] for t in ts]
        wins = sum(1 for p in pnls if p > 0)
        return {
            "N": len(ts),
            "WR": round(100.0 * wins / len(ts), 1),
            "AvgPnL": round(np.mean(pnls), 2),
            "TotalPnL": round(sum(pnls), 2),
        }

    return {"up_gaps_short": _s(up), "down_gaps_long": _s(dn)}


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_table(title: str, rows: list[dict]):
    if not rows:
        print(f"\n{title}: no data")
        return
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    keys = list(rows[0].keys())
    col_w = max(12, max(len(k) for k in keys) + 2)
    header = "".join(f"{k:<{col_w}}" for k in keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = ""
        for k, v in row.items():
            if isinstance(v, float):
                line += f"{v:<{col_w}.2f}"
            else:
                line += f"{str(v):<{col_w}}"
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_grid_sweep(daily: dict) -> list[dict]:
    """Full parameter grid sweep. Returns list of result dicts sorted by Sharpe."""
    import itertools

    g_stars        = [0.2, 0.3, 0.4, 0.5, 0.7]
    st_thresholds  = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]   # 0.0 = disabled
    clv_thresholds = [0.7, 0.75, 0.8]
    ey_thresholds  = [0.0, 1.0, 1.1]                     # 0.0 = disabled
    time_stops     = [11, 12, 13]
    open_fail      = [True, False]

    total = (len(g_stars) * len(st_thresholds) * len(clv_thresholds)
             * len(ey_thresholds) * len(time_stops) * len(open_fail))

    print(f"\n{'='*75}")
    print(f"  PARAMETER GRID SWEEP  ({total} combinations)")
    print(f"{'='*75}")

    rows = []
    for gs, st, clv, ey, ts_hr, of in itertools.product(
        g_stars, st_thresholds, clv_thresholds, ey_thresholds, time_stops, open_fail
    ):
        trades = run_backtest(
            daily, g_star=gs, clv_thresh=clv, ey_thresh=ey,
            st_thresh=st, use_opening_failure=of, time_stop_hour=ts_hr,
        )
        stats = compute_stats(trades)
        rows.append({
            "g_star": gs, "st_thresh": st, "clv_thresh": clv,
            "ey_thresh": ey, "ts_hr": ts_hr, "open_fail": of,
            **stats,
        })

    # Sort by Sharpe descending, require N >= 3
    qualified = [r for r in rows if r["N"] >= 3]
    qualified.sort(key=lambda x: x["Sharpe"], reverse=True)

    # Print top 25
    print(f"\n{'g*':>5} {'st':>5} {'clv':>5} {'ey':>5} {'ts':>4} {'of':>5}  "
          f"{'N':>4}  {'WR%':>6}  {'AvgPnL':>8}  {'Sharpe':>7}  {'MaxDD':>9}  {'TotalPnL':>10}")
    print("-" * 90)
    for r in qualified[:25]:
        print(
            f"{r['g_star']:>5.2f} {r['st_thresh']:>5.2f} {r['clv_thresh']:>5.2f} "
            f"{r['ey_thresh']:>5.2f} {r['ts_hr']:>4d} {str(r['open_fail']):>5}  "
            f"{r['N']:>4}  {r['WR']:>5.1f}%  "
            f"${r['AvgPnL']:>7.2f}  {r['Sharpe']:>7.3f}  "
            f"${r['MaxDD']:>8.2f}  ${r['TotalPnL']:>9.2f}"
        )

    if qualified:
        best = qualified[0]
        print(f"\nBest config: g*={best['g_star']}, st={best['st_thresh']}, "
              f"clv={best['clv_thresh']}, ey={best['ey_thresh']}, "
              f"ts_hr={best['ts_hr']}, open_fail={best['open_fail']}")
        print(f"  N={best['N']}, WR={best['WR']}%, AvgPnL=${best['AvgPnL']:.2f}, "
              f"Sharpe={best['Sharpe']:.3f}, MaxDD=${best['MaxDD']:.2f}")

        # Print all trades for best config
        best_trades = run_backtest(
            daily, g_star=best["g_star"], clv_thresh=best["clv_thresh"],
            ey_thresh=best["ey_thresh"], st_thresh=best["st_thresh"],
            use_opening_failure=best["open_fail"], time_stop_hour=best["ts_hr"],
        )
        print(f"\n--- All trades for best config ---")
        print(f"  {'Date':<12} {'Dir':<6} {'G_t':>6} {'CLV':>6} {'E_y':>5} {'S_t':>6}  "
              f"{'Entry':>7} {'Exit':>7} {'Reason':<10} {'PnL':>8}")
        print("  " + "-" * 80)
        for t in best_trades:
            print(
                f"  {t['date']:<12} {t['direction']:<6} {t['G_t']:>6.3f} "
                f"{t['CLV_y']:>6.3f} {t['E_y']:>5.2f} {t['S_t']:>6.3f}  "
                f"{t['entry_price']:>7.2f} {t['exit_price']:>7.2f} "
                f"{t['exit_reason']:<10} ${t['pnl_usd']:>7.2f}"
            )

        # Monotonicity & asymmetry for best config
        mono = monotonicity_buckets(best_trades)
        print(f"\n--- Monotonicity (best config) ---")
        print(f"  {'Bucket':<12} {'N':>4}  {'WR%':>6}  {'AvgPnL':>8}")
        for m in mono:
            print(f"  {m['bucket']:<12} {m['N']:>4}  {m['WR']:>5.1f}%  ${m['AvgPnL']:>7.2f}")

        asym = asymmetry_stats(best_trades)
        print(f"\n--- Asymmetry (best config) ---")
        for lbl, s in [("Shorts", asym["up_gaps_short"]), ("Longs", asym["down_gaps_long"])]:
            print(f"  {lbl}: N={s['N']}, WR={s['WR']}%, AvgPnL=${s['AvgPnL']:.2f}")

    else:
        print("\nNo config produced N >= 3 trades.")
        best = None
        best_trades = []
        mono = []
        asym = {}

    return rows, best, best_trades, mono, asym


def main():
    print("Loading 1-min MES bars...")
    df = load_bars()
    print(f"  Loaded {len(df):,} bars, {df.index[0]} to {df.index[-1]}")

    print("Building daily structures...")
    daily = build_daily_data(df)
    print(f"  {len(daily)} trading days with full data")

    # ------------------------------------------------------------------
    # Grid sweep
    # ------------------------------------------------------------------
    all_rows, best, best_trades, mono, asym = run_grid_sweep(daily)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "grid_sweep": [
            {k: (v if not isinstance(v, bool) else int(v)) for k, v in r.items()}
            for r in all_rows
        ],
        "best_config": best,
        "best_trades": best_trades,
        "monotonicity": mono,
        "asymmetry": asym,
        "days_in_dataset": len(daily),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
