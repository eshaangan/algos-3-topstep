"""
VVR (Void Velocity Reversion) Backtest — MNQ
Runs on two datasets:
  1. MNQ 1-min  (56 days, Jan–Mar 2026)
  2. MNQ 5-min  (162 days, Aug 2025–Mar 2026)
Results saved to rule_based_v1/diagnostics/vvr_mnq_results.json
"""

import json
import math
from collections import deque
from datetime import time as dtime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared constants (override per-run via run_vvr arguments)
# ---------------------------------------------------------------------------
SHOCK_WINDOW = 60
ATR_WINDOW = 20
DEG_WINDOW = 60
ORX_WINDOW = 60
SHOCK_PCTL = 80
DEG_PCTL = 50
ORX_PCTL = 50
NO_ENTRY_AFTER = dtime(14, 45)
MAX_LOSING_TRADES_SESSION = 3
EOD_EXIT_1MIN = dtime(15, 55)
EOD_EXIT_5MIN = dtime(15, 55)
MAX_ACTIVE_GAPS = 3
PAUSE_AFTER_CONSEC_LOSSES = 2
SLIP = 0.25
N_CONTRACTS = 1

RESULTS_PATH = "rule_based_v1/diagnostics/vvr_mnq_results.json"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_mnq_1min() -> pd.DataFrame:
    df = pd.read_hdf(
        "data/processed/mnq_2026ytd_1min.h5", key="/bars_1min"
    )
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            raise ValueError("MNQ 1-min: no timestamp index or column found")
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    df = df.sort_index()

    minute_of_day = df.index.hour * 60 + df.index.minute
    df = df[(minute_of_day >= 570) & (minute_of_day <= 959)].copy()
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    return df


def load_mnq_5min() -> pd.DataFrame:
    df = pd.read_hdf(
        "data/processed/mnq_5min_aug25_mar26.h5", key="/bars_5min"
    )
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            raise ValueError("MNQ 5-min: no timestamp index or column found")
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    df = df.sort_index()

    # RTH: 09:30–15:55 ET (minute 570–955)
    minute_of_day = df.index.hour * 60 + df.index.minute
    df = df[(minute_of_day >= 570) & (minute_of_day <= 955)].copy()
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    return df


# ---------------------------------------------------------------------------
# Rolling metric arrays
# ---------------------------------------------------------------------------
def compute_rolling_metrics(df: pd.DataFrame):
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    ranges = high - low
    atr_arr = np.full(n, np.nan)
    med_range = np.full(n, np.nan)

    atr_buf = deque(maxlen=ATR_WINDOW)
    shock_buf = deque(maxlen=SHOCK_WINDOW)

    for i in range(n):
        atr_buf.append(ranges[i])
        shock_buf.append(ranges[i])
        atr_arr[i] = np.mean(atr_buf)
        med_range[i] = np.median(shock_buf)

    return atr_arr, med_range


# ---------------------------------------------------------------------------
# Core backtest — parameterised by point_value, time_stop_bars, pause_bars
# ---------------------------------------------------------------------------
def run_vvr(
    bars_df: pd.DataFrame,
    point_value: float,
    time_stop_bars: int,
    pause_bars: int,
    label: str,
    shock_pctl: int = SHOCK_PCTL,
    deg_pctl: int = DEG_PCTL,
    collect_trades: bool = True,
) -> list:
    """
    Run the VVR strategy on bars_df and return a list of trade dicts.

    Parameters
    ----------
    bars_df        : DataFrame with columns open/high/low/close and a
                     tz-aware 'timestamp' column (US/Eastern).
    point_value    : Dollar value per point per contract.
    time_stop_bars : Number of bars after entry before time-stop fires.
    pause_bars     : Number of bars to pause after PAUSE_AFTER_CONSEC_LOSSES.
    label          : Human-readable label stored in each trade dict.
    """
    ts_series = bars_df["timestamp"]
    open_ = bars_df["open"].values
    high = bars_df["high"].values
    low = bars_df["low"].values
    close = bars_df["close"].values
    n = len(bars_df)

    atr_arr, med_range = compute_rolling_metrics(bars_df)

    bar_hour = ts_series.dt.hour.values
    bar_minute = ts_series.dt.minute.values
    bar_date = ts_series.dt.date.values

    eps = 1e-9
    trades = []

    shock_history = deque(maxlen=SHOCK_WINDOW)
    deg_history = deque(maxlen=DEG_WINDOW)
    orx_history = deque(maxlen=ORX_WINDOW)

    current_date = None
    consec_losses = 0
    session_losses = 0
    pause_until_bar = -1
    active_trade = None
    active_gaps = []

    i = 2
    while i < n:
        today = bar_date[i]

        if today != current_date:
            current_date = today
            consec_losses = 0
            session_losses = 0
            pause_until_bar = -1
            active_gaps = []
            if active_trade is not None:
                active_trade = None

        # --- Manage active trade exits ---
        if active_trade is not None:
            entry_price = active_trade["entry_price"]
            stop_price = active_trade["stop_price"]
            profit_target = active_trade["profit_target"]
            time_stop_bar = active_trade["time_stop_bar"]
            k = active_trade["entry_bar"]

            exited = False
            exit_price = None
            exit_reason = None

            bar_t = dtime(int(bar_hour[i]), int(bar_minute[i]))
            if bar_t >= EOD_EXIT_1MIN:
                exit_price = close[i] - SLIP
                exit_reason = "eod"
                exited = True
            elif low[i] <= stop_price:
                exit_price = stop_price
                exit_reason = "stop"
                exited = True
            elif high[i] >= profit_target:
                exit_price = profit_target
                exit_reason = "tp1"
                exited = True
            elif i >= time_stop_bar:
                exit_price = close[i] - SLIP
                exit_reason = "time"
                exited = True

            if exited:
                pnl = (exit_price - entry_price) * point_value * N_CONTRACTS
                is_win = pnl > 0

                if is_win:
                    consec_losses = 0
                else:
                    consec_losses += 1
                    session_losses += 1

                if collect_trades:
                    gap_idx = active_trade["gap_bar_idx"]
                    trades.append({
                        "label": label,
                        "date": str(bar_date[gap_idx]),
                        "entry_time": str(ts_series.iloc[k]),
                        "direction": "LONG",
                        "Shock": float(active_trade["Shock"]),
                        "Deg": float(active_trade["Deg"]),
                        "ORX": float(active_trade["ORX"]),
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_price),
                        "exit_reason": exit_reason,
                        "pnl": float(round(pnl, 2)),
                    })

                active_trade = None

        # --- Gap detection at bar i ---
        if i >= 2 and atr_arr[i] > 0 and med_range[i] > 0:
            H3 = max(high[i - 2], high[i - 1], high[i])
            L3 = min(low[i - 2], low[i - 1], low[i])
            three_bar_range = H3 - L3

            direction_ok = close[i] < open_[i - 2]

            if direction_ok:
                Shock = (three_bar_range / (med_range[i] + eps)) * (
                    abs(close[i] - open_[i - 2]) / (three_bar_range + eps)
                )
            else:
                Shock = 0.0

            gap_exists = high[i] < low[i - 2]

            if gap_exists and Shock > 0:
                Gap = low[i - 2] - high[i]
                Deg = Gap / (3 * atr_arr[i] + eps)

                orx_num = 0.0
                orx_den = 0.0
                for jj in [i - 2, i - 1, i]:
                    bar_lo = min(open_[jj], close[jj])
                    orx_num += (bar_lo - low[jj]) ** 2
                    orx_den += (high[jj] - low[jj]) ** 2
                ORX = orx_num / (orx_den + eps)

                shock_history.append(Shock)
                deg_history.append(Deg)
                orx_history.append(ORX)

                if len(shock_history) >= 10:
                    shock_p = np.percentile(list(shock_history), shock_pctl)
                    deg_p = np.percentile(list(deg_history), deg_pctl)
                    orx_p = np.percentile(list(orx_history), ORX_PCTL)

                    qualifies = (
                        Shock > shock_p
                        and Deg < deg_p
                        and ORX > orx_p
                    )

                    if qualifies and len(active_gaps) < MAX_ACTIVE_GAPS:
                        active_gaps.append({
                            "gap_bar_idx": i,
                            "void_lo": high[i],
                            "void_hi": low[i - 2],
                            "gap_mid": (high[i] + low[i - 2]) / 2,
                            "Gap": Gap,
                            "Shock": Shock,
                            "Deg": Deg,
                            "ORX": ORX,
                            "state": "waiting_revisit",
                            "revisit_bar_idx": None,
                            "revisit_high": None,
                        })

        # --- Process active gap watches ---
        new_active_gaps = []
        for gap in active_gaps:
            if gap["state"] == "done":
                continue

            gap_bar_idx = gap["gap_bar_idx"]

            if bar_date[gap_bar_idx] != today:
                continue

            void_lo = gap["void_lo"]
            void_hi = gap["void_hi"]
            gap_mid = gap["gap_mid"]

            if gap["state"] == "waiting_revisit":
                if i > gap_bar_idx and high[i] >= void_lo:
                    bottom_q = low[i] + 0.25 * (high[i] - low[i])
                    valid_revisit = close[i] > bottom_q
                    if valid_revisit:
                        gap["state"] = "waiting_entry"
                        gap["revisit_bar_idx"] = i
                        gap["revisit_high"] = high[i]
                        new_active_gaps.append(gap)
                    else:
                        continue
                else:
                    new_active_gaps.append(gap)

            elif gap["state"] == "waiting_entry":
                revisit_idx = gap["revisit_bar_idx"]
                revisit_high = gap["revisit_high"]
                j_bar = revisit_idx

                if i == revisit_idx + 1:
                    if high[i] >= revisit_high:
                        can_enter = True
                        bar_t = dtime(int(bar_hour[i]), int(bar_minute[i]))

                        if bar_t >= NO_ENTRY_AFTER:
                            can_enter = False
                        if session_losses >= MAX_LOSING_TRADES_SESSION:
                            can_enter = False
                        if consec_losses >= PAUSE_AFTER_CONSEC_LOSSES:
                            if i < pause_until_bar:
                                can_enter = False
                            else:
                                pause_until_bar = -1
                                consec_losses = 0
                        if active_trade is not None:
                            can_enter = False

                        if can_enter:
                            entry_price = revisit_high + SLIP

                            sl_dist = min(
                                entry_price - (low[j_bar] - SLIP),
                                0.6 * gap["Gap"],
                            )
                            sl_dist = max(sl_dist, 2 * SLIP)
                            stop_price = entry_price - sl_dist

                            profit_target = gap_mid
                            if profit_target <= entry_price:
                                profit_target = void_hi + SLIP

                            if profit_target > entry_price:
                                t_stop_bar = i + time_stop_bars

                                active_trade = {
                                    "gap_bar_idx": gap_bar_idx,
                                    "revisit_bar": j_bar,
                                    "entry_bar": i,
                                    "entry_price": entry_price,
                                    "stop_price": stop_price,
                                    "profit_target": profit_target,
                                    "time_stop_bar": t_stop_bar,
                                    "Shock": gap["Shock"],
                                    "Deg": gap["Deg"],
                                    "ORX": gap["ORX"],
                                }

                    gap["state"] = "done"
                    continue
                else:
                    new_active_gaps.append(gap)

        active_gaps = new_active_gaps

        if consec_losses >= PAUSE_AFTER_CONSEC_LOSSES and pause_until_bar < i:
            pause_until_bar = i + pause_bars

        i += 1

    return trades


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def compute_stats(trades: list, n_rth_bars: int, n_days: int) -> dict:
    if not trades:
        return {
            "N": 0, "WR": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0,
            "max_dd": 0.0, "sharpe_trade": 0.0, "sharpe_daily": 0.0,
            "avg_trades_per_day": 0.0,
        }

    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls > 0
    N = len(pnls)
    WR = float(wins.mean())
    avg_pnl = float(pnls.mean())
    total_pnl = float(pnls.sum())

    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(dd.min())

    if pnls.std() > 0:
        sharpe_trade = float(
            (pnls.mean() / pnls.std()) * math.sqrt(252 * (N / n_days))
        )
    else:
        sharpe_trade = 0.0

    daily_pnl: dict = {}
    for t in trades:
        d = t["date"]
        daily_pnl[d] = daily_pnl.get(d, 0.0) + t["pnl"]
    daily_arr = np.array(list(daily_pnl.values()))
    if len(daily_arr) > 1 and daily_arr.std() > 0:
        sharpe_daily = float(
            (daily_arr.mean() / daily_arr.std()) * math.sqrt(252)
        )
    else:
        sharpe_daily = 0.0

    return {
        "N": int(N),
        "WR": round(WR, 4),
        "avg_pnl": round(avg_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "max_dd": round(max_dd, 2),
        "sharpe_trade": round(sharpe_trade, 3),
        "sharpe_daily": round(sharpe_daily, 3),
        "avg_trades_per_day": round(float(N / n_days), 3),
    }


def exit_breakdown(trades: list) -> list:
    reasons: dict = {}
    for t in trades:
        r = t["exit_reason"]
        if r not in reasons:
            reasons[r] = []
        reasons[r].append(t["pnl"])

    result = []
    for r, pnls in sorted(reasons.items()):
        arr = np.array(pnls)
        result.append({
            "reason": r,
            "N": int(len(arr)),
            "WR": round(float((arr > 0).mean()), 4),
            "avg_pnl": round(float(arr.mean()), 2),
        })
    return result


def monthly_pnl(trades: list) -> dict:
    monthly: dict = {}
    for t in trades:
        month = t["date"][:7]
        monthly[month] = monthly.get(month, 0.0) + t["pnl"]
    return {k: round(v, 2) for k, v in sorted(monthly.items())}


def monotonicity_test(trades: list) -> dict:
    if len(trades) < 10:
        return {
            "monotonicity_shock": [],
            "monotonicity_deg": [],
            "monotonicity_orx": [],
        }

    def decile_breakdown(field: str, invert: bool = False) -> list:
        values = np.array([t[field] for t in trades])
        pnls = np.array([t["pnl"] for t in trades])
        if invert:
            values = -values

        pctls = np.percentile(values, np.arange(0, 110, 10))
        result = []
        for d in range(10):
            lo = pctls[d]
            hi = pctls[d + 1]
            if d < 9:
                mask = (values >= lo) & (values < hi)
            else:
                mask = values >= lo
            subset = pnls[mask]
            if len(subset) == 0:
                result.append({
                    "decile": f"{d*10}-{(d+1)*10}",
                    "N": 0,
                    "WR": None,
                    "avg_pnl": None,
                })
            else:
                result.append({
                    "decile": f"{d*10}-{(d+1)*10}",
                    "N": int(len(subset)),
                    "WR": round(float((subset > 0).mean()), 4),
                    "avg_pnl": round(float(subset.mean()), 2),
                })
        return result

    return {
        "monotonicity_shock": decile_breakdown("Shock"),
        "monotonicity_deg": decile_breakdown("Deg", invert=True),
        "monotonicity_orx": decile_breakdown("ORX"),
    }


def threshold_sweep(
    df: pd.DataFrame,
    point_value: float,
    time_stop_bars: int,
    pause_bars: int,
    label: str,
    n_days: int,
) -> dict:
    shock_results = []
    for sp in range(60, 100, 10):
        t = run_vvr(
            df, point_value, time_stop_bars, pause_bars, label,
            shock_pctl=sp, deg_pctl=DEG_PCTL, collect_trades=True,
        )
        if not t:
            shock_results.append({
                "shock_pctl": sp, "N": 0,
                "WR": None, "avg_pnl": None, "sharpe": None,
            })
            continue
        pnls = np.array([x["pnl"] for x in t])
        daily_pnl: dict = {}
        for tr in t:
            d = tr["date"]
            daily_pnl[d] = daily_pnl.get(d, 0.0) + tr["pnl"]
        daily_arr = np.array(list(daily_pnl.values()))
        sharpe = (
            round(
                float((daily_arr.mean() / daily_arr.std()) * math.sqrt(252)), 3
            )
            if len(daily_arr) > 1 and daily_arr.std() > 0
            else 0.0
        )
        shock_results.append({
            "shock_pctl": sp,
            "N": int(len(pnls)),
            "WR": round(float((pnls > 0).mean()), 4),
            "avg_pnl": round(float(pnls.mean()), 2),
            "sharpe": sharpe,
        })

    deg_results = []
    for dp in range(30, 80, 10):
        t = run_vvr(
            df, point_value, time_stop_bars, pause_bars, label,
            shock_pctl=SHOCK_PCTL, deg_pctl=dp, collect_trades=True,
        )
        if not t:
            deg_results.append({
                "deg_pctl": dp, "N": 0,
                "WR": None, "avg_pnl": None, "sharpe": None,
            })
            continue
        pnls = np.array([x["pnl"] for x in t])
        daily_pnl = {}
        for tr in t:
            d = tr["date"]
            daily_pnl[d] = daily_pnl.get(d, 0.0) + tr["pnl"]
        daily_arr = np.array(list(daily_pnl.values()))
        sharpe = (
            round(
                float((daily_arr.mean() / daily_arr.std()) * math.sqrt(252)), 3
            )
            if len(daily_arr) > 1 and daily_arr.std() > 0
            else 0.0
        )
        deg_results.append({
            "deg_pctl": dp,
            "N": int(len(pnls)),
            "WR": round(float((pnls > 0).mean()), 4),
            "avg_pnl": round(float(pnls.mean()), 2),
            "sharpe": sharpe,
        })

    return {
        "threshold_sweep_shock": shock_results,
        "threshold_sweep_deg": deg_results,
    }


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------
def print_summary(results: dict):
    meta = results["meta"]
    ov = results["overall"]
    label = meta.get("label", meta.get("instrument", "?"))
    width = 64
    print("\n" + "=" * width)
    print(f"  VVR (Void Velocity Reversion) Backtest — {label}")
    print("=" * width)
    print(f"  Data          : {meta['data_range']}")
    print(
        f"  Days / Bars   : {meta['n_days']} RTH days, "
        f"{meta['n_rth_bars']:,} bars"
    )
    print(f"  Trades        : {ov['N']}")
    if ov["N"] == 0:
        print("  (no trades found)")
        print("=" * width)
        return
    print(f"  Win Rate      : {ov['WR']*100:.1f}%")
    print(f"  Avg PnL/trade : ${ov['avg_pnl']:.2f}")
    print(f"  Total PnL     : ${ov['total_pnl']:.2f}")
    print(f"  Max Drawdown  : ${ov['max_dd']:.2f}")
    print(f"  Sharpe(trade) : {ov['sharpe_trade']:.3f}")
    print(f"  Sharpe(daily) : {ov['sharpe_daily']:.3f}")
    print(f"  Avg trades/day: {ov['avg_trades_per_day']:.3f}")
    print()

    print("  Exit Breakdown:")
    print(f"  {'Reason':<8} {'N':>5} {'WR':>7} {'AvgPnL':>9}")
    print("  " + "-" * 33)
    for row in results["exit_breakdown"]:
        wr_str = f"{row['WR']*100:.1f}%" if row["WR"] is not None else "  N/A"
        avg_str = (
            f"${row['avg_pnl']:.2f}" if row["avg_pnl"] is not None else "  N/A"
        )
        print(f"  {row['reason']:<8} {row['N']:>5} {wr_str:>7} {avg_str:>9}")
    print()

    print("  Monthly PnL:")
    for month, pnl in results["monthly_pnl"].items():
        sign = "+" if pnl >= 0 else ""
        bar = "#" * max(0, int(abs(pnl) / 30))
        print(f"  {month}  {sign}${pnl:>8.2f}  {bar}")
    print()

    for display_label, key in [
        ("Shock Decile Monotonicity (higher Shock = better?)", "monotonicity_shock"),
        ("Deg Decile Monotonicity (lower Deg = better, inverted)", "monotonicity_deg"),
        ("ORX Decile Monotonicity (higher ORX = better?)", "monotonicity_orx"),
    ]:
        rows = results.get(key, [])
        if not rows:
            continue
        print(f"  {display_label}:")
        print(f"  {'Decile':<10} {'N':>5} {'WR':>7} {'AvgPnL':>9}")
        print("  " + "-" * 35)
        for row in rows:
            wr_str = (
                f"{row['WR']*100:.1f}%" if row["WR"] is not None else "  N/A"
            )
            avg_str = (
                f"${row['avg_pnl']:.2f}"
                if row["avg_pnl"] is not None
                else "  N/A"
            )
            print(
                f"  {row['decile']:<10} {row['N']:>5} {wr_str:>7} {avg_str:>9}"
            )
        print()

    print("  Threshold Sweep -- SHOCK_PCTL:")
    print(
        f"  {'SHOCK_PCTL':<12} {'N':>5} {'WR':>7} {'AvgPnL':>9} {'Sharpe':>8}"
    )
    print("  " + "-" * 45)
    for row in results["threshold_sweep_shock"]:
        wr_str = f"{row['WR']*100:.1f}%" if row["WR"] is not None else "  N/A"
        avg_str = (
            f"${row['avg_pnl']:.2f}" if row["avg_pnl"] is not None else "  N/A"
        )
        sh_str = (
            f"{row['sharpe']:.3f}" if row["sharpe"] is not None else "  N/A"
        )
        print(
            f"  {row['shock_pctl']:<12} {row['N']:>5} {wr_str:>7} "
            f"{avg_str:>9} {sh_str:>8}"
        )
    print()

    print("  Threshold Sweep -- DEG_PCTL:")
    print(
        f"  {'DEG_PCTL':<12} {'N':>5} {'WR':>7} {'AvgPnL':>9} {'Sharpe':>8}"
    )
    print("  " + "-" * 45)
    for row in results["threshold_sweep_deg"]:
        wr_str = f"{row['WR']*100:.1f}%" if row["WR"] is not None else "  N/A"
        avg_str = (
            f"${row['avg_pnl']:.2f}" if row["avg_pnl"] is not None else "  N/A"
        )
        sh_str = (
            f"{row['sharpe']:.3f}" if row["sharpe"] is not None else "  N/A"
        )
        print(
            f"  {row['deg_pctl']:<12} {row['N']:>5} {wr_str:>7} "
            f"{avg_str:>9} {sh_str:>8}"
        )
    print("=" * width)


# ---------------------------------------------------------------------------
# Build a full results dict for one dataset
# ---------------------------------------------------------------------------
def build_results(
    df: pd.DataFrame,
    point_value: float,
    time_stop_bars: int,
    pause_bars: int,
    label: str,
) -> dict:
    ts_s = df["timestamp"]
    n_rth_bars = len(df)
    dates = set(ts_s.dt.date)
    n_days = len(dates)
    data_range = f"{ts_s.dt.date.min()} to {ts_s.dt.date.max()}"

    print(f"  {n_rth_bars:,} RTH bars | {n_days} days | {data_range}")

    print(f"  Running base backtest [{label}]...")
    trades = run_vvr(
        df, point_value, time_stop_bars, pause_bars, label,
        shock_pctl=SHOCK_PCTL, deg_pctl=DEG_PCTL, collect_trades=True,
    )
    print(f"  {len(trades)} trades found")

    overall = compute_stats(trades, n_rth_bars, n_days)
    eb = exit_breakdown(trades)
    monthly = monthly_pnl(trades)
    mono = monotonicity_test(trades)

    print(f"  Running threshold sweeps [{label}]...")
    sweeps = threshold_sweep(
        df, point_value, time_stop_bars, pause_bars, label, n_days
    )

    return {
        "meta": {
            "label": label,
            "instrument": "MNQ",
            "data_range": data_range,
            "n_days": n_days,
            "n_rth_bars": n_rth_bars,
            "point_value": point_value,
            "time_stop_bars": time_stop_bars,
            "pause_bars": pause_bars,
        },
        "overall": overall,
        "exit_breakdown": eb,
        "monthly_pnl": monthly,
        "monotonicity_shock": mono.get("monotonicity_shock", []),
        "monotonicity_deg": mono.get("monotonicity_deg", []),
        "monotonicity_orx": mono.get("monotonicity_orx", []),
        "threshold_sweep_shock": sweeps["threshold_sweep_shock"],
        "threshold_sweep_deg": sweeps["threshold_sweep_deg"],
        "trade_log": trades,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- MNQ 1-min ---
    print("\nLoading MNQ 1-min RTH bars...")
    df_1min = load_mnq_1min()
    results_1min = build_results(
        df_1min,
        point_value=2.0,
        time_stop_bars=10,
        pause_bars=30,
        label="MNQ 1-min",
    )

    # --- MNQ 5-min ---
    print("\nLoading MNQ 5-min RTH bars...")
    df_5min = load_mnq_5min()
    results_5min = build_results(
        df_5min,
        point_value=2.0,
        time_stop_bars=3,
        pause_bars=6,
        label="MNQ 5-min",
    )

    # --- Save ---
    combined = {
        "mnq_1min": {k: v for k, v in results_1min.items() if k != "trade_log"},
        "mnq_5min": {k: v for k, v in results_5min.items() if k != "trade_log"},
        "trade_log_1min": results_1min["trade_log"],
        "trade_log_5min": results_5min["trade_log"],
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nResults written to {RESULTS_PATH}")

    # --- Print both summaries ---
    print_summary(results_1min)
    print_summary(results_5min)


if __name__ == "__main__":
    main()
