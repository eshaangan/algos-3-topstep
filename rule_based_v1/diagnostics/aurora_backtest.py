"""AURORA (Auction Regime Open Range Algorithm) Backtest — MES 1-min bars.

Strategy: classify each day into CONTINUATION or REPAIR branch based on
overnight compression, gap size, and opening drive efficiency, then trade
accordingly.

Instrument: MES, $5/pt, 1 contract, 0.25 pt slippage per side.
Data: data/processed/mes_1m_bars_cache.h5, key /bars_1m
      2025-06-30 to 2025-12-26 ET

Run:
    python rule_based_v1/diagnostics/aurora_backtest.py
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
RESULTS_PATH = DIAG_DIR / "aurora_results.json"

for p in [str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POINT_VALUE = 5.0    # $/pt MES
SLIP = 0.25          # slippage per side (pts)
N_CONTRACTS = 1

# Absolute thresholds
ABS_C_CONT = 0.80
ABS_G_CONT = 0.40
ABS_E_CONT = 0.65
ABS_G_REPAIR = 0.60
ABS_C_REPAIR = 1.25

# Percentile thresholds
PCT_C_CONT = 40      # c_pct < 40
PCT_G_CONT = 45      # abs_g_pct < 45
PCT_E_CONT = 55      # e_pct > 55
PCT_G_REPAIR = 65    # abs_g_pct > 65
PCT_C_REPAIR = 70    # c_pct > 70


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


def is_rth_minute(hour: int, minute: int) -> bool:
    m = hour * 60 + minute
    return 570 <= m <= 959


# ---------------------------------------------------------------------------
# Build per-day feature structures
# ---------------------------------------------------------------------------

def build_daily_features(df: pd.DataFrame) -> list[dict]:
    """Return list of per-day feature dicts, sorted by date."""
    ET = df.index.tzinfo

    # RTH mask
    minutes = df.index.hour * 60 + df.index.minute
    rth_mask = (minutes >= 570) & (minutes <= 959)
    rth = df[rth_mask]

    rth_dates = sorted(pd.DatetimeIndex(rth.index.normalize()).unique())

    # Build per-date RTH lookup
    rth_by_date: dict = {}
    for d in rth_dates:
        day_rth = rth[rth.index.normalize() == d]
        if len(day_rth) > 0:
            rth_by_date[d] = day_rth

    date_list = sorted(rth_by_date.keys())

    # Prior RTH daily ranges for ATR
    ranges = {}
    for d in date_list:
        dr = rth_by_date[d]
        ranges[d] = dr["high"].max() - dr["low"].min()
    range_series = pd.Series(ranges)

    days_out = []

    for i, d in enumerate(date_list):
        if i < 1:
            continue

        prev_d = date_list[i - 1]
        if prev_d not in rth_by_date:
            continue

        # --- Prior RTH stats ---
        prev_rth = rth_by_date[prev_d]
        C_y = float(prev_rth["close"].iloc[-1])
        H_y = float(prev_rth["high"].max())
        L_y = float(prev_rth["low"].min())
        clv_y = (C_y - L_y) / (H_y - L_y + 1e-9)

        # 20-day rolling mean of prior ranges (using index up to prev_d inclusive)
        idx_prev = range_series.index.get_loc(prev_d)
        window_start = max(0, idx_prev - 19)
        atr_window = range_series.iloc[window_start: idx_prev + 1]
        if len(atr_window) < 1:
            continue
        ATR_d = float(atr_window.mean())
        if ATR_d <= 0:
            continue

        # --- ETH overnight ---
        eth_start = prev_d.replace(hour=16, minute=0, second=0, microsecond=0)
        eth_end = d.replace(hour=9, minute=29, second=59, microsecond=0)
        eth_bars = df[(df.index >= eth_start) & (df.index <= eth_end)]
        if len(eth_bars) == 0:
            continue

        H_eth = float(eth_bars["high"].max())
        L_eth = float(eth_bars["low"].min())
        eth_range = H_eth - L_eth

        # --- Opening Range (OR): 9:30–9:44 ---
        cur_rth = rth_by_date[d]
        or_mask = (cur_rth.index.hour == 9) & (cur_rth.index.minute >= 30) & (cur_rth.index.minute <= 44)
        or_bars = cur_rth[or_mask]
        if len(or_bars) < 15:
            # Allow slightly fewer bars (e.g. holiday truncation) — need at least 5
            if len(or_bars) < 5:
                continue

        O_open = float(or_bars["open"].iloc[0])
        H_or = float(or_bars["high"].max())
        L_or = float(or_bars["low"].min())
        C_or = float(or_bars["close"].iloc[-1])
        or_range = H_or - L_or
        if or_range <= 0:
            continue

        # Gap
        g = (O_open - C_y) / ATR_d

        # Drive efficiency
        e = abs(C_or - O_open) / (or_range + 1e-9)

        # Drive direction
        drive_dir = 1 if C_or > O_open else -1

        # Micro-pivot: last 5 bars of OR (9:40–9:44)
        micro_mask = (cur_rth.index.hour == 9) & (cur_rth.index.minute >= 40) & (cur_rth.index.minute <= 44)
        micro_bars = cur_rth[micro_mask]
        if len(micro_bars) == 0:
            micro_high = H_or
            micro_low = L_or
        else:
            micro_high = float(micro_bars["high"].max())
            micro_low = float(micro_bars["low"].min())

        days_out.append({
            "date": d,
            "C_y": C_y,
            "H_y": H_y,
            "L_y": L_y,
            "clv_y": clv_y,
            "ATR_d": ATR_d,
            "H_eth": H_eth,
            "L_eth": L_eth,
            "eth_range": eth_range,
            "O_open": O_open,
            "H_or": H_or,
            "L_or": L_or,
            "C_or": C_or,
            "or_range": or_range,
            "g": g,
            "e": e,
            "drive_dir": drive_dir,
            "micro_high": micro_high,
            "micro_low": micro_low,
            "cur_rth": cur_rth,
        })

    return days_out


# ---------------------------------------------------------------------------
# Rolling percentile calibration
# ---------------------------------------------------------------------------

def percentile_rank(value: float, history: list[float]) -> float:
    """Fraction of history values strictly less than value, × 100."""
    if len(history) == 0:
        return 50.0
    arr = np.array(history)
    return float(np.mean(arr < value) * 100.0)


def add_rolling_percentiles(days: list[dict], window: int = 60) -> list[dict]:
    """Attach c_pct, abs_g_pct, e_pct to each day using prior `window` days."""
    c_vals: list[float] = []
    absg_vals: list[float] = []
    e_vals: list[float] = []

    enriched = []
    for day in days:
        eth_range_median = np.median(c_vals[-window:]) if c_vals else day["eth_range"]
        c = day["eth_range"] / (eth_range_median + 1e-9)

        # Rolling medians for c
        prior_c = c_vals[-window:]
        prior_absg = absg_vals[-window:]
        prior_e = e_vals[-window:]

        c_pct = percentile_rank(c, prior_c)
        abs_g_pct = percentile_rank(abs(day["g"]), prior_absg)
        e_pct = percentile_rank(day["e"], prior_e)

        c_vals.append(c)
        absg_vals.append(abs(day["g"]))
        e_vals.append(day["e"])

        enriched.append({
            **day,
            "c": c,
            "c_pct": c_pct,
            "abs_g_pct": abs_g_pct,
            "e_pct": e_pct,
        })

    return enriched


# ---------------------------------------------------------------------------
# Branch classification
# ---------------------------------------------------------------------------

def classify_branch_absolute(day: dict) -> Optional[str]:
    """Return 'CONTINUATION', 'REPAIR', or None."""
    c = day["c"]
    g = day["g"]
    e = day["e"]
    H_or = day["H_or"]
    L_or = day["L_or"]
    H_eth = day["H_eth"]
    L_eth = day["L_eth"]
    C_or = day["C_or"]
    or_range = day["or_range"]
    drive_dir = day["drive_dir"]

    # Check REPAIR first (priority)
    repair_gap_cond = (abs(g) > ABS_G_REPAIR) or (c > ABS_C_REPAIR)
    repair_inside = (L_eth <= C_or <= H_eth)
    if g > 0:
        wick_cond = (H_or - max(day["O_open"], C_or)) >= 0.45 * or_range
    else:
        wick_cond = (min(day["O_open"], C_or) - L_or) >= 0.45 * or_range
    if repair_gap_cond and repair_inside and wick_cond:
        return "REPAIR"

    # Check CONTINUATION
    cont_quiet = c < ABS_C_CONT
    cont_gap = abs(g) < ABS_G_CONT
    cont_eff = e > ABS_E_CONT
    if drive_dir == 1:
        cont_outside = C_or > H_eth + 0.10 * or_range
    else:
        cont_outside = C_or < L_eth - 0.10 * or_range
    if cont_quiet and cont_gap and cont_eff and cont_outside:
        return "CONTINUATION"

    return None


def classify_branch_percentile(day: dict) -> Optional[str]:
    """Return 'CONTINUATION', 'REPAIR', or None using percentile thresholds."""
    c_pct = day["c_pct"]
    abs_g_pct = day["abs_g_pct"]
    e_pct = day["e_pct"]
    c = day["c"]
    g = day["g"]
    H_or = day["H_or"]
    L_or = day["L_or"]
    H_eth = day["H_eth"]
    L_eth = day["L_eth"]
    C_or = day["C_or"]
    or_range = day["or_range"]
    drive_dir = day["drive_dir"]

    # REPAIR first (priority)
    repair_gap_cond = (abs_g_pct > PCT_G_REPAIR) or (c_pct > PCT_C_REPAIR)
    repair_inside = (L_eth <= C_or <= H_eth)
    if g > 0:
        wick_cond = (H_or - max(day["O_open"], C_or)) >= 0.45 * or_range
    else:
        wick_cond = (min(day["O_open"], C_or) - L_or) >= 0.45 * or_range
    if repair_gap_cond and repair_inside and wick_cond:
        return "REPAIR"

    # CONTINUATION
    cont_quiet = c_pct < PCT_C_CONT
    cont_gap = abs_g_pct < PCT_G_CONT
    cont_eff = e_pct > PCT_E_CONT
    if drive_dir == 1:
        cont_outside = C_or > H_eth + 0.10 * or_range
    else:
        cont_outside = C_or < L_eth - 0.10 * or_range
    if cont_quiet and cont_gap and cont_eff and cont_outside:
        return "CONTINUATION"

    return None


def classify_branch_forced(day: dict, force: str) -> Optional[str]:
    """Force all days into 'CONTINUATION' or 'REPAIR' for baseline comparison."""
    return force


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

def simulate_continuation(day: dict) -> Optional[dict]:
    """Scan 9:45–10:10 for pullback entry. Return trade dict or None."""
    cur_rth = day["cur_rth"]
    H_or = day["H_or"]
    L_or = day["L_or"]
    or_range = day["or_range"]
    ATR_d = day["ATR_d"]
    drive_dir = day["drive_dir"]
    d = day["date"]
    ET = cur_rth.index.tzinfo

    # Scan window
    scan_start = d.replace(hour=9, minute=45, second=0, microsecond=0)
    scan_end = d.replace(hour=10, minute=10, second=59, microsecond=0)
    scan_bars = cur_rth[(cur_rth.index >= scan_start) & (cur_rth.index <= scan_end)]

    entry_price = None
    entry_time = None

    for ts, bar in scan_bars.iterrows():
        if drive_dir == 1:
            # LONG: bar low touches top of OR (H_or)
            if bar["low"] <= H_or:
                entry_price = H_or + SLIP
                entry_time = ts
                break
        else:
            # SHORT: bar high touches bottom of OR (L_or)
            if bar["high"] >= L_or:
                entry_price = L_or - SLIP
                entry_time = ts
                break

    if entry_price is None:
        return None

    # Stop and targets
    raw_stop = 0.75 * or_range
    capped_stop = min(raw_stop, 2.0 * ATR_d)
    stop_dist = capped_stop

    if drive_dir == 1:
        sl_price = entry_price - stop_dist
        tp1_price = entry_price + 1.2 * stop_dist
        tp2_price = entry_price + 1.8 * stop_dist
    else:
        sl_price = entry_price + stop_dist
        tp1_price = entry_price - 1.2 * stop_dist
        tp2_price = entry_price - 1.8 * stop_dist

    # Exit scan: bars after entry up to 13:00
    flat_time = d.replace(hour=13, minute=0, second=0, microsecond=0)
    exit_bars = cur_rth[(cur_rth.index > entry_time) & (cur_rth.index <= flat_time)]

    exit_price, exit_reason = _scan_exits_tp1(exit_bars, entry_price, drive_dir,
                                               sl_price, tp1_price, flat_time)

    pnl_pts = (exit_price - entry_price) * drive_dir - 2 * SLIP
    pnl_usd = pnl_pts * POINT_VALUE * N_CONTRACTS

    return {
        "date": str(d.date()),
        "branch": "CONTINUATION",
        "direction": "LONG" if drive_dir == 1 else "SHORT",
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "sl_price": round(sl_price, 4),
        "tp1_price": round(tp1_price, 4),
        "stop_dist": round(stop_dist, 4),
        "pnl_pts": round(pnl_pts, 4),
        "pnl_usd": round(pnl_usd, 2),
        "exit_reason": exit_reason,
        "c": round(day.get("c", float("nan")), 4),
        "g": round(day["g"], 4),
        "e": round(day["e"], 4),
        "c_pct": round(day.get("c_pct", float("nan")), 1),
        "abs_g_pct": round(day.get("abs_g_pct", float("nan")), 1),
        "e_pct": round(day.get("e_pct", float("nan")), 1),
    }


def simulate_repair(day: dict) -> Optional[dict]:
    """Scan 9:45–10:00 for micro-pivot break entry. Return trade dict or None."""
    cur_rth = day["cur_rth"]
    micro_high = day["micro_high"]
    micro_low = day["micro_low"]
    H_or = day["H_or"]
    L_or = day["L_or"]
    C_y = day["C_y"]
    g = day["g"]
    d = day["date"]

    # Repair direction: opposite of gap
    repair_dir = -1 if g > 0 else 1

    # Scan window: 9:45–10:00
    scan_start = d.replace(hour=9, minute=45, second=0, microsecond=0)
    scan_end = d.replace(hour=10, minute=0, second=59, microsecond=0)
    scan_bars = cur_rth[(cur_rth.index >= scan_start) & (cur_rth.index <= scan_end)]

    entry_price = None
    entry_time = None

    for ts, bar in scan_bars.iterrows():
        if repair_dir == 1:
            # LONG repair (gap was down): high breaks micro_high
            if bar["high"] > micro_high:
                entry_price = micro_high + SLIP
                entry_time = ts
                break
        else:
            # SHORT repair (gap was up): low breaks micro_low
            if bar["low"] < micro_low:
                entry_price = micro_low - SLIP
                entry_time = ts
                break

    if entry_price is None:
        return None

    # Stop: session extreme + 2 ticks (0.50 pts)
    TICK2 = 0.50
    if repair_dir == 1:
        sl_price = L_or - TICK2
    else:
        sl_price = H_or + TICK2

    stop_dist = abs(entry_price - sl_price)
    if stop_dist <= 0:
        stop_dist = 1.0

    # TP: prior close OR 1.5 × stop_dist, whichever is closer to entry
    if repair_dir == 1:
        tp_close = C_y
        tp_rr = entry_price + 1.5 * stop_dist
        tp_price = tp_close if (tp_close - entry_price) < (tp_rr - entry_price) and tp_close > entry_price else tp_rr
    else:
        tp_close = C_y
        tp_rr = entry_price - 1.5 * stop_dist
        tp_price = tp_close if (entry_price - tp_close) < (entry_price - tp_rr) and tp_close < entry_price else tp_rr

    # Exit scan up to 13:00
    flat_time = d.replace(hour=13, minute=0, second=0, microsecond=0)
    exit_bars = cur_rth[(cur_rth.index > entry_time) & (cur_rth.index <= flat_time)]

    exit_price, exit_reason = _scan_exits_tp1(exit_bars, entry_price, repair_dir,
                                               sl_price, tp_price, flat_time)

    pnl_pts = (exit_price - entry_price) * repair_dir - 2 * SLIP
    pnl_usd = pnl_pts * POINT_VALUE * N_CONTRACTS

    return {
        "date": str(d.date()),
        "branch": "REPAIR",
        "direction": "LONG" if repair_dir == 1 else "SHORT",
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "sl_price": round(sl_price, 4),
        "tp1_price": round(tp_price, 4),
        "stop_dist": round(stop_dist, 4),
        "pnl_pts": round(pnl_pts, 4),
        "pnl_usd": round(pnl_usd, 2),
        "exit_reason": exit_reason,
        "c": round(day.get("c", float("nan")), 4),
        "g": round(day["g"], 4),
        "e": round(day["e"], 4),
        "c_pct": round(day.get("c_pct", float("nan")), 1),
        "abs_g_pct": round(day.get("abs_g_pct", float("nan")), 1),
        "e_pct": round(day.get("e_pct", float("nan")), 1),
    }


def _scan_exits_tp1(
    bars: pd.DataFrame,
    entry: float,
    direction: int,
    sl: float,
    tp: float,
    flat_time: pd.Timestamp,
) -> tuple[float, str]:
    """Check bar-by-bar for TP/SL hit. TP wins on same bar. Flat at end."""
    for ts, bar in bars.iterrows():
        if direction == 1:
            tp_hit = bar["high"] >= tp
            sl_hit = bar["low"] <= sl
        else:
            tp_hit = bar["low"] <= tp
            sl_hit = bar["high"] >= sl

        if tp_hit and sl_hit:
            return tp, "TP"  # optimistic: TP wins same bar
        if tp_hit:
            return tp, "TP"
        if sl_hit:
            return sl, "SL"

    # Time exit: return close of last bar before/at flat_time
    bars_to_flat = bars[bars.index <= flat_time]
    if len(bars_to_flat) > 0:
        return float(bars_to_flat["close"].iloc[-1]), "FLAT"
    return entry, "FLAT_NO_BARS"


# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------

def run_backtest(
    days: list[dict],
    classify_fn,
    min_prior_days: int = 20,
    require_pct: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Run full backtest. Returns (trades, day_summaries)."""
    trades: list[dict] = []
    day_summaries: list[dict] = []

    for i, day in enumerate(days):
        if i < min_prior_days:
            continue
        # Require at least 60 prior days for percentile branch
        if require_pct and i < 60:
            continue

        branch = classify_fn(day)
        trade = None

        if branch == "CONTINUATION":
            trade = simulate_continuation(day)
        elif branch == "REPAIR":
            trade = simulate_repair(day)

        if trade is not None:
            trade["threshold_mode"] = "percentile" if require_pct else "absolute"
            trades.append(trade)

        day_summaries.append({
            "date": str(day["date"].date()),
            "branch": branch,
            "traded": trade is not None,
        })

    return trades, day_summaries


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[dict], label: str = "") -> dict:
    if not trades:
        return {
            "N": 0, "WR": 0.0, "AvgPnL": 0.0, "TotalPnL": 0.0,
            "Sharpe": 0.0, "MaxDD": 0.0,
        }

    pnls = [t["pnl_usd"] for t in trades]
    dates = [t["date"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    n = len(pnls)
    wr = wins / n * 100
    avg_pnl = float(np.mean(pnls))
    total_pnl = float(np.sum(pnls))

    # Daily PnL for Sharpe (sum per date)
    daily: dict[str, float] = {}
    for t in trades:
        daily[t["date"]] = daily.get(t["date"], 0.0) + t["pnl_usd"]
    daily_pnls = np.array(list(daily.values()))
    if len(daily_pnls) > 1 and np.std(daily_pnls) > 0:
        sharpe = float(np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Max drawdown on cumulative equity
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(dd.min())

    return {
        "N": n,
        "WR": round(wr, 1),
        "AvgPnL": round(avg_pnl, 2),
        "TotalPnL": round(total_pnl, 2),
        "Sharpe": round(sharpe, 3),
        "MaxDD": round(max_dd, 2),
    }


# ---------------------------------------------------------------------------
# Monotonicity validation
# ---------------------------------------------------------------------------

def monotonicity_continuation(trades: list[dict]) -> list[dict]:
    cont_trades = [t for t in trades if t["branch"] == "CONTINUATION"]
    results = []

    # e_pct buckets
    e_buckets = [(0, 33, "e_pct 0-33"), (33, 66, "e_pct 33-66"), (66, 100, "e_pct 66-100")]
    for lo, hi, label in e_buckets:
        sub = [t for t in cont_trades if lo <= t["e_pct"] < hi or (hi == 100 and t["e_pct"] == 100)]
        m = compute_metrics(sub)
        results.append({"bucket": label, "dimension": "e_pct", **m})

    # c_pct buckets
    c_buckets = [(0, 33, "c_pct 0-33"), (33, 66, "c_pct 33-66")]
    for lo, hi, label in c_buckets:
        sub = [t for t in cont_trades if lo <= t["c_pct"] < hi]
        m = compute_metrics(sub)
        results.append({"bucket": label, "dimension": "c_pct", **m})

    return results


def monotonicity_repair(trades: list[dict]) -> list[dict]:
    repair_trades = [t for t in trades if t["branch"] == "REPAIR"]
    results = []

    absg_buckets = [
        (50, 65, "abs_g_pct 50-65"),
        (65, 80, "abs_g_pct 65-80"),
        (80, 100, "abs_g_pct 80-100"),
    ]
    for lo, hi, label in absg_buckets:
        sub = [t for t in repair_trades if lo <= t["abs_g_pct"] < hi or (hi == 100 and t["abs_g_pct"] == 100)]
        m = compute_metrics(sub)
        results.append({"bucket": label, "dimension": "abs_g_pct", **m})

    c_buckets = [
        (65, 80, "c_pct 65-80"),
        (80, 100, "c_pct 80-100"),
    ]
    for lo, hi, label in c_buckets:
        sub = [t for t in repair_trades if lo <= t["c_pct"] < hi or (hi == 100 and t["c_pct"] == 100)]
        m = compute_metrics(sub)
        results.append({"bucket": label, "dimension": "c_pct", **m})

    return results


# ---------------------------------------------------------------------------
# Classifier value-add
# ---------------------------------------------------------------------------

def classifier_value_add(days: list[dict], min_prior: int = 60) -> dict:
    """Compare combined vs always-cont vs always-repair using percentile branch."""
    eligible = [d for i, d in enumerate(days) if i >= min_prior]

    combined_trades = []
    cont_only_trades = []
    repair_only_trades = []

    for day in eligible:
        branch = classify_branch_percentile(day)

        if branch == "CONTINUATION":
            t = simulate_continuation(day)
            if t:
                t["branch"] = "CONTINUATION"
                combined_trades.append(t)
        elif branch == "REPAIR":
            t = simulate_repair(day)
            if t:
                t["branch"] = "REPAIR"
                combined_trades.append(t)

        # Always continuation
        t_cont = simulate_continuation(day)
        if t_cont:
            t_cont["branch"] = "CONTINUATION"
            cont_only_trades.append(t_cont)

        # Always repair
        t_repair = simulate_repair(day)
        if t_repair:
            t_repair["branch"] = "REPAIR"
            repair_only_trades.append(t_repair)

    return {
        "combined": compute_metrics(combined_trades),
        "always_continuation": compute_metrics(cont_only_trades),
        "always_repair": compute_metrics(repair_only_trades),
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_table(title: str, metrics: dict):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        if k not in ("N",):
            print(f"  {k:<20} {v}")
        else:
            print(f"  {k:<20} {v}")


def print_bucket_table(title: str, rows: list[dict]):
    print(f"\n{'-'*75}")
    print(f"  {title}")
    print(f"{'Bucket':<25} {'N':>5} {'WR%':>7} {'AvgPnL':>9} {'TotalPnL':>11} {'Sharpe':>8} {'MaxDD':>10}")
    print(f"{'-'*75}")
    for r in rows:
        print(
            f"  {r['bucket']:<23} {r['N']:>5} {r['WR']:>7.1f} "
            f"{r['AvgPnL']:>9.2f} {r['TotalPnL']:>11.2f} "
            f"{r['Sharpe']:>8.3f} {r['MaxDD']:>10.2f}"
        )


def print_branch_breakdown(trades: list[dict]):
    print(f"\n{'='*55}")
    print("  Branch Breakdown")
    print(f"{'='*55}")
    for branch in ["CONTINUATION", "REPAIR"]:
        sub = [t for t in trades if t["branch"] == branch]
        m = compute_metrics(sub)
        print(f"\n  [{branch}]")
        for k, v in m.items():
            print(f"    {k:<18} {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading MES 1-min bars...")
    df = load_bars()
    print(f"  Total bars: {len(df):,}  |  Range: {df.index[0]} → {df.index[-1]}")

    print("\nBuilding per-day features...")
    days_raw = build_daily_features(df)
    print(f"  Days with full OR data: {len(days_raw)}")

    print("Adding rolling percentile calibration (60-day window)...")
    days = add_rolling_percentiles(days_raw, window=60)
    days_in_dataset = len(days)
    print(f"  Total feature-enriched days: {days_in_dataset}")

    # -----------------------------------------------------------------------
    # 1. ABSOLUTE THRESHOLDS
    # -----------------------------------------------------------------------
    print("\n" + "="*55)
    print("  [1] ABSOLUTE THRESHOLD BACKTEST")
    print("="*55)

    trades_abs, _ = run_backtest(days, classify_branch_absolute, min_prior_days=20, require_pct=False)
    metrics_abs = compute_metrics(trades_abs, "Absolute")
    trades_per_day_abs = len(trades_abs) / max(days_in_dataset - 20, 1)

    print_table("Absolute Thresholds", metrics_abs)
    print(f"  {'Trades/day':<20} {trades_per_day_abs:.3f}")

    # -----------------------------------------------------------------------
    # 2. PERCENTILE THRESHOLDS
    # -----------------------------------------------------------------------
    print("\n" + "="*55)
    print("  [2] PERCENTILE THRESHOLD BACKTEST")
    print("="*55)

    trades_pct, _ = run_backtest(days, classify_branch_percentile, min_prior_days=60, require_pct=True)
    metrics_pct = compute_metrics(trades_pct, "Percentile")
    trades_per_day_pct = len(trades_pct) / max(days_in_dataset - 60, 1)

    print_table("Percentile Thresholds", metrics_pct)
    print(f"  {'Trades/day':<20} {trades_per_day_pct:.3f}")

    # -----------------------------------------------------------------------
    # 3. Branch breakdown (use percentile trades)
    # -----------------------------------------------------------------------
    print_branch_breakdown(trades_pct)

    branch_breakdown = {
        "continuation": compute_metrics([t for t in trades_pct if t["branch"] == "CONTINUATION"]),
        "repair": compute_metrics([t for t in trades_pct if t["branch"] == "REPAIR"]),
    }

    # -----------------------------------------------------------------------
    # 4. Monotonicity validation
    # -----------------------------------------------------------------------
    print(f"\n{'='*55}")
    print("  [3] MONOTONICITY VALIDATION")

    mono_cont = monotonicity_continuation(trades_pct)
    print_bucket_table("Continuation: e_pct & c_pct buckets", mono_cont)

    mono_rep = monotonicity_repair(trades_pct)
    print_bucket_table("Repair: abs_g_pct & c_pct buckets", mono_rep)

    # -----------------------------------------------------------------------
    # 5. Classifier value-add
    # -----------------------------------------------------------------------
    print(f"\n{'='*55}")
    print("  [4] CLASSIFIER VALUE-ADD")
    print(f"{'='*55}")

    cva = classifier_value_add(days, min_prior=60)
    for label, m in cva.items():
        print(f"\n  [{label}]")
        for k, v in m.items():
            print(f"    {k:<18} {v}")

    # -----------------------------------------------------------------------
    # All trades combined for JSON (use both absolute and percentile)
    # -----------------------------------------------------------------------
    all_trades_json = trades_abs + trades_pct  # include both runs

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "absolute_thresholds": metrics_abs,
        "percentile_thresholds": metrics_pct,
        "branch_breakdown": branch_breakdown,
        "monotonicity_continuation": mono_cont,
        "monotonicity_repair": mono_rep,
        "classifier_value_add": cva,
        "all_trades": all_trades_json,
        "days_in_dataset": days_in_dataset,
    }

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {RESULTS_PATH}")
    print(f"Total trades (abs): {metrics_abs['N']}  |  Total trades (pct): {metrics_pct['N']}")
    print(f"TotalPnL (abs): ${metrics_abs['TotalPnL']:.2f}  |  TotalPnL (pct): ${metrics_pct['TotalPnL']:.2f}")


if __name__ == "__main__":
    main()
