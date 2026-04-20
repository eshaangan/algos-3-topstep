"""NOVA (Nighttime Order-imbalance Value Arbitrage) Backtest — MES 1-min bars.

Strategy: trade MES futures in the overnight session based on end-of-day
close imbalances. Computes a Session Close Imbalance (SCI) score from the
RTH close window (15:30-15:59 ET), then looks for price repair in the
02:00-02:04 ET window the following night.

Instruments: MES, $5/pt, 1 contract, 0.25 pt slippage per side.
Data:
    Primary  (Jun-Dec 2025): data/processed/mes_1m_bars_cache.h5, /bars_1m
    Secondary (Jan-Feb 2026): data/processed/jan_feb_2026_oos_test_1m.h5, /bars_1min

Run:
    python rule_based_v1/diagnostics/nova_backtest.py
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
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
PRIMARY_PATH = DATA_DIR / "mes_1m_bars_cache.h5"
SECONDARY_PATH = DATA_DIR / "jan_feb_2026_oos_test_1m.h5"
RESULTS_PATH = DIAG_DIR / "nova_results.json"

for p in [str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POINT_VALUE = 5.0        # MES $/pt
SLIP = 0.25              # slippage per side (pts)
N_CONTRACTS = 1
SCI_PCTL_WINDOW = 30
SCI_MIN_HISTORY = 10
REPAIR_LO = 0.05
REPAIR_HI = 0.45
TP_MULT = 0.6
SL_MULT = 0.45
HARD_EXIT_HOUR = 3
HARD_EXIT_MIN = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sign(x: float) -> int:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def agg_1m_to_5m(bars_1m: pd.DataFrame) -> dict:
    """Aggregate a slice of 1-min bars into a single pseudo-5-min bar dict."""
    if len(bars_1m) == 0:
        return {}
    return {
        "open": float(bars_1m["open"].iloc[0]),
        "high": float(bars_1m["high"].max()),
        "low": float(bars_1m["low"].min()),
        "close": float(bars_1m["close"].iloc[-1]),
        "volume": float(bars_1m["volume"].sum()),
        "n_bars": len(bars_1m),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bars() -> pd.DataFrame:
    """Load and combine primary + secondary 1-min bars, tz-converted to ET."""
    # --- Primary ---
    with pd.HDFStore(str(PRIMARY_PATH), mode="r") as store:
        df1 = store["/bars_1m"].copy()
    # timestamp is a column with int64 index; set as index
    df1 = df1.set_index("timestamp")
    df1.index = df1.index.tz_convert("US/Eastern")

    # --- Secondary ---
    with pd.HDFStore(str(SECONDARY_PATH), mode="r") as store:
        df2 = store["/bars_1min"].copy()
    # Index already tz-aware
    df2.index = df2.index.tz_convert("US/Eastern")

    # --- Combine ---
    combined = pd.concat([df1, df2])
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()
    return combined


# ---------------------------------------------------------------------------
# Step 1 — Per-day SCI computation
# ---------------------------------------------------------------------------

def compute_rth_daily_ranges(df: pd.DataFrame) -> pd.Series:
    """Return Series of daily RTH range (H-L) indexed by date (tz-naive)."""
    minutes = df.index.hour * 60 + df.index.minute
    rth_mask = (minutes >= 9 * 60 + 30) & (minutes <= 15 * 60 + 59)
    rth = df[rth_mask]
    if len(rth) == 0:
        return pd.Series(dtype=float)
    rth_dates = rth.index.normalize().unique()
    ranges = {}
    for d in rth_dates:
        day_bars = rth[rth.index.normalize() == d]
        if len(day_bars) > 0:
            ranges[d.date()] = float(day_bars["high"].max() - day_bars["low"].min())
    return pd.Series(ranges)


def compute_sci_for_day(
    day_rth: pd.DataFrame,
    atr_d: float,
) -> Optional[dict]:
    """
    Compute SCI from RTH bars of one day.
    Returns dict with sci, sci_dir, C_end, C_start, close_range or None.
    """
    if atr_d <= 0:
        return None

    # RTH extremes
    H_RTH = float(day_rth["high"].max())
    L_RTH = float(day_rth["low"].min())

    # Close window: 15:30-15:59 ET
    cw_mask = (day_rth.index.hour == 15) & (day_rth.index.minute >= 30) & (day_rth.index.minute <= 59)
    cw_bars = day_rth[cw_mask]

    if len(cw_bars) < 10:
        return None

    C_start = float(cw_bars["close"].iloc[0])
    C_end = float(cw_bars["close"].iloc[-1])
    O_start = float(cw_bars["open"].iloc[0])
    H_win = float(cw_bars["high"].max())
    L_win = float(cw_bars["low"].min())

    r_c = (C_end - C_start) / atr_d
    if r_c == 0:
        return None

    sd = sign(r_c)

    e_c = abs(C_end - O_start) / (H_win - L_win + 1e-9)

    # x_c: per-bar signed CLV weighted by direction
    clv_vals = []
    for _, bar in cw_bars.iterrows():
        bar_range = bar["high"] - bar["low"]
        clv = (2 * (bar["close"] - bar["low"]) / (bar_range + 1e-9)) - 1
        clv_vals.append(sd * clv)
    x_c = float(np.mean(clv_vals))

    # d_c: where did close end relative to full RTH range
    d_c = sd * (2 * (C_end - L_RTH) / (H_RTH - L_RTH + 1e-9) - 1)

    sci = 0.35 * abs(r_c) + 0.25 * e_c + 0.20 * x_c + 0.20 * d_c

    return {
        "sci": float(sci),
        "sci_dir": sd,          # 1 = rally → SHORT; -1 = sell-off → LONG
        "C_end": C_end,
        "C_start": C_start,
        "close_range": abs(C_end - C_start),
        "r_c": float(r_c),
        "e_c": float(e_c),
        "x_c": float(x_c),
        "d_c": float(d_c),
    }


# ---------------------------------------------------------------------------
# Step 3 — Repair measurement
# ---------------------------------------------------------------------------

def measure_repair(
    df: pd.DataFrame,
    rth_date: pd.Timestamp,
    C_end: float,
    C_start: float,
    sci_dir: int,
) -> Optional[float]:
    """
    Look at the 01:55 ET bar of the next calendar day.
    Returns Repair float or None if no bar found.
    """
    next_date = rth_date + timedelta(days=1)

    # Build the target timestamp: next_date at 01:55 ET
    target_ts = next_date.replace(hour=1, minute=55, second=0, microsecond=0)
    # Allow up to 01:57 in case of slight gaps
    window_end = next_date.replace(hour=1, minute=57, second=0, microsecond=0)

    repair_bars = df[(df.index >= target_ts) & (df.index <= window_end)]
    if len(repair_bars) == 0:
        return None

    C_1955 = float(repair_bars["close"].iloc[0])
    repair = -sci_dir * (C_1955 - C_end) / (abs(C_end - C_start) + 1e-9)
    return float(repair)


# ---------------------------------------------------------------------------
# Step 4 — Entry confirmation
# ---------------------------------------------------------------------------

def get_confirmation_bars(
    df: pd.DataFrame,
    next_date: pd.Timestamp,
) -> Optional[dict]:
    """
    Aggregate the four 5-min synthetic bars needed for entry confirmation.
    Returns dict of bar_1945, bar_1950, bar_1955, bar_0200 or None if any missing.
    """
    windows = [
        ("bar_1945", next_date.replace(hour=1, minute=45), next_date.replace(hour=1, minute=49, second=59)),
        ("bar_1950", next_date.replace(hour=1, minute=50), next_date.replace(hour=1, minute=54, second=59)),
        ("bar_1955", next_date.replace(hour=1, minute=55), next_date.replace(hour=1, minute=59, second=59)),
        ("bar_0200", next_date.replace(hour=2, minute=0),  next_date.replace(hour=2, minute=4,  second=59)),
    ]

    result = {}
    for name, t_start, t_end in windows:
        slice_bars = df[(df.index >= t_start) & (df.index <= t_end)]
        if len(slice_bars) == 0:
            return None
        result[name] = agg_1m_to_5m(slice_bars)

    return result


# ---------------------------------------------------------------------------
# Step 5 — Stops and targets
# ---------------------------------------------------------------------------

def compute_stops(
    df: pd.DataFrame,
    rth_date: pd.Timestamp,
    next_date: pd.Timestamp,
    C_end: float,
    entry_price: float,
    direction: int,
    close_range: float,
) -> dict:
    """Compute stop and TP levels."""
    # ETH bars: from rth_date 16:00 to next_date 02:00 ET
    eth_start = rth_date.replace(hour=16, minute=0, second=0, microsecond=0)
    eth_end = next_date.replace(hour=2, minute=0, second=0, microsecond=0)
    eth_bars = df[(df.index >= eth_start) & (df.index <= eth_end)]

    if direction == 1:  # LONG
        overnight_extreme = float(eth_bars["low"].min()) if len(eth_bars) > 0 else entry_price - 10
        overnight_extreme_dist = entry_price - overnight_extreme + SLIP
    else:  # SHORT
        overnight_extreme = float(eth_bars["high"].max()) if len(eth_bars) > 0 else entry_price + 10
        overnight_extreme_dist = overnight_extreme - entry_price + SLIP

    sl_dist = min(SL_MULT * close_range, overnight_extreme_dist)
    sl_dist = max(sl_dist, 4 * SLIP)  # at least 4 ticks

    tp_dist = TP_MULT * close_range
    profit_target = entry_price + direction * tp_dist
    stop_price = entry_price - direction * sl_dist

    return {
        "sl_dist": sl_dist,
        "tp_dist": tp_dist,
        "profit_target": profit_target,
        "stop_price": stop_price,
        "overnight_extreme": overnight_extreme,
    }


# ---------------------------------------------------------------------------
# Step 6 — Trade simulation
# ---------------------------------------------------------------------------

def simulate_trade(
    df: pd.DataFrame,
    next_date: pd.Timestamp,
    entry_price: float,
    direction: int,
    stop_price: float,
    profit_target: float,
) -> tuple[float, str]:
    """
    Scan 02:05 ET to 02:59 ET for TP/SL; hard exit at 03:00.
    Returns (exit_price, exit_reason).
    """
    scan_start = next_date.replace(hour=2, minute=5, second=0, microsecond=0)
    scan_end = next_date.replace(hour=2, minute=59, second=59, microsecond=0)
    hard_exit_ts = next_date.replace(hour=HARD_EXIT_HOUR, minute=HARD_EXIT_MIN, second=0, microsecond=0)

    scan_bars = df[(df.index >= scan_start) & (df.index <= scan_end)]

    for ts, bar in scan_bars.iterrows():
        if direction == 1:  # LONG
            sl_hit = bar["low"] <= stop_price
            tp_hit = bar["high"] >= profit_target
        else:  # SHORT
            sl_hit = bar["high"] >= stop_price
            tp_hit = bar["low"] <= profit_target

        if tp_hit and sl_hit:
            return profit_target - SLIP if direction == 1 else profit_target + SLIP, "tp"
        if tp_hit:
            return profit_target - SLIP if direction == 1 else profit_target + SLIP, "tp"
        if sl_hit:
            return stop_price - SLIP if direction == 1 else stop_price + SLIP, "stop"

    # Hard exit: use open of 03:00 bar if available, else last close
    hard_bars = df[(df.index >= hard_exit_ts) & (df.index < hard_exit_ts + timedelta(minutes=1))]
    if len(hard_bars) > 0:
        exit_price = float(hard_bars["open"].iloc[0])
    else:
        # Fallback: last bar before 03:00
        pre_bars = df[(df.index >= scan_start) & (df.index < hard_exit_ts)]
        if len(pre_bars) > 0:
            exit_price = float(pre_bars["close"].iloc[-1])
        else:
            exit_price = entry_price
    return exit_price - SLIP if direction == 1 else exit_price + SLIP, "hard_exit"


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame) -> list[dict]:
    """Run NOVA backtest. Returns list of trade dicts."""

    # Compute daily RTH ranges for ATR
    daily_ranges = compute_rth_daily_ranges(df)  # keyed by date (no tz)

    # Get RTH dates
    minutes = df.index.hour * 60 + df.index.minute
    rth_mask = (minutes >= 9 * 60 + 30) & (minutes <= 15 * 60 + 59)
    rth = df[rth_mask]
    rth_dates_raw = sorted(rth.index.normalize().unique())

    # Build per-RTH-date DataFrame lookup
    rth_by_date = {}
    for d in rth_dates_raw:
        day_bars = rth[rth.index.normalize() == d]
        if len(day_bars) > 0:
            rth_by_date[d] = day_bars

    date_list = sorted(rth_by_date.keys())

    # Rolling SCI history for percentile
    sci_history: list[float] = []
    trades: list[dict] = []

    for i, rth_ts in enumerate(date_list):
        rth_date = rth_ts  # tz-aware Timestamp (normalized, but still ET)

        # Need at least SCI_MIN_HISTORY days of history before qualifying
        day_rth = rth_by_date[rth_date]
        rth_date_key = rth_date.date()

        # --- ATR: rolling 20-day mean of prior daily ranges ---
        idx_in_ranges = daily_ranges.index.tolist()
        try:
            pos = idx_in_ranges.index(rth_date_key)
        except ValueError:
            continue
        window_start = max(0, pos - 19)
        atr_window = daily_ranges.iloc[window_start: pos]
        if len(atr_window) < 5:
            # Append None to SCI history for burn-in accounting (no SCI value yet)
            continue
        atr_d = float(atr_window.mean())

        # --- Step 1: Compute SCI ---
        sci_result = compute_sci_for_day(day_rth, atr_d)
        if sci_result is None:
            continue

        sci_val = sci_result["sci"]

        # --- Step 2: Rolling SCI percentile ---
        if len(sci_history) >= SCI_MIN_HISTORY:
            window = sci_history[-SCI_PCTL_WINDOW:]
            sci_p70 = float(np.percentile(window, 70))
            qualified = sci_val > sci_p70
        else:
            qualified = False

        # Append to history AFTER computing percentile (no look-ahead)
        sci_history.append(sci_val)

        if not qualified:
            continue

        # --- Determine overnight window ---
        next_date = rth_date + timedelta(days=1)

        # Weekend check: if rth_date is Friday (weekday=4), next_date is Saturday → skip
        if rth_date.weekday() == 4:
            continue
        # Also skip if next_date is Sunday (should not happen with daily advance from non-Fri)
        if next_date.weekday() == 6:
            continue

        sci_dir = sci_result["sci_dir"]
        C_end = sci_result["C_end"]
        C_start = sci_result["C_start"]
        close_range = sci_result["close_range"]

        # --- Step 3: Repair measurement ---
        repair = measure_repair(df, rth_date, C_end, C_start, sci_dir)
        if repair is None:
            continue
        if not (REPAIR_LO <= repair <= REPAIR_HI):
            continue

        # --- Step 4: Entry confirmation ---
        conf_bars = get_confirmation_bars(df, next_date)
        if conf_bars is None:
            continue

        bar_1945 = conf_bars["bar_1945"]
        bar_1950 = conf_bars["bar_1950"]
        bar_1955 = conf_bars["bar_1955"]
        bar_0200 = conf_bars["bar_0200"]

        # Midpoint of prior 3 bars
        prior_high = max(bar_1945["high"], bar_1950["high"], bar_1955["high"])
        prior_low = min(bar_1945["low"], bar_1950["low"], bar_1955["low"])
        mid_3 = (prior_high + prior_low) / 2.0

        direction: int
        entry_trigger_price: float

        if sci_dir == -1:  # close sell-off → LONG reversal
            confirmed = bar_0200["close"] > mid_3
            if not confirmed:
                continue
            direction = 1
            entry_trigger_price = bar_0200["high"] + SLIP  # buy 1 tick above

        else:  # sci_dir == 1, close rally → SHORT reversal
            confirmed = bar_0200["close"] < mid_3
            if not confirmed:
                continue
            direction = -1
            entry_trigger_price = bar_0200["low"] - SLIP  # sell 1 tick below

        # Check if entry trigger actually gets hit (scan 02:05 onwards)
        scan_start = next_date.replace(hour=2, minute=5, second=0, microsecond=0)
        scan_end_trigger = next_date.replace(hour=2, minute=59, second=59, microsecond=0)
        trigger_bars = df[(df.index >= scan_start) & (df.index <= scan_end_trigger)]

        entry_filled = False
        entry_price = entry_trigger_price

        if direction == 1:
            # Triggered when bar breaks above bar_0200.high
            trigger_level = bar_0200["high"]
            for ts_t, bar_t in trigger_bars.iterrows():
                if bar_t["high"] >= trigger_level:
                    entry_price = trigger_level + SLIP
                    entry_filled = True
                    break
        else:
            trigger_level = bar_0200["low"]
            for ts_t, bar_t in trigger_bars.iterrows():
                if bar_t["low"] <= trigger_level:
                    entry_price = trigger_level - SLIP
                    entry_filled = True
                    break

        if not entry_filled:
            continue

        # --- Step 5: Stops and targets ---
        stops = compute_stops(
            df, rth_date, next_date, C_end, entry_price, direction, close_range
        )

        # --- Step 6: Trade simulation ---
        exit_price, exit_reason = simulate_trade(
            df, next_date, entry_price, direction,
            stops["stop_price"], stops["profit_target"]
        )

        raw_pnl_pts = (exit_price - entry_price) * direction
        pnl_usd = raw_pnl_pts * POINT_VALUE * N_CONTRACTS

        trades.append({
            "date": str(rth_date.date()),
            "next_date": str(next_date.date()),
            "direction": "LONG" if direction == 1 else "SHORT",
            "sci": round(sci_val, 4),
            "sci_dir": sci_dir,
            "repair": round(repair, 4),
            "C_end": round(C_end, 4),
            "C_start": round(C_start, 4),
            "close_range": round(close_range, 4),
            "mid_3": round(mid_3, 4),
            "entry_price": round(entry_price, 4),
            "stop_price": round(stops["stop_price"], 4),
            "profit_target": round(stops["profit_target"], 4),
            "sl_dist": round(stops["sl_dist"], 4),
            "tp_dist": round(stops["tp_dist"], 4),
            "exit_price": round(exit_price, 4),
            "exit_reason": exit_reason,
            "pnl_pts": round(raw_pnl_pts, 4),
            "pnl_usd": round(pnl_usd, 2),
            "atr_d": round(atr_d, 4),
        })

    return trades


# ---------------------------------------------------------------------------
# Step 7 — Statistics
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {
            "N": 0, "WR": 0.0, "avg_pnl": 0.0,
            "total_pnl": 0.0, "max_dd": 0.0, "sharpe": 0.0,
        }

    pnls = [t["pnl_usd"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    n = len(pnls)
    wr = wins / n * 100.0
    avg_pnl = float(np.mean(pnls))
    total_pnl = float(np.sum(pnls))

    # Sharpe (trade-level)
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Max drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(dd.min())

    return {
        "N": n,
        "WR": round(wr, 1),
        "avg_pnl": round(avg_pnl, 2),
        "total_pnl": round(total_pnl, 2),
        "max_dd": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
    }


def exit_breakdown(trades: list[dict]) -> list[dict]:
    """Summarize trades by exit reason."""
    reasons = {}
    for t in trades:
        r = t["exit_reason"]
        if r not in reasons:
            reasons[r] = {"reason": r, "N": 0, "wins": 0, "total_pnl": 0.0}
        reasons[r]["N"] += 1
        if t["pnl_usd"] > 0:
            reasons[r]["wins"] += 1
        reasons[r]["total_pnl"] += t["pnl_usd"]

    result = []
    for r in ["tp", "stop", "hard_exit"]:
        if r in reasons:
            d = reasons[r]
            n = d["N"]
            wr = d["wins"] / n * 100.0 if n > 0 else 0.0
            result.append({
                "reason": r,
                "N": n,
                "WR": round(wr, 1),
                "total_pnl": round(d["total_pnl"], 2),
                "avg_pnl": round(d["total_pnl"] / n, 2) if n > 0 else 0.0,
            })
    return result


def monthly_pnl(trades: list[dict]) -> dict:
    monthly: dict[str, float] = {}
    for t in trades:
        ym = t["date"][:7]  # "2025-06"
        monthly[ym] = monthly.get(ym, 0.0) + t["pnl_usd"]
    return {k: round(v, 2) for k, v in sorted(monthly.items())}


def monotonicity_by_sci(trades: list[dict]) -> list[dict]:
    """Sort qualifying trades by SCI decile."""
    if not trades:
        return []
    sci_vals = [t["sci"] for t in trades]
    decile_edges = np.percentile(sci_vals, np.arange(0, 101, 10))

    rows = []
    for dec in range(10):
        lo = decile_edges[dec]
        hi = decile_edges[dec + 1]
        if dec == 9:
            sub = [t for t in trades if t["sci"] >= lo]
        else:
            sub = [t for t in trades if lo <= t["sci"] < hi]
        n = len(sub)
        if n == 0:
            rows.append({"decile": f"{dec*10}-{(dec+1)*10}pct", "N": 0, "WR": None, "avg_pnl": None})
            continue
        wins = sum(1 for t in sub if t["pnl_usd"] > 0)
        wr = wins / n * 100.0
        avg_p = float(np.mean([t["pnl_usd"] for t in sub]))
        rows.append({
            "decile": f"{dec*10}-{(dec+1)*10}pct",
            "N": n,
            "WR": round(wr, 1),
            "avg_pnl": round(avg_p, 2),
        })
    return rows


def monotonicity_by_repair(trades: list[dict]) -> list[dict]:
    """Sort trades by Repair band."""
    bands = [
        (0.05, 0.15),
        (0.15, 0.25),
        (0.25, 0.35),
        (0.35, 0.45),
    ]
    rows = []
    for lo, hi in bands:
        sub = [t for t in trades if lo <= t["repair"] <= hi]
        n = len(sub)
        if n == 0:
            rows.append({"band": f"{lo:.2f}-{hi:.2f}", "N": 0, "WR": None, "avg_pnl": None})
            continue
        wins = sum(1 for t in sub if t["pnl_usd"] > 0)
        wr = wins / n * 100.0
        avg_p = float(np.mean([t["pnl_usd"] for t in sub]))
        rows.append({
            "band": f"{lo:.2f}-{hi:.2f}",
            "N": n,
            "WR": round(wr, 1),
            "avg_pnl": round(avg_p, 2),
        })
    return rows


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_summary_table(label: str, metrics: dict):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")


def print_monotonicity(title: str, rows: list[dict], key_col: str):
    print(f"\n{'-'*65}")
    print(f"  {title}")
    hdr_key = "Decile" if key_col == "decile" else "Band"
    print(f"  {hdr_key:<18} {'N':>5} {'WR%':>7} {'AvgPnL':>10}")
    print(f"  {'-'*42}")
    for r in rows:
        wr_str = f"{r['WR']:7.1f}" if r["WR"] is not None else "    N/A"
        avg_str = f"{r['avg_pnl']:10.2f}" if r["avg_pnl"] is not None else "       N/A"
        print(f"  {r[key_col]:<18} {r['N']:>5} {wr_str} {avg_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading MES 1-min bars (primary + secondary)...")
    df = load_bars()
    print(f"  Total bars: {len(df):,}  |  Range: {df.index[0]} -> {df.index[-1]}")

    print("\nRunning NOVA backtest...")
    trades = run_backtest(df)
    print(f"  Total trades: {len(trades)}")

    if len(trades) == 0:
        print("\nNo trades generated. Check data and parameters.")
        return

    # --- Overall metrics ---
    overall = compute_metrics(trades)
    long_trades = [t for t in trades if t["direction"] == "LONG"]
    short_trades = [t for t in trades if t["direction"] == "SHORT"]
    long_metrics = compute_metrics(long_trades)
    short_metrics = compute_metrics(short_trades)

    print_summary_table("NOVA Overall", overall)
    print_summary_table("NOVA Long Only", long_metrics)
    print_summary_table("NOVA Short Only", short_metrics)

    # --- Exit breakdown ---
    exit_bkdn = exit_breakdown(trades)
    print(f"\n{'-'*55}")
    print("  Exit Breakdown")
    print(f"  {'Reason':<12} {'N':>5} {'WR%':>7} {'AvgPnL':>10} {'TotalPnL':>12}")
    print(f"  {'-'*46}")
    for row in exit_bkdn:
        print(f"  {row['reason']:<12} {row['N']:>5} {row['WR']:>7.1f} {row['avg_pnl']:>10.2f} {row['total_pnl']:>12.2f}")

    # --- Monthly PnL ---
    monthly = monthly_pnl(trades)
    print(f"\n{'-'*45}")
    print("  Monthly PnL")
    print(f"  {'Month':<12} {'PnL':>10}")
    print(f"  {'-'*24}")
    for ym, pnl in monthly.items():
        print(f"  {ym:<12} {pnl:>10.2f}")

    # --- Monotonicity ---
    sci_mono = monotonicity_by_sci(trades)
    repair_mono = monotonicity_by_repair(trades)
    print_monotonicity("Monotonicity by SCI Decile", sci_mono, "decile")
    print_monotonicity("Monotonicity by Repair Band", repair_mono, "band")

    # --- Data range for meta ---
    data_start = str(df.index[0].date())
    data_end = str(df.index[-1].date())

    # --- Save results ---
    results = {
        "meta": {
            "instrument": "MES",
            "data_range": f"{data_start} to {data_end}",
            "n_days": len(set(t["date"] for t in trades)),
            "sci_pctl_window": SCI_PCTL_WINDOW,
            "repair_lo": REPAIR_LO,
            "repair_hi": REPAIR_HI,
            "tp_mult": TP_MULT,
            "sl_mult": SL_MULT,
            "n_contracts": N_CONTRACTS,
        },
        "overall": overall,
        "long_only": long_metrics,
        "short_only": short_metrics,
        "exit_breakdown": exit_bkdn,
        "monotonicity_by_sci": sci_mono,
        "monotonicity_by_repair": repair_mono,
        "monthly_pnl": monthly,
        "trade_log": trades,
    }

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"N={overall['N']}  WR={overall['WR']}%  TotalPnL=${overall['total_pnl']:.2f}  "
          f"MaxDD=${overall['max_dd']:.2f}  Sharpe={overall['sharpe']:.3f}")


if __name__ == "__main__":
    main()
