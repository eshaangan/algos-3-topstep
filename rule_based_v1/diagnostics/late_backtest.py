"""LATE Backtest — Late Auction Timing Engine diagnostic.

Strategy: Afternoon momentum continuation play on 1-min bars.
Signal window: 3:00 PM – 3:35 PM ET.  Hard exit: 4:03 PM ET.

Variant families:
  A — Revised LATE (active midday + wider 16-tick stop)
  B — Pure late-momentum (no compression filter, 3:00 PM entry)

Data sources:
  MES: data/processed/mes_1m_bars_cache.h5  /bars_1m   (Jun–Dec 2025)
  MNQ: data/processed/mnq_2026ytd_1min.h5  /bars_1min (Jan–Mar 2026)

Run (from project root):
    python rule_based_v1/diagnostics/late_backtest.py
    python rule_based_v1/diagnostics/late_backtest.py --stage A2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for _p in [str(ROOT), str(RBV1)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Results path
# ---------------------------------------------------------------------------
RESULTS_PATH = ROOT / "rule_based_v1" / "diagnostics" / "late_results.json"

# ---------------------------------------------------------------------------
# Instrument definitions
# ---------------------------------------------------------------------------
INSTRUMENTS = {
    "MES": {
        "name": "MES",
        "label": "MES (Jun-Dec 2025)",
        "h5_path": ROOT / "data" / "processed" / "mes_1m_bars_cache.h5",
        "h5_key": "/bars_1m",
        "tick_size": 0.25,
        "point_value": 5.0,
        "slippage_ticks": 1,
        "has_timestamp_col": True,   # integer index, timestamp column
        "tz": "US/Eastern",
    },
    "MNQ": {
        "name": "MNQ",
        "label": "MNQ (Jan-Mar 2026)",
        "h5_path": ROOT / "data" / "processed" / "mnq_2026ytd_1min.h5",
        "h5_key": "/bars_1min",
        "tick_size": 0.25,
        "point_value": 2.0,
        "slippage_ticks": 1,
        "has_timestamp_col": False,  # already has DatetimeIndex in ET
        "tz": "US/Eastern",
    },
}

# ---------------------------------------------------------------------------
# Minute-of-day boundaries
# ---------------------------------------------------------------------------
RTH_OPEN_MIN  = 570   # 9:30
RTH_CLOSE_MIN = 960   # 4:00
AM_END_MIN    = 720   # noon
MID_END_MIN   = 870   # 2:30 (bars up to 14:29 inclusive)
MID_CLOSE_MIN = 869   # 2:29 PM bar
MID_START_MIN = 720   # 12:00 PM bar (for mid_drift calc)
SIG_A_MIN     = 920   # 3:20 PM (A-variant market entry bar)
SIG_B_MIN     = 900   # 3:00 PM (B-variant market entry bar)
TBOX_START_MIN= 900   # 3:00 PM
TBOX_END_MIN  = 920   # 3:20 PM (last bar of trigger box)
SCAN_START_MIN= 921   # 3:21 PM (first breakout scan bar)
SCAN_END_MIN  = 935   # 3:35 PM (last scan bar)
HARD_EXIT_MIN = 963   # 4:03 PM

WARMUP_DAYS   = 20

# ---------------------------------------------------------------------------
# Stage metadata
# ---------------------------------------------------------------------------
STAGE_NOTES = {
    "A1": "r>0.3, market 3:20, 16tk stop",
    "A2": "A1 + Comp>0.45",
    "A3": "A2 + D>p80_D",
    "A4": "A2 + trigger-box breakout 3:21-3:35",
    "A5": "A4 + price pos + AM extreme",
    "B1": "r>0.3, market 3:00, 16tk stop",
    "B2": "r>0.2, market 3:00, 16tk stop",
    "B3": "r>0.3 + mid_drift>0, market 3:00, 16tk stop",
    "B4": "r>0.3 + trigger-box breakout 3:21-3:35",
}

ALL_STAGES = list(STAGE_NOTES.keys())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bars(inst: dict) -> pd.DataFrame:
    """Load 1-min bars, normalise to ET DatetimeIndex + date/min_of_day cols."""
    h5_path = inst["h5_path"]
    h5_key  = inst["h5_key"]

    with pd.HDFStore(str(h5_path), mode="r") as store:
        df = store[h5_key].copy()

    if inst["has_timestamp_col"]:
        # MES: integer index, timestamp column (ns UTC)
        df["dt"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(inst["tz"])
        df = df.sort_values("dt").reset_index(drop=True)
    else:
        # MNQ: DatetimeIndex already in ET
        if df.index.tz is None:
            df.index = df.index.tz_localize(inst["tz"])
        else:
            df.index = df.index.tz_convert(inst["tz"])
        df = df.sort_index().reset_index()
        # rename the index column to 'dt'
        idx_col = df.columns[0]
        df = df.rename(columns={idx_col: "dt"})

    df["date"]       = df["dt"].dt.date
    df["min_of_day"] = df["dt"].dt.hour * 60 + df["dt"].dt.minute
    return df


# ---------------------------------------------------------------------------
# Daily aggregate builder
# ---------------------------------------------------------------------------

def build_daily_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per trading date with all filter variables."""
    rows = []
    grouped = df.groupby("date")
    dates = sorted(grouped.groups.keys())

    for i, date in enumerate(dates):
        day_df = grouped.get_group(date)

        # RTH bars
        rth = day_df[
            (day_df["min_of_day"] >= RTH_OPEN_MIN) &
            (day_df["min_of_day"] < RTH_CLOSE_MIN)
        ]
        if rth.empty:
            continue

        rth_high = rth["high"].max()
        rth_low  = rth["low"].min()

        # AM (9:30 – 11:59)
        am = rth[rth["min_of_day"] < AM_END_MIN]
        am_high = am["high"].max() if not am.empty else np.nan
        am_low  = am["low"].min()  if not am.empty else np.nan
        am_high_time = am.loc[am["high"].idxmax(), "min_of_day"] if not am.empty else np.nan
        am_low_time  = am.loc[am["low"].idxmin(),  "min_of_day"] if not am.empty else np.nan

        # 9:30 open
        open_930 = rth.iloc[0]["open"]

        # Close of first bar at/after 10:00
        after_10   = rth[rth["min_of_day"] >= 600]
        r10_close  = after_10.iloc[0]["close"] if not after_10.empty else np.nan

        # ETH (prior day 18:00 + pre-market same day)
        eth_today = day_df[day_df["min_of_day"] < RTH_OPEN_MIN]
        if i > 0:
            prev_df  = grouped.get_group(dates[i - 1])
            prev_eth = prev_df[prev_df["min_of_day"] >= 1080]
            eth_all  = pd.concat([prev_eth, eth_today])
        else:
            eth_all = eth_today
        eth_high = eth_all["high"].max() if not eth_all.empty else np.nan
        eth_low  = eth_all["low"].min()  if not eth_all.empty else np.nan

        # MID range (noon to 2:29 PM)
        mid = day_df[
            (day_df["min_of_day"] >= AM_END_MIN) &
            (day_df["min_of_day"] < MID_END_MIN)
        ]
        R_mid = mid["high"].max() - mid["low"].min() if not mid.empty else np.nan

        # mid_drift: (close_2:29 - close_12:00) * sign(r_open)
        # stored raw; sign(r_open) applied after r_open is computed
        bar_1200 = day_df[day_df["min_of_day"] == MID_START_MIN]
        bar_1429 = day_df[day_df["min_of_day"] == MID_CLOSE_MIN]
        close_1200 = bar_1200.iloc[-1]["close"] if not bar_1200.empty else np.nan
        close_1429 = bar_1429.iloc[-1]["close"] if not bar_1429.empty else np.nan
        mid_diff = close_1429 - close_1200  # raw; multiply by sign(r_open) later

        rows.append({
            "date":         date,
            "rth_high":     rth_high,
            "rth_low":      rth_low,
            "rth_range":    rth_high - rth_low,
            "am_high":      am_high,
            "am_low":       am_low,
            "am_high_time": am_high_time,
            "am_low_time":  am_low_time,
            "eth_high":     eth_high,
            "eth_low":      eth_low,
            "open_930":     open_930,
            "r10_close":    r10_close,
            "R_mid":        R_mid,
            "mid_diff":     mid_diff,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# D statistic at 3:00 PM
# ---------------------------------------------------------------------------

def compute_D_at_3pm(rth_day: pd.DataFrame) -> float:
    bars = rth_day[rth_day["min_of_day"] < SIG_B_MIN].copy()
    if len(bars) < 2:
        return np.nan
    n     = len(bars)
    idx_h = bars["high"].values.argmax()
    idx_l = bars["low"].values.argmin()
    u_H   = idx_h / (n - 1)
    u_L   = idx_l / (n - 1)
    return np.sqrt(u_H * (1 - u_H)) + np.sqrt(u_L * (1 - u_L))


# ---------------------------------------------------------------------------
# Exit simulation
# ---------------------------------------------------------------------------

def simulate_exit(
    rth: pd.DataFrame,
    entry_row_idx: int,
    entry_price: float,
    stop: float,
    pt: float,
    direction: int,
    point_value: float,
    slippage_pts: float,
) -> float:
    """Walk 1-min bars forward; return PnL in $ (single contract)."""
    rt_slip = slippage_pts * 2 * point_value  # exit slippage cost

    future = rth.loc[rth.index > entry_row_idx]
    for _, bar in future.iterrows():
        if bar["min_of_day"] >= HARD_EXIT_MIN:
            exit_price = bar["open"] - direction * slippage_pts
            raw = (exit_price - entry_price) * direction * point_value
            return raw - rt_slip

        if direction == 1 and bar["low"] <= stop:
            raw = (stop - entry_price) * point_value
            return raw - rt_slip
        if direction == -1 and bar["high"] >= stop:
            raw = (entry_price - stop) * point_value
            return raw - rt_slip

        if direction == 1 and bar["high"] >= pt:
            raw = (pt - entry_price) * point_value
            return raw - rt_slip
        if direction == -1 and bar["low"] <= pt:
            raw = (entry_price - pt) * point_value
            return raw - rt_slip

    # Reached end of RTH
    if not rth.empty:
        last_close = rth.iloc[-1]["close"]
        exit_price  = last_close - direction * slippage_pts
        raw = (exit_price - entry_price) * direction * point_value
        return raw - rt_slip
    return 0.0


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_stage(
    df: pd.DataFrame,
    daily: pd.DataFrame,
    stage: str,
    inst: dict,
) -> tuple[list[dict], list[float]]:
    """
    Run backtest for one variant/stage combination.
    Returns (trades, daily_pnls).
    """
    tick_size    = inst["tick_size"]
    point_value  = inst["point_value"]
    slip_pts     = inst["slippage_ticks"] * tick_size
    stop_16tick  = 16 * tick_size   # 4.0 pts
    max_tbox_cap = 20 * tick_size   # 5.0 pts

    variant = stage[0]   # 'A' or 'B'

    # ---- Compute rolling features ----
    daily = daily.copy().reset_index(drop=True)
    daily["sigma_20"] = daily["rth_range"].rolling(20, min_periods=20).mean()

    # D statistic per day
    grouped = df.groupby("date")
    D_values = []
    for date in daily["date"]:
        if date not in grouped.groups:
            D_values.append(np.nan)
            continue
        day_bars = grouped.get_group(date)
        rth_bars = day_bars[
            (day_bars["min_of_day"] >= RTH_OPEN_MIN) &
            (day_bars["min_of_day"] < RTH_CLOSE_MIN)
        ]
        D_values.append(compute_D_at_3pm(rth_bars))
    daily["D"]    = D_values
    daily["D_p80"] = daily["D"].rolling(20, min_periods=20).quantile(0.80)

    # r_open
    daily["r_open"] = (daily["r10_close"] - daily["open_930"]) / (
        daily["sigma_20"] + 1e-9
    )

    # Comp = R_mid / am_range
    am_range    = (daily["am_high"] - daily["am_low"]).clip(lower=0)
    daily["Comp"] = daily["R_mid"] / (am_range + 1e-9)

    # mid_drift = (close_1429 - close_1200) * sign(r_open)
    daily["mid_drift"] = daily["mid_diff"] * np.sign(daily["r_open"])

    daily_idx = daily.set_index("date")
    dates_all = daily["date"].tolist()

    trades: list[dict]      = []
    daily_pnl_map: dict = {}

    for date in dates_all:
        if date not in daily_idx.index:
            continue
        row = daily_idx.loc[date]

        # Warmup guard
        if pd.isna(row["sigma_20"]) or pd.isna(row["D_p80"]):
            daily_pnl_map[date] = 0.0
            continue
        if date not in grouped.groups:
            daily_pnl_map[date] = 0.0
            continue

        day_bars = grouped.get_group(date)
        rth = day_bars[
            (day_bars["min_of_day"] >= RTH_OPEN_MIN) &
            (day_bars["min_of_day"] < RTH_CLOSE_MIN)
        ].copy()
        if rth.empty:
            daily_pnl_map[date] = 0.0
            continue

        r_open    = row["r_open"]
        comp      = row["Comp"]
        D         = row["D"]
        D_p80     = row["D_p80"]
        mid_drift = row["mid_drift"]

        # ---- Variant A filters ----
        if variant == "A":
            # A1: |r_open| > 0.3
            if abs(r_open) < 0.3:
                daily_pnl_map[date] = 0.0
                continue
            direction = 1 if r_open > 0 else -1

            if stage in ("A2", "A3", "A4", "A5"):
                if comp <= 0.45:
                    daily_pnl_map[date] = 0.0
                    continue

            if stage == "A3":
                if pd.isna(D) or D <= D_p80:
                    daily_pnl_map[date] = 0.0
                    continue

            # A1 & A2 & A3: market entry at 3:20 PM, 16-tick stop
            if stage in ("A1", "A2", "A3"):
                entry_bar_df = rth[rth["min_of_day"] == SIG_A_MIN]
                if entry_bar_df.empty:
                    daily_pnl_map[date] = 0.0
                    continue
                entry_bar   = entry_bar_df.iloc[0]
                entry_price = entry_bar["open"] + direction * slip_pts
                stop_dist   = stop_16tick
                pt_dist     = stop_dist * 1.5
                stop        = entry_price - direction * stop_dist
                pt          = entry_price + direction * pt_dist
                pnl = simulate_exit(
                    rth, entry_bar.name, entry_price, stop, pt,
                    direction, point_value, slip_pts,
                )
                trades.append({
                    "date":       str(date),
                    "variant":    variant,
                    "stage":      stage,
                    "direction":  "LONG" if direction == 1 else "SHORT",
                    "entry_time": entry_bar["dt"].strftime("%H:%M"),
                    "entry":      round(entry_price, 4),
                    "stop":       round(stop, 4),
                    "pt":         round(pt, 4),
                    "stop_dist":  round(stop_dist, 4),
                    "pnl":        round(pnl, 2),
                })
                daily_pnl_map[date] = pnl
                continue

            # A4 & A5: trigger-box breakout (3:21-3:35 PM scan)
            tbox = rth[
                (rth["min_of_day"] >= TBOX_START_MIN) &
                (rth["min_of_day"] <= TBOX_END_MIN)
            ]
            tbox_high = tbox["high"].max() if not tbox.empty else np.nan
            tbox_low  = tbox["low"].min()  if not tbox.empty else np.nan

            traded = False
            scan_bars = rth[
                (rth["min_of_day"] >= SCAN_START_MIN) &
                (rth["min_of_day"] <= SCAN_END_MIN)
            ]
            for _, bar in scan_bars.iterrows():
                if traded:
                    break

                # A5 extra filters
                if stage == "A5":
                    # Price position: upper/lower 35% of RTH range at 3:00 PM
                    rth_to_3 = rth[rth["min_of_day"] < SIG_B_MIN]
                    if not rth_to_3.empty:
                        rth_hi  = rth_to_3["high"].max()
                        rth_lo  = rth_to_3["low"].min()
                        rth_rng = rth_hi - rth_lo
                        close_3pm_bar = rth[rth["min_of_day"] == SIG_B_MIN]
                        price_3pm = close_3pm_bar.iloc[-1]["close"] if not close_3pm_bar.empty else bar["close"]
                        if direction == 1 and price_3pm < rth_lo + 0.65 * rth_rng:
                            daily_pnl_map[date] = 0.0
                            break
                        if direction == -1 and price_3pm > rth_lo + 0.35 * rth_rng:
                            daily_pnl_map[date] = 0.0
                            break
                    # RTH extreme set before 11:00 AM (660 = 11:00)
                    am_high_time = row["am_high_time"]
                    am_low_time  = row["am_low_time"]
                    if direction == 1:
                        if pd.isna(am_high_time) or am_high_time >= 660:
                            daily_pnl_map[date] = 0.0
                            break
                    else:
                        if pd.isna(am_low_time) or am_low_time >= 660:
                            daily_pnl_map[date] = 0.0
                            break

                bar_close = bar["close"]
                if direction == 1:
                    if pd.isna(tbox_high) or bar_close <= tbox_high:
                        continue
                    entry_price = bar_close + slip_pts
                else:
                    if pd.isna(tbox_low) or bar_close >= tbox_low:
                        continue
                    entry_price = bar_close - slip_pts

                # Stop: other side of trigger box, capped at 20 ticks
                if not pd.isna(tbox_high) and not pd.isna(tbox_low):
                    tbox_range = tbox_high - tbox_low
                    stop_dist  = min(tbox_range, max_tbox_cap)
                    stop_dist  = max(stop_dist, tick_size)
                else:
                    stop_dist = stop_16tick
                pt_dist = stop_dist * 1.5
                stop    = entry_price - direction * stop_dist
                pt      = entry_price + direction * pt_dist

                pnl = simulate_exit(
                    rth, bar.name, entry_price, stop, pt,
                    direction, point_value, slip_pts,
                )
                trades.append({
                    "date":       str(date),
                    "variant":    variant,
                    "stage":      stage,
                    "direction":  "LONG" if direction == 1 else "SHORT",
                    "entry_time": bar["dt"].strftime("%H:%M"),
                    "entry":      round(entry_price, 4),
                    "stop":       round(stop, 4),
                    "pt":         round(pt, 4),
                    "stop_dist":  round(stop_dist, 4),
                    "pnl":        round(pnl, 2),
                })
                daily_pnl_map[date] = pnl
                traded = True

            if not traded:
                daily_pnl_map.setdefault(date, 0.0)

        # ---- Variant B filters ----
        else:
            # B1: |r_open| > 0.3
            # B2: |r_open| > 0.2
            # B3: |r_open| > 0.3 + mid_drift > 0
            # B4: |r_open| > 0.3 + trigger-box breakout

            r_thresh = 0.2 if stage == "B2" else 0.3
            if abs(r_open) < r_thresh:
                daily_pnl_map[date] = 0.0
                continue
            direction = 1 if r_open > 0 else -1

            if stage == "B3":
                if mid_drift <= 0:
                    daily_pnl_map[date] = 0.0
                    continue

            # B1, B2, B3: market entry at 3:00 PM
            if stage in ("B1", "B2", "B3"):
                entry_bar_df = rth[rth["min_of_day"] == SIG_B_MIN]
                if entry_bar_df.empty:
                    daily_pnl_map[date] = 0.0
                    continue
                entry_bar   = entry_bar_df.iloc[0]
                entry_price = entry_bar["open"] + direction * slip_pts
                stop_dist   = stop_16tick
                pt_dist     = stop_dist * 1.5
                stop        = entry_price - direction * stop_dist
                pt          = entry_price + direction * pt_dist
                pnl = simulate_exit(
                    rth, entry_bar.name, entry_price, stop, pt,
                    direction, point_value, slip_pts,
                )
                trades.append({
                    "date":       str(date),
                    "variant":    variant,
                    "stage":      stage,
                    "direction":  "LONG" if direction == 1 else "SHORT",
                    "entry_time": entry_bar["dt"].strftime("%H:%M"),
                    "entry":      round(entry_price, 4),
                    "stop":       round(stop, 4),
                    "pt":         round(pt, 4),
                    "stop_dist":  round(stop_dist, 4),
                    "pnl":        round(pnl, 2),
                })
                daily_pnl_map[date] = pnl
                continue

            # B4: trigger-box breakout (same as A4 without Comp filter)
            tbox = rth[
                (rth["min_of_day"] >= TBOX_START_MIN) &
                (rth["min_of_day"] <= TBOX_END_MIN)
            ]
            tbox_high = tbox["high"].max() if not tbox.empty else np.nan
            tbox_low  = tbox["low"].min()  if not tbox.empty else np.nan

            traded = False
            scan_bars = rth[
                (rth["min_of_day"] >= SCAN_START_MIN) &
                (rth["min_of_day"] <= SCAN_END_MIN)
            ]
            for _, bar in scan_bars.iterrows():
                if traded:
                    break
                bar_close = bar["close"]
                if direction == 1:
                    if pd.isna(tbox_high) or bar_close <= tbox_high:
                        continue
                    entry_price = bar_close + slip_pts
                else:
                    if pd.isna(tbox_low) or bar_close >= tbox_low:
                        continue
                    entry_price = bar_close - slip_pts

                if not pd.isna(tbox_high) and not pd.isna(tbox_low):
                    tbox_range = tbox_high - tbox_low
                    stop_dist  = min(tbox_range, max_tbox_cap)
                    stop_dist  = max(stop_dist, tick_size)
                else:
                    stop_dist = stop_16tick
                pt_dist = stop_dist * 1.5
                stop    = entry_price - direction * stop_dist
                pt      = entry_price + direction * pt_dist

                pnl = simulate_exit(
                    rth, bar.name, entry_price, stop, pt,
                    direction, point_value, slip_pts,
                )
                trades.append({
                    "date":       str(date),
                    "variant":    variant,
                    "stage":      stage,
                    "direction":  "LONG" if direction == 1 else "SHORT",
                    "entry_time": bar["dt"].strftime("%H:%M"),
                    "entry":      round(entry_price, 4),
                    "stop":       round(stop, 4),
                    "pt":         round(pt, 4),
                    "stop_dist":  round(stop_dist, 4),
                    "pnl":        round(pnl, 2),
                })
                daily_pnl_map[date] = pnl
                traded = True

            if not traded:
                daily_pnl_map.setdefault(date, 0.0)

    daily_pnls = [daily_pnl_map.get(d, 0.0) for d in dates_all]
    return trades, daily_pnls


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[dict], daily_pnls: list[float]) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": 0.0, "avg_pnl": 0.0, "sharpe": None, "max_dd": 0.0}

    pnls    = np.array([t["pnl"] for t in trades])
    wins    = (pnls > 0).sum()
    wr      = wins / n

    daily_arr = np.array(daily_pnls)
    std       = daily_arr.std()
    sharpe    = (daily_arr.mean() / (std + 1e-9)) * np.sqrt(252) if std > 0 else 0.0

    cum    = np.cumsum(daily_arr)
    peak   = np.maximum.accumulate(cum)
    max_dd = (cum - peak).min()

    return {
        "n":       n,
        "wr":      round(wr, 4),
        "avg_pnl": round(pnls.mean(), 2),
        "sharpe":  round(sharpe, 3),
        "max_dd":  round(max_dd, 2),
    }


# ---------------------------------------------------------------------------
# Per-trade print helper
# ---------------------------------------------------------------------------

def print_trade_log(trades: list[dict]) -> None:
    print(
        f"\n  {'Date':<12}{'Dir':<6}{'Time':<6}{'Entry':>8}"
        f"{'Stop':>8}{'PT':>8}{'Dist':>6}{'PnL':>9}"
    )
    for t in trades:
        print(
            f"  {t['date']:<12}{t['direction']:<6}"
            f"{t.get('entry_time','?'):>6}"
            f"{t['entry']:>8.2f}{t['stop']:>8.2f}"
            f"{t['pt']:>8.2f}{t['stop_dist']:>6.2f}"
            f"  ${t['pnl']:>8.2f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LATE strategy backtest — A/B variants, MES + MNQ")
    parser.add_argument(
        "--stage",
        type=str,
        default="",
        help="Run only this stage (e.g. A1, B3). Default = all stages.",
    )
    args = parser.parse_args()

    stages_to_run = (
        [args.stage.upper()] if args.stage.upper() in ALL_STAGES else ALL_STAGES
    )

    all_results: dict = {}

    for inst_key, inst in INSTRUMENTS.items():
        print(f"\nLoading {inst['name']} bars …")
        df    = load_bars(inst)
        n_days = df["date"].nunique()
        print(f"  {len(df):,} bars | {n_days} trading dates")
        print(f"  Range: {df['date'].min()} → {df['date'].max()}")

        print("Building daily aggregates …")
        daily = build_daily_aggregates(df)
        print(f"  {len(daily)} days with RTH data")

        inst_results: dict = {}

        # Track best Sharpe per variant for per-trade log
        best_per_variant: dict[str, tuple[float, str, list[dict]]] = {}

        header = (
            f"  {'Variant':<8}{'Stage':<6}{'N':>5}  {'WR':>7}  "
            f"{'AvgPnL':>9}  {'Sharpe':>7}  {'MaxDD':>10}  Notes"
        )
        sep = "  " + "-" * (len(header) - 2)

        label   = inst["label"]
        days_lbl = f"{n_days} days"
        print(f"\n=== {label} — {days_lbl} ===")
        print(header)
        print(sep)

        for s in stages_to_run:
            variant = s[0]
            trades, daily_pnls = run_stage(df, daily, s, inst)
            m    = compute_metrics(trades, daily_pnls)
            note = STAGE_NOTES.get(s, "")

            if m["n"] < 5:
                sharpe_str = " (N too small)"
                sharpe_val = None
            else:
                sharpe_str = f"{m['sharpe']:>7.2f}"
                sharpe_val = m["sharpe"]

            wr_str  = f"{m['wr']*100:>6.1f}%" if m["n"] > 0 else "   N/A "
            dd_str  = f"${m['max_dd']:>9.2f}"
            avg_str = f"${m['avg_pnl']:>8.2f}"

            if m["n"] < 5:
                print(
                    f"  {variant:<8}{s:<6}{m['n']:>5}  {wr_str}  "
                    f"{avg_str}  {sharpe_str}  {dd_str}  {note}"
                )
            else:
                print(
                    f"  {variant:<8}{s:<6}{m['n']:>5}  {wr_str}  "
                    f"{avg_str}  {sharpe_str}  {dd_str}  {note}"
                )

            inst_results[s] = {**m, "notes": note}

            # Track best per variant
            if sharpe_val is not None:
                prev_best_sharpe = best_per_variant.get(variant, (-1e9, "", []))[0]
                if sharpe_val > prev_best_sharpe:
                    best_per_variant[variant] = (sharpe_val, s, trades)

        all_results[inst_key] = inst_results

        # Per-trade log for best stage of each variant
        for variant in sorted(best_per_variant.keys()):
            sharpe_val, best_stage, best_trades = best_per_variant[variant]
            if best_trades:
                print(
                    f"\n  --- Variant {variant} best stage: {best_stage} "
                    f"(Sharpe={sharpe_val:.2f}) per-trade log ---"
                )
                print_trade_log(best_trades)

    # Save JSON
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "generated": pd.Timestamp.now().isoformat(),
        "stages_run": stages_to_run,
        "results":    all_results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
