"""
Impulse-Retest Backtest — 1-min MES/MNQ
=========================================
Edge hypothesis: a significant above-average-range bar with a volume spike creates
a structural reference level. Waiting for a controlled RETEST of the impulse bar's
midpoint (not entering on the impulse itself) and confirming the original direction
produces a high-probability entry with a tight structural stop.

Regime routing (computed at 9:45 ET using first 6 5-min bars):
  - "trend"    : continuation in impulse direction
  - "negative" : continuation SHORTs only (bear impulses)
  - "chop"     : fade back to session VWAP

Data:
  Primary train : data/processed/mes_1m_bars_cache.h5  key=/bars_1m
  OOS           : data/processed/mnq_2026ytd_1min_eth.h5  key=/bars_1min_eth
  Stress proxy  : data/processed/es_bars_2010_2025.h5  key=/bars_5min

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/impulse_retest_backtest.py
"""
from __future__ import annotations

import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent.parent.parent
MES_PATH = ROOT / "data/processed/mes_1m_bars_cache.h5"
MNQ_PATH = ROOT / "data/processed/mnq_2026ytd_1min_eth.h5"
ES_PATH  = ROOT / "data/processed/es_bars_2010_2025.h5"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MES_PV   = 5.0
MNQ_PV   = 2.0
ES_PV    = 50.0
COST_RT  = 6.25    # round-trip commission + slippage, 1 contract

# Default single-run config (used for Section 1 baseline)
TR_MULT          = 2.0
VOL_MULT         = 1.5
RETEST_DEPTH     = 0.50    # fraction of impulse range from far end (0.50 = midpoint)
TIME_LIMIT_BARS  = 5       # max bars to wait for retest after impulse
STOP_ATR_MULT    = 0.5
TIME_STOP_BARS   = 8
MAX_TRADES_DAY   = 3

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

print("Loading MES 1-min (primary train)...")
with pd.HDFStore(str(MES_PATH), "r") as s:
    df_mes_raw = s["/bars_1m"].set_index("timestamp")
df_mes_raw.index = pd.to_datetime(df_mes_raw.index, utc=True).tz_convert("US/Eastern")
df_mes_raw = df_mes_raw[["open", "high", "low", "close", "volume"]].sort_index()

print("Loading MNQ 2026 YTD (OOS)...")
with pd.HDFStore(str(MNQ_PATH), "r") as s:
    df_mnq_raw = s["/bars_1min_eth"].copy()
if df_mnq_raw.index.tz is None:
    df_mnq_raw.index = df_mnq_raw.index.tz_localize("UTC").tz_convert("US/Eastern")
elif str(df_mnq_raw.index.tz) != "US/Eastern":
    df_mnq_raw.index = df_mnq_raw.index.tz_convert("US/Eastern")
df_mnq_raw = df_mnq_raw[["open", "high", "low", "close", "volume"]].sort_index()

print("Loading ES 5-min (stress proxy, 15yr)...")
with pd.HDFStore(str(ES_PATH), "r") as s:
    df_es = s["/bars_5min"].copy()
df_es = df_es.set_index("timestamp").sort_index()
df_es.index = df_es.index.tz_convert("US/Eastern")
mask_rth_es = (
    ((df_es.index.hour == 9) & (df_es.index.minute >= 30))
    | ((df_es.index.hour > 9) & (df_es.index.hour < 16))
)
df_es = df_es[mask_rth_es][["open", "high", "low", "close", "volume"]].copy()

# ---------------------------------------------------------------------------
# Helpers: RTH filter, resample
# ---------------------------------------------------------------------------

def rth(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        ((df.index.hour == 9) & (df.index.minute >= 30))
        | ((df.index.hour > 9) & (df.index.hour < 16))
    )
    return df[mask].copy()


def resample_5m(df1m: pd.DataFrame) -> pd.DataFrame:
    df = df1m[["open", "high", "low", "close", "volume"]].copy()
    df["_date"]   = df.index.date
    df["_bucket"] = df.index.floor("5min")
    g = df.groupby(["_date", "_bucket"]).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),   close=("close", "last"),
        volume=("volume", "sum"),
    )
    g = g.reset_index(level="_date", drop=True)
    g.index.name = "timestamp"
    return g


# Apply RTH filter and resample
mes_1m_rth = rth(df_mes_raw)
mnq_1m_rth = rth(df_mnq_raw)

mes_5m = resample_5m(mes_1m_rth)
mnq_5m = resample_5m(mnq_1m_rth)

print(f"MES 1m RTH: {len(mes_1m_rth):,} bars  "
      f"({mes_1m_rth.index[0].date()} – {mes_1m_rth.index[-1].date()})")
print(f"MNQ 1m RTH: {len(mnq_1m_rth):,} bars  "
      f"({mnq_1m_rth.index[0].date()} – {mnq_1m_rth.index[-1].date()})")
print(f"ES  5m RTH: {len(df_es):,} bars  "
      f"({df_es.index[0].date()} – {df_es.index[-1].date()})")

# ---------------------------------------------------------------------------
# Rich session stats builder
# Replicates rcaf_orb_combined.py::build_session_stats() but returns dict[date]
# ---------------------------------------------------------------------------

def build_stats_rich(bars_5m: pd.DataFrame) -> dict:
    """
    Returns dict[date -> stat_dict] with all fields needed by classify_regime_full:
      atr_20d, vol_first_30m, rolling_20d_avg_vol_first_30m,
      gap_pct, prev_1d_ret, prev_3d_ret, prev_close, prev_high, prev_low
    """
    sessions = sorted(bars_5m.groupby(bars_5m.index.date), key=lambda x: x[0])
    rows = []
    for date, sess in sessions:
        if len(sess) < 2:
            continue
        rows.append({
            "date":          date,
            "open":          float(sess["open"].iloc[0]),
            "high":          float(sess["high"].max()),
            "low":           float(sess["low"].min()),
            "close":         float(sess["close"].iloc[-1]),
            "vol_first_30m": float(sess["volume"].iloc[:6].sum()),
        })

    result = {}
    prev_close_val = None
    atr_val        = None
    vol_history: list[float] = []

    for i, row in enumerate(rows):
        hi, lo, cl = row["high"], row["low"], row["close"]

        tr = (hi - lo) if prev_close_val is None else max(
            hi - lo, abs(hi - prev_close_val), abs(lo - prev_close_val))
        atr_val = tr if atr_val is None else (atr_val * 13 + tr) / 14

        vol_history.append(row["vol_first_30m"])
        avg_vol_30m = float(np.mean(vol_history[-20:])) if len(vol_history) >= 20 else None

        prev_1d_ret = (row["close"] - rows[i-1]["close"]) / (rows[i-1]["close"] + 1e-8) if i >= 1 else 0.0
        prev_3d_ret = (row["close"] - rows[i-3]["close"]) / (rows[i-3]["close"] + 1e-8) if i >= 3 else 0.0
        gap_pct     = (row["open"] - rows[i-1]["close"]) / (rows[i-1]["close"] + 1e-8) if i >= 1 else 0.0

        result[row["date"]] = {
            "atr_20d":                     atr_val,
            "vol_first_30m":               row["vol_first_30m"],
            "rolling_20d_avg_vol_first_30m": avg_vol_30m,
            "gap_pct":                     gap_pct,
            "prev_1d_ret":                 prev_1d_ret,
            "prev_3d_ret":                 prev_3d_ret,
            "prev_close":  rows[i-1]["close"] if i > 0 else None,
            "prev_high":   rows[i-1]["high"]  if i > 0 else None,
            "prev_low":    rows[i-1]["low"]   if i > 0 else None,
        }
        prev_close_val = cl

    return result


print("\nBuilding session stats...")
mes_stats = build_stats_rich(mes_5m)
mnq_stats = build_stats_rich(mnq_5m)
es_stats  = build_stats_rich(df_es)
print(f"  MES: {len(mes_stats)} sessions  MNQ: {len(mnq_stats)} sessions  ES: {len(es_stats)} sessions")

# Pre-build session DataFrames dict for regime lookup
mes_5m_by_date = {d: g for d, g in mes_5m.groupby(mes_5m.index.date)}
mnq_5m_by_date = {d: g for d, g in mnq_5m.groupby(mnq_5m.index.date)}

# ---------------------------------------------------------------------------
# Cumulative intraday VWAP (resets each session)
# Copied from vwap_strategy_backtest.py::compute_session_vwap()
# ---------------------------------------------------------------------------

def compute_session_vwap(bars: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP from 09:30 each day. Returns pd.Series aligned with bars.index."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    pv      = typical * bars["volume"]
    vwap    = pd.Series(np.nan, index=bars.index)
    for date, grp in bars.groupby(bars.index.date):
        idx     = grp.index
        cum_pv  = pv.loc[idx].cumsum()
        cum_vol = bars["volume"].loc[idx].cumsum()
        vwap.loc[idx] = (cum_pv / cum_vol.replace(0, np.nan)).values
    return vwap


print("Computing session VWAPs...")
mes_vwap = compute_session_vwap(mes_1m_rth)
mnq_vwap = compute_session_vwap(mnq_1m_rth)

# ---------------------------------------------------------------------------
# Regime classifier (adapted from rcaf_orb_combined.py::classify_regime_full)
# Uses first 6 5-min bars (30 min) — no adaptation needed, same function
# ---------------------------------------------------------------------------

def classify_regime(sess_5m: pd.DataFrame, stat: dict) -> str:
    """
    Wrapper around rcaf_orb_combined classify_regime_full logic.
    sess_5m: full 5-min session DataFrame
    stat: rich stat dict from build_stats_rich()
    Returns: "trend" | "chop" | "negative"
    """
    if len(sess_5m) < 7 or stat["atr_20d"] is None:
        return "trend"

    first6 = sess_5m.iloc[:6]
    open_price = float(sess_5m["open"].iloc[0])
    atr_20d    = stat["atr_20d"]

    first_30m_range = float(first6["high"].max() - first6["low"].min())
    range_ratio     = first_30m_range / (atr_20d + 1e-8)

    # 30-min VWAP slope
    tp6   = (first6["high"] + first6["low"] + first6["close"]) / 3
    vol6  = first6["volume"]
    vwap_30m      = float((tp6 * vol6).sum() / (vol6.sum() + 1e-8))
    vwap_slope_30m = (vwap_30m - open_price) / (open_price + 1e-8)

    # Directional efficiency
    sum_tr6 = float((first6["high"] - first6["low"]).sum()) + 1e-8
    close6  = float(first6["close"].iloc[-1])
    de = abs(close6 - open_price) / sum_tr6

    # Relative volume
    avg_vol = stat["rolling_20d_avg_vol_first_30m"]
    rel_vol = (stat["vol_first_30m"] / (avg_vol + 1e-8)) if avg_vol else 1.0

    neg_count = (
        int(stat["gap_pct"] < -0.002)
        + int(stat["prev_1d_ret"] < -0.005)
        + int(stat["prev_3d_ret"] < -0.01)
        + int(range_ratio > 1.3)
        + (int(rel_vol > 1.3) if avg_vol else 0)
        + int(vwap_slope_30m < -0.0003)
    )

    if neg_count >= 2 and de > 0.25:
        return "negative"
    elif de < 0.25 and abs(vwap_slope_30m) < 0.0005:
        return "chop"
    else:
        return "trend"

# ---------------------------------------------------------------------------
# Vectorized impulse detection
# ---------------------------------------------------------------------------

def detect_impulses(bars_1m_rth: pd.DataFrame,
                    tr_mult: float = TR_MULT,
                    vol_mult: float = VOL_MULT,
                    roll_window: int = 30) -> pd.DataFrame:
    """
    Returns DataFrame with same index as bars_1m_rth:
      is_bull, is_bear, bar_range, close_loc, body_frac,
      vol_ratio, range_ratio, imp_high, imp_low, imp_mid
    """
    h = bars_1m_rth["high"]; l = bars_1m_rth["low"]
    o = bars_1m_rth["open"]; c = bars_1m_rth["close"]
    v = bars_1m_rth["volume"]

    bar_range  = h - l
    close_loc  = (c - l) / bar_range.replace(0, np.nan)
    body_frac  = (c - o).abs() / bar_range.replace(0, np.nan)

    roll_rng_med = bar_range.rolling(roll_window, min_periods=10).median()
    roll_vol_med = v.rolling(roll_window, min_periods=10).median()

    range_ratio = bar_range / roll_rng_med.replace(0, np.nan)
    vol_ratio   = v / roll_vol_med.replace(0, np.nan)

    # Time gate: 9:45–12:30 ET
    idx = bars_1m_rth.index
    in_gate = (
        ((idx.hour == 9)  & (idx.minute >= 45))
        | ((idx.hour > 9) & (idx.hour < 12))
        | ((idx.hour == 12) & (idx.minute <= 30))
    )

    quality = (range_ratio > tr_mult) & (vol_ratio > vol_mult) & (body_frac > 0.40) & in_gate
    bull_cand = quality & (close_loc > 0.65)
    bear_cand = quality & (close_loc < 0.35)

    # Cooldown: min 5 bars between impulses, reset at session boundaries
    accepted           = pd.Series(False, index=bars_1m_rth.index)
    last_iloc          = -999
    last_date          = None
    iloc_map           = {ts: i for i, ts in enumerate(bars_1m_rth.index)}

    for ts in bars_1m_rth.index[bull_cand | bear_cand]:
        cur_date = ts.date()
        if cur_date != last_date:
            last_iloc, last_date = -999, cur_date
        iloc_ = iloc_map[ts]
        if iloc_ - last_iloc >= 5:
            accepted[ts] = True
            last_iloc = iloc_

    return pd.DataFrame({
        "is_bull":     accepted & bull_cand,
        "is_bear":     accepted & bear_cand,
        "bar_range":   bar_range,
        "close_loc":   close_loc,
        "body_frac":   body_frac,
        "vol_ratio":   vol_ratio,
        "range_ratio": range_ratio,
        "imp_high":    h,
        "imp_low":     l,
        "imp_mid":     (h + l) / 2.0,
    }, index=bars_1m_rth.index)

# ---------------------------------------------------------------------------
# Event extraction: retest + confirmation entry
# ---------------------------------------------------------------------------

def extract_events(bars_1m_rth: pd.DataFrame,
                   imp_df: pd.DataFrame,
                   session_vwap: pd.Series,
                   stats_rich: dict,
                   bars_5m_by_date: dict,
                   retest_depth: float = RETEST_DEPTH,
                   time_limit_bars: int = TIME_LIMIT_BARS) -> list:
    """
    For each impulse bar, find a retest + confirmation entry.
    Returns list of event dicts (same structure as oiar_backtest.py events).
    """
    events = []
    session_dates = sorted(set(bars_1m_rth.index.date))

    for date in session_dates:
        stat    = stats_rich.get(date)
        sess_5m = bars_5m_by_date.get(date)
        if stat is None or sess_5m is None or stat["atr_20d"] is None:
            continue

        # Regime from 5-min first 30 min
        regime = classify_regime(sess_5m, stat)

        bar_atr = stat["atr_20d"] / 13.0

        mask_day  = bars_1m_rth.index.date == date
        sess_1m   = bars_1m_rth[mask_day]
        sess_imp  = imp_df[mask_day]
        sess_v    = session_vwap[mask_day]

        if len(sess_1m) < 16:
            continue

        bars_list = list(sess_1m.iterrows())
        n         = len(bars_list)
        imp_arr   = sess_imp[["is_bull", "is_bear", "imp_high", "imp_low",
                               "imp_mid", "range_ratio", "vol_ratio"]].values
        # columns: 0=is_bull, 1=is_bear, 2=imp_high, 3=imp_low, 4=imp_mid, 5=range_ratio, 6=vol_ratio

        trades_today = 0

        for imp_i in range(n):
            if trades_today >= MAX_TRADES_DAY:
                break

            is_bull = bool(imp_arr[imp_i, 0])
            is_bear = bool(imp_arr[imp_i, 1])
            if not (is_bull or is_bear):
                continue

            imp_high = float(imp_arr[imp_i, 2])
            imp_low  = float(imp_arr[imp_i, 3])
            imp_mid  = float(imp_arr[imp_i, 4])
            imp_rng  = imp_high - imp_low
            if imp_rng < 1e-6:
                continue

            imp_ts = bars_list[imp_i][0]

            # ----------------------------------------------------------------
            # Regime routing
            # ----------------------------------------------------------------
            if regime in ("trend", "negative"):
                mode = "continuation"
                if is_bull and regime == "negative":
                    continue   # negative regime: only bear impulses → SHORT
                trade_dir = +1 if is_bull else -1
            else:   # chop
                mode = "fade"
                trade_dir = -1 if is_bull else +1   # fade the impulse

            # ----------------------------------------------------------------
            # Retest threshold
            # retest_depth = fraction of impulse range from the FAR end
            # LONG: must pull back to  imp_high - retest_depth * imp_rng
            # SHORT: must pull back to imp_low  + retest_depth * imp_rng
            # ----------------------------------------------------------------
            if trade_dir == +1:
                rt_thresh = imp_high - retest_depth * imp_rng
            else:
                rt_thresh = imp_low + retest_depth * imp_rng

            # ----------------------------------------------------------------
            # Scan for retest + confirmation (up to time_limit_bars + 1)
            # ----------------------------------------------------------------
            entry_px     = None
            conf_i       = -1
            retest_found = False
            cutoff_i     = min(imp_i + time_limit_bars + 2, n)

            for scan_i in range(imp_i + 1, cutoff_i):
                scan_ts, scan_bar = bars_list[scan_i]
                # Hard time cutoff: no new entries after 12:30 ET
                if scan_ts.hour > 12 or (scan_ts.hour == 12 and scan_ts.minute > 30):
                    break

                sl = float(scan_bar["low"])
                sh = float(scan_bar["high"])
                sc = float(scan_bar["close"])

                if not retest_found:
                    # Check if this bar touches the retest threshold
                    hit = (sl <= rt_thresh) if trade_dir == +1 else (sh >= rt_thresh)
                    if hit:
                        retest_found = True
                        # Check immediate confirmation (close recovers past mid on same bar)
                        if (trade_dir == +1 and sc > imp_mid) or \
                           (trade_dir == -1 and sc < imp_mid):
                            entry_px = sc
                            conf_i   = scan_i
                            break
                        # else wait for next bar
                else:
                    # Retest already found — check confirmation
                    if (trade_dir == +1 and sc > imp_mid) or \
                       (trade_dir == -1 and sc < imp_mid):
                        entry_px = sc
                        conf_i   = scan_i
                        break
                    # Timeout: exceeded time_limit_bars since impulse
                    if (scan_i - imp_i) > time_limit_bars:
                        break

            if entry_px is None or conf_i < 0:
                continue

            conf_ts = bars_list[conf_i][0]

            # VWAP at confirmation bar
            vwap_at_entry = float(sess_v.iloc[conf_i]) if conf_i < len(sess_v) else np.nan

            # ----------------------------------------------------------------
            # Target price
            # ----------------------------------------------------------------
            if mode == "continuation":
                tp_ref = entry_px + imp_rng if trade_dir == +1 else entry_px - imp_rng
            else:
                # Fade: target = session VWAP at entry (locked snapshot)
                tp_ref = vwap_at_entry if not np.isnan(vwap_at_entry) else \
                         (entry_px + imp_rng * 0.5 if trade_dir == +1 else entry_px - imp_rng * 0.5)

            # ----------------------------------------------------------------
            # Subsequent bars (up to 15) with VWAP for invalidation exit
            # ----------------------------------------------------------------
            subsequent = []
            for j in range(conf_i + 1, min(conf_i + 16, n)):
                _, sb = bars_list[j]
                v_j = float(sess_v.iloc[j]) if j < len(sess_v) else vwap_at_entry
                subsequent.append({
                    "h": float(sb["high"]), "l": float(sb["low"]),
                    "c": float(sb["close"]), "vwap": v_j,
                })

            events.append({
                "date":          str(date),
                "impulse_ts":    str(imp_ts),
                "entry_ts":      str(conf_ts),
                "regime":        regime,
                "mode":          mode,
                "direction":     trade_dir,
                "imp_high":      imp_high,
                "imp_low":       imp_low,
                "imp_mid":       imp_mid,
                "bar_range":     imp_rng,
                "bar_atr":       bar_atr,
                "range_ratio":   float(imp_arr[imp_i, 5]),
                "vol_ratio":     float(imp_arr[imp_i, 6]),
                "entry_px":      entry_px,
                "stop_ref":      imp_low if trade_dir == +1 else imp_high,
                "tp_ref":        tp_ref,
                "vwap_at_entry": vwap_at_entry,
                "subsequent":    subsequent,
            })
            trades_today += 1

    return events

# ---------------------------------------------------------------------------
# Exit simulator
# ---------------------------------------------------------------------------

def simulate(events: list,
             stop_atr_mult: float = STOP_ATR_MULT,
             time_stop_bars: int  = TIME_STOP_BARS,
             mode_filter: str     = "both",       # "continuation", "fade", "both"
             regime_filter: str   = "all",         # "trend", "negative", "chop", "all"
             direction_filter: str = "both",       # "long", "short", "both"
             use_vwap_exit: bool  = True,
             point_value: float   = MES_PV) -> dict:
    trades = []

    for ev in events:
        if mode_filter != "both" and ev["mode"] != mode_filter:
            continue
        if regime_filter != "all" and ev["regime"] != regime_filter:
            continue
        if direction_filter == "long"  and ev["direction"] != +1:
            continue
        if direction_filter == "short" and ev["direction"] != -1:
            continue

        entry  = ev["entry_px"]
        dir_   = ev["direction"]
        ba     = ev["bar_atr"]
        stop   = (ev["stop_ref"] - stop_atr_mult * ba) if dir_ == +1 \
                 else (ev["stop_ref"] + stop_atr_mult * ba)
        tp     = ev["tp_ref"]
        risk   = (entry - stop) * dir_
        if risk <= 0:
            continue

        pnl = reason = None
        for i_b, fb in enumerate(ev["subsequent"][:time_stop_bars]):
            fh, fl, fc = fb["h"], fb["l"], fb["c"]
            fv = fb.get("vwap", np.nan)

            # VWAP invalidation: price closes past VWAP against direction
            if use_vwap_exit and not np.isnan(fv):
                if dir_ == +1 and fc < fv:
                    pnl = (fc - entry) * point_value - COST_RT
                    reason = "vwap_exit"; break
                elif dir_ == -1 and fc > fv:
                    pnl = (entry - fc) * point_value - COST_RT
                    reason = "vwap_exit"; break

            # Stop
            if dir_ == +1 and fl <= stop:
                pnl = (stop - entry) * point_value - COST_RT; reason = "stop"; break
            if dir_ == -1 and fh >= stop:
                pnl = (entry - stop) * point_value - COST_RT; reason = "stop"; break

            # Target
            if dir_ == +1 and fh >= tp:
                pnl = (tp - entry) * point_value - COST_RT; reason = "tp"; break
            if dir_ == -1 and fl <= tp:
                pnl = (entry - tp) * point_value - COST_RT; reason = "tp"; break

            # Time stop (last bar in window)
            if i_b == min(time_stop_bars, len(ev["subsequent"])) - 1:
                pnl = ((fc - entry) if dir_ == +1 else (entry - fc)) * point_value - COST_RT
                reason = "time"

        if pnl is None and ev["subsequent"]:
            last = ev["subsequent"][-1]
            pnl  = ((last["c"] - entry) if dir_ == +1 else (entry - last["c"])) * point_value - COST_RT
            reason = "time"

        if pnl is not None:
            trades.append({
                "date":   ev["date"],
                "pnl":    round(pnl, 2),
                "exit":   reason,
                "mode":   ev["mode"],
                "regime": ev["regime"],
                "dir":    "long" if dir_ == +1 else "short",
            })

    return _stats(trades)


def _stats(trades: list) -> dict:
    if not trades:
        return {"N": 0, "WR": 0.0, "E": 0.0, "PnL": 0.0, "Sharpe": 0.0,
                "MaxDD": 0.0, "avg_tpd": 0.0, "exit_dist": {}, "stress": {}}

    p  = np.array([t["pnl"] for t in trades])
    WR = float((p > 0).mean())
    E  = float(p.mean())

    by_day: dict[str, float] = {}
    for t in trades:
        by_day[t["date"]] = by_day.get(t["date"], 0.0) + t["pnl"]
    daily  = np.array(list(by_day.values()))
    sharpe = float((daily.mean() / (daily.std() + 1e-8)) * np.sqrt(252)) if len(daily) > 1 else 0.0
    cum    = np.cumsum(p)
    max_dd = float((cum - np.maximum.accumulate(cum)).min())

    tpd_arr = np.array([sum(1 for t in trades if t["date"] == d) for d in by_day])
    avg_tpd = float(tpd_arr.mean()) if len(tpd_arr) else 0.0

    exit_dist: dict[str, int] = {}
    for t in trades:
        exit_dist[t["exit"]] = exit_dist.get(t["exit"], 0) + 1

    stress: dict[str, float] = {}
    for period, (y0, y1) in [("2018-2019", ("2018", "2019")),
                               ("2020",      ("2020", "2020")),
                               ("2022",      ("2022", "2022"))]:
        sub = [t for t in trades if y0 <= t["date"][:4] <= y1]
        stress[period] = round(float(sum(t["pnl"] for t in sub)), 2) if sub else None

    return {"N": len(p), "WR": round(WR, 4), "E": round(E, 2),
            "PnL": round(float(p.sum()), 2), "Sharpe": round(sharpe, 3),
            "MaxDD": round(max_dd, 2), "avg_tpd": round(avg_tpd, 2),
            "exit_dist": exit_dist, "stress": stress}


def print_s(label: str, s: dict, extra: str = "") -> None:
    if s.get("N", 0) == 0:
        print(f"  {label:48s} | NO TRADES")
        return
    print(f"  {label:48s} | N={s['N']:5d} | WR={s['WR']:.1%} | E=${s['E']:+6.0f} "
          f"| Sharpe={s['Sharpe']:+6.2f} | MaxDD=${s['MaxDD']:,.0f} "
          f"| {s['avg_tpd']:.1f}t/d{extra}")


# ---------------------------------------------------------------------------
# ORB simple runner for correlation (adapted from oiar_backtest.py)
# ---------------------------------------------------------------------------

def run_orb_simple(bars_5m: pd.DataFrame, stats_d: dict, point_value: float) -> list:
    trades = []
    session_dates = sorted(set(bars_5m.index.date))
    pv_flag: dict = {}
    for i, d in enumerate(session_dates):
        if i == 0:
            pv_flag[d] = False; continue
        prev = session_dates[i-1]
        ps = bars_5m[bars_5m.index.date == prev]
        if len(ps) < 2:
            pv_flag[d] = False; continue
        tp_ = (ps["high"] + ps["low"] + ps["close"]) / 3
        pvwap = float((tp_ * ps["volume"]).sum() / (ps["volume"].sum() + 1e-8))
        pv_flag[d] = float(ps["close"].iloc[-1]) > pvwap

    for d in session_dates:
        stat = stats_d.get(d)
        if stat is None or stat["atr_20d"] is None:
            continue
        if not pv_flag.get(d, False):
            continue
        sess = bars_5m[bars_5m.index.date == d].copy()
        if len(sess) <= 7:
            continue
        or_h = float(sess.iloc[:7]["high"].max())
        or_l = float(sess.iloc[:7]["low"].min())
        if (or_h - or_l) < 0.3 * stat["atr_20d"]:
            continue
        in_t = False; ep = sp = tp2 = 0.0; eb = 0
        for i_b, (ts_b, bar) in enumerate(sess.iterrows()):
            if i_b < 7:
                continue
            if ts_b.hour >= 12 and not in_t:
                break
            if in_t:
                if float(bar["high"]) >= tp2:
                    pnl = (tp2 - ep) * point_value - COST_RT
                    trades.append({"date": str(d), "pnl": round(pnl, 2)}); in_t = False
                elif float(bar["low"]) <= sp:
                    pnl = (sp - ep) * point_value - COST_RT
                    trades.append({"date": str(d), "pnl": round(pnl, 2)}); in_t = False
                elif (i_b - eb) >= 24 or i_b == len(sess) - 1:
                    pnl = (float(bar["close"]) - ep) * point_value - COST_RT
                    trades.append({"date": str(d), "pnl": round(pnl, 2)}); in_t = False
                continue
            if float(bar["close"]) > or_h:
                ep = float(bar["close"]); sp = ep - 1.5 * stat["atr_20d"]
                tp2 = ep + 2.0 * stat["atr_20d"]; in_t = True; eb = i_b
    return trades

# ---------------------------------------------------------------------------
# ES 5-min impulse detection (stress proxy — roll_window=10, wider time gate)
# ---------------------------------------------------------------------------

def detect_impulses_5m(bars_5m: pd.DataFrame,
                       tr_mult: float = TR_MULT,
                       vol_mult: float = VOL_MULT,
                       roll_window: int = 10) -> pd.DataFrame:
    """Same logic as detect_impulses() but tuned for 5-min bars."""
    h = bars_5m["high"]; l = bars_5m["low"]
    o = bars_5m["open"]; c = bars_5m["close"]
    v = bars_5m["volume"]

    bar_range  = h - l
    close_loc  = (c - l) / bar_range.replace(0, np.nan)
    body_frac  = (c - o).abs() / bar_range.replace(0, np.nan)

    roll_rng_med = bar_range.rolling(roll_window, min_periods=5).median()
    roll_vol_med = v.rolling(roll_window, min_periods=5).median()
    range_ratio  = bar_range / roll_rng_med.replace(0, np.nan)
    vol_ratio    = v / roll_vol_med.replace(0, np.nan)

    idx = bars_5m.index
    # Time gate: 10:00–13:00 (avoid first-30m ORB overlap)
    in_gate = (idx.hour >= 10) & (idx.hour < 13)

    quality   = (range_ratio > tr_mult) & (vol_ratio > vol_mult) & (body_frac > 0.40) & in_gate
    bull_cand = quality & (close_loc > 0.65)
    bear_cand = quality & (close_loc < 0.35)

    accepted  = pd.Series(False, index=bars_5m.index)
    last_iloc = -999; last_date = None
    iloc_map  = {ts: i for i, ts in enumerate(bars_5m.index)}

    for ts in bars_5m.index[bull_cand | bear_cand]:
        cur_date = ts.date()
        if cur_date != last_date:
            last_iloc, last_date = -999, cur_date
        iloc_ = iloc_map[ts]
        if iloc_ - last_iloc >= 3:   # 5-min bars: 3-bar cooldown = 15 min
            accepted[ts] = True
            last_iloc = iloc_

    return pd.DataFrame({
        "is_bull": accepted & bull_cand, "is_bear": accepted & bear_cand,
        "bar_range": bar_range, "close_loc": close_loc, "body_frac": body_frac,
        "vol_ratio": vol_ratio, "range_ratio": range_ratio,
        "imp_high": h, "imp_low": l, "imp_mid": (h + l) / 2.0,
    }, index=bars_5m.index)


def extract_events_5m(bars_5m: pd.DataFrame,
                      imp_df: pd.DataFrame,
                      stats_rich: dict,
                      retest_depth: float = RETEST_DEPTH,
                      time_limit_bars: int = 3) -> list:
    """Impulse-retest events on 5-min bars for ES stress test (no regime routing)."""
    events = []
    session_dates = sorted(set(bars_5m.index.date))

    for date in session_dates:
        stat = stats_rich.get(date)
        if stat is None or stat["atr_20d"] is None:
            continue
        bar_atr = stat["atr_20d"] / 13.0

        mask    = bars_5m.index.date == date
        sess    = bars_5m[mask]
        simp    = imp_df[mask]
        if len(sess) < 8:
            continue

        bars_list = list(sess.iterrows())
        n         = len(bars_list)
        imp_vals  = simp[["is_bull", "is_bear", "imp_high", "imp_low",
                           "imp_mid", "range_ratio", "vol_ratio"]].values
        trades_today = 0

        for imp_i in range(n):
            if trades_today >= MAX_TRADES_DAY:
                break
            is_bull = bool(imp_vals[imp_i, 0])
            is_bear = bool(imp_vals[imp_i, 1])
            if not (is_bull or is_bear):
                continue

            imp_high = float(imp_vals[imp_i, 2])
            imp_low  = float(imp_vals[imp_i, 3])
            imp_mid  = float(imp_vals[imp_i, 4])
            imp_rng  = imp_high - imp_low
            if imp_rng < 1e-6:
                continue
            imp_ts = bars_list[imp_i][0]
            trade_dir = +1 if is_bull else -1

            rt_thresh = (imp_high - retest_depth * imp_rng) if trade_dir == +1 \
                        else (imp_low + retest_depth * imp_rng)

            entry_px = None; conf_i = -1; retest_found = False
            for scan_i in range(imp_i + 1, min(imp_i + time_limit_bars + 2, n)):
                scan_ts, scan_bar = bars_list[scan_i]
                if scan_ts.hour >= 13:
                    break
                sl = float(scan_bar["low"]); sh = float(scan_bar["high"])
                sc = float(scan_bar["close"])
                if not retest_found:
                    hit = (sl <= rt_thresh) if trade_dir == +1 else (sh >= rt_thresh)
                    if hit:
                        retest_found = True
                        if (trade_dir == +1 and sc > imp_mid) or (trade_dir == -1 and sc < imp_mid):
                            entry_px = sc; conf_i = scan_i; break
                else:
                    if (trade_dir == +1 and sc > imp_mid) or (trade_dir == -1 and sc < imp_mid):
                        entry_px = sc; conf_i = scan_i; break
                    if (scan_i - imp_i) > time_limit_bars:
                        break

            if entry_px is None:
                continue

            tp_ref = entry_px + imp_rng if trade_dir == +1 else entry_px - imp_rng
            subsequent = []
            for j in range(conf_i + 1, min(conf_i + 10, n)):
                _, sb = bars_list[j]
                subsequent.append({"h": float(sb["high"]), "l": float(sb["low"]),
                                   "c": float(sb["close"])})

            events.append({
                "date": str(date), "impulse_ts": str(imp_ts),
                "regime": "all", "mode": "continuation", "direction": trade_dir,
                "imp_high": imp_high, "imp_low": imp_low, "imp_mid": imp_mid,
                "bar_range": imp_rng, "bar_atr": bar_atr,
                "range_ratio": float(imp_vals[imp_i, 5]),
                "vol_ratio":   float(imp_vals[imp_i, 6]),
                "entry_px": entry_px,
                "stop_ref": imp_low if trade_dir == +1 else imp_high,
                "tp_ref":   tp_ref,
                "vwap_at_entry": np.nan,
                "subsequent": subsequent,
            })
            trades_today += 1

    return events

# ===========================================================================
# SECTION 1: Raw signal baseline (MES, default config)
# ===========================================================================

print("\n" + "=" * 80)
print("SECTION 1: RAW SIGNAL BASELINE (MES 1-min, default config)")
print(f"  TR_MULT={TR_MULT}  VOL_MULT={VOL_MULT}  RETEST_DEPTH={RETEST_DEPTH}  "
      f"TIME_LIMIT={TIME_LIMIT_BARS}b  STOP={STOP_ATR_MULT}×ATR  TS={TIME_STOP_BARS}b")
print("=" * 80)

print("\nDetecting impulses (MES)...")
mes_imp = detect_impulses(mes_1m_rth)
n_bull = int(mes_imp["is_bull"].sum())
n_bear = int(mes_imp["is_bear"].sum())
print(f"  Bull impulses: {n_bull}  Bear impulses: {n_bear}  "
      f"Total: {n_bull + n_bear}  across {len(mes_stats)} sessions "
      f"({(n_bull + n_bear) / max(1, len(mes_stats)):.1f}/session)")

print("Extracting retest events (MES)...")
mes_events = extract_events(mes_1m_rth, mes_imp, mes_vwap,
                             mes_stats, mes_5m_by_date)
print(f"  Events: {len(mes_events)}  "
      f"({len(mes_events) / max(1, len(mes_stats)):.1f}/session)")

# Regime distribution
from collections import Counter
regime_counts = Counter(e["regime"] for e in mes_events)
mode_counts   = Counter(e["mode"]   for e in mes_events)
dir_counts    = Counter(e["direction"] for e in mes_events)
print(f"  Regime: {dict(regime_counts)}")
print(f"  Mode:   {dict(mode_counts)}")
print(f"  Dir:    {{+1: {dir_counts[1]}, -1: {dir_counts[-1]}}}")

s_all = simulate(mes_events)
print(f"\n  {'Config':48s}  N     WR      E      Sharpe  MaxDD      t/d")
print("  " + "-" * 80)
print_s("All events (both modes, both dirs)", s_all)
print_s("Continuation only", simulate(mes_events, mode_filter="continuation"))
print_s("Fade only",         simulate(mes_events, mode_filter="fade"))
print_s("Long only",         simulate(mes_events, direction_filter="long"))
print_s("Short only",        simulate(mes_events, direction_filter="short"))
print_s("Negative regime",   simulate(mes_events, regime_filter="negative"))
print_s("Chop regime",       simulate(mes_events, regime_filter="chop"))
print_s("Trend regime",      simulate(mes_events, regime_filter="trend"))

# Exit distribution
s_all_trades = s_all["exit_dist"]
total_t = sum(s_all_trades.values())
if total_t:
    print(f"\n  Exit distribution ({total_t} trades):")
    for reason, cnt in sorted(s_all_trades.items(), key=lambda x: -x[1]):
        print(f"    {reason:15s}: {cnt:4d} ({cnt/total_t:.1%})")

# ===========================================================================
# SECTION 2: Parameter sweep (MES)
# ===========================================================================

print("\n" + "=" * 80)
print("SECTION 2: PARAMETER SWEEP (MES 1-min)")
print("=" * 80)

sweep_grid = {
    "tr_mult":        [1.6, 2.0, 2.4],
    "vol_mult":       [1.3, 1.5, 2.0],
    "retest_depth":   [0.33, 0.50, 0.67],
    "time_limit":     [3, 5],
}

print(f"\n  Sweeping {3*3*3*2} configs...")
sweep_results = []

for tr, vm, rd, tl in product(sweep_grid["tr_mult"], sweep_grid["vol_mult"],
                               sweep_grid["retest_depth"], sweep_grid["time_limit"]):
    imp_sw  = detect_impulses(mes_1m_rth, tr_mult=tr, vol_mult=vm)
    evts_sw = extract_events(mes_1m_rth, imp_sw, mes_vwap,
                              mes_stats, mes_5m_by_date,
                              retest_depth=rd, time_limit_bars=tl)
    s = simulate(evts_sw)
    if s["N"] < 50 or s["Sharpe"] < 1.0:
        continue

    neg_s  = simulate(evts_sw, regime_filter="negative")
    chop_s = simulate(evts_sw, regime_filter="chop")
    trend_s = simulate(evts_sw, regime_filter="trend")

    sweep_results.append({
        "tr": tr, "vm": vm, "rd": rd, "tl": tl,
        **s,
        "neg_pnl":   neg_s["PnL"],   "neg_wr":   neg_s["WR"],   "neg_n":   neg_s["N"],
        "chop_pnl":  chop_s["PnL"],  "chop_wr":  chop_s["WR"],  "chop_n":  chop_s["N"],
        "trend_pnl": trend_s["PnL"], "trend_wr": trend_s["WR"], "trend_n": trend_s["N"],
    })

sweep_results.sort(key=lambda x: x["Sharpe"], reverse=True)

hdr = f"  {'Config':38s}  {'N':>5}  {'WR':>6}  {'E':>6}  {'Sharpe':>7}  {'MaxDD':>8}  {'t/d':>4}  Reg(neg/chop/trend)"
print(hdr)
print("  " + "-" * 115)
for r in sweep_results[:25]:
    cfg = f"tr={r['tr']:.1f}x vol={r['vm']:.1f}x rt={r['rd']:.2f} tl={r['tl']}b"
    all_stress_pass = (
        r["stress"].get("2018-2019") is not None and r["stress"]["2018-2019"] >= 0 and
        r["stress"].get("2020") is not None and r["stress"]["2020"] > 0 and
        r["stress"].get("2022") is not None and r["stress"]["2022"] > 0
    )
    star = " ★" if all_stress_pass and r["neg_pnl"] > 0 and r["chop_pnl"] > 0 else ""
    print(f"  {cfg:38s}  {r['N']:5d}  {r['WR']:.1%}  ${r['E']:5.0f}  {r['Sharpe']:7.3f}  "
          f"${r['MaxDD']:8,.0f}  {r['avg_tpd']:4.1f}  "
          f"${r['neg_pnl']:+,.0f}/${r['chop_pnl']:+,.0f}/${r['trend_pnl']:+,.0f}{star}")

best = sweep_results[0] if sweep_results else None

# ===========================================================================
# SECTION 3: Year-by-year (best config, MES)
# ===========================================================================

print("\n" + "=" * 80)
print("SECTION 3: YEAR-BY-YEAR BREAKDOWN (best Sharpe config, MES)")
print("=" * 80)

if best:
    print(f"  Config: tr={best['tr']:.1f}x vol={best['vm']:.1f}x "
          f"rt={best['rd']:.2f} tl={best['tl']}b")
    best_imp  = detect_impulses(mes_1m_rth, tr_mult=best["tr"], vol_mult=best["vm"])
    best_evts = extract_events(mes_1m_rth, best_imp, mes_vwap,
                                mes_stats, mes_5m_by_date,
                                retest_depth=best["rd"], time_limit_bars=best["tl"])

    years = sorted({e["date"][:4] for e in best_evts})
    for yr in years:
        yr_ev = [e for e in best_evts if e["date"][:4] == yr]
        s_yr  = simulate(yr_ev)
        if s_yr["N"] == 0:
            print(f"    {yr}  | NO TRADES")
        else:
            print(f"    {yr}  | N={s_yr['N']:4d} | WR={s_yr['WR']:.1%} | "
                  f"E=${s_yr['E']:+6.0f} | PnL=${s_yr['PnL']:+8,.0f} | "
                  f"Sharpe={s_yr['Sharpe']:+6.2f} | MaxDD=${s_yr['MaxDD']:,.0f}")
else:
    print("  No qualifying configs found.")
    best_evts = mes_events

# ===========================================================================
# SECTION 4: OOS MNQ transfer
# ===========================================================================

print("\n" + "=" * 80)
print("SECTION 4: OOS MNQ JAN–MAR 2026")
print("=" * 80)

mnq_imp  = detect_impulses(mnq_1m_rth,
                            tr_mult=best["tr"] if best else TR_MULT,
                            vol_mult=best["vm"] if best else VOL_MULT)
mnq_events = extract_events(mnq_1m_rth, mnq_imp, mnq_vwap,
                              mnq_stats, mnq_5m_by_date,
                              retest_depth=best["rd"] if best else RETEST_DEPTH,
                              time_limit_bars=best["tl"] if best else TIME_LIMIT_BARS)
print(f"  MNQ events: {len(mnq_events)}  "
      f"({len(mnq_events) / max(1, len(mnq_stats)):.1f}/session)")

mnq_s = simulate(mnq_events, point_value=MNQ_PV)
print_s("MNQ OOS all",          mnq_s)
print_s("  continuation",       simulate(mnq_events, mode_filter="continuation", point_value=MNQ_PV))
print_s("  fade",               simulate(mnq_events, mode_filter="fade", point_value=MNQ_PV))
print_s("  negative regime",    simulate(mnq_events, regime_filter="negative", point_value=MNQ_PV))
print_s("  chop regime",        simulate(mnq_events, regime_filter="chop", point_value=MNQ_PV))

# ORB vs impulse correlation on MES
print("\nORB vs Impulse-Retest daily PnL correlation (MES)...")
orb_trades = run_orb_simple(mes_5m, mes_stats, MES_PV)

def daily_pnl_dict(trades: list) -> dict:
    d: dict[str, float] = {}
    for t in trades:
        d[t["date"]] = d.get(t["date"], 0.0) + t["pnl"]
    return d

orb_d  = daily_pnl_dict(orb_trades)
imp_d  = daily_pnl_dict([t for t in (
    lambda evts: [{
        "date": ev["date"],
        "pnl": simulate([ev])["PnL"]
    } for ev in evts]
)(best_evts) if t["pnl"] != 0])

all_dates = sorted(set(orb_d) | set(imp_d))
if len(all_dates) > 5:
    corr = float(np.corrcoef(
        [orb_d.get(d, 0.0) for d in all_dates],
        [imp_d.get(d, 0.0) for d in all_dates],
    )[0, 1])
    print(f"  ORB vs Impulse-Retest correlation: {corr:.3f}")
else:
    corr = 0.0
    print("  Insufficient data for correlation")

# ===========================================================================
# SECTION 5: ES 5-min stress proxy (2010–2025)
# ===========================================================================

print("\n" + "=" * 80)
print("SECTION 5: ES 5-MIN STRESS PROXY (2010–2025)")
print("=" * 80)

es_imp_5m   = detect_impulses_5m(df_es,
                                  tr_mult=best["tr"] if best else TR_MULT,
                                  vol_mult=best["vm"] if best else VOL_MULT)
es_events_5m = extract_events_5m(df_es, es_imp_5m, es_stats,
                                  retest_depth=best["rd"] if best else RETEST_DEPTH,
                                  time_limit_bars=3)
print(f"  ES events: {len(es_events_5m)} across {len(es_stats)} sessions "
      f"({len(es_events_5m)/max(1,len(es_stats)):.1f}/session)")

s_es = simulate(es_events_5m, use_vwap_exit=False, point_value=ES_PV)
print_s("ES 5m all", s_es)

# Stress windows
for period in ["2018-2019", "2020", "2022"]:
    val = s_es["stress"].get(period)
    flag = "✅" if (val is not None and val > 0) else "❌"
    print(f"  {period:12s}: ${val:+,.0f}  {flag}" if val is not None else f"  {period:12s}: N/A")

# Year-by-year ES
print()
years_es = sorted({e["date"][:4] for e in es_events_5m})
for yr in years_es:
    yr_ev = [e for e in es_events_5m if e["date"][:4] == yr]
    s_yr  = simulate(yr_ev, use_vwap_exit=False, point_value=ES_PV)
    if s_yr["N"] > 0:
        print(f"    {yr}  | N={s_yr['N']:4d} | WR={s_yr['WR']:.1%} | "
              f"E=${s_yr['E']:+5.0f} | PnL=${s_yr['PnL']:+7,.0f} | Sharpe={s_yr['Sharpe']:+5.2f}")

# ===========================================================================
# SECTION 6: Acceptance criteria
# ===========================================================================

print("\n" + "=" * 80)
print("SECTION 6: ACCEPTANCE CRITERIA")
print("=" * 80)

n_pass = 0

def chk(name: str, passed: bool, detail: str = "") -> None:
    global n_pass
    if passed:
        n_pass += 1
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}]  {name}" + (f"  ({detail})" if detail else ""))

ref = simulate(best_evts) if best else _stats([])
ref_neg  = simulate(best_evts, regime_filter="negative") if best else _stats([])
ref_chop = simulate(best_evts, regime_filter="chop")     if best else _stats([])

chk("Avg trades/day ≥ 3",
    ref.get("avg_tpd", 0) >= 3.0,
    f"{ref.get('avg_tpd', 0):.1f} t/day")
chk("WR > 48%",
    ref.get("WR", 0) > 0.48,
    f"WR={ref.get('WR', 0):.1%}")
chk("Sharpe > 2.0",
    ref.get("Sharpe", 0) > 2.0,
    f"Sharpe={ref.get('Sharpe', 0):.3f}")
chk("Chop regime PnL > 0",
    ref_chop.get("PnL", -1) > 0,
    f"${ref_chop.get('PnL', 0):+,.0f}")
chk("Negative regime PnL > 0",
    ref_neg.get("PnL", -1) > 0,
    f"${ref_neg.get('PnL', 0):+,.0f}")
chk("ORB correlation < 0.35",
    abs(corr) < 0.35,
    f"corr={corr:.3f}")
chk("ES 2020 stress positive",
    (s_es.get("stress", {}).get("2020") or -1) > 0,
    f"${(s_es.get('stress', {}).get('2020') or 0):+,.0f}")
chk("ES 2022 stress positive",
    (s_es.get("stress", {}).get("2022") or -1) > 0,
    f"${(s_es.get('stress', {}).get('2022') or 0):+,.0f}")

if n_pass >= 7:
    verdict = "STRONG — deploy to live"
elif n_pass >= 5:
    verdict = "MODERATE — promising, refine parameters"
elif n_pass >= 3:
    verdict = "WEAK — partial signal, investigate failures"
else:
    verdict = "KILL — no viable edge at 1-min impulse family"

print(f"\n  Passed {n_pass}/8 → {verdict}")

# ===========================================================================
# Save results
# ===========================================================================

out = {
    "best_config": {k: v for k, v in (best or {}).items()
                    if k not in ("exit_dist", "stress", "neg_pnl", "chop_pnl",
                                 "trend_pnl", "neg_wr", "chop_wr", "trend_wr",
                                 "neg_n", "chop_n", "trend_n")},
    "mes_baseline": {k: v for k, v in ref.items() if k != "exit_dist"},
    "mnq_oos": {k: v for k, v in mnq_s.items() if k != "exit_dist"},
    "es_stress": {k: v for k, v in s_es.items() if k != "exit_dist"},
    "orb_correlation": round(corr, 3),
    "criteria_passed": n_pass,
    "verdict": verdict,
    "top_configs": [
        {k: v for k, v in r.items() if k not in ("exit_dist", "stress")}
        for r in sweep_results[:10]
    ],
}
out_path = ROOT / "rule_based_v1/diagnostics/impulse_retest_results.json"
out_path.write_text(json.dumps(out, indent=2, default=str))
print(f"\nSaved → {out_path}")
