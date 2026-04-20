"""
RCAF + ORB Combined Portfolio Backtest
Tests whether RCAF chop+neg SHORT is additive to ORB.
Regime-complementary hypothesis: ORB fires on breakout days, RCAF fires on chop/negative days.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

with pd.HDFStore(str(ROOT / "data/processed/mes_1m_bars_cache.h5"), "r") as s:
    df_train_raw = s["/bars_1m"].set_index("timestamp")
df_train_raw.index = pd.to_datetime(df_train_raw.index, utc=True).tz_convert("US/Eastern")
df_train_raw = df_train_raw.sort_index()

with pd.HDFStore(str(ROOT / "data/processed/jan_feb_2026_oos_test_1m.h5"), "r") as s:
    df_oos_raw = s["/bars_1min"].copy()
df_oos_raw.index = pd.to_datetime(df_oos_raw.index, utc=True).tz_convert("US/Eastern")
df_oos_raw = df_oos_raw.sort_index()

with pd.HDFStore(str(ROOT / "data/processed/mnq_2026ytd_1min.h5"), "r") as s:
    df_mnq_raw = s["/bars_1min"].copy()
df_mnq_raw.index.name = "timestamp"


def rth(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        ((df.index.hour == 9) & (df.index.minute >= 30))
        | ((df.index.hour > 9) & (df.index.hour < 16))
    )
    return df[mask].copy()


def resample_5m(bars_1m: pd.DataFrame) -> pd.DataFrame:
    df = bars_1m[["open", "high", "low", "close", "volume"]].copy()
    df["_date"] = df.index.date
    df["_bucket"] = df.index.floor("5min")
    g = df.groupby(["_date", "_bucket"]).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    g = g.reset_index(level="_date", drop=True)
    g.index.name = "timestamp"
    return g


train_5m = resample_5m(rth(df_train_raw))
oos_5m   = resample_5m(rth(df_oos_raw))
mnq_5m   = resample_5m(rth(df_mnq_raw))

print(f"Train 5m bars : {len(train_5m):,}  ({train_5m.index[0].date()} – {train_5m.index[-1].date()})")
print(f"OOS   5m bars : {len(oos_5m):,}  ({oos_5m.index[0].date()} – {oos_5m.index[-1].date()})")
print(f"MNQ   5m bars : {len(mnq_5m):,}  ({mnq_5m.index[0].date()} – {mnq_5m.index[-1].date()})")


# ---------------------------------------------------------------------------
# ORB configuration
# ---------------------------------------------------------------------------

ORB_CFG = {
    "or_end_bar":        7,       # bars 0–6 = 9:30–10:00; first signal at bar 7 (10:05)
    "min_range_atr":     0.3,     # OR range must be >= 0.3x daily ATR
    "pt_mult":           2.0,     # profit target
    "sl_mult":           1.5,     # stop loss
    "time_stop_bars":    24,      # bars before time-exit
    "entry_cutoff_h":    12,      # no entries at or after noon
    "long_only":         True,
    "require_prev_vwap": True,    # only trade when prev session close > prev session VWAP
}


# ---------------------------------------------------------------------------
# Session stats builder (inline — includes prev_close)
# ---------------------------------------------------------------------------

def build_session_stats(bars_5m: pd.DataFrame) -> list:
    """Build one dict per session with daily OHLC, ATR, volume features, and prev_close."""
    sessions = sorted(bars_5m.groupby(bars_5m.index.date), key=lambda x: x[0])
    rows = []
    for date, sess in sessions:
        if len(sess) < 2:
            continue
        op = float(sess["open"].iloc[0])
        cl = float(sess["close"].iloc[-1])
        hi = float(sess["high"].max())
        lo = float(sess["low"].min())
        rows.append({
            "date": date,
            "open": op,
            "high": hi,
            "low": lo,
            "close": cl,
            "vol_first_30m": float(sess["volume"].iloc[:6].sum()),
        })

    stats_list = []
    prev_close_val = None
    atr_val = None
    vol_first_30m_history = []

    for i, row in enumerate(rows):
        hi = row["high"]
        lo = row["low"]
        cl = row["close"]

        if prev_close_val is None:
            tr = hi - lo
        else:
            tr = max(hi - lo, abs(hi - prev_close_val), abs(lo - prev_close_val))

        if atr_val is None:
            atr_val = tr
        else:
            atr_val = (atr_val * 13 + tr) / 14

        vol_first_30m_history.append(row["vol_first_30m"])
        if len(vol_first_30m_history) >= 20:
            avg_vol_30m = float(np.mean(vol_first_30m_history[-20:]))
        else:
            avg_vol_30m = None

        if i >= 1:
            prev_1d_ret = (row["close"] - rows[i - 1]["close"]) / (rows[i - 1]["close"] + 1e-8)
        else:
            prev_1d_ret = 0.0

        if i >= 3:
            prev_3d_ret = (row["close"] - rows[i - 3]["close"]) / (rows[i - 3]["close"] + 1e-8)
        else:
            prev_3d_ret = 0.0

        if i >= 1:
            gap_pct = (row["open"] - rows[i - 1]["close"]) / (rows[i - 1]["close"] + 1e-8)
        else:
            gap_pct = 0.0

        stats_list.append({
            "date": row["date"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "true_range": tr,
            "atr_20d": atr_val,
            "vol_first_30m": row["vol_first_30m"],
            "rolling_20d_avg_vol_first_30m": avg_vol_30m,
            "prev_1d_ret": prev_1d_ret,
            "prev_3d_ret": prev_3d_ret,
            "gap_pct": gap_pct,
            "prev_close": rows[i - 1]["close"] if i > 0 else None,
        })
        prev_close_val = cl

    return stats_list


# ---------------------------------------------------------------------------
# Session VWAP helper
# ---------------------------------------------------------------------------

def compute_session_vwap(session_5m: pd.DataFrame) -> float:
    """Full-session VWAP for a single session DataFrame."""
    tp = (session_5m["high"] + session_5m["low"] + session_5m["close"]) / 3
    vol = session_5m["volume"]
    return float((tp * vol).sum() / (vol.sum() + 1e-8))


# ---------------------------------------------------------------------------
# Regime classifier (returns neg_count too)
# ---------------------------------------------------------------------------

def classify_regime_full(session: pd.DataFrame, sess_stat: dict, atr: float) -> tuple:
    """
    Classify session regime at the 30-min mark.
    Returns (regime_str, vwap_slope_30m, neg_count).
    """
    if len(session) < 7:
        return "trend", 0.0, 0

    first6 = session.iloc[:6]
    open_price = float(session["open"].iloc[0])

    first_30m_high = float(first6["high"].max())
    first_30m_low  = float(first6["low"].min())
    first_30m_range = first_30m_high - first_30m_low

    atr_20d = sess_stat["atr_20d"]
    first_30m_range_ratio = first_30m_range / (atr_20d + 1e-8)

    vwap_num = 0.0
    vwap_den = 0.0
    for _, b in first6.iterrows():
        typical = (b["high"] + b["low"] + b["close"]) / 3.0
        vwap_num += typical * b["volume"]
        vwap_den += b["volume"]
    vwap_30m = vwap_num / (vwap_den + 1e-8) if vwap_den > 0 else (
        (first6["high"] + first6["low"] + first6["close"]).mean() / 3.0
    )
    vwap_slope_30m = (vwap_30m - open_price) / (open_price + 1e-8)

    vol_first_30m = sess_stat["vol_first_30m"]
    avg_vol_30m   = sess_stat["rolling_20d_avg_vol_first_30m"]
    if avg_vol_30m is not None and avg_vol_30m > 0:
        rel_volume_30m = vol_first_30m / avg_vol_30m
        use_rel_vol = True
    else:
        rel_volume_30m = 1.0
        use_rel_vol = False

    sum_tr_first6 = float(first6.apply(lambda r: r["high"] - r["low"], axis=1).sum()) + 1e-8
    close_bar6 = float(first6["close"].iloc[-1])
    directional_efficiency = abs(close_bar6 - open_price) / sum_tr_first6

    gap_pct    = sess_stat["gap_pct"]
    prev_1d_ret = sess_stat["prev_1d_ret"]
    prev_3d_ret = sess_stat["prev_3d_ret"]

    neg_count = (
        int(gap_pct < -0.002)
        + int(prev_1d_ret < -0.005)
        + int(prev_3d_ret < -0.01)
        + int(first_30m_range_ratio > 1.3)
        + (int(rel_volume_30m > 1.3) if use_rel_vol else 0)
        + int(vwap_slope_30m < -0.0003)
    )

    if neg_count >= 2 and directional_efficiency > 0.25:
        regime = "negative"
    elif directional_efficiency < 0.25 and abs(vwap_slope_30m) < 0.0005:
        regime = "chop"
    else:
        regime = "trend"

    return regime, vwap_slope_30m, neg_count


# ---------------------------------------------------------------------------
# ORB session engine
# ---------------------------------------------------------------------------

def run_orb_session(session: pd.DataFrame, atr: float, prev_vwap_passed: bool,
                    cfg: dict, point_value: float, cost_rt: float) -> list:
    """Run ORB on one session. Returns list of trade dicts."""
    if not prev_vwap_passed and cfg["require_prev_vwap"]:
        return []

    or_end = cfg["or_end_bar"]
    if len(session) <= or_end:
        return []

    or_bars  = session.iloc[:or_end]
    or_high  = float(or_bars["high"].max())
    or_low   = float(or_bars["low"].min())
    or_range = or_high - or_low

    if or_range < cfg["min_range_atr"] * atr:
        return []

    trades = []
    in_trade    = False
    entry_px    = stop_px = target_px = 0.0
    entry_bar_i = 0

    for i, (ts, bar) in enumerate(session.iterrows()):
        if i < or_end:
            continue
        if ts.hour >= cfg["entry_cutoff_h"] and not in_trade:
            break

        # Exit check
        if in_trade:
            force_end = (i == len(session) - 1)
            if force_end:
                pnl = (bar["close"] - entry_px) * point_value - cost_rt
                trades.append({"date": str(ts.date()), "pnl": round(pnl, 2),
                               "exit_reason": "session_end", "strategy": "orb"})
                in_trade = False
            elif bar["high"] >= target_px:
                pnl = (target_px - entry_px) * point_value - cost_rt
                trades.append({"date": str(ts.date()), "pnl": round(pnl, 2),
                               "exit_reason": "target", "strategy": "orb"})
                in_trade = False
            elif bar["low"] <= stop_px:
                pnl = (stop_px - entry_px) * point_value - cost_rt
                trades.append({"date": str(ts.date()), "pnl": round(pnl, 2),
                               "exit_reason": "stop", "strategy": "orb"})
                in_trade = False
            elif (i - entry_bar_i) >= cfg["time_stop_bars"]:
                pnl = (bar["close"] - entry_px) * point_value - cost_rt
                trades.append({"date": str(ts.date()), "pnl": round(pnl, 2),
                               "exit_reason": "time", "strategy": "orb"})
                in_trade = False

        # Entry check (long only, close-based signal)
        if not in_trade and i >= or_end and ts.hour < cfg["entry_cutoff_h"]:
            if cfg["long_only"] and bar["close"] > or_high:
                in_trade    = True
                entry_px    = float(bar["close"])
                stop_px     = entry_px - cfg["sl_mult"] * atr
                target_px   = entry_px + cfg["pt_mult"] * atr
                entry_bar_i = i

    return trades


# ---------------------------------------------------------------------------
# RCAF chop + neg SHORT session engine
# ---------------------------------------------------------------------------

def run_rcaf_session(session: pd.DataFrame, sess_stat: dict, vwap_slope_30m: float,
                     neg_count: int, regime: str,
                     point_value: float, cost_rt: float) -> list:
    """
    Run RCAF chop SHORT + negative SHORT on one session.
    regime: 'negative' | 'chop' | 'trend'
    Returns list of trade dicts.
    """
    if regime == "trend":
        return []

    atr_20d  = sess_stat["atr_20d"]
    bar_atr  = atr_20d / 13.0

    trades      = []
    vwap_num    = vwap_den = 0.0
    vol_history = []
    session_vwap_devs = []
    in_trade    = False
    trade_dir   = 0
    entry_px    = stop_px = target_px = 0.0
    entry_bar   = time_stop_bars = 0
    trades_today = 0
    was_below_vwap_early = False
    consec_above_vwap = prev_consec_above_vwap = 0

    for bar_i, (ts, bar) in enumerate(session.iterrows()):
        tp       = (bar["high"] + bar["low"] + bar["close"]) / 3
        vwap_num += tp * max(float(bar["volume"]), 1e-8)
        vwap_den += max(float(bar["volume"]), 1e-8)
        vwap     = vwap_num / (vwap_den + 1e-8)

        vol_history.append(float(bar["volume"]))
        vol_median = float(np.median(vol_history[-20:]))

        session_vwap_devs.append(float(bar["close"]) - vwap)
        vwap_std = float(np.std(session_vwap_devs)) if len(session_vwap_devs) >= 3 else bar_atr * 0.5

        if bar_i < 6 and bar["low"] < vwap:
            was_below_vwap_early = True

        if bar_i < 6:
            continue

        force_exit = (bar_i == len(session) - 1)

        # Exit
        if in_trade:
            exit_px = None
            exit_reason = None
            if trade_dir == -1:
                if force_exit:
                    exit_px = float(bar["close"])
                    exit_reason = "session_end"
                elif bar["low"] <= target_px:
                    exit_px = target_px
                    exit_reason = "target"
                elif bar["high"] >= stop_px:
                    exit_px = stop_px
                    exit_reason = "stop"
                elif (bar_i - entry_bar) >= time_stop_bars:
                    exit_px = float(bar["close"])
                    exit_reason = "time"
            if exit_px is not None:
                pnl = trade_dir * (exit_px - entry_px) * point_value - cost_rt
                trades.append({"date": str(ts.date()), "pnl": round(pnl, 2),
                               "exit_reason": exit_reason, "strategy": "rcaf",
                               "regime": regime})
                in_trade = False

        # Consecutive VWAP tracker (update before entry check)
        prev_consec_above_vwap = consec_above_vwap
        consec_above_vwap = consec_above_vwap + 1 if float(bar["close"]) > vwap else 0

        if in_trade or trades_today >= 2:
            continue

        rng_b     = bar["high"] - bar["low"] + 1e-6
        clv       = (2 * bar["close"] - bar["high"] - bar["low"]) / rng_b
        close_loc = (bar["close"] - bar["low"]) / rng_b
        body      = abs(bar["close"] - bar["open"])
        body_frac = body / rng_b
        body_signed = float(bar["close"]) - float(bar["open"])

        def fs(direction: int) -> float:
            uw = bar["high"] - max(bar["open"], bar["close"])
            lw = min(bar["open"], bar["close"]) - bar["low"]
            return (-direction * clv) + (direction * (uw - lw) / rng_b) + (-direction * body_signed / rng_b)

        if regime == "negative":
            fsu = fs(+1)
            if (was_below_vwap_early and prev_consec_above_vwap >= 3
                    and close_loc < 0.35 and bar["volume"] > 1.2 * vol_median and fsu > 0.3):
                in_trade    = True
                trade_dir   = -1
                entry_px    = float(bar["close"])
                stop_px     = entry_px + 1.0 * bar_atr
                target_px   = entry_px - 2.0 * bar_atr
                entry_bar   = bar_i
                time_stop_bars = 8
                trades_today += 1

        elif regime == "chop":
            fsc = fs(+1)
            if (float(bar["close"]) > vwap
                    and abs(float(bar["close"]) - vwap) > 1.5 * vwap_std
                    and fsc > 0.5 and body_frac < 0.45):
                in_trade    = True
                trade_dir   = -1
                entry_px    = float(bar["close"])
                stop_px     = entry_px + 0.5 * bar_atr
                target_px   = vwap
                entry_bar   = bar_i
                time_stop_bars = 6
                trades_today += 1

    return trades


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def stats(trade_recs: list) -> dict:
    if not trade_recs:
        return dict(N=0, n_day=0, WR=0, AvgW=0, AvgL=0, PF=0, PnL=0, Sharpe=0, MaxDD=0)
    p    = np.array([t["pnl"] for t in trade_recs])
    wins = p[p > 0]
    losses = p[p <= 0]
    WR   = len(wins) / len(p)
    avg_w = float(wins.mean()) if len(wins) else 0.0
    avg_l = float(losses.mean()) if len(losses) else 0.0
    pf    = float(wins.sum()) / (float(abs(losses.sum())) + 1e-8) if len(losses) else float("inf")

    by_day = {}
    for t in trade_recs:
        by_day.setdefault(t["date"], 0.0)
        by_day[t["date"]] += t["pnl"]
    daily  = np.array(list(by_day.values()))
    n_day  = len(daily)
    sharpe = (daily.mean() / (daily.std() + 1e-8)) * np.sqrt(252) if n_day > 1 else 0.0

    cum      = np.cumsum(p)
    roll_max = np.maximum.accumulate(cum)
    max_dd   = float((cum - roll_max).min())

    return dict(
        N=len(p), n_day=n_day, WR=round(WR, 4),
        AvgW=round(avg_w, 2), AvgL=round(avg_l, 2), PF=round(pf, 3),
        PnL=round(float(p.sum()), 2), Sharpe=round(float(sharpe), 3),
        MaxDD=round(max_dd, 2),
    )


def daily_pnl_series(trade_recs: list, all_dates) -> dict:
    """Returns dict[date_str -> float] including zero-PnL days."""
    by_day = {str(d): 0.0 for d in all_dates}
    for t in trade_recs:
        by_day[t["date"]] = by_day.get(t["date"], 0.0) + t["pnl"]
    return by_day


def combined_stats(orb_daily: dict, rcaf_daily: dict, all_dates) -> dict:
    """Merge two daily-PnL dicts and compute portfolio stats."""
    all_d = sorted(set(list(orb_daily.keys()) + list(rcaf_daily.keys()) + [str(d) for d in all_dates]))
    combined_pnl = []
    for d in all_d:
        combined_pnl.append((orb_daily.get(d, 0.0) + rcaf_daily.get(d, 0.0)))
    pnl_arr = np.array(combined_pnl)
    active  = pnl_arr[pnl_arr != 0.0]
    n_active = int((pnl_arr != 0.0).sum())
    day_wr   = float((active > 0).mean()) if len(active) else 0.0
    sharpe   = (pnl_arr.mean() / (pnl_arr.std() + 1e-8)) * np.sqrt(252) if len(pnl_arr) > 1 else 0.0
    cum      = np.cumsum(pnl_arr)
    roll_max = np.maximum.accumulate(cum) if len(cum) else np.array([0.0])
    max_dd   = float((cum - roll_max).min()) if len(cum) else 0.0
    return dict(
        N_active_days=n_active,
        day_WR=round(day_wr, 4),
        PnL=round(float(pnl_arr.sum()), 2),
        Sharpe=round(float(sharpe), 3),
        MaxDD=round(max_dd, 2),
    )


def sharpe_from_daily(daily_pnl_dict: dict, all_dates) -> float:
    pnl_arr = np.array([daily_pnl_dict.get(str(d), 0.0) for d in all_dates])
    if len(pnl_arr) < 2 or pnl_arr.std() == 0:
        return 0.0
    return float((pnl_arr.mean() / (pnl_arr.std() + 1e-8)) * np.sqrt(252))


def max_dd_from_daily(daily_pnl_dict: dict, all_dates) -> float:
    pnl_arr = np.array([daily_pnl_dict.get(str(d), 0.0) for d in all_dates])
    cum      = np.cumsum(pnl_arr)
    roll_max = np.maximum.accumulate(cum) if len(cum) else np.array([0.0])
    return float((cum - roll_max).min())


# ---------------------------------------------------------------------------
# Main combined backtest loop
# ---------------------------------------------------------------------------

def run_combined(bars_5m: pd.DataFrame, point_value: float, cost_rt: float,
                 warmup: int = 21) -> tuple:
    """
    Run ORB + RCAF on the same 5-min bar dataset.
    Returns (orb_trades, rcaf_trades, all_dates).
    """
    sessions_list = sorted(bars_5m.groupby(bars_5m.index.date), key=lambda x: x[0])
    sess_stats    = build_session_stats(bars_5m)
    date_to_stat  = {s["date"]: s for s in sess_stats}

    # Pre-compute per-session VWAP for prev_vwap filter
    session_vwaps = {}
    for date, sess in sessions_list:
        session_vwaps[date] = compute_session_vwap(sess)

    all_dates  = [date for date, _ in sessions_list]
    orb_trades  = []
    rcaf_trades = []

    for sess_idx, (date, session) in enumerate(sessions_list):
        if sess_idx < warmup or len(session) < 8:
            continue

        stat = date_to_stat.get(date)
        if stat is None:
            continue

        atr = stat["atr_20d"]

        # ORB prev_vwap filter
        prev_vwap_passed = False
        if sess_idx > 0:
            prev_date      = sessions_list[sess_idx - 1][0]
            prev_close_val = stat.get("prev_close")
            prev_vwap_val  = session_vwaps.get(prev_date, 0.0)
            if prev_close_val is not None:
                prev_vwap_passed = (prev_close_val > prev_vwap_val)

        orb_trades.extend(
            run_orb_session(session, atr, prev_vwap_passed, ORB_CFG, point_value, cost_rt)
        )

        # RCAF regime classification
        regime, vwap_slope_30m, neg_count = classify_regime_full(session, stat, atr)
        rcaf_trades.extend(
            run_rcaf_session(session, stat, vwap_slope_30m, neg_count, regime,
                             point_value, cost_rt)
        )

    return orb_trades, rcaf_trades, all_dates


# ---------------------------------------------------------------------------
# Monte Carlo (Topstep combine limits)
# ---------------------------------------------------------------------------

def monte_carlo_combined(daily_pnl_arr: np.ndarray, n_paths: int = 10_000,
                         n_days: int = 60,
                         profit_target: float = 3000.0,
                         trail_dd_limit: float = -2000.0,
                         daily_dd_limit: float = -1000.0) -> dict:
    np.random.seed(42)
    passed = busted_dd = busted_daily = 0
    finish_days = []

    for _ in range(n_paths):
        sample  = np.random.choice(daily_pnl_arr, size=n_days, replace=True)
        cum_pnl = 0.0
        peak    = 0.0
        result  = "timeout"
        day_res = n_days

        for d_idx, dpnl in enumerate(sample):
            if dpnl < daily_dd_limit:
                busted_daily += 1
                result  = "bust_daily"
                day_res = d_idx + 1
                break
            cum_pnl += dpnl
            peak    = max(peak, cum_pnl)
            trail   = cum_pnl - peak
            if trail <= trail_dd_limit:
                busted_dd += 1
                result  = "bust_dd"
                day_res = d_idx + 1
                break
            if cum_pnl >= profit_target:
                result  = "pass"
                day_res = d_idx + 1
                break
        else:
            day_res = n_days

        if result == "pass":
            passed += 1
            finish_days.append(day_res)

    p_pass         = passed / n_paths
    p_bust_dd      = busted_dd / n_paths
    p_bust_daily   = busted_daily / n_paths
    median_days    = int(np.median(finish_days)) if finish_days else n_days
    return dict(
        p_pass=round(p_pass, 4),
        p_bust_dd=round(p_bust_dd, 4),
        p_bust_daily=round(p_bust_daily, 4),
        median_days_to_pass=median_days,
    )


# ---------------------------------------------------------------------------
# Run training set
# ---------------------------------------------------------------------------

MES_PV  = 5.0
MES_CR  = 2.50
MNQ_PV  = 2.0
MNQ_CR  = 1.00

print("\n" + "=" * 80)
print("RUNNING COMBINED BACKTEST — TRAINING (MES)")
print("=" * 80)

orb_train, rcaf_train, train_dates = run_combined(
    train_5m, point_value=MES_PV, cost_rt=MES_CR, warmup=21
)

print(f"ORB  trades (train): {len(orb_train)}")
print(f"RCAF trades (train): {len(rcaf_train)}")

# ---------------------------------------------------------------------------
# Section 1: Standalone performance
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 1: STANDALONE PERFORMANCE (training)")
print("=" * 80)

orb_st  = stats(orb_train)
rcaf_st = stats(rcaf_train)
rcaf_chop_st = stats([t for t in rcaf_train if t["regime"] == "chop"])
rcaf_neg_st  = stats([t for t in rcaf_train if t["regime"] == "negative"])

def fmt(label: str, s: dict):
    if s["N"] == 0:
        print(f"  {label:35s} | NO TRADES")
        return
    print(
        f"  {label:35s} | N={s['N']:4d} | WR={s['WR']:.1%}"
        f" | Sharpe={s['Sharpe']:.2f} | PnL=${s['PnL']:,.0f} | MaxDD=${s['MaxDD']:,.0f}"
    )

fmt("ORB standalone", orb_st)
fmt("RCAF standalone", rcaf_st)
fmt("  RCAF chop", rcaf_chop_st)
fmt("  RCAF negative", rcaf_neg_st)

# ---------------------------------------------------------------------------
# Section 2: Day-level overlap analysis
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 2: DAY-LEVEL OVERLAP ANALYSIS (training)")
print("=" * 80)

orb_days  = set(t["date"] for t in orb_train)
rcaf_days = set(t["date"] for t in rcaf_train)
both_days = orb_days & rcaf_days
total_days = len(train_dates)

rcaf_chop_days = set(t["date"] for t in rcaf_train if t["regime"] == "chop")
rcaf_neg_days  = set(t["date"] for t in rcaf_train if t["regime"] == "negative")

print(f"  Total sessions in dataset:    {total_days}")
print(f"  ORB active days:              {len(orb_days)}")
print(f"  RCAF active days:             {len(rcaf_days)}")
print(f"  Both active same day:         {len(both_days)}")
print(f"  RCAF-only days:               {len(rcaf_days - orb_days)}")
print(f"  ORB-only days:                {len(orb_days - rcaf_days)}")

if rcaf_chop_days:
    orb_on_chop = len(orb_days & rcaf_chop_days) / len(rcaf_chop_days)
    print(f"  ORB fires on chop-regime days: {orb_on_chop:.1%}")
if rcaf_neg_days:
    orb_on_neg  = len(orb_days & rcaf_neg_days) / len(rcaf_neg_days)
    print(f"  ORB fires on neg-regime days:  {orb_on_neg:.1%}")

overlap_pct = len(both_days) / max(len(rcaf_days), 1)
print(f"  Overlap (both/RCAF active):   {overlap_pct:.1%}")

# ---------------------------------------------------------------------------
# Section 3: Daily PnL correlation
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 3: DAILY PnL CORRELATION (days both active, training)")
print("=" * 80)

orb_daily_dict  = daily_pnl_series(orb_train,  train_dates)
rcaf_daily_dict = daily_pnl_series(rcaf_train, train_dates)

if both_days:
    orb_on_both  = np.array([orb_daily_dict[d] for d in sorted(both_days)])
    rcaf_on_both = np.array([rcaf_daily_dict[d] for d in sorted(both_days)])
    corr_val = float(np.corrcoef(orb_on_both, rcaf_on_both)[0, 1]) if len(both_days) >= 3 else float("nan")
    print(f"  Correlation (days both active): r = {corr_val:.3f}")
    print(f"  (negative = regime-complementary, positive = same-regime exposure)")
else:
    corr_val = float("nan")
    print("  No overlapping days — cannot compute correlation.")

# ---------------------------------------------------------------------------
# Section 4: Combined portfolio stats
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 4: COMBINED PORTFOLIO STATS (training)")
print("=" * 80)

comb_st = combined_stats(orb_daily_dict, rcaf_daily_dict, train_dates)

# Compute standalone daily Sharpe on full calendar (including zero days)
orb_sharpe_full  = sharpe_from_daily(orb_daily_dict,  train_dates)
rcaf_sharpe_full = sharpe_from_daily(rcaf_daily_dict, train_dates)
orb_dd_full      = max_dd_from_daily(orb_daily_dict,  train_dates)
rcaf_dd_full     = max_dd_from_daily(rcaf_daily_dict, train_dates)

# Day WR standalones
orb_active_pnls  = [v for v in orb_daily_dict.values()  if v != 0.0]
rcaf_active_pnls = [v for v in rcaf_daily_dict.values() if v != 0.0]
orb_day_wr       = float(np.mean([1 if v > 0 else 0 for v in orb_active_pnls])) if orb_active_pnls else 0.0
rcaf_day_wr      = float(np.mean([1 if v > 0 else 0 for v in rcaf_active_pnls])) if rcaf_active_pnls else 0.0

orb_pnl_full  = sum(orb_daily_dict.values())
rcaf_pnl_full = sum(rcaf_daily_dict.values())

header = f"{'Strategy':<32} | {'Sharpe':>7} | {'PnL':>8} | {'MaxDD':>9} | {'Day WR':>7}"
sep    = "-" * len(header)
print(header)
print(sep)
print(f"{'ORB standalone':<32} | {orb_sharpe_full:>7.2f} | ${orb_pnl_full:>7,.0f} | ${orb_dd_full:>8,.0f} | {orb_day_wr:>6.1%}")
print(f"{'RCAF standalone':<32} | {rcaf_sharpe_full:>7.2f} | ${rcaf_pnl_full:>7,.0f} | ${rcaf_dd_full:>8,.0f} | {rcaf_day_wr:>6.1%}")
print(f"{'Combined portfolio':<32} | {comb_st['Sharpe']:>7.2f} | ${comb_st['PnL']:>7,.0f} | ${comb_st['MaxDD']:>8,.0f} | {comb_st['day_WR']:>6.1%}")
delta_sharpe = comb_st["Sharpe"] - orb_sharpe_full
delta_pnl    = comb_st["PnL"] - orb_pnl_full
delta_dd     = comb_st["MaxDD"] - orb_dd_full
delta_wr     = comb_st["day_WR"] - orb_day_wr
sign_s = "+" if delta_sharpe >= 0 else ""
sign_p = "+" if delta_pnl >= 0 else ""
sign_d = "+" if delta_dd >= 0 else ""
sign_w = "+" if delta_wr >= 0 else ""
print(f"{'Delta (combined - ORB)':<32} | {sign_s}{delta_sharpe:>6.2f} | {sign_p}${delta_pnl:>6,.0f} | {sign_d}${delta_dd:>7,.0f} | {sign_w}{delta_wr:>5.1%}")

# ---------------------------------------------------------------------------
# Section 5: OOS MES (Jan-Feb 2026)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 5: OOS VALIDATION — MES Jan-Feb 2026")
print("=" * 80)

orb_oos, rcaf_oos, oos_dates = run_combined(
    oos_5m, point_value=MES_PV, cost_rt=MES_CR, warmup=5
)
orb_oos_daily  = daily_pnl_series(orb_oos,  oos_dates)
rcaf_oos_daily = daily_pnl_series(rcaf_oos, oos_dates)
comb_oos_st    = combined_stats(orb_oos_daily, rcaf_oos_daily, oos_dates)

orb_oos_sh   = sharpe_from_daily(orb_oos_daily,  oos_dates)
rcaf_oos_sh  = sharpe_from_daily(rcaf_oos_daily, oos_dates)
orb_oos_dd   = max_dd_from_daily(orb_oos_daily,  oos_dates)
rcaf_oos_dd  = max_dd_from_daily(rcaf_oos_daily, oos_dates)
orb_oos_pnl  = sum(orb_oos_daily.values())
rcaf_oos_pnl = sum(rcaf_oos_daily.values())

orb_oos_active  = [v for v in orb_oos_daily.values()  if v != 0.0]
rcaf_oos_active = [v for v in rcaf_oos_daily.values() if v != 0.0]
orb_oos_dwr  = float(np.mean([1 if v > 0 else 0 for v in orb_oos_active]))  if orb_oos_active  else 0.0
rcaf_oos_dwr = float(np.mean([1 if v > 0 else 0 for v in rcaf_oos_active])) if rcaf_oos_active else 0.0

print(header)
print(sep)
print(f"{'OOS ORB standalone':<32} | {orb_oos_sh:>7.2f} | ${orb_oos_pnl:>7,.0f} | ${orb_oos_dd:>8,.0f} | {orb_oos_dwr:>6.1%}")
print(f"{'OOS RCAF standalone':<32} | {rcaf_oos_sh:>7.2f} | ${rcaf_oos_pnl:>7,.0f} | ${rcaf_oos_dd:>8,.0f} | {rcaf_oos_dwr:>6.1%}")
print(f"{'OOS Combined':<32} | {comb_oos_st['Sharpe']:>7.2f} | ${comb_oos_st['PnL']:>7,.0f} | ${comb_oos_st['MaxDD']:>8,.0f} | {comb_oos_st['day_WR']:>6.1%}")
print(f"  OOS ORB trades: {len(orb_oos)}, OOS RCAF trades: {len(rcaf_oos)}")

# ---------------------------------------------------------------------------
# Section 6: MNQ transfer
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 6: MNQ TRANSFER (1 contract each, point_value=2.0)")
print("=" * 80)

orb_mnq, rcaf_mnq, mnq_dates = run_combined(
    mnq_5m, point_value=MNQ_PV, cost_rt=MNQ_CR, warmup=5
)
orb_mnq_daily  = daily_pnl_series(orb_mnq,  mnq_dates)
rcaf_mnq_daily = daily_pnl_series(rcaf_mnq, mnq_dates)
comb_mnq_st    = combined_stats(orb_mnq_daily, rcaf_mnq_daily, mnq_dates)

orb_mnq_sh   = sharpe_from_daily(orb_mnq_daily,  mnq_dates)
rcaf_mnq_sh  = sharpe_from_daily(rcaf_mnq_daily, mnq_dates)
orb_mnq_dd   = max_dd_from_daily(orb_mnq_daily,  mnq_dates)
rcaf_mnq_dd  = max_dd_from_daily(rcaf_mnq_daily, mnq_dates)
orb_mnq_pnl  = sum(orb_mnq_daily.values())
rcaf_mnq_pnl = sum(rcaf_mnq_daily.values())

orb_mnq_active  = [v for v in orb_mnq_daily.values()  if v != 0.0]
rcaf_mnq_active = [v for v in rcaf_mnq_daily.values() if v != 0.0]
orb_mnq_dwr  = float(np.mean([1 if v > 0 else 0 for v in orb_mnq_active]))  if orb_mnq_active  else 0.0
rcaf_mnq_dwr = float(np.mean([1 if v > 0 else 0 for v in rcaf_mnq_active])) if rcaf_mnq_active else 0.0

print(header)
print(sep)
print(f"{'MNQ ORB standalone':<32} | {orb_mnq_sh:>7.2f} | ${orb_mnq_pnl:>7,.0f} | ${orb_mnq_dd:>8,.0f} | {orb_mnq_dwr:>6.1%}")
print(f"{'MNQ RCAF standalone':<32} | {rcaf_mnq_sh:>7.2f} | ${rcaf_mnq_pnl:>7,.0f} | ${rcaf_mnq_dd:>8,.0f} | {rcaf_mnq_dwr:>6.1%}")
print(f"{'MNQ Combined':<32} | {comb_mnq_st['Sharpe']:>7.2f} | ${comb_mnq_st['PnL']:>7,.0f} | ${comb_mnq_st['MaxDD']:>8,.0f} | {comb_mnq_st['day_WR']:>6.1%}")
print(f"  MNQ ORB trades: {len(orb_mnq)}, MNQ RCAF trades: {len(rcaf_mnq)}")

# ---------------------------------------------------------------------------
# Section 7: Monte Carlo — combined portfolio (training daily PnLs)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 7: MONTE CARLO — COMBINED PORTFOLIO (10k paths × 60 days, training)")
print("=" * 80)

combined_daily_arr = np.array([
    orb_daily_dict.get(str(d), 0.0) + rcaf_daily_dict.get(str(d), 0.0)
    for d in train_dates
])

mc_results = monte_carlo_combined(
    combined_daily_arr,
    n_paths=10_000, n_days=60,
    profit_target=3000.0,
    trail_dd_limit=-2000.0,
    daily_dd_limit=-1000.0,
)
print(f"  P(pass)        = {mc_results['p_pass']:.1%}")
print(f"  P(bust_dd)     = {mc_results['p_bust_dd']:.1%}")
print(f"  P(bust_daily)  = {mc_results['p_bust_daily']:.1%}")
print(f"  Median days to pass (of passers): {mc_results['median_days_to_pass']}")

# Also run ORB-only MC for comparison
orb_only_arr = np.array([orb_daily_dict.get(str(d), 0.0) for d in train_dates])
mc_orb_only  = monte_carlo_combined(
    orb_only_arr,
    n_paths=10_000, n_days=60,
    profit_target=3000.0,
    trail_dd_limit=-2000.0,
    daily_dd_limit=-1000.0,
)
print(f"\n  ORB-only P(pass): {mc_orb_only['p_pass']:.1%}  (for comparison)")
print(f"  Combined P(pass): {mc_results['p_pass']:.1%}")

# ---------------------------------------------------------------------------
# Section 8: Verdict
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 8: VERDICT")
print("=" * 80)

check_sharpe   = comb_st["Sharpe"] > orb_sharpe_full
check_dd       = comb_st["MaxDD"]  >= orb_dd_full * 1.20   # not more than 20% worse
check_rcaf_own = rcaf_sharpe_full > 0.0
check_overlap  = overlap_pct < 0.30
check_oos      = comb_oos_st["Sharpe"] > 0.0

def check_mark(b: bool) -> str:
    return "PASS" if b else "FAIL"

print(f"  Key question: does RCAF improve ORB?\n")
print(f"  {check_mark(check_sharpe):<4}  Combined Sharpe ({comb_st['Sharpe']:.2f}) > ORB standalone ({orb_sharpe_full:.2f})")
print(f"  {check_mark(check_dd):<4}  Combined MaxDD (${comb_st['MaxDD']:,.0f}) not worse than ORB (${orb_dd_full:,.0f}) × 1.20")
print(f"  {check_mark(check_rcaf_own):<4}  RCAF-only days have positive Sharpe ({rcaf_sharpe_full:.2f})")
print(f"  {check_mark(check_overlap):<4}  Overlap < 30% (actual: {overlap_pct:.1%})")
print(f"  {check_mark(check_oos):<4}  OOS combined Sharpe > 0 (actual: {comb_oos_st['Sharpe']:.2f})")

n_pass = sum([check_sharpe, check_dd, check_rcaf_own, check_overlap, check_oos])
if n_pass >= 4:
    verdict = "ADDITIVE"
elif n_pass >= 2:
    verdict = "NEUTRAL"
else:
    verdict = "HARMFUL"

print(f"\n  *** VERDICT: {verdict} ***")
if verdict == "ADDITIVE":
    print("  RCAF adds diversification without hurting ORB DD.")
elif verdict == "NEUTRAL":
    print("  RCAF adds marginal value; consider regime-selective deployment.")
else:
    print("  RCAF degrades combined portfolio; deploy ORB standalone only.")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

results = {
    "verdict": verdict,
    "checks": {
        "combined_sharpe_better": check_sharpe,
        "combined_dd_not_worse":  check_dd,
        "rcaf_positive_sharpe":   check_rcaf_own,
        "overlap_lt_30pct":       check_overlap,
        "oos_combined_positive":  check_oos,
    },
    "training_mes": {
        "orb_standalone":  dict(
            N=orb_st["N"], WR=orb_st["WR"], Sharpe=orb_sharpe_full,
            PnL=round(orb_pnl_full, 2), MaxDD=round(orb_dd_full, 2), day_WR=round(orb_day_wr, 4)
        ),
        "rcaf_standalone": dict(
            N=rcaf_st["N"], WR=rcaf_st["WR"], Sharpe=rcaf_sharpe_full,
            PnL=round(rcaf_pnl_full, 2), MaxDD=round(rcaf_dd_full, 2), day_WR=round(rcaf_day_wr, 4)
        ),
        "rcaf_chop":     dict(N=rcaf_chop_st["N"], WR=rcaf_chop_st["WR"],
                              Sharpe=rcaf_chop_st["Sharpe"], PnL=rcaf_chop_st["PnL"],
                              MaxDD=rcaf_chop_st["MaxDD"]),
        "rcaf_negative": dict(N=rcaf_neg_st["N"], WR=rcaf_neg_st["WR"],
                              Sharpe=rcaf_neg_st["Sharpe"], PnL=rcaf_neg_st["PnL"],
                              MaxDD=rcaf_neg_st["MaxDD"]),
        "combined":      comb_st,
        "delta_sharpe":  round(delta_sharpe, 3),
        "delta_pnl":     round(delta_pnl, 2),
        "delta_dd":      round(delta_dd, 2),
        "overlap_days":  len(both_days),
        "overlap_pct":   round(overlap_pct, 4),
        "correlation_both_active": round(corr_val, 4) if not np.isnan(corr_val) else None,
    },
    "oos_mes": {
        "orb_standalone":  dict(N=len(orb_oos), Sharpe=round(orb_oos_sh, 3),
                                PnL=round(orb_oos_pnl, 2), MaxDD=round(orb_oos_dd, 2)),
        "rcaf_standalone": dict(N=len(rcaf_oos), Sharpe=round(rcaf_oos_sh, 3),
                                PnL=round(rcaf_oos_pnl, 2), MaxDD=round(rcaf_oos_dd, 2)),
        "combined":        comb_oos_st,
    },
    "mnq_transfer": {
        "orb_standalone":  dict(N=len(orb_mnq), Sharpe=round(orb_mnq_sh, 3),
                                PnL=round(orb_mnq_pnl, 2), MaxDD=round(orb_mnq_dd, 2)),
        "rcaf_standalone": dict(N=len(rcaf_mnq), Sharpe=round(rcaf_mnq_sh, 3),
                                PnL=round(rcaf_mnq_pnl, 2), MaxDD=round(rcaf_mnq_dd, 2)),
        "combined":        comb_mnq_st,
    },
    "monte_carlo_combined": mc_results,
    "monte_carlo_orb_only": mc_orb_only,
}

out = ROOT / "rule_based_v1/diagnostics/rcaf_orb_results.json"
out.write_text(json.dumps(results, indent=2, default=str))
print(f"\nSaved → {out}")
