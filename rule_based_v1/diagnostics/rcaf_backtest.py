"""
RCAF (Regime-Conditioned Auction Failure) Backtest
Session-level regime classifier routes each day into one of 3 playbooks:
  negative  — short failed rallies + capitulation longs
  chop      — VWAP fade
  trend     — VWAP pullback continuation
Volume is a first-class signal throughout.
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


# Apply RTH then resample to 5-min
train_5m = resample_5m(rth(df_train_raw))
oos_5m = resample_5m(rth(df_oos_raw))
mnq_5m = resample_5m(rth(df_mnq_raw))

print(f"Train 5m bars : {len(train_5m):,}  ({train_5m.index[0].date()} – {train_5m.index[-1].date()})")
print(f"OOS   5m bars : {len(oos_5m):,}  ({oos_5m.index[0].date()} – {oos_5m.index[-1].date()})")
print(f"MNQ   5m bars : {len(mnq_5m):,}  ({mnq_5m.index[0].date()} – {mnq_5m.index[-1].date()})")


# ---------------------------------------------------------------------------
# Session summary builder
# ---------------------------------------------------------------------------

def build_session_stats(bars_5m: pd.DataFrame) -> list:
    """Build one dict per session with daily OHLC, ATR, volume features."""
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

    # Compute daily true range and 20-day Wilder ATR
    stats_list = []
    prev_close = None
    atr_val = None
    vol_first_30m_history = []

    for i, row in enumerate(rows):
        hi = row["high"]
        lo = row["low"]
        cl = row["close"]

        if prev_close is None:
            tr = hi - lo
        else:
            tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))

        # Wilder's smoothing
        if atr_val is None:
            atr_val = tr
        else:
            atr_val = (atr_val * 13 + tr) / 14  # 20-session Wilder (period=14 for stability)

        vol_first_30m_history.append(row["vol_first_30m"])
        # Rolling 20-session avg of first-30m volume
        if len(vol_first_30m_history) >= 20:
            avg_vol_30m = float(np.mean(vol_first_30m_history[-20:]))
        else:
            avg_vol_30m = None  # not enough history

        # Prior returns
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
        })
        prev_close = cl

    return stats_list


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------

def classify_regime(session: pd.DataFrame, sess_stat: dict, bar_atr: float) -> tuple:
    """
    Classify a session's regime at the 30-min mark (bar index 6).
    Returns (regime_str, vwap_slope_30m).
    """
    if len(session) < 7:
        return "trend", 0.0

    first6 = session.iloc[:6]
    open_price = float(session["open"].iloc[0])

    # first_30m_range
    first_30m_high = float(first6["high"].max())
    first_30m_low = float(first6["low"].min())
    first_30m_range = first_30m_high - first_30m_low

    atr_20d = sess_stat["atr_20d"]
    first_30m_range_ratio = first_30m_range / (atr_20d + 1e-8)

    # VWAP at bar 6
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

    # Relative volume (early 30m)
    vol_first_30m = sess_stat["vol_first_30m"]
    avg_vol_30m = sess_stat["rolling_20d_avg_vol_first_30m"]
    if avg_vol_30m is not None and avg_vol_30m > 0:
        rel_volume_30m = vol_first_30m / avg_vol_30m
        use_rel_vol = True
    else:
        rel_volume_30m = 1.0
        use_rel_vol = False

    # Directional efficiency over first 6 bars
    sum_tr_first6 = float(first6.apply(
        lambda r: r["high"] - r["low"], axis=1
    ).sum()) + 1e-8
    close_bar6 = float(first6["close"].iloc[-1])
    directional_efficiency = abs(close_bar6 - open_price) / sum_tr_first6

    # Negative score
    gap_pct = sess_stat["gap_pct"]
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

    return regime, vwap_slope_30m


# ---------------------------------------------------------------------------
# Failure score
# ---------------------------------------------------------------------------

def failure_score(bar: pd.Series, direction: int, vwap: float, atr: float) -> float:
    """
    Returns a failure probability proxy. Higher = more likely to fail.
    direction: +1 for upside breakout assessment, -1 for downside.
    """
    rng = bar["high"] - bar["low"] + 1e-6
    body = abs(bar["close"] - bar["open"])

    clv = (2 * bar["close"] - bar["high"] - bar["low"]) / rng
    clv_score = -direction * clv

    uw = bar["high"] - max(bar["open"], bar["close"])
    lw = min(bar["open"], bar["close"]) - bar["low"]
    wick_score = direction * (uw - lw) / rng

    # Signed pressure: bar closed against the breakout direction = failure
    body_signed = float(bar["close"]) - float(bar["open"])
    pressure_score = -direction * (body_signed / rng)   # +1 if bar closed against direction

    return clv_score + wick_score + pressure_score


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def stats(trade_recs: list) -> dict:
    if not trade_recs:
        return dict(N=0, n_day=0, WR=0, AvgW=0, AvgL=0, PF=0, PnL=0, Sharpe=0, MaxDD=0)
    p = np.array([t["pnl"] for t in trade_recs])
    wins = p[p > 0]
    losses = p[p <= 0]
    WR = len(wins) / len(p)
    avg_w = float(wins.mean()) if len(wins) else 0.0
    avg_l = float(losses.mean()) if len(losses) else 0.0
    gross_w = float(wins.sum()) if len(wins) else 0.0
    gross_l = float(abs(losses.sum())) if len(losses) else 1e-8
    pf = gross_w / gross_l

    by_day = {}
    for t in trade_recs:
        by_day.setdefault(t["date"], 0.0)
        by_day[t["date"]] += t["pnl"]
    daily = np.array(list(by_day.values()))
    n_day = len(daily)
    sharpe = (daily.mean() / (daily.std() + 1e-8)) * np.sqrt(252) if n_day > 1 else 0.0

    cum = np.cumsum(p)
    roll_max = np.maximum.accumulate(cum)
    max_dd = float((cum - roll_max).min())

    return dict(
        N=len(p),
        n_day=n_day,
        WR=round(WR, 4),
        AvgW=round(avg_w, 2),
        AvgL=round(avg_l, 2),
        PF=round(pf, 3),
        PnL=round(float(p.sum()), 2),
        Sharpe=round(float(sharpe), 3),
        MaxDD=round(max_dd, 2),
    )


def print_stats_line(label: str, s: dict):
    if s["N"] == 0:
        print(f"  {label:35s} | NO TRADES")
        return
    trades_per_day = s["N"] / max(s["n_day"], 1)
    print(
        f"  {label:35s} | N={s['N']:4d} ({trades_per_day:.2f}/day)"
        f" | WR={s['WR']:.1%} | AvgW=${s['AvgW']:.0f}"
        f" | AvgL=${s['AvgL']:.0f} | PnL=${s['PnL']:,.0f}"
        f" | Sharpe={s['Sharpe']:.2f} | MaxDD=${s['MaxDD']:,.0f}"
    )


# ---------------------------------------------------------------------------
# Main RCAF backtest
# ---------------------------------------------------------------------------

def run_rcaf(
    bars_5m: pd.DataFrame,
    point_value: float = 5.0,
    cost_rt: float = 2.50,
    warmup_sessions: int = 21,
) -> tuple:
    """
    Run RCAF backtest.
    Returns (all_trades, regime_counts).
    """
    sessions = sorted(bars_5m.groupby(bars_5m.index.date), key=lambda x: x[0])
    sess_stat_list = build_session_stats(bars_5m)

    # Build date -> stats index
    date_to_stat = {s["date"]: s for s in sess_stat_list}

    all_trades = []
    regime_counts = {"negative": 0, "chop": 0, "trend": 0}

    for sess_idx, (date, session) in enumerate(sessions):
        if sess_idx < warmup_sessions:
            continue

        if len(session) < 8:
            continue

        sess_stat = date_to_stat.get(date)
        if sess_stat is None:
            continue

        atr_20d = sess_stat["atr_20d"]
        bar_atr = atr_20d / 13.0  # scale daily ATR to ~bar-level

        # Classify regime at 30-min mark
        regime, vwap_slope_30m = classify_regime(session, sess_stat, bar_atr)
        regime_counts[regime] += 1

        # Trend direction for trend regime (stored from 30m classification)
        trend_dir = 1 if vwap_slope_30m > 0 else -1
        # Confirm with close at bar 6 vs open
        open_session = float(session["open"].iloc[0])
        close_bar6 = float(session["close"].iloc[5]) if len(session) > 5 else open_session
        if regime == "trend":
            trend_dir = 1 if (close_bar6 > open_session and vwap_slope_30m > 0) else -1

        # Intraday loop state
        vwap_num = 0.0
        vwap_den = 0.0
        vol_history = []
        session_vwap_devs = []

        in_trade = False
        trade_dir = 0
        entry_px = 0.0
        stop_px = 0.0
        target_px = 0.0
        entry_bar = 0
        time_stop_bars = 0
        trade_subtype = ""
        trade_fs = 0.0

        trades_today = 0
        was_below_vwap_early = False
        consec_above_vwap = 0        # consecutive bar closes above VWAP
        prev_consec_above_vwap = 0   # value from previous bar (before this bar's close)

        if sess_idx % 20 == 0:
            print(f"  [sess {sess_idx}] {date}  regime={regime}  bar_atr={bar_atr:.2f}")

        for bar_i, (ts, bar) in enumerate(session.iterrows()):
            # Update running VWAP
            typical = (bar["high"] + bar["low"] + bar["close"]) / 3.0
            if bar["volume"] > 0 or vwap_den == 0:
                vwap_num += typical * max(bar["volume"], 1e-8)
                vwap_den += max(bar["volume"], 1e-8)
            vwap = vwap_num / (vwap_den + 1e-8)

            # Volume history for rolling median
            vol_history.append(float(bar["volume"]))
            vol_median = float(np.median(vol_history[-20:]))

            # Session VWAP std
            session_vwap_devs.append(float(bar["close"]) - vwap)
            if len(session_vwap_devs) >= 3:
                vwap_std = float(np.std(session_vwap_devs))
            else:
                vwap_std = bar_atr * 0.5

            # Track early weakness (bars 0-5)
            if bar_i < 6 and bar["low"] < vwap:
                was_below_vwap_early = True

            # Don't trade during regime classification period
            if bar_i < 6:
                continue

            force_exit = (bar_i == len(session) - 1)

            # ---- EXIT CHECK ----
            if in_trade:
                exit_px_val = None
                exit_reason = None

                if trade_dir == 1:  # LONG
                    if force_exit:
                        exit_px_val = float(bar["close"])
                        exit_reason = "session_end"
                    elif bar["high"] >= target_px:
                        exit_px_val = target_px
                        exit_reason = "target"
                    elif bar["low"] <= stop_px:
                        exit_px_val = stop_px
                        exit_reason = "stop"
                    elif (bar_i - entry_bar) >= time_stop_bars:
                        exit_px_val = float(bar["close"])
                        exit_reason = "time"
                else:  # SHORT (trade_dir == -1)
                    if force_exit:
                        exit_px_val = float(bar["close"])
                        exit_reason = "session_end"
                    elif bar["low"] <= target_px:
                        exit_px_val = target_px
                        exit_reason = "target"
                    elif bar["high"] >= stop_px:
                        exit_px_val = stop_px
                        exit_reason = "stop"
                    elif (bar_i - entry_bar) >= time_stop_bars:
                        exit_px_val = float(bar["close"])
                        exit_reason = "time"

                if exit_px_val is not None:
                    pnl = trade_dir * (exit_px_val - entry_px) * point_value - cost_rt
                    all_trades.append({
                        "date": str(date),
                        "bar": bar_i,
                        "regime": regime,
                        "subtype": trade_subtype,
                        "dir": trade_dir,
                        "entry": round(entry_px, 4),
                        "exit": round(exit_px_val, 4),
                        "exit_reason": exit_reason,
                        "fs": round(trade_fs, 4),
                        "pnl": round(pnl, 2),
                    })
                    in_trade = False

            # Track consecutive closes above/below VWAP (update BEFORE entry check)
            prev_consec_above_vwap = consec_above_vwap
            if float(bar["close"]) > vwap:
                consec_above_vwap += 1
            else:
                consec_above_vwap = 0

            # ---- ENTRY CHECK ----
            if in_trade or trades_today >= 2:
                continue

            # Compute current bar metrics
            rng_bar = bar["high"] - bar["low"] + 1e-6
            clv = (2.0 * bar["close"] - bar["high"] - bar["low"]) / rng_bar
            close_loc = (bar["close"] - bar["low"]) / rng_bar  # 0=at low, 1=at high
            body = abs(bar["close"] - bar["open"])
            body_frac = body / rng_bar
            lw = float(min(bar["open"], bar["close"])) - float(bar["low"])

            if regime == "negative":
                # --- Short failed rally ---
                fs_up = failure_score(bar, +1, vwap, bar_atr)
                if (
                    was_below_vwap_early
                    and prev_consec_above_vwap >= 3   # real rally: ≥3 bars above VWAP
                    and close_loc < 0.35              # current bar closes weak (rejection)
                    and bar["volume"] > 1.2 * vol_median
                    and fs_up > 0.3
                ):
                    in_trade = True
                    trade_dir = -1
                    entry_px = float(bar["close"])
                    stop_px = entry_px + 1.0 * bar_atr    # tighter: R:R = 2:1, breakeven = 33%
                    target_px = entry_px - 2.0 * bar_atr
                    entry_bar = bar_i
                    time_stop_bars = 8
                    trades_today += 1
                    trade_subtype = "neg_short"
                    trade_fs = fs_up

                # --- Capitulation long ---
                elif (
                    bar["close"] < vwap - 1.5 * vwap_std
                    and bar["close"] > bar["open"]  # closes up
                    and lw > 2.0 * (body + 1e-8)  # large lower wick
                    and bar["volume"] > 2.0 * vol_median
                ):
                    fs_down = failure_score(bar, -1, vwap, bar_atr)
                    in_trade = True
                    trade_dir = +1
                    entry_px = float(bar["close"])
                    stop_px = entry_px - 1.5 * bar_atr
                    target_px = vwap
                    entry_bar = bar_i
                    time_stop_bars = 6
                    trades_today += 1
                    trade_subtype = "neg_long_cap"
                    trade_fs = fs_down

            elif regime == "chop":
                # Only fade the SHORT side: upside extensions in chop revert more reliably.
                # Downside extensions ("buy the dip") fail in risk-off / negative tape.
                fade_dir = -1 if bar["close"] > vwap else +1
                fs_chop = failure_score(bar, -fade_dir, vwap, bar_atr)
                if (
                    fade_dir == -1                            # SHORT fades only
                    and abs(bar["close"] - vwap) > 1.5 * vwap_std
                    and fs_chop > 0.5
                    and body_frac < 0.45
                ):
                    in_trade = True
                    trade_dir = fade_dir
                    entry_px = float(bar["close"])
                    stop_px = entry_px - fade_dir * 0.5 * bar_atr  # tight: 0.5 ATR beyond entry
                    target_px = vwap
                    entry_bar = bar_i
                    time_stop_bars = 6
                    trades_today += 1
                    trade_subtype = "chop_fade"
                    trade_fs = fs_chop

            elif regime == "trend":
                pass  # trend continuation disabled — no edge confirmed in data

    return all_trades, regime_counts


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def regime_stats(trades: list, regime: str) -> dict:
    sub = [t for t in trades if t["regime"] == regime]
    return stats(sub)


def subtype_stats(trades: list, subtype: str) -> dict:
    sub = [t for t in trades if t["subtype"] == subtype]
    return stats(sub)


def direction_stats(trades: list, direction: int) -> dict:
    sub = [t for t in trades if t["dir"] == direction]
    return stats(sub)


def count_days(bars_5m: pd.DataFrame) -> int:
    return len(set(bars_5m.index.date))


# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("RUNNING RCAF — TRAINING (MES, warmup=21 sessions)")
print("=" * 80)

train_trades, train_regime_counts = run_rcaf(
    train_5m, point_value=5.0, cost_rt=2.50, warmup_sessions=21
)
train_days = count_days(train_5m)
train_st = stats(train_trades)

print(f"\nTotal training trades: {len(train_trades)}, over {train_days} days")

# ---------------------------------------------------------------------------
# Section 1 — Regime distribution
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 1: REGIME DISTRIBUTION (training)")
print("=" * 80)

total_regimes = sum(train_regime_counts.values())
for r, c in train_regime_counts.items():
    pct = c / max(total_regimes, 1)
    print(f"  {r:10s}: {c:4d} sessions ({pct:.1%})")

# ---------------------------------------------------------------------------
# Section 2 — Performance by regime
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 2: PERFORMANCE BY REGIME (training)")
print("=" * 80)

for regime in ("negative", "chop", "trend"):
    s = regime_stats(train_trades, regime)
    print_stats_line(f"Regime={regime}", s)

print_stats_line("ALL REGIMES", train_st)

# ---------------------------------------------------------------------------
# Section 3 — Direction split
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 3: DIRECTION SPLIT BY REGIME (training)")
print("=" * 80)

for regime in ("negative", "chop", "trend"):
    for subtype in ("neg_short", "neg_long_cap", "chop_fade", "trend_cont"):
        sub = [t for t in train_trades if t["regime"] == regime and t["subtype"] == subtype]
        if sub:
            s = stats(sub)
            print_stats_line(f"  {regime}/{subtype}", s)
    longs = [t for t in train_trades if t["regime"] == regime and t["dir"] == 1]
    shorts = [t for t in train_trades if t["regime"] == regime and t["dir"] == -1]
    if longs:
        print_stats_line(f"  {regime} LONG", stats(longs))
    if shorts:
        print_stats_line(f"  {regime} SHORT", stats(shorts))

# ---------------------------------------------------------------------------
# Section 4 — Failure score monotonicity (negative-regime shorts)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 4: FAILURE SCORE MONOTONICITY (neg-regime shorts)")
print("=" * 80)

neg_shorts = [t for t in train_trades if t["subtype"] == "neg_short"]
fs_mono_pass = False
if neg_shorts:
    fs_vals = np.array([abs(t["fs"]) for t in neg_shorts])
    pnls = np.array([t["pnl"] for t in neg_shorts])

    p33 = np.percentile(fs_vals, 33.3)
    p66 = np.percentile(fs_vals, 66.6)
    bucket_wrs = []
    for b_lo, b_hi, label in [(0, p33, "B1 (low FS)"), (p33, p66, "B2 (mid FS)"), (p66, 999, "B3 (high FS)")]:
        mask = (fs_vals >= b_lo) & (fs_vals < b_hi) if b_hi < 999 else (fs_vals >= b_lo)
        sub_pnl = pnls[mask]
        n = len(sub_pnl)
        wr = float((sub_pnl > 0).mean()) if n > 0 else 0.0
        avg_pnl = float(sub_pnl.mean()) if n > 0 else 0.0
        bucket_wrs.append(wr)
        print(f"  {label}: N={n}, WR={wr:.1%}, avg_pnl=${avg_pnl:.0f}")
    if len(bucket_wrs) == 3:
        fs_mono_pass = bucket_wrs[2] >= bucket_wrs[0]
        print(f"  Monotonicity (B3 WR >= B1 WR): {'PASS' if fs_mono_pass else 'FAIL'}")
else:
    print("  No negative-regime shorts to assess.")

# ---------------------------------------------------------------------------
# Section 5 — OOS MES (Jan-Feb 2026)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 5: OOS MES (Jan-Feb 2026, warmup=5 sessions)")
print("=" * 80)

oos_trades, oos_regime_counts = run_rcaf(
    oos_5m, point_value=5.0, cost_rt=2.50, warmup_sessions=5
)
oos_days = count_days(oos_5m)
oos_st = stats(oos_trades)
print_stats_line("OOS MES", oos_st)

print("\n  OOS regime distribution:")
oos_total_regimes = sum(oos_regime_counts.values())
for r, c in oos_regime_counts.items():
    pct = c / max(oos_total_regimes, 1)
    print(f"    {r:10s}: {c:4d} ({pct:.1%})")

for regime in ("negative", "chop", "trend"):
    s = regime_stats(oos_trades, regime)
    print_stats_line(f"  OOS {regime}", s)

# ---------------------------------------------------------------------------
# Section 6 — MNQ transfer
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 6: MNQ TRANSFER (2026 YTD, point_value=2.0, warmup=5)")
print("=" * 80)

mnq_trades, mnq_regime_counts = run_rcaf(
    mnq_5m, point_value=2.0, cost_rt=1.00, warmup_sessions=5
)
mnq_days = count_days(mnq_5m)
mnq_st = stats(mnq_trades)
print_stats_line("MNQ 2026 YTD", mnq_st)

print("\n  MNQ regime distribution:")
mnq_total_regimes = sum(mnq_regime_counts.values())
for r, c in mnq_regime_counts.items():
    pct = c / max(mnq_total_regimes, 1)
    print(f"    {r:10s}: {c:4d} ({pct:.1%})")

for regime in ("negative", "chop", "trend"):
    s = regime_stats(mnq_trades, regime)
    print_stats_line(f"  MNQ {regime}", s)

# ---------------------------------------------------------------------------
# Section 7 — Combined regime stats vs ORB baseline
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 7: COMBINED REGIME STATS vs ORB BASELINE")
print("=" * 80)

ORB_BASELINE = {
    "label": "ORB baseline (OOS Jan-Feb 2026, MNQ n=2)",
    "N": 32,
    "WR": 0.562,
    "PnL": 922.0,
    "Sharpe": 3.46,
    "MaxDD": -574.0,
}

print(f"  ORB baseline : N={ORB_BASELINE['N']}, WR={ORB_BASELINE['WR']:.1%}, "
      f"PnL=${ORB_BASELINE['PnL']:,.0f}, Sharpe={ORB_BASELINE['Sharpe']:.2f}, "
      f"MaxDD=${ORB_BASELINE['MaxDD']:,.0f}")
print_stats_line("RCAF Train MES", train_st)
print_stats_line("RCAF OOS  MES", oos_st)
print_stats_line("RCAF MNQ YTD", mnq_st)

# ---------------------------------------------------------------------------
# Section 8 — Monte Carlo (Topstep 50k combine)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 8: MONTE CARLO (10k paths × 60 days, Topstep 50k limits)")
print("=" * 80)

mc_pass = 0
mc_bust = 0
mc_finish_days = []

if train_trades:
    df_train = pd.DataFrame(train_trades)
    daily_pnl_series = df_train.groupby("date")["pnl"].sum().values

    np.random.seed(42)
    n_paths = 10_000
    n_days = 60
    profit_target = 3_000.0
    trail_dd_limit = -2_000.0
    daily_dd_limit = -1_000.0

    for _ in range(n_paths):
        sample = np.random.choice(daily_pnl_series, size=n_days, replace=True)
        cum_pnl = 0.0
        peak_pnl = 0.0
        result = "timeout"
        day_result = n_days
        for d_idx, dpnl in enumerate(sample):
            dpnl = max(dpnl, daily_dd_limit)
            cum_pnl += dpnl
            peak_pnl = max(peak_pnl, cum_pnl)
            trail_dd = cum_pnl - peak_pnl
            if trail_dd <= trail_dd_limit:
                result = "bust"
                day_result = d_idx + 1
                break
            if cum_pnl >= profit_target:
                result = "pass"
                day_result = d_idx + 1
                break
        if result == "pass":
            mc_pass += 1
            mc_finish_days.append(day_result)
        elif result == "bust":
            mc_bust += 1

    p_pass = mc_pass / n_paths
    p_bust = mc_bust / n_paths
    median_days = int(np.median(mc_finish_days)) if mc_finish_days else n_days

    print(f"  P(pass)                    = {p_pass:.1%}")
    print(f"  P(bust)                    = {p_bust:.1%}")
    print(f"  Median days to pass        = {median_days}")
    print(f"  (profit_target=${profit_target:,.0f}, trail_dd={trail_dd_limit:,.0f}, "
          f"daily_dd={daily_dd_limit:,.0f})")
else:
    p_pass = 0.0
    p_bust = 1.0
    median_days = n_days
    print("  No trades — skipping Monte Carlo.")

# ---------------------------------------------------------------------------
# Section 9 — Verdict
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

train_sharpe = train_st["Sharpe"]
oos_sharpe = oos_st["Sharpe"]
neg_trades = [t for t in train_trades if t["regime"] == "negative"]
neg_wr = float((np.array([t["pnl"] for t in neg_trades]) > 0).mean()) if neg_trades else 0.0

verdict_pass = (
    train_sharpe > 1.5
    and oos_sharpe > 0.0
    and neg_wr >= 0.52
)

print(f"  Train Sharpe          : {train_sharpe:.3f}  (need >1.5)")
print(f"  OOS Sharpe            : {oos_sharpe:.3f}  (need >0.0)")
print(f"  Negative regime WR    : {neg_wr:.1%}  (need >=52%)")
print(f"  FS monotonicity       : {'PASS' if fs_mono_pass else 'FAIL'}")
print(f"  P(pass combine)       : {p_pass:.1%}")
print(f"\n  *** VERDICT: {'PASS' if verdict_pass else 'KILL'} ***")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

results = {
    "verdict": "PASS" if verdict_pass else "KILL",
    "criteria": {
        "train_sharpe_gt_1p5": train_sharpe > 1.5,
        "oos_sharpe_gt_0": oos_sharpe > 0.0,
        "neg_regime_wr_ge_52pct": neg_wr >= 0.52,
    },
    "regime_distribution_train": train_regime_counts,
    "regime_distribution_oos": oos_regime_counts,
    "regime_distribution_mnq": mnq_regime_counts,
    "train_stats": {
        "all": train_st,
        "negative": regime_stats(train_trades, "negative"),
        "chop": regime_stats(train_trades, "chop"),
        "trend": regime_stats(train_trades, "trend"),
        "neg_short": subtype_stats(train_trades, "neg_short"),
        "neg_long_cap": subtype_stats(train_trades, "neg_long_cap"),
        "chop_fade": subtype_stats(train_trades, "chop_fade"),
        "trend_cont": subtype_stats(train_trades, "trend_cont"),
    },
    "oos_stats": {
        "all": oos_st,
        "negative": regime_stats(oos_trades, "negative"),
        "chop": regime_stats(oos_trades, "chop"),
        "trend": regime_stats(oos_trades, "trend"),
    },
    "mnq_stats": {
        "all": mnq_st,
        "negative": regime_stats(mnq_trades, "negative"),
        "chop": regime_stats(mnq_trades, "chop"),
        "trend": regime_stats(mnq_trades, "trend"),
    },
    "failure_score_monotonicity": {
        "neg_shorts_total": len(neg_shorts),
        "mono_pass": bool(fs_mono_pass),
    },
    "monte_carlo": {
        "p_pass": round(p_pass, 4),
        "p_bust": round(p_bust, 4),
        "median_days_to_pass": median_days,
        "n_paths": 10_000,
        "n_days": 60,
        "profit_target": 3_000,
        "trail_dd_limit": -2_000,
        "daily_dd_limit": -1_000,
    },
    "orb_baseline": ORB_BASELINE,
}

out = ROOT / "rule_based_v1/diagnostics/rcaf_results.json"
out.write_text(json.dumps(results, indent=2, default=str))
print(f"\nSaved → {out}")
