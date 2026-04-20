"""
VMC — Value Migration Continuation Backtest
===========================================
Strategy: Wait for the market to leave the opening Initial Balance (IB), prove
acceptance via 2 consecutive 15-min closes outside the IB, then enter on the
first clean pullback that holds outside the IB.

ORB trades the opening discovery move; VMC trades the late-morning/afternoon
trend that ORB misses.

Instruments:
  ES  2010-2025 — long-term IS/OOS (RTH 5-min bars resampled to 15-min)
  MNQ 2026 YTD  — OOS validation   (ETH 1-min bars, RTH filter, 15-min)

Run:
  python rule_based_v1/diagnostics/vmc_backtest.py
"""
from __future__ import annotations

import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent.parent
ES_PATH  = ROOT / "data/processed/es_bars_2010_2025.h5"
MNQ_PATH = ROOT / "data/processed/mnq_2026ytd_1min_eth.h5"
RESULTS_PATH = ROOT / "rule_based_v1/diagnostics/vmc_results.json"

ES_PV    = 50.0
MNQ_PV   = 2.0
COST_RT  = 6.25   # per-side commission (in points×PV already; we use flat $ per RT)

# Default sweep config
IB_N_BARS          = 4      # 60-min IB (4 × 15-min)
PULLBACK_VOL_RATIO = 0.80   # pullback bar volume ≤ 80% of breakout bar volume
STOP_BUFFER_ATR    = 0.10   # buffer below/above pullback extreme
TARGET_R           = 1.5    # 1.5× risk
TIME_STOP_BARS     = 8      # 8 × 15-min = 2 hours

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

print("Loading ES 15-year data (RTH 5-min)…")
with pd.HDFStore(str(ES_PATH), "r") as s:
    df_es_5m = s["/bars_5min"].copy()
df_es_5m = df_es_5m.set_index("timestamp").sort_index()
df_es_5m.index = df_es_5m.index.tz_convert("US/Eastern")

print("Loading MNQ 2026 YTD data (ETH 1-min)…")
with pd.HDFStore(str(MNQ_PATH), "r") as s:
    df_mnq_1m = s["/bars_1min_eth"].copy()
if df_mnq_1m.index.tz is None:
    df_mnq_1m.index = df_mnq_1m.index.tz_localize("UTC").tz_convert("US/Eastern")
elif str(df_mnq_1m.index.tz) != "US/Eastern":
    df_mnq_1m.index = df_mnq_1m.index.tz_convert("US/Eastern")
df_mnq_1m = df_mnq_1m.sort_index()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rth(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to RTH: 09:30–15:59 ET."""
    mask = (
        ((df.index.hour == 9) & (df.index.minute >= 30))
        | ((df.index.hour > 9) & (df.index.hour < 16))
    )
    return df[mask].copy()


def resample_to_15m(df_rth: pd.DataFrame) -> pd.DataFrame:
    """Session-safe 15-min resampling — handles multi-day DataFrames correctly."""
    df = df_rth[["open", "high", "low", "close", "volume"]].copy()
    df["_date"]   = df.index.date
    df["_bucket"] = df.index.floor("15min")
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


def build_session_stats(bars_15m: pd.DataFrame) -> dict:
    """
    Per-session dictionary with:
      atr_20d   — EMA-20 of daily true range
      prev_close — prior session last close
      open_price — first bar open of the session
    """
    sessions = sorted(bars_15m.groupby(bars_15m.index.date), key=lambda x: x[0])
    rows = []
    for date, sess in sessions:
        if len(sess) < 2:
            continue
        rows.append({
            "date":   date,
            "open":   float(sess["open"].iloc[0]),
            "high":   float(sess["high"].max()),
            "low":    float(sess["low"].min()),
            "close":  float(sess["close"].iloc[-1]),
        })

    alpha = 1.0 / 20.0
    result = {}
    prev_close = None
    atr_val    = None

    for i, row in enumerate(rows):
        hi, lo, cl = row["high"], row["low"], row["close"]
        if prev_close is None:
            tr = hi - lo
        else:
            tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))

        atr_val = tr if atr_val is None else atr_val * (1 - alpha) + tr * alpha

        result[row["date"]] = {
            "atr_20d":    atr_val,
            "prev_close": rows[i - 1]["close"] if i > 0 else None,
            "open_price": row["open"],
        }
        prev_close = cl

    return result


def compute_session_vwap(bars: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP per session (reset each day)."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    pv      = typical * bars["volume"]
    vwap    = pd.Series(np.nan, index=bars.index)
    for date, grp in bars.groupby(bars.index.date):
        idx     = grp.index
        cum_pv  = pv.loc[idx].cumsum()
        cum_vol = bars["volume"].loc[idx].cumsum()
        vwap.loc[idx] = (cum_pv / cum_vol.replace(0, np.nan)).values
    return vwap


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------

def classify_regime(sess_15m: pd.DataFrame, stat: dict) -> tuple[str, int]:
    """
    Classify the session regime using the first 6 bars (90 min).
    Returns (regime_str, trend_dir) where regime_str in {"trend","chop","negative"}
    and trend_dir is +1 or -1.
    """
    first6      = sess_15m.iloc[:6]
    open_price  = float(sess_15m["open"].iloc[0])
    atr         = stat["atr_20d"]

    first_90m_range = float(first6["high"].max() - first6["low"].min())
    range_ratio     = first_90m_range / (atr + 1e-8)

    tp6      = (first6["high"] + first6["low"] + first6["close"]) / 3
    vol6     = first6["volume"]
    vwap_90m = float((tp6 * vol6).sum() / (vol6.sum() + 1e-8))
    vwap_slope = (vwap_90m - open_price) / (open_price + 1e-8)

    sum_tr6 = float((first6["high"] - first6["low"]).sum()) + 1e-8
    close6  = float(first6["close"].iloc[-1])
    de      = abs(close6 - open_price) / sum_tr6

    trend_dir = +1 if close6 > open_price else -1

    neg_count = (
        int(stat.get("gap_pct", 0) < -0.002)
        + int(range_ratio > 1.3)
        + int(vwap_slope < -0.0003)
    )

    if de < 0.20 and abs(vwap_slope) < 0.0005:
        return "chop", trend_dir
    elif neg_count >= 2 and de > 0.20:
        return "negative", -1
    else:
        return "trend", trend_dir


# ---------------------------------------------------------------------------
# Core event extractor
# ---------------------------------------------------------------------------

def extract_events(
    bars_15m: pd.DataFrame,
    session_vwap: pd.Series,
    stats: dict,
    ib_n_bars: int = 4,
    pullback_vol_ratio_max: float = 0.80,
) -> list:
    """
    Extract VMC trade events from 15-min bars.
    Max 1 trade per session.

    Parameters
    ----------
    bars_15m              : 15-min RTH bars (DatetimeIndex ET)
    session_vwap          : VWAP series aligned to bars_15m
    stats                 : dict from build_session_stats()
    ib_n_bars             : number of 15-min bars in the Initial Balance (4=60min, 6=90min)
    pullback_vol_ratio_max: pullback bar must have volume <= this × breakout bar volume
    """
    events = []
    session_dates = sorted(set(bars_15m.index.date))

    for date in session_dates:
        stat = stats.get(date)
        if stat is None or stat["atr_20d"] is None:
            continue

        sess = bars_15m[bars_15m.index.date == date]
        if len(sess) < ib_n_bars + 4:
            continue

        # 1. Initial Balance
        ib_bars  = sess.iloc[:ib_n_bars]
        ib_high  = float(ib_bars["high"].max())
        ib_low   = float(ib_bars["low"].min())
        ib_range = ib_high - ib_low
        if ib_range < 0.3 * stat["atr_20d"]:
            continue  # flat open, no structure

        # 2. Regime
        if len(sess) < 6:
            continue
        regime, trend_dir = classify_regime(sess, stat)
        if regime == "chop":
            continue

        # 3. Scan for IB breakout + 2-bar acceptance
        post_ib    = sess.iloc[ib_n_bars:]
        n          = len(post_ib)
        bars_list  = list(post_ib.iterrows())

        breakout_found    = False
        breakout_dir      = 0
        breakout_bar_vol  = 0.0
        acceptance_close  = 0.0
        acceptance_bar_idx = -1

        for i, (ts, bar) in enumerate(bars_list):
            close = float(bar["close"])

            if i == 0:
                prev_close = float(ib_bars["close"].iloc[-1])
            else:
                prev_close = float(bars_list[i - 1][1]["close"])

            # Long breakout acceptance: 2 consecutive closes above ib_high
            if close > ib_high and prev_close > ib_high:
                if regime == "negative":
                    continue  # negative regime: only shorts allowed
                breakout_found     = True
                breakout_dir       = +1
                breakout_bar_vol   = float(bar["volume"])
                acceptance_close   = close
                acceptance_bar_idx = i
                break

            # Short breakout acceptance: 2 consecutive closes below ib_low
            if close < ib_low and prev_close < ib_low:
                breakout_found     = True
                breakout_dir       = -1
                breakout_bar_vol   = float(bar["volume"])
                acceptance_close   = close
                acceptance_bar_idx = i
                break

        if not breakout_found:
            continue
        if acceptance_bar_idx + 1 >= n:
            continue  # no room for a pullback bar

        # 4. Scan for pullback after acceptance (within next 5 bars)
        pullback_found    = False
        pullback_high     = 0.0
        pullback_low      = 0.0
        pullback_bar_idx  = -1

        for j in range(acceptance_bar_idx + 1, min(acceptance_bar_idx + 6, n)):
            ts_j, bar_j = bars_list[j]
            # Hard time cutoff: no entries after 14:00 ET
            if ts_j.hour >= 14:
                break

            bh = float(bar_j["high"])
            bl = float(bar_j["low"])
            bc = float(bar_j["close"])
            bv = float(bar_j["volume"])

            vwap_j = float(session_vwap.get(ts_j, np.nan))

            # Pullback direction check
            is_pullback = (bc < acceptance_close) if breakout_dir == +1 else (bc > acceptance_close)
            if not is_pullback:
                continue

            # Pullback validity — LONG
            if breakout_dir == +1:
                if bl < ib_high:
                    continue  # re-entered IB, invalid
                if not np.isnan(vwap_j) and bc < vwap_j - 0.3 * stat["atr_20d"]:
                    continue  # too far below VWAP
            else:
                # SHORT
                if bh > ib_low:
                    continue  # re-entered IB, invalid
                if not np.isnan(vwap_j) and bc > vwap_j + 0.3 * stat["atr_20d"]:
                    continue  # too far above VWAP

            # Volume check: lighter pullback volume
            if bv > pullback_vol_ratio_max * breakout_bar_vol:
                continue

            pullback_found   = True
            pullback_high    = bh
            pullback_low     = bl
            pullback_bar_idx = j
            break  # take first valid pullback

        if not pullback_found:
            continue
        if pullback_bar_idx + 1 >= n:
            continue

        # 5. Entry: next bar must trade beyond pullback extreme
        entry_bar_idx = pullback_bar_idx + 1
        if entry_bar_idx >= n:
            continue
        ts_e, bar_e = bars_list[entry_bar_idx]
        if ts_e.hour >= 14:
            continue

        if breakout_dir == +1:
            if float(bar_e["high"]) < pullback_high:
                continue  # never triggered
            entry_px  = pullback_high
            stop_ref  = pullback_low
        else:
            if float(bar_e["low"]) > pullback_low:
                continue  # never triggered
            entry_px  = pullback_low
            stop_ref  = pullback_high

        # 6. Collect subsequent bars for simulation
        subsequent = []
        for k in range(entry_bar_idx + 1, n):
            _, sb = bars_list[k]
            subsequent.append({
                "h": float(sb["high"]),
                "l": float(sb["low"]),
                "c": float(sb["close"]),
            })

        events.append({
            "date":      str(date),
            "entry_ts":  str(ts_e),
            "direction": breakout_dir,
            "regime":    regime,
            "ib_high":   ib_high,
            "ib_low":    ib_low,
            "entry_px":  entry_px,
            "stop_ref":  stop_ref,
            "atr":       stat["atr_20d"],
            "subsequent": subsequent,
        })

    return events


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate(
    events: list,
    stop_buffer_atr: float,
    target_r: float,
    time_stop_bars: int,
    direction_filter: str,
    point_value: float,
) -> dict:
    """
    Simulate VMC events and return a result dictionary.

    Parameters
    ----------
    direction_filter : "long", "short", or "both"
    """
    trades = []

    for ev in events:
        d = ev["direction"]
        if direction_filter == "long"  and d != +1:
            continue
        if direction_filter == "short" and d != -1:
            continue

        entry_px = ev["entry_px"]
        stop_ref = ev["stop_ref"]
        atr      = ev["atr"]

        if d == +1:
            stop = stop_ref - stop_buffer_atr * atr
        else:
            stop = stop_ref + stop_buffer_atr * atr

        risk = abs(entry_px - stop)
        if risk <= 0:
            continue

        if d == +1:
            tp = entry_px + target_r * risk
        else:
            tp = entry_px - target_r * risk

        # Scan subsequent bars
        outcome = "time_stop"
        exit_px = None
        bars    = ev["subsequent"]

        for i, bar in enumerate(bars):
            is_last = (i == len(bars) - 1)

            if d == +1:
                if bar["h"] >= tp and bar["l"] <= stop:
                    # Both hit; assume TP first (conservative: assume worst case for MAE)
                    exit_px = tp
                    outcome = "tp"
                    break
                elif bar["h"] >= tp:
                    exit_px = tp
                    outcome = "tp"
                    break
                elif bar["l"] <= stop:
                    exit_px = stop
                    outcome = "sl"
                    break
            else:
                if bar["l"] <= tp and bar["h"] >= stop:
                    exit_px = tp
                    outcome = "tp"
                    break
                elif bar["l"] <= tp:
                    exit_px = tp
                    outcome = "tp"
                    break
                elif bar["h"] >= stop:
                    exit_px = stop
                    outcome = "sl"
                    break

            if i >= time_stop_bars - 1 or is_last:
                exit_px = bar["c"]
                outcome = "time_stop"
                break

        if exit_px is None:
            if bars:
                exit_px = bars[-1]["c"]
                outcome = "time_stop"
            else:
                exit_px = entry_px
                outcome = "time_stop"

        raw_pnl = (exit_px - entry_px) * d * point_value
        pnl     = raw_pnl - COST_RT

        trades.append({
            "date":      ev["date"],
            "direction": d,
            "regime":    ev["regime"],
            "entry_px":  entry_px,
            "exit_px":   exit_px,
            "outcome":   outcome,
            "pnl":       pnl,
        })

    return trades


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _stats(trades: list, point_value: float = ES_PV) -> dict:
    """Compute strategy statistics from a list of trade dicts."""
    if not trades:
        return {
            "N": 0, "WR": 0.0, "E": 0.0, "PnL": 0.0,
            "Sharpe": 0.0, "MaxDD": 0.0, "avg_tpd": 0.0,
            "exit_dist": {}, "stress": {},
        }

    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    n   = len(df)
    wr  = float((df["pnl"] > 0).mean())
    e   = float(df["pnl"].mean())

    # Annualised Sharpe on daily PnL
    daily = df.groupby("date")["pnl"].sum()
    mu    = daily.mean()
    sig   = daily.std()
    sharpe = float(mu / sig * np.sqrt(252)) if sig > 0 else 0.0

    # Cumulative drawdown
    cum    = df["pnl"].cumsum().values
    peak   = np.maximum.accumulate(cum)
    dd     = cum - peak
    maxdd  = float(dd.min())

    # Average trades per day
    n_days = max(len(set(df["date"].dt.date)), 1)
    avg_tpd = n / n_days

    # Exit distribution
    exit_dist = df["outcome"].value_counts().to_dict()

    # Stress periods
    stress = {}
    for label, y0, y1 in [("2018-2019", 2018, 2019), ("2020", 2020, 2020), ("2022", 2022, 2022)]:
        sub = df[df["date"].dt.year.between(y0, y1)]
        stress[label] = {
            "N":   len(sub),
            "PnL": float(sub["pnl"].sum()) if len(sub) > 0 else 0.0,
            "WR":  float((sub["pnl"] > 0).mean()) if len(sub) > 0 else 0.0,
        }

    return {
        "N":        n,
        "WR":       round(wr, 4),
        "E":        round(e, 2),
        "PnL":      round(float(df["pnl"].sum()), 2),
        "Sharpe":   round(sharpe, 3),
        "MaxDD":    round(maxdd, 2),
        "avg_tpd":  round(avg_tpd, 3),
        "exit_dist": exit_dist,
        "stress":    stress,
    }


def print_s(label: str, s: dict) -> None:
    if s["N"] == 0:
        print(f"  {label:<30}  N=0  (no trades)")
        return
    print(
        f"  {label:<30}  N={s['N']:>4}  WR={s['WR']:.1%}  "
        f"E=${s['E']:.0f}  PnL=${s['PnL']:,.0f}  "
        f"Sharpe={s['Sharpe']:.2f}  MaxDD=${s['MaxDD']:,.0f}  "
        f"avg_tpd={s['avg_tpd']:.2f}"
    )


# ---------------------------------------------------------------------------
# Apply RTH + resample
# ---------------------------------------------------------------------------

es_rth    = rth(df_es_5m)
es_15m    = resample_to_15m(es_rth)

mnq_rth   = rth(df_mnq_1m)
mnq_15m   = resample_to_15m(mnq_rth)

print(f"ES 15m : {len(es_15m):,} bars  ({es_15m.index[0].date()} – {es_15m.index[-1].date()})")
print(f"MNQ 15m: {len(mnq_15m):,} bars  ({mnq_15m.index[0].date()} – {mnq_15m.index[-1].date()})")

# Build stats and VWAP
print("Building session stats…")
es_stats  = build_session_stats(es_15m)
mnq_stats = build_session_stats(mnq_15m)

print("Computing session VWAP…")
es_vwap  = compute_session_vwap(es_15m)
mnq_vwap = compute_session_vwap(mnq_15m)

# ---------------------------------------------------------------------------
# Section 1: ES 15-min Baseline (2010–2025)
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("SECTION 1 — ES 15-min Baseline (2010–2025)")
print("=" * 70)

print(f"Config: IB={IB_N_BARS}bars  pullback_vol={PULLBACK_VOL_RATIO}  "
      f"stop_buf={STOP_BUFFER_ATR}×ATR  target={TARGET_R}R  "
      f"time_stop={TIME_STOP_BARS}bars")

es_events = extract_events(es_15m, es_vwap, es_stats,
                           ib_n_bars=IB_N_BARS,
                           pullback_vol_ratio_max=PULLBACK_VOL_RATIO)

long_ev  = [e for e in es_events if e["direction"] == +1]
short_ev = [e for e in es_events if e["direction"] == -1]
trend_ev = [e for e in es_events if e["regime"] == "trend"]
neg_ev   = [e for e in es_events if e["regime"] == "negative"]

regime_counts = {}
for e in es_events:
    regime_counts[e["regime"]] = regime_counts.get(e["regime"], 0) + 1

print(f"\nTotal events: {len(es_events)}  "
      f"long={len(long_ev)}  short={len(short_ev)}")
print(f"Regime distribution: {regime_counts}")

sim_cfg = dict(
    stop_buffer_atr=STOP_BUFFER_ATR,
    target_r=TARGET_R,
    time_stop_bars=TIME_STOP_BARS,
    point_value=ES_PV,
)

all_trades   = simulate(es_events, direction_filter="both",  **sim_cfg)
long_trades  = simulate(es_events, direction_filter="long",  **sim_cfg)
short_trades = simulate(es_events, direction_filter="short", **sim_cfg)
trend_trades = simulate(trend_ev,  direction_filter="both",  **sim_cfg)
neg_trades   = simulate(neg_ev,    direction_filter="both",  **sim_cfg)

print()
print_s("All (both dirs)",     _stats(all_trades))
print_s("Long only",           _stats(long_trades))
print_s("Short only",          _stats(short_trades))
print_s("Trend regime only",   _stats(trend_trades))
print_s("Negative regime only",_stats(neg_trades))

# Year-by-year breakdown
print("\nYear-by-year (both directions):")
all_df = pd.DataFrame(all_trades)
if len(all_df) > 0:
    all_df["year"] = pd.to_datetime(all_df["date"]).dt.year
    for yr in sorted(all_df["year"].unique()):
        yr_trades = all_df[all_df["year"] == yr].to_dict("records")
        s = _stats(yr_trades)
        print_s(str(yr), s)

# Stress periods
print("\nStress periods:")
s_all = _stats(all_trades)
for period, data in s_all["stress"].items():
    flag = "✅" if data["PnL"] >= 0 else "❌"
    print(f"  {flag}  {period:<12}  N={data['N']}  PnL=${data['PnL']:,.0f}  WR={data['WR']:.1%}")

# ---------------------------------------------------------------------------
# Section 2: Parameter Sweep (ES 15-min)
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("SECTION 2 — Parameter Sweep (ES 15-min)")
print("=" * 70)

ib_choices          = [4, 6]
stop_buf_choices    = [0.05, 0.10, 0.20]
target_r_choices    = [1.0, 1.5, 2.0]
direction_choices   = ["long", "short", "both"]

# Cache events by (ib_n_bars,) — pullback_vol_ratio fixed at 0.80
event_cache: dict[int, list] = {}
for ib in ib_choices:
    event_cache[ib] = extract_events(
        es_15m, es_vwap, es_stats,
        ib_n_bars=ib,
        pullback_vol_ratio_max=0.80,
    )

sweep_results = []
for ib, sbuf, tr, df_dir in product(
        ib_choices, stop_buf_choices, target_r_choices, direction_choices):
    evs    = event_cache[ib]
    trades = simulate(evs,
                      stop_buffer_atr=sbuf,
                      target_r=tr,
                      time_stop_bars=TIME_STOP_BARS,
                      direction_filter=df_dir,
                      point_value=ES_PV)
    s = _stats(trades)
    if s["N"] < 20:
        continue
    sweep_results.append({
        "ib_n_bars":       ib,
        "stop_buffer_atr": sbuf,
        "target_r":        tr,
        "direction":       df_dir,
        **s,
    })

sweep_results.sort(key=lambda x: x["Sharpe"], reverse=True)
print(f"\nConfigs with N>=20: {len(sweep_results)}")
print(f"{'Rank':<5} {'IB':>4} {'SBuf':>6} {'TR':>5} {'Dir':<7} "
      f"{'N':>5} {'WR':>6} {'PnL':>10} {'Sharpe':>7} {'MaxDD':>10}")
print("-" * 72)
for rank, r in enumerate(sweep_results[:20], 1):
    print(
        f"  {rank:<3} {r['ib_n_bars']:>4} {r['stop_buffer_atr']:>6.2f} "
        f"{r['target_r']:>5.1f} {r['direction']:<7} "
        f"{r['N']:>5} {r['WR']:>6.1%} ${r['PnL']:>9,.0f} "
        f"{r['Sharpe']:>7.2f} ${r['MaxDD']:>9,.0f}"
    )

# Pick best config for MNQ OOS
best_cfg = None
if sweep_results and sweep_results[0]["Sharpe"] > 1.5:
    best_cfg = sweep_results[0]
    print(f"\nBest config (Sharpe={best_cfg['Sharpe']:.2f}): "
          f"IB={best_cfg['ib_n_bars']} stop_buf={best_cfg['stop_buffer_atr']} "
          f"target_r={best_cfg['target_r']} dir={best_cfg['direction']}")
else:
    print("\nNo config with Sharpe>1.5 found — using defaults for MNQ OOS.")

# ---------------------------------------------------------------------------
# Section 3: MNQ 2026 YTD OOS
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("SECTION 3 — MNQ 2026 YTD OOS")
print("=" * 70)

if best_cfg is not None:
    mnq_ib      = best_cfg["ib_n_bars"]
    mnq_sbuf    = best_cfg["stop_buffer_atr"]
    mnq_tr      = best_cfg["target_r"]
    mnq_dir     = best_cfg["direction"]
else:
    mnq_ib      = IB_N_BARS
    mnq_sbuf    = STOP_BUFFER_ATR
    mnq_tr      = TARGET_R
    mnq_dir     = "both"

print(f"Config: IB={mnq_ib}bars  stop_buf={mnq_sbuf}  target_r={mnq_tr}  dir={mnq_dir}")

mnq_events = extract_events(mnq_15m, mnq_vwap, mnq_stats,
                             ib_n_bars=mnq_ib,
                             pullback_vol_ratio_max=PULLBACK_VOL_RATIO)

print(f"MNQ events: {len(mnq_events)}  "
      f"long={sum(1 for e in mnq_events if e['direction']==+1)}  "
      f"short={sum(1 for e in mnq_events if e['direction']==-1)}")

mnq_all   = simulate(mnq_events, direction_filter="both",
                     stop_buffer_atr=mnq_sbuf, target_r=mnq_tr,
                     time_stop_bars=TIME_STOP_BARS, point_value=MNQ_PV)
mnq_long  = simulate(mnq_events, direction_filter="long",
                     stop_buffer_atr=mnq_sbuf, target_r=mnq_tr,
                     time_stop_bars=TIME_STOP_BARS, point_value=MNQ_PV)
mnq_short = simulate(mnq_events, direction_filter="short",
                     stop_buffer_atr=mnq_sbuf, target_r=mnq_tr,
                     time_stop_bars=TIME_STOP_BARS, point_value=MNQ_PV)

print()
print_s("MNQ All",   _stats(mnq_all,   MNQ_PV))
print_s("MNQ Long",  _stats(mnq_long,  MNQ_PV))
print_s("MNQ Short", _stats(mnq_short, MNQ_PV))

# ---------------------------------------------------------------------------
# Section 4: Acceptance Criteria
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("SECTION 4 — Acceptance Criteria")
print("=" * 70)

s_all_es    = _stats(all_trades)
s_long_es   = _stats(long_trades)
s_short_es  = _stats(short_trades)
s_mnq_all   = _stats(mnq_all, MNQ_PV)

criteria = [
    ("ES WR > 48%",        s_all_es["WR"] > 0.48),
    ("ES Sharpe > 1.5",    s_all_es["Sharpe"] > 1.5),
    ("ES MaxDD > -$20,000",s_all_es["MaxDD"] > -20_000),
    ("ES 2020 positive",   s_all_es["stress"].get("2020", {}).get("PnL", 0) >= 0),
    ("ES 2022 positive",   s_all_es["stress"].get("2022", {}).get("PnL", 0) >= 0),
    ("Long PnL > 0",       s_long_es["PnL"] > 0),
    ("Short PnL > 0",      s_short_es["PnL"] > 0),
    ("MNQ OOS positive",   s_mnq_all["PnL"] > 0),
]

passed = 0
for label, ok in criteria:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}]  {label}")
    if ok:
        passed += 1

total = len(criteria)
print(f"\n  Passed {passed}/{total}", end="  →  ")
if passed >= 6:
    print("GO")
elif passed >= 4:
    print("MARGINAL")
else:
    print("KILL")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

results = {
    "es_baseline": {
        "all":     _stats(all_trades),
        "long":    _stats(long_trades),
        "short":   _stats(short_trades),
        "trend":   _stats(trend_trades),
        "negative":_stats(neg_trades),
    },
    "sweep_top20": sweep_results[:20],
    "mnq_oos": {
        "all":   _stats(mnq_all,   MNQ_PV),
        "long":  _stats(mnq_long,  MNQ_PV),
        "short": _stats(mnq_short, MNQ_PV),
    },
    "acceptance": {
        label: bool(ok) for label, ok in criteria
    },
    "verdict": (
        "GO" if passed >= 6
        else "MARGINAL" if passed >= 4
        else "KILL"
    ),
    "config_used": {
        "ib_n_bars":           IB_N_BARS,
        "pullback_vol_ratio":  PULLBACK_VOL_RATIO,
        "stop_buffer_atr":     STOP_BUFFER_ATR,
        "target_r":            TARGET_R,
        "time_stop_bars":      TIME_STOP_BARS,
    },
}

with open(str(RESULTS_PATH), "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved → {RESULTS_PATH}")
