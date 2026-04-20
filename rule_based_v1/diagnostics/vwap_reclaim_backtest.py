"""
VWAP Reclaim After Gap
======================
Standalone research backtest — pure pandas/numpy, no BacktestEngine.

Edge hypothesis: When the overnight gap is meaningful (>= min_gap x ATR),
the first time price reclaims the session VWAP during RTH represents
institutional absorption of the gap direction. Entering on the VWAP
reclaim close (in the gap direction) gives a structural entry with
prev_close (gap fill) as a natural target.

Mode A — gap-down reclaim (LONG):
  open < prev_close by >= min_gap_atr x ATR → wait for first bar that
  CLOSES above session VWAP → enter LONG → target = prev_close (gap fill)

Mode B — gap-up reclaim (SHORT):
  open > prev_close by >= min_gap_atr x ATR → wait for first bar that
  CLOSES below session VWAP → enter SHORT → target = prev_close (gap fill)

Long-term test: ES 5-min 2010-2025 (3,995 sessions).
Transfer tests: MES 1-min, MNQ 1-min 2026 YTD.
"""
import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ES_PV   = 50.0
MES_PV  = 5.0
MNQ_PV  = 2.0
COST_RT = 6.25

# Default config
MIN_GAP_ATR    = 0.20
STOP_ATR_MULT  = 0.75
TIME_GATE_END  = 12   # exclusive hour
TIME_STOP_BARS = 24   # bars after entry (5-min = 2 hr)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
print("Loading ES 15-year data…")
with pd.HDFStore(str(ROOT / "data/processed/es_bars_2010_2025.h5"), "r") as _s:
    df_es = _s["/bars_5min"].copy()
df_es = df_es.set_index("timestamp").sort_index()
df_es.index = df_es.index.tz_convert("US/Eastern")

print("Loading MES 1-min data…")
with pd.HDFStore(str(ROOT / "data/processed/mes_1m_bars_cache.h5"), "r") as _s:
    df_mes_raw = _s["/bars_1m"].set_index("timestamp")
df_mes_raw.index = pd.to_datetime(df_mes_raw.index, utc=True).tz_convert("US/Eastern")
df_mes_raw = df_mes_raw.sort_index()

print("Loading MNQ 1-min data…")
with pd.HDFStore(str(ROOT / "data/processed/mnq_2026ytd_1min_eth.h5"), "r") as _s:
    df_mnq_raw = _s["/bars_1min_eth"].copy()
if df_mnq_raw.index.tz is None:
    df_mnq_raw.index = df_mnq_raw.index.tz_localize("UTC").tz_convert("US/Eastern")
elif str(df_mnq_raw.index.tz) != "US/Eastern":
    df_mnq_raw.index = df_mnq_raw.index.tz_convert("US/Eastern")
df_mnq_raw = df_mnq_raw.sort_index()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rth(df: pd.DataFrame) -> pd.DataFrame:
    """RTH mask 09:30–16:00 ET."""
    mask = (
        ((df.index.hour == 9) & (df.index.minute >= 30))
        | ((df.index.hour > 9) & (df.index.hour < 16))
    )
    return df[mask].copy()


def resample_1m_to_bar(df1m: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    df = df1m[["open", "high", "low", "close", "volume"]].copy()
    df["_date"]   = df.index.date
    df["_bucket"] = df.index.floor(freq)
    g = df.groupby(["_date", "_bucket"]).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),    close=("close", "last"),
        volume=("volume", "sum"),
    )
    g = g.reset_index(level="_date", drop=True)
    g.index.name = "timestamp"
    return g


# Apply RTH and resample 1-min datasets
df_es_rth  = rth(df_es)
df_mes_rth = rth(df_mes_raw)
df_mnq_rth = rth(df_mnq_raw)

mes_bars = resample_1m_to_bar(df_mes_rth, "5min")
mnq_bars = resample_1m_to_bar(df_mnq_rth, "1min")

print(f"ES  5m RTH: {len(df_es_rth):,} bars, "
      f"{df_es_rth.index[0].date()} – {df_es_rth.index[-1].date()}")
print(f"MES 5m RTH: {len(mes_bars):,} bars, "
      f"{mes_bars.index[0].date()} – {mes_bars.index[-1].date()}")
print(f"MNQ 1m RTH: {len(mnq_bars):,} bars, "
      f"{mnq_bars.index[0].date()} – {mnq_bars.index[-1].date()}")


# ---------------------------------------------------------------------------
# Session stats
# ---------------------------------------------------------------------------

def build_session_stats(bars_rth: pd.DataFrame) -> dict:
    """
    Per-session dict:
      atr_20d   : 20-period EMA of true range (using daily OHLC from session bars)
      prev_close: previous session's last close
      open_price: today's first bar open
      gap_pts   : open_price - prev_close
      gap_atr   : abs(gap_pts) / atr_20d
    """
    sessions = sorted(bars_rth.groupby(bars_rth.index.date), key=lambda x: x[0])
    rows = []
    for date, sess in sessions:
        if len(sess) < 2:
            continue
        rows.append({
            "date":  date,
            "open":  float(sess["open"].iloc[0]),
            "high":  float(sess["high"].max()),
            "low":   float(sess["low"].min()),
            "close": float(sess["close"].iloc[-1]),
        })

    result = {}
    prev_close = None
    atr_val    = None

    for i, row in enumerate(rows):
        hi, lo, cl = row["high"], row["low"], row["close"]
        tr = (hi - lo) if prev_close is None else max(
            hi - lo, abs(hi - prev_close), abs(lo - prev_close)
        )
        atr_val = tr if atr_val is None else (atr_val * 19 + tr) / 20

        gap_pts    = (row["open"] - rows[i - 1]["close"]) if i > 0 else 0.0
        gap_atr    = abs(gap_pts) / atr_val if atr_val and atr_val > 0 else 0.0
        result[row["date"]] = {
            "atr_20d":    float(atr_val),
            "prev_close": float(rows[i - 1]["close"]) if i > 0 else None,
            "open_price": float(row["open"]),
            "gap_pts":    float(gap_pts),
            "gap_atr":    float(gap_atr),
        }
        prev_close = cl

    return result


# ---------------------------------------------------------------------------
# Session VWAP
# ---------------------------------------------------------------------------

def compute_session_vwap(bars: pd.DataFrame) -> pd.Series:
    """Cumulative (typical_price x volume) / cumulative_volume, reset each session."""
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
# Event extraction
# ---------------------------------------------------------------------------

def extract_events(
    bars_rth: pd.DataFrame,
    session_vwap: pd.Series,
    stats: dict,
    min_gap_atr: float = 0.20,
    time_gate_end_hour: int = 12,
    max_subsequent: int = 30,
) -> list:
    """
    For each session with a qualifying gap, scan for the first VWAP reclaim bar
    and record the event with up to max_subsequent subsequent bars for simulation.
    """
    events = []

    for date, grp in bars_rth.groupby(bars_rth.index.date):
        stat = stats.get(date)
        if stat is None or stat["prev_close"] is None:
            continue
        if stat["atr_20d"] <= 0:
            continue

        gap_atr = stat["gap_atr"]
        # Skip if gap too small or extreme (earnings / circuit breakers)
        if gap_atr < min_gap_atr or gap_atr > 3.0:
            continue

        gap_pts    = stat["gap_pts"]
        prev_close = stat["prev_close"]
        atr        = stat["atr_20d"]

        gap_down = gap_pts < 0  # LONG candidate
        gap_up   = gap_pts > 0  # SHORT candidate

        bars_arr   = grp
        vwap_arr   = session_vwap.loc[grp.index]

        # Start from second bar (index 1) — need 1-bar lookback within session
        for i in range(1, len(bars_arr)):
            bar      = bars_arr.iloc[i]
            bar_prev = bars_arr.iloc[i - 1]
            ts       = bars_arr.index[i]

            # Time gate: must be >= 09:35 and < time_gate_end_hour
            if ts.hour < 9 or (ts.hour == 9 and ts.minute < 35):
                continue
            if ts.hour >= time_gate_end_hour:
                break

            vwap_now  = vwap_arr.iloc[i]
            vwap_prev = vwap_arr.iloc[i - 1]

            if np.isnan(vwap_now) or np.isnan(vwap_prev):
                continue

            reclaim_long  = False
            reclaim_short = False

            if gap_down:
                # Previous bar close was below VWAP, this bar close crosses above
                if bar_prev["close"] < vwap_prev and bar["close"] > vwap_now:
                    reclaim_long = True

            if gap_up:
                # Previous bar close was above VWAP, this bar close crosses below
                if bar_prev["close"] > vwap_prev and bar["close"] < vwap_now:
                    reclaim_short = True

            if not (reclaim_long or reclaim_short):
                continue

            direction  = +1 if reclaim_long else -1
            entry_px   = float(bar["close"])
            target_px  = float(prev_close)

            # Collect up to max_subsequent bars after entry
            subsequent_bars = bars_arr.iloc[i + 1: i + 1 + max_subsequent]
            subsequent = [
                {
                    "h": float(r["high"]),
                    "l": float(r["low"]),
                    "c": float(r["close"]),
                }
                for _, r in subsequent_bars.iterrows()
            ]

            events.append({
                "date":       str(date),
                "entry_ts":   str(ts),
                "direction":  direction,
                "gap_pts":    float(gap_pts),
                "gap_atr":    float(gap_atr),
                "entry_px":   entry_px,
                "target_px":  target_px,
                "atr":        float(atr),
                "subsequent": subsequent,
            })
            break  # max one trade per session

    return events


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    events: list,
    stop_atr_mult: float = 0.75,
    time_stop_bars: int = 24,
    direction_filter: str = "both",
    point_value: float = ES_PV,
) -> dict:
    """
    Simulate events with stop/target/time-stop exits.
    Returns dict with trades list and summary stats.
    """
    trades = []

    for ev in events:
        d = ev["direction"]
        if direction_filter == "long"  and d != +1:
            continue
        if direction_filter == "short" and d != -1:
            continue

        entry_px  = ev["entry_px"]
        target_px = ev["target_px"]
        atr       = ev["atr"]
        subs      = ev["subsequent"]

        stop = (entry_px - stop_atr_mult * atr) if d == +1 else (entry_px + stop_atr_mult * atr)

        # Sanity: stop must be on the correct side
        if d == +1 and stop >= entry_px:
            continue
        if d == -1 and stop <= entry_px:
            continue

        # Target must offer positive R (if target is on wrong side of entry, skip)
        if d == +1 and target_px <= entry_px:
            continue
        if d == -1 and target_px >= entry_px:
            continue

        exit_px     = None
        exit_reason = "time"
        window      = subs[:time_stop_bars]

        for bar in window:
            h, l, c = bar["h"], bar["l"], bar["c"]
            if d == +1:
                if h >= target_px:
                    exit_px     = target_px
                    exit_reason = "tp"
                    break
                if l <= stop:
                    exit_px     = stop
                    exit_reason = "sl"
                    break
            else:
                if l <= target_px:
                    exit_px     = target_px
                    exit_reason = "tp"
                    break
                if h >= stop:
                    exit_px     = stop
                    exit_reason = "sl"
                    break

        if exit_px is None:
            # Time stop: exit at last bar's close
            if window:
                exit_px     = window[-1]["c"]
                exit_reason = "time"
            else:
                continue  # no bars to simulate

        pnl = (d * (exit_px - entry_px)) * point_value - COST_RT

        trades.append({
            "date":        ev["date"],
            "entry_ts":    ev["entry_ts"],
            "direction":   d,
            "gap_atr":     ev["gap_atr"],
            "entry_px":    entry_px,
            "exit_px":     exit_px,
            "pnl":         pnl,
            "exit_reason": exit_reason,
        })

    return {"trades": trades}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _stats(trades: list) -> dict:
    if not trades:
        return {
            "N": 0, "WR": 0.0, "E": 0.0, "PnL": 0.0,
            "Sharpe": 0.0, "MaxDD": 0.0, "avg_tpd": 0.0,
            "exit_dist": {}, "stress": {},
        }

    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls > 0
    N    = len(pnls)
    WR   = float(wins.sum()) / N
    E    = float(pnls.mean())
    PnL  = float(pnls.sum())

    by_day: dict[str, float] = {}
    for t in trades:
        day = t["date"]
        by_day[day] = by_day.get(day, 0.0) + t["pnl"]

    daily_pnl = np.array(list(by_day.values()))
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        ann_factor = np.sqrt(252)
        Sharpe = float((daily_pnl.mean() / daily_pnl.std()) * ann_factor)
    else:
        Sharpe = 0.0

    equity = np.cumsum(pnls)
    roll_max = np.maximum.accumulate(equity)
    dd       = equity - roll_max
    MaxDD    = float(dd.min())

    # Average trades per trading day (trading days = all days that appear in the full
    # session stats, approximated here by unique trade dates)
    trade_dates = set(t["date"] for t in trades)
    avg_tpd = N / max(len(trade_dates), 1)

    exit_dist: dict[str, int] = {}
    for t in trades:
        exit_dist[t["exit_reason"]] = exit_dist.get(t["exit_reason"], 0) + 1

    stress: dict[str, float] = {}
    for period, years in [
        ("2018-2019", {"2018", "2019"}),
        ("2020",      {"2020"}),
        ("2022",      {"2022"}),
    ]:
        period_pnl = sum(t["pnl"] for t in trades if t["date"][:4] in years)
        stress[period] = period_pnl

    return {
        "N":        N,
        "WR":       WR,
        "E":        E,
        "PnL":      PnL,
        "Sharpe":   Sharpe,
        "MaxDD":    MaxDD,
        "avg_tpd":  avg_tpd,
        "exit_dist": exit_dist,
        "stress":   stress,
    }


def print_s(label: str, s: dict) -> None:
    N      = s["N"]
    WR     = s["WR"]
    E      = s["E"]
    Sharpe = s["Sharpe"]
    MaxDD  = s["MaxDD"]
    tpd    = s["avg_tpd"]
    print(
        f"  {label:48s} | N={N:5d} | WR={WR:.1%} | "
        f"E=${E:+6.0f} | Sharpe={Sharpe:+6.2f} | "
        f"MaxDD=${MaxDD:,.0f} | {tpd:.1f}t/d"
    )


# ===========================================================================
# SECTION 1: ES 5-min baseline (2010–2025)
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 1: ES 5-min baseline 2010–2025")
print("=" * 70)

es_stats    = build_session_stats(df_es_rth)
es_vwap     = compute_session_vwap(df_es_rth)
es_events   = extract_events(
    df_es_rth, es_vwap, es_stats,
    min_gap_atr=MIN_GAP_ATR,
    time_gate_end_hour=TIME_GATE_END,
)

n_events    = len(es_events)
n_long      = sum(1 for e in es_events if e["direction"] == +1)
n_short     = sum(1 for e in es_events if e["direction"] == -1)
n_years     = len(set(e["date"][:4] for e in es_events))
avg_per_yr  = n_events / max(n_years, 1)

print(f"\nTotal events: {n_events}  |  gap-down (LONG): {n_long}  |  "
      f"gap-up (SHORT): {n_short}  |  avg/yr: {avg_per_yr:.1f}")
print()

for label, direction in [
    ("All (both dirs)", "both"),
    ("Long only (gap-down reclaim)", "long"),
    ("Short only (gap-up reclaim)", "short"),
]:
    sim = simulate(
        es_events,
        stop_atr_mult=STOP_ATR_MULT,
        time_stop_bars=TIME_STOP_BARS,
        direction_filter=direction,
        point_value=ES_PV,
    )
    s = _stats(sim["trades"])
    print_s(label, s)

# Per-year breakdown
print()
print(f"  {'Year':<48s} | {'N':>5s} | WR      | E        | PnL")
print("  " + "-" * 68)
for year in range(2010, 2026):
    yr_events = [e for e in es_events if e["date"].startswith(str(year))]
    if not yr_events:
        print(f"  {str(year):<48s} | {'—':>5s}")
        continue
    sim = simulate(
        yr_events,
        stop_atr_mult=STOP_ATR_MULT,
        time_stop_bars=TIME_STOP_BARS,
        direction_filter="both",
        point_value=ES_PV,
    )
    s = _stats(sim["trades"])
    print(
        f"  {str(year):<48s} | N={s['N']:4d} | WR={s['WR']:.1%} | "
        f"E=${s['E']:+6.0f} | PnL=${s['PnL']:+8,.0f} | Sharpe={s['Sharpe']:+5.2f}"
    )

# ES baseline sim for acceptance criteria
_es_sim_both  = simulate(es_events, STOP_ATR_MULT, TIME_STOP_BARS, "both",  ES_PV)
_es_sim_long  = simulate(es_events, STOP_ATR_MULT, TIME_STOP_BARS, "long",  ES_PV)
_es_sim_short = simulate(es_events, STOP_ATR_MULT, TIME_STOP_BARS, "short", ES_PV)

es_stats_both  = _stats(_es_sim_both["trades"])
es_stats_long  = _stats(_es_sim_long["trades"])
es_stats_short = _stats(_es_sim_short["trades"])

# ===========================================================================
# SECTION 2: Parameter sweep on ES 5-min
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 2: Parameter sweep — ES 5-min")
print("=" * 70)

_gap_grid   = [0.15, 0.20, 0.30, 0.40]
_stop_grid  = [0.5, 0.75, 1.0]
_gate_grid  = [11, 12, 13]
_dir_grid   = ["long", "short", "both"]

# Pre-extract event pools for each (min_gap_atr, time_gate_end) combo
_event_cache: dict[tuple, list] = {}
for _gap, _gate in product(_gap_grid, _gate_grid):
    key = (_gap, _gate)
    if key not in _event_cache:
        _event_cache[key] = extract_events(
            df_es_rth, es_vwap, es_stats,
            min_gap_atr=_gap,
            time_gate_end_hour=_gate,
        )

sweep_results = []
for _gap, _stop, _gate, _dir in product(_gap_grid, _stop_grid, _gate_grid, _dir_grid):
    evs = _event_cache[(_gap, _gate)]
    sim = simulate(evs, _stop, TIME_STOP_BARS, _dir, ES_PV)
    s   = _stats(sim["trades"])
    if s["N"] < 30:
        continue
    sweep_results.append({
        "min_gap_atr":   _gap,
        "stop_atr_mult": _stop,
        "time_gate_end": _gate,
        "direction":     _dir,
        **s,
    })

sweep_results.sort(key=lambda x: x["Sharpe"], reverse=True)

print(f"\nTop 25 configs (N >= 30, sorted by Sharpe):\n")
header = (
    f"  {'gap':>6s}  {'stop':>6s}  {'gate':>4s}  {'dir':>5s}  | "
    f"{'N':>5s} | WR      | E        | Sharpe   | MaxDD"
)
print(header)
print("  " + "-" * 72)
for r in sweep_results[:25]:
    print(
        f"  gap>={r['min_gap_atr']:.2f}  stop={r['stop_atr_mult']:.2f}x  "
        f"gate<={r['time_gate_end']:02d}h  dir={r['direction']:5s}  | "
        f"N={r['N']:5d} | WR={r['WR']:.1%} | "
        f"E=${r['E']:+6.0f} | Sharpe={r['Sharpe']:+6.2f} | MaxDD=${r['MaxDD']:,.0f}"
    )

# Determine best config to carry forward
best_cfg = sweep_results[0] if sweep_results else None
if best_cfg and best_cfg["Sharpe"] > 2.0:
    USE_GAP  = best_cfg["min_gap_atr"]
    USE_STOP = best_cfg["stop_atr_mult"]
    USE_GATE = best_cfg["time_gate_end"]
    USE_DIR  = best_cfg["direction"]
    print(f"\nBest config (Sharpe={best_cfg['Sharpe']:+.2f}) carried forward to transfer tests.")
else:
    USE_GAP  = MIN_GAP_ATR
    USE_STOP = STOP_ATR_MULT
    USE_GATE = TIME_GATE_END
    USE_DIR  = "both"
    print(f"\nNo config exceeded Sharpe > 2.0 — using default config for transfer tests.")

# ===========================================================================
# SECTION 3: MES 1-min transfer test
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 3: MES 1-min transfer test")
print("=" * 70)

TIME_STOP_BARS_1M = 48  # 48 minutes (1-min bars)

mes_session_stats = build_session_stats(mes_bars)
mes_vwap          = compute_session_vwap(mes_bars)
mes_events        = extract_events(
    mes_bars, mes_vwap, mes_session_stats,
    min_gap_atr=USE_GAP,
    time_gate_end_hour=USE_GATE,
)
print(f"\nMES events: {len(mes_events)}  |  "
      f"long: {sum(1 for e in mes_events if e['direction']==+1)}  |  "
      f"short: {sum(1 for e in mes_events if e['direction']==-1)}")
print()

for label, direction in [
    ("All (both dirs)", "both"),
    ("Long only (gap-down reclaim)", "long"),
    ("Short only (gap-up reclaim)", "short"),
]:
    sim = simulate(
        mes_events,
        stop_atr_mult=USE_STOP,
        time_stop_bars=TIME_STOP_BARS_1M,
        direction_filter=direction,
        point_value=MES_PV,
    )
    s = _stats(sim["trades"])
    print_s(label, s)

# ===========================================================================
# SECTION 4: MNQ 2026 YTD OOS
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 4: MNQ 2026 YTD OOS")
print("=" * 70)

mnq_session_stats = build_session_stats(mnq_bars)
mnq_vwap          = compute_session_vwap(mnq_bars)
mnq_events        = extract_events(
    mnq_bars, mnq_vwap, mnq_session_stats,
    min_gap_atr=USE_GAP,
    time_gate_end_hour=USE_GATE,
)
print(f"\nMNQ events: {len(mnq_events)}  |  "
      f"long: {sum(1 for e in mnq_events if e['direction']==+1)}  |  "
      f"short: {sum(1 for e in mnq_events if e['direction']==-1)}")
print()

_mnq_results = {}
for label, direction in [
    ("All (both dirs)", "both"),
    ("Long only (gap-down reclaim)", "long"),
    ("Short only (gap-up reclaim)", "short"),
]:
    sim = simulate(
        mnq_events,
        stop_atr_mult=USE_STOP,
        time_stop_bars=TIME_STOP_BARS_1M,
        direction_filter=direction,
        point_value=MNQ_PV,
    )
    s = _stats(sim["trades"])
    print_s(label, s)
    _mnq_results[direction] = s

mnq_oos_pnl = _mnq_results["both"]["PnL"]

# ===========================================================================
# SECTION 5: Acceptance criteria
# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 5: Acceptance criteria")
print("=" * 70)
print()

criteria = [
    ("ES WR > 48%",         es_stats_both["WR"]    > 0.48,  f"WR={es_stats_both['WR']:.1%}"),
    ("ES Sharpe > 2.0",     es_stats_both["Sharpe"] > 2.0,  f"Sharpe={es_stats_both['Sharpe']:+.2f}"),
    ("ES MaxDD > -$15,000", es_stats_both["MaxDD"]  > -15000, f"${es_stats_both['MaxDD']:,.0f}"),
    ("ES 2020 positive",    es_stats_both["stress"].get("2020", 0) > 0,
                            f"${es_stats_both['stress'].get('2020', 0):,.0f}"),
    ("ES 2022 positive",    es_stats_both["stress"].get("2022", 0) > 0,
                            f"${es_stats_both['stress'].get('2022', 0):,.0f}"),
    ("Long PnL > 0",        es_stats_long["PnL"]  > 0, f"${es_stats_long['PnL']:,.0f}"),
    ("Short PnL > 0",       es_stats_short["PnL"] > 0, f"${es_stats_short['PnL']:,.0f}"),
    ("MNQ OOS positive",    mnq_oos_pnl           > 0, f"${mnq_oos_pnl:,.0f}"),
]

passed = 0
for criterion, ok, value in criteria:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag:4s}]  {criterion:<30s}  ({value})")
    if ok:
        passed += 1

total = len(criteria)
verdict = "GO" if passed >= 6 else ("MARGINAL" if passed >= 4 else "KILL")
print(f"\n  Passed {passed}/{total} → [{verdict}]")
print(f"  (>=6 = GO, 4-5 = MARGINAL, <4 = KILL)")

# ===========================================================================
# Save results
# ===========================================================================
OUT_PATH = ROOT / "rule_based_v1/diagnostics/vwap_reclaim_results.json"

output = {
    "config": {
        "min_gap_atr":    USE_GAP,
        "stop_atr_mult":  USE_STOP,
        "time_gate_end":  USE_GATE,
        "direction":      USE_DIR,
        "time_stop_bars": TIME_STOP_BARS,
    },
    "es_baseline": {
        "both":  es_stats_both,
        "long":  es_stats_long,
        "short": es_stats_short,
    },
    "sweep_top25": sweep_results[:25],
    "mnq_oos": _mnq_results,
    "acceptance": {
        "passed":  passed,
        "total":   total,
        "verdict": verdict,
        "detail":  [
            {"criterion": c, "pass": ok, "value": v}
            for c, ok, v in criteria
        ],
    },
}

with open(str(OUT_PATH), "w") as _f:
    json.dump(output, _f, indent=2, default=str)

print(f"\nResults saved to: {OUT_PATH}")
