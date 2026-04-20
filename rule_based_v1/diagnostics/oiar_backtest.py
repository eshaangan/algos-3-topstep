"""
OIAR — Overnight Inventory Acceptance / Rejection
==================================================
Standalone strategy using a different information source than ORB.

Edge hypothesis: a large overnight gap contains an inventory dislocation.
RTH first 30 minutes (first 6 bars) decide whether that displacement is
ACCEPTED (held) or REJECTED (unwound). Both outcomes are tradeable.

Event: day with |gap / atr_20d| >= GAP_ATR_MIN (significant overnight move)

Mode A — Rejection unwind:
  - Bullish gap + RTH fails to hold above on_midpoint in first 6 bars → SHORT
  - Target: prior_close (full gap fill)
  - Stop: above RTH session high so far

Mode B — Acceptance continuation:
  - Bullish gap + RTH holds above on_midpoint after 6 bars → LONG
  - Target: open + gap_size (measured move extension)
  - Stop: below session low so far

Mirror image for bearish gaps.

Regime routing (optional gate):
  - negative: only rejection SHORTs and acceptance SHORTs
  - chop: only rejection fades, both sides
  - trend: acceptance continuation both sides
  - (no regime gate = all events tested first)

ETH proxies (RTH-only data):
  - on_midpoint  = (prev_close + open) / 2         ≈ overnight VWAP
  - on_range_proxy = gap_size (minimum overnight range)
  - acceptance = first 6 bars hold above on_midpoint (bullish)
  - rejection  = any bar in first 6 closes below on_midpoint (bullish)

Long-term test uses ES 2010-2025 (RTH 5-min, 3,995 sessions).
Short-term tests use MES 1-min resampled to 5-min.
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
# Parameters
# ---------------------------------------------------------------------------
GAP_ATR_MIN   = 0.50   # |gap / atr_20d| threshold — significant overnight move
OR_END_BAR    = 6      # first 6 bars (30 min) = acceptance/rejection window
CUTOFF_H      = 13     # no new entries after 1pm

ES_PV         = 50.0
MES_PV        = 5.0
MNQ_PV        = 2.0
COST_RT       = 6.25

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

print("Loading ES 15-year data (RTH)…")
with pd.HDFStore(str(ROOT / "data/processed/es_bars_2010_2025.h5"), "r") as s:
    df_es = s["/bars_5min"].copy()
df_es.index = pd.to_datetime(df_es["timestamp"], unit="s", utc=True).dt.tz_convert("US/Eastern")
df_es = df_es.drop(columns=["timestamp"]).sort_index()
mask_rth = (
    ((df_es.index.hour == 9) & (df_es.index.minute >= 30))
    | ((df_es.index.hour > 9) & (df_es.index.hour < 16))
)
df_es = df_es[mask_rth].copy()
print(f"ES: {len(df_es):,} bars, {len(set(df_es.index.date)):,} sessions")

print("Loading MES 1-min data (full ETH)…")
with pd.HDFStore(str(ROOT / "data/processed/mes_1m_bars_cache.h5"), "r") as s:
    df_mes_raw = s["/bars_1m"].set_index("timestamp")
df_mes_raw.index = pd.to_datetime(df_mes_raw.index, utc=True).tz_convert("US/Eastern")
df_mes_raw = df_mes_raw.sort_index()

print("Loading OOS 1-min data…")
with pd.HDFStore(str(ROOT / "data/processed/jan_feb_2026_oos_test_1m.h5"), "r") as s:
    df_oos_raw = s["/bars_1min"].copy()
df_oos_raw.index = pd.to_datetime(df_oos_raw.index, utc=True).tz_convert("US/Eastern")
df_oos_raw = df_oos_raw.sort_index()

print("Loading MNQ 1-min data (full ETH)…")
with pd.HDFStore(str(ROOT / "data/processed/mnq_2026ytd_1min_eth.h5"), "r") as s:
    df_mnq_raw = s["/bars_1min_eth"].copy()
if df_mnq_raw.index.tz is None:
    df_mnq_raw.index = df_mnq_raw.index.tz_localize("UTC").tz_convert("US/Eastern")
elif str(df_mnq_raw.index.tz) != "US/Eastern":
    df_mnq_raw.index = df_mnq_raw.index.tz_convert("US/Eastern")
df_mnq_raw = df_mnq_raw.sort_index()


def rth(df):
    mask = (
        ((df.index.hour == 9) & (df.index.minute >= 30))
        | ((df.index.hour > 9) & (df.index.hour < 16))
    )
    return df[mask].copy()


def resample_5m(df1m):
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


mes_5m = resample_5m(rth(df_mes_raw))
oos_5m = resample_5m(rth(df_oos_raw))
mnq_5m = resample_5m(rth(df_mnq_raw))

print(f"MES 5m: {len(mes_5m):,}  ({mes_5m.index[0].date()} – {mes_5m.index[-1].date()})")
print(f"OOS 5m: {len(oos_5m):,}  ({oos_5m.index[0].date()} – {oos_5m.index[-1].date()})")
print(f"MNQ 5m: {len(mnq_5m):,}  ({mnq_5m.index[0].date()} – {mnq_5m.index[-1].date()})")


# ---------------------------------------------------------------------------
# Session stats + ATR
# ---------------------------------------------------------------------------

def build_stats(bars_5m: pd.DataFrame) -> dict:
    sessions = sorted(bars_5m.groupby(bars_5m.index.date), key=lambda x: x[0])
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
            "vol30": float(sess["volume"].iloc[:6].sum()),
        })

    result = {}
    prev_close = None
    atr_val    = None

    for i, row in enumerate(rows):
        hi, lo, cl = row["high"], row["low"], row["close"]
        tr      = (hi - lo) if prev_close is None else max(
            hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        atr_val = tr if atr_val is None else (atr_val * 13 + tr) / 14

        gap_pts = (row["open"] - rows[i-1]["close"]) if i > 0 else 0.0
        result[row["date"]] = {
            "atr_20d":    atr_val,
            "gap_pts":    gap_pts,          # open - prev_close in points
            "prev_close": rows[i-1]["close"] if i > 0 else None,
            "prev_high":  rows[i-1]["high"]  if i > 0 else None,
            "prev_low":   rows[i-1]["low"]   if i > 0 else None,
        }
        prev_close = cl

    return result


# Optional: compute richer overnight metrics from full ETH 1-min data
def compute_eth_metrics(df_1m_full: pd.DataFrame, stats: dict) -> dict:
    """
    For each RTH session date, compute overnight metrics from ETH bars:
      on_high, on_low, on_range, on_vwap, on_volume, on_efficiency
    Overnight = 16:01 ET (prior day) to 09:29 ET (current day)
    """
    eth_metrics = {}
    session_dates = sorted(stats.keys())

    for date in session_dates:
        stat = stats[date]
        if stat["prev_close"] is None:
            continue

        # Overnight window: prior day 16:00 → current day 09:30 (exclusive of RTH)
        ts_start = pd.Timestamp(date - pd.Timedelta(days=3), tz="US/Eastern").replace(
            hour=16, minute=0)  # rough start — will be filtered
        ts_end   = pd.Timestamp(date, tz="US/Eastern").replace(hour=9, minute=29)

        # Filter to approximate overnight (last 24h before RTH open, excluding RTH)
        on_bars = df_1m_full[
            (df_1m_full.index < ts_end)
            & (df_1m_full.index >= ts_end - pd.Timedelta(hours=20))
            & ~(
                ((df_1m_full.index.hour == 9) & (df_1m_full.index.minute >= 30))
                | ((df_1m_full.index.hour > 9) & (df_1m_full.index.hour < 16))
            )
        ]

        if len(on_bars) < 10:
            continue

        on_high = float(on_bars["high"].max())
        on_low  = float(on_bars["low"].min())
        on_vol  = float(on_bars["volume"].sum())
        tp      = (on_bars["high"] + on_bars["low"] + on_bars["close"]) / 3
        on_vwap = float((tp * on_bars["volume"]).sum() / (on_vol + 1e-8))

        sum_tr  = float(on_bars.apply(lambda r: r["high"] - r["low"], axis=1).sum()) + 1e-8
        on_open_price  = float(on_bars["open"].iloc[0])
        on_close_price = float(on_bars["close"].iloc[-1])
        on_efficiency  = abs(on_close_price - on_open_price) / sum_tr

        eth_metrics[date] = {
            "on_high":      on_high,
            "on_low":       on_low,
            "on_range":     on_high - on_low,
            "on_vwap":      on_vwap,
            "on_volume":    on_vol,
            "on_efficiency": on_efficiency,
            "on_close_loc": (on_close_price - on_low) / (on_high - on_low + 1e-6),
        }

    return eth_metrics


# ---------------------------------------------------------------------------
# OIAR event extractor
# ---------------------------------------------------------------------------

def extract_oiar_events(bars_5m: pd.DataFrame, stats: dict,
                        eth_metrics: dict | None = None) -> list:
    """
    For each qualifying session (large gap), extract the OIAR event:
      - mode: "rejection" or "acceptance"
      - direction: +1 (long) or -1 (short)
      - entry candidates: subsequent bars with PnL simulation data

    Qualifying session: |gap_pts / atr_20d| >= GAP_ATR_MIN
    """
    events = []
    session_dates = sorted(set(bars_5m.index.date))

    for date in session_dates:
        stat = stats.get(date)
        if stat is None or stat["atr_20d"] is None or stat["prev_close"] is None:
            continue

        atr_20d    = stat["atr_20d"]
        bar_atr    = atr_20d / 13.0
        gap_pts    = stat["gap_pts"]        # open - prev_close
        prev_close = stat["prev_close"]

        # Require significant overnight gap
        if abs(gap_pts) / (atr_20d + 1e-8) < GAP_ATR_MIN:
            continue

        gap_bullish = gap_pts > 0
        on_midpoint = prev_close + gap_pts / 2.0   # approx overnight VWAP

        # Enrich with ETH metrics if available
        eth = eth_metrics.get(date) if eth_metrics else None
        if eth:
            on_high     = eth["on_high"]
            on_low      = eth["on_low"]
            on_vwap     = eth["on_vwap"]
            on_midpoint = on_vwap  # use true overnight VWAP as midpoint
        else:
            on_high = max(prev_close, prev_close + gap_pts)
            on_low  = min(prev_close, prev_close + gap_pts)

        sess      = bars_5m[bars_5m.index.date == date].copy()
        sess_bars = list(sess.iterrows())
        n_bars    = len(sess_bars)

        if n_bars < OR_END_BAR + 3:
            continue

        rth_open = float(sess["open"].iloc[0])

        # ------------------------------------------------------------------
        # Evaluate first 6 bars (30 min) for acceptance or rejection
        # ------------------------------------------------------------------
        first6 = sess.iloc[:OR_END_BAR]
        min_close_first6 = float(first6["close"].min())
        max_close_first6 = float(first6["close"].max())

        # Directional efficiency of first 30 min
        sum_tr6 = float(first6.apply(lambda r: r["high"] - r["low"], axis=1).sum()) + 1e-8
        close6  = float(first6["close"].iloc[-1])
        de6     = abs(close6 - rth_open) / sum_tr6

        # Acceptance/rejection classification
        if gap_bullish:
            # Rejection: any first-6 close falls below the midpoint
            rejection = min_close_first6 < on_midpoint
            # Strong rejection: any first-6 close falls BELOW prev_close (gap filled)
            strong_rejection = min_close_first6 < prev_close
        else:
            # Rejection: any first-6 close rises above the midpoint (bear gap fading up)
            rejection = max_close_first6 > on_midpoint
            strong_rejection = max_close_first6 > prev_close

        # Accept = held past the midpoint after 30 min without rejection
        acceptance = not rejection and (
            (gap_bullish and close6 > on_midpoint)
            or (not gap_bullish and close6 < on_midpoint)
        )

        # ------------------------------------------------------------------
        # Find entry bar (first qualifying entry after 30-min window)
        # ------------------------------------------------------------------
        entry_found = False
        entry_info  = {}

        for i_bar in range(OR_END_BAR, n_bars):
            ts, bar = sess_bars[i_bar]
            if ts.hour >= CUTOFF_H:
                break

            h = float(bar["high"])
            l = float(bar["low"])
            c = float(bar["close"])
            rng = h - l + 1e-6
            cl  = (c - l) / rng

            session_hi = float(sess.iloc[:i_bar]["high"].max())
            session_lo = float(sess.iloc[:i_bar]["low"].min())

            # Compute subsequent bars for exit simulation
            subsequent = []
            for j in range(i_bar + 1, min(i_bar + 30, n_bars)):
                _, fb = sess_bars[j]
                subsequent.append({
                    "h": float(fb["high"]), "l": float(fb["low"]), "c": float(fb["close"])
                })

            if rejection and not entry_found:
                if gap_bullish:
                    # SHORT: bar close is below midpoint (confirmation of rejection)
                    if c < on_midpoint and cl < 0.45:
                        entry_info = {
                            "mode":          "rejection",
                            "direction":     -1,
                            "entry_px":      c,
                            "stop_ref":      session_hi,   # stop above session high
                            "tp_ref":        prev_close,   # target: full gap fill
                            "on_midpoint":   on_midpoint,
                            "bar_atr":       bar_atr,
                            "subsequent":    subsequent,
                            "de6":           de6,
                            "strong":        strong_rejection,
                        }
                        entry_found = True
                else:
                    # LONG: bar close is above midpoint (bear gap rejection)
                    if c > on_midpoint and cl > 0.55:
                        entry_info = {
                            "mode":          "rejection",
                            "direction":     1,
                            "entry_px":      c,
                            "stop_ref":      session_lo,
                            "tp_ref":        prev_close,
                            "on_midpoint":   on_midpoint,
                            "bar_atr":       bar_atr,
                            "subsequent":    subsequent,
                            "de6":           de6,
                            "strong":        strong_rejection,
                        }
                        entry_found = True

            elif acceptance and not entry_found:
                if gap_bullish:
                    # LONG continuation: first pullback that holds above on_midpoint
                    if c > on_midpoint and cl < 0.50:   # pullback bar (close < midbar)
                        target = rth_open + abs(gap_pts)  # measured move
                        entry_info = {
                            "mode":          "acceptance",
                            "direction":     1,
                            "entry_px":      c,
                            "stop_ref":      session_lo,
                            "tp_ref":        target,
                            "on_midpoint":   on_midpoint,
                            "bar_atr":       bar_atr,
                            "subsequent":    subsequent,
                            "de6":           de6,
                            "strong":        False,
                        }
                        entry_found = True
                else:
                    # SHORT continuation: first bounce that fails below on_midpoint
                    if c < on_midpoint and cl > 0.50:
                        target = rth_open - abs(gap_pts)
                        entry_info = {
                            "mode":          "acceptance",
                            "direction":     -1,
                            "entry_px":      c,
                            "stop_ref":      session_hi,
                            "tp_ref":        target,
                            "on_midpoint":   on_midpoint,
                            "bar_atr":       bar_atr,
                            "subsequent":    subsequent,
                            "de6":           de6,
                            "strong":        False,
                        }
                        entry_found = True

            if entry_found:
                break

        if not entry_found:
            continue

        events.append({
            "date":       str(date),
            "gap_pts":    gap_pts,
            "gap_norm":   gap_pts / (atr_20d + 1e-8),
            "gap_bull":   gap_bullish,
            **entry_info,
        })

    return events


# ---------------------------------------------------------------------------
# Exit simulator
# ---------------------------------------------------------------------------

def simulate(events: list, stop_atr_mult: float, time_stop_bars: int,
             mode_filter: str = "both",   # "rejection", "acceptance", "both"
             direction_filter: str = "both",  # "short", "long", "both"
             strong_only: bool = False,
             min_gap_norm: float = GAP_ATR_MIN,
             point_value: float = ES_PV) -> dict:
    trades = []

    for ev in events:
        if mode_filter != "both" and ev["mode"] != mode_filter:
            continue
        if direction_filter == "short" and ev["direction"] != -1:
            continue
        if direction_filter == "long" and ev["direction"] != 1:
            continue
        if strong_only and not ev.get("strong", False):
            continue
        if abs(ev["gap_norm"]) < min_gap_norm:
            continue

        entry = ev["entry_px"]
        dir_  = ev["direction"]
        ba    = ev["bar_atr"]
        ref   = ev["stop_ref"]
        tp    = ev["tp_ref"]

        if dir_ == -1:
            stop = ref + stop_atr_mult * ba   # above session high + buffer
            risk = stop - entry
            if risk <= 0:
                continue
        else:
            stop = ref - stop_atr_mult * ba
            risk = entry - stop
            if risk <= 0:
                continue

        pnl    = None
        reason = None
        for i_b, fb in enumerate(ev["subsequent"][:time_stop_bars]):
            if dir_ == -1:
                if fb["h"] >= stop:
                    pnl = (entry - stop) * point_value - COST_RT; reason = "stop"; break
                elif fb["l"] <= tp:
                    pnl = (entry - tp) * point_value - COST_RT; reason = "tp"; break
            else:
                if fb["l"] <= stop:
                    pnl = (stop - entry) * point_value - COST_RT; reason = "stop"; break
                elif fb["h"] >= tp:
                    pnl = (tp - entry) * point_value - COST_RT; reason = "tp"; break
            if i_b == min(time_stop_bars, len(ev["subsequent"])) - 1:
                pnl = (entry - fb["c"]) * point_value - COST_RT if dir_ == -1 \
                      else (fb["c"] - entry) * point_value - COST_RT
                reason = "time"

        if pnl is None and ev["subsequent"]:
            last = ev["subsequent"][-1]
            pnl  = (entry - last["c"]) * point_value - COST_RT if dir_ == -1 \
                   else (last["c"] - entry) * point_value - COST_RT
            reason = "time"

        if pnl is not None:
            trades.append({
                "date": ev["date"], "pnl": pnl, "exit": reason,
                "mode": ev["mode"], "dir": "short" if dir_ == -1 else "long",
            })

    return _stats(trades)


def _stats(trades: list) -> dict:
    if not trades:
        return {"N": 0, "WR": 0.0, "E": 0.0, "PnL": 0.0, "Sharpe": 0.0, "MaxDD": 0.0,
                "stress": {}}
    p = np.array([t["pnl"] for t in trades])
    WR = float((p > 0).mean())
    E  = float(p.mean())

    by_day = {}
    for t in trades:
        by_day.setdefault(t["date"], 0.0)
        by_day[t["date"]] += t["pnl"]
    daily  = np.array(list(by_day.values()))
    sharpe = float((daily.mean() / (daily.std() + 1e-8)) * np.sqrt(252)) if len(daily) > 1 else 0.0

    cum    = np.cumsum(p)
    max_dd = float((cum - np.maximum.accumulate(cum)).min())

    stress = {}
    for period, yr_range in [("2018-2019", ("2018", "2019")),
                              ("2020",      ("2020", "2020")),
                              ("2022",      ("2022", "2022"))]:
        sub = [t for t in trades if yr_range[0] <= t["date"][:4] <= yr_range[1]]
        stress[period] = round(float(np.array([t["pnl"] for t in sub]).sum()), 2) if sub else 0.0

    return {"N": len(p), "WR": round(WR, 4), "E": round(E, 2),
            "PnL": round(float(p.sum()), 2), "Sharpe": round(sharpe, 3),
            "MaxDD": round(max_dd, 2), "stress": stress}


def print_s(label, s, pv=None):
    if s.get("N", 0) == 0:
        print(f"  {label:45s} | NO TRADES")
        return
    print(f"  {label:45s} | N={s['N']:5d} | WR={s['WR']:.1%} | E=${s['E']:+6.0f}"
          f" | PnL=${s['PnL']:+8,.0f} | Sharpe={s['Sharpe']:+6.2f} | MaxDD=${s['MaxDD']:,.0f}")


# ---------------------------------------------------------------------------
# Build stats + extract events
# ---------------------------------------------------------------------------

print("\nBuilding session stats…")
es_stats   = build_stats(df_es)
mes_stats  = build_stats(mes_5m)
oos_stats  = build_stats(oos_5m)
mnq_stats  = build_stats(mnq_5m)

print("Computing ETH overnight metrics (MES, OOS, MNQ)…")
mes_eth = compute_eth_metrics(df_mes_raw, mes_stats)
oos_eth = compute_eth_metrics(df_oos_raw, oos_stats)
mnq_eth = compute_eth_metrics(df_mnq_raw, mnq_stats)
print(f"  ETH metrics computed: MES={len(mes_eth)} days, OOS={len(oos_eth)} days, MNQ={len(mnq_eth)} days")

print("Extracting OIAR events…")
es_events  = extract_oiar_events(df_es, es_stats, eth_metrics=None)  # RTH proxy
mes_events = extract_oiar_events(mes_5m, mes_stats, eth_metrics=mes_eth)
oos_events = extract_oiar_events(oos_5m, oos_stats, eth_metrics=oos_eth)
mnq_events = extract_oiar_events(mnq_5m, mnq_stats, eth_metrics=mnq_eth)

print(f"  ES 15yr events  : {len(es_events):,}")
print(f"  MES train events: {len(mes_events):,}")
print(f"  OOS events      : {len(oos_events):,}")
print(f"  MNQ events      : {len(mnq_events):,}")

# Mode split
es_rej  = [e for e in es_events if e["mode"] == "rejection"]
es_acc  = [e for e in es_events if e["mode"] == "acceptance"]
es_bull = [e for e in es_events if e["gap_bull"]]
es_bear = [e for e in es_events if not e["gap_bull"]]
print(f"  ES breakdown: rejection={len(es_rej)}, acceptance={len(es_acc)}, "
      f"bull-gap={len(es_bull)}, bear-gap={len(es_bear)}")


# ---------------------------------------------------------------------------
# SECTION 1: Raw signal quality (no exit grid yet)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 1: RAW SIGNAL — EVENT BASELINE CONFIGS (ES 15yr)")
print("=" * 80)

# Baseline: stop=0.5×bar_atr, time=12 bars
for mode in ["both", "rejection", "acceptance"]:
    for direction in ["both", "short", "long"]:
        s = simulate(es_events, 0.5, 12, mode, direction, point_value=ES_PV)
        if s["N"] >= 20:
            label = f"mode={mode:10s} dir={direction:5s}"
            print_s(label, s)

print()
# Strong rejection only
s = simulate(es_events, 0.5, 12, "rejection", "both", strong_only=True, point_value=ES_PV)
print_s("rejection STRONG only (gap fully filled first 30m)", s)


# ---------------------------------------------------------------------------
# SECTION 2: Grid sweep — ES 15yr
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 2: STOP × TIME GRID SWEEP (ES 15yr, rejection mode)")
print("=" * 80)

stop_mults  = [0.10, 0.25, 0.50, 0.75, 1.00]
time_stops  = [3, 6, 9, 12, 18, 24]
modes       = ["rejection", "acceptance", "both"]
directions  = ["short", "long", "both"]
strong_opts = [False, True]

results = []
for sm, ts, mode, dir_, strong in product(stop_mults, time_stops, modes, directions, strong_opts):
    r = simulate(es_events, sm, ts, mode, dir_, strong, point_value=ES_PV)
    if r["N"] < 30:
        continue
    r.update({"sm": sm, "ts": ts, "mode": mode, "dir": dir_, "strong": strong})
    results.append(r)

results_sorted = sorted(results, key=lambda x: x["Sharpe"], reverse=True)

print(f"\n{'Config':55s}  {'N':>5}  {'WR':>6}  {'E':>7}  {'Sharpe':>7}  Stress(18/20/22)")
print("-" * 110)
for r in results_sorted[:25]:
    cfg = (f"stop={r['sm']:.2f}× ts={r['ts']:2d}b {r['mode']:10s} {r['dir']:5s} "
           f"{'strong' if r['strong'] else '      '}")
    s18 = r["stress"].get("2018-2019", 0)
    s20 = r["stress"].get("2020", 0)
    s22 = r["stress"].get("2022", 0)
    print(f"  {cfg:55s}  {r['N']:5d}  {r['WR']:.1%}  ${r['E']:6.0f}  {r['Sharpe']:7.3f}  "
          f"${s18:+,.0f}/${s20:+,.0f}/${s22:+,.0f}")

stress_survivors = [r for r in results if
                    r["stress"].get("2018-2019", -1) > 0
                    and r["stress"].get("2020", -1) > 0
                    and r["stress"].get("2022", -1) > 0
                    and r["N"] >= 50
                    and r["Sharpe"] > 0]
print(f"\n  Configs surviving ALL 3 stress windows (Sharpe>0, N>=50): {len(stress_survivors)}")
if stress_survivors:
    best = max(stress_survivors, key=lambda x: x["Sharpe"])
    print(f"  Best: stop={best['sm']}× ts={best['ts']}b {best['mode']} {best['dir']} "
          f"{'strong' if best['strong'] else ''}")
    print(f"    N={best['N']}, WR={best['WR']:.1%}, E=${best['E']:.0f}, "
          f"Sharpe={best['Sharpe']:.3f}, PnL=${best['PnL']:,.0f}")


# ---------------------------------------------------------------------------
# SECTION 3: Year-by-year with best overall config
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 3: YEAR-BY-YEAR BREAKDOWN (best Sharpe config)")
print("=" * 80)

best_overall = results_sorted[0] if results_sorted else None
if best_overall:
    print(f"  Config: stop={best_overall['sm']}× ts={best_overall['ts']}b "
          f"{best_overall['mode']} {best_overall['dir']} "
          f"{'strong' if best_overall['strong'] else ''}")
    years = sorted(set(e["date"][:4] for e in es_events))
    for yr in years:
        yr_ev = [e for e in es_events if e["date"].startswith(yr)]
        r = simulate(yr_ev, best_overall["sm"], best_overall["ts"],
                     best_overall["mode"], best_overall["dir"],
                     best_overall["strong"], point_value=ES_PV)
        print_s(f"  {yr}", r)


# ---------------------------------------------------------------------------
# SECTION 4: MES training + OOS + MNQ transfer
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 4: MES TRAINING / OOS / MNQ TRANSFER")
print("=" * 80)

if best_overall:
    sm, ts_, mode_, dir_, strong_ = (best_overall["sm"], best_overall["ts"],
                                     best_overall["mode"], best_overall["dir"],
                                     best_overall["strong"])

    # ORB trades for correlation
    def run_orb_simple(bars_5m, stats_d, point_value):
        """Minimal ORB runner for correlation analysis."""
        trades = []
        session_dates = sorted(set(bars_5m.index.date))
        pv_flag = {}
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
            if stat is None or stat["atr_20d"] is None: continue
            if not pv_flag.get(d, False): continue
            sess = bars_5m[bars_5m.index.date == d].copy()
            if len(sess) <= 7: continue
            or_h = float(sess.iloc[:7]["high"].max())
            or_l = float(sess.iloc[:7]["low"].min())
            if (or_h - or_l) < 0.3 * stat["atr_20d"]: continue
            in_t = False; ep = sp = tp2 = 0.0; eb = 0
            for i, (ts, bar) in enumerate(sess.iterrows()):
                if i < 7: continue
                if ts.hour >= 12 and not in_t: break
                if in_t:
                    if float(bar["high"]) >= tp2:
                        pnl = (tp2 - ep) * point_value - COST_RT
                        trades.append({"date": str(d), "pnl": round(pnl, 2)}); in_t = False
                    elif float(bar["low"]) <= sp:
                        pnl = (sp - ep) * point_value - COST_RT
                        trades.append({"date": str(d), "pnl": round(pnl, 2)}); in_t = False
                    elif (i - eb) >= 24 or i == len(sess) - 1:
                        pnl = (float(bar["close"]) - ep) * point_value - COST_RT
                        trades.append({"date": str(d), "pnl": round(pnl, 2)}); in_t = False
                    continue
                if float(bar["close"]) > or_h:
                    ep = float(bar["close"]); sp = ep - 1.5 * stat["atr_20d"]
                    tp2 = ep + 2.0 * stat["atr_20d"]; in_t = True; eb = i
        return trades

    for label, evts, stats_d, pv in [
        ("MES training", mes_events, mes_stats, MES_PV),
        ("OOS MES Jan-Feb 2026", oos_events, oos_stats, MES_PV),
        ("MNQ Jan-Mar 2026", mnq_events, mnq_stats, MNQ_PV),
    ]:
        oiar_s = simulate(evts, sm, ts_, mode_, dir_, strong_, point_value=pv)
        print_s(f"OIAR {label}", oiar_s)
        if len(evts) > 0:
            rej = [e for e in evts if e["mode"] == "rejection"]
            acc = [e for e in evts if e["mode"] == "acceptance"]
            print_s(f"  rejection", simulate(rej, sm, ts_, "rejection", dir_, strong_, point_value=pv))
            print_s(f"  acceptance", simulate(acc, sm, ts_, "acceptance", dir_, strong_, point_value=pv))
        print()

    # Correlation with ORB on MES training
    print("ORB vs OIAR daily PnL correlation (MES training)…")
    orb_t = run_orb_simple(mes_5m, mes_stats, MES_PV)
    oiar_t = []
    for ev in mes_events:
        if mode_ != "both" and ev["mode"] != mode_: continue
        if dir_ == "short" and ev["direction"] != -1: continue
        if dir_ == "long"  and ev["direction"] != 1:  continue
        entry = ev["entry_px"]; dir2 = ev["direction"]; ba = ev["bar_atr"]
        ref = ev["stop_ref"]
        stop = (ref + sm * ba) if dir2 == -1 else (ref - sm * ba)
        tp2  = ev["tp_ref"]
        risk = (stop - entry) if dir2 == -1 else (entry - stop)
        if risk <= 0: continue
        pnl = None
        for i_b, fb in enumerate(ev["subsequent"][:ts_]):
            if dir2 == -1:
                if fb["h"] >= stop: pnl = (entry - stop) * MES_PV - COST_RT; break
                elif fb["l"] <= tp2: pnl = (entry - tp2) * MES_PV - COST_RT; break
            else:
                if fb["l"] <= stop: pnl = (stop - entry) * MES_PV - COST_RT; break
                elif fb["h"] >= tp2: pnl = (tp2 - entry) * MES_PV - COST_RT; break
            if i_b == min(ts_, len(ev["subsequent"])) - 1:
                pnl = (entry - fb["c"]) * MES_PV - COST_RT if dir2 == -1 else (fb["c"] - entry) * MES_PV - COST_RT
        if pnl is None and ev["subsequent"]:
            pnl = (entry - ev["subsequent"][-1]["c"]) * MES_PV - COST_RT if dir2 == -1 \
                  else (ev["subsequent"][-1]["c"] - entry) * MES_PV - COST_RT
        if pnl is not None:
            oiar_t.append({"date": ev["date"], "pnl": pnl})

    def daily_pnl(trades):
        d = {}
        for t in trades:
            d[t["date"]] = d.get(t["date"], 0.0) + t["pnl"]
        return d

    orb_d  = daily_pnl(orb_t)
    oiar_d = daily_pnl(oiar_t)
    all_d  = sorted(set(orb_d) | set(oiar_d))
    if len(all_d) > 2:
        corr = float(np.corrcoef(
            [orb_d.get(d, 0.0) for d in all_d],
            [oiar_d.get(d, 0.0) for d in all_d]
        )[0, 1])
        print(f"  ORB vs OIAR daily PnL correlation: {corr:.3f}")

    orb_loss_dates   = {d for d, p in orb_d.items() if p < 0}
    orb_notrd_dates  = set(all_d) - set(orb_d)
    oiar_on_orb_loss = [t for t in oiar_t if t["date"] in orb_loss_dates]
    oiar_on_notrd    = [t for t in oiar_t if t["date"] in orb_notrd_dates]
    print_s("  OIAR on ORB-losing days", _stats(oiar_on_orb_loss))
    print_s("  OIAR on ORB-no-trade days", _stats(oiar_on_notrd))


# ---------------------------------------------------------------------------
# SECTION 5: Acceptance criteria
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 5: ACCEPTANCE CRITERIA")
print("=" * 80)

best_s    = best_overall if best_overall else {}
n_pass    = 0

def chk(name, passed, detail=""):
    global n_pass
    if passed:
        n_pass += 1
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}]  {name}" + (f"  ({detail})" if detail else ""))

es_best = simulate(es_events, best_s.get("sm", 0.5), best_s.get("ts", 12),
                   best_s.get("mode", "both"), best_s.get("dir", "both"),
                   best_s.get("strong", False), point_value=ES_PV) if best_s else {"N": 0}

chk("N >= 100 (15yr)", es_best.get("N", 0) >= 100, f"N={es_best.get('N', 0)}")
chk("WR > 50%",        es_best.get("WR", 0) > 0.50, f"WR={es_best.get('WR', 0):.1%}")
chk("Sharpe > 1.5",    es_best.get("Sharpe", 0) > 1.5, f"Sharpe={es_best.get('Sharpe', 0):.3f}")
chk("Profitable in 2022 stress", es_best.get("stress", {}).get("2022", -1) > 0,
    f"${es_best.get('stress', {}).get('2022', 0):+,.0f}")
chk("Profitable in 2020 stress", es_best.get("stress", {}).get("2020", -1) > 0,
    f"${es_best.get('stress', {}).get('2020', 0):+,.0f}")
chk("Configs surviving all 3 stress windows > 0", len(stress_survivors) > 0,
    f"{len(stress_survivors)} configs")

if n_pass >= 5:
    verdict = "STRONG — test live deployment"
elif n_pass >= 4:
    verdict = "MODERATE — promising, refine entry quality gate"
elif n_pass >= 2:
    verdict = "WEAK — partial signal, investigate mode breakdown"
else:
    verdict = "KILL"

print(f"\n  Passed {n_pass}/6 → {verdict}")

# ---------------------------------------------------------------------------
# SECTION 6: Frequency expansion sweep
# Three levers to squeeze more trades from the rejection edge:
#   A — Add rejection SHORT (bullish gap fails → SHORT)
#   B — Lower gap threshold with de6 quality gate
#   C — Grid: gap_norm_min × de6_min × direction
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SECTION 6: FREQUENCY EXPANSION — MORE TRADES WITHOUT KILLING EDGE")
print("=" * 80)

# Re-extract ES events at wider thresholds (0.30, 0.35, 0.40)
# keeping the same best stop/time config
_sm, _ts = best_s.get("sm", 0.75), best_s.get("ts", 18)

def _year_pnl(trades_list, year):
    return sum(t["pnl"] for t in trades_list if t["date"][:4] == year)

def _stress(trades_list):
    s18 = sum(t["pnl"] for t in trades_list if "2018" <= t["date"][:4] <= "2019")
    s20 = sum(t["pnl"] for t in trades_list if t["date"][:4] == "2020")
    s22 = sum(t["pnl"] for t in trades_list if t["date"][:4] == "2022")
    return s18, s20, s22

def _sim_with_de6(events, sm, ts, mode, direction, min_gap_norm, min_de6):
    """Like simulate() but also applies a minimum de6 filter."""
    trades = []
    for ev in events:
        if mode != "both" and ev["mode"] != mode:
            continue
        if direction == "short" and ev["direction"] != -1:
            continue
        if direction == "long" and ev["direction"] != 1:
            continue
        if abs(ev["gap_norm"]) < min_gap_norm:
            continue
        if ev.get("de6", 0.0) < min_de6:
            continue

        entry = ev["entry_px"]
        dir_  = ev["direction"]
        ba    = ev["bar_atr"]
        ref   = ev["stop_ref"]
        tp    = ev["tp_ref"]

        if dir_ == -1:
            stop = ref + sm * ba
            risk = stop - entry
        else:
            stop = ref - sm * ba
            risk = entry - stop
        if risk <= 0:
            continue

        pnl = reason = None
        for i_b, fb in enumerate(ev["subsequent"][:ts]):
            if dir_ == -1:
                if fb["h"] >= stop:
                    pnl = (entry - stop) * ES_PV - COST_RT; reason = "stop"; break
                elif fb["l"] <= tp:
                    pnl = (entry - tp) * ES_PV - COST_RT; reason = "tp"; break
            else:
                if fb["l"] <= stop:
                    pnl = (stop - entry) * ES_PV - COST_RT; reason = "stop"; break
                elif fb["h"] >= tp:
                    pnl = (tp - entry) * ES_PV - COST_RT; reason = "tp"; break
            if i_b == min(ts, len(ev["subsequent"])) - 1:
                pnl = ((entry - fb["c"]) if dir_ == -1 else (fb["c"] - entry)) * ES_PV - COST_RT
                reason = "time"
        if pnl is None and ev["subsequent"]:
            last = ev["subsequent"][-1]
            pnl = ((entry - last["c"]) if dir_ == -1 else (last["c"] - entry)) * ES_PV - COST_RT
            reason = "time"
        if pnl is not None:
            trades.append({"date": ev["date"], "pnl": pnl, "exit": reason,
                           "mode": ev["mode"], "dir": "short" if dir_ == -1 else "long"})
    return trades

# ── A: Both directions at current threshold ──────────────────────────────────
print(f"\n── A: Add rejection SHORT (both directions, GAP_ATR_MIN={GAP_ATR_MIN:.2f}) ──")
print(f"  {'Config':50s}  {'N':>5}  {'WR':>6}  {'E':>7}  {'Sharpe':>7}  Stress(18-19/20/22)")
print("  " + "-" * 95)

for direction in ["long", "short", "both"]:
    t = _sim_with_de6(es_events, _sm, _ts, "rejection", direction,
                      min_gap_norm=GAP_ATR_MIN, min_de6=0.0)
    if not t:
        continue
    s = _stats(t)
    s18, s20, s22 = _stress(t)
    lbl = f"rejection {direction:5s}  gap>={GAP_ATR_MIN:.2f}  de6>=0.00"
    print(f"  {lbl:50s}  {s['N']:5d}  {s['WR']:.1%}  ${s['E']:6.0f}  {s['Sharpe']:7.3f}  "
          f"${s18:+,.0f}/${s20:+,.0f}/${s22:+,.0f}")

# ── B: Lower gap threshold + de6 quality gate ────────────────────────────────
print(f"\n── B: Lower threshold + de6 gate (rejection LONG) ──")
print(f"  {'Config':50s}  {'N':>5}  {'WR':>6}  {'E':>7}  {'Sharpe':>7}  Stress(18-19/20/22)")
print("  " + "-" * 95)

# Extract wider event set at threshold 0.30 (includes everything at 0.35, 0.40, 0.50 too)
_es_events_wide = extract_oiar_events(df_es, es_stats, eth_metrics=None)  # uses module GAP_ATR_MIN
# We need to temporarily override by re-calling with custom threshold
# Since extract_oiar_events uses the module-level GAP_ATR_MIN, patch it inline
_orig_gap = GAP_ATR_MIN

import types

def _extract_wide(bars_5m, stats, min_gap_norm):
    """Re-run extract with a custom min_gap_norm (no module-level override)."""
    events = []
    session_dates = sorted(set(bars_5m.index.date))
    for date in session_dates:
        stat = stats.get(date)
        if stat is None or stat["atr_20d"] is None or stat["prev_close"] is None:
            continue
        atr_20d = stat["atr_20d"]; bar_atr = atr_20d / 13.0
        gap_pts = stat["gap_pts"]; prev_close = stat["prev_close"]
        if abs(gap_pts) / (atr_20d + 1e-8) < min_gap_norm:
            continue
        gap_bullish = gap_pts > 0
        on_midpoint = prev_close + gap_pts / 2.0
        sess = bars_5m[bars_5m.index.date == date].copy()
        sess_bars = list(sess.iterrows()); n_bars = len(sess_bars)
        if n_bars < OR_END_BAR + 3:
            continue
        rth_open = float(sess["open"].iloc[0])
        first6 = sess.iloc[:OR_END_BAR]
        min_c6 = float(first6["close"].min()); max_c6 = float(first6["close"].max())
        sum_tr6 = float(first6.apply(lambda r: r["high"] - r["low"], axis=1).sum()) + 1e-8
        close6 = float(first6["close"].iloc[-1])
        de6 = abs(close6 - rth_open) / sum_tr6
        if gap_bullish:
            rejection = min_c6 < on_midpoint
            strong_rejection = min_c6 < prev_close
        else:
            rejection = max_c6 > on_midpoint
            strong_rejection = max_c6 > prev_close
        if not rejection:
            continue  # only rejection events for this sweep
        entry_found = False
        for i_bar in range(OR_END_BAR, n_bars):
            ts_b, bar = sess_bars[i_bar]
            if ts_b.hour >= CUTOFF_H:
                break
            h = float(bar["high"]); l = float(bar["low"]); c = float(bar["close"])
            rng = h - l + 1e-6; cl = (c - l) / rng
            session_hi = float(sess.iloc[:i_bar]["high"].max())
            session_lo = float(sess.iloc[:i_bar]["low"].min())
            subsequent = [{"h": float(sess_bars[j][1]["high"]),
                           "l": float(sess_bars[j][1]["low"]),
                           "c": float(sess_bars[j][1]["close"])}
                          for j in range(i_bar+1, min(i_bar+30, n_bars))]
            if gap_bullish and c < on_midpoint and cl < 0.45:
                events.append({
                    "date": str(date), "gap_pts": gap_pts,
                    "gap_norm": gap_pts / (atr_20d + 1e-8), "gap_bull": gap_bullish,
                    "mode": "rejection", "direction": -1, "entry_px": c,
                    "stop_ref": session_hi, "tp_ref": prev_close,
                    "on_midpoint": on_midpoint, "bar_atr": bar_atr,
                    "subsequent": subsequent, "de6": de6, "strong": strong_rejection,
                }); entry_found = True; break
            elif not gap_bullish and c > on_midpoint and cl > 0.55:
                events.append({
                    "date": str(date), "gap_pts": gap_pts,
                    "gap_norm": gap_pts / (atr_20d + 1e-8), "gap_bull": gap_bullish,
                    "mode": "rejection", "direction": 1, "entry_px": c,
                    "stop_ref": session_lo, "tp_ref": prev_close,
                    "on_midpoint": on_midpoint, "bar_atr": bar_atr,
                    "subsequent": subsequent, "de6": de6, "strong": strong_rejection,
                }); entry_found = True; break
    return events

es_events_030 = _extract_wide(df_es, es_stats, min_gap_norm=0.30)
print(f"  Wide event pool: {len(es_events_030)} rejection events at gap>=0.30")

# de6 distribution across the wider pool
de6_vals = [e["de6"] for e in es_events_030]
p25, p50, p75 = np.percentile(de6_vals, [25, 50, 75])
print(f"  de6 distribution: p25={p25:.2f} p50={p50:.2f} p75={p75:.2f}")

for gap_min in [0.30, 0.35, 0.40, 0.45, 0.50]:
    for de6_min in [0.00, 0.10, 0.15, 0.20, 0.25, 0.30]:
        t = _sim_with_de6(es_events_030, _sm, _ts, "rejection", "long",
                          min_gap_norm=gap_min, min_de6=de6_min)
        if len(t) < 20:
            continue
        s = _stats(t)
        if s["Sharpe"] < 2.0:
            continue  # skip weak configs
        s18, s20, s22 = _stress(t)
        lbl = f"rejection long  gap>={gap_min:.2f}  de6>={de6_min:.2f}"
        print(f"  {lbl:50s}  {s['N']:5d}  {s['WR']:.1%}  ${s['E']:6.0f}  {s['Sharpe']:7.3f}  "
              f"${s18:+,.0f}/${s20:+,.0f}/${s22:+,.0f}")

# ── C: Both directions + de6 gate ────────────────────────────────────────────
print(f"\n── C: Both directions + de6 gate ──")
print(f"  {'Config':50s}  {'N':>5}  {'WR':>6}  {'E':>7}  {'Sharpe':>7}  Stress(18-19/20/22)")
print("  " + "-" * 95)

es_events_030_both = _extract_wide(df_es, es_stats, min_gap_norm=0.30)
# Also need SHORT events — _extract_wide already returns both directions
for gap_min in [0.35, 0.40, 0.50]:
    for de6_min in [0.00, 0.15, 0.20]:
        t = _sim_with_de6(es_events_030, _sm, _ts, "rejection", "both",
                          min_gap_norm=gap_min, min_de6=de6_min)
        if len(t) < 30:
            continue
        s = _stats(t)
        if s["Sharpe"] < 2.0:
            continue
        s18, s20, s22 = _stress(t)
        survivors = all([s18 >= 0, s20 > 0, s22 > 0])
        flag = " ★ ALL STRESS PASS" if survivors else ""
        lbl = f"rejection both  gap>={gap_min:.2f}  de6>={de6_min:.2f}"
        print(f"  {lbl:50s}  {s['N']:5d}  {s['WR']:.1%}  ${s['E']:6.0f}  {s['Sharpe']:7.3f}  "
              f"${s18:+,.0f}/${s20:+,.0f}/${s22:+,.0f}{flag}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n── Summary: current best vs best expansion configs ──")
baseline = _sim_with_de6(es_events, _sm, _ts, "rejection", "long",
                         min_gap_norm=GAP_ATR_MIN, min_de6=0.0)
sb = _stats(baseline); s18b, s20b, s22b = _stress(baseline)
print(f"  Baseline (gap>=0.50 long):  N={sb['N']} WR={sb['WR']:.1%} Sharpe={sb['Sharpe']:.2f}  "
      f"18-19=${s18b:+,.0f} 2020=${s20b:+,.0f} 2022=${s22b:+,.0f}")

# Save
out = {
    "events": {"es": len(es_events), "mes": len(mes_events),
               "oos": len(oos_events), "mnq": len(mnq_events)},
    "es_mode_split": {"rejection": len(es_rej), "acceptance": len(es_acc)},
    "best_config": {k: v for k, v in best_s.items() if k not in ("stress",)},
    "best_stats": {k: v for k, v in es_best.items() if k != "stress"},
    "stress": es_best.get("stress", {}),
    "stress_survivors": len(stress_survivors),
    "criteria_passed": n_pass,
    "verdict": verdict,
}
out_path = ROOT / "rule_based_v1/diagnostics/oiar_results.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nSaved → {out_path}")
