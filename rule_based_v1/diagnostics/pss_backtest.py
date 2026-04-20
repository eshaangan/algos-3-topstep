"""
PSS — Prior Session Sweep Micro-Edge Backtest
=============================================
Isolates the `prev_sess_sweep` kernel from IFR and tests it with:

  Variant A: standalone with a stop/TP grid sweep
    - Tighter stop options: [0.1, 0.25, 0.5, 0.75] × bar_atr
    - Faster TP options: [0.5R, 0.75R, 1.0R, 1.5R, VWAP]
    - Time stop options: [2, 3, 4, 6] bars
    - Short-only vs both-sides
    - Higher quality gate: FS threshold sweep [0.35, 0.50, 0.65, 0.80]

  Variant B: ORB filter
    - Does a prior-session sweep REJECTION predict ORB failure?
    - Does a prior-session sweep ACCEPTANCE (continuation) predict ORB success?
    - Test: filter ORB trades by sweep outcome from previous bars in the session

Acceptance criteria (hard):
  1. Positive 15-year expectancy
  2. Profitable in 2018-2019, 2020, 2022 stress windows
  3. N >= 100 trades total
  4. Daily PnL correlation to ORB < 0.30
"""
import json
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

print("Loading ES 15-year data…")
with pd.HDFStore(str(ROOT / "data/processed/es_bars_2010_2025.h5"), "r") as s:
    df_es = s["/bars_5min"].copy()

df_es.index = pd.to_datetime(df_es["timestamp"], unit="s", utc=True).dt.tz_convert("US/Eastern")
df_es = df_es.drop(columns=["timestamp"]).sort_index()
mask = (
    ((df_es.index.hour == 9) & (df_es.index.minute >= 30))
    | ((df_es.index.hour > 9) & (df_es.index.hour < 16))
)
df_es = df_es[mask].copy()
print(f"ES: {len(df_es):,} bars, {len(set(df_es.index.date)):,} sessions  "
      f"({df_es.index[0].date()} – {df_es.index[-1].date()})")

ES_PV   = 50.0
COST_RT = 6.25

# ---------------------------------------------------------------------------
# Session stats
# ---------------------------------------------------------------------------

def build_stats(bars: pd.DataFrame) -> dict:
    sessions = sorted(bars.groupby(bars.index.date), key=lambda x: x[0])
    rows = []
    for date, sess in sessions:
        if len(sess) < 2:
            continue
        rows.append({
            "date": date,
            "open":  float(sess["open"].iloc[0]),
            "high":  float(sess["high"].max()),
            "low":   float(sess["low"].min()),
            "close": float(sess["close"].iloc[-1]),
            "vol30": float(sess["volume"].iloc[:6].sum()),
        })

    result = {}
    prev_close = None
    atr_val = None
    vol30_hist = []

    for i, row in enumerate(rows):
        hi, lo, cl = row["high"], row["low"], row["close"]
        tr = (hi - lo) if prev_close is None else max(
            hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        atr_val = tr if atr_val is None else (atr_val * 13 + tr) / 14

        vol30_hist.append(row["vol30"])
        avg_vol = float(np.mean(vol30_hist[-20:])) if len(vol30_hist) >= 20 else None

        prev_1d = (row["close"] - rows[i-1]["close"]) / (rows[i-1]["close"] + 1e-8) if i >= 1 else 0.0
        prev_3d = (row["close"] - rows[i-3]["close"]) / (rows[i-3]["close"] + 1e-8) if i >= 3 else 0.0
        gap     = (row["open"]  - rows[i-1]["close"]) / (rows[i-1]["close"] + 1e-8) if i >= 1 else 0.0

        result[row["date"]] = {
            "atr_20d":   atr_val,
            "vol30":     row["vol30"],
            "avg_vol30": avg_vol,
            "prev_1d":   prev_1d,
            "prev_3d":   prev_3d,
            "gap":       gap,
            "prev_hi":   rows[i-1]["high"] if i > 0 else None,
            "prev_lo":   rows[i-1]["low"]  if i > 0 else None,
        }
        prev_close = cl

    return result


def classify_regime(sess: pd.DataFrame, stat: dict) -> str:
    if len(sess) < 7 or stat["atr_20d"] is None:
        return "trend"
    first6 = sess.iloc[:6]
    op = float(sess["open"].iloc[0])
    atr = stat["atr_20d"]

    rng6 = float(first6["high"].max()) - float(first6["low"].min())
    vwap_num = sum((b["high"] + b["low"] + b["close"]) / 3.0 * b["volume"] for _, b in first6.iterrows())
    vwap_den = float(first6["volume"].sum())
    vwap = vwap_num / (vwap_den + 1e-8)
    vs = (vwap - op) / (op + 1e-8)

    avg_v = stat["avg_vol30"]
    rv = stat["vol30"] / avg_v if avg_v else 1.0

    sum_tr6 = float(first6.apply(lambda r: r["high"] - r["low"], axis=1).sum()) + 1e-8
    de = abs(float(first6["close"].iloc[-1]) - op) / sum_tr6

    neg = (int(stat["gap"] < -0.002) + int(stat["prev_1d"] < -0.005)
           + int(stat["prev_3d"] < -0.01) + int(rng6 / atr > 1.3)
           + (int(rv > 1.3) if avg_v else 0) + int(vs < -0.0003))

    if neg >= 2 and de > 0.25:
        return "negative"
    if de < 0.25 and abs(vs) < 0.0005:
        return "chop"
    return "trend"


def fs_short(bar: pd.Series) -> float:
    """Failure score for upside sweep → SHORT."""
    rng = float(bar["high"]) - float(bar["low"]) + 1e-6
    clv = (2 * float(bar["close"]) - float(bar["high"]) - float(bar["low"])) / rng
    uw = float(bar["high"]) - max(float(bar["open"]), float(bar["close"]))
    lw = min(float(bar["open"]), float(bar["close"])) - float(bar["low"])
    body_signed = float(bar["close"]) - float(bar["open"])
    return -clv + (uw - lw) / rng - body_signed / rng


def fs_long(bar: pd.Series) -> float:
    """Failure score for downside sweep → LONG."""
    rng = float(bar["high"]) - float(bar["low"]) + 1e-6
    clv = (2 * float(bar["close"]) - float(bar["high"]) - float(bar["low"])) / rng
    uw = float(bar["high"]) - max(float(bar["open"]), float(bar["close"]))
    lw = min(float(bar["open"]), float(bar["close"])) - float(bar["low"])
    body_signed = float(bar["close"]) - float(bar["open"])
    return clv - (uw - lw) / rng + body_signed / rng


# ---------------------------------------------------------------------------
# PSS event detector — extract all sweep events per session
# ---------------------------------------------------------------------------

OR_END_BAR   = 7
CUTOFF_H     = 14
CL_SHORT_MAX = 0.45
CL_LONG_MIN  = 0.55
BODY_MAX     = 0.55
REL_VOL_MIN  = 1.10


def extract_pss_events(bars: pd.DataFrame, stats: dict,
                        direction: str = "short") -> list:
    """
    Return list of PSS sweep events with all bar-level data needed to
    simulate any exit configuration.

    Each event dict contains:
      date, sweep_bar_i, entry_bar_i (i+1), entry_px, stop_extreme,
      bar_atr, vwap_at_entry, regime, fs,
      subsequent_bars: list of (ts, h, l, c)
    """
    events = []
    session_dates = sorted(set(bars.index.date))

    for date in session_dates:
        sess = bars[bars.index.date == date].copy()
        if len(sess) < OR_END_BAR + 3:
            continue
        stat = stats.get(date)
        if stat is None or stat["atr_20d"] is None:
            continue

        atr_20d  = stat["atr_20d"]
        bar_atr  = atr_20d / 13.0
        prev_hi  = stat["prev_hi"]
        prev_lo  = stat["prev_lo"]

        if direction == "short" and prev_hi is None:
            continue
        if direction == "long" and prev_lo is None:
            continue

        regime = classify_regime(sess, stat)
        if regime == "trend":
            continue

        vwap_num, vwap_den = 0.0, 0.0
        vol_hist: list = []
        triggered = False  # one event per session

        sess_bars = list(sess.iterrows())
        n_bars = len(sess_bars)

        for i_bar, (ts, bar) in enumerate(sess_bars):
            h = float(bar["high"])
            l = float(bar["low"])
            c = float(bar["close"])
            v = float(bar["volume"])

            typical   = (h + l + c) / 3.0
            vwap_num += typical * v
            vwap_den += v
            vwap = vwap_num / (vwap_den + 1e-8)

            vol_hist.append(v)
            vol_med = float(np.median(vol_hist[-20:])) if len(vol_hist) >= 5 else v
            rel_vol = v / (vol_med + 1e-8)

            if i_bar < OR_END_BAR:
                continue
            if ts.hour >= CUTOFF_H:
                break
            if triggered:
                continue

            rng       = h - l + 1e-6
            body_frac = abs(c - float(bar["open"])) / rng
            cl_val    = (c - l) / rng

            if direction == "short":
                quality = (cl_val < CL_SHORT_MAX and body_frac < BODY_MAX
                           and rel_vol > REL_VOL_MIN and h > prev_hi)
                if not quality:
                    continue
                fs = fs_short(bar)
            else:
                quality = (cl_val > CL_LONG_MIN and body_frac < BODY_MAX
                           and rel_vol > REL_VOL_MIN and l < prev_lo)
                if not quality:
                    continue
                fs = fs_long(bar)

            # Confirmation: next bar must confirm failure
            # (don't enter on sweep bar itself — wait for close inside range)
            if i_bar + 1 >= n_bars:
                continue
            next_ts, next_bar = sess_bars[i_bar + 1]
            next_c = float(next_bar["close"])

            if direction == "short" and next_c >= prev_hi:
                continue   # price still above swept level — no confirmation
            if direction == "long"  and next_c <= prev_lo:
                continue

            entry_px      = next_c
            sweep_extreme = h if direction == "short" else l
            entry_bar_i   = i_bar + 1

            # Collect subsequent bars for exit simulation
            subsequent = []
            for j in range(entry_bar_i + 1, min(entry_bar_i + 20, n_bars)):
                _, fb = sess_bars[j]
                subsequent.append({
                    "h": float(fb["high"]),
                    "l": float(fb["low"]),
                    "c": float(fb["close"]),
                })

            events.append({
                "date":          str(date),
                "regime":        regime,
                "direction":     direction,
                "entry_px":      entry_px,
                "sweep_extreme": sweep_extreme,
                "bar_atr":       bar_atr,
                "vwap":          vwap,
                "fs":            fs,
                "subsequent":    subsequent,
            })
            triggered = True

    return events


# ---------------------------------------------------------------------------
# Simulate exit configuration on extracted events
# ---------------------------------------------------------------------------

def simulate(events: list, stop_atr_mult: float, tp_r_mult: float,
             time_stop_bars: int, min_fs: float,
             direction: str = "short",
             tp_mode: str = "fixed") -> dict:
    """
    tp_mode: "fixed" = entry ± tp_r_mult * risk
             "vwap"  = vwap (target) or tp_r_mult * risk, whichever closer
    """
    trades = []
    for ev in events:
        if ev["fs"] < min_fs:
            continue
        if ev["direction"] != direction:
            continue

        entry    = ev["entry_px"]
        ext      = ev["sweep_extreme"]
        bar_atr  = ev["bar_atr"]
        vwap     = ev["vwap"]

        if direction == "short":
            stop = ext + stop_atr_mult * bar_atr
            risk = stop - entry
            if risk <= 0:
                continue
            tp_fixed = entry - tp_r_mult * risk
            if tp_mode == "vwap" and vwap < entry and (entry - vwap) >= 0.5 * risk:
                tp = vwap
            else:
                tp = tp_fixed
        else:
            stop = ext - stop_atr_mult * bar_atr
            risk = entry - stop
            if risk <= 0:
                continue
            tp_fixed = entry + tp_r_mult * risk
            if tp_mode == "vwap" and vwap > entry and (vwap - entry) >= 0.5 * risk:
                tp = vwap
            else:
                tp = tp_fixed

        pnl = None
        exit_r = None
        for i_bar, fb in enumerate(ev["subsequent"][:time_stop_bars]):
            if direction == "short":
                if fb["h"] >= stop:
                    pnl = (entry - stop) * ES_PV - COST_RT
                    exit_r = "stop"
                    break
                elif fb["l"] <= tp:
                    pnl = (entry - tp) * ES_PV - COST_RT
                    exit_r = "tp"
                    break
            else:
                if fb["l"] <= stop:
                    pnl = (stop - entry) * ES_PV - COST_RT
                    exit_r = "stop"
                    break
                elif fb["h"] >= tp:
                    pnl = (tp - entry) * ES_PV - COST_RT
                    exit_r = "tp"
                    break

            if i_bar == min(time_stop_bars, len(ev["subsequent"])) - 1:
                pnl = (entry - fb["c"]) * ES_PV - COST_RT if direction == "short" \
                      else (fb["c"] - entry) * ES_PV - COST_RT
                exit_r = "time"

        if pnl is None and ev["subsequent"]:
            last = ev["subsequent"][-1]
            pnl = (entry - last["c"]) * ES_PV - COST_RT if direction == "short" \
                  else (last["c"] - entry) * ES_PV - COST_RT
            exit_r = "time"

        if pnl is not None:
            trades.append({"date": ev["date"], "pnl": pnl, "exit": exit_r,
                           "regime": ev["regime"]})

    if not trades:
        return {"N": 0, "Sharpe": -99.0, "E": 0.0, "WR": 0.0,
                "PnL": 0.0, "MaxDD": 0.0}

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

    # Stress-window check
    stress = {}
    for period, yr_range in [("2018-2019", ("2018", "2019")),
                              ("2020",      ("2020", "2020")),
                              ("2022",      ("2022", "2022"))]:
        sub = [t for t in trades if yr_range[0] <= t["date"][:4] <= yr_range[1]]
        stress[period] = round(float(np.array([t["pnl"] for t in sub]).sum()), 2) if sub else 0.0

    return {
        "N": len(p), "WR": round(WR, 4), "E": round(E, 2),
        "PnL": round(float(p.sum()), 2), "Sharpe": round(sharpe, 3),
        "MaxDD": round(max_dd, 2), "stress": stress,
    }


# ---------------------------------------------------------------------------
# Extract events
# ---------------------------------------------------------------------------

print("\nBuilding session stats…")
stats = build_stats(df_es)

print("Extracting PSS events (short)…")
events_short = extract_pss_events(df_es, stats, direction="short")
print(f"  Short sweep events (chop/neg, quality gate): {len(events_short)}")

print("Extracting PSS events (long)…")
events_long  = extract_pss_events(df_es, stats, direction="long")
print(f"  Long  sweep events (chop/neg, quality gate): {len(events_long)}")


# ---------------------------------------------------------------------------
# VARIANT A: Stop/TP/Time-stop/FS grid sweep
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("VARIANT A: Stop × TP × Time × FS Grid Sweep (Short-Only)")
print("=" * 80)

stop_mults  = [0.10, 0.25, 0.50, 0.75]
tp_r_mults  = [0.50, 0.75, 1.00, 1.50, 2.00]
time_stops  = [2, 3, 4, 6]
fs_thresholds = [0.35, 0.50, 0.65, 0.80]
tp_modes    = ["fixed", "vwap"]

best = {"Sharpe": -99.0}
results = []

for stop_m, tp_r, ts, fs_t, tp_mode in product(stop_mults, tp_r_mults, time_stops, fs_thresholds, tp_modes):
    r = simulate(events_short, stop_m, tp_r, ts, fs_t, "short", tp_mode)
    if r["N"] < 30:
        continue
    r.update({"stop_m": stop_m, "tp_r": tp_r, "ts": ts, "fs_t": fs_t, "tp_mode": tp_mode})
    results.append(r)
    if r["Sharpe"] > best["Sharpe"]:
        best = r

# Show top 15 by Sharpe
results_sorted = sorted(results, key=lambda x: x["Sharpe"], reverse=True)

print(f"\n{'Config':50s}  {'N':>5}  {'WR':>6}  {'E':>7}  {'Sharpe':>7}  {'MaxDD':>9}  Stress(18/20/22)")
print("-" * 120)
for r in results_sorted[:20]:
    cfg = (f"stop={r['stop_m']:.2f}× tp={r['tp_r']:.2f}R ts={r['ts']}b "
           f"fs>={r['fs_t']:.2f} {r['tp_mode']:5s}")
    s18 = r["stress"].get("2018-2019", 0)
    s20 = r["stress"].get("2020", 0)
    s22 = r["stress"].get("2022", 0)
    print(f"  {cfg:50s}  {r['N']:5d}  {r['WR']:.1%}  ${r['E']:6.0f}  {r['Sharpe']:7.3f}  "
          f"${r['MaxDD']:8,.0f}  ${s18:+,.0f}/${s20:+,.0f}/${s22:+,.0f}")

print(f"\n  BEST CONFIG: stop={best.get('stop_m','?')}× tp={best.get('tp_r','?')}R "
      f"ts={best.get('ts','?')}b fs>={best.get('fs_t','?')} {best.get('tp_mode','?')}")
print(f"  Sharpe={best.get('Sharpe',0):.3f}, E=${best.get('E',0):.2f}/trade, N={best.get('N',0)}")

# Stress gate: must be positive in all three
stress_survivors = [r for r in results if
                    r["stress"].get("2018-2019", -1) > 0
                    and r["stress"].get("2020", -1) > 0
                    and r["stress"].get("2022", -1) > 0
                    and r["N"] >= 50]

print(f"\n  Configs surviving ALL 3 stress windows (N>={50}): {len(stress_survivors)}")
if stress_survivors:
    best_stress = max(stress_survivors, key=lambda x: x["Sharpe"])
    print(f"  Best stress-survivor: stop={best_stress['stop_m']}× tp={best_stress['tp_r']}R "
          f"ts={best_stress['ts']}b fs>={best_stress['fs_t']} {best_stress['tp_mode']}")
    print(f"    Sharpe={best_stress['Sharpe']:.3f}, E=${best_stress['E']:.2f}, "
          f"N={best_stress['N']}, PnL=${best_stress['PnL']:,.0f}")


# ---------------------------------------------------------------------------
# VARIANT A2: Long-side test with best short config
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("VARIANT A2: Long-side test")
print("=" * 80)

long_results = []
for stop_m, tp_r, ts, fs_t, tp_mode in product(stop_mults, tp_r_mults, time_stops, fs_thresholds, tp_modes):
    r = simulate(events_long, stop_m, tp_r, ts, fs_t, "long", tp_mode)
    if r["N"] < 20:
        continue
    r.update({"stop_m": stop_m, "tp_r": tp_r, "ts": ts, "fs_t": fs_t, "tp_mode": tp_mode})
    long_results.append(r)

long_survivors = [r for r in long_results if
                  r["stress"].get("2018-2019", -1) > 0
                  and r["stress"].get("2020", -1) > 0
                  and r["stress"].get("2022", -1) > 0
                  and r["Sharpe"] > 0
                  and r["N"] >= 30]
print(f"  Long configs passing all stress windows AND Sharpe>0: {len(long_survivors)}")
if long_survivors:
    bl = max(long_survivors, key=lambda x: x["Sharpe"])
    print(f"  Best: stop={bl['stop_m']}× tp={bl['tp_r']}R ts={bl['ts']}b fs>={bl['fs_t']} "
          f"Sharpe={bl['Sharpe']:.3f} E=${bl['E']:.2f} N={bl['N']}")
else:
    print("  No long configuration passes all stress windows. LONG SIDE: KILL.")


# ---------------------------------------------------------------------------
# VARIANT B: ORB filter — does PSS predict ORB outcome?
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("VARIANT B: ORB Filter — does prev_sess_sweep predict ORB outcome?")
print("=" * 80)

ORB_CFG = {
    "or_end_bar": 7, "min_range_atr": 0.3, "pt_mult": 2.0, "sl_mult": 1.5,
    "time_stop_bars": 24, "entry_cutoff_h": 12, "long_only": True, "require_prev_vwap": True,
}


def run_orb(bars: pd.DataFrame, stats: dict, cfg: dict) -> list:
    session_dates = sorted(set(bars.index.date))

    # prev_vwap per date
    pv_flag = {}
    for i, date in enumerate(session_dates):
        if i == 0:
            pv_flag[date] = False
            continue
        prev = session_dates[i-1]
        ps = bars[bars.index.date == prev]
        if len(ps) < 2:
            pv_flag[date] = False
            continue
        tp = (ps["high"] + ps["low"] + ps["close"]) / 3
        pvwap = float((tp * ps["volume"]).sum() / (ps["volume"].sum() + 1e-8))
        pv_flag[date] = float(ps["close"].iloc[-1]) > pvwap

    trades = []
    for date in session_dates:
        sess = bars[bars.index.date == date].copy()
        stat = stats.get(date)
        if stat is None or stat["atr_20d"] is None:
            continue
        if cfg["require_prev_vwap"] and not pv_flag.get(date, False):
            continue

        atr = stat["atr_20d"]
        or_end = cfg["or_end_bar"]
        if len(sess) <= or_end:
            continue

        or_h = float(sess.iloc[:or_end]["high"].max())
        or_l = float(sess.iloc[:or_end]["low"].min())
        if (or_h - or_l) < cfg["min_range_atr"] * atr:
            continue

        in_trade = False
        entry_px = stop_px = tgt_px = 0.0
        eb = 0
        direction = 0

        for i, (ts, bar) in enumerate(sess.iterrows()):
            if i < or_end:
                continue
            if ts.hour >= cfg["entry_cutoff_h"] and not in_trade:
                break
            if in_trade:
                force = (i == len(sess) - 1)
                if float(bar["high"]) >= tgt_px:
                    pnl = (tgt_px - entry_px) * direction * ES_PV - COST_RT
                    trades.append({"date": str(date), "pnl": round(pnl, 2), "exit": "target"})
                    in_trade = False
                elif float(bar["low"]) <= stop_px:
                    pnl = (stop_px - entry_px) * direction * ES_PV - COST_RT
                    trades.append({"date": str(date), "pnl": round(pnl, 2), "exit": "stop"})
                    in_trade = False
                elif (i - eb) >= cfg["time_stop_bars"] or force:
                    pnl = (float(bar["close"]) - entry_px) * direction * ES_PV - COST_RT
                    trades.append({"date": str(date), "pnl": round(pnl, 2), "exit": "time"})
                    in_trade = False
                continue

            if float(bar["close"]) > or_h:
                entry_px, direction = float(bar["close"]), 1
                stop_px  = entry_px - cfg["sl_mult"] * atr
                tgt_px   = entry_px + cfg["pt_mult"] * atr
                in_trade, eb = True, i
            elif not cfg["long_only"] and float(bar["close"]) < or_l:
                entry_px, direction = float(bar["close"]), -1
                stop_px  = entry_px + cfg["sl_mult"] * atr
                tgt_px   = entry_px - cfg["pt_mult"] * atr
                in_trade, eb = True, i

    return trades


# PSS event dates — dates where a prev_sess_sweep occurred in chop/negative
pss_short_dates = set(ev["date"] for ev in events_short)
pss_long_dates  = set(ev["date"] for ev in events_long)

orb_trades = run_orb(df_es, stats, ORB_CFG)
print(f"  ORB total trades: {len(orb_trades)}")

# Split ORB trades by whether a PSS event occurred the same day
orb_on_pss_short = [t for t in orb_trades if t["date"] in pss_short_dates]
orb_off_pss_short = [t for t in orb_trades if t["date"] not in pss_short_dates]


def qs(trades):
    if not trades:
        return "N=0"
    p = np.array([t["pnl"] for t in trades])
    WR = float((p > 0).mean())
    E  = float(p.mean())
    by_day = {}
    for t in trades:
        by_day.setdefault(t["date"], 0.0)
        by_day[t["date"]] += t["pnl"]
    daily = np.array(list(by_day.values()))
    sh = float((daily.mean() / (daily.std() + 1e-8)) * np.sqrt(252)) if len(daily) > 1 else 0.0
    return (f"N={len(p)}, WR={WR:.1%}, E=${E:.0f}/trade, "
            f"PnL=${p.sum():,.0f}, Sharpe={sh:.2f}")


print(f"\n  ORB all:                        {qs(orb_trades)}")
print(f"  ORB on PSS-short days:          {qs(orb_on_pss_short)}")
print(f"  ORB on non-PSS-short days:      {qs(orb_off_pss_short)}")

# Chi-square style comparison: does PSS predict ORB failure?
orb_pss_wins  = [t for t in orb_on_pss_short if t["pnl"] > 0]
orb_npss_wins = [t for t in orb_off_pss_short if t["pnl"] > 0]
pss_wr  = len(orb_pss_wins) / max(len(orb_on_pss_short), 1)
npss_wr = len(orb_npss_wins) / max(len(orb_off_pss_short), 1)
print(f"\n  ORB WR on PSS days:      {pss_wr:.1%}  ({len(orb_on_pss_short)} trades)")
print(f"  ORB WR on non-PSS days:  {npss_wr:.1%}  ({len(orb_off_pss_short)} trades)")

delta_wr = npss_wr - pss_wr
print(f"\n  WR delta (non-PSS minus PSS): {delta_wr:+.1%}")
if delta_wr > 0.05:
    print("  → PSS days PREDICT WORSE ORB outcomes. PSS as ORB veto filter: VIABLE")
    orb_filtered = orb_off_pss_short
    print(f"  ORB after PSS veto: {qs(orb_filtered)}")
elif delta_wr < -0.05:
    print("  → PSS days PREDICT BETTER ORB outcomes. PSS as confirmation: VIABLE")
else:
    print("  → PSS has no predictive power for ORB outcome. ORB FILTER: NOT VIABLE")

# Bonus: Correlation of daily PnL
orb_daily = {}
for t in orb_trades:
    orb_daily[t["date"]] = orb_daily.get(t["date"], 0.0) + t["pnl"]

# Short PSS trades using best config (or baseline)
best_cfg = best if best.get("N", 0) > 0 else {"stop_m": 0.25, "tp_r": 1.0, "ts": 3, "fs_t": 0.50, "tp_mode": "fixed"}
pss_trades_best = []
for ev in events_short:
    if ev["fs"] < best_cfg.get("fs_t", 0.50):
        continue
    entry  = ev["entry_px"]
    ext    = ev["sweep_extreme"]
    ba     = ev["bar_atr"]
    stop   = ext + best_cfg.get("stop_m", 0.25) * ba
    risk   = stop - entry
    if risk <= 0:
        continue
    tp = entry - best_cfg.get("tp_r", 1.0) * risk

    pnl = None
    for i_b, fb in enumerate(ev["subsequent"][:best_cfg.get("ts", 3)]):
        if fb["h"] >= stop:
            pnl = (entry - stop) * ES_PV - COST_RT; break
        elif fb["l"] <= tp:
            pnl = (entry - tp) * ES_PV - COST_RT; break
        if i_b == min(best_cfg.get("ts", 3), len(ev["subsequent"])) - 1:
            pnl = (entry - fb["c"]) * ES_PV - COST_RT

    if pnl is None and ev["subsequent"]:
        pnl = (entry - ev["subsequent"][-1]["c"]) * ES_PV - COST_RT

    if pnl is not None:
        pss_trades_best.append({"date": ev["date"], "pnl": pnl})

pss_daily = {}
for t in pss_trades_best:
    pss_daily[t["date"]] = pss_daily.get(t["date"], 0.0) + t["pnl"]

all_dates = sorted(set(orb_daily) | set(pss_daily))
if len(all_dates) > 2:
    orb_arr = np.array([orb_daily.get(d, 0.0) for d in all_dates])
    pss_arr = np.array([pss_daily.get(d, 0.0) for d in all_dates])
    corr = float(np.corrcoef(orb_arr, pss_arr)[0, 1])
    print(f"\n  Daily PnL correlation (ORB vs PSS best config): {corr:.3f}")


# ---------------------------------------------------------------------------
# Final acceptance assessment
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL ASSESSMENT")
print("=" * 80)

def passes_all_stress(r):
    return (r["stress"].get("2018-2019", -1) > 0
            and r["stress"].get("2020", -1) > 0
            and r["stress"].get("2022", -1) > 0)

best_overall_sharpe = max((r for r in results if r["N"] >= 30),
                          key=lambda x: x["Sharpe"], default={"Sharpe": -99})
best_stress_pass    = stress_survivors[0] if stress_survivors else None

print(f"  Best Sharpe found (short, any stress): "
      f"Sharpe={best_overall_sharpe.get('Sharpe', 0):.3f}, "
      f"E=${best_overall_sharpe.get('E', 0):.0f}/trade, "
      f"N={best_overall_sharpe.get('N', 0)}")

print(f"  Best config passing ALL stress windows: "
      + (f"Sharpe={best_stress_pass['Sharpe']:.3f}, E=${best_stress_pass['E']:.0f}/trade, "
         f"N={best_stress_pass['N']}"
         if best_stress_pass else "NONE FOUND"))

long_viable = len(long_survivors) > 0
short_viable = (best_stress_pass is not None and
                best_stress_pass.get("Sharpe", 0) > 0.5 and
                best_stress_pass.get("E", 0) > 0)
orb_filter_viable = abs(delta_wr) > 0.05

print(f"\n  Short PSS viable:    {'YES' if short_viable else 'NO'}")
print(f"  Long  PSS viable:    {'YES' if long_viable  else 'NO'}")
print(f"  ORB filter viable:   {'YES' if orb_filter_viable else 'NO'}")

if not any([short_viable, long_viable, orb_filter_viable]):
    print("\n  *** VERDICT: KILL prev_sess_sweep. No configuration survives stress + quality gates. ***")
    print("       The kernel has no extractable edge. ORB is the only validated strategy.")
elif orb_filter_viable:
    print("\n  *** VERDICT: prev_sess_sweep as ORB veto VIABLE. Test deployment as ORB pre-filter. ***")
elif short_viable:
    print("\n  *** VERDICT: Short PSS has narrow edge. Deploy with strict parameters above. ***")
else:
    print("\n  *** VERDICT: MARGINAL. Review stress periods before deploying. ***")

# Save results
out = {
    "short_events": len(events_short),
    "long_events":  len(events_long),
    "best_short_sharpe": {k: v for k, v in best_overall_sharpe.items() if k != "stress"},
    "best_stress_survivor": {k: v for k, v in best_stress_pass.items() if k != "stress"} if best_stress_pass else None,
    "stress_survivor_count": len(stress_survivors),
    "long_viable":         long_viable,
    "orb_wr_on_pss_days":  round(pss_wr, 4),
    "orb_wr_off_pss_days": round(npss_wr, 4),
    "orb_filter_viable":   orb_filter_viable,
}
out_path = ROOT / "rule_based_v1/diagnostics/pss_results.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nSaved → {out_path}")
