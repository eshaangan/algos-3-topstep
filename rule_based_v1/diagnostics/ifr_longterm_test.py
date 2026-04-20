"""
IFR Long-Term Validation — ES 2010-2025 (3,995 sessions)
=========================================================
Tests the core IFR signal on 15 years of ES data:
  - Regime-gated (chop/negative only)
  - prev_sess_high/low sweep + quality gate
  - Rolling 20-bar intraday extension sweep
  - SHORT only

Acceptance threshold: Sharpe > 1.5 with N > 200 trades and positive WR.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Data loading — ES 15-year (already RTH filtered)
# ---------------------------------------------------------------------------

print("Loading ES 15-year data…")
with pd.HDFStore(str(ROOT / "data/processed/es_bars_2010_2025.h5"), "r") as s:
    df_es = s["/bars_5min"].copy()

df_es.index = pd.to_datetime(df_es["timestamp"], unit="s", utc=True).dt.tz_convert("US/Eastern")
df_es = df_es.drop(columns=["timestamp"]).sort_index()

# Restrict to RTH (already filtered but be safe)
mask = (
    ((df_es.index.hour == 9) & (df_es.index.minute >= 30))
    | ((df_es.index.hour > 9) & (df_es.index.hour < 16))
)
df_es = df_es[mask].copy()
print(f"ES bars: {len(df_es):,}  ({df_es.index[0].date()} – {df_es.index[-1].date()})")
print(f"ES sessions: {len(set(df_es.index.date)):,}")

# ---------------------------------------------------------------------------
# Session stats builder
# ---------------------------------------------------------------------------

def build_session_stats(bars_5m: pd.DataFrame) -> dict:
    """Returns dict keyed by date."""
    sessions = sorted(bars_5m.groupby(bars_5m.index.date), key=lambda x: x[0])
    rows = []
    for date, sess in sessions:
        if len(sess) < 2:
            continue
        rows.append({
            "date": date,
            "open": float(sess["open"].iloc[0]),
            "high": float(sess["high"].max()),
            "low": float(sess["low"].min()),
            "close": float(sess["close"].iloc[-1]),
            "vol_first_30m": float(sess["volume"].iloc[:6].sum()),
        })

    result = {}
    prev_close_val = None
    atr_val = None
    vol_history = []

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
            "date": row["date"],
            "atr_20d": atr_val,
            "vol_first_30m": row["vol_first_30m"],
            "rolling_20d_avg_vol_first_30m": avg_vol_30m,
            "prev_1d_ret": prev_1d_ret,
            "prev_3d_ret": prev_3d_ret,
            "gap_pct": gap_pct,
            "prev_sess_high": rows[i-1]["high"] if i > 0 else None,
            "prev_sess_low":  rows[i-1]["low"]  if i > 0 else None,
        }
        prev_close_val = cl

    return result


def classify_regime(session: pd.DataFrame, stat: dict) -> str:
    if len(session) < 7:
        return "trend"
    atr_20d = stat["atr_20d"]
    if atr_20d is None or atr_20d <= 0:
        return "trend"

    first6 = session.iloc[:6]
    open_price = float(session["open"].iloc[0])

    first_30m_range = float(first6["high"].max()) - float(first6["low"].min())
    range_ratio = first_30m_range / (atr_20d + 1e-8)

    vwap_num = sum((b["high"] + b["low"] + b["close"]) / 3.0 * b["volume"] for _, b in first6.iterrows())
    vwap_den = float(first6["volume"].sum())
    vwap_30m = vwap_num / (vwap_den + 1e-8)
    vwap_slope = (vwap_30m - open_price) / (open_price + 1e-8)

    avg_vol = stat["rolling_20d_avg_vol_first_30m"]
    rel_vol = stat["vol_first_30m"] / avg_vol if avg_vol else 1.0
    use_rv = avg_vol is not None

    sum_tr6 = float(first6.apply(lambda r: r["high"] - r["low"], axis=1).sum()) + 1e-8
    close6 = float(first6["close"].iloc[-1])
    de = abs(close6 - open_price) / sum_tr6

    neg_count = (
        int(stat["gap_pct"] < -0.002)
        + int(stat["prev_1d_ret"] < -0.005)
        + int(stat["prev_3d_ret"] < -0.01)
        + int(range_ratio > 1.3)
        + (int(rel_vol > 1.3) if use_rv else 0)
        + int(vwap_slope < -0.0003)
    )

    if neg_count >= 2 and de > 0.25:
        return "negative"
    elif de < 0.25 and abs(vwap_slope) < 0.0005:
        return "chop"
    return "trend"


def failure_score(bar: pd.Series) -> float:
    """Failure score for upside sweep (SHORT setup)."""
    rng = float(bar["high"]) - float(bar["low"]) + 1e-6
    clv = (2 * float(bar["close"]) - float(bar["high"]) - float(bar["low"])) / rng
    clv_score = -clv  # +1 if closed near bottom

    uw = float(bar["high"]) - max(float(bar["open"]), float(bar["close"]))
    lw = min(float(bar["open"]), float(bar["close"])) - float(bar["low"])
    wick_score = (uw - lw) / rng  # +1 if large upper wick

    body_signed = float(bar["close"]) - float(bar["open"])
    pressure = -(body_signed / rng)  # +1 if closed down (bear bar on upside sweep)

    return clv_score + wick_score + pressure


# ---------------------------------------------------------------------------
# IFR engine
# ---------------------------------------------------------------------------

OR_END_BAR    = 7
TIME_STOP     = 8
CUTOFF_H      = 14
CL_MAX        = 0.45
BODY_MAX      = 0.55
REL_VOL_MIN   = 1.10
FS_MIN        = 0.35
POINT_VALUE   = 50.0   # ES full contract
N_CONTRACTS   = 1
COST_RT       = 6.25   # ~0.25 ES ticks round-trip (2× $3.14 ~ $6)


def run_ifr_es(bars_5m: pd.DataFrame, stats_by_date: dict) -> list:
    all_trades = []
    session_dates = sorted(set(bars_5m.index.date))

    for date in session_dates:
        sess = bars_5m[bars_5m.index.date == date].copy()
        if len(sess) < OR_END_BAR + 3:
            continue
        stat = stats_by_date.get(date)
        if stat is None or stat["atr_20d"] is None or stat["atr_20d"] <= 0:
            continue

        regime = classify_regime(sess, stat)
        if regime == "trend":
            continue

        atr_20d  = stat["atr_20d"]
        bar_atr  = atr_20d / 13.0
        prev_hi  = stat["prev_sess_high"]
        prev_lo  = stat["prev_sess_low"]

        # Running state
        vwap_num, vwap_den = 0.0, 0.0
        vol_hist: list = []
        bar_highs: list = []
        bar_lows: list  = []
        in_trade = False
        trade_meta: dict = {}
        pending: dict = {}
        confirm_left = 0

        for i_bar, (ts, bar) in enumerate(sess.iterrows()):
            h = float(bar["high"])
            l = float(bar["low"])
            c = float(bar["close"])
            o = float(bar["open"])
            v = float(bar["volume"])

            # Running VWAP
            typical   = (h + l + c) / 3.0
            vwap_num += typical * v
            vwap_den += v
            vwap = vwap_num / (vwap_den + 1e-8)

            # Relative volume
            vol_hist.append(v)
            vol_med = float(np.median(vol_hist[-20:])) if len(vol_hist) >= 5 else v
            rel_vol = v / (vol_med + 1e-8)

            # Rolling intraday extremes
            bar_highs.append(h)
            bar_lows.append(l)
            roll_high = max(bar_highs[max(0, i_bar - 19):i_bar]) if i_bar > 0 else h
            roll_low  = min(bar_lows[max(0, i_bar - 19):i_bar])  if i_bar > 0 else l

            # ----------------------------------------------------------------
            # Exit
            # ----------------------------------------------------------------
            if in_trade:
                m = trade_meta
                pnl = None
                reason = None
                force_end = (i_bar == len(sess) - 1)

                if h >= m["stop"]:
                    pnl = (m["entry"] - m["stop"]) * POINT_VALUE * N_CONTRACTS - COST_RT
                    reason = "stop"
                elif l <= m["tp"]:
                    pnl = (m["entry"] - m["tp"]) * POINT_VALUE * N_CONTRACTS - COST_RT
                    reason = "tp"
                elif (i_bar - m["entry_bar"]) >= TIME_STOP or force_end:
                    pnl = (m["entry"] - c) * POINT_VALUE * N_CONTRACTS - COST_RT
                    reason = "time" if not force_end else "session_end"

                if pnl is not None:
                    all_trades.append({
                        "date":        str(date),
                        "regime":      regime,
                        "setup":       m["setup"],
                        "pnl":         round(pnl, 2),
                        "exit_reason": reason,
                        "entry":       m["entry"],
                        "stop":        m["stop"],
                        "tp":          m["tp"],
                    })
                    in_trade = False
                    pending = {}
                    confirm_left = 0
                continue

            if i_bar < OR_END_BAR:
                continue
            if ts.hour >= CUTOFF_H:
                continue

            # ----------------------------------------------------------------
            # Confirmation check
            # ----------------------------------------------------------------
            if pending and confirm_left > 0:
                confirm_left -= 1
                if c < pending["swept_level"]:  # price back inside range
                    entry_px = c
                    stop  = pending["sweep_extreme"] + 0.5 * bar_atr
                    risk  = stop - entry_px
                    # TP: vwap if >= 1R reward, else 1.5R
                    tp_fixed = entry_px - 1.5 * risk
                    tp = min(vwap, tp_fixed) if (vwap < entry_px and (entry_px - vwap) >= risk) else tp_fixed

                    in_trade = True
                    trade_meta = {
                        "entry":     entry_px,
                        "stop":      stop,
                        "tp":        tp,
                        "entry_bar": i_bar,
                        "setup":     pending["setup"],
                    }
                    pending = {}
                    confirm_left = 0
                elif confirm_left <= 0:
                    pending = {}
                continue

            # ----------------------------------------------------------------
            # Sweep detection
            # ----------------------------------------------------------------
            rng       = h - l + 1e-6
            body_frac = abs(c - o) / rng
            cl_val    = (c - l) / rng

            quality_ok = (cl_val < CL_MAX and body_frac < BODY_MAX and rel_vol > REL_VOL_MIN)

            if quality_ok and not pending:
                fs = failure_score(bar)
                if fs >= FS_MIN:
                    if prev_hi and h > prev_hi:
                        pending = {"swept_level": prev_hi, "sweep_extreme": h,
                                   "setup": "prev_sess_sweep"}
                        confirm_left = 3
                    elif i_bar >= OR_END_BAR and h > roll_high:
                        pending = {"swept_level": roll_high, "sweep_extreme": h,
                                   "setup": "roll_high_sweep"}
                        confirm_left = 3

    return all_trades


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print("Building session stats…")
stats_by_date = build_session_stats(df_es)

print("Running IFR on ES 2010-2025…")
trades = run_ifr_es(df_es, stats_by_date)
print(f"Total trades: {len(trades)}")


def compute_stats(trades):
    if not trades:
        return {}
    p = np.array([t["pnl"] for t in trades])
    wins = p[p > 0]
    losses = p[p <= 0]
    WR = len(wins) / len(p)

    by_day = {}
    for t in trades:
        by_day.setdefault(t["date"], 0.0)
        by_day[t["date"]] += t["pnl"]
    daily = np.array(list(by_day.values()))
    n_day = len(daily)
    sharpe = float((daily.mean() / (daily.std() + 1e-8)) * np.sqrt(252)) if n_day > 1 else 0.0
    cum = np.cumsum(p)
    roll_max = np.maximum.accumulate(cum)
    max_dd = float((cum - roll_max).min())
    expectancy = float(p.mean())

    return {
        "N": len(p),
        "n_day": n_day,
        "WR": round(WR, 4),
        "AvgW": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "AvgL": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "PF": round(float(wins.sum()) / (abs(float(losses.sum())) + 1e-8), 3),
        "PnL": round(float(p.sum()), 2),
        "Sharpe": round(sharpe, 3),
        "MaxDD": round(max_dd, 2),
        "Expectancy": round(expectancy, 2),
    }


def print_stats(label, s):
    if not s or s.get("N", 0) == 0:
        print(f"  {label:40s} | NO TRADES")
        return
    tpd = s["N"] / max(s["n_day"], 1)
    print(f"  {label:40s} | N={s['N']:5d} ({tpd:.2f}/day)"
          f" | WR={s['WR']:.1%} | AvgW=${s['AvgW']:.0f} | AvgL=${s['AvgL']:.0f}"
          f" | PnL=${s['PnL']:,.0f} | Sharpe={s['Sharpe']:.2f} | MaxDD=${s['MaxDD']:,.0f}"
          f" | E[trade]=${s['Expectancy']:.0f}")


# Overall
s_all = compute_stats(trades)
print("\n" + "=" * 80)
print("FULL PERIOD 2010-2025 — ES IFR Signal")
print("=" * 80)
print_stats("IFR overall", s_all)

# By regime
neg_trades  = [t for t in trades if t["regime"] == "negative"]
chop_trades = [t for t in trades if t["regime"] == "chop"]
print_stats("  Negative regime", compute_stats(neg_trades))
print_stats("  Chop regime",     compute_stats(chop_trades))

# By setup type
ps_trades  = [t for t in trades if t["setup"] == "prev_sess_sweep"]
rh_trades  = [t for t in trades if t["setup"] == "roll_high_sweep"]
print_stats("  prev_sess_sweep", compute_stats(ps_trades))
print_stats("  roll_high_sweep", compute_stats(rh_trades))

# Exit distribution
exit_dist = {}
for t in trades:
    exit_dist[t["exit_reason"]] = exit_dist.get(t["exit_reason"], 0) + 1
print(f"\n  Exit distribution: {exit_dist}")

# Year-by-year breakdown
print("\n" + "=" * 80)
print("YEAR-BY-YEAR BREAKDOWN")
print("=" * 80)
years = sorted(set(t["date"][:4] for t in trades))
yearly_results = {}
for yr in years:
    yr_trades = [t for t in trades if t["date"].startswith(yr)]
    s = compute_stats(yr_trades)
    yearly_results[yr] = s
    print_stats(f"  {yr}", s)

# Market regime analysis — 2008-2009 crisis, 2020 covid, 2022 bear
print("\n" + "=" * 80)
print("PERIOD BREAKDOWN")
print("=" * 80)
periods = {
    "2010-2012 (recovery)":     [t for t in trades if "2010" <= t["date"][:4] <= "2012"],
    "2013-2017 (bull)":         [t for t in trades if "2013" <= t["date"][:4] <= "2017"],
    "2018-2019 (vol spike)":    [t for t in trades if "2018" <= t["date"][:4] <= "2019"],
    "2020 (COVID)":             [t for t in trades if t["date"].startswith("2020")],
    "2021 (bull)":              [t for t in trades if t["date"].startswith("2021")],
    "2022 (bear/vol)":          [t for t in trades if t["date"].startswith("2022")],
    "2023-2024 (recovery)":     [t for t in trades if "2023" <= t["date"][:4] <= "2024"],
    "2025 (latest)":            [t for t in trades if t["date"].startswith("2025")],
}
for name, subset in periods.items():
    print_stats(f"  {name}", compute_stats(subset))

# Acceptance gate
print("\n" + "=" * 80)
print("ACCEPTANCE GATE")
print("=" * 80)

def chk(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}]  {name}" + (f"  ({detail})" if detail else ""))

s = s_all
chk("N > 200 trades",             s.get("N",0) > 200,      f"N={s.get('N',0)}")
chk("WR > 50%",                   s.get("WR",0) > 0.50,    f"WR={s.get('WR',0):.1%}")
chk("Sharpe > 1.5",               s.get("Sharpe",0) > 1.5, f"Sharpe={s.get('Sharpe',0):.2f}")
chk("E[trade] > $0",              s.get("Expectancy",0) > 0, f"E=${s.get('Expectancy',0):.0f}")
chk("Profitable in 2022 bear",    compute_stats(periods["2022 (bear/vol)"]).get("PnL",0) > 0,
    f"PnL=${compute_stats(periods['2022 (bear/vol)']).get('PnL',0):,.0f}")
chk("Profitable in 2020 COVID",   compute_stats(periods["2020 (COVID)"]).get("PnL",0) > 0,
    f"PnL=${compute_stats(periods['2020 (COVID)']).get('PnL',0):,.0f}")

n_passed = sum([
    s.get("N",0) > 200,
    s.get("WR",0) > 0.50,
    s.get("Sharpe",0) > 1.5,
    s.get("Expectancy",0) > 0,
    compute_stats(periods["2022 (bear/vol)"]).get("PnL",0) > 0,
    compute_stats(periods["2020 (COVID)"]).get("PnL",0) > 0,
])

if n_passed >= 5:
    verdict = "STRONG — deploy IFR alongside ORB"
elif n_passed >= 4:
    verdict = "MODERATE — review weak criteria, reduce size"
elif n_passed >= 2:
    verdict = "WEAK — do not deploy, needs redesign"
else:
    verdict = "KILL"

print(f"\n  Passed {n_passed}/6 criteria  →  {verdict}")

# Save
out = {
    "overall": s_all,
    "by_regime": {
        "negative": compute_stats(neg_trades),
        "chop":     compute_stats(chop_trades),
    },
    "by_setup": {
        "prev_sess_sweep": compute_stats(ps_trades),
        "roll_high_sweep": compute_stats(rh_trades),
    },
    "by_year": yearly_results,
    "exit_distribution": exit_dist,
    "verdict": verdict,
    "criteria_passed": int(n_passed),
}
out_path = ROOT / "rule_based_v1/diagnostics/ifr_longterm_results.json"
out_path.write_text(json.dumps(out, indent=2, default=str))
print(f"\nSaved → {out_path}")
