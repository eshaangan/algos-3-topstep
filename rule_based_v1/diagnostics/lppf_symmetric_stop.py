"""LPPF Symmetric Stop Sweep
===========================
Fixes the exit asymmetry found in lppf_backtest.py:
  Current: SL = sl_k × sqrt(P_entry_pp)  → avg -$15 (Kalman variance-based, too wide)
  Fix:     SL = sl_rr × |p_entry|         → proportional to detected pressure magnitude

Since p_entry passed monotonicity (B3 WR=48%), tying the SL to p_entry is the
natural choice.  TP fires when pressure decays (1-tp_frac)×|p_entry|.
For symmetric R:R (sl_rr = 1-tp_frac):  breakeven WR = 50%.

Datasets:
  Training : MES 1-min RTH  Jun–Dec 2025
  OOS      : MES 1-min RTH  Jan–Feb 2026
  Transfer : MNQ 1-min RTH  Jan–Mar 2026
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
for p in [str(ROOT), str(ROOT / "rule_based_v1")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
with pd.HDFStore(str(ROOT / "data/processed/mes_1m_bars_cache.h5"), "r") as s:
    _raw = s["/bars_1m"].set_index("timestamp")
_raw.index = pd.to_datetime(_raw.index, utc=True).tz_convert("US/Eastern")
_raw = _raw.sort_index()

with pd.HDFStore(str(ROOT / "data/processed/jan_feb_2026_oos_test_1m.h5"), "r") as s:
    _oos = s["/bars_1min"].copy()
_oos.index = pd.to_datetime(_oos.index, utc=True).tz_convert("US/Eastern")

with pd.HDFStore(str(ROOT / "data/processed/mnq_2026ytd_1min.h5"), "r") as s:
    _mnq = s["/bars_1min"].copy()


def rth(df: pd.DataFrame) -> pd.DataFrame:
    m = ((df.index.hour == 9) & (df.index.minute >= 30)) | \
        ((df.index.hour > 9) & (df.index.hour < 16))
    return df[m].copy()


train = rth(_raw)
oos   = rth(_oos)
mnq   = _mnq.copy()

print(f"Train: {len(train):,} bars  {train.index[0].date()} – {train.index[-1].date()}")
print(f"OOS:   {len(oos):,} bars  {oos.index[0].date()} – {oos.index[-1].date()}")
print(f"MNQ:   {len(mnq):,} bars  {mnq.index[0].date()} – {mnq.index[-1].date()}")

# ---------------------------------------------------------------------------
# Pre-compute bar signals (same as lppf_backtest.py)
# ---------------------------------------------------------------------------

def precompute(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars[["open", "high", "low", "close"]].copy()
    df.columns = ["O", "H", "L", "C"]
    rng = df["H"] - df["L"]
    df["sigma_bar"] = rng.rolling(20, min_periods=1).mean()
    df["med60"]     = rng.rolling(60, min_periods=1).median()
    rng_s = rng + 1e-6
    CLV      = 2 * (df["C"] - df["L"]) / rng_s - 1
    Body     = (df["C"] - df["O"]) / rng_s
    uw = df["H"] - df[["O","C"]].max(axis=1)
    lw = df[["O","C"]].min(axis=1) - df["L"]
    WickSkew = (uw - lw) / rng_s
    size_sc  = np.sqrt(rng / (df["med60"] + 1e-6))
    df["q"]        = np.tanh(1.2*CLV + 0.8*Body - 0.4*WickSkew) * size_sc
    df["y1"]       = df["C"]
    df["y2"]       = (df["H"] + df["L"] + 2*df["C"]) / 4
    wick_sq        = (uw**2 + lw**2) / 2
    df["wick_var"] = wick_sq.rolling(20, min_periods=1).mean()
    return df


train_pre = precompute(train)
oos_pre   = precompute(oos)
mnq_pre   = precompute(mnq)

# ---------------------------------------------------------------------------
# Kalman update (identical to lppf_backtest.py)
# ---------------------------------------------------------------------------

def kalman_update(x_hat, P, phi, q_t, y_obs, sigma_bar, wick_var):
    F     = np.array([[1.0, 0.0], [0.0, phi]])
    B     = np.array([0.0, 1.0])
    Q     = np.diag([(0.15*sigma_bar)**2, (0.60*sigma_bar)**2])
    H_obs = np.array([[1.0, 1.0], [1.0, 0.5]])
    wv    = max(wick_var, (0.2*sigma_bar)**2)
    R     = np.diag([max(wv*1.0, (0.5*sigma_bar)**2),
                     max(wv*0.4, (0.25*sigma_bar)**2)])
    x_p   = F @ x_hat + B * q_t
    P_p   = F @ P @ F.T + Q
    S     = H_obs @ P_p @ H_obs.T + R
    K     = P_p @ H_obs.T @ np.linalg.inv(S)
    x_new = x_p + K @ (y_obs - H_obs @ x_p)
    P_new = (np.eye(2) - K @ H_obs) @ P_p
    Z_p   = x_new[1] / np.sqrt(max(P_new[1,1], 1e-8))
    return x_new, P_new, float(Z_p), float(x_new[1])

# ---------------------------------------------------------------------------
# Backtest engine — symmetric pressure-proportional stop
# ---------------------------------------------------------------------------

def run_lppf_sym(
    bars_pre: pd.DataFrame,
    phi: float = 0.80,
    z_star: float = 1.0,
    tp_frac: float = 0.80,   # TP when pressure decays (1-tp_frac)|p_entry|
    sl_rr: float = 1.0,      # SL dist = sl_rr × |p_entry|  (KEY CHANGE)
    sl_floor_ticks: int = 4, # minimum SL in ticks (avoid zero-stop)
    time_stop_bars: int = 6,
    decay_bars: int = 1,
    only_long: bool = False,
    only_short: bool = False,
    n_contracts: int = 1,
    point_value: float = 5.0,
    commission: float = 0.62,
    slippage_ticks: int = 1,
    tick_size: float = 0.25,
    daily_loss_floor: float = -400.0,
    max_session_losses: int = 3,
    high_pressure_only: bool = False,  # only trade top-tercile |p_entry|
) -> list[dict]:

    slip = slippage_ticks * tick_size
    cost = 2 * commission
    sl_floor = sl_floor_ticks * tick_size

    trades: list[dict] = []
    dates = sorted(set(bars_pre.index.date))

    # collect |p_entry| values to compute tercile threshold if needed
    # Two-pass: first estimate tercile from a dry run, then trade.
    # For simplicity: compute rolling 100-trade |p_entry| median online.
    p_entry_history: list[float] = []

    for day in dates:
        day_bars = bars_pre[bars_pre.index.date == day]
        if len(day_bars) < 10:
            continue

        sb0      = float(day_bars["sigma_bar"].iloc[0])
        x_hat    = np.array([float(day_bars["O"].iloc[0]), 0.0])
        P        = np.diag([sb0**2 * 4, sb0**2 * 2])

        session_losses = 0
        session_pnl    = 0.0
        in_trade       = False
        direction      = 0
        entry_price    = 0.0
        p_entry_val    = 0.0
        p_entry_pp     = 0.0
        Z_entry_val    = 0.0
        bars_held      = 0
        p_hist         = [0.0]
        D_hist: list[float] = []

        for bar_i, (ts, row) in enumerate(day_bars.iterrows()):
            sb   = float(row["sigma_bar"])
            wv   = float(row["wick_var"])
            q_t  = float(row["q"])
            yobs = np.array([row["y1"], row["y2"]])
            H, L, C = float(row["H"]), float(row["L"]), float(row["C"])

            x_hat, P, Z_p, p_f = kalman_update(x_hat, P, phi, q_t, yobs, sb, wv)
            D_t = p_f - p_hist[-1]
            p_hist.append(p_f)
            D_hist.append(D_t)

            h, m_ = ts.hour, ts.minute
            sess_close = (h == 15 and m_ >= 55)

            # ---- EXIT ----
            if in_trade:
                bars_held += 1
                ep    = entry_price
                # KEY CHANGE: SL proportional to |p_entry|
                sl_d  = max(sl_rr * abs(p_entry_val), sl_floor)
                tp_d  = (1.0 - tp_frac) * abs(p_entry_val)  # price gain at TP
                tp_p_thresh = -tp_frac * abs(p_entry_val) if direction == 1 \
                              else  tp_frac * abs(p_entry_val)

                exit_p, reason = None, None

                if direction == 1:   # LONG
                    if L <= ep - sl_d:
                        exit_p, reason = ep - sl_d, "SL"
                    elif p_f >= 0 or p_f >= tp_p_thresh:
                        exit_p, reason = C - slip, "TP"
                    elif bars_held >= time_stop_bars:
                        exit_p, reason = C, "time_stop"
                    elif sess_close:
                        exit_p, reason = C, "session_close"
                else:                # SHORT
                    if H >= ep + sl_d:
                        exit_p, reason = ep + sl_d, "SL"
                    elif p_f <= 0 or p_f <= tp_p_thresh:
                        exit_p, reason = C + slip, "TP"
                    elif bars_held >= time_stop_bars:
                        exit_p, reason = C, "time_stop"
                    elif sess_close:
                        exit_p, reason = C, "session_close"

                if exit_p is not None:
                    pnl = direction*(exit_p - entry_price)*n_contracts*point_value - cost*n_contracts
                    session_pnl += pnl
                    if pnl < 0:
                        session_losses += 1
                    trades.append({
                        "date": str(ts.date()),
                        "direction": direction,
                        "entry": round(entry_price, 4),
                        "exit":  round(exit_p, 4),
                        "pnl":   round(pnl, 4),
                        "reason": reason,
                        "Z_entry":   round(Z_entry_val, 4),
                        "p_entry":   round(p_entry_val, 4),
                        "bars_held": bars_held,
                        "sl_dist":   round(sl_d, 4),
                        "tp_dist":   round(tp_d, 4),
                    })
                    in_trade = False

            # ---- ENTRY ----
            if (not in_trade
                and bar_i >= 5
                and not (h == 9 and m_ < 35)
                and not (h >= 15 and m_ >= 40)
                and session_losses < max_session_losses
                and session_pnl > daily_loss_floor):

                # decay confirmation
                if len(D_hist) >= decay_bars:
                    dec_ok_long  = all(d > 0 for d in D_hist[-decay_bars:])
                    dec_ok_short = all(d < 0 for d in D_hist[-decay_bars:])
                else:
                    dec_ok_long = dec_ok_short = False

                # high-pressure filter: require |p_f| above recent median
                if high_pressure_only and len(p_entry_history) >= 20:
                    thresh = np.percentile([abs(x) for x in p_entry_history[-100:]], 67)
                    hp_ok = abs(p_f) >= thresh
                else:
                    hp_ok = True

                if not only_short and Z_p < -z_star and dec_ok_long and hp_ok:
                    direction   = 1
                    entry_price = C + slip
                    p_entry_val = p_f
                    p_entry_pp  = P[1,1]
                    Z_entry_val = Z_p
                    bars_held   = 0
                    in_trade    = True
                    p_entry_history.append(p_f)

                elif not only_long and Z_p > z_star and dec_ok_short and hp_ok:
                    direction   = -1
                    entry_price = C - slip
                    p_entry_val = p_f
                    p_entry_pp  = P[1,1]
                    Z_entry_val = Z_p
                    bars_held   = 0
                    in_trade    = True
                    p_entry_history.append(p_f)

    return trades


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def stats(trades: list[dict], label: str = "") -> dict:
    if not trades:
        return {"label": label, "N": 0, "WR": 0.0, "sharpe": 0.0,
                "total_pnl": 0.0, "n_per_day": 0.0, "max_dd": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "pf": 0.0}
    pnls  = [t["pnl"] for t in trades]
    wins  = [p for p in pnls if p > 0]
    losses= [p for p in pnls if p <= 0]
    daily: dict[str, float] = {}
    for t in trades:
        daily[t["date"]] = daily.get(t["date"], 0.0) + t["pnl"]
    active = [v for v in daily.values() if v != 0]
    mu = np.mean(active) if active else 0.0
    sd = np.std(active, ddof=1) if len(active) > 1 else 1e-9
    sharpe = mu / sd * np.sqrt(252) if sd > 1e-9 else 0.0
    eq = np.cumsum(pnls)
    pk = np.maximum.accumulate(eq)
    maxdd = float((eq - pk).min()) if len(eq) else 0.0
    n_days = len(set(t["date"] for t in trades))
    return {
        "label":     label,
        "N":         len(pnls),
        "n_days":    n_days,
        "n_per_day": round(len(pnls) / max(n_days, 1), 2),
        "WR":        round(100 * len(wins) / max(len(pnls),1), 1),
        "avg_win":   round(np.mean(wins),  2) if wins   else 0.0,
        "avg_loss":  round(np.mean(losses),2) if losses else 0.0,
        "pf":        round(sum(wins)/abs(sum(losses)),2) if losses else 999.0,
        "total_pnl": round(sum(pnls), 2),
        "sharpe":    round(sharpe, 3),
        "max_dd":    round(maxdd, 2),
        "daily":     daily,
    }


def print_stats(s: dict) -> None:
    print(f"  [{s['label']}]")
    print(f"    N={s['N']}  n/day={s['n_per_day']}  WR={s['WR']}%  "
          f"AvgW=${s['avg_win']:+.2f}  AvgL=${s['avg_loss']:+.2f}  PF={s['pf']:.2f}")
    print(f"    TotalPnL=${s['total_pnl']:+,.2f}  Sharpe={s['sharpe']:.3f}  MaxDD=${s['max_dd']:,.2f}")


def monotonicity(trades: list[dict], field: str, label: str) -> bool:
    vals = [abs(t[field]) for t in trades]
    p33, p67 = np.percentile(vals, 33), np.percentile(vals, 67)
    b1 = [t for t in trades if abs(t[field]) < p33]
    b2 = [t for t in trades if p33 <= abs(t[field]) < p67]
    b3 = [t for t in trades if abs(t[field]) >= p67]
    rows = []
    for bn, bt in [("B1_low",b1),("B2_mid",b2),("B3_high",b3)]:
        if not bt:
            rows.append((bn,0,0,0))
            continue
        wr  = 100*sum(1 for t in bt if t["pnl"]>0)/len(bt)
        avg = np.mean([t["pnl"] for t in bt])
        rows.append((bn, len(bt), round(wr,1), round(avg,2)))
    wrs = [r[2] for r in rows]
    passes = wrs[0] <= wrs[1] <= wrs[2] and wrs[0] < wrs[2]
    print(f"\n  {label} monotonicity:")
    for bn, n, wr, avg in rows:
        print(f"    {bn}: N={n}  WR={wr}%  avg_pnl=${avg:+.2f}")
    print(f"  → {'PASS ✓' if passes else 'FAIL ✗'}")
    return passes


# ============================================================================
# SECTION 1 — Symmetric Stop Sweep (training data)
# ============================================================================
print("\n" + "="*72)
print("  SECTION 1: SYMMETRIC STOP SWEEP  (training, phi=0.80, z*=1.0)")
print("="*72)

GRID_SL_RR  = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
GRID_TP     = [0.5, 0.6, 0.7, 0.80, 0.90]
PHI, ZSTAR  = 0.80, 1.0

rows = []
total = len(GRID_SL_RR) * len(GRID_TP)
done  = 0
for sl_rr in GRID_SL_RR:
    for tp_frac in GRID_TP:
        t = run_lppf_sym(train_pre, phi=PHI, z_star=ZSTAR,
                         tp_frac=tp_frac, sl_rr=sl_rr)
        s = stats(t, f"sl{sl_rr}_tp{tp_frac}")
        rows.append((sl_rr, tp_frac, s["N"], s["n_per_day"],
                     s["WR"], s["avg_win"], s["avg_loss"],
                     s["pf"], s["total_pnl"], s["sharpe"], s["max_dd"]))
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{total} done...")

rows.sort(key=lambda r: r[9], reverse=True)
print(f"\n  {'sl_rr':>6} {'tp_fr':>6} {'N':>5} {'N/d':>5} "
      f"{'WR':>6} {'AvgW':>7} {'AvgL':>7} {'PF':>5} "
      f"{'PnL':>9} {'Sharpe':>8} {'MaxDD':>9}")
print("  " + "-"*82)
for r in rows[:20]:
    sl_rr,tp_f,N,npd,wr,aw,al,pf,pnl,sh,dd = r
    flag = " ◄" if sh > 0 else ""
    print(f"  {sl_rr:>6.2f} {tp_f:>6.2f} {N:>5} {npd:>5.2f} "
          f"{wr:>5.1f}% ${aw:>6.2f} ${al:>6.2f} {pf:>5.2f} "
          f"${pnl:>8,.0f} {sh:>8.3f} ${dd:>8,.0f}{flag}")

# Best config
best = rows[0]
best_sl, best_tp = best[0], best[1]
print(f"\n  Best: sl_rr={best_sl}, tp_frac={best_tp}  "
      f"Sharpe={best[9]:.3f}  WR={best[4]}%  PnL=${best[8]:+,.0f}")

# ============================================================================
# SECTION 2 — Best Config Deep Analysis
# ============================================================================
print("\n" + "="*72)
print(f"  SECTION 2: DEEP ANALYSIS  (best: sl_rr={best_sl}, tp_frac={best_tp})")
print("="*72)

best_trades = run_lppf_sym(train_pre, phi=PHI, z_star=ZSTAR,
                            tp_frac=best_tp, sl_rr=best_sl)
bs = stats(best_trades, f"Training best (sl_rr={best_sl}, tp={best_tp})")
print_stats(bs)

# exit reason breakdown
reasons: dict[str,list] = {}
for t in best_trades:
    reasons.setdefault(t["reason"],[]).append(t["pnl"])
print("\n  Exit reasons:")
for r, pnls in sorted(reasons.items()):
    wr = 100*sum(1 for p in pnls if p>0)/len(pnls)
    print(f"    {r:<14} N={len(pnls):>4}  WR={wr:>5.1f}%  avg=${np.mean(pnls):+.2f}")

# avg SL and TP distances
sl_dists = [t["sl_dist"] for t in best_trades]
tp_dists = [t["tp_dist"] for t in best_trades]
print(f"\n  Avg SL dist = ${np.mean(sl_dists):.2f}  (vs old: ~$15)")
print(f"  Avg TP dist = ${np.mean(tp_dists):.2f}")
print(f"  R:R ratio   = {np.mean(sl_dists)/max(np.mean(tp_dists),0.01):.2f}:1")

# monotonicity
m1 = monotonicity(best_trades, "p_entry", "|p_entry|")
m2 = monotonicity(best_trades, "Z_entry", "|Z_entry|")

# ============================================================================
# SECTION 3 — Direction Split
# ============================================================================
print("\n" + "="*72)
print("  SECTION 3: DIRECTION SPLIT")
print("="*72)
for label, kw in [("LONG only", {"only_long":True}),
                  ("SHORT only",{"only_short":True})]:
    t = run_lppf_sym(train_pre, phi=PHI, z_star=ZSTAR,
                      tp_frac=best_tp, sl_rr=best_sl, **kw)
    s = stats(t, label)
    print_stats(s)

# ============================================================================
# SECTION 4 — High-Pressure-Only Filter (B3 trades, WR=48.2% in original)
# ============================================================================
print("\n" + "="*72)
print("  SECTION 4: HIGH-PRESSURE-ONLY FILTER (top 33% |p_entry|)")
print("="*72)
t_hp = run_lppf_sym(train_pre, phi=PHI, z_star=ZSTAR,
                     tp_frac=best_tp, sl_rr=best_sl, high_pressure_only=True)
s_hp = stats(t_hp, "High-pressure only (B3)")
print_stats(s_hp)
monotonicity(t_hp, "p_entry", "|p_entry| (HP only)")

# ============================================================================
# SECTION 5 — OOS Validation (MES Jan-Feb 2026)
# ============================================================================
print("\n" + "="*72)
print("  SECTION 5: OOS VALIDATION  (MES Jan-Feb 2026)")
print("="*72)
t_oos = run_lppf_sym(oos_pre, phi=PHI, z_star=ZSTAR,
                      tp_frac=best_tp, sl_rr=best_sl)
s_oos = stats(t_oos, "OOS MES Jan-Feb 2026")
print_stats(s_oos)
print(f"\n  Train WR={bs['WR']}%  OOS WR={s_oos['WR']}%  "
      f"gap={abs(bs['WR']-s_oos['WR']):.1f}pp  "
      f"{'GOOD' if abs(bs['WR']-s_oos['WR'])<=8 else 'BAD'}")

# ============================================================================
# SECTION 6 — MNQ Transfer (Jan-Mar 2026)
# ============================================================================
print("\n" + "="*72)
print("  SECTION 6: MNQ TRANSFER  (Jan-Mar 2026, point_value=2.0)")
print("="*72)
t_mnq = run_lppf_sym(mnq_pre, phi=PHI, z_star=ZSTAR,
                      tp_frac=best_tp, sl_rr=best_sl,
                      point_value=2.0, tick_size=0.25)
s_mnq = stats(t_mnq, "MNQ Jan-Mar 2026")
print_stats(s_mnq)
m_mnq = monotonicity(t_mnq, "p_entry", "|p_entry| MNQ")

# ============================================================================
# SECTION 7 — phi sensitivity (best exit params)
# ============================================================================
print("\n" + "="*72)
print("  SECTION 7: phi SENSITIVITY (pressure decay half-life)")
print("="*72)
print(f"  {'phi':>6}  {'N':>5}  {'WR':>6}  {'Sharpe':>8}  {'PnL':>9}")
print("  " + "-"*40)
for phi_val in [0.35, 0.50, 0.65, 0.80]:
    t = run_lppf_sym(train_pre, phi=phi_val, z_star=ZSTAR,
                      tp_frac=best_tp, sl_rr=best_sl)
    s = stats(t)
    flag = " ◄" if phi_val == 0.80 else ""
    print(f"  {phi_val:>6.2f}  {s['N']:>5}  {s['WR']:>5.1f}%  "
          f"{s['sharpe']:>8.3f}  ${s['total_pnl']:>8,.0f}{flag}")

# ============================================================================
# SECTION 8 — Best config per-day stats + Monte Carlo
# ============================================================================
print("\n" + "="*72)
print("  SECTION 8: PER-DAY STATS + COMBINE MONTE CARLO")
print("="*72)

daily_pnls = list(bs["daily"].values())
active_days = [d for d in daily_pnls if d != 0]
print(f"\n  Active trading days: {len(active_days)}")
print(f"  Daily PnL: min=${min(active_days):,.0f}  "
      f"mean=${np.mean(active_days):,.0f}  "
      f"max=${max(active_days):,.0f}  "
      f"std=${np.std(active_days):,.0f}")

# per-session trade count
from collections import Counter
daily_counts = Counter(t["date"] for t in best_trades)
cnts = list(daily_counts.values())
print(f"  Trades/day: min={min(cnts)}  p25={int(np.percentile(cnts,25))}  "
      f"median={int(np.median(cnts))}  p75={int(np.percentile(cnts,75))}  max={max(cnts)}")

# best/worst sessions
sorted_days = sorted(bs["daily"].items(), key=lambda x: x[1])
print(f"\n  Worst 5 sessions:")
for d, p in sorted_days[:5]:
    print(f"    {d}  ${p:+,.2f}")
print(f"  Best 5 sessions:")
for d, p in sorted_days[-5:]:
    print(f"    {d}  ${p:+,.2f}")

# Monte Carlo
print(f"\n  Monte Carlo (10k paths × 60 days):")
START, TRAIL_DD, DAILY_LIM, TARGET = 50_000, 2_000, 1_000, 3_000
rng_mc = np.random.default_rng(42)
sample = np.array(active_days)
paths  = rng_mc.choice(sample, size=(10_000, 60), replace=True)

n_pass = n_bust_dd = n_bust_daily = n_timeout = 0
days_to_pass: list[int] = []

for path in paths:
    equity = START
    peak   = START
    result = "timeout"
    for day_i, dpnl in enumerate(path):
        if abs(dpnl) > DAILY_LIM:
            result = "bust_daily"; break
        equity += dpnl
        peak    = max(peak, equity)
        if peak - equity >= TRAIL_DD:
            result = "bust_dd"; break
        if equity - START >= TARGET:
            result = "pass"
            days_to_pass.append(day_i + 1)
            break

    if result == "pass":         n_pass      += 1
    elif result == "bust_dd":    n_bust_dd   += 1
    elif result == "bust_daily": n_bust_daily+= 1
    else:                        n_timeout   += 1

print(f"    P(pass)       = {n_pass/100:.1f}%")
print(f"    P(bust_dd)    = {n_bust_dd/100:.1f}%")
print(f"    P(bust_daily) = {n_bust_daily/100:.1f}%")
print(f"    P(timeout)    = {n_timeout/100:.1f}%")
if days_to_pass:
    print(f"    Median days   = {np.median(days_to_pass):.0f}")

# ============================================================================
# VERDICT
# ============================================================================
print("\n" + "="*72)
print("  VERDICT")
print("="*72)
wr_gap = abs(bs["WR"] - s_oos["WR"])
verdict_lines = [
    f"  Best sl_rr    : {best_sl}",
    f"  Best tp_frac  : {best_tp}",
    f"  Training WR   : {bs['WR']}%  (need ≥52% PASS, ≥48% MARGINAL)",
    f"  Training Sharpe: {bs['sharpe']:.3f}  (need >0)",
    f"  OOS WR gap    : {wr_gap:.1f}pp  (need ≤8pp)",
    f"  p_entry mono  : {'PASS' if m1 else 'FAIL'}",
    f"  Z_entry mono  : {'PASS' if m2 else 'FAIL'}",
    f"  MNQ transfer  : {'PASS' if m_mnq else 'FAIL'}  WR={s_mnq['WR']}%",
    f"  P(pass combine): {n_pass/100:.1f}%",
]
for l in verdict_lines:
    print(l)

if (bs["sharpe"] > 0 and bs["WR"] >= 52
        and wr_gap <= 8 and m1):
    print("\n  >>> VERDICT: PASS <<<")
elif (bs["sharpe"] > 0 and bs["WR"] >= 48 and m1):
    print("\n  >>> VERDICT: MARGINAL — further validation needed <<<")
elif bs["sharpe"] > -1.0 and bs["WR"] >= 45:
    print("\n  >>> VERDICT: WEAK — monitor, do not deploy <<<")
else:
    print("\n  >>> VERDICT: KILL <<<")

# Save
out = {
    "fix": "sl_rr × |p_entry| replaces sl_k × sqrt(P_pp)",
    "best_sl_rr":  best_sl,
    "best_tp_frac": best_tp,
    "train": {k: v for k, v in bs.items() if k != "daily"},
    "oos":   {k: v for k, v in s_oos.items() if k != "daily"},
    "mnq":   {k: v for k, v in s_mnq.items() if k != "daily"},
    "hp_filter": {k: v for k, v in s_hp.items() if k != "daily"},
    "monte_carlo": {"p_pass": n_pass/100, "p_bust_dd": n_bust_dd/100,
                    "median_days": float(np.median(days_to_pass)) if days_to_pass else None},
    "p_entry_mono": m1, "Z_entry_mono": m2, "mnq_mono": m_mnq,
}
out_path = ROOT / "rule_based_v1/diagnostics/lppf_symmetric_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\n  Saved → {out_path.relative_to(ROOT)}")
