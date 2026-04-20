"""
COMET Horizon Extension Test
Tests whether 1-min OHLC signals carry predictive power at longer holding horizons.

Two approaches:
  A) 1-min COMET signal, extend h from 2 → 30 bars (2-30 min)
  B) Resample to 5-min bars, run COMET with h = 1-8 bars (5-40 min)

Kill criterion: if WR stays below breakeven at all h, resolution mismatch is confirmed.
Pass criterion: some h shows WR >= breakeven AND positive Sharpe on OOS.
"""
import json
import warnings
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
with pd.HDFStore(str(ROOT / "data/processed/mes_1m_bars_cache.h5"), "r") as _s:
    _df_tr = _s["/bars_1m"].set_index("timestamp")
_df_tr.index = pd.to_datetime(_df_tr.index, utc=True).tz_convert("US/Eastern")
_df_tr = _df_tr.sort_index()

with pd.HDFStore(str(ROOT / "data/processed/jan_feb_2026_oos_test_1m.h5"), "r") as _s:
    _df_oo = _s["/bars_1min"].copy()
_df_oo.index = pd.to_datetime(_df_oo.index, utc=True).tz_convert("US/Eastern")
_df_oo = _df_oo.sort_index()

with pd.HDFStore(str(ROOT / "data/processed/mnq_2026ytd_1min.h5"), "r") as _s:
    _df_mn = _s["/bars_1min"].copy()
_df_mn.index.name = "timestamp"


def rth(df):
    mask = ((df.index.hour == 9) & (df.index.minute >= 30)) | ((df.index.hour > 9) & (df.index.hour < 16))
    return df[mask].copy()


tr1m = rth(_df_tr)
oo1m = rth(_df_oo)
mn1m = _df_mn.copy()

print(f"1-min train: {len(tr1m):,}  {tr1m.index[0].date()} – {tr1m.index[-1].date()}")
print(f"1-min OOS:   {len(oo1m):,}  {oo1m.index[0].date()} – {oo1m.index[-1].date()}")
print(f"1-min MNQ:   {len(mn1m):,}  {mn1m.index[0].date()} – {mn1m.index[-1].date()}")

# ─────────────────────────────────────────────────────────────────────────────
# 5-min resampling (session-aware: no cross-session bars)
# ─────────────────────────────────────────────────────────────────────────────

def resample_5m(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min RTH bars to 5-min. Each bucket stays within its session date."""
    df = bars_1m[["open", "high", "low", "close"]].copy()
    df["_date"]   = df.index.date
    df["_bucket"] = df.index.floor("5min")
    g = df.groupby(["_date", "_bucket"]).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),    close=("close", "last"),
    )
    # flatten to single timestamp index
    g = g.reset_index(level="_date", drop=True)
    g.index.name = "timestamp"
    return g


tr5m = resample_5m(tr1m)
oo5m = resample_5m(oo1m)
mn5m = resample_5m(mn1m)

print(f"5-min train: {len(tr5m):,}  {tr5m.index[0].date()} – {tr5m.index[-1].date()}")
print(f"5-min OOS:   {len(oo5m):,}  {oo5m.index[0].date()} – {oo5m.index[-1].date()}")

# ─────────────────────────────────────────────────────────────────────────────
# Feature pre-computation (same COMET architecture, tf-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
N_EXPERTS = 4
N_FEAT    = 7


def precompute(bars: pd.DataFrame, mu_tod=None, sig_tod=None, tod_slots=390):
    O = bars["open"].values.astype(float)
    H = bars["high"].values.astype(float)
    L = bars["low"].values.astype(float)
    C = bars["close"].values.astype(float)
    n = len(bars)

    # minute-of-day [0, tod_slots-1]
    tod = np.clip(bars.index.hour * 60 + bars.index.minute - 9 * 60 - 30, 0, tod_slots - 1)

    # day-boundary
    dates      = bars.index.date
    is_new_day = np.array([False] + [dates[i] != dates[i - 1] for i in range(1, n)])

    r = np.zeros(n)
    r[1:] = C[1:] / C[:-1] - 1.0
    r[is_new_day] = 0.0

    if mu_tod is None:
        mu_tod  = np.zeros(tod_slots)
        sig_tod = np.full(tod_slots, 1e-4)
        for slot in range(tod_slots):
            mask_s = tod == slot
            if mask_s.sum() > 1:
                mu_tod[slot]  = r[mask_s].mean()
                sig_tod[slot] = max(r[mask_s].std(), 1e-6)

    r_tilde = (r - mu_tod[tod]) / (sig_tod[tod] + 1e-8)
    r_tilde[is_new_day] = 0.0

    rng      = H - L
    rng_safe = rng + 1e-6
    CLV      = (2 * C - H - L) / rng_safe
    Body     = (C - O) / rng_safe
    uw       = H - np.maximum(O, C)
    lw       = np.minimum(O, C) - L
    WickSkew = (uw - lw) / rng_safe
    med60    = pd.Series(rng).rolling(60, min_periods=1).median().values
    rr       = rng / (med60 + 1e-6)

    # experts
    r_t1 = np.roll(r_tilde, 1); r_t1[is_new_day] = 0.0
    r_t2 = np.roll(r_tilde, 2); r_t2[is_new_day] = 0.0
    r_t3 = np.roll(r_tilde, 3); r_t3[is_new_day] = 0.0
    s1 = np.clip(-r_t1, -3, 3) / 3.0
    mom3 = r_t1 + 0.7 * r_t2 + 0.49 * r_t3
    s2 = np.sign(mom3) * (np.abs(mom3) > 0.05).astype(float)
    s3 = np.sign(CLV) * (rr > 1.2).astype(float)
    s4 = -np.sign(Body) * (np.abs(Body) > 0.5).astype(float)

    S = np.column_stack([s1, s2, s3, s4])

    tod_norm = tod / (tod_slots - 1)
    gsig     = sig_tod[sig_tod > 1e-5].mean() if (sig_tod > 1e-5).any() else 1e-4
    dv       = np.clip(sig_tod[tod] / (gsig + 1e-8), 0, 5)
    rrc      = np.clip(rr, 0, 5)

    X = np.column_stack([np.ones(n), tod_norm, dv, rrc, CLV, Body, WickSkew])

    return C, S, X, tod, mu_tod, sig_tod


# ─────────────────────────────────────────────────────────────────────────────
# COMET loop (session-exit safe: exit at session end if h would cross day)
# ─────────────────────────────────────────────────────────────────────────────

def run_comet(C, S, X, idx,
              point_value, cost_rt,
              h=2, tau_pct=0.80, eta=0.005, lam=0.002, warmup=200):
    n     = len(C)
    theta = np.zeros((N_EXPERTS, N_FEAT))

    dates      = idx.date
    abs_a_buf  = deque(maxlen=200)
    update_q   = deque()

    in_trade  = False; dir_ = 0; px_ = 0.0; bar_ = -1
    pnl_list  = []; recs = []; a_arr = np.zeros(n)

    for i in range(n):
        logits = theta @ X[i]
        logits -= logits.max()
        w = np.exp(logits); w /= (w.sum() + 1e-12)

        a_t      = float(w @ S[i])
        a_arr[i] = a_t
        abs_a_buf.append(abs(a_t))

        # delayed gate update
        while update_q and update_q[0][0] <= i - h:
            j, x_j, s_j, w_j = update_q.popleft()
            R_j = C[i] - C[j]
            pi_k = s_j * R_j; pi_bar = float(w_j @ pi_k)
            theta += eta * np.outer(pi_k - pi_bar, x_j)
            theta *= (1.0 - lam)
        update_q.append((i, X[i].copy(), S[i].copy(), w.copy()))

        ready  = len(abs_a_buf) >= warmup
        thresh = np.percentile(list(abs_a_buf), tau_pct * 100) if ready else 1e9

        # exit: on h-bar horizon OR end of session, whichever comes first
        if in_trade:
            session_end = (i == n - 1) or (dates[i + 1] != dates[bar_])
            time_exit   = (i - bar_) >= h
            if time_exit or session_end:
                pnl = dir_ * (C[i] - px_) * point_value - cost_rt
                pnl_list.append(pnl)
                recs.append({"date": str(dates[bar_]), "bar": bar_,
                             "dir": dir_, "a": round(float(a_arr[bar_]), 5),
                             "entry": px_, "exit": C[i], "pnl": round(pnl, 2)})
                in_trade = False

        # entry
        if not in_trade and ready and abs(a_t) > thresh:
            dir_ = int(np.sign(a_t)) if a_t != 0 else 1
            px_  = C[i]; bar_ = i; in_trade = True

    return pnl_list, recs, a_arr


def stats(recs):
    if not recs:
        return dict(N=0, n_day=0.0, WR=0.0, AvgW=0.0, AvgL=0.0, bkeven_wr=0.0,
                    PF=0.0, PnL=0.0, Sharpe=0.0, MaxDD=0.0)
    p = np.array([r["pnl"] for r in recs])
    n = len(p)
    wins   = p[p > 0]; losses = p[p <= 0]
    cum    = np.cumsum(p); dd = (cum - np.maximum.accumulate(cum)).min()
    pf     = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 9.99
    by_day = {}
    for r in recs:
        by_day.setdefault(r["date"], 0.0); by_day[r["date"]] += r["pnl"]
    daily = np.array(list(by_day.values()))
    sharpe = (daily.mean() / (daily.std() + 1e-8)) * np.sqrt(252) if len(daily) > 1 else 0.0
    n_day  = n / len(daily) if daily.size else 0.0
    avg_w  = wins.mean() if len(wins) else 0.0
    avg_l  = abs(losses.mean()) if len(losses) else 1e-6
    bkeven = avg_l / (avg_w + avg_l) if (avg_w + avg_l) > 0 else 0.5
    return dict(N=n, n_day=round(n_day, 1), WR=round(len(wins)/n, 3),
                AvgW=round(avg_w, 2), AvgL=round(-abs(losses.mean()) if len(losses) else 0, 2),
                bkeven_wr=round(bkeven, 3),
                PF=round(pf, 3), PnL=round(float(p.sum()), 2),
                Sharpe=round(sharpe, 3), MaxDD=round(float(dd), 2))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: 1-min COMET, extended hold horizon sweep
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SECTION A: 1-MIN COMET — HOLD HORIZON SWEEP  (training)")
print("=" * 72)

C1, S1, X1, tod1, mu1, sig1 = precompute(tr1m)
C1o, S1o, X1o = precompute(oo1m, mu1, sig1)[:3]
C1m, S1m, X1m = precompute(mn1m, mu1, sig1)[:3]

MES_PV = 5.0; MES_COST = 2.50
MNQ_PV = 2.0; MNQ_COST = 1.00

print(f"\n  {'h':>3}  {'N':>5}  {'n/d':>5}  {'WR':>6}  {'bkeven':>7}  "
      f"{'gap':>6}  {'Sharpe':>8}  {'PnL':>9}")
print("  " + "-" * 65)

h_sweep_1m = [1, 2, 3, 5, 7, 10, 15, 20, 30]
rows_1m = []
for h in h_sweep_1m:
    _, recs, _ = run_comet(C1, S1, X1, tr1m.index, MES_PV, MES_COST,
                           h=h, tau_pct=0.80, eta=0.005, lam=0.002, warmup=200)
    st = stats(recs)
    gap = st["WR"] - st["bkeven_wr"]
    rows_1m.append({"h": h, **st, "gap": round(gap, 3)})
    marker = " ◄" if st["Sharpe"] > 0 else ("  →" if gap > 0 else "")
    print(f"  {h:>3}  {st['N']:>5}  {st['n_day']:>5.1f}  {st['WR']:>6.1%}  "
          f"{st['bkeven_wr']:>7.1%}  {gap:>+6.1%}  {st['Sharpe']:>8.3f}  "
          f"${st['PnL']:>8,.0f}{marker}")

best_1m = max(rows_1m, key=lambda r: r["Sharpe"])
print(f"\n  Best 1-min: h={best_1m['h']}  Sharpe={best_1m['Sharpe']:.3f}  WR={best_1m['WR']:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: 5-min COMET — hold horizon sweep
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SECTION B: 5-MIN COMET — HOLD HORIZON SWEEP  (training)")
print("=" * 72)

# 5-min bars: tod slots = 78 (9:30..15:55 every 5 min)
C5, S5, X5, tod5, mu5, sig5 = precompute(tr5m, tod_slots=78)
C5o, S5o, X5o = precompute(oo5m, mu5, sig5, tod_slots=78)[:3]
C5m, S5m, X5m = precompute(mn5m, mu5, sig5, tod_slots=78)[:3]

print(f"  5-min train: {len(tr5m):,}  |  5-min OOS: {len(oo5m):,}")

print(f"\n  {'h':>3}  {'min':>5}  {'N':>5}  {'n/d':>5}  {'WR':>6}  {'bkeven':>7}  "
      f"{'gap':>6}  {'Sharpe':>8}  {'PnL':>9}")
print("  " + "-" * 72)

h_sweep_5m = [1, 2, 3, 4, 5, 6, 8, 10, 12]
rows_5m = []
for h in h_sweep_5m:
    _, recs, _ = run_comet(C5, S5, X5, tr5m.index, MES_PV, MES_COST,
                           h=h, tau_pct=0.80, eta=0.005, lam=0.002, warmup=40)
    st = stats(recs)
    gap = st["WR"] - st["bkeven_wr"]
    rows_5m.append({"h": h, **st, "gap": round(gap, 3)})
    marker = " ◄ POSITIVE" if st["Sharpe"] > 0 else ("  ← near" if gap > -0.02 else "")
    print(f"  {h:>3}  {h*5:>5}m  {st['N']:>5}  {st['n_day']:>5.1f}  {st['WR']:>6.1%}  "
          f"{st['bkeven_wr']:>7.1%}  {gap:>+6.1%}  {st['Sharpe']:>8.3f}  "
          f"${st['PnL']:>8,.0f}{marker}")

best_5m = max(rows_5m, key=lambda r: r["Sharpe"])
print(f"\n  Best 5-min: h={best_5m['h']} ({best_5m['h']*5}min)  "
      f"Sharpe={best_5m['Sharpe']:.3f}  WR={best_5m['WR']:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: 5-min COMET — eta/tau sweep at best h
# ─────────────────────────────────────────────────────────────────────────────
BH5 = best_5m["h"]
print("\n" + "=" * 72)
print(f"  SECTION C: 5-MIN COMET PARAM SWEEP  (h={BH5}, training)")
print("=" * 72)

print(f"\n  {'tau':>5}  {'eta':>6}  {'N':>5}  {'n/d':>5}  {'WR':>6}  {'bkeven':>7}  {'Sharpe':>8}  {'PnL':>9}")
print("  " + "-" * 68)

best_c = None
for tau in [0.65, 0.75, 0.80, 0.85, 0.90]:
    for eta in [0.002, 0.005, 0.010, 0.020]:
        _, recs, _ = run_comet(C5, S5, X5, tr5m.index, MES_PV, MES_COST,
                               h=BH5, tau_pct=tau, eta=eta, lam=0.002, warmup=40)
        st = stats(recs)
        gap = st["WR"] - st["bkeven_wr"]
        if best_c is None or st["Sharpe"] > best_c["Sharpe"]:
            best_c = {"tau": tau, "eta": eta, **st}
        mark = " ◄" if st["Sharpe"] > 0 else ""
        print(f"  {tau:>5.2f}  {eta:>6.3f}  {st['N']:>5}  {st['n_day']:>5.1f}  "
              f"{st['WR']:>6.1%}  {st['bkeven_wr']:>7.1%}  {st['Sharpe']:>8.3f}  "
              f"${st['PnL']:>8,.0f}{mark}")

print(f"\n  Best: tau={best_c['tau']}, eta={best_c['eta']}  Sharpe={best_c['Sharpe']:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: Deep analysis on best 5-min config
# ─────────────────────────────────────────────────────────────────────────────
BTAU = best_c["tau"]
BETA = best_c["eta"]

print("\n" + "=" * 72)
print(f"  SECTION D: DEEP ANALYSIS  (5-min, h={BH5}, tau={BTAU}, eta={BETA})")
print("=" * 72)

_, recs_5tr, a5 = run_comet(C5, S5, X5, tr5m.index, MES_PV, MES_COST,
                             h=BH5, tau_pct=BTAU, eta=BETA, lam=0.002, warmup=40)
st5 = stats(recs_5tr)
print(f"\n  Training: N={st5['N']}  n/day={st5['n_day']}  WR={st5['WR']:.1%}  "
      f"AvgW=${st5['AvgW']:+.2f}  AvgL=${st5['AvgL']:+.2f}  "
      f"Sharpe={st5['Sharpe']:.3f}  MaxDD=${st5['MaxDD']:,.2f}")

# |a_t| monotonicity
a_vals  = np.array([abs(r["a"]) for r in recs_5tr])
pnl_arr = np.array([r["pnl"] for r in recs_5tr])
print("\n  |a_t| monotonicity (5-min training):")
if len(a_vals) >= 9:
    q33, q67 = np.percentile(a_vals, [33.3, 66.7])
    b1 = pnl_arr[a_vals <= q33]; b2 = pnl_arr[(a_vals > q33) & (a_vals <= q67)]; b3 = pnl_arr[a_vals > q67]
    wr1 = (b1 > 0).mean() if len(b1) else 0
    wr2 = (b2 > 0).mean() if len(b2) else 0
    wr3 = (b3 > 0).mean() if len(b3) else 0
    print(f"    B1: N={len(b1):3d}  WR={wr1:.1%}  avg=${b1.mean() if len(b1) else 0:.2f}")
    print(f"    B2: N={len(b2):3d}  WR={wr2:.1%}  avg=${b2.mean() if len(b2) else 0:.2f}")
    print(f"    B3: N={len(b3):3d}  WR={wr3:.1%}  avg=${b3.mean() if len(b3) else 0:.2f}")
    mono5 = (wr3 > wr2 > wr1) or (wr3 > wr1 and wr3 > wr2 * 0.95)
    print(f"    → {'PASS ✓' if mono5 else 'FAIL ✗'}")
else:
    mono5 = False

# Direction split
long_r  = [r for r in recs_5tr if r["dir"] > 0]
short_r = [r for r in recs_5tr if r["dir"] < 0]
sl = stats(long_r); ss = stats(short_r)
print(f"\n  LONG:  N={sl['N']}  WR={sl['WR']:.1%}  Sharpe={sl['Sharpe']:.3f}  PnL=${sl['PnL']:,.0f}")
print(f"  SHORT: N={ss['N']}  WR={ss['WR']:.1%}  Sharpe={ss['Sharpe']:.3f}  PnL=${ss['PnL']:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION E: OOS + MNQ transfer (5-min best config)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SECTION E: OOS MES + MNQ TRANSFER  (5-min best config)")
print("=" * 72)

_, recs_5oo, _ = run_comet(C5o, S5o, X5o, oo5m.index, MES_PV, MES_COST,
                            h=BH5, tau_pct=BTAU, eta=BETA, lam=0.002, warmup=10)
st5oo = stats(recs_5oo)
print(f"\n  OOS MES: N={st5oo['N']}  WR={st5oo['WR']:.1%}  bkeven={st5oo['bkeven_wr']:.1%}  "
      f"Sharpe={st5oo['Sharpe']:.3f}  PnL=${st5oo['PnL']:,.0f}")

_, recs_5mn, _ = run_comet(C5m, S5m, X5m, mn5m.index, MNQ_PV, MNQ_COST,
                            h=BH5, tau_pct=BTAU, eta=BETA, lam=0.002, warmup=10)
st5mn = stats(recs_5mn)
print(f"  MNQ:     N={st5mn['N']}  WR={st5mn['WR']:.1%}  bkeven={st5mn['bkeven_wr']:.1%}  "
      f"Sharpe={st5mn['Sharpe']:.3f}  PnL=${st5mn['PnL']:,.0f}")

wr_gap = abs(st5["WR"] - st5oo["WR"])
print(f"\n  Train WR={st5['WR']:.1%}  OOS WR={st5oo['WR']:.1%}  gap={wr_gap:.1%}  "
      f"{'OK ≤8pp' if wr_gap <= 0.08 else 'OVERFIT >8pp'}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION F: WR vs h curve summary (both timeframes)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SECTION F: WR vs HOLD-HORIZON CURVE SUMMARY")
print("=" * 72)
print(f"\n  1-min bars:")
print(f"  {'h (bars)':>10}  {'WR':>6}  {'bkeven':>7}  {'gap':>6}  {'Sharpe':>8}")
for row in rows_1m:
    g = row["WR"] - row["bkeven_wr"]
    print(f"  {row['h']:>10}  {row['WR']:>6.1%}  {row['bkeven_wr']:>7.1%}  {g:>+6.1%}  {row['Sharpe']:>8.3f}")

print(f"\n  5-min bars:")
print(f"  {'h (bars)':>10}  {'h (min)':>8}  {'WR':>6}  {'bkeven':>7}  {'gap':>6}  {'Sharpe':>8}")
for row in rows_5m:
    g = row["WR"] - row["bkeven_wr"]
    print(f"  {row['h']:>10}  {row['h']*5:>8}m  {row['WR']:>6.1%}  {row['bkeven_wr']:>7.1%}  {g:>+6.1%}  {row['Sharpe']:>8.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION G: Per-day + Monte Carlo (5-min best if positive, else 1-min best)
# ─────────────────────────────────────────────────────────────────────────────
best_overall_sharpe = max(best_5m["Sharpe"], best_1m["Sharpe"])
print("\n" + "=" * 72)
print(f"  SECTION G: MONTE CARLO  (best overall config, Sharpe={best_overall_sharpe:.3f})")
print("=" * 72)

use_recs = recs_5tr if best_5m["Sharpe"] >= best_1m["Sharpe"] else None
if use_recs is None:
    # fall back to best 1-min h
    BH1_BEST = best_1m["h"]
    _, use_recs, _ = run_comet(C1, S1, X1, tr1m.index, MES_PV, MES_COST,
                               h=BH1_BEST, tau_pct=0.80, eta=0.005, lam=0.002, warmup=200)

by_day = {}
for r in use_recs:
    by_day.setdefault(r["date"], 0.0); by_day[r["date"]] += r["pnl"]
daily = np.array(list(by_day.values()))
n_days = len(daily)
print(f"  Active days: {n_days}  mean=${daily.mean():.0f}  std=${daily.std():.0f}  "
      f"day_WR={(daily > 0).mean():.1%}")

rng_mc = np.random.default_rng(42)
samp = rng_mc.choice(daily, size=(10_000, 60), replace=True)
n_pass = n_bust_dd = n_bust_dl = n_timeout = 0
for path in samp:
    eq = 50_000; peak = 50_000; bust = False; passed = False
    for dp in path:
        eq += dp; peak = max(peak, eq)
        if dp < -1_000: n_bust_dl += 1; bust = True; break
        if peak - eq > 2_000: n_bust_dd += 1; bust = True; break
        if eq >= 53_000: passed = True; break
    if passed: n_pass += 1
    elif not bust: n_timeout += 1

print(f"  MC P(pass)={n_pass/10_000:.1%}  P(bust_dd)={n_bust_dd/10_000:.1%}  "
      f"P(bust_daily)={n_bust_dl/10_000:.1%}  P(timeout)={n_timeout/10_000:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  VERDICT")
print("=" * 72)

any_positive_1m = any(r["Sharpe"] > 0 for r in rows_1m)
any_positive_5m = any(r["Sharpe"] > 0 for r in rows_5m)
any_above_bkeven = any(r["WR"] > r["bkeven_wr"] + 0.01 for r in rows_5m)

print(f"\n  1-min horizon: any Sharpe>0   = {'YES ✓' if any_positive_1m else 'NO ✗'}")
print(f"  5-min horizon: any Sharpe>0   = {'YES ✓' if any_positive_5m else 'NO ✗'}")
print(f"  5-min horizon: WR > bkeven    = {'YES ✓' if any_above_bkeven else 'NO ✗'}")
print(f"  5-min OOS Sharpe              = {st5oo['Sharpe']:.3f}  {'✓' if st5oo['Sharpe'] > 0 else '✗'}")
print(f"  MNQ transfer Sharpe           = {st5mn['Sharpe']:.3f}  {'✓' if st5mn['Sharpe'] > 0 else '✗'}")

if (any_positive_5m or any_positive_1m) and st5oo["Sharpe"] > 0:
    verdict = "PASS — extend horizon to 5-min, positive OOS Sharpe confirmed"
elif any_above_bkeven:
    verdict = "MARGINAL — edge exists but not statistically robust; gather more data"
else:
    verdict = "KILL — OHLC signals have no predictive power at any tested horizon"

print(f"\n  >>> VERDICT: {verdict} <<<")

# save
out = ROOT / "rule_based_v1/diagnostics/comet_horizon_results.json"
out.write_text(json.dumps({
    "best_1m": {"h": best_1m["h"], "Sharpe": best_1m["Sharpe"], "WR": best_1m["WR"]},
    "best_5m": {"h": best_5m["h"], "Sharpe": best_5m["Sharpe"], "WR": best_5m["WR"]},
    "best_5m_oos": {"Sharpe": st5oo["Sharpe"], "WR": st5oo["WR"]},
    "best_5m_mnq": {"Sharpe": st5mn["Sharpe"], "WR": st5mn["WR"]},
    "any_positive_1m": bool(any_positive_1m),
    "any_positive_5m": bool(any_positive_5m),
    "mc_p_pass": round(n_pass / 10_000, 3),
    "verdict": verdict,
    "rows_1m": rows_1m,
    "rows_5m": rows_5m,
}, indent=2))
print(f"\n  Saved → {out}")
