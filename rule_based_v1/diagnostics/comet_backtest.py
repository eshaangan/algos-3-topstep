"""
COMET — Contextual Online Mixture of Experts Trader
Adaptive softmax gate over 4 weak OHLC experts, updated via online mirror descent.

Experts:
  s1 = -r̃_{t-1}                              (micro reversal)
  s2 = sign(r̃_{t-1} + 0.7r̃_{t-2} + 0.49r̃_{t-3})  (3-bar continuation)
  s3 = sign(CLV) × I[range > 1.2×med60]       (range-expansion / close-location)
  s4 = -sign(Body) × I[|Body| > 0.5]          (overreaction repair)

Gate:  w_k = softmax(θ_k · x_t),  x_t ∈ ℝ^7
Update: θ_k += η · x_t · (πk - π̄);  πk = sk · R_{t→t+h}
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
    _df_train = _s["/bars_1m"].set_index("timestamp")
_df_train.index = pd.to_datetime(_df_train.index, utc=True).tz_convert("US/Eastern")
_df_train = _df_train.sort_index()

with pd.HDFStore(str(ROOT / "data/processed/jan_feb_2026_oos_test_1m.h5"), "r") as _s:
    _df_oos = _s["/bars_1min"].copy()
_df_oos.index = pd.to_datetime(_df_oos.index, utc=True).tz_convert("US/Eastern")
_df_oos = _df_oos.sort_index()

with pd.HDFStore(str(ROOT / "data/processed/mnq_2026ytd_1min.h5"), "r") as _s:
    _df_mnq = _s["/bars_1min"].copy()
_df_mnq.index.name = "timestamp"


def rth(df: pd.DataFrame) -> pd.DataFrame:
    mask = ((df.index.hour == 9) & (df.index.minute >= 30)) | ((df.index.hour > 9) & (df.index.hour < 16))
    return df[mask].copy()


train = rth(_df_train)
oos   = rth(_df_oos)
mnq   = _df_mnq.copy()

print(f"Train: {len(train):,} bars  {train.index[0].date()} – {train.index[-1].date()}")
print(f"OOS:   {len(oos):,} bars  {oos.index[0].date()} – {oos.index[-1].date()}")
print(f"MNQ:   {len(mnq):,} bars  {mnq.index[0].date()} – {mnq.index[-1].date()}")

N_EXPERTS = 4
N_FEAT    = 7

# ─────────────────────────────────────────────────────────────────────────────
# Vectorised feature pre-computation
# ─────────────────────────────────────────────────────────────────────────────

def precompute(bars: pd.DataFrame,
               mu_tod:  np.ndarray = None,
               sig_tod: np.ndarray = None):
    """
    Returns: C, S (n×4 experts), X (n×7 context), tod, mu_tod, sig_tod
    mu_tod / sig_tod are returned so they can be frozen for OOS/MNQ.
    """
    O = bars["open"].values.astype(float)
    H = bars["high"].values.astype(float)
    L = bars["low"].values.astype(float)
    C = bars["close"].values.astype(float)
    n = len(bars)

    # minute-of-day slot [0, 389]
    tod = np.clip(bars.index.hour * 60 + bars.index.minute - 9 * 60 - 30, 0, 389)

    # day-boundary mask: first bar of each session → zero out lagged signals
    dates   = bars.index.date
    is_new_day = np.array([False] + [dates[i] != dates[i - 1] for i in range(1, n)])

    # simple returns; zero at day boundary
    r = np.zeros(n)
    r[1:] = C[1:] / C[:-1] - 1.0
    r[is_new_day] = 0.0

    # ── de-seasonalisation ───────────────────────────────────────────────────
    if mu_tod is None:
        mu_tod  = np.zeros(390)
        sig_tod = np.full(390, 1e-4)
        for slot in range(390):
            mask_s = tod == slot
            if mask_s.sum() > 1:
                mu_tod[slot]  = r[mask_s].mean()
                sig_tod[slot] = max(r[mask_s].std(), 1e-6)

    r_tilde = (r - mu_tod[tod]) / (sig_tod[tod] + 1e-8)
    r_tilde[is_new_day] = 0.0          # no leakage across sessions

    # ── OHLC geometry ────────────────────────────────────────────────────────
    rng      = H - L
    rng_safe = rng + 1e-6
    CLV      = (2 * C - H - L) / rng_safe
    Body     = (C - O) / rng_safe
    uw       = H - np.maximum(O, C)
    lw       = np.minimum(O, C) - L
    WickSkew = (uw - lw) / rng_safe
    med60    = pd.Series(rng).rolling(60, min_periods=1).median().values
    range_ratio = rng / (med60 + 1e-6)

    # ── Expert signals ────────────────────────────────────────────────────────
    r_t1 = np.roll(r_tilde, 1); r_t1[is_new_day] = 0.0
    r_t2 = np.roll(r_tilde, 2); r_t2[is_new_day] = 0.0
    r_t3 = np.roll(r_tilde, 3); r_t3[is_new_day] = 0.0

    # s1: micro reversal
    s1 = np.clip(-r_t1, -3, 3) / 3.0

    # s2: 3-bar continuation
    mom3 = r_t1 + 0.7 * r_t2 + 0.49 * r_t3
    s2   = np.sign(mom3) * (np.abs(mom3) > 0.05).astype(float)

    # s3: range-expansion / close-location
    s3 = np.sign(CLV) * (range_ratio > 1.2).astype(float)

    # s4: overreaction repair
    s4 = -np.sign(Body) * (np.abs(Body) > 0.5).astype(float)

    S = np.column_stack([s1, s2, s3, s4])   # (n, 4)

    # ── Context vector ────────────────────────────────────────────────────────
    tod_norm   = tod / 389.0
    global_sig = sig_tod[sig_tod > 1e-5].mean() if (sig_tod > 1e-5).any() else 1e-4
    deseas_vol = np.clip(sig_tod[tod] / (global_sig + 1e-8), 0.0, 5.0)
    rr_clamp   = np.clip(range_ratio, 0.0, 5.0)

    X = np.column_stack([
        np.ones(n),    # bias
        tod_norm,      # time-of-day normalised
        deseas_vol,    # seasonal vol ratio
        rr_clamp,      # range ratio
        CLV,           # close-location value
        Body,          # bar body
        WickSkew,      # wick asymmetry
    ])                                       # (n, 7)

    return C, S, X, tod, mu_tod, sig_tod


# ─────────────────────────────────────────────────────────────────────────────
# Online COMET loop
# ─────────────────────────────────────────────────────────────────────────────

def run_comet(C, S, X, idx,
              point_value: float, cost_rt: float,
              h: int = 2, tau_pct: float = 0.75,
              eta: float = 0.01, lam: float = 0.002,
              warmup: int = 200):
    """
    C         : (n,) close prices
    S         : (n, 4) expert signals ∈ [-1, 1]
    X         : (n, 7) context features
    idx       : pandas DatetimeIndex aligned with bars
    """
    n     = len(C)
    theta = np.zeros((N_EXPERTS, N_FEAT))

    abs_a_buf = deque(maxlen=200)
    update_q  = deque()    # (bar_idx, x_j, s_j, w_j)

    in_trade   = False
    trade_dir  = 0
    trade_px   = 0.0
    trade_bar  = -999

    pnl_list   = []
    trade_recs = []
    a_arr      = np.zeros(n)
    w_arr      = np.zeros((n, N_EXPERTS))

    for i in range(n):
        # ── gate weights ─────────────────────────────────────────────────────
        logits = theta @ X[i]
        logits -= logits.max()
        w = np.exp(logits); w /= (w.sum() + 1e-12)

        a_t      = float(w @ S[i])
        a_arr[i] = a_t
        w_arr[i] = w
        abs_a_buf.append(abs(a_t))

        # ── delayed gate update for bar (i-h) ────────────────────────────────
        while update_q and update_q[0][0] <= i - h:
            j, x_j, s_j, w_j = update_q.popleft()
            R_j    = C[i] - C[j]
            pi_k   = s_j * R_j
            pi_bar = float(w_j @ pi_k)
            theta += eta * np.outer(pi_k - pi_bar, x_j)
            theta *= (1.0 - lam)

        update_q.append((i, X[i].copy(), S[i].copy(), w.copy()))

        # ── threshold ────────────────────────────────────────────────────────
        ready = len(abs_a_buf) >= warmup
        thresh = np.percentile(list(abs_a_buf), tau_pct * 100) if ready else 1e9

        # ── exit ─────────────────────────────────────────────────────────────
        if in_trade and (i - trade_bar) >= h:
            raw = trade_dir * (C[i] - trade_px) * point_value
            pnl = raw - cost_rt
            pnl_list.append(pnl)
            trade_recs.append({
                "date":  str(idx[trade_bar].date()),
                "bar":   trade_bar,
                "dir":   trade_dir,
                "a":     round(float(a_arr[trade_bar]), 5),
                "entry": round(trade_px, 2),
                "exit":  round(C[i], 2),
                "pnl":   round(pnl, 2),
            })
            in_trade = False

        # ── entry ─────────────────────────────────────────────────────────────
        if not in_trade and ready and abs(a_t) > thresh:
            trade_dir = int(np.sign(a_t)) if a_t != 0 else 1
            trade_px  = C[i]
            trade_bar = i
            in_trade  = True

    # force-close open trade
    if in_trade:
        pnl = trade_dir * (C[-1] - trade_px) * point_value - cost_rt
        pnl_list.append(pnl)

    return pnl_list, trade_recs, a_arr, w_arr


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def stats(trade_recs):
    if not trade_recs:
        return dict(N=0, n_day=0.0, WR=0.0, AvgW=0.0, AvgL=0.0, PF=0.0, PnL=0.0, Sharpe=0.0, MaxDD=0.0)
    p    = np.array([t["pnl"] for t in trade_recs])
    n    = len(p)
    wins = p[p > 0]; losses = p[p <= 0]
    cum  = np.cumsum(p)
    dd   = (cum - np.maximum.accumulate(cum)).min()
    pf   = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 9.99

    # daily Sharpe
    by_day = {}
    for t in trade_recs:
        by_day.setdefault(t["date"], 0.0)
        by_day[t["date"]] += t["pnl"]
    daily = np.array(list(by_day.values()))
    n_days = len(daily)
    sharpe = (daily.mean() / (daily.std() + 1e-8)) * np.sqrt(252) if n_days > 1 else 0.0
    n_day  = n / n_days if n_days else 0.0

    return dict(N=n, n_day=round(n_day, 2), WR=round(len(wins)/n, 3),
                AvgW=round(wins.mean() if len(wins) else 0, 2),
                AvgL=round(losses.mean() if len(losses) else 0, 2),
                PF=round(pf, 3), PnL=round(float(p.sum()), 2),
                Sharpe=round(sharpe, 3), MaxDD=round(float(dd), 2))


def print_st(st, label=""):
    print(f"  [{label}]")
    print(f"    N={st['N']}  n/day={st['n_day']}  WR={st['WR']:.1%}  "
          f"AvgW=${st['AvgW']:+.2f}  AvgL=${st['AvgL']:+.2f}  "
          f"PF={st['PF']:.2f}  PnL=${st['PnL']:,.2f}  "
          f"Sharpe={st['Sharpe']:.3f}  MaxDD=${st['MaxDD']:,.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute features
# ─────────────────────────────────────────────────────────────────────────────
MES_PV   = 5.0   # $5 per MES point
MNQ_PV   = 2.0   # $2 per MNQ point
MES_COST = 2.50  # round-trip cost (slippage + commissions)
MNQ_COST = 1.00

print("\nPre-computing features ...")
C_tr, S_tr, X_tr, tod_tr, mu_tod, sig_tod = precompute(train)
C_oo, S_oo, X_oo, tod_oo, _, _           = precompute(oos,  mu_tod, sig_tod)
C_mn, S_mn, X_mn, tod_mn, _, _           = precompute(mnq,  mu_tod, sig_tod)
print("  done.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: Static expert baselines (no gate learning, h=2, tau=0.75)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SECTION 0: STATIC EXPERT BASELINES  (training, h=2, tau=75th pct)")
print("=" * 72)

WARMUP = 200

def run_static(C, S_col, idx, pv, cost, h=2, tau_pct=0.75):
    """Trade a single expert signal with rolling-percentile threshold."""
    n   = len(C)
    sig = S_col  # (n,) signal ∈ [-1, 1]
    abs_buf  = deque(maxlen=200)
    in_trade = False; dir_ = 0; px_ = 0.0; bar_ = -1
    recs = []
    for i in range(n):
        a = float(sig[i])
        abs_buf.append(abs(a))
        ready  = len(abs_buf) >= WARMUP
        thresh = np.percentile(list(abs_buf), tau_pct * 100) if ready else 1e9
        if in_trade and (i - bar_) >= h:
            pnl = dir_ * (C[i] - px_) * pv - cost
            recs.append({"date": str(idx[bar_].date()), "bar": bar_,
                         "dir": dir_, "a": a, "entry": px_, "exit": C[i], "pnl": pnl})
            in_trade = False
        if not in_trade and ready and abs(a) > thresh:
            dir_ = int(np.sign(a)) if a != 0 else 1
            px_ = C[i]; bar_ = i; in_trade = True
    return recs


def run_uniform(C, S, idx, pv, cost, h=2, tau_pct=0.75):
    """Uniform blend: a_t = mean(s_k) regardless of context."""
    sig = S.mean(axis=1)
    return run_static(C, sig, idx, pv, cost, h=h, tau_pct=tau_pct)


expert_names = ["s1_reversal", "s2_continuation", "s3_range_expansion", "s4_overreaction"]
baseline_results = {}

for k, name in enumerate(expert_names):
    recs = run_static(C_tr, S_tr[:, k], train.index, MES_PV, MES_COST)
    st   = stats(recs)
    baseline_results[name] = st
    print_st(st, name)

recs_uniform = run_uniform(C_tr, S_tr, train.index, MES_PV, MES_COST)
st_uniform   = stats(recs_uniform)
baseline_results["uniform_blend"] = st_uniform
print_st(st_uniform, "uniform_blend (no gate)")

best_static_sharpe = max(v["Sharpe"] for v in baseline_results.values())
print(f"\n  Best static Sharpe = {best_static_sharpe:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: COMET SWEEP  (training)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SECTION 1: COMET PARAMETER SWEEP  (training)")
print("=" * 72)

sweep_grid = {
    "h":        [1, 2, 3],
    "tau_pct":  [0.65, 0.75, 0.80],
    "eta":      [0.005, 0.010, 0.020],
}

rows = []
total = len(sweep_grid["h"]) * len(sweep_grid["tau_pct"]) * len(sweep_grid["eta"])
done  = 0

for h in sweep_grid["h"]:
    for tau in sweep_grid["tau_pct"]:
        for eta in sweep_grid["eta"]:
            pnl_list, recs, _, _ = run_comet(C_tr, S_tr, X_tr, train.index,
                                              MES_PV, MES_COST,
                                              h=h, tau_pct=tau, eta=eta,
                                              lam=0.002, warmup=WARMUP)
            st = stats(recs)
            rows.append({"h": h, "tau": tau, "eta": eta, **st})
            done += 1
            if done % 9 == 0:
                print(f"  {done}/{total} done...")

rows_sorted = sorted(rows, key=lambda r: -r["Sharpe"])

print(f"\n  {'h':>2}  {'tau':>5}  {'eta':>6}  {'N':>5}  {'n/d':>5}  "
      f"{'WR':>6}  {'Sharpe':>7}  {'PnL':>9}  {'MaxDD':>9}")
print("  " + "-" * 72)
for r in rows_sorted[:15]:
    print(f"  {r['h']:>2}  {r['tau']:>5.2f}  {r['eta']:>6.3f}  {r['N']:>5}  "
          f"{r['n_day']:>5.1f}  {r['WR']:>6.1%}  {r['Sharpe']:>7.3f}  "
          f"${r['PnL']:>8,.0f}  ${r['MaxDD']:>8,.0f}")

best = rows_sorted[0]
print(f"\n  Best COMET: h={best['h']}, tau={best['tau']}, eta={best['eta']}  "
      f"Sharpe={best['Sharpe']:.3f}  vs best_static={best_static_sharpe:.3f}")

BEAT_STATIC = best["Sharpe"] > best_static_sharpe
print(f"  Beats best static: {'YES ✓' if BEAT_STATIC else 'NO ✗'}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DEEP ANALYSIS  (best config, training)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"  SECTION 2: DEEP ANALYSIS  (h={best['h']}, tau={best['tau']}, eta={best['eta']})")
print("=" * 72)

BH   = best["h"]
BTAU = best["tau"]
BETA = best["eta"]

pnl_tr, recs_tr, a_tr, w_tr = run_comet(C_tr, S_tr, X_tr, train.index,
                                          MES_PV, MES_COST,
                                          h=BH, tau_pct=BTAU, eta=BETA,
                                          lam=0.002, warmup=WARMUP)
st_tr = stats(recs_tr)
print_st(st_tr, f"Training (best COMET)")

# ── |a_t| monotonicity ────────────────────────────────────────────────────
print("\n  |a_t| monotonicity (training trades):")
a_vals  = np.array([abs(r["a"]) for r in recs_tr])
pnl_arr = np.array([r["pnl"] for r in recs_tr])
if len(a_vals) >= 9:
    q33, q67 = np.percentile(a_vals, [33.3, 66.7])
    b1 = pnl_arr[a_vals <= q33]
    b2 = pnl_arr[(a_vals > q33) & (a_vals <= q67)]
    b3 = pnl_arr[a_vals > q67]
    wr1 = (b1 > 0).mean() if len(b1) else 0
    wr2 = (b2 > 0).mean() if len(b2) else 0
    wr3 = (b3 > 0).mean() if len(b3) else 0
    print(f"    B1 (low |a|):  N={len(b1):3d}  WR={wr1:.1%}  avg=${b1.mean() if len(b1) else 0:.2f}")
    print(f"    B2 (mid |a|):  N={len(b2):3d}  WR={wr2:.1%}  avg=${b2.mean() if len(b2) else 0:.2f}")
    print(f"    B3 (high |a|): N={len(b3):3d}  WR={wr3:.1%}  avg=${b3.mean() if len(b3) else 0:.2f}")
    a_mono = (wr3 > wr2 > wr1) or (wr3 > wr1 and wr3 > wr2 * 0.95)
    print(f"    → {'PASS ✓' if a_mono else 'FAIL ✗'}")
else:
    a_mono = False
    print("    → Too few trades")

# ── Expert weight interpretation ──────────────────────────────────────────
print("\n  Average gate weights by time-of-day context:")
# Use w_arr from run_comet (weights at every bar, not just trades)
# Segment by morning/midday/afternoon
tod_arr = np.clip(train.index.hour * 60 + train.index.minute - 9 * 60 - 30, 0, 389)
labels_seg = ["morning (9:30-10:30)", "midday (10:30-13:00)", "afternoon (13:00-16:00)"]
seg_masks  = [tod_arr < 60, (tod_arr >= 60) & (tod_arr < 210), tod_arr >= 210]
for seg_label, mask in zip(labels_seg, seg_masks):
    w_seg = w_tr[mask]
    avg_w = w_seg.mean(axis=0)
    print(f"    {seg_label}: s1={avg_w[0]:.3f}  s2={avg_w[1]:.3f}  "
          f"s3={avg_w[2]:.3f}  s4={avg_w[3]:.3f}")

# ── Direction split ───────────────────────────────────────────────────────
print("\n  Direction split:")
long_recs  = [r for r in recs_tr if r["dir"] > 0]
short_recs = [r for r in recs_tr if r["dir"] < 0]
print_st(stats(long_recs),  "LONG")
print_st(stats(short_recs), "SHORT")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: OOS VALIDATION  (MES Jan–Feb 2026)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SECTION 3: OOS VALIDATION  (MES Jan–Feb 2026)")
print("=" * 72)

# Freeze theta from end of training run, then re-run OOS with online update continuing
# (clean: retrain from scratch on training data to get final theta, then run OOS)
pnl_oo, recs_oo, a_oo, w_oo = run_comet(C_oo, S_oo, X_oo, oos.index,
                                          MES_PV, MES_COST,
                                          h=BH, tau_pct=BTAU, eta=BETA,
                                          lam=0.002, warmup=50)  # shorter warmup for OOS
st_oo = stats(recs_oo)
print_st(st_oo, "OOS MES Jan-Feb 2026")

tr_wr = st_tr["WR"]
oo_wr = st_oo["WR"]
wr_gap = abs(tr_wr - oo_wr)
print(f"\n  Train WR={tr_wr:.1%}  OOS WR={oo_wr:.1%}  gap={wr_gap:.1%}  "
      f"{'GOOD (≤8pp)' if wr_gap <= 0.08 else 'OVERFIT (>8pp)'}")

# ── |a_t| monotonicity OOS ────────────────────────────────────────────────
if len(recs_oo) >= 9:
    a_oo_vals = np.array([abs(r["a"]) for r in recs_oo])
    p_oo_vals = np.array([r["pnl"] for r in recs_oo])
    q33o, q67o = np.percentile(a_oo_vals, [33.3, 66.7])
    b1o = p_oo_vals[a_oo_vals <= q33o]
    b2o = p_oo_vals[(a_oo_vals > q33o) & (a_oo_vals <= q67o)]
    b3o = p_oo_vals[a_oo_vals > q67o]
    wr1o = (b1o > 0).mean() if len(b1o) else 0
    wr3o = (b3o > 0).mean() if len(b3o) else 0
    print(f"\n  OOS |a_t| mono: B1 WR={wr1o:.1%} → B3 WR={wr3o:.1%}  "
          f"{'PASS ✓' if wr3o > wr1o else 'FAIL ✗'}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MNQ TRANSFER  (Jan–Mar 2026)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SECTION 4: MNQ TRANSFER  (Jan–Mar 2026)")
print("=" * 72)

pnl_mn, recs_mn, a_mn, w_mn = run_comet(C_mn, S_mn, X_mn, mnq.index,
                                          MNQ_PV, MNQ_COST,
                                          h=BH, tau_pct=BTAU, eta=BETA,
                                          lam=0.002, warmup=50)
st_mn = stats(recs_mn)
print_st(st_mn, "MNQ Jan-Mar 2026")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: ADAPTIVE vs STATIC NET BENEFIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SECTION 5: COMET vs UNIFORM BLEND vs BEST STATIC")
print("=" * 72)
print_st(st_uniform, "uniform_blend")
print_st(st_tr,      "COMET (adaptive gate)")
delta_sharpe = st_tr["Sharpe"] - st_uniform["Sharpe"]
print(f"\n  Gate adaptation lift: Δ Sharpe = {delta_sharpe:+.3f}  "
      f"({'significant' if delta_sharpe > 0.5 else 'marginal' if delta_sharpe > 0 else 'none'})")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: ETA / LAM SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"  SECTION 6: LEARNING RATE SENSITIVITY  (h={BH}, tau={BTAU})")
print("=" * 72)
print(f"  {'eta':>6}  {'lam':>6}  {'N':>5}  {'WR':>6}  {'Sharpe':>8}  {'PnL':>9}")
for eta in [0.001, 0.005, 0.010, 0.020, 0.050]:
    for lam in [0.0005, 0.002, 0.005]:
        _, rr, _, _ = run_comet(C_tr, S_tr, X_tr, train.index,
                                MES_PV, MES_COST, h=BH, tau_pct=BTAU,
                                eta=eta, lam=lam, warmup=WARMUP)
        st_ = stats(rr)
        print(f"  {eta:>6.3f}  {lam:>6.4f}  {st_['N']:>5}  {st_['WR']:>6.1%}  "
              f"{st_['Sharpe']:>8.3f}  ${st_['PnL']:>8,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: PER-DAY + MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SECTION 7: PER-DAY STATS + COMBINE MONTE CARLO  (training)")
print("=" * 72)

by_day: dict = {}
for rec in recs_tr:
    by_day.setdefault(rec["date"], 0.0)
    by_day[rec["date"]] += rec["pnl"]
daily_pnl = np.array(list(by_day.values()))
n_active   = len(daily_pnl)

print(f"  Active days: {n_active}")
print(f"  Daily PnL: min=${daily_pnl.min():.0f}  mean=${daily_pnl.mean():.0f}  "
      f"max=${daily_pnl.max():.0f}  std=${daily_pnl.std():.0f}")
print(f"  Day WR: {(daily_pnl > 0).mean():.1%}")

print("\n  Worst 5 sessions:")
for d, v in sorted(by_day.items(), key=lambda x: x[1])[:5]:
    print(f"    {d}  ${v:+.2f}")
print("  Best 5 sessions:")
for d, v in sorted(by_day.items(), key=lambda x: -x[1])[:5]:
    print(f"    {d}  ${v:+.2f}")

# Monte Carlo combine simulation
INIT_EQ  = 50000
PT_EQ    = 53000
TRAIL_DD = 2000
DAILY_LL = 1000
N_PATHS  = 10_000
N_DAYS   = 60

rng_mc = np.random.default_rng(42)
samples = rng_mc.choice(daily_pnl, size=(N_PATHS, N_DAYS), replace=True)

pass_count = 0; bust_dd = 0; bust_daily = 0; timeout_count = 0
for path in samples:
    equity = INIT_EQ
    trail_peak = INIT_EQ
    busted = False
    passed = False
    for dp in path:
        equity += dp
        trail_peak = max(trail_peak, equity)
        if dp < -DAILY_LL:
            bust_daily += 1; busted = True; break
        if (trail_peak - equity) > TRAIL_DD:
            bust_dd += 1; busted = True; break
        if equity >= PT_EQ:
            passed = True; break
    if passed:
        pass_count += 1
    elif not busted:
        timeout_count += 1

print(f"\n  Monte Carlo ({N_PATHS:,} paths × {N_DAYS} days):")
print(f"    P(pass)       = {pass_count/N_PATHS:.1%}")
print(f"    P(bust_dd)    = {bust_dd/N_PATHS:.1%}")
print(f"    P(bust_daily) = {bust_daily/N_PATHS:.1%}")
print(f"    P(timeout)    = {timeout_count/N_PATHS:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  VERDICT")
print("=" * 72)

verdict_items = {
    "Training Sharpe":        (st_tr["Sharpe"],     "> 0",          st_tr["Sharpe"] > 0),
    "Training WR":            (st_tr["WR"],          "≥ 52%",        st_tr["WR"] >= 0.52),
    "Beats best static":      (best["Sharpe"],       f"> {best_static_sharpe:.3f}", BEAT_STATIC),
    "Gate lift vs uniform":   (delta_sharpe,         "> 0",          delta_sharpe > 0),
    "|a_t| monotonicity":     (None,                 "PASS",         a_mono),
    "OOS WR gap":             (wr_gap,               "≤ 8pp",        wr_gap <= 0.08),
    "MNQ transfer Sharpe":    (st_mn["Sharpe"],      "> 0",          st_mn["Sharpe"] > 0),
    "P(pass combine)":        (pass_count/N_PATHS,   "> 50%",        pass_count/N_PATHS > 0.50),
}

passes = sum(1 for v in verdict_items.values() if v[2])
total_checks = len(verdict_items)

for k, (val, need, ok) in verdict_items.items():
    val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
    print(f"  {'✓' if ok else '✗'}  {k:<30}  {val_str:<10}  (need {need})")

print(f"\n  {passes}/{total_checks} checks pass")
if passes >= 6:
    verdict = "PASS — deploy with 1 MES, monitor weight drift"
elif passes >= 4:
    verdict = "MARGINAL — promising but needs more OOS data"
else:
    verdict = "KILL — adaptive gate adds no net value"

print(f"\n  >>> VERDICT: {verdict} <<<")

# Save results
results = {
    "best_config": {"h": BH, "tau_pct": BTAU, "eta": BETA, "lam": 0.002},
    "training": st_tr,
    "oos_mes": st_oo,
    "mnq": st_mn,
    "uniform_blend": st_uniform,
    "static_best_sharpe": best_static_sharpe,
    "beat_static": bool(BEAT_STATIC),
    "gate_lift_sharpe": round(delta_sharpe, 3),
    "a_monotonicity": bool(a_mono),
    "wr_gap_oos": round(wr_gap, 3),
    "mc_p_pass": round(pass_count / N_PATHS, 3),
    "verdict": verdict,
    "sweep": rows_sorted[:10],
}
out = ROOT / "rule_based_v1/diagnostics/comet_results.json"
out.write_text(json.dumps(results, indent=2))
print(f"\n  Saved → {out}")
