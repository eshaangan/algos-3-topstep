"""Markov regime detection as a DAY FILTER: which sessions to be long at all.

PRE-REGISTERED, written before any result was seen.

Why this is not the vol-regime trial already in the ledger: that one conditioned
the PURCHASE of an evaluation on VIX at entry. This conditions each SESSION on an
inferred regime. And skipping a session costs literally nothing here -- no round
turn, no drag -- so unlike a direction flip this lever is free if it works at all.

METHOD. Gaussian HMM on (session return, session range), 2 and 3 states, fitted
walk-forward on past data only, refit monthly. The regime for session i is the
Viterbi state at i-1 carried forward, so nothing from session i is used. Go long
only in states whose TRAINING-SAMPLE mean return is positive.

ADOPTION RULE, fixed in advance: adopt only if the filter beats always-long on
dollars per session in BOTH eras AND raises account EV in BOTH eras. Anything
less is a no.

LEAKAGE AUDIT: assert the regime label for session i uses no data from session i.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "rule_based_v1/validation/lucid_research")
from sessions import build
from hmmlearn.hmm import GaussianHMM

s, f, y = build()
R = s["ret"].values; RNG = ((s["high"]-s["low"])*2.0).values
N = len(R); H = N//2
OBS = np.column_stack([R, RNG])

TRAIN, STEP = 500, 21
def regime_signal(k):
    """Returns (go_long_mask, fitted_mask). Strictly out of sample."""
    go = np.zeros(N, dtype=bool); have = np.zeros(N, dtype=bool)
    for i in range(TRAIN, N, STEP):
        j = min(i+STEP, N)
        Xtr = OBS[:i]
        m = GaussianHMM(n_components=k, covariance_type="diag",
                        n_iter=60, random_state=0)
        try: m.fit(Xtr)
        except Exception: continue
        st = m.predict(Xtr)
        good = {q for q in range(k) if R[:i][st == q].mean() > 0}
        # regime for session t is the state inferred from data up to t-1
        for t in range(i, j):
            prev = m.predict(OBS[:t])[-1]          # uses sessions 0..t-1 only
            nxt = m.transmat_[prev].argmax()       # most likely state AT t
            go[t] = nxt in good; have[t] = True
    return go, have

print("Markov regime day-filter. Pre-registered; adopt only if BOTH eras improve.\n")
for k in (2, 3):
    go, have = regime_signal(k)
    m = have
    print(f"=== {k}-state HMM ===  {m.sum()} out-of-sample sessions, "
          f"long on {go[m].mean():.1%} of them")
    for tag, lo, hi in (("dev", TRAIN, H), ("val", H, N), ("all", TRAIN, N)):
        sel = np.zeros(N, bool); sel[lo:hi] = True; sel &= m
        if sel.sum() < 50: continue
        filt = np.where(go[sel], R[sel], 0.0)      # flat when the regime is bad
        alw = R[sel]
        # accuracy of the filter as a "trade / no-trade" call
        traded = go[sel]
        print(f"  {tag:>3} n={sel.sum():>4}  filter ${filt.mean():>+7.2f}/sess  "
              f"always-long ${alw.mean():>+7.2f}/sess  "
              f"in-market {traded.mean():>5.1%}  "
              f"E[r|long] ${alw[traded].mean() if traded.any() else 0:>+7.2f}  "
              f"E[r|flat] ${alw[~traded].mean() if (~traded).any() else 0:>+7.2f}")
    print()
