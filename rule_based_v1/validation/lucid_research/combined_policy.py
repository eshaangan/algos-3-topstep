"""End-to-end test of the two policies together, split by era.

Both pieces were optimised separately, which is the standard route to a result that
does not survive. This runs them jointly:

  EVAL    n_hi micros until the floor LOCKS at 100,100, then n_lo
  FUNDED  n_f micros, withdraw nothing until balance >= W, floor-jump rule ON

Reported against the honest baseline (flat 2 micros, withdraw immediately) and with
dev/val split. Beta-stripped, live breach, measured cost throughout.
"""
import sys, numpy as np
sys.path.insert(0, "rule_based_v1/validation/lucid_research")
from account import RET, LOW, N, rt

START, TARGET, MLL, LOCK, FEE = 100_000., 106_000., 3_000., 100_100., 158.

def evaluate(i0, n_lo, n_hi, ret, low, max_s=400):
    bal = START; floor = START - MLL
    for k in range(max_s):
        i = i0 + k
        if i >= len(ret): return None, k
        n = n_lo if floor >= LOCK - 1 else n_hi
        c = rt(n); pnl = ret[i]*n - c; worst = min(low[i]*n - c, pnl)
        if bal + worst <= floor: return False, k+1
        bal += pnl
        if bal >= TARGET: return True, k+1
        floor = max(floor, min(bal - MLL, LOCK))
    return None, max_s

def funded(i0, n, ret, low, W, max_s=400):
    c = rt(n); bal = START; floor = START - MLL; q = 0; bank = 0.0
    for k in range(max_s):
        i = i0 + k
        if i >= len(ret): break
        pnl = ret[i]*n - c; worst = min(low[i]*n - c, pnl)
        if bal + worst <= floor: return bank
        bal += pnl
        if pnl >= 200.0: q += 1
        if q >= 5 and bal >= W and bal > START:
            take = min(bal - START, 2500.0)
            if take >= 200:
                bank += take; bal -= take; q = 0
                if floor < LOCK: floor = LOCK      # floor-jump penalty
        floor = max(floor, min(bal - MLL, LOCK))
    return bank

def trial(n_lo, n_hi, n_f, W, ret, low, lo, hi):
    ps, cash = [], []
    for s in range(lo, hi, 3):
        ok, used = evaluate(s, n_lo, n_hi, ret, low)
        if ok is None: continue
        ps.append(ok)
        cash.append(funded(s+used, n_f, ret, low, W)*0.90 if ok else 0.0)
    if not ps: return 0, 0, 0
    p = np.mean(ps); c = np.mean(cash)
    return p, c, c - FEE

mu = RET.mean(); rd = RET-mu; ld = LOW-mu; H = N//2
CONFIGS = [
    ("BASELINE flat-2, withdraw immediately", 2, 2, 2, 100_200),
    ("patient withdrawal only",               2, 2, 2, 102_000),
    ("bold sizing only",                      4, 10, 4, 100_200),
    ("BOTH: bold 10->4, withdraw at 102k",    4, 10, 2, 102_000),
    ("BOTH, funded at 4 micros",              4, 10, 4, 102_000),
    ("BOTH, leg1=20",                         4, 20, 2, 102_000),
]
print("End-to-end, beta-stripped. EV is net of the $158 fee.\n")
print(f"{'config':>38} {'era':>4} {'P(pass)':>8} {'E[cash]':>8} {'EV':>7}")
for lab, nlo, nhi, nf, W in CONFIGS:
    for tag, a, b in (("dev", 0, H), ("val", H, N-420), ("all", 0, N-420)):
        p, c, ev = trial(nlo, nhi, nf, W, rd, ld, a, b)
        print(f"{lab if tag=='dev' else '':>38} {tag:>4} {p:>8.1%} {c:>8,.0f} {ev:>7,.0f}")
