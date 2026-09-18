"""Withdrawal policy in the FUNDED phase, with the floor-jump rule modelled.

The attribution says payout friction is the single biggest loss (~$430 of the
~$758 that separates theory from reality). Every funded-phase simulation so far
has withdrawn as soon as 5 qualifying days existed -- and has NOT modelled the
rule the ledger already recorded:

  requesting a payout while the MLL is still TRAILING instantly jumps the floor
  to 100,100, discarding up to $2,900 of buffer.

So the previous numbers are optimistic (no penalty applied) AND the policy tested
was the worst one available. This grids the withdrawal threshold W: take nothing
until balance >= W. W = 100,200 reproduces the old impatient behaviour; W = 103,100
is the ledger's "never take a payout before the floor locks" rule.

Beta-stripped, live breach, measured cost. Pre-registered: report all eras.
"""
import sys, numpy as np
sys.path.insert(0, "rule_based_v1/validation/lucid_research")
from account import RET, LOW, N, rt

START, TARGET, MLL, LOCK = 100_000., 106_000., 3_000., 100_100.
TRAIL_END = 103_100.          # above this the floor is locked at 100,100 anyway

def evaluate(i0, n, ret, low, max_s=400):
    c = rt(n); bal = START; floor = START - MLL
    for k in range(max_s):
        i = i0 + k
        if i >= len(ret): return None, k
        pnl = ret[i]*n - c; worst = min(low[i]*n - c, pnl)
        if bal + worst <= floor: return False, k+1
        bal += pnl
        if bal >= TARGET: return True, k+1
        floor = max(floor, min(bal - MLL, LOCK))
    return None, max_s

def funded(i0, n, ret, low, W, jump, max_s=400):
    """W = withdraw only at or above this balance. jump = model the floor-jump rule."""
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
                if jump and floor < LOCK:      # payout taken while still trailing
                    floor = LOCK
        floor = max(floor, min(bal - MLL, LOCK))
    return bank

mu = RET.mean(); rd = RET-mu; ld = LOW-mu; H = N//2
POL = [100_200, 101_000, 102_000, 103_000, 103_100, 104_000, 105_000]
print("Funded-phase withdrawal policy. 'jump OFF' = the old (wrong) model.\n")
for n in (2, 4, 10):
    print(f"n={n} micros")
    print(f"{'withdraw at >=':>15} {'E[cash] jump OFF':>17} {'jump ON':>9} "
          f"{'dev':>8} {'val':>8}")
    for W in POL:
        rows = list(range(0, N-520, 3))
        off, on, dev, val = [], [], [], []
        for s in rows:
            ok, used = evaluate(s, n, rd, ld)
            if not ok: continue
            a = funded(s+used, n, rd, ld, W, False)*0.90
            b = funded(s+used, n, rd, ld, W, True)*0.90
            off.append(a); on.append(b)
            (dev if s < H else val).append(b)
        print(f"{W:>15,} {np.mean(off):>17,.0f} {np.mean(on):>9,.0f} "
              f"{np.mean(dev) if dev else 0:>8,.0f} {np.mean(val) if val else 0:>8,.0f}")
    print()
