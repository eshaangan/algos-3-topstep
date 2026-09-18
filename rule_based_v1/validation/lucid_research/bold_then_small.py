"""BIG size through the ratchet, SMALL size after it locks. The untested direction.

The eval is two barriers in series:
  leg 1  100,000 -> 103,100  floor TRAILS behind you (ratchets on every new high)
  leg 2  103,100 -> 106,000  floor LOCKED at 100,100, i.e. a STATIC barrier

Under a trailing floor, survival decays with the NUMBER of ratchets -- roughly
exp(-T/D) in the small-step limit -- so many small steps is the worst way to cross
leg 1. Bold play crosses it in few steps. Leg 2 is a static barrier where optional
stopping applies and size only adds cost drag, so small is right there.

The earlier two-phase test only ever ran small-then-LARGE (lo <= hi). This runs the
reverse. Beta-stripped, live breach, measured cost.
"""
import sys, numpy as np
sys.path.insert(0, "rule_based_v1/validation/lucid_research")
from account import RET, LOW, N, rt

START, TARGET, MLL, LOCK = 100_000., 106_000., 3_000., 100_100.

def evaluate(i0, n_lo, n_hi, ret, low, max_s=400):
    """n_hi until the floor locks, then n_lo."""
    bal = START; floor = START - MLL
    for k in range(max_s):
        i = i0 + k
        if i >= len(ret): return None
        n = n_lo if floor >= LOCK - 1 else n_hi
        c = rt(n)
        pnl = ret[i]*n - c; worst = min(low[i]*n - c, pnl)
        if bal + worst <= floor: return False
        bal += pnl
        if bal >= TARGET: return True
        floor = max(floor, min(bal - MLL, LOCK))
    return None

mu = RET.mean(); rd = RET-mu; ld = LOW-mu; H = N//2
SIZES = (2, 4, 6, 10, 20, 40, 60)
print("P(pass), beta-stripped. Rows = size through the RATCHET (leg 1),")
print("columns = size after the floor LOCKS (leg 2). Diagonal = flat sizing.\n")
HDR = "leg1 / leg2"
print(f"{HDR:>11} " + " ".join(f"{s:>7}" for s in SIZES))
best = None
for hi in SIZES:
    row = []
    for lo in SIZES:
        r = [evaluate(s, lo, hi, rd, ld) for s in range(0, N-420, 3)]
        r = [x for x in r if x is not None]
        p = np.mean(r) if r else 0.0
        row.append(p)
        if best is None or p > best[0]: best = (p, hi, lo)
    print(f"{hi:>10} " + " ".join(f"{v:>6.1%} " for v in row))
print(f"\nbest cell: {best[0]:.1%} with {best[1]} micros through the ratchet, "
      f"{best[2]} after the lock")
print(f"flat-2 reference: {np.mean([x for x in (evaluate(s,2,2,rd,ld) for s in range(0,N-420,3)) if x is not None]):.1%}")
