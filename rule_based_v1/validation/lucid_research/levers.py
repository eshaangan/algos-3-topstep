import sys, numpy as np
sys.path.insert(0,"/private/tmp/claude-501/-Users-eshaanganguly-Documents-projects-algos-3-topstep/f6967d40-9bb8-470d-b3bc-1b7c827a9ace/scratchpad")
from account import *

LONG = np.ones(N)
print("Lucid 100k eval, always-long, real tape. P(pass):\n")
print("A) FLAT SIZE")
print(f"{'micros':>7} {'P(pass)':>8}")
best=None
for n in (1,2,3,4,6,8,10,14,20,30,40,60):
    p,_ = rate(LONG, flat(n)); print(f"{n:>7} {p:>8.1%}")
    if best is None or p>best[1]: best=(n,p)
print(f"best flat: {best[0]} micros at {best[1]:.1%}")

print("\nB) TWO-PHASE SIZE  (small until the floor LOCKS at 100,100, then switch)")
print(f"{'lo':>4} {'hi':>4} {'P(pass)':>8}  {'vs flat lo':>10}")
for lo in (2,3,4,6):
    base,_ = rate(LONG, flat(lo))
    for hi in (lo,6,10,20,40,60):
        if hi < lo: continue
        p,_ = rate(LONG, twophase(lo,hi))
        print(f"{lo:>4} {hi:>4} {p:>8.1%}  {p-base:>+9.1f}pp")

print("\nC) PROTECTIVE STOP  (cap the intraday excursion so it cannot touch the floor)")
print(f"{'micros':>7} " + " ".join(f"{f'${s}' if s else 'none':>8}" for s in (None,2000,1500,1000,750,500,300)))
for n in (2,4,6,10,20,40):
    row=[rate(LONG, flat(n), stop=s)[0] for s in (None,2000,1500,1000,750,500,300)]
    print(f"{n:>7} " + " ".join(f"{v:>8.1%}" for v in row))
