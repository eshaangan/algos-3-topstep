"""LucidFlex 100k evaluation, simulated exactly as the rulebook reads.

  start 100,000, target 106,000, MLL 3,000 trailing on the CLOSING balance,
  floor LOCKS at 100,100 once balance exceeds 103,100,
  BREACH is evaluated on LIVE intraday equity (confirmed by the firm).

Direction, size schedule and a protective stop are all pluggable so the levers
can be compared on one tape.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/private/tmp/claude-501/-Users-eshaanganguly-Documents-projects-algos-3-topstep/f6967d40-9bb8-470d-b3bc-1b7c827a9ace/scratchpad")
from sessions import build

START, TARGET, MLL, LOCK = 100_000., 106_000., 3_000., 100_100.
BOOK = {1:1.0,2:1.0,4:1.0,6:1.0,9:1.111,13:1.269,20:1.525,40:2.188,60:2.792}
XS = np.array(sorted(BOOK)); YS = np.array([BOOK[k] for k in sorted(BOOK)])
def rt(n): return (float(np.interp(n, XS, YS)) + 1.00) * n

s, f, y = build()
RET = s["ret"].values; LOW = s["low_x"].values; HIGH = s["high_x"].values
N = len(RET)

def episode(i0, direction, size_fn, stop=None, max_s=400):
    """One evaluation attempt starting at session i0. Returns 'pass'/'bust'/'timeout'."""
    bal = START; floor = START - MLL
    for k in range(max_s):
        i = i0 + k
        if i >= N: return "timeout"
        n = size_fn(bal, floor)
        if n <= 0: return "timeout"
        c = rt(n); dr = direction[i]
        # adverse excursion and final P&L for this side, in dollars
        adverse = (LOW[i] if dr > 0 else -HIGH[i]) * n - c
        pnl = dr * RET[i] * n - c
        if stop is not None and adverse <= -stop:      # protective stop fires
            pnl = -stop - c; adverse = -stop - c
        if bal + adverse <= floor: return "bust"
        bal += pnl
        if bal >= TARGET: return "pass"
        floor = max(floor, min(bal - MLL, LOCK))       # EOD ratchet, capped at lock
    return "timeout"

def rate(direction, size_fn, stop=None, stride=3):
    outs = [episode(i, direction, size_fn, stop) for i in range(0, N-420, stride)]
    return outs.count("pass") / len(outs), len(outs)

def flat(n):   return lambda bal, floor: n
def twophase(n_lo, n_hi):
    return lambda bal, floor: (n_hi if floor >= LOCK - 1 else n_lo)

TRUE = np.sign(RET); TRUE[TRUE == 0] = 1
def oracle(p, seed=0):
    rng = np.random.default_rng(seed)
    flip = rng.random(N) > p
    d = TRUE.copy(); d[flip] *= -1
    return d
