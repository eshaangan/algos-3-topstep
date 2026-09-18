"""Attribution: where does the LucidFlex 100k lose its theoretical value?

Theory for a driftless fair game with a STATIC floor and zero cost:
    P(pass)      = D/(T+D) = 3,000/9,000 = 33.3%
    E[cash|pass] <= split * D = 0.90 * 3,000 = $2,700
    EV           = 0.333 * 2,700 - 158 = +$742

Measured reality is negative. This turns the four mechanics on one at a time to
say which one eats what. Beta-stripped throughout, so nothing here is a market call.
"""
import sys, numpy as np
sys.path.insert(0, "rule_based_v1/validation/lucid_research")
from account import RET, LOW, N, rt

START, TARGET, MLL, LOCK = 100_000., 106_000., 3_000., 100_100.

def sim(i0, n, ret, low, trailing, live, cost, harvest, max_s=100_000):
    c = rt(n) if cost else 0.0
    bal = START; floor = START - MLL; q = 0; bank = 0.0
    for k in range(max_s):
        i = i0 + k
        if i >= len(ret): return ("timeout", bank, k)
        pnl = ret[i]*n - c
        worst = min(low[i]*n - c, pnl) if live else pnl
        if bal + worst <= floor: return ("bust", bank, k+1)
        bal += pnl
        if not harvest and bal >= TARGET: return ("pass", bank, k+1)
        if harvest:
            if pnl >= 200.0: q += 1
            if q >= 5 and bal > START:
                take = min(bal-START, 2500.0); bank += take; bal -= take; q = 0
        if trailing: floor = max(floor, min(bal-MLL, LOCK))
    return ("timeout", bank, max_s)

mu = RET.mean(); rd = RET-mu; ld = LOW-mu
n = 2
STEPS = [
    ("theory: static floor, no live breach, no cost", False, False, False),
    ("+ LIVE intraday breach",                        False, True,  False),
    ("+ TRAILING floor (the ratchet)",                True,  True,  False),
    ("+ measured COST drag",                          True,  True,  True),
]
print(f"Attribution at n={n} micros, beta-stripped. Theory says 33.3% / $2,700 / +$742.\n")
print(f"{'mechanic added':>46} {'P(pass)':>8} {'E[cash|pass]':>13} {'EV':>7} {'delta EV':>9}")
prev = None
for lab, tr, lv, co in STEPS:
    ps, cash = [], []
    for s in range(0, N-520, 3):
        o, _, used = sim(s, n, rd, ld, tr, lv, co, False)
        ok = (o == "pass"); ps.append(ok)
        if ok:
            _, w, _ = sim(s+used, n, rd, ld, tr, lv, co, True)
            cash.append(w*0.90)
    p = np.mean(ps); e = np.mean(cash) if cash else 0.0; ev = p*e - 158
    d = "" if prev is None else f"{ev-prev:>+9,.0f}"
    print(f"{lab:>46} {p:>8.1%} {e:>13,.0f} {ev:>7,.0f} {d:>9}")
    prev = ev
print(f"\nbound on E[cash|pass] is split*D = 0.90 * 3,000 = $2,700")
