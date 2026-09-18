"""Payout POLICY, not signal: make daily P&L uniform to satisfy the consistency rule.

THE MECHANISM. ETF's DTF pays only on days that clear $500 AND are >= 50% of your
BEST day. Always-long produces wildly dispersed days, so one +$3,000 session raises
the bar to $1,500 for every other qualifying day and disqualifies most of them.

A daily profit cap removes that. Exit at +$X and stop for the session: every winning
day is then ~$X, the best day is ~$X, and the 50% test becomes free. It costs no
extra round turn (you were flattening at the close anyway) so drag is UNCHANGED.

This is path-independent: with a target and no stop, you are filled at X if and only
if the session high reached X. No intrabar ordering assumption is needed.

Pre-registered: adopt only if it beats the uncapped policy in BOTH eras.
"""
import sys, numpy as np
sys.path.insert(0, "rule_based_v1/validation/lucid_research")
from account import RET, LOW, HIGH, N, rt

START, FLOOR, NET = 100_000., 95_000., 5_100.
ATD_N, ATD_MIN, ATD_CONS, CAP = 20, 500.0, 0.50, 25_000.

def run(i0, n, ret, high, low, cap_x, live=True, max_s=500):
    c = rt(n); bal = START; good = []; bank = 0.0; safe = False
    for k in range(max_s):
        i = i0 + k
        if i >= len(ret): break
        if cap_x and high[i]*n >= cap_x:      # target filled intraday
            pnl = cap_x - c; worst = min(low[i]*n - c, pnl)
        else:
            pnl = ret[i]*n - c; worst = min(low[i]*n - c, pnl)
        if live and bal + worst <= FLOOR: return bank
        bal += pnl
        if bal <= FLOOR: return bank
        if pnl >= ATD_MIN: good.append(pnl)
        if not safe and bal - START >= NET: safe = True
        if safe and len(good) >= ATD_N:
            best = max(good); q = [d for d in good if d >= ATD_CONS*best]
            if len(q) >= ATD_N:
                avail = bal - START - NET
                if avail >= 500:
                    take = min(avail, CAP - bank)
                    bank += take; bal -= take; good = []
                    if bank >= CAP: return bank
    return bank

mu = RET.mean(); rd = RET-mu; hd = HIGH-mu; ld = LOW-mu; H = N//2
print("ETF DTF 100k, BETA-STRIPPED, live-breach (the conservative reading).")
print("EV at $77.10. 'none' = the uncapped always-long policy measured earlier.\n")
CAPS = [None, 6000, 5000, 4000, 3000, 2500, 2000, 1750, 1500]
hdr = " ".join(f"{('none' if x is None else f'${x}'):>7}" for x in CAPS)
for n in (4, 6, 10, 20):
    print(f"n={n} micros      {hdr}")
    for tag, lo, hi in (("dev", 0, H), ("val", H, N-520), ("all", 0, N-520)):
        row = [np.mean([run(s, n, rd, hd, ld, x) for s in range(lo, hi, 3)]) - 77.10
               for x in CAPS]
        print(f"  {tag:>3}           " + " ".join(f"{v:>7,.0f}" for v in row))
    print()
