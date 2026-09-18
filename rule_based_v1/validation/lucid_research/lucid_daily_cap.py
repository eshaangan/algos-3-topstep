"""Daily profit cap on the LUCID FLEX 100k you already own.

Different mechanism from the ETF test. Lucid breaches on LIVE intraday equity, so
the moment you take profit and go flat you CANNOT breach for the rest of that
session. A profit target therefore buys real protection -- unlike the 20 time-based
exits already tested and killed, which cut winners and losers indiscriminately.

Path-independent: with a target and no stop you fill at X iff the session high
reached X.

  start 100,000  target 106,000  MLL 3,000 trailing on the CLOSE, locks at 100,100
  breach on LIVE equity (confirmed by the firm)
  funded: 5 qualifying days >= $200, payout cap $2,500, split 0.90

Pre-registered: adopt only if it beats the uncapped policy in BOTH eras.
"""
import sys, numpy as np
sys.path.insert(0, "rule_based_v1/validation/lucid_research")
from account import RET, LOW, HIGH, N, rt

START, TARGET, MLL, LOCK = 100_000., 106_000., 3_000., 100_100.

def day(i, n, c, ret, high, low, cap):
    """(pnl, worst) for one session, long, with an optional profit cap."""
    if cap and high[i]*n >= cap:
        pnl = cap - c
        # flat after the target: the adverse excursion can only precede the fill
        worst = min(low[i]*n - c, pnl) if low[i]*n < 0 else pnl
        return pnl, worst
    pnl = ret[i]*n - c
    return pnl, min(low[i]*n - c, pnl)

def evaluate(i0, n, ret, high, low, cap, max_s=400):
    c = rt(n); bal = START; floor = START - MLL
    for k in range(max_s):
        i = i0 + k
        if i >= len(ret): return None, 0
        pnl, worst = day(i, n, c, ret, high, low, cap)
        if bal + worst <= floor: return False, k
        bal += pnl
        if bal >= TARGET: return True, k + 1   # k+1 sessions consumed; funded must start AFTER the winning session
        floor = max(floor, min(bal - MLL, LOCK))
    return None, max_s

def funded(i0, n, ret, high, low, cap, max_s=400):
    c = rt(n); bal = START; floor = START - MLL; q = 0; bank = 0.0
    for k in range(max_s):
        i = i0 + k
        if i >= len(ret): break
        pnl, worst = day(i, n, c, ret, high, low, cap)
        if bal + worst <= floor: return bank
        bal += pnl
        if pnl >= 200.0: q += 1
        if q >= 5 and bal > START:
            take = min(bal - START, 2_500.0)
            if take >= 200: bank += take; bal -= take; q = 0
        floor = max(floor, min(bal - MLL, LOCK))
    return bank

mu = RET.mean(); rd = RET-mu; hd = HIGH-mu; ld = LOW-mu; H = N//2
print("LUCID FLEX 100k (the account you own), BETA-STRIPPED. EV at the $158 fee.\n")
CAPS = [None, 3000, 2000, 1500, 1000, 750, 500, 400, 300]
hdr = " ".join(f"{('none' if x is None else f'${x}'):>7}" for x in CAPS)
for n in (2, 4, 6, 10):
    print(f"n={n} micros      {hdr}")
    for tag, lo, hi in (("dev", 0, H), ("val", H, N-420), ("all", 0, N-420)):
        pr, ev = [], []
        for x in CAPS:
            ps, cash = [], []
            for s in range(lo, hi, 3):
                ok, used = evaluate(s, n, rd, hd, ld, x)
                if ok is None: continue
                ps.append(ok)
                cash.append(funded(s+used, n, rd, hd, ld, x)*0.90 if ok else 0.0)
            pr.append(np.mean(ps) if ps else 0)
            ev.append((np.mean(cash) if cash else 0) - 158)
        print(f"  {tag:>3} P(pass)  " + " ".join(f"{v:>6.1%} " for v in pr))
        print(f"  {tag:>3} EV       " + " ".join(f"{v:>7,.0f}" for v in ev))
    print()
