"""Measure the Elite Trader Funding DTF 100k, the top row of the screen.

Its screen score (+$2,708, 35x per dollar) uses an EXTRAPOLATED multiplier and
ignores the payout friction entirely. This prices it on the real tape with the
firm's actual rules:

  no evaluation, start 100,000, STATIC floor 95,000
  breach on the CLOSING balance, not intraday      <- the mechanic Lucid lacked
  safety net: +$5,100 REALISED before any withdrawal, not itself withdrawable
  20 qualifying days per cycle, each >= $500 AND >= 50% of the best day
  $25,000 lifetime cap, split up to 100%
"""
import sys, numpy as np
sys.path.insert(0, "rule_based_v1/validation/lucid_research")
from account import RET, LOW, N, rt

START, FLOOR, NET = 100_000., 95_000., 5_100.
ATD_N, ATD_MIN, ATD_CONS, CAP = 20, 500.0, 0.50, 25_000.

def run(i0, n, ret, max_s=500):
    """Returns (reached_safety_net, cash_withdrawn)."""
    c = rt(n); bal = START; days = []; bank = 0.0; safe = False
    for k in range(max_s):
        i = i0 + k
        if i >= len(ret): break
        pnl = ret[i]*n - c
        bal += pnl
        if bal <= FLOOR: return safe, bank          # breach on the CLOSE only
        days.append(pnl)
        if not safe and bal - START >= NET: safe = True
        if safe:
            # a cycle pays when 20 days clear $500 AND 50% of the best of them
            good = [d for d in days if d >= ATD_MIN]
            if len(good) >= ATD_N:
                best = max(good)
                q = [d for d in good if d >= ATD_CONS*best]
                if len(q) >= ATD_N:
                    avail = bal - START - NET
                    if avail >= 500:
                        take = min(avail, CAP - bank)
                        bank += take; bal -= take; days = []
                        if bank >= CAP: return safe, bank
    return safe, bank

mu = RET.mean()
print("ETF DTF 100k, always long, measured. 'beta off' = session returns demeaned.\n")
print(f"{'micros':>7} {'beta':>5} {'reach net':>10} {'E[cash]':>9} {'bound':>7} "
      f"{'EV@77':>8} {'EV@257':>8}")
for tag, r_ in (("on", RET), ("off", RET-mu)):
    for n in (2, 4, 10, 20, 40):
        res = [run(s, n, r_) for s in range(0, N-520, 3)]
        reach = np.mean([a for a, _ in res]); cash = np.mean([b for _, b in res])
        print(f"{n:>7} {tag:>5} {reach:>10.1%} {cash:>9,.0f} {5_000:>7,} "
              f"{cash-77.10:>8,.0f} {cash-257:>8,.0f}")
print("\nNOTE: EV here is E[cash] - fee directly; there is no separate P(pass)")
print("because DTF has no evaluation phase. The bound on E[cash] is the gifted")
print("drawdown, $5,000. Anything above that is beta, not mechanism.")
