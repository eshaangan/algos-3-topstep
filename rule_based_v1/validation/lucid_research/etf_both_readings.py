"""ETF DTF 100k priced under BOTH readings of the unresolved breach rule.

The firm says "if your account ever closes at or below that fixed floor, the
evaluation ends" and never defines whether "closes" means the session close or
any touch. Lucid's identical ambiguity flipped that entire thesis, so this
prices both columns instead of picking one.

  CLOSE  breach only if the END-OF-DAY balance is at or below the floor
  LIVE   breach if intraday equity (unrealised included) ever touches it

If both columns are positive the unknown does not block a decision.
"""
import sys, numpy as np
sys.path.insert(0, "rule_based_v1/validation/lucid_research")
from account import RET, LOW, N, rt

START, FLOOR, NET = 100_000., 95_000., 5_100.
ATD_N, ATD_MIN, ATD_CONS, CAP = 20, 500.0, 0.50, 25_000.

def run(i0, n, ret, low, live, max_s=500):
    c = rt(n); bal = START; good = []; bank = 0.0; safe = False
    for k in range(max_s):
        i = i0 + k
        if i >= len(ret): break
        pnl = ret[i]*n - c
        if live and bal + min(low[i]*n - c, pnl) <= FLOOR:   # intraday touch
            return bank
        bal += pnl
        if bal <= FLOOR: return bank                          # close at/below floor
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

mu = RET.mean(); rd = RET - mu; ld = LOW - mu; H = N // 2
print("ETF DTF 100k, BETA-STRIPPED, always long. EV at the $77.10 promo price.")
print("A decision is blocked only if the two columns disagree in sign.\n")
print(f"{'micros':>7} {'era':>5} {'EV close-breach':>16} {'EV live-breach':>15}")
for n in (2, 4, 10, 20):
    for tag, lo, hi in (("dev", 0, H), ("val", H, N-520), ("all", 0, N-520)):
        rows = range(lo, hi, 3)
        if not len(rows): continue
        cl = np.mean([run(s, n, rd, ld, False) for s in rows]) - 77.10
        lv = np.mean([run(s, n, rd, ld, True)  for s in rows]) - 77.10
        print(f"{n:>7} {tag:>5} {cl:>16,.0f} {lv:>15,.0f}")
