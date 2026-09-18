"""Three refinements to the validated +$57 policy.

1. THE THRESHOLD SHOULD APPLY TO THE FIRST PAYOUT ONLY. The floor-jump penalty
   fires once -- taking a payout while the floor still trails moves it to 100,100.
   After that the floor IS 100,100 and further payouts are free. My model applied
   the patience threshold to EVERY payout, needlessly starving later cycles.

2. THE LEG-SWITCH POINT. I switched from bold to small exactly when the floor
   locked. There is no reason that is optimal; switching on balance directly lets
   the boundary move either side of it.

3. CONTINUOUS SIZING. Two discrete legs is a crude approximation to "bet in
   proportion to the room you have". Size = round(k * (balance - floor) / 1000)
   is the natural generalisation and needs no regime boundary at all.

Beta-stripped, live breach, measured cost, floor-jump ON, dev/val split.
"""
import sys, numpy as np
sys.path.insert(0, "rule_based_v1/validation/lucid_research")
from account import RET, LOW, N, rt

START, TARGET, MLL, LOCK, FEE = 100_000., 106_000., 3_000., 100_100., 158.
SZ = np.array([1,2,3,4,6,8,10,13,16,20,25,30,40,50,60])
def snap(x): return int(SZ[np.argmin(np.abs(SZ - np.clip(x, 1, 60)))])

def evaluate(i0, ret, low, mode, p, max_s=400):
    bal = START; floor = START - MLL
    for k in range(max_s):
        i = i0 + k
        if i >= len(ret): return None, k
        if mode == "two_leg":
            n_lo, n_hi, switch = p
            n = n_lo if bal >= switch else n_hi
        else:                                  # proportional to room
            n = snap(p * (bal - floor) / 1000.0)
        c = rt(n); pnl = ret[i]*n - c; worst = min(low[i]*n - c, pnl)
        if bal + worst <= floor: return False, k+1
        bal += pnl
        if bal >= TARGET: return True, k+1
        floor = max(floor, min(bal - MLL, LOCK))
    return None, max_s

def funded(i0, n, ret, low, W, first_only, max_s=400):
    c = rt(n); bal = START; floor = START - MLL; q = 0; bank = 0.0; paid = False
    for k in range(max_s):
        i = i0 + k
        if i >= len(ret): break
        pnl = ret[i]*n - c; worst = min(low[i]*n - c, pnl)
        if bal + worst <= floor: return bank
        bal += pnl
        if pnl >= 200.0: q += 1
        gate = W if (not paid or not first_only) else 100_200.
        if q >= 5 and bal >= gate and bal > START:
            take = min(bal - START, 2500.0)
            if take >= 200:
                bank += take; bal -= take; q = 0; paid = True
                if floor < LOCK: floor = LOCK
        floor = max(floor, min(bal - MLL, LOCK))
    return bank

mu = RET.mean(); rd = RET-mu; ld = LOW-mu; H = N//2
def run(mode, p, nf, W, first_only):
    out = {}
    for tag, a, b in (("dev",0,H), ("val",H,N-420), ("all",0,N-420)):
        cash, ps = [], []
        for s in range(a, b, 3):
            ok, used = evaluate(s, rd, ld, mode, p)
            if ok is None: continue
            ps.append(ok)
            cash.append(funded(s+used, nf, rd, ld, W, first_only)*0.90 if ok else 0.0)
        out[tag] = (np.mean(ps), np.mean(cash)-FEE)
    return out

print("Refinements to the validated +$57 policy (20->4 eval, funded 2, W=102k).\n")
print(f"{'variant':>44} {'P(pass)':>8} {'dev':>7} {'val':>7} {'all':>7}")
def show(lab, r):
    print(f"{lab:>44} {r['all'][0]:>8.1%} {r['dev'][1]:>7,.0f} "
          f"{r['val'][1]:>7,.0f} {r['all'][1]:>7,.0f}")

show("VALIDATED BASE (W on every payout)", run("two_leg",(4,20,103_100),2,102_000,False))
show("  + threshold on FIRST payout only", run("two_leg",(4,20,103_100),2,102_000,True))
for sw in (101_500, 102_500, 103_100, 104_000, 105_000):
    show(f"  switch to small at {sw:,}", run("two_leg",(4,20,sw),2,102_000,True))
for k in (2, 4, 6, 8, 12):
    show(f"  CONTINUOUS size = {k} per $1k of room", run("prop",k,2,102_000,True))
