import json, numpy as np
TARGET=6000.0; MLL=3000.0; FOMC_PER_YR=8.0
rng=np.random.default_rng(7)
L=json.load(open("/Users/jg/auction/runs/legs_2mic_600.json"))
wk=np.array([r["pnl"] for r in L["weekend"]]); fo=np.array([r["pnl"] for r in L["fomc"]])
print(f"weekend n={len(wk)} mean=${wk.mean():.2f} best=${wk.max():.2f} ({100*wk.max()/TARGET:.1f}% of target)")
print(f"fomc    n={len(fo)} mean=${fo.mean():.2f} best=${fo.max():.2f} ({100*fo.max()/TARGET:.1f}% of target)")

def sim(weeks, mode, n=40000):
    """mode: none | gross (denom = sum of winning days) | net (denom = net profit)"""
    passed=np.zeros(n,bool); busted=np.zeros(n,bool); blocked=np.zeros(n,bool)
    wtp=np.full(n,np.nan); p_fomc=FOMC_PER_YR/52.0
    for i in range(n):
        eq=0.0; peak=0.0; floor=-MLL; days=[]; ever_blocked=False
        for w in range(weeks):
            trades=[wk[rng.integers(len(wk))]]
            if rng.random()<p_fomc: trades.append(fo[rng.integers(len(fo))])
            dead=False
            for t in trades:
                eq+=t; days.append(t)
                if eq<=floor: busted[i]=True; dead=True; break
            if dead: break
            peak=max(peak,eq); floor=max(floor,min(peak-MLL,0.0))
            if eq>=TARGET:
                prof=[x for x in days if x>0]
                if mode=="none": ok=True
                elif mode=="gross": ok=(not prof) or max(prof)<=0.5*sum(prof)
                else:              ok=(not prof) or max(prof)<=0.5*eq
                if ok: passed[i]=True; wtp[i]=w+1; break
                ever_blocked=True
        if ever_blocked and not passed[i]: blocked[i]=True
    return (passed.mean(), busted.mean(), blocked.mean(),
            float(np.nanmedian(wtp)) if passed.any() else None)

print()
print(f"{chr(39)}horizon{chr(39):>0}".replace(chr(39),""), end="")
print(f"{'horizon':>8} {'rule':>7} {'p_pass':>8} {'bust':>7} {'blocked':>8} {'median wk':>10}")
for weeks in (16,32,52):
    for mode in ("none","gross","net"):
        p,b,bl,m=sim(weeks,mode)
        print(f"{weeks:>8} {mode:>7} {p:>8.3f} {b:>7.3f} {bl:>8.3f} {str(m):>10}")

# --- robustness: my rebuilt weekend leg means $161.85/wk but the frozen spec
# --- means $253.20 (ledger: $126.60/micro stopped x 2). A HIGHER mean reaches
# --- the target in FEWER trades = MORE concentration, so re-test at the higher
# --- mean by shifting the level and leaving dispersion/tail untouched.
print()
print("ROBUSTNESS at the frozen-spec mean ($253.20/wk weekend):")
wk = wk + (253.20 - wk.mean())
print(f"  shifted weekend: mean=${wk.mean():.2f} best=${wk.max():.2f} worst=${wk.min():.2f}")
print(f"{chr(32)*8}{chr(39)}{chr(39)}", end="")
print(f"\n{'horizon':>8} {'rule':>7} {'p_pass':>8} {'bust':>7} {'blocked':>8} {'median wk':>10}")
for weeks in (16,32):
    for mode in ("none","net"):
        p,b,bl,m=sim(weeks,mode)
        print(f"{weeks:>8} {mode:>7} {p:>8.3f} {b:>7.3f} {bl:>8.3f} {str(m):>10}")
