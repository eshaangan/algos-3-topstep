"""Is there anything here beyond 2020-2026 bull beta?

The martingale bound is exact: E[total withdrawn] = 100,000 - E[balance at
absorption]. The MLL LOCKS at 100,100, i.e. ABOVE the starting balance. So in a
genuinely fair game the funded account's expected extraction is ~ -$100, not $3,000.
Any positive number my earlier runs produced must therefore be drift, not structure.

Test: corrected mechanics, direction RANDOMISED (true fair game), vs real direction.
"""
import numpy as np
exec(open("/tmp/lucid_game.py").read().split('print("=== PHASE 1')[0])
exec(open("/tmp/lucid_v2.py").read().split('MLL0,')[1].split('for intraday')[0].join(
     ["MLL0,",""]))

days=np.array([s["day"] for s in sess]); yrs=np.array([d.year for d in days])

def series(n, sgn=None, dollars=200.0):
    """(eval pnl, eval running-min, funded pnl, funded running-min) for size n."""
    c=rt(n); ep=[];em=[];fp=[];fm=[]
    for i,s in enumerate(sess):
        d = 1 if sgn is None else sgn[i]
        p=s["path"]*n*d
        ep.append(p[-1]-c); em.append(p.min()-c)
        j=np.argmax(p>=dollars+c)
        v=(dollars+c if p[j]>=dollars+c else p[-1])-c
        fp.append(v); fm.append(min(p[:j+1].min()-c, v) if p[j]>=dollars+c else p.min()-c)
    return map(np.array,(ep,em,fp,fm))

def ev(n, sgn, intraday, mask=None):
    ep,em,fp,fm=series(n,sgn)
    if mask is not None: ep,em,fp,fm=ep[mask],em[mask],fp[mask],fm[mask]
    e=[x[0] for x in eval_run(ep,em,intraday)]; pp=e.count("pass")/max(len(e),1)
    g=np.array([x[0] for x in funded_run(fp,fm,intraday,True)]).mean()
    return pp,g,pp*g

rng=np.random.default_rng(0)
print(f"{'scenario':>26} {'sz':>3} {'P(pass)':>8} {'E[cash]':>9} {'grossEV':>8} {'EV@158':>8}")
for intraday in (False,True):
    print(f"\n--- breach on {'LIVE INTRADAY equity' if intraday else 'CLOSING balance'} ---")
    for n in (4,20,60):
        pp,g,x=ev(n,None,intraday)
        print(f"{'real direction, 2020-26':>26} {n:>3} {pp:>8.1%} {g:>9.0f} {x:>8.0f} {x-158:>8.0f}")
    for n in (4,20,60):
        xs=[];ps=[];gs=[]
        for sd in range(3):
            sgn=np.random.default_rng(sd).choice([-1,1],len(sess))
            pp,g,x=ev(n,sgn,intraday); xs.append(x);ps.append(pp);gs.append(g)
        print(f"{'BETA STRIPPED (fair game)':>26} {n:>3} {np.mean(ps):>8.1%} "
              f"{np.mean(gs):>9.0f} {np.mean(xs):>8.0f} {np.mean(xs)-158:>8.0f}")
    m=(yrs==2022)
    for n in (4,20):
        pp,g,x=ev(n,None,intraday,m)
        print(f"{'2022 bear, real direction':>26} {n:>3} {pp:>8.1%} {g:>9.0f} {x:>8.0f} {x-158:>8.0f}")
