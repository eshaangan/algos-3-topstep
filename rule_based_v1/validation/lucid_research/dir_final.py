"""Lagged direction rules on the Lucid barrier game. Same pre-registration as
direction.py; the only change is that every rule may use information only up to
the close of session i-1. Self-contained.

R1 long / R2 prev were already lagged correctly and act as controls.
"""
import numpy as np, pandas as pd

ET="America/New_York"
COST={1:2.047,2:2.101,4:2.191,6:2.269,9:2.388,13:2.558,20:2.844,40:3.562,60:4.205}
XS=np.array(sorted(COST)); YS=np.array([COST[k] for k in sorted(COST)])
def rt(n): return float(np.interp(n,XS,YS))*n
TRAIL,LOCK=103100.,100100.

def build(path,pv):
    b=pd.read_parquet(path)
    if "et" in b.columns: b=b.set_index(pd.DatetimeIndex(b["et"])).drop(columns=["et"])
    b=b.tz_convert(ET).sort_index(); b=b[~b.index.duplicated(keep="last")]
    m=b.index.hour*60+b.index.minute
    b=b.assign(sid=(m>=18*60).cumsum())
    S=[]
    for sid,g in b.groupby("sid"):
        g=g[(g.index.hour*60+g.index.minute<=16*60+45)|(g.index.hour>=18)]
        if len(g)<200: continue
        o=g["open"].iloc[0]; c=g["close"].iloc[-1]
        S.append(dict(day=g.index[-1].date(),close=c,ret=(c-o)*pv,
                      path=(g["close"].to_numpy()-o)*pv))
    return S

def directions(S):
    cl=pd.Series([s["close"] for s in S]); r=np.array([s["ret"] for s in S])
    lag=cl.shift(1)
    d={"R1 long":np.ones(len(S))}
    p=np.sign(np.r_[0.0,r[:-1]]); p[p==0]=1; d["R2 prev"]=p
    for n,lab in ((50,"R3 ma50"),(200,"R4 ma200")):
        v=np.sign((lag-lag.rolling(n).mean()).to_numpy())
        v[np.isnan(v)]=1; v[v==0]=1; d[lab]=v
    v=np.sign(lag.pct_change(252).to_numpy()); v[np.isnan(v)]=1; v[v==0]=1
    d["R5 tsmom252"]=v
    return d

def evalrun(pnl):
    N=len(pnl); out=[]
    for st in range(0,N-60):
        bal=peak=100000.; dmax=0.
        for k in range(st,min(st+500,N)):
            bal+=pnl[k]
            if pnl[k]>dmax: dmax=pnl[k]
            peak=max(peak,bal)
            if bal<=min(peak-3000.,LOCK): out.append(0); break
            pr=bal-100000.
            if pr>=6000. and (k-st)>=1 and dmax<=.5*pr: out.append(1); break
        else: out.append(0)
    return float(np.mean(out))

def fundedrun(pnl):
    N=len(pnl); out=[]
    for st in range(0,N-60):
        bal=peak=100000.; cash=0.; good=0; po=0; lk=False; last=100000.
        for k in range(st,min(st+500,N)):
            bal+=pnl[k]
            if pnl[k]>=200.: good+=1
            peak=max(peak,bal)
            if bal<=(LOCK if lk else min(peak-3000.,LOCK)): break
            if good>=5 and bal>last and bal>=TRAIL:
                req=min(.5*(bal-100000.),2500.)
                if req>=500:
                    bal-=req; cash+=.9*req; po+=1; good=0; last=bal; lk=True
                    if po>=5: break
        out.append(cash)
    return float(np.mean(out))

def price(S,sgn,n,mask):
    c=rt(n); ep=[];fp=[]
    for i,s in enumerate(S):
        p=s["path"]*n*sgn[i]
        ep.append(p[-1]-c)
        j=int(np.argmax(p>=200.+c))
        fp.append((200.+c if p[j]>=200.+c else p[-1])-c)
    return evalrun(np.array(ep)[mask]), fundedrun(np.array(fp)[mask])

for lab,pth,pv in (("MNQ","/Users/jg/auction/data/mnq_1m_all.parquet",2.0),
                   ("ES tape","/Users/jg/auction/data/es_1min_eth_frontmonth.parquet",5.0)):
    S=build(pth,pv); D=directions(S); r=np.array([s["ret"] for s in S])
    yr=np.array([s["day"].year for s in S])
    print(f"\n{'='*70}\n{lab}: {len(S)} sessions {S[0]['day']}..{S[-1]['day']}\n{'='*70}")
    print("  leakage audit  corr(direction, SAME-session return) -- must be ~0")
    for k,v in D.items():
        c=np.corrcoef(v,r)[0,1] if v.std()>0 else 0.0
        t=c*np.sqrt(len(r)-2)/np.sqrt(max(1e-12,1-c*c))
        print(f"    {k:>12}  corr {c:+.4f}  t {t:+6.2f}"+("  <-- LEAK" if abs(t)>4 else ""))
    for era,mk in (("dev 2020-22",yr<=2022),("val 2023-26",yr>=2023)):
        if mk.sum()<250: continue
        print(f"\n  {era}   {'rule':>12} {'sz':>3} {'P(pass)':>8} {'E[cash]':>9} "
              f"{'grossEV':>8} {'EV@158':>8}")
        for rule,sgn in D.items():
            for n in (4,60):
                pp,cash=price(S,sgn,n,mk)
                print(f"{'':>14} {rule:>12} {n:>3} {pp:>8.1%} {cash:>9.0f} "
                      f"{pp*cash:>8.0f} {pp*cash-158:>8.0f}")
