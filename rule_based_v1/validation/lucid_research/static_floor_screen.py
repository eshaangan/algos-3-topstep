import numpy as np, pandas as pd
ET="America/New_York"; PV=2.0
BOOK={1:1.0,2:1.0,4:1.0,6:1.0,9:1.111,13:1.269,20:1.525,40:2.188,60:2.792}
XS=np.array(sorted(BOOK)); YS=np.array([BOOK[k] for k in sorted(BOOK)])
def rt(n): return (float(np.interp(n,XS,YS))+1.00)*n
b=pd.read_parquet("data/processed/mnq_1m_all.parquet").tz_convert(ET).sort_index()
b=b[~b.index.duplicated(keep="last")]
mm=b.index.hour*60+b.index.minute; b=b.assign(sid=(mm>=18*60).cumsum(),mins=mm)
RET=[];LOW=[]
for _,g in b.groupby("sid"):
    g=g[(g["mins"]<=16*60+45)|(g["mins"]>=18*60)]
    if len(g)<200: continue
    o=g["open"].iloc[0]
    RET.append((g["close"].iloc[-1]-o)*PV); LOW.append((g["low"].min()-o)*PV)
RET=np.array(RET); LOW=np.array(LOW); N=len(RET)

def walk(ret,low,i0,start,dd,target,n,trailing,lock,harvest,min_days,min_daily,cap):
    """Returns (outcome, withdrawn, sessions_used). Live intraday breach."""
    c=rt(n); bal=start; floor=start-dd; qual=0; bank=0.0; i=i0
    while i < len(ret):
        pnl=ret[i]*n-c; worst=min(low[i]*n-c,pnl); i+=1
        if bal+worst<=floor: return("bust",bank,i-i0)
        bal+=pnl
        if not harvest and bal>=target: return("pass",bank,i-i0)
        if harvest:
            if pnl>=min_daily: qual+=1
            if qual>=min_days and bal>start:
                amt=bal-start
                if cap: amt=min(amt,cap)
                bank+=amt; bal-=amt; qual=0
        if trailing: floor=max(floor,min(bal-dd,lock))
    return("timeout",bank,i-i0)

def run(ret,low,size,tgt,dd,n,trailing,split,lock,fee,pay):
    pas=0;tot=0;wd=[]
    for s in range(0,N-500,5):
        o,_,used=walk(ret,low,s,size,dd,size+tgt,n,trailing,lock,False,0,0,None)
        tot+=1
        if o!="pass": continue
        pas+=1
        _,w,_=walk(ret,low,s+used,size,dd,None,n,trailing,lock,True,**pay)
        wd.append(w*split)
    p=pas/tot; e=float(np.mean(wd)) if wd else 0.0
    return p,e,split*dd,p*e-fee

mu=RET.mean()
CASES=[("Lucid 100k TRAIL",100_000,6_000,3_000,True,0.90,100_100,158,
        dict(min_days=5,min_daily=200.0,cap=2_500.0)),
       ("ETF 50k STATIC",50_000,4_000,2_000,False,0.80,None,150,
        dict(min_days=5,min_daily=100.0,cap=None))]
print(f"{'plan':>18} {'n':>3} {'beta':>4} {'P(pass)':>8} {'E[cash]':>8} {'bound':>6} {'eff':>5} {'EV':>7}")
print("-"*66)
for label,size,tgt,dd,tr,sp,lock,fee,pay in CASES:
    for tag,r_,l_ in (("on",RET,LOW),("off",RET-mu,LOW-mu)):
        for n in (4,10,20):
            p,e,bnd,ev=run(r_,l_,size,tgt,dd,n,tr,sp,lock,fee,pay)
            print(f"{label:>18} {n:>3} {tag:>4} {p:>8.1%} {e:>8,.0f} {bnd:>6,.0f} {e/bnd:>5.2f} {ev:>7,.0f}")

# --- how long does a campaign last? monthly billing is the open risk -----------
print("\nETF 50k STATIC, beta-stripped, sessions to outcome (21 sessions ~ 1 month):")
rd,ld=RET-mu,LOW-mu
for n in (4,10,20):
    ev_s=[];fu_s=[];tot=[]
    for s in range(0,N-500,5):
        o,_,u1=walk(rd,ld,s,50_000,2_000,54_000,n,False,None,False,0,0,None)
        ev_s.append(u1)
        if o=="pass":
            _,_,u2=walk(rd,ld,s+u1,50_000,2_000,None,n,False,None,True,5,100.0,None)
            fu_s.append(u2); tot.append(u1+u2)
    print(f"  n={n:>2}  eval med {np.median(ev_s):>5.0f}  funded med "
          f"{np.median(fu_s) if fu_s else 0:>5.0f}  total(pass) med "
          f"{np.median(tot) if tot else 0:>5.0f}  -> ~{(np.median(tot) if tot else 0)/21:.1f} months")
