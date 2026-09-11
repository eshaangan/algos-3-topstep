"""PRE-REGISTERED. Optimise the LucidFlex 100k under the CONFIRMED mechanics:
breach on LIVE equity, floor trails on closing balances, locks at 100,100.

MECHANISM. Under a close-only breach, how long you hold is irrelevant to survival --
only the close can kill you. Under a LIVE breach it is the dominant variable, because
death is caused by the intraday EXCURSION, and excursion grows with holding time
roughly as sqrt(t) while the drag per session stays fixed at one round turn.

So shortening the hold trades two opposing effects:
  + fewer spurious deaths (touching the floor on a session that would have closed fine)
  - more sessions to cover the same distance, hence more total drag
That trade was invisible under the old assumption. It is now the main lever.

A hard stop above the floor is NOT tested: optional stopping says it cannot change
P(pass) in a fair game, and it adds round turns, so it is strictly worse.

TRIALS: 5 hold windows x 4 sizes = 20. Costs are the MEASURED RTH curve.
BASELINE: 4 micros, full session, the best live-breach cell so far (EV +$20).
DECISION: adopt only if it beats the baseline in BOTH eras.
"""
import numpy as np, pandas as pd
ET="America/New_York"; PV=2.0; LOCK=100100.; TRAIL=103100.
# measured book cost $/contract (RTH, 15 days, 407M events) + Lucid $0.50/side
BOOK={1:1.000,2:1.000,4:1.000,6:1.000,9:1.111,13:1.269,20:1.525,40:2.188,60:2.792}
XS=np.array(sorted(BOOK)); YS=np.array([BOOK[k] for k in sorted(BOOK)])
def rt(n): return (float(np.interp(n,XS,YS))+1.00)*n

b=pd.read_parquet("/Users/jg/auction/data/mnq_1m_all.parquet")
if "et" in b.columns: b=b.set_index(pd.DatetimeIndex(b["et"])).drop(columns=["et"])
b=b.tz_convert(ET).sort_index(); b=b[~b.index.duplicated(keep="last")]
m=b.index.hour*60+b.index.minute
b=b.assign(sid=(m>=18*60).cumsum(), mins=m)

WINDOWS=[("09:30 +1h",570,60),("09:30 +2h",570,120),("09:30 +4h",570,240),
         ("09:30 +6.5h",570,390),("18:00 full",1080,None)]
S={w[0]:[] for w in WINDOWS}; days=[]
for sid,g in b.groupby("sid"):
    gg=g[(g["mins"]<=16*60+45)|(g["mins"]>=18*60)]
    if len(gg)<200: continue
    days.append(gg.index[-1].date())
    for lab,start,dur in WINDOWS:
        if dur is None: w=gg
        else:
            w=gg[(gg["mins"]>=start)&(gg["mins"]<start+dur)]
        if len(w)<10: S[lab].append(None); continue
        o=w["open"].iloc[0]; c=w["close"].to_numpy()
        S[lab].append(((c[-1]-o)*PV, (w["low"].min()-o)*PV, (c-o)*PV))
days=np.array(days); yr=np.array([d.year for d in days])

def series(lab,n,tgt=None):
    c=rt(n); out=[]
    for e in S[lab]:
        if e is None: out.append((0.0,0.0)); continue
        ret,lo,path=e
        if tgt is None: out.append((ret*n-c, lo*n-c))
        else:
            p=path*n; j=int(np.argmax(p>=tgt+c))
            hit=p[j]>=tgt+c
            out.append(((tgt+c if hit else p[-1])-c,
                        (min(p[:j+1].min(),0.0) if hit else lo*n)-c))
    a=np.array(out); return a[:,0],a[:,1]

def run(pnl,lo,mask,funded):
    idx=np.where(mask)[0]; P=[];C=[]
    for si in range(0,len(idx)-60):
        bal=peak=100000.; dmax=0.; cash=0.; good=0; po=0; lk=False; last=100000.; res=0
        for k in idx[si:si+500]:
            mll=LOCK if lk else min(peak-3000.,LOCK)
            if bal+lo[k]<=mll: break                    # LIVE breach on the excursion
            bal+=pnl[k]
            if funded and pnl[k]>=200.: good+=1
            if pnl[k]>dmax: dmax=pnl[k]
            peak=max(peak,bal); mll=LOCK if lk else min(peak-3000.,LOCK)
            if bal<=mll: break
            if funded:
                if good>=5 and bal>last and bal>=TRAIL:
                    r=min(.5*(bal-100000.),2500.)
                    if r>=500: bal-=r; cash+=.9*r; po+=1; good=0; last=bal; lk=True
                    if po>=5: break
            else:
                pr=bal-100000.
                if pr>=6000. and dmax<=.5*pr: res=1; break
        P.append(res); C.append(cash)
    return float(np.mean(P)), float(np.mean(C))

for era,mk in (("dev<=2022",yr<=2022),("val>=2023",yr>=2023)):
    print(f"\n=== {era} ===")
    print(f"{'window':>12} {'sz':>3} {'$/RT':>6} {'P(pass)':>8} {'E[cash]':>9} {'EV@158':>8}")
    for lab,_,_ in WINDOWS:
        for n in (4,10,20,40):
            ep,el=series(lab,n); fp,fl=series(lab,n,tgt=200.)
            pp,_=run(ep,el,mk,False); _,cash=run(fp,fl,mk,True)
            print(f"{lab:>12} {n:>3} {rt(n):>6.0f} {pp:>8.1%} {cash:>9.0f} "
                  f"{pp*cash-158:>8.0f}")
