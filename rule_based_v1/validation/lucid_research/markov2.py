"""Exact first passage, corrected.

Two bugs in v1, both silent:
  1. Floors were dict keys of type float, looked up with a recomputed float. Most
     lookups MISSED and `continue` then discarded that probability mass entirely --
     it became neither pass nor fail. That biased P(pass) DOWN, hardest at small
     sizes where the floor ratchets in many small steps. Fixed by indexing floors
     by integer.
  2. The unlocked band was binned as [0, 3000) in 30 bins, so r = 3000 -- which is
     the STARTING state, balance 100,000 on a 97,000 floor -- fell outside the grid.
     Fixed by putting grid points AT j*step, j = 1..30.
"""
import numpy as np, pandas as pd

ET="America/New_York"; PV=2.0; LOCK=100100.; MLL=3000.; TARGET=106000.; START=100000.
BOOK={1:1.0,2:1.0,4:1.0,6:1.0,9:1.111,13:1.269,20:1.525,40:2.188,60:2.792}
XS=np.array(sorted(BOOK)); YS=np.array([BOOK[k] for k in sorted(BOOK)])
def rt(n): return (float(np.interp(n,XS,YS))+1.00)*n

b=pd.read_parquet("/Users/jg/auction/data/mnq_1m_all.parquet")
if "et" in b.columns: b=b.set_index(pd.DatetimeIndex(b["et"])).drop(columns=["et"])
b=b.tz_convert(ET).sort_index(); b=b[~b.index.duplicated(keep="last")]
mm_=b.index.hour*60+b.index.minute
b=b.assign(sid=(mm_>=18*60).cumsum(), mins=mm_)
R=[];M=[];RNG=[]
for sid,g in b.groupby("sid"):
    g=g[(g["mins"]<=16*60+45)|(g["mins"]>=18*60)]
    if len(g)<200: continue
    o=g["open"].iloc[0]
    R.append((g["close"].iloc[-1]-o)*PV); M.append((g["low"].min()-o)*PV)
    RNG.append((g["high"].max()-g["low"].min())*PV)
R=np.array(R); M=np.array(M); RNG=np.array(RNG)
hi=RNG>np.median(RNG); v=hi.astype(int)
P=np.zeros((2,2))
for a in (0,1):
    for c in (0,1): P[a,c]=np.sum((v[:-1]==a)&(v[1:]==c))
P/=P.sum(1,keepdims=True)
print(f"{len(R)} sessions.  vol persistence P(lo|lo)={P[0,0]:.3f} P(hi|hi)={P[1,1]:.3f}")

STEP=100.0
def solve(n, trailing=True):
    c=rt(n); ret=R*n-c; mn=np.minimum(M*n-c, ret)
    F=int(round((LOCK-(START-MLL))/STEP))+1          # 32 floor levels
    fl=np.array([(START-MLL)+k*STEP for k in range(F)])
    NB={k:(int(MLL/STEP) if k<F-1 else int((TARGET-LOCK)/STEP)) for k in range(F)}
    if not trailing: F=1; fl=np.array([START-MLL]); NB={0:int((TARGET-fl[0])/STEP)}
    val={}
    for k in range(F-1,-1,-1):
        f=fl[k]; nb=NB[k]; rs=(np.arange(nb)+1)*STEP
        A=np.zeros((nb*2,nb*2)); rhs=np.zeros(nb*2)
        for vi in (0,1):
            sel=v==vi; rr=ret[sel]; mo=mn[sel]; w=1.0/len(rr)
            for i,r0 in enumerate(rs):
                row=i*2+vi; A[row,row]+=1.0
                alive=(r0+mo)>0
                r1=r0+rr; bal=f+r1
                for vj in (0,1):
                    pv=P[vi,vj]*w
                    ok=alive&(r1>0)
                    win=ok&(bal>=TARGET); rhs[row]+=pv*win.sum()
                    go=ok&~win
                    if not go.any(): continue
                    bb=bal[go]
                    nf=np.maximum(f,np.minimum(bb-MLL,LOCK)) if trailing else np.full(bb.shape,f)
                    kk=np.clip(np.rint((nf-fl[0])/STEP).astype(int),k,F-1)
                    rr2=bb-fl[kk]
                    for kp in np.unique(kk):
                        s2=kk==kp
                        jj=np.clip(np.rint(rr2[s2]/STEP).astype(int)-1,0,NB[kp]-1)
                        if kp==k:
                            np.add.at(A,(row,jj*2+vj),-pv)
                        else:
                            rhs[row]+=pv*val[kp][jj*2+vj].sum()
        val[k]=np.linalg.solve(A,rhs)
    j=int(round((START-fl[0])/STEP))-1
    p0=float(np.mean(~hi))
    return p0*val[0][j*2]+(1-p0)*val[0][j*2+1]

print(f"\n{'size':>5} {'$/RT':>6} {'EXACT trailing':>15} {'simulated':>10} "
      f"{'EXACT static':>13} {'trailing costs':>15}")
for n,sim in ((4,0.228),(10,0.161),(20,0.116),(40,0.056)):
    t=solve(n,True); s=solve(n,False)
    print(f"{n:>5} {rt(n):>6.0f} {t:>15.1%} {sim:>10.1%} {s:>13.1%} "
          f"{s-t:>14.1f}pp")
print(f"\ntheory: static floor, zero cost => D/(T+D) = {3000/9000:.1%}")
