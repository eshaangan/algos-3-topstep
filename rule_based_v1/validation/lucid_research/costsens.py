"""Two questions, one run.

Q1 DIRECTION. How accurate would a session-direction signal have to be to matter?
   drift = (2p-1) * E|session return|. Calibrate against beta's +$17.78/micro.

Q2 COST. The funded phase burns one round turn per session at 60 micros, which the
   measured curve puts at $252. Six NQ minis are the SAME $120/point exposure at
   $1.75/side instead of $5.00/side-equivalent. How much is that switch worth?
"""
import numpy as np
exec(open("/tmp/dir_final.py").read().split('"""',2)[2].split("for lab,pth,pv in")[0])

S=build("/Users/jg/auction/data/mnq_1m_all.parquet",2.0)
r=np.array([s["ret"] for s in S])
print(f"MNQ {len(S)} sessions.  mean {r.mean():+.2f}  sd {r.std():.1f}  "
      f"E|r| {np.abs(r).mean():.1f}  $/micro\n")

print("Q1: what session-direction accuracy would be worth having?")
print(f"{'accuracy':>9} {'drift $/micro/sess':>19} {'vs beta (+17.78)':>18}")
for p in (0.50,0.51,0.52,0.55,0.60):
    d=(2*p-1)*np.abs(r).mean()
    print(f"{p:>9.0%} {d:>19.2f} {d/17.78:>17.1f}x")
print("  beta alone is equivalent to a 52.5% accurate direction call.")
print("  tonight's 5 rules -- incl. 252-day momentum, the most replicated direction")
print("  anomaly in finance -- all scored BELOW always-long. A book-state signal")
print("  would have to beat that over a 22-HOUR horizon, 13x longer than the")
print("  longest horizon the L3 programme ever found anything at.\n")

def funded(pnl_cost, n=60):
    """E[cash] in the funded phase with an arbitrary round-turn cost."""
    fp=[]
    for s in S:
        p=s["path"]*n
        j=int(np.argmax(p>=200.+pnl_cost))
        fp.append((200.+pnl_cost if p[j]>=200.+pnl_cost else p[-1])-pnl_cost)
    fp=np.array(fp); N=len(fp); out=[]
    for st in range(0,N-60):
        bal=peak=100000.; cash=0.; good=0; po=0; lk=False; last=100000.
        for k in range(st,min(st+500,N)):
            bal+=fp[k]
            if fp[k]>=200.: good+=1
            peak=max(peak,bal)
            if bal<=(LOCK if lk else min(peak-3000.,LOCK)): break
            if good>=5 and bal>last and bal>=TRAIL:
                req=min(.5*(bal-100000.),2500.)
                if req>=500:
                    bal-=req; cash+=.9*req; po+=1; good=0; last=bal; lk=True
                    if po>=5: break
        out.append(cash)
    return float(np.mean(out))

print("Q2: funded-phase E[cash] vs round-turn cost at $120/point exposure")
print(f"{'route':>34} {'$/RT':>7} {'E[cash]':>9} {'EV @ P(pass)=36.1%':>20}")
for lab,c in (("60 MNQ micros (measured curve)",252.3),
              ("if the size column is 1.65x off",416.3),
              ("6 NQ minis, spread as micros",187.3),
              ("6 NQ minis, commission+1 tick",51.0),
              ("frictionless bound",0.0)):
    e=funded(c); print(f"{lab:>34} {c:>7.0f} {e:>9.0f} {0.361*e-158:>20.0f}")
