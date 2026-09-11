"""Beat Lucid, not the market.

Objective is NOT expectancy vs a $520/wk bar. It is:

    EV per evaluation purchased = P(pass) * E[cash extracted | funded] - fee

Both phases are FIRST PASSAGE on a gifted loss allowance you never repay.
Zero-edge play, contiguous historical windows only, measured cost model.

Rules (verified at support.lucidtrading.com, 2026-09-10):
  eval   100k: target +6,000, max loss 3,000 EOD-trailing, floor locks at
               100,000 once peak >= 103,000, 50% consistency, 2 min days,
               no time limit, one-time fee $225 (~$158 with code)
  funded 100k: no target, max loss 3,000 EOD-trailing + lock, 90/10 split,
               NO consistency, payout needs 5 separate days each >= $200
               profit (resets after each payout) and positive cycle net,
               payout = 50% of profit capped $2,500, min $500, max 5 payouts.
"""
import numpy as np, pandas as pd

ET="America/New_York"; PV=2.0
COST={1:2.047,2:2.101,4:2.191,6:2.269,9:2.388,13:2.558,20:2.844,40:3.562,60:4.205}
XS=np.array(sorted(COST)); YS=np.array([COST[k] for k in sorted(COST)])
def rt(n): return float(np.interp(n,XS,YS))*n      # $ per round turn, n micros

bars=pd.read_parquet("/Users/jg/auction/data/mnq_1m_all.parquet")
if "et" in bars.columns: bars=bars.set_index(pd.DatetimeIndex(bars["et"])).drop(columns=["et"])
bars=bars.tz_convert(ET).sort_index(); bars=bars[~bars.index.duplicated(keep="last")]

# ---- sessions: enter 18:00 ET, mandatory flatten 16:45 ET -------------------
m=bars.index.hour*60+bars.index.minute
sess_id=(m>=18*60).cumsum()          # 18:00 starts a new session
bars=bars.assign(sid=sess_id)
sess=[]
for sid,g in bars.groupby("sid"):
    g=g[(g.index.hour*60+g.index.minute<=16*60+45)|(g.index.hour>=18)]
    if len(g)<200: continue
    o=g["open"].iloc[0]
    sess.append(dict(day=g.index[-1].date(),
                     ret=(g["close"].iloc[-1]-o)*PV,          # $/micro, hold to flatten
                     path=(g["close"].to_numpy()-o)*PV))      # $/micro running
print(f"{len(sess)} sessions {sess[0]['day']} .. {sess[-1]['day']}")
rets=np.array([s["ret"] for s in sess])
print(f"session $/micro: mean {rets.mean():+.2f}  sd {rets.std():.2f}\n")

def hit_or_flatten(n, dollars):
    """P&L of 'hold long until +$dollars, else flatten', per session, n micros."""
    out=np.empty(len(sess)); c=rt(n)
    for i,s in enumerate(sess):
        p=s["path"]*n
        j=np.argmax(p>=dollars+c)
        out[i]=(dollars+c if p[j]>=dollars+c else p[-1])-c
    return out

def run(pnl_by_sess, target, floor0=97000.0, lock=100000.0, maxsess=500,
        payout_mode=False, min_day=200.0):
    """Contiguous-window first passage. Returns per-start outcomes."""
    N=len(pnl_by_sess); res=[]
    for st in range(0,N-60):
        bal=100000.0; peak=100000.0; floor=floor0
        cash=0.0; gooddays=0; last_payout_bal=100000.0; payouts=0
        daymax=0.0; k=0
        for k in range(st,min(st+maxsess,N)):
            p=pnl_by_sess[k]; bal+=p
            if p>0: daymax=max(daymax,p)
            peak=max(peak,bal); floor=min(peak-3000.0,lock)
            if bal<=floor:
                res.append(("bust",k-st,cash,payouts)); break
            if payout_mode:
                if p>=min_day: gooddays+=1
                prof=bal-last_payout_bal
                if gooddays>=5 and prof>0:
                    req=min(0.5*(bal-100000.0),2500.0)
                    if req>=500:
                        bal-=req; cash+=0.9*req; payouts+=1
                        gooddays=0; last_payout_bal=bal
                        if payouts>=5: res.append(("live",k-st,cash,payouts)); break
            else:
                prof=bal-100000.0
                if prof>=target and (k-st)>=1 and daymax<=0.5*prof:
                    res.append(("pass",k-st,cash,payouts)); break
        else:
            res.append(("timeout",k-st,cash,payouts))
    return res

print("=== PHASE 1: EVALUATION.  target +6,000 before the trailing floor ===")
print("theory: a fair game gives P(pass)=3000/9000=33.3% for ANY strategy.")
print("the only lever is cost drag, which is n*cost_at(n) per SESSION.\n")
print(f"{'micros':>6} {'$RT':>7} {'drag/sess':>10} {'P(pass)':>8} {'P(bust)':>8} "
      f"{'P(t/o)':>7} {'med days':>9}")
for n in (2,4,6,10,20,40,60):
    p=rets*n-rt(n)
    r=run(p,6000.0)
    lab=[x[0] for x in r]; d=[x[1] for x in r if x[0]=="pass"]
    print(f"{n:>6} {rt(n):>7.2f} {rt(n):>10.2f} "
          f"{lab.count('pass')/len(r):>8.1%} {lab.count('bust')/len(r):>8.1%} "
          f"{lab.count('timeout')/len(r):>7.1%} {np.median(d) if d else float('nan'):>9.0f}")

print("\n=== PHASE 2: FUNDED.  how much cash comes out before the floor takes it ===")
print("strategy: hold long until +$200 (the payout day-minimum), else flatten.")
print("bound: E[total withdrawn] = 100,000 - E[floor at bust] <= $3,000, minus")
print("       every dollar of cost. the 5-day rule FORCES round turns.\n")
print(f"{'micros':>6} {'$RT':>7} {'E[cash]':>9} {'P(>=1 payout)':>14} {'P(bust $0)':>11} "
      f"{'med payouts':>12} {'E[sessions]':>12}")
funded={}
for n in (2,4,6,10,20,40,60):
    p=hit_or_flatten(n,200.0)
    r=run(p,None,payout_mode=True,maxsess=500)
    cash=np.array([x[2] for x in r]); po=np.array([x[3] for x in r])
    funded[n]=cash.mean()
    print(f"{n:>6} {rt(n):>7.2f} {cash.mean():>9.0f} {(po>=1).mean():>14.1%} "
          f"{(cash==0).mean():>11.1%} {np.median(po):>12.0f} "
          f"{np.mean([x[1] for x in r]):>12.0f}")

print("\n=== THE WHOLE GAME: EV per evaluation purchased ===")
print(f"{'micros':>6} {'P(pass)':>8} {'E[cash|funded]':>15} {'gross EV':>9} "
      f"{'EV @ $225':>10} {'EV @ $158':>10}")
for n in (2,4,6,10,20,40,60):
    pe=[x[0] for x in run(rets*n-rt(n),6000.0)]
    pp=pe.count("pass")/len(pe)
    g=pp*funded[n]
    print(f"{n:>6} {pp:>8.1%} {funded[n]:>15.0f} {g:>9.0f} {g-225:>10.0f} {g-158:>10.0f}")
