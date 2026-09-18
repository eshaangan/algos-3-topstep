import sys, numpy as np
sys.path.insert(0,"/private/tmp/claude-501/-Users-eshaanganguly-Documents-projects-algos-3-topstep/f6967d40-9bb8-470d-b3bc-1b7c827a9ace/scratchpad")
from account import *
LONG=np.ones(N); HALF=N//2

def rate_era(lo,hi,n,stop):
    o=[episode(i,LONG,flat(n),stop) for i in range(lo,hi-420,3)]
    return o.count("pass")/max(1,len(o)), len(o)

print("Protective stop, FINE grid, split eras. A real mechanism is monotone-ish")
print("and survives in BOTH halves. dev = 2020-2023, val = 2023-2026.\n")
STOPS=[None,1200,900,700,600,500,450,400,350,300,250,200,150]
hdr=" ".join(f"{('none' if s is None else s):>6}" for s in STOPS)
for n in (1,2,3,4):
    print(f"n={n} micros            {hdr}")
    for tag,lo,hi in (("dev",0,HALF+420),("val",HALF,N),("all",0,N)):
        r=[rate_era(lo,hi,n,s)[0] for s in STOPS]
        k=rate_era(lo,hi,n,None)[1]
        print(f"  {tag} (starts={k:>3})   " + " ".join(f"{v:>5.1%}" for v in r))
    print()
