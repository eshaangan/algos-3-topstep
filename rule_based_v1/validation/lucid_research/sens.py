import sys, numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-eshaanganguly-Documents-projects-algos-3-topstep/f6967d40-9bb8-470d-b3bc-1b7c827a9ace/scratchpad")
from account import *

print("P(pass) vs DIRECTIONAL ACCURACY  (accuracy-matched oracle, 6 seeds)")
print("an oracle flatters a real model: it is right on big and small moves alike\n")
print(f"{'accuracy':>9} " + " ".join(f"{f'n={n}':>8}" for n in (2,4,10,20)))
for p in (0.50, 0.525, 0.55, 0.575, 0.60, 0.65, 0.70, 0.80, 1.00):
    row = []
    for n in (2, 4, 10, 20):
        rs = [rate(oracle(p, sd), flat(n))[0] for sd in range(6 if p < 1 else 1)]
        row.append(np.mean(rs))
    print(f"{p:>9.1%} " + " ".join(f"{v:>8.1%}" for v in row))
print("\nreference: always-long is 54.4% accurate; best walk-forward model 53.8%")
