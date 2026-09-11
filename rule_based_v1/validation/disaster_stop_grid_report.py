import json, pandas as pd
d = pd.DataFrame(json.load(open("/Users/jg/auction/runs/disaster_stop_repaired.json")))
d["stop"] = d["stop"].fillna(-1)
print("=== THE FROZEN SPEC'S OWN CELL ($600 stop, weekend x2, fomc x2) ===")
f = d[(d.stop == 600.0) & (d.weekend == 2) & (d.fomc == 2)]
print(f[["weeks","p_pass","p_bust","median_weeks","p_pass_decay50","worst_weekend","per_week"]].to_string(index=False))
print("   claimed on the damaged tape: 50.4% @16wk, 73.4% @32wk")
print()
print("=== BEST CELL AT EACH HORIZON, AND THE SAFEST GOOD ONE ===")
for w in (16, 32):
    s = d[d.weeks == w].sort_values("p_pass", ascending=False)
    b = s.iloc[0]
    print(f"{w}wk  best      stop=${b.stop:.0f} wk x{b.weekend} fomc x{b.fomc}  "
          f"p={b.p_pass:.3f} bust={b.p_bust:.3f} decay50={b.p_pass_decay50:.3f} "
          f"worst_wknd=${b.worst_weekend:.0f}")
    # best cell whose bust probability is under a third
    q = s[s.p_bust < 0.33]
    if len(q):
        c = q.iloc[0]
        print(f"{w}wk  bust<33%  stop=${c.stop:.0f} wk x{c.weekend} fomc x{c.fomc}  "
              f"p={c.p_pass:.3f} bust={c.p_bust:.3f} decay50={c.p_pass_decay50:.3f} "
              f"worst_wknd=${c.worst_weekend:.0f}")
    else:
        print(f"{w}wk  bust<33%  NO CELL")
print()
print("=== ceiling check: does ANY cell clear 60% / 70% / 85%? ===")
for thr in (0.60, 0.70, 0.85):
    n = (d.p_pass >= thr).sum()
    print(f"  p_pass >= {thr:.0%}: {n} of {len(d)} cells")
print()
print("=== how P(pass) moves with the stop, at the 32wk optimum size (wk x2, fomc x4) ===")
z = d[(d.weeks == 32) & (d.weekend == 2) & (d.fomc == 4)].sort_values("stop")
z = z.assign(stop=z["stop"].replace(-1, float("nan")))
print(z[["stop","p_pass","p_bust","worst_weekend","per_week"]].to_string(index=False))
