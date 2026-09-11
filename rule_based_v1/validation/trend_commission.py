"""How much of the intraday trend survives at a lower commission?"""
import json
import numpy as np

COMM_NOW = 0.62          # $/side, current
MEASURED = {1: 2.047, 2: 2.101, 4: 2.191, 6: 2.269, 9: 2.388,
            13: 2.558, 20: 2.844, 40: 3.562, 60: 4.205}

def cost_at(size, comm):
    """Measured round turn, with the commission component swapped out."""
    xs = np.array(sorted(MEASURED)); ys = np.array([MEASURED[k] for k in xs])
    total = float(np.interp(size, xs, ys))
    spread_and_depth = total - 2 * COMM_NOW      # what the book itself charges
    return spread_and_depth + 2 * comm

for label, path, per_wk in (("MNQ", "/Users/jg/auction/runs/intraday_trend.json", None),
                            ("MES (ES tape)", "/Users/jg/auction/runs/intraday_trend_es.json", None)):
    rows = json.load(open(path))
    print(f"\n=== {label}: net $/micro/trade at three commission rates ===")
    print(f"{'horiz':>6} {'autocorr':>9} {'gross':>8} {'size':>5} "
          f"{'$0.62':>9} {'$0.35':>9} {'$0.25':>9} {'$/wk @0.25':>11}")
    for r in rows:
        H, g, s = r["horizon"], r["gross"], r["size"]
        trades_wk = (390 / H) * 5
        line = [g - cost_at(s, c) for c in (0.62, 0.35, 0.25)]
        wk = line[2] * s * trades_wk
        print(f"{H:>6} {r['autocorr']:>9.4f} {g:>8.3f} {s:>5} "
              f"{line[0]:>9.3f} {line[1]:>9.3f} {line[2]:>9.3f} {wk:>11.0f}")
print("\n$/wk column is at the tail-legal size, against a $520/wk requirement.")
