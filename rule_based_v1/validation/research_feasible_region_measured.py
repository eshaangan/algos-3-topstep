"""Feasibility map rebuilt on MEASURED fill cost instead of a flat $2.06.

The original map (research_feasible_region.py, now lost from the repo) charged
every horizon the same $2.06/contract. That is the cost of ONE lot at the touch.
The map then sized each horizon at 9-20 contracts, where the book charges more.
So the map understated cost exactly where it sized largest.

    f = (target_per_week / (N * S) + cost(S)) / M
    S = floor(risk_budget / adverse_p05)

f is the fraction of the mean absolute move a strategy must capture. It is a
rescaled per-trade Sharpe, so it is also reported as a required annualised
Sharpe traded continuously.
"""
import json
import numpy as np, pandas as pd

TAPE = "/Users/jg/auction/data/mnq_1m_all.parquet"
COSTS = "/Users/jg/auction/runs/fill_costs.parquet"
PV = 2.0                 # $/point per MNQ micro
RISK_BUDGET = 1800.0     # $ tail budget against the $3,000 MLL
TARGET_WK = 520.0        # $/week the 16-week bar requires
HORIZONS = [15, 30, 60, 120, 240]

bars = pd.read_parquet(TAPE)
rth = bars[(bars.index.hour * 60 + bars.index.minute >= 570) &
           (bars.index.hour * 60 + bars.index.minute < 960) &
           (bars.index.dayofweek < 5)]
print(f"RTH 1-min bars: {len(rth):,}  {rth.index.min().date()} -> {rth.index.max().date()}")

# measured cost curve, latency 0, pooled over the 15 L3 days
c = pd.read_parquet(COSTS)
c = c[c.latency_ms == 0]
curve = c.groupby("size")["rt"].mean()
print("measured cost curve ($/contract):", {int(k): round(v, 3) for k, v in curve.items()})

def cost_at(S: int) -> float:
    """Linear interpolation on the measured curve; flat beyond its ends."""
    xs = curve.index.to_numpy(dtype=float); ys = curve.to_numpy()
    return float(np.interp(float(S), xs, ys))

rows = []
for H in HORIZONS:
    # non-overlapping blocks within a single session
    g = rth.groupby(rth.index.date)
    moves, adverse = [], []
    for _, day in g:
        o = day["open"].to_numpy(); hi = day["high"].to_numpy()
        lo = day["low"].to_numpy(); cl = day["close"].to_numpy()
        n = len(day) // H
        for b in range(n):
            s, e = b * H, (b + 1) * H
            entry = o[s]
            moves.append(abs(cl[e - 1] - entry) * PV)
            # adverse excursion for a position held the whole block, worse side
            adverse.append(max(entry - lo[s:e].min(), hi[s:e].max() - entry) * PV)
    M = float(np.mean(moves))
    p05 = float(np.percentile(adverse, 95))     # 95th pct of adverse = p05 of P/L
    S = max(int(RISK_BUDGET // p05), 1)
    N = (390 / H) * 5                            # RTH blocks per week
    cost_new, cost_old = cost_at(S), 2.06
    f_new = (TARGET_WK / (N * S) + cost_new) / M
    f_old = (TARGET_WK / (N * S) + cost_old) / M
    # f is mean/|move|; mean|move| ~ 0.8 sd  =>  per-trade Sharpe ~ f/0.8
    sharpe = lambda f: (f / 0.8) * np.sqrt(N * 52)
    rows.append(dict(horizon_min=H, trades_per_wk=round(N, 1), mean_abs_move=round(M, 2),
                     adverse_p05=round(-p05, 0), size=S,
                     cost_assumed=cost_old, cost_measured=round(cost_new, 3),
                     f_assumed=round(f_old, 4), f_measured=round(f_new, 4),
                     harder_by=round(f_new / f_old - 1, 3),
                     sharpe_assumed=round(sharpe(f_old), 2),
                     sharpe_measured=round(sharpe(f_new), 2)))

df = pd.DataFrame(rows)
print()
print("=== FEASIBILITY MAP ON MEASURED COST (MNQ, $520/wk, $1,800 tail budget) ===")
print(df.to_string(index=False))
json.dump(rows, open("/Users/jg/auction/runs/feasible_measured.json", "w"), indent=2)
print()
print("f = fraction of the mean |move| a strategy must capture.")
print("harder_by = how much the measured cost raises that bar vs the flat $2.06.")
