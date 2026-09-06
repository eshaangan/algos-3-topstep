# GO — portfolio reallocation: weekend×1 + FOMC×2 (2026-09-05)

**This is a configuration result, not a new edge.** The hunt found no new signal.
What it found is that the project's existing validated edges are allocated wrongly:
the current plan bets on the weaker leg and under-sizes the stronger one.

**Run:**
```
python3 rule_based_v1/validation/research_portfolio_mc.py --out runs/portfolio_mc.json
```

## The two legs, measured on the same MNQ tape

| leg | n | mean | t |
|---|---|---|---|
| weekend_hold_v1 (Sun 18:00 → Mon 15:59) | 202 | +67.15 pts | +4.58 |
| fomc_drift_v1 (14:00 day-before → 13:55 decision day) | 47 | +87.97 pts | +3.46 |

`monday_rth_v1` is deliberately excluded: its window sits entirely **inside**
weekend_hold_v1, so trading both is the same exposure counted twice, not
diversification.

## Sizing grid, $3k target against a $3k EOD-trailing buffer

| weekend | fomc | $/wk | P(pass) | median |
|---|---|---|---|---|
| 1 | 0 | $132 | 96.2% | 20wk |
| **1** | **2** | **$185** | **96.4%** | **15wk** |
| 2 | 0 | $264 | 79.5% | 9wk |
| 2 | 2 *(current plan)* | $318 | 81.6% | 8wk |
| 3 | 2 | $450 | 73.0% | 5wk |

1+2 strictly dominates 1+0 — higher pass probability *and* five weeks faster.

## Why this is the recommendation: the decay stress

Every number above is bootstrapped from post-2020 MNQ. This session established
on the new 15.5-year ES tape that the weekend leg paid **−$8.80/wk pre-2020** and
+$65 after (see `weekend_hold_15y_ES_CHECK.md`), so its forward mean is the single
most uncertain input in the whole plan. Shrinking that mean while holding
dispersion and MAE fixed:

| config | ×1.00 | ×0.75 | ×0.50 | ×0.25 | ×0.00 |
|---|---|---|---|---|---|
| **1wk+2fomc** | 96.4% | 93.0% | 87.0% | 76.7% | **62.3%** |
| 2wk+2fomc *(current)* | 81.6% | 75.2% | 67.0% | 56.9% | 45.5% |
| 1wk+0fomc | 96.2% | 90.6% | 78.4% | 58.3% | 35.4% |
| 2wk+0fomc | 79.5% | 71.4% | 61.1% | 48.9% | 36.3% |

**1wk+2fomc still passes 62.3% with the weekend edge at exactly zero**, because
the FOMC leg carries the account on its own. The current 2wk+2fomc plan falls to
45.5% in the same scenario, and weekend-only collapses to 35%.

That is the whole argument. The configuration change moves the account's
dependence off the leg with a documented regime problem and onto the leg that
passed dev *and* a sealed holdout, and it costs nothing in pass probability
today — it gains 15 points.

## What changes operationally

1. Cut the weekend runner from 2 micros to **1**.
2. Raise the FOMC runner to **2** micros (it is the load-bearing leg now).
3. Expect ~15 weeks median rather than ~8. This is the deliberate trade: the
   faster configurations are all more fragile to weekend decay.

## Honest limits

- Not a new edge. Four papers and a 540-cell clock sweep produced no new signal
  this session; this is a reallocation of what already existed.
- The MC bootstraps iid weekly draws from post-2020 MNQ. It does not model serial
  correlation in regimes, so the ×1.00 column is optimistic in the same way the
  ledger's earlier 63.1% was conservative. The decay columns are the honest range,
  and the recommendation is chosen to be the one that survives all of them.
- FOMC n=47 on this tape. Small. It is the same 8-per-year limitation the ledger
  already records, and the reason the weekend leg is kept at 1 rather than 0.
