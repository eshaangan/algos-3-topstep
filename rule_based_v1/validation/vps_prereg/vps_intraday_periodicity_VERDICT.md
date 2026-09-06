# vps_intraday_periodicity — NO-GO. Dead in both directions, by a factor of 2.5–24×.

Script `research_intraday_periodicity.py`, run `runs/periodicity.json`.
All 26 pre-registered cells computed and counted. Bonferroni bar t = 3.09.

## Result: no day-lagged, time-matched autocorrelation exists to trade

| instrument | n | conditional $/trade | pooled t | vs unconditional | vs shuffled-lag |
|---|---|---|---|---|---|
| ES 2010–2025 | 48,708 | −$7.79 | **−23.21** | t=−1.37 | t=−1.06 |
| MNQ 2020–2026 | 21,005 | −$6.28 | **−4.53** | t=−1.48 | t=−2.49 |

Every one of the 26 buckets is negative on ES. The conditional strategy is **worse than
simply being long** the same bucket, and **worse than the shuffled-lag control** — the
previous session's sign carries less information than a random other day's sign. That is
the cleanest possible refutation: the lag structure HKS and Bogousslavsky predict is not
merely absent, it is mildly counterproductive.

## Reversing the signal does not rescue it — cost is the whole story

Stripping cost out of the conditional mean gives the actual predictable component, and
then charging it against the round turn in either direction:

| instrument | round turn (2 micros) | predictable gross | reversed, net of cost | cost ÷ \|signal\| |
|---|---|---|---|---|
| ES | $7.48 | −$0.31 | −$7.17 | **24.5×** |
| MNQ | $4.48 | −$1.80 | −$2.68 | **2.5×** |

**The entire predictable component is 1/2.5 to 1/24 of the round-turn cost.** There is no
version of this family — long, short, or filtered — that survives retail execution. This
independently reproduces the project's meta-finding that short-horizon barrier trading
loses in both directions on every instrument tested.

## Why it was still worth a slot

This was NOT a re-run of the killed 540-cell clock map. That map tested unconditional
cell means; a bucket can have a zero mean and still carry strong day-over-day
autocorrelation at its own clock time, which the map could not detect. The distinction
was real and the test was the right one — the answer is just no.

It was also the last high-velocity candidate available: 13 buckets × ~250 days ≈ 3,250
opportunities/year, against the existing legs' 52 and 8. The required-edge frontier says
only a high-velocity family could supply the ~$350/week the deadline needs. **This was
the shot, and it missed by more than an order of magnitude on ES.**

## Note on the cost-floor gate

The gate did its job as a *screen* and should be kept: on ES the round turn is 11.9–21.5%
of the mean absolute bucket move, on MNQ only 1.9–4.2%. MNQ looked far more promising on
that screen and was still dead — because what matters is cost against the **predictable**
component, not against total movement. Record that distinction: a favourable
cost/volatility ratio is necessary, never sufficient.

**Verdict: NO-GO. Do not re-run. The intraday periodicity family is closed.**
