# Edge audit — 2026-09-10

## Verdict

There is a statistically defensible **sparse long-premium book** in the corrected
MNQ data, but there is no honest high-probability way for it to pass a Topstep
50K Combine in three months.  The best evidence-backed components are:

1. long one MNQ from 09:31 to 15:59 ET on Mondays;
2. long one MNQ from 18:01 ET on the day before a scheduled FOMC decision to
   13:55 ET on decision day.

Both use a fixed $250-per-contract protective stop and measured $2.05 round-trip
cost.  They are flat before the FOMC announcement and before Topstep's weekday
maintenance close.  This is a research/paper candidate, not authorization for
live trading.

## Evidence after the tape repair

| leg | events | mean / contract | win rate | t-stat | losing years |
|---|---:|---:|---:|---:|---|
| Monday RTH | 304 | $47.37 | 55.3% | 3.01 | 2022, 2024 |
| legal pre-FOMC | 51 | $105.90 | 64.7% | 2.40 | 2021 |

The stop caps an ordinary fill at about -$252.05 per contract.  Minute bars
cannot prove real stop fills during a discontinuity, so this remains an
optimistic limit-order assumption despite punitive gap-through handling.

The FOMC calendar now includes the official 2026 decision dates.  Four of those
meetings are present before the corrected minute tape ends on 2026-07-09; the
remaining dates are schedule inputs, not backtest observations.

## Topstep 50K path simulation

Rules modelled: $3,000 target, $2,000 end-of-day trailing Maximum Loss Limit
that locks at the starting balance, real-time unrealized-loss checks, and a best
day no greater than 50% of total profit.  Paths use four-week block resampling
of the historical event calendar so regimes and FOMC frequency are not sampled
as independent trades.  Each cell has 20,000 paths.

| horizon | assumption | best pass cell | pass | bust | lower-risk cell |
|---|---|---:|---:|---:|---|
| 13 weeks | historical mean | 4 contracts | 37.2% | 52.8% | 2 contracts: 22.8% pass / 18.7% bust |
| 13 weeks | 50% mean decay | 4 contracts | 25.2% | 64.7% | 1 contract: 0.7% pass / 1.6% bust |
| 26 weeks | historical mean | 2 contracts | 51.7% | 31.0% | 1 contract: 17.7% pass / 3.1% bust |
| 52 weeks | historical mean | 2 contracts | 63.9% | 34.9% | 1 contract: 58.3% pass / 7.9% bust |

The apparent 13-week optimum is rejected: an account-bust probability above
50% is not a high-likelihood strategy.  Even one year does not produce an 85%
cell.  The account geometry forces a choice between low pass probability and
unacceptable failure risk.

## What the new VPS data can and cannot establish

- The MNQ L3 archive contains 15 complete weekday recordings from 2026-08-19
  through 2026-09-09: roughly 419 million depth events, 20.4 million trades,
  and 1.54 million BBO rows.
- The older NQ MBP-1 archive has 56 dates from January through July 2026, but
  each file contains only 75 seconds around the RTH open.  It is not three
  months of full-session outcomes.
- A causal, depth-scaled opening OFI continuation signal was preregistered before
  opening the final 15-day partition.  It failed: 69 sessions, -$16.96 average,
  -$1,170 total, t=-1.38; the final holdout averaged -$40.93.
- The paper-specified close-momentum signal also failed after costs: MNQ
  2020-2026 averaged -$2.01 over 1,623 events; ES/MES 2010-2025 averaged -$2.11
  over 3,804 events.

## Invalidated claims

- The old weekend strategy is not deployable.  Missing non-DST Sunday reopens
  made the original sample DST-heavy; repaired 2021-2024 performance is weak and
  the 16-week pass estimate fell to roughly 23%.
- The old 84%-91% Combine pass claims came from selected/resampled candidates,
  stale account assumptions, or damaged data.  They are not current evidence.
- Raw one-second OFI, ORB variants, Treasury-auction concession, ECB drift,
  cross-asset rate conditioning, and close momentum all failed their relevant
  holdout or control tests.

## Reproducible artifacts

- `rule_based_v1/validation/research_topstep_50k_sparse_book.py`
- `runs/topstep_50k_sparse_events.csv`
- `runs/topstep_50k_sparse_summary.json`
- `rule_based_v1/validation/PREREG_opening_ofi_v1.md`
- `rule_based_v1/validation/research_opening_ofi.py`
- `runs/opening_ofi_v1_summary.json`

## Decision

Paper-track one contract only.  Require at least eight new Mondays and two new
FOMC observations with positive combined net PnL before reconsidering a Combine
attempt.  Do not deploy the old weekend book, ORB/MSITE/GIRE sizing, or any
four-contract "fast pass" cell from this report.
