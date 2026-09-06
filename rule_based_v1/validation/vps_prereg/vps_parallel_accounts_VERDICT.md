# vps_parallel_accounts — NO-GO. The campaign route saturates at 76.2%.

Script `research_parallel_accounts.py`, run `runs/parallel.json`.

## The simulation is trustworthy — identity check passed exactly

Every account trades ONE shared weekly stream. The mandatory control:

| | P(campaign) |
|---|---|
| 1 account | 59.2350% |
| 3 identical accounts, same start | 59.2350% |
| **difference** | **0.00e+00 — PASS** |

Same-start, same-size parallelism gives *exactly* the single-account probability, as it
must: those are the same account three times. Independent per-account draws would have
fabricated diversification that does not exist. The plumbing is correct, so the rest of
the run means something.

## Staggering is a real lever — and it saturates below the bar

25k tier, $500/micro stop, 2-week stagger, 16-week horizon:

| accounts | P(campaign) | marginal | eval fees |
|---|---|---|---|
| 1 | 59.2% | — | $150 |
| 2 | 65.2% | +6.0% | $300 |
| 3 | 69.9% | +4.7% | $450 |
| 4 | 73.0% | +3.0% | $600 |
| 6 | 75.9% | +2.9% | $900 |
| 8 | **76.2%** | +0.4% | $1,200 |
| 12 | 76.2% | **+0.0%** | $1,800 |
| 16 | 76.2% | **+0.0%** | $2,400 |

**Hard ceiling at 76.2%, reached by 8 accounts, and no number of further evaluations moves
it.** With a 2-week stagger, eight accounts already occupy every distinct phase available
inside a 16-week window; account nine is an exact duplicate of account one. The residual
~24% is the set of streams where the shared weekend sequence is bad enough that *every*
phase fails — and buying more accounts cannot diversify a risk they all share.

The economics fail independently: 8 evaluations cost roughly $1,200 to chase a $1,250
target.

## Other campaign shapes

| campaign | accounts | P | median wk |
|---|---|---|---|
| 4× 25k, stagger 2wk | 4 | 73.0% | 8 |
| 3× 25k stagger 4wk + 1× 100k | 4 | 72.0% | 9 |
| 1× 25k + 1× 100k (size mix, same start) | 2 | 59.3% | 8 |
| 2× 100k wk1/fo2, stagger 4wk | 2 | 4.5% | 14 |

Size mixing adds essentially nothing (59.3% vs 59.2% for the 25k alone) — the 100k cannot
finish inside 16 weeks, so it contributes no winning paths.

## Why this closes the question

This was the last structural lever. The five routes to the bar and their ceilings:

| route | ceiling within 16wk |
|---|---|
| more edge | needs 3.2–4×; both high-velocity families dead |
| more size | ~50% (peak at 3 micros) |
| different firm (no flatten) | single-event risk binds identically everywhere |
| smaller tier | ~64% at any horizon |
| **more accounts in parallel** | **76.2%, saturated** |

Every route is measured, and every one falls short. **The 85%/16-week bar is not
reachable with this book by any structural means.**

**Verdict: NO-GO. Do not buy a campaign of evaluations expecting the deadline.**
