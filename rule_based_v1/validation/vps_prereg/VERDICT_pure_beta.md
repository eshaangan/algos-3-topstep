# VERDICT — "just follow the market" = NO-GO, and the reason is the useful part

    date:   2026-09-10
    script: rule_based_v1/validation/research_pure_beta.py
    runs:   runs/pure_beta.json (MNQ 6.5y), runs/pure_beta_mes.json (ES 15.5y)

## The question, which is a good one

The evaluation is not a contest against the market. It is a **first-passage
problem**: reach +$6,000 before a $3,000 trailing drawdown catches you. Alpha is
only one way to bias that walk, and the equity risk premium is free. The ledger
half-reached this in Sep 2026 ("beta is not scarce — the binding constraint is
fitting the PATH inside a $3k trailing buffer") and then never tested it.

## What "following the market" actually costs here

Lucid force-flattens at 16:45 ET daily, so **buy-and-hold is not available**. The
closest legal thing is a chain of 18:00 -> 16:45 sessions, which is **five forced
round trips a week**, charged per contract.

## Result — it tops out near 30% and busts two thirds of the time

Best cell over all sizes, contiguous historical windows:

| tape | horizon | best P(pass) | at size | P(bust) |
|---|---|---:|---:|---:|
| MNQ, 6.5y, 1,654 sessions | 16wk | 30.2% | 3 | 69.8% |
| MNQ | 32wk | **31.7%** | 3 | 68.3% |
| MES via ES, 15.5y, 3,841 sessions | 16wk | 24.5% | 20 | 75.5% |
| MES | 32wk | **28.3%** | 8 | 71.3% |

Against the two-leg event book on the same rules: **57.5% pass, 34% bust.**
Pure beta is roughly half the pass probability at double the bust rate, on both
indices, at every size.

## 🔑 WHY — the metric that decides a first-passage problem is not Sharpe

The trailing floor **locks at the starting balance**, so the game has two phases:
get from $0 to +$3,000 with a cushion behind you, then get from +$3,000 to
+$6,000 **without ever giving back to breakeven.** Phase two is what kills beta.

Raw annualised Sharpe of the hold is 0.63 (MNQ) and 0.59 (MES) — perfectly
respectable, and irrelevant. What matters is **return per unit of adverse
excursion**, and the ledger already measured it: **fomc 0.47, weekend 0.18,
beta 0.04.** A 12x gap in exactly the quantity that decides whether you reach
the target before the drawdown. **Beta is abundant but lumpy; the events are
rare but smooth, and smooth is what this game pays for.**

## 🔑 SIZE IS NOT A LEVER WITHOUT AN EDGE

P(pass) is nearly FLAT in size above 3-4 contracts (MNQ 22-32%, MES 23-28%).
That is the first-passage property: at near-zero edge, scaling the bet scales
target and barrier together, so the outcome depends on the barrier RATIO, not
the stake. The ledger's "size is the whole lever" finding applies only to
edge-bearing strategies. Here it buys nothing.

## ⚠️ METHOD WARNING — iid bootstraps flatter first-passage estimates

Contiguous windows vs an iid bootstrap of the same sessions, MES 32wk:

| size | contiguous | iid bootstrap |
|---|---:|---:|
| 3 | **13.0%** | 27.8% |
| 4 | **16.7%** | 26.7% |
| 8 | 28.3% | 23.7% |

Bootstrapping destroys the autocorrelation and drawdown-clustering that decide
first passage, and at small sizes it roughly DOUBLES the apparent pass rate.
**Every P(pass) in this project's ledger, including the book's 57.5%, comes from
an iid bootstrap.** Those numbers should be re-estimated on contiguous windows
before anyone relies on them.

## Bottom line

The reframe is right and worth keeping: optimise the path, not the edge. But
the answer it gives is that the event book already IS the path-optimised
strategy, because its return-to-drawdown ratio is an order of magnitude better
than beta's. Following the market is strictly worse here.
