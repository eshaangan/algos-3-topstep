# Session verdict, 2026-09-05 — NO-GO on 85%/16wk, with the frontier mapped

## The bar
LucidFlex 100k: $6,000 target, $3,000 EOD-trailing MLL, 50% consistency.
GO = P(pass) ≥ 85% within 16 weeks, legal, no single event able to breach the MLL,
surviving control + both era halves + cross-instrument.

**Not achieved. Six independent attacks, all pre-registered, all with mandatory controls.**

## What was tried and what happened

| attack | outcome |
|---|---|
| **MLL-lock bug in all 4 MCs** | **Fixed** — every prior P(pass) in this repo was too pessimistic |
| Recover a legal FOMC window | **Partial win** — 18:00 entry legal; 42wk → 34wk free |
| Track B: survey 4 prop firms | **Win** — Phidias ~halves time-to-pass; no firm hits 16wk |
| Post-release continuation (~250/yr) | Dead — 12 cells, best \|t\|=0.64 vs 2.87 bar |
| Disaster stop to unlock size | **Works, adopt** — but P(pass) peaks at 48.6% |
| Intraday periodicity (~3,250/yr) | Dead — 26 cells; cost is 2.5–24× the predictable component |
| Smaller account tiers (25k/50k) | **Better geometry, worse ceiling** — 25k asymptotes at ~64% |

## The binding constraint, in three steps

**1. It is not the flatten.** Single-event risk binds first and binds identically at every
firm, because it is a property of the strategy and the MLL, not the rules. Worst weekend
at 2 micros is −$3,801 = 127% of a $3,000 MLL — an account kill even at Phidias, which
has no flatten and no consistency rule.

**2. It is not the tail either — that one is fixable.** A $600/micro stop keeps 93% of
the premium while cutting the worst trade 46% (−$1,117 → −$603), with t rising to 4.36
and only 8.9% of weekends stopped. The premium is not in the deep excursions.

**3. It is the target/MLL geometry against this book's Sharpe.** Holding dispersion and
tail fixed and scaling only the mean, at the largest tail-legal size:

| edge quality | $/wk | P(pass) @16wk |
|---|---|---|
| ×1.0 (actual) | 162 | 5.7% |
| ×2.5 | 405 | 59.1% |
| ×3.0 | 486 | 80.5% |
| **×3.2** | **~520** | **~85%** |

**85%/16wk needs ~3.2× the edge at the same risk.** No remaining family can supply it:
both high-velocity candidates are now dead, one by more than an order of magnitude.

## The speed/safety frontier across account tiers

No tier gives both. The ratio is target ÷ MLL.

| tier | ratio | best config | P(pass) | median wks | ceiling |
|---|---|---|---|---|---|
| 25k | **1.25** | wk×1 + fomc×1, $500 stop | 57.1% @16wk | **8** | ~64% at any horizon |
| 50k | 1.50 | wk×2 + fomc×2, $500 stop | 54.2% @16wk | 8 | — |
| **100k** | 2.00 | wk×1 + fomc×2, $600 stop | **94.4%** | 35 | — |

The 25k is fast (8-week median) and permanently unsafe — its $1,000 MLL busts ~36% of
paths early regardless of horizon. The 100k is safe and slow. **This is a real frontier,
not a tuning failure.**

## Recommended plan (none of it needs new alpha)

0. **Re-read any P(pass) written before 2026-09-05 as too LOW.** The MLL-lock bug is
   fixed; the incumbent is 95.0%/42wk, not 92.4%/42wk.
1. **Adopt the $600/micro disaster stop unconditionally.** Improves P(pass)
   (92.0→94.4%), decay-robustness (70.9→73.2%) and tail (37%→20% of MLL) simultaneously.
2. **Add the legal 18:00 FOMC leg at LucidFlex now** — 42wk → 34wk, no firm change.
3. **Consider a firm with no flatten.** Corrected-lock results, 104wk horizon:

| firm | best ≥85% config | P(pass) | median wks |
|---|---|---|---|
| LucidFlex / MyFundedFutures | wk×1 + fomc×3 @18:00 | **92.7%** | 31 |
| Phidias Swing 100k | wk×1 + fomc×4 @16:00 | 90.6% | 24 |
| Elite Trader Funding Diamond | wk×1 + fomc×3 @14:00 | 89.4% | **22** |

Realistic: **Phidias or ETF Diamond, weekend×1 + FOMC×3–4, $600 stop, ~22–24 weeks at
~90%.** Do not size up to chase 16 weeks — every configuration fast enough peaks below
50%, even with the lock corrected.

## Two caveats that govern all of the above

- **The ES dev half is negative for both surviving legs.** This is a post-2020 regime bet;
  the decay-stressed ~73% is the honest figure, not the 94% headline.
- **Every firm rule here is `verified: false`** — web-sourced, unconfirmed against a
  dashboard.

## Bugs fixed this session

- `tzguard.check_window_legal()` tested whether a hold *crossed* the flatten but not
  whether the entry could be **placed**. It blessed 17:00 — the single best-looking cell
  in the scan (t=+2.97, efficiency 0.407) — whose entire advantage came from the
  untradeable 17:00–18:00 CME maintenance halt. Now enforces `trading_resumes_et`.
- The disaster-stop MC carried the **unstopped** MAE into the intra-trade bust check,
  charging the account for excursions a stopped position was never in. A stop changes the
  MAE, not just the exit.


## Postscript: the MLL-lock bug (found last, mattered everywhere)

`account_rules.yaml` has recorded `drawdown.locks_at_starting_balance: true` all along —
the EOD-trailing floor stops rising once it reaches the starting balance. **All four Monte
Carlos in this repo trailed it forever**: `research_corrected_rules_mc.py`,
`research_firm_choice_mc.py`, `research_portfolio_mc.py`,
`research_vol_target_weekend.py`.

Direction of the error: every P(pass) this project has ever reported was **too
pessimistic**, and most so at large sizes, since those build the biggest peak and were
therefore penalised with the highest phantom floor. Effect on the incumbent: 92.4% → 95.0%
at 1 micro; 65.4% → 73.4% at 2.

It does not change the verdict — the corrected 16-week ceiling is still ~50% — but it
changes the recommendation (FOMC×3–4 now beats ×2) and it invalidates the *level* of every
historical P(pass) in the ledger. **Any pass probability in this repo written before
2026-09-05 should be read as a lower bound.**
