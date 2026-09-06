# Track B — prop-firm survey: does any firm's rules pass the existing book?

Rules in `rule_based_v1/configs/prop_firm_rules.yaml` (machine-readable, never prose).
MC in `research_firm_choice_mc.py`; runs `runs/firm_choice.json` (16wk),
`runs/firm_choice_104.json` (104wk). Trade construction imported from
`research_portfolio_mc.py` — not reimplemented.

**All firm rules are web-sourced and `verified: false`. Confirm in-app before paying.**

## The four firms

| firm | target | MLL | consistency | daily flatten | overnight | weekend | news |
|---|---|---|---|---|---|---|---|
| LucidFlex 100k (incumbent) | $6,000 | $3,000 EOD | 50% | **16:45 ET** | no | no | none |
| MyFundedFutures Pro 100k | $6,000 | $3,000 EOD | 50% | **16:10 ET** | no | no | Tier-1 flat ±2min |
| **Phidias Swing 100k** | $6,000 | $3,000 EOD | **none** | **NONE** | yes | yes | none |
| Elite Trader Funding Diamond 100k | $5,000 | $3,500 EOD | none | NONE | yes | yes | unknown |

**MyFundedFutures is a trap.** Aggregators advertise it as allowing overnight holds. Its
own help centre says open positions are auto-closed at **16:10 ET every regular trading
day** — stricter than Lucid — and it adds a Tier-1 news restriction (flat 2 min either
side of FOMC/CPI/NFP) that Lucid does not have. It is worse on both axes that matter.

**Phidias is the real find:** identical headline arithmetic to LucidFlex — same $6,000
target, same $3,000 EOD-trailing MLL — but no daily flatten and **no evaluation
consistency rule**. Every window this project validated becomes legal, including the
14:00/16:00 `fomc_drift_v1` entries Lucid force-closes.

## Result: no firm reaches the bar

Nothing clears **85% within 16 weeks at any firm.** Not one configuration.

Best configuration at each firm (≥85% P(pass), 104-week horizon, sizes with a
single-event kill already disqualified):

| firm | best config | P(pass) | median weeks | $/wk |
|---|---|---|---|---|
| LucidFlex — as the ledger had it | weekend ×1 only | 92.4% | **42** | $132 |
| LucidFlex — **with the legal 18:00 FOMC leg** | wk×1 + fomc×2 @18:00 | 92.4% | **34** | $166 |
| MyFundedFutures | identical to Lucid (same legal windows) | 92.4% | 34 | $166 |
| **Phidias** | wk×1 + fomc×2 @16:00 | **94.0%** | 31 | $182 |
| **Phidias — fastest ≥85%** | wk×1 + fomc×4 @16:00 | 86.2% | **23** | $232 |

## Why no firm gets there: the flatten was never the binding constraint

The ledger blames the 16:45 flatten. The MC says otherwise. **Single-event risk binds
first, and it binds identically at every firm**, because it is a property of the strategy
and the $3,000 MLL, not of the rules:

| size | worst weekend | vs $3,000 MLL |
|---|---|---|
| 1 micro | −$1,900 | 63% |
| 2 micros | −$3,801 | **127% — one-weekend account kill** |
| 3 micros | −$5,702 | 190% |

Two micros on the weekend leg is disqualified at **every** firm surveyed, including the
ones with no flatten and no consistency rule. So the book is capped at 1 micro on its
highest-velocity leg, which caps it at **~$132–232/week against the $375–500/week the
16-week arithmetic requires.**

Removing the flatten buys the FOMC leg back and cuts 42 weeks → 23. It cannot buy size,
and size is what the deadline needs.

## What Track B is actually worth

Two genuine wins, neither requiring new alpha:

1. **The 18:00 FOMC leg is legal at LucidFlex right now** and takes the incumbent from
   92.4%/42wk to 92.4%/34wk. Free, no firm change. (See
   `vps_legal_fomc_window_VERDICT.md`.)
2. **Phidias, if its published rules hold up in-app,** takes it to 94.0%/31wk, or
   86.2%/23wk if you want speed. That is roughly **half the incumbent's time-to-pass**
   for the price of an evaluation.

Neither reaches 85% in 16 weeks. Track B improves the answer substantially; it does not
meet the bar.

## Before acting

Phidias's numbers come from two independent aggregator reviews; its own rules page
(`phidiaspropfirm.com/rules`, `/swing-allowed`) returns 403 to automated fetch and must
be read in a browser. Confirm, in this order: the $3,000 EOD-trailing MLL on the Swing
100k, that the evaluation truly has no consistency rule, that weekend holds are permitted
on the **evaluation** and not only on funded accounts, and the commission schedule.
