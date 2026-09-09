# Pre-registration — free, non-OHLCV covariates for `weekend_hold_v1`

Written **before** looking at any of the three results. Dated 2026-09-08.

## Why pre-register

`weekend_hold_v1` is the only leg in the ledger that survives the audited 16:45 ET
mandatory flatten, so it is the only live target. That makes it the thing most at
risk of being p-hacked, and this project's recorded history is exactly that failure
mode (Quant Guild lecture 97; ~74 NO-GO configs).

One trial has **already been spent** on this sample this session: prior-week
*realized* volatility conditioning, which was falsified (HIGH-vol excl. 2020/2026:
−$15.54/wk, t=−0.14). That trial counts against the budget below.

## Sample (fixed)

- Trades: 202 weekend holds, MNQ, Sunday 18:00 → Monday 15:59 ET, 1 micro, net of
  costs, built by `rule_based_v1/validation/research_portfolio_mc.weekend_trades`.
- Span: 2020-01 → 2026-07. Tape `data/processed/mnq_1m_all.parquet`
  (America/Chicago, DST-aware — see the tzguard two-convention note).

## Hypotheses (exactly three; no others will be tested on this sample)

**H1 — crypto closure-window return.** Equity futures are shut Friday 17:00 →
Sunday 18:00 ET. Crypto trades throughout. Information arriving in that window is
impounded in BTC and *cannot* be in the futures tape. If the Sunday reopen
under-reacts, BTC's closure-window return predicts weekend_hold P/L.
Covariate: log return of Coinbase BTC-USD, Friday 17:00 ET → Sunday 17:00 ET
(one hour **before** the 18:00 entry, so it is strictly observable at entry).

**H2 — implied-vol term structure.** The weekend premium is compensation for gap
risk; the market's forward price of that risk is the VIX term structure.
Covariate: `VIX3M / VIX` at the Friday close preceding entry.
Note: this is *not* a retest of realized vol, but it may proxy the same state.

**H3 — speculative positioning.** COT is positioning, not price. Tuesday's data is
released Friday 15:30 ET, i.e. before a Sunday 18:00 entry.
Covariate: non-commercial net as % of open interest, 52-week trailing z-score
(`cot_positioning`, z-window excludes the current observation), E-MINI S&P 500,
using the most recent report **released** before entry (a ≥3-day release lag is
enforced, not the report date).

## Test procedure (fixed)

For each covariate `x`, standardized on a trailing expanding window (no lookahead):

    pnl_t = a + b * x_t + e_t     OLS, Newey-West HAC errors

Primary statistic: `t(b)`. Secondary: mean P/L in the top vs bottom tercile of `x`,
with terciles from expanding-window quantiles.

## Multiple-testing budget

Trials on this sample: 3 pre-registered + 1 already spent (realized vol) = **4**.
Bonferroni threshold: **α = 0.05 / 4 = 0.0125**, two-sided.

## Decision rule (fixed, all three conditions required)

A covariate is a GO candidate only if:

1. `p(b) < 0.0125` on the full sample, **and**
2. the sign of `b` is the same in both halves of the sample, **and**
3. it still clears `p < 0.05` after **excluding 2020 and 2026** — the two periods
   already shown to carry the strategy (excl. 2020+2026 the base strategy is
   t=2.28, and 2021–2024 alone is t=0.84).

Condition 3 is the load-bearing one. A covariate that only works in 2020 and 2026
is re-describing the periods that already carry the edge, not adding information.

Anything failing any condition is recorded NO-GO and **not** re-tested with a
variant. Variants are new trials and would require a new pre-registration with a
widened budget.

## What a pass would and would not mean

A pass would mean the covariate carries information about weekend P/L that the
futures tape does not. It would **not** on its own justify sizing: the base
strategy's own robustness problem (flat 2021–2024) is unaffected by finding a
filter, and any filter reduces `n`, which makes the already-thin trade count
thinner. Sizing would still require the corrected-rules path-aware MC.

---

## RESULT (2026-09-08) — 0 of 3 GO. Recorded after execution, protocol unchanged.

`ml_intraday_v3/results/edge_screens/weekend_covariates.json`
Reproduce: `python -m ml_intraday_v3.diagnostics.test_weekend_covariates`

n=172 of 202 (the first 30 are consumed by the expanding-window standardisation).

| covariate | full b / t / p | excl 2020+2026 | sign stable | verdict |
|---|---|---|---|---|
| H1 crypto closure return | +61.40 / 1.20 / 0.231 | t=1.45, p=0.150 | yes | **NO_GO** |
| H2 VIX3M/VIX slope | −44.23 / −0.99 / 0.323 | t=−0.15, p=0.882 | no | **NO_GO** |
| H3 COT non-comm z | +13.34 / 0.32 / 0.753 | t=−0.03, p=0.975 | no | **NO_GO** |

None clears the Bonferroni threshold of 0.0125; none is close to even an
unadjusted 0.05.

**H1 is the only one that behaves coherently** — correct sign, stable across both
halves, and terciles ordered monotonically (bottom $91.9, mid $87.2, top $167.6,
n=45/68/59). It is simply not significant at n=172. That is a "not enough
evidence" result rather than a falsification, but under the pre-registered rule it
is NO_GO and **must not be re-tested with a variant** (different window, different
coin, different threshold). Doing so is the exact data-snooping loop this document
exists to prevent. Re-opening it requires new data, not a new specification.

**H2 and H3 flip sign between halves**, which is the signature of noise. Both are
falsified, not merely underpowered.

### What this says about the "get better data" question

The data was genuinely new and genuinely free — implied vol back to 1990,
positioning back to 1997, and 49h/week of crypto tape covering exactly the window
when equity futures are shut. None of it rescued the strategy. The binding
constraint on `weekend_hold_v1` is not the feature set; it is that the underlying
effect averages $24.75/week across 2021–2024. A filter cannot manufacture a
premium that is not being paid, and every filter reduces an already thin n.
