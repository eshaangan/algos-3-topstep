# VERDICT — Treasury auction concession & reversal: **NO-GO**

    prereg:   vps_treasury_auction_concession.md  (registered before any statistic)
    scripts:  rule_based_v1/validation/research_auction_concession.py
    runs:     runs/zn_all.json (566-auction pooled), runs/zn_10y.json (matched tenor)
    data:     data/processed/td_coupon_auctions.csv (566 nominal coupon auctions,
              Fiscal Data API, free + keyless, verified 2026-09-09)
    verdict:  KILLED on all four pre-registered kill conditions, and separately
              closed by cost arithmetic that no signal quality can fix.

## Why this family was worth a slot

It was the only untested family with the velocity to fix the project's actual
problem. The required-Sharpe table says `fomc_drift_v1` is velocity-starved: a
per-trade Sharpe of 0.515 fired 8 times a year annualises to 1.46 against a
requirement near 2.1. **Treasury coupon auctions run ~84 a year — 10.5x the
FOMC** — so even a much weaker per-trade edge would have annualised above the
bar. The mechanism (Lou-Yan-Zhang 2013 RFS; Fleming-Liu-Nguyen FRBNY SR 1188) is
a dealer inventory concession, economically the same shape as the one edge this
project has validated.

## Result — pooled, n=543 auctions on ZN, 2020-01 .. 2026-07

| leg | event | control | event − control | t | net of cost |
|---|---:|---:|---:|---:|---:|
| concession (short into 13:00) | −$11.80 | −$6.22 | −$5.57 | **−0.57** | −$5.83 |
| reversal (long after result) | −$8.09 | −$3.23 | −$4.85 | **−0.59** | −$25.71 |

Matched-tenor 10-Year only (n=79): concession diff t=−0.51, reversal diff t=+0.50.

1. **Significance.** Best |t| anywhere is 0.59 against the standing 2.5 bar. At
   n=543 this is absence, not low power.
2. **Sign.** The **reversal leg is wrong-signed** in the pooled sample (−$8.09
   where the mechanism requires positive). The mechanism picks the sign; flipping
   it would be fitting, not trading.
3. **Era split.** Concession dev 2020-23 −$20.27 vs val 2024-26 **+$1.64** — sign
   flip. Per year it changes sign four times (−32.81, −28.12, −12.83, −7.81,
   −22.97, **+14.10, +24.31**).
4. **Cost.** Round turn $17.62/contract against a mean |3-hour move| of $130,
   a **13.5% cost ratio** — versus the 1.9-4.2% that made MNQ cost-viable and
   the 30-70% that ruled out MCL/M6E.

## 🔒 The arithmetic that closes it independent of significance

**The entire measured concession is $11.80 per contract. The round turn is
$17.62.** A trader who captured the whole effect, perfectly, on every auction,
never being adversely selected, nets **−$5.82 per auction**.

At the free-est possible cost — zero commission, paying only ZN's one-tick spread
— the round turn is still $15.63, and the effect still does not clear it.

This survives a change of vehicle. $11.80 on ZN is about **0.17 basis points**
(ZN DV01 ≈ $70). On the micro 10-Year yield future ($10/bp) that is $1.80 against
a round turn near $2.24. The cost-to-DV01 ratio is what binds, and it is the same
across the curve's retail instruments. ⇒ **the family is closed by arithmetic,
not by evidence — the same shape as the market-making kill. Do not re-test it
with different windows, tenors, auction sizes, or bid-to-cover conditioning.**

## Method notes worth keeping

* **🚨 `data/processed/zn_1min.parquet` IS MIS-TIMEZONED AND MUST NOT BE USED.**
  Its maintenance halt returns at 18:00 ET in winter but 19:00 ET in summer — the
  fixed `-05:00` landmine the ledger records for `cost_floor.py`, present in this
  tape too. The raw `data/hist_1m24_zn/` files put the halt return at 17:01 local
  in BOTH seasons, which pins the source to DST-aware America/Chicago. Rebuilt
  correctly; `research_auction_concession.assert_halt()` now aborts on any tape
  that fails the two-season check, rather than warning.
* **The control must exclude EVERY tenor, not just the tested one.** Treasury
  runs 3y/10y/30y on consecutive refunding days, so a control built from
  "non-10-Year weekdays" would have carried the effect inside the control and
  biased the test toward null for a spurious reason.
* **Internal check that the machinery works:** the reversal leg's net t is
  **−4.04**, i.e. an ordinary intraday hold losing to cost at high significance.
  That independently reproduces the project's standing "5-min barrier trading
  loses both directions" and "conditional intraday momentum loses to cost"
  findings, so the cost model and windowing are behaving.
* **Free data unlocked:** `api.fiscaldata.treasury.gov` auctions_query is keyless,
  paginates properly, carries `closing_time_comp`, and reaches back to 1979.
  TreasuryDirect's own `TA_WS` endpoint silently ignores date filters and caps at
  250 rows — it looks like it worked and returns the wrong sample.

## Trial accounting

Trial 1 of the family; the pre-declared extensions (all tenors, MNQ as vehicle)
are not run, because the cost arithmetic closes them in advance. Family CLOSED.
