# PREREGISTRATION — Treasury auction concession & reversal

    registered:  2026-09-09  (BEFORE any event statistic was computed)
    family:      scheduled-supply events (NEW family — never tested in this project)
    status:      CLOSED 2026-09-09 — NO-GO

## 1. Why this family, and why now

The kill ledger's own arithmetic (`runs/feasible_region.json`, and the
required-Sharpe table) says the binding problem is **velocity, not edge
quality**: `fomc_drift_v1` has a per-trade Sharpe of 0.515 and fires 8 times a
year, giving an annualised Sharpe of 1.46 against a requirement of ~2.1.

Treasury coupon auctions are the only untested event family with the velocity to
close that gap. TreasuryDirect's own calendar (free, keyless, verified reachable
2026-09-09) shows **159 Note+Bond auctions in the 17.5 months to 2026-09-09
≈ 109 per year — 13.6x the FOMC's velocity.** At `fomc_drift_v1`'s per-trade
quality an event family of that size would annualise to Sharpe ~5. It does not
need to be nearly that good to matter.

## 2. Mechanism (stated before any result)

Lou, Yan & Zhang (2013, RFS) *Anticipated and Repeated Shocks in Liquid Markets*
and Fleming, Liu & Nguyen (FRBNY Staff Report 1188) document a **price
concession**: dealers who must absorb a known quantity of new supply demand a
lower price going into the auction, and the price **reverses** once the supply is
placed. It is an inventory / limited-risk-bearing-capacity effect, not an
information effect.

This is the same economic shape as `fomc_drift_v1` — a scheduled resolution of a
known uncertainty, paid to whoever carries risk across it — and it is explicitly
NOT the shape that died in this project: the HPWZ data-release family
(NFP/ISM/GDP/CPI) failed because the *clock window* paid on non-event days too.
That is exactly what the control in §4 is designed to catch.

## 3. Predictions (directional, committed now)

Matched instrument for a 10-Year note auction is **ZN**, the 10-year note future.
Auction close is 13:00 ET, results ~13:02-13:05 ET.

* **H1 CONCESSION** — ZN return over the 180 min ending at auction close is
  **NEGATIVE**, and more negative than the same clock window on matched
  non-auction weekdays.
* **H2 REVERSAL** — ZN return from `result_release + 5 min` to `+180 min` is
  **POSITIVE**, and more positive than the matched control.

A result with the right magnitude but the WRONG SIGN is a kill, not a
short-the-other-way trade. The mechanism predicts the sign; reversing it would
be fitting.

## 4. Statistic, and the bar

* Primary statistic is **event minus matched control**, never the raw event mean.
  Control = the identical clock window on non-auction weekdays, same era. This is
  the test that killed HPWZ ISM/GDP after their point estimates replicated.
* Counted cells: **2** (H1, H2) on one instrument. Bonferroni alpha = 0.025,
  i.e. the project's standing **t >= 2.5** gate.
* Era split committed now: **DEV 2020-01-01 .. 2023-12-31**, **VAL 2024-01-01 ..
  2026-07-09** (end of tape). Both halves must be same-signed. One look at VAL.
* Every trade is charged a full round turn as a taker. No net-of-nothing numbers.

## 5. Data, and the guard that had to come first

* Bars: `data/processed/zn_1m_all.parquet`, rebuilt in this session from
  `data/hist_1m24_zn/` (1,838 daily files, 2020-01-02 .. 2026-07-09).
* Auctions: `data/processed/td_10y_auctions.csv`, 118 10-Year auctions with
  explicit `auction_close` / `result_release` timestamps.
* 🚨 **The `data/processed/zn_1min.parquet` already on disk is MIS-TIMEZONED and
  must not be used.** Its maintenance halt starts 17:00 ET in winter (correct)
  but 18:00 ET in summer (one hour late) — the fixed `-05:00` landmine the ledger
  records for `cost_floor.py` / `replay_hold_strategies.py`, present here too.
  The raw files put the halt at 16:00 local in BOTH seasons, which pins the
  source to DST-aware America/Chicago. Using the old file would have shifted
  every summer auction window by an hour, silently, on roughly half the sample.

## 6. Kill conditions (any one is fatal)

1. Either hypothesis's event-minus-control |t| < 2.5.
2. Dev and val disagree in sign on either leg.
3. The concession is real but smaller than the ZN round-turn cost (measured in
   §7 BEFORE the event statistics, per the ledger's standing cost-floor rule).
4. The effect is carried by a single era or a handful of auctions (Gini /
   drop-one, as in `analysis/edge_stress.py`).

## 7. Cost floor — computed BEFORE the event test

Measured on the rebuilt tape: mean |180-min ZN move| = **$130** into the
auction and **$105** after the release, against a round turn of **$17.62**
(1 tick spread + $1.00/side commission) = a **13.5% / 16.7% cost ratio**. For
comparison the ledger's viable band on MNQ is 1.9-4.2% and it ruled MCL/M6E out
at 30-70%. This is marginal before any statistic is computed, and §6.3 was
therefore live from the start.

    status: CLOSED — see VERDICT_treasury_auction_concession.md

## 8. Trial accounting

This is trial **1** of a new family. If H1/H2 survive, the pre-committed
extensions are: all coupon tenors (2/3/5/7/10/20/30y, ~109/yr) as a
**confirmation** on independent events, and MNQ as a **vehicle** test. Neither is
a second bite at the same cherry; both are declared here so they cannot be
presented later as fresh.
