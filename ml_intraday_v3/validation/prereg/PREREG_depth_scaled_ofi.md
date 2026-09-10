# Pre-registration — depth-scaled order-flow imbalance on MNQ L3

**Written 2026-09-09, after a one-day feature smoke run and BEFORE the DEV screen
is executed.** The one-day numbers that motivated this are disclosed in §7 rather
than hidden, because they are the reason the hypothesis is stated the way it is.
Nothing in §2–§5 may be edited after the DEV screen runs. Amendments go in a
dated appendix.

---

## 1. Why this exists, and why it is not the killed screen again

`project_mbo_l2_vps_sep2026` records a kill: "OFI DEAD: IC ≈ 0 all horizons
(1/5/10/30s), sign hit-rate 0.455". That verdict rested on a measurement with two
defects, both now identified in code and fixed:

1. **OFI was computed by differencing one-second bar endpoints.**
   Cont-Kukanov-Stoikov define OFI as a sum of per-event contributions over every
   best-quote transition in the interval. MNQ produces ~940 book events per
   second. A unit test (`test_ofi_accumulates_every_event_not_just_bar_endpoints`)
   locks the size of the error: a bid walking up ten ticks accumulates 100 units
   of true OFI and 10 by endpoint differencing.

2. **OFI was never divided by depth.** The CKS result is `dP = beta * OFI / AD`.
   The impact coefficient is inversely proportional to available liquidity, so
   regressing raw OFI blends thin-book states, where an imbalance moves price a
   long way, with thick-book states, where it does not.

The old verdict is therefore **untestable rather than false**, and this is a
first test, not a re-test. That framing does not lower the bar: the prior on
microstructure taker alpha remains poor, for the reason in §6.

## 2. Cost floor — measured, and binding

Measured on MNQ RTH 2026-09-04 from the recorded BBO stream, not assumed:

| quantity | value |
|---|---|
| spread = 1 tick | 37.9% of RTH quote updates |
| spread = 2 ticks | 60.6% |
| mean spread | 1.64 ticks = $0.82 |
| commission (repo constant, $0.62/side) | $1.24 round trip |
| **taker round-trip cost** | **$2.06/contract = 1.03 MNQ points** |

Commission per side ($0.62) **exceeds the half-spread ($0.41)**. A perfect market
maker who always earned the full spread and was never adversely selected would
net −$0.42 per round trip. **The maker route is closed by arithmetic**, so every
hypothesis below is taker-only and must clear $2.06/contract.

## 3. Hypotheses — declared, ordered, and falsifiable

> **H1 (the mechanism).** `ofi_scaled` = OFI / average depth has materially
> higher rank correlation with forward return than raw `ofi`, and the gap widens
> as depth falls.

H1 is a test of whether the CKS relation describes MNQ at all. It is *not*
tradeable on its own.

> **H2 (the only tradeable claim).** In the low-depth tail, the conditional move
> following an extreme depth-scaled imbalance exceeds $2.06/contract net, on
> non-overlapping trades, while the high-depth bulk does not.

**Declared outcomes, binding:**

| outcome | reading | action |
|---|---|---|
| H1 fails | CKS scaling does not describe this instrument | do not interrogate H2; record the corrected-measurement result and **close the OFI family for good** |
| H1 holds, H2 fails on DEV | mechanism real, not monetisable at retail cost | close H2; the mechanism may still be used as a *filter* elsewhere, which is a separate pre-registration |
| H1 holds, H2 passes DEV | one cell goes to VAL, once | then OOE, once |

## 4. The grid — declared in full, every cell reported

Nothing may be added after the DEV run. Cell count is reported with the result.

- `signal` ∈ {ofi, ofi_scaled, signed_vol, signed_vol_scaled, queue_imbalance} (5)
- `depth_tail` ∈ {0.10, 0.25, 0.50, 1.00} trailing depth percentile ceiling (4)
- `signal_q` ∈ {0.90, 0.95, 0.99} quantile of |signal| within the bucket (3)
- `horizon_s` ∈ {1, 5, 10, 30, 60, 300} seconds (6)

**360 cells.** With that many, a t of 2 means nothing. The DEV winner is chosen
by **net dollars per contract**, and the decision rule at VAL is stated in §5.

## 5. Splits and the decision rule

- **DEV** = first 60% of available L3 session dates (2026-08-19 onward).
- **VAL** = remaining 40%. **One look**, on a single frozen cell, with the
  absolute signal threshold carried over from DEV rather than recomputed.
- **OOE** = the MBP-1 days, 2026-01 → 2026-07. A different era and a thinner
  schema, so quote-side features only. One look. The feature code path is shared
  and locked by `test_mbp1_and_mbo_paths_produce_identical_features`.

**Promotion requires all of:** VAL net > $2.06/contract equivalent (i.e. net
positive after the measured cost), VAL t ≥ 2.5, DEV and VAL agree in sign, OOE
net positive, and net/|MAE p5| ≥ 0.10. The last is the ledger's criterion: the
binding constraint is fitting the path inside a $3k trailing buffer, so rank by
premium per unit of adverse excursion, not by t.

**Trades are non-overlapping.** Once entered, no new entry until the hold
expires. Overlapping 1-second entries on a 60-second hold would inflate every t
by roughly √60, which is the standard way a screen of this shape fools itself.

**Costs are charged at the prevailing spread of the entry bar**, not an average:
long enters at the ask and exits at the bid, short the reverse, commission both
sides.

## 6. Honest prior, recorded before the result

Poor. Three independent reasons:

1. The literature is explicit that book-imbalance effects do not beat the spread
   plus fees for a taker. Multi-level OFI work says the same in as many words.
2. This repo has killed the family twice, and the neighbouring lab ran a
   seventeen-feature L3 battery (including a hidden-liquidity score) as a trade
   filter and got AUC 0.580, p=0.054, null.
3. §2 shows commission alone exceeds the half-spread, which is a brutal cost
   structure for anything at this horizon.

**What would make it interesting anyway:** the corrected measurement is genuinely
different, and nobody has looked at the low-depth conditional. That is worth one
properly-run screen and no more.

## 7. Disclosure — the one-day smoke run that preceded this

2026-09-03 RTH only, n=23,160 one-second bars, Spearman IC vs forward mid move:

| signal | 1s | 5s | 10s | 30s | 60s |
|---|---|---|---|---|---|
| ofi, endpoint method (the killed screen's estimator) | 0.0017 | −0.0042 | −0.0045 | −0.0081 | −0.0096 |
| ofi, event-level (corrected) | 0.0102 | −0.0049 | −0.0061 | −0.0062 | −0.0065 |
| ofi_scaled | 0.0096 | −0.0050 | −0.0063 | −0.0058 | −0.0055 |
| signed_vol | 0.0012 | −0.0159 | −0.0131 | −0.0173 | −0.0115 |
| queue_imbalance | 0.0137 | 0.0034 | 0.0040 | 0.0017 | −0.0028 |

**What this already says, on one day:** the correction is real (1-second IC rises
6×), but **depth scaling does not widen the gap** — `ofi_scaled` tracks `ofi`
almost exactly, and in the thinnest 10% both go *more negative*, not more
positive. That is evidence against H1 as stated. It is one day, it is not the
verdict, and the DEV screen runs as specified regardless.

**Recorded now so it cannot be claimed later:** if the DEV screen reproduces this
shape, H1 fails and §3's binding clause closes the family. Having built the
extractor is not a reason to keep searching it.

## 8. Artifacts

- feature code `ml_intraday_v3/features/depth_scaled_flow.py` (20 tests)
- screen `ml_intraday_v3/diagnostics/depth_scaled_screen.py`
- features `data/processed/mbo_features/features_YYYYMMDD.parquet`
- results `ml_intraday_v3/runs/depth_scaled_screen/{ic,cells,meta}_{dev,val}.*`

Data is READ from the research VPS and processed locally. Nothing is built or
run on that machine.

---

## Appendix A — amendments after inventorying the pulled corpus (2026-09-09)

Written after the 13GB mirror completed and BEFORE the DEV screen ran. Disclosed
as an amendment rather than folded into §5, per the rule at the top.

### A.1 The declared OOE set does not exist as described. Corrected.

§5 named "the MBP-1 days, 2026-01 → 2026-07" as the out-of-era check, on the
strength of 56 files named `mbp1_2026-*.parquet`. On inspection those files cover
**09:29:00 → 09:30:14 ET only — a 74-second window at the cash open — and they
are NQ, not MNQ.** They are an opening-auction study someone else ran. They
cannot test a 1-second signal across an RTH session, and the instrument is
different, so §5's OOE as written is void.

**Replacement OOE, declared now:** `mnq_l2_data/reference/bbo_MNQ_*.parquet` —
**9 MNQ days, 2026-06-15 → 2026-07-31**, ~4.5M best-quote updates per day,
schema `recv_ns bid_px bid_sz bid_ord ask_px ask_sz ask_ord`. Same instrument,
6-10 weeks before the DEV era, and it feeds `replay_bbo_from_mbp1` directly.

**Two limitations, recorded before use, both of which weaken the check:**

1. These files run from the prior evening to roughly **10:32 ET**, so the OOE
   window is **09:30–10:30 only**, not the full session. The first hour is the
   most volatile and thinnest part of the day, so it is a favourable window for a
   thin-book hypothesis. An OOE pass here is therefore weaker evidence than a
   full-session pass would have been, and must be reported as such.
2. No trade stream, so `signed_vol`, `signed_vol_scaled` and the sweep columns
   **cannot be checked out of era at all**. If the DEV/VAL winner is one of
   those, the OOE step is unavailable and the cell cannot be promoted under §5.

### A.2 Corpus actually held, for the record

| set | instrument | dates | content |
|---|---|---|---|
| L3 depth + fill-level trades | MNQ | 2026-08-19 → 2026-09-09, 15-16 days | full book, trades with aggressor |
| reference BBO | MNQ | 9 days, 2026-06-15 → 2026-07-31 | best quote only, to ~10:32 ET |
| daily BBO | MNQ | ~25 days | best quote, small files |
| mbp1 open window | **NQ** | 56 days, 2026-01 → 2026-07 | 74 s at the open, unusable for §3 |
