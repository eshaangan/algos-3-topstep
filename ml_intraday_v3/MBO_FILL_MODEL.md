# Measured fill cost for MNQ — replacing the $2.06 assumption

    built:   2026-09-09
    module:  ml_intraday_v3/features/mbo_fill_model.py  (11 tests)
    driver:  rule_based_v1/validation/research_feasible_region_measured.py
    runs:    runs/fill_costs.parquet (9.5M priced orders), runs/feasible_measured.json
    data:    15 MNQ L3 days 2026-08-19 .. 2026-09-09, ~35M book events/day

## What it does

Replays the full L3 order book and walks the ladder, so the cost of a marketable
order of Q contracts is measured rather than assumed:

    effective round trip (Q) = buy_vwap(Q) − sell_vwap(Q) + 2 × commission

`latency_ns` moves the arrival away from the decision, so the order fills against
the book it MEETS rather than the one it SAW.

## Validation — this is the part that matters

The repo's standing cost figure came from averaging the QUOTED spread. Before
trusting a replacement, the replayed book has to be shown correct.

1. **It reproduces the established number at the size where it should.** Q=1
   mean round trip is **$2.047** against the repo's $2.060, and mean spread is
   **1.638 ticks** against the recorded 1.64.
2. **Trade containment.** 85.3% of 1,591,808 RTH trades print **exactly at a
   replayed touch**, split 42.75% ask / 42.55% bid. The symmetry rules out a
   side bias.
3. **Clock alignment.** Containment peaks at exactly **zero** offset between the
   trade and depth streams and falls away on both sides, so the two streams
   share a clock.
4. **Causal ordering.** Shifting the query 200µs later collapses at-a-touch to
   57% while the mean outside-book distance FALLS. That is the book correctly
   showing liquidity consumed immediately after each print.

⚠️ **The `bbo_MNQ_*.parquet` files cannot validate this and should not be used
for it.** They carry ~102,700 rows/day against the book's ~35.6M, i.e. roughly
4 updates/second against ~1,400. At 1-second marks they disagree with the replay
40% of the time purely from their own resolution. Their aggregate spread (1.673
ticks) does agree with the replay (1.638), and the signed difference has median
exactly 0, which is the correct reading: the reference is coarse, not wrong.

## Result — measured round-turn cost, $/contract

| Q | median | mean | p95 | p99 |
|---:|---:|---:|---:|---:|
| 1 | 2.240 | **2.047** | 2.240 | 2.740 |
| 6 | 2.240 | 2.269 | 2.573 | 2.990 |
| 9 | 2.351 | 2.388 | 2.796 | 3.240 |
| 13 | 2.509 | 2.558 | 3.048 | 3.586 |
| 20 | 2.765 | 2.844 | 3.490 | 4.115 |
| 60 | 4.040 | 4.205 | 5.515 | 6.515 |

**The $2.06 assumption is right for ONE lot and wrong above it.** At the 9-13
contracts the feasibility map wants to trade, the book charges 16-24% more.

**Latency is irrelevant here.** Q=13 costs $2.558 at 0ms and $2.558 at 50ms.
For MNQ at these sizes the book is deep and stable enough that arrival timing
does not change the fill, so the whole assumed-vs-measured gap is DEPTH, not
timing. That is a useful negative: it closes "we are being picked off on
latency" as an explanation for anything.

**Time of day is not.** At Q=13 the 09:30-10:00 hour costs **$3.115** against
**$2.44** in the afternoon, a 28% penalty. At Q=1 the same comparison is 2.22 vs
2.01. Size and the open compound.

## Consequence for the feasibility map

The map charged a flat $2.06 at every horizon while sizing each one at 3-9
contracts, so it understated cost exactly where it sized largest. Rebuilt on the
measured curve:

| horizon | size | f assumed | f measured | harder by |
|---|---:|---:|---:|---:|
| 15m | 9 | 5.40% | 6.10% | **+13.1%** |
| 30m | 6 | 5.10% | 5.41% | +6.2% |
| 60m | 4 | 6.36% | 6.50% | +2.2% |
| 120m | 3 | 9.24% | 9.30% | +0.7% |
| 240m | 2 | 15.12% | 15.13% | +0.1% |

**The correction bites hardest at the short end, which is where the map claimed
the open zone was.** At 60 minutes and beyond it is noise.

## ⚠️ The map's own inputs could not be reproduced

`research_feasible_region.py` is lost from the repo (see the untracked-validation
problem), so only its recorded outputs survive. Recomputing directly from
`mnq_1m_all.parquet`, 2020-2026 RTH:

| statistic | ledger records | recomputed here |
|---|---:|---:|
| 15m mean \|move\| | $28.31 | **$46.41** |
| 15m adverse p05 | −$88 | **−$149** (long-only) / **−$183** (two-sided) |

Both recorded figures are smaller by a consistent factor of about 1.65. I cannot
determine why without the script and am not going to guess. The resulting `f`
values happen to land close (15m 8.0% vs 6.1%, 60m 6.5% vs 6.5%) because the
discrepancy partly cancels in the ratio — but **the map's inputs are unverified,
and its sizes are the load-bearing part.** Treat the size column with suspicion
until the script is recovered or rewritten.

## Limits of the model

Not modelled: the reaction of other participants to the order itself beyond the
sweep, and hidden or iceberg liquidity, which this Rithmic feed cannot express.
Those push the true cost in OPPOSITE directions, so this is an estimate rather
than a bound. The book also starts cold, so deep levels are understated early;
a 15-minute warmup is applied and the first bars of the session are excluded.
