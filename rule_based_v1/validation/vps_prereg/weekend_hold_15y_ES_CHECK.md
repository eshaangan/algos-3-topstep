# weekend_hold_v1 / monday_rth_v1 on 15.5 years of ES — regime warning (2026-09-05)

Not a new strategy. A robustness check on the two live survivors, made possible by
`data/processed/es_1min_eth_frontmonth.parquet` (built earlier today). Until now
every overnight test in this project was capped at MNQ 2020-2026, because
`es_bars_2010_2025.h5` is RTH-only. The decade before the strategy was discovered
was simply invisible.

**Run:**
```
python3 rule_based_v1/validation/replay_hold_strategies.py \
    --source es-eth --econ mes --by-era \
    --data-dir data/processed/es_1min_eth_frontmonth.parquet --out-prefix runs/es15y
```
Same `replay_weekend` / `replay_monday` functions the live paper runners are
reconciled against — extended, not reimplemented, with the MNQ defaults verified
arithmetically unchanged ($4.48 round turn, 2 contracts, $2/pt).

## Result

ES 2010-06 → 2025-12, 2 MES, entry/exit identical to the live spec.

| strategy | full sample | pre-2020 | 2020+ |
|---|---|---|---|
| weekend_hold_v1 (Sun 18:00 → Mon 15:59) | n=710, +$19.58, **t=+1.61** | n=438, **−$8.80, t=−1.09**, WR 49.8% | n=272, +$65.28, t=+2.27, WR 59.2% |
| monday_rth_v1 (Mon 09:30 → 16:00) | n=728, +$9.03, **t=+1.03** | n=450, **−$11.18, t=−1.80**, WR 50.9% | n=278, +$41.76, t=+2.04, WR 57.9% |

Pre-2020 by year, weekend_hold: 2010 −8, 2011 −16, 2012 −1, 2013 −3, 2014 −14,
2015 +9, 2016 −2, 2017 +11, 2018 −43, 2019 −20. **Eight of ten years negative or
flat**, across the strongest equity bull market in the sample.

## The control: the weekday ranking inverted

To separate "weekends stopped being special" from "the whole tape was different",
the same prior-18:00 → 15:59 hold was run for every weekday.

| era | Mon | Tue | Wed | Thu | Fri | Mon − other |
|---|---|---|---|---|---|---|
| pre-2020 | **−$6.64** | +$7.77 | +$4.17 | +$11.14 | −$0.09 | −$12.41, t=−1.40 |
| 2020+ | **+$67.00** | −$12.68 | +$28.67 | −$33.10 | +$5.61 | +$69.85, t=+2.19, p=0.029 |

Pre-2020, Monday was the **worst** weekday of the five. Post-2020 it is the best.
This is not a level shift in the whole tape — the ordering flipped.

## What this does and does not establish

**Does:** on the S&P tape the weekend effect is a post-2020 phenomenon, not a
durable structural premium. n=448 pre-2020 Mondays is a large sample, so t=−0.83
is a real null rather than insufficient power. The full 15.5-year sample gives
t=+1.61 and t=+1.03, both below this project's t≥2.5 gate.

**Does not:** prove the MNQ version is dead. These are different instruments and
there is no MNQ data before 2020, so this is the only available window on that
decade. Forward weekends remain the actual test, and the existing kill rule
(8 forward weekends with mean < 0) still stands.

**But it does damage the mechanism story.** The ledger's case for
`weekend_hold_v1` is an *equity* weekend-risk premium, supported by cross-asset
placebos (gold, crude, euro all flat) showing the effect is equity-specific. If
that is what it is, it should be present in the S&P too. It is — for six years,
and not for the ten before them. A risk premium that pays nothing across
2010-2019 while equities compound is hard to call compensation for carrying
weekend gap risk.

## Recommended handling

1. Do not treat n=220 MNQ weekends as a 7-year structural result. The effective
   sample for the mechanism is the post-2020 regime on both instruments.
2. Size to a ~6-year effective sample, not a structural constant. The Monte Carlo
   pass probability (63.1% at 2 micros) is conditioned on post-2020 statistics.
3. Keep the forward paper weekends running — they are now the only clean evidence.
4. If a pre-2020 MNQ tape ever becomes reachable (Rithmic probe found ≥2020 only),
   re-run this exact check on it. That is the test that would settle it.
