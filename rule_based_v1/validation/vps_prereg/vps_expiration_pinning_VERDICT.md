# vps_expiration_pinning — NO-GO (2026-09-05)

Golez & Jackwerth (2012) expiration-day strike pinning, tested per
`vps_expiration_pinning.yaml` (pre-registered before any run).

**Run:**
```
python3 rule_based_v1/validation/research_expiration_pinning.py --instrument es  --out runs/pinning_es.json
python3 rule_based_v1/validation/research_expiration_pinning.py --instrument mnq --out runs/pinning_mnq.json
```

Samples: ES 5-min RTH 2010-06-07 → 2025-12-19, 3,870 sessions, 124 serial + 63
quarterly expirations. MNQ 1-min 2020-01-02 → 2026-07-09, 1,681 sessions,
52 serial + 26 quarterly.

## Verdict

Dead on both instruments, at every grid, decision time, band and distance
threshold. Nothing cleared the `min_t_stat: 2.5` gate; nothing came close.

### Part 1 — clustering (the paper's own statistic)

| instrument | grid | serial exp vs control | quarterly (placebo, must be flat) |
|---|---|---|---|
| ES | 5.0 (paper-faithful) | 16.9% vs 13.6%, **z=+1.04, p=0.30** | 6.3% vs 13.9%, z=−1.70 |
| ES | 25.0 | 8.1% vs 15.3%, z=−2.22 (wrong sign) | 15.9% vs 14.7%, z=+0.27 |
| ES | 50.0 | 11.3% vs 14.7%, z=−1.06 | 15.9% vs 16.2%, z=−0.06 |
| MNQ | 50.0 | 23.1% vs 11.5%, z=+2.50 | 23.1% vs 8.8%, **z=+2.41 — placebo fails** |
| MNQ | 100.0 | 25.0% vs 11.2%, z=+3.00 | 15.4% vs 11.9%, z=+0.53 |

The paper-faithful ES specification has the right sign and is insignificant over
15.5 years. Every wider ES grid flips to anti-pinning. MNQ shows elevated
clustering, but at grid=50 the quarterly placebo is **as strong as the serial
treatment** (z=+2.41 vs +2.50) — i.e. it is not the mechanism, since the paper's
own structural argument is that quarterly expirations cannot pin the front
future. The one cell with a clean placebo separation (MNQ grid=100, z=+3.00) has
a *tradeable* version of −$54.54/event.

### Part 2 — tradeability (what actually decides GO)

Enter at T toward the nearest strike, exit at settlement, 2 contracts, standard
cost model ($7.48 RT on ES, $4.48 on MNQ).

Best cell out of 18: ES grid=25, T=14:00 → +$9.19/event, **t=+0.84**. Its own
quarterly placebo is −$53.43 at t=−2.83. Dev/val halves flip sign in most cells
(ES grid=50 T=15:00: dev −$19.06 t=−4.52, val +$25.25 t=+1.76).

### Part 3 — distance sweep

A real magnet must pull harder the further price sits from it. Across 72 cells
(3 grids × 3 times × 4 distance thresholds × 2 instruments) there is no monotone
strengthening in `min_dist_frac` anywhere. The largest |t| values are negative
(MNQ grid=25 T=14:00 d≥0.75: −$185.48, t=−2.64 — pinning's exact opposite).

## Why — and this was written down before the run

The strike grid never widened with the index. Golez & Jackwerth measure
Nov 1992 – Nov 2009, when a 5-point grid on an S&P of 400–1550 was **0.3%–1.2%**
wide. At ES 6900 the same 5-point grid is **0.072%** — roughly 6× narrower than
their sample median.

That collapses the mechanism's economics regardless of whether the hedging flow
still exists:

- half a grid interval = **$25.00** gross, the absolute theoretical maximum;
- round-turn cost = **$7.48**, or 30% of that ceiling;
- best realized gross = **+$4.56**, 18% of the ceiling and *below* the cost line.

So the honest reason is economic, not statistical. Even a fully intact pinning
force on today's grid could not pay for its own execution. The wider 25/50-point
grids were tested as a proxy for where option open interest actually
concentrates, and they are where the sign flips to anti-pinning — those grids buy
room to pay at the cost of no longer being strikes.

Secondary and unmeasured here: since ~2022 ES/NQ have daily-expiring options, so
"expiration day" is no longer a monthly event and the serial-vs-quarterly
contrast the paper relies on is structurally diluted.

## Follow-ups explicitly NOT worth a slot

- Re-running with real option open-interest strikes. The ceiling arithmetic above
  binds on grid width alone; better strike identification cannot lift a $25 cap
  over a $7.48 cost by enough to clear t=2.5.
- Anti-pinning as a strategy. The negative cells do not replicate across
  instrument, era or decision time.
