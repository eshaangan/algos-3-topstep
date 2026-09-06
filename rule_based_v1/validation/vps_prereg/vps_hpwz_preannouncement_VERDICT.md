# vps_hpwz_preannouncement — NO-GO on both events (2026-09-05)

Hu, Pan, Wang & Zhu (NBER w25817) pre-announcement drift, tested per
`vps_hpwz_preannouncement.yaml` (pre-registered before any run). This was step 3
of the Sep-2026 paper pull: use the HPWZ screen to pick at most two new event
types, pre-register, one shot each.

**Events chosen by the screen, before looking at data:** HPWZ's documented
positives are exactly NFP, ISM and GDP. NFP is already dead here (both eras,
2026-07-10). CPI and PPI never appear in their paper at all, which is consistent
with this project's own CPI kill. That leaves **ISM and GDP**, neither previously
tested.

**Run:**
```
python3 rule_based_v1/validation/build_es_eth_1min.py \
    --csv data/raw/GLBX-20251220-LWFB9HCEL5/glbx-mdp3-20100606-20251219.ohlcv-1m.csv \
    --out data/processed/es_1min_eth_frontmonth.parquet
python3 rule_based_v1/validation/fetch_bea_gdp_calendar.py --pages 45 \
    --out data/processed/bea_gdp_advance_releases.csv
python3 rule_based_v1/validation/research_hpwz_preannouncement.py \
    --instrument {es,mnq} --event ism --out runs/hpwz_ism_{es,mnq}.json
python3 rule_based_v1/validation/research_hpwz_preannouncement.py \
    --instrument {es,mnq} --event gdp \
    --calendar data/processed/bea_gdp_advance_releases.csv --out runs/hpwz_gdp_{es,mnq}.json
```

Window is the paper's: previous session's 16:00 ET close → 5 minutes before the
release. Long only. 2 contracts, standard cost model.

## Result: the point estimates replicate, the edge does not

HPWZ report ISM +9.1 bps and GDP +7.5 bps (Sep 1994 – May 2018). On ES 2010–2025
we get **ISM +8.70 bps** (n=184) and **GDP +5.32 bps** (n=60). The magnitudes land
almost exactly where the paper says they should. That is a genuine replication of
the *statistic*.

It is not an edge, because the same clock window on non-event days pays too:

| instrument | event | n | event bps | control bps | event − control | p |
|---|---|---|---|---|---|---|
| ES | ISM | 184 | +8.70 | +2.70 | t=+0.22 | 0.83 |
| ES | GDP | 60 | +5.32 | +2.95 | t=+0.35 | 0.73 |
| MNQ | ISM | 77 | +9.90 | +4.92 | t=+0.06 | 0.95 |
| MNQ | GDP | 25 | +15.68 | +5.53 | t=+0.45 | 0.66 |

No cell reaches the pre-registered t≥2.5, and no cell separates from its control.
Nothing here would have been promoted even before the era split.

## The era split kills it independently

HPWZ's sample ends May 2018, so our val half is genuinely out of their sample.

- **ES ISM:** dev +12.19 bps vs control +1.21 (diff t=1.10, p=0.27) → val +5.42 vs
  control +4.11 (diff t=−0.18). Gone.
- **ES GDP:** dev −0.92 bps (t=−0.86) → val +11.55 bps (t=0.76). Sign flip.
- **MNQ ISM:** dev event t=+2.02 (diff p=0.073, the closest anything came) → val
  **−5.85 bps**, negative.
- **MNQ GDP:** dev +32.27 bps → val −13.83 bps. Sign flip.

Every instrument-event pair that looks alive in dev is dead or inverted in val.

## What the control actually shows, and why it does not reopen anything

The control window — prior 16:00 ET to the pre-release mark on *every* day — is
significantly positive on MNQ: ISM-clock control net **+$35.03/day, t=+2.16**
(n=1546); GDP-clock control (pure overnight, 16:00→08:25) net **+$34.51/day,
t=+2.50** (n=1597). The control is *stronger and better-powered than either event*.

That is the overnight equity drift already recorded in
`project_structural_taker_verdict_jul2026` as real but unharvestable, and this run
corroborates that verdict rather than reopening it: MAE p05 is −$1,136 and the
worst is **−$4,506** per 2 MNQ, against a $3k buffer. The drift is real; the path
still busts the account.

This is exactly why the placebo was pre-registered. Without it, ES ISM at +8.70 bps
matching the paper's +9.1 bps would have looked like a clean replication worth
paper-trading. Two thirds of it is just being long overnight.

## Verdict

Both events NO-GO. The HPWZ screen is now spent: all three of its documented
positives (NFP, ISM, GDP) are dead in this project. `fomc_drift_v1` remains the
only member of the pre-announcement family that survives here, and the reason is
visible in these numbers — its window is a *policy decision*, not a data release,
which is the distinction the ledger already recorded on 2026-07-09.

## Byproducts worth keeping

- `data/processed/es_1min_eth_frontmonth.parquet` — **ES 1-min ETH, front-month
  pinned by session volume, 2010-06-06 → 2025-12-19, 5.43M rows, 4,162 trade
  dates, 40 contracts.** The existing `es_bars_2010_2025.h5` is RTH-only and cannot
  express any overnight window; this can. Spread symbols dropped, one contract per
  CME trade date so rolls switch at a session boundary.
- `data/processed/bea_gdp_advance_releases.csv` — 62 GDP advance-estimate dates
  2010–2025 with BEA's own release timestamps, advance estimates only (second and
  third estimates revise already-seen numbers and are not the HPWZ event). 4/year
  except 2019 and 2025, both government-shutdown years.
