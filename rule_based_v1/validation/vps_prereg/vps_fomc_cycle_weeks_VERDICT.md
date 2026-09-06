# FOMC-cycle even-week effect (CMVJ 2019) — NO-GO (2026-09-05)

The item flagged as the highest-value unobtained paper in the Sep-2026 pull.
Obtained, read, and tested. It does not survive.

**Papers** (in `ml_intraday_v3/research papers/edge_hunt_sep2026/`):
- `cieslak_morse_vj_stock_returns_fomc_cycle.pdf` — the CMVJ working paper
  (Feb 2018 draft), author-hosted on Adair Morse's Berkeley faculty page. The
  published JF 2019 version is closed access with no OA location anywhere.
- `uppal_does_fomc_cycle_still_drive_returns.pdf` — Uppal (Imperial, Mar 2025),
  a direct out-of-sample challenge, acknowledging Vissing-Jørgensen herself.

**Run:**
```
python3 rule_based_v1/validation/fetch_fomc_calendar.py --start 2010 --end 2025 \
    --out data/processed/fomc_announcements.csv
python3 rule_based_v1/validation/research_fomc_cycle_weeks.py --instrument es  --out runs/fomc_cycle_es.json
python3 rule_based_v1/validation/research_fomc_cycle_weeks.py --instrument mnq --out runs/fomc_cycle_mnq.json
```

## The claim, and why it mattered here

CMVJ: since 1994 the entire US equity premium is earned in weeks 0, 2, 4 and 6 of
FOMC cycle time. That would have been the **velocity fix** for `fomc_drift_v1`,
whose stated limit is 8 events/yr — even weeks give ~26 windows/yr.

The catch, which CMVJ state themselves: week 0 spans days −1..+3 and therefore
*contains* the Lucca-Moench pre-FOMC drift. In their words, that drift "is thus
part of a broader bi-weekly pattern." So week 0 is not new — it is
`fomc_drift_v1` under another name. The only new velocity is **weeks 2, 4, 6**,
and that is the quantity tested below.

## Uppal's four findings

1. Even-week result does not hold out-of-sample; loses significance from **2004**,
   coefficient negative or zero after the financial crisis.
2. In 1994–2003, where it does hold, a few outlier *days* drive it; remove them
   and it dies there too.
3. No even-week effect in Treasuries, Fed Funds futures or Eurodollar futures —
   markets that must respond if biweekly Fed leaks were the mechanism.
4. The mechanism itself ended: Board of Governors meetings stopped being
   biweekly after 2004.

## Independent verification on our own tape

ES 5-min RTH 2010-06-07 → 2025-12-19 (3,870 sessions, 124 announcements in
range, 99.4% of sessions mapped to a cycle week), cycle time per CMVJ / Uppal
Table 1 counted in trading days. Honest note: this was run *after* reading
Uppal, so it is confirmation rather than a blind test — but the specification
comes from CMVJ's own table, and the finding is a kill, which is the
conservative direction.

**Close-to-close (CMVJ-equivalent), ES:**

| era | even (0,2,4,6) vs odd | weeks (2,4,6) vs odd |
|---|---|---|
| FULL 2010–2025 | −0.34 pts, **t=−0.31, p=0.76** | −0.28 pts, t=−0.23, p=0.82 |
| 2010–2017 | +1.63 pts, t=+2.48, p=0.013 | +1.94 pts, t=+2.66, p=0.008 |
| 2018–2025 | −2.23 pts, **t=−1.07** | −2.40 pts, **t=−1.02** |

Over the full 15.5 years odd weeks actually pay *more* than even weeks
(odd t=+2.08 vs even t=+1.48). The first half still catches a decaying tail of
the effect; the second half flips sign outright. That is Uppal's result,
reproduced on our own data with a calendar built from the Fed's own pages.

**RTH open→close, ES:** same shape — dev (2,4,6) vs odd t=+2.87 p=0.004,
val t=+0.43 p=0.669.

**MNQ 2020-01 → 2026-07 (1,681 sessions, 47 announcements):** nothing at all.
Even vs odd t=−0.06 (c2c), t=+0.11 (intraday); weeks (2,4,6) vs odd t=−0.65 and
t=+0.12.

### The one cell that stayed alive, and why it is not a strategy

ES **week 2, intraday** is positive in every split: dev t=+2.60, val t=+2.10,
full t=+2.83 (+$27.77/day per 2 MES). It is the only cell that does not collapse.
It still fails as a candidate:

- 16 cells were tested (8 weeks × 2 return measures); at that count a single
  |t|≈2.8 is roughly what noise produces. Bonferroni needs t≈3.0.
- It does **not** replicate cross-instrument — MNQ week 2 intraday is t=+1.33.
- Its val half is t=+2.10, below the project's t≥2.5 gate.
- Its own close-to-close twin is weaker (dev t=+2.18, val t=+1.00), so the
  effect is not stable across the two obvious ways of measuring the same week.

Recorded as observed, not promoted. Re-testing it would need a differentiated
mechanism hypothesis for why week 2 specifically, not a re-run.

## Verdict

`fomc_drift_v1` stays exactly as it is: 8 events/yr, week 0 only. There is no
biweekly extension to buy. The velocity limit noted in the ledger is real and
this does not lift it.

## Byproduct worth keeping

`data/processed/fomc_announcements.csv` — 127 scheduled FOMC announcement dates
2010–2025, parsed from federalreserve.gov statement-release URLs, excluding
conference calls, unscheduled meetings, notation votes and the cancelled
2020-03-17/18 meeting. 8/year except 2020 (7). This replaces the model-memory
dates the ledger flagged as unverified, and `fetch_fomc_calendar.py` regenerates
and extends it.
