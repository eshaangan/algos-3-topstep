# DATA REPAIR + CONSEQUENCE — the winter Sunday hour, and what it does to the book

    date:    2026-09-10
    scripts: fetch_sunday_gap.py, weekend_leg_repaired.py, weekend_winter_hole.py,
             weekend_final_book.py, weekend_sunday_audit_all.py
    runs:    runs/weekend_repaired.csv
    tape:    data/processed/mnq_1m_all.parquet REPLACED (old kept as
             mnq_1m_all.DAMAGED_backup.parquet)

## Root cause — a guard meant to skip holidays deleted a third of the calendar

`fetch_mnq_vol.py` requested **one UTC day at a time** and kept the result only
if `len(bars) > 100`. A Sunday UTC day contains nothing but the Globex reopen,
and the UTC window ends an hour earlier in ET during winter:

    summer  Sun 00:00-24:00 UTC = Sat 20:00 -> Sun 20:00 EDT  ->  18:00-20:00 ET  ~121 bars  KEPT
    winter  Sun 00:00-24:00 UTC = Sat 19:00 -> Sun 19:00 EST  ->  18:00-19:00 ET   ~61 bars  DISCARDED

**Every non-DST Sunday therefore lost the 18:00-19:00 ET hour — which is exactly
the `weekend_hold_v1` entry.** 112 of 126 non-DST Sunday files were absent.

**The same guard is in every day fetcher and damaged every instrument tape**
(zn, mgc, mcl, m6e, mes all missing 112-118 non-DST Sundays). All six are now
patched to `len(bars) >= 30`, with the reason in a comment.

Neither standing tz guard catches this and both passed throughout: halt-return
18:01 ET and RTH-open volume peak 09:31 ET, in both seasons. **A one-hour
seasonal hole is invisible to row counts, span checks and both guards. Check
bars-per-session-hour split by DST flag.**

## Repair

113 Sundays re-fetched, 4 correctly skipped as genuinely closed (Christmas Eve
and New Year's Eve Sundays), 0 errors. Staged to a separate directory, checked
for collisions (0), then merged. Sunday 18:00 ET bars per session went
**0.0 -> 59.0** on non-DST Sundays; total Sunday bars 299 -> 359 against DST's
358. Both tz guards still pass.

## 🚨 CONSEQUENCE — the weekend edge is DST-only, and the worst weekend was missing

The 100 recovered winter weekends are an **accidental but genuine out-of-sample
season**: no prior test, tuning or selection ever saw them.

| sample | n | mean $/wk @2 micros | t |
|---|---:|---:|---:|
| DST (the entire validated history) | 204 | $225.66 | 3.92 |
| **non-DST (NEW)** | **100** | **−$16.01** | **−0.18** |
| all weekends | 304 | $146.16 | 3.00 |

**The premium is absent in winter.** There is no mechanism reason to expect a
DST effect — the known equity seasonal ("sell in May") points the other way — so
read this as evidence against the strategy, NOT as a licence to add a
DST-only filter. Filtering to the season that worked, after seeing the season
that did not, is the snooping loop.

**And the tail got worse: the single worst weekend in the sample,
2020-03-15 (COVID), was one of the deleted files.** Worst trade is now
**−$1,763 at 2 micros = −$881/micro**, against the frozen spec's −$603/micro —
46% worse, and 59% of the $3,000 MLL in one weekend.

Stress cuts, re-run on the full tape: excl 2026 **t=2.28**; excl 2020+2026
**t=1.40**; 2021-2024 only **t=0.78** (mean $39.76). Top 20 of 304 weekends =
**87%** of all P/L.

## 🛑 P(pass) FALLS ~20 POINTS — the frozen spec was measured on damaged data

Same MC, same $600 stop, weekend x2 + FOMC x2:

| horizon | frozen spec (damaged tape) | repaired tape |
|---|---:|---:|
| 16 weeks | 50.4% | **23.4%** |
| 32 weeks | **73.4%** | **53.1%** |
| 52 weeks | — | 62.9% |

Two independent causes, both artifacts of the same bug: the winter half pays
nothing (mean −42%), and the true worst weekend was missing (tail +46%).

**DO NOT DEPLOY THE FROZEN SPEC.** Its 73.4% was computed on a tape missing a
third of the calendar and its worst historical event. The book is not dead, but
it is a coin flip over 32 weeks, not three-in-four.
