# New-session prompt: the 60-120 minute intraday trend, and the cost wall in front of it

Copy everything below the line into a fresh session.

---

I want to pursue one specific thread: **there is a real, measured intraday trend
on index futures at 60-120 minutes, and it is currently worth slightly less than
my transaction costs.** I want to know whether that gap can be closed, and I want
you to tell me plainly if it cannot.

## What is already established — do not re-derive or re-litigate this

**The account.** LucidFlex 100k evaluation, `LFE100-Z96Y99ZR-TEST001`. Target
+$6,000. Max loss limit $3,000, end-of-day trailing, and the floor **locks at the
starting balance** once peak equity reaches +$3,000. Mandatory flatten 16:45 ET
Mon-Fri, reopen 18:00 ET Sun-Thu. 50% consistency rule during evaluation only,
which has been modelled and costs under half a point, so ignore it. Max size 6
minis or 60 micros.

**The measured signal.** Autocorrelation of returns on non-overlapping
regular-hours blocks:

| horizon | MNQ (6.5y) | ES (15.5y) |
|---|---|---|
| 5 min | −0.0059 (t −2.12) | −0.0129 (t −7.16) |
| 15 min | +0.0109 (2.26) | −0.0004 (−0.11) |
| 30 min | +0.0142 (2.09) | +0.0031 (0.70) |
| 60 min | **+0.0241 (2.40)** | **+0.0267 (4.09)** |
| 120 min | **+0.0342 (2.40)** | **+0.0375 (4.07)** |

Monotone in horizon, on two instruments over independent samples. At 5 minutes it
is *reversal*, which is why every breakout and opening-range system in this repo
lost money — they traded the wrong sign.

**The cost, measured rather than assumed.** I built an L3 order-book fill model
(`ml_intraday_v3/features/mbo_fill_model.py`, 11 tests) that replays the book and
walks the ladder. Round trip per contract on MNQ: $2.047 at 1 lot, $2.269 at 6,
$2.388 at 9, $2.558 at 13. Commission is $0.62 a side, which is **60% of that
cost**. Latency is irrelevant at these sizes; the 09:30-10:00 hour costs 28% more
than the afternoon at size.

**The gap.** Naive sign-following at MNQ 60 min grosses $1.857 per micro per
trade against a $2.19 round trip at the tail-legal 4 micros. Net −$0.33. At
$0.35/side it is +$0.21. At $0.25/side it is +$0.41, which is about **$53/week**
against a $520/week requirement.

**The ceiling.** Theoretical max for a sign-following trade is roughly
`corr × sd × sqrt(2/π)` ≈ $2.28 at MNQ 60 min. The naive trade already captures
$1.857, or **80%**. So better entries, filters and stops are competing for the
remaining fifth. Bear that in mind before proposing a clever overlay.

## What I want you to do

In priority order.

1. **Establish whether the cost side can move.** What is the $0.62 actually
   composed of, what do other routes and firms charge per micro contract, and
   does the funded stage differ from evaluation? This is research about the
   market for execution, not about markets. If the answer is that $0.62 is near
   the floor for prop-firm micros, say so and stop.

2. **Test whether the trend can be harvested better than naively**, given the 80%
   ceiling. Specifically worth trying, and only these: conditioning entry on
   whether the prior block's move was accompanied by volume expansion; skipping
   the first 30 minutes of the session, where cost is 28% higher; and holding
   through to the flatten rather than a fixed 60 minutes. Pre-register before
   running. Do not grid-search parameters.

3. **Check whether the signal is stronger on an instrument I have not measured.**
   I have repaired tapes for MNQ, MES, MGC, MCL, M6E and ZN. Same measurement,
   same code, one run.

4. If 1 through 3 do not produce something that clears cost by a real margin,
   **tell me the family is closed and stop.** I would rather have that than
   another marginal configuration.

## Traps that have already cost me real time

- **Never use an iid bootstrap for a pass-probability or barrier question.** It
  destroys the autocorrelation and drawdown clustering that decide first passage,
  and at small sizes it roughly doubles the apparent pass rate. Use contiguous
  historical windows. Every pass probability in my older notes is flattered this
  way.
- **Timezone landmines.** Raw Rithmic day files are DST-aware America/Chicago,
  not fixed −05:00. Verify any tape two ways before trusting it: the CME
  maintenance halt must return at 18:0x ET in *both* seasons, and the regular-hours
  volume peak must sit at 09:31 ET in *both* seasons.
- **A tolerance parameter can silently resurrect a fixed bug.** A 90-minute
  "first bar after" window re-admitted 100 weekends that a previous fix had
  correctly removed, and moved a strategy's mean by 40%. Prefer a strict window
  and drop the event.
- **Check bars-per-session-hour split by DST flag on any new tape.** A one-hour
  seasonal hole is invisible to row counts, span checks and both guards above. I
  found one that had deleted a third of the calendar.
- Cost must be charged against the *predictable* component, never against total
  volatility. A favourable cost-to-volatility ratio is necessary and never
  sufficient.

## Where things live

Compute on the Mac mini, `ssh jg@100.113.240.72`, venv `~/quant_venv`, raw tapes
in `~/.svc-3hKye0/hist_1m24*`. **Do not do multi-file data work on the laptop** —
the repo sits in iCloud Drive and reading 1,800 small files stalls for ten
minutes there versus four seconds on the mini.

Relevant code, all in the repo: `research_intraday_trend.py`,
`trend_commission.py`, `research_pure_beta.py`, `mbo_fill_model.py`,
`research_disaster_stop_v2.py`.

## Protocol

Mechanism first, pre-registered before results, dev and validation eras split in
advance, one look at validation, all trials counted, t ≥ 2.5, cost model always
applied. If a mechanism makes a prediction testable on data I already have, test
that before proposing any purchase. Tell me when something is dead.
