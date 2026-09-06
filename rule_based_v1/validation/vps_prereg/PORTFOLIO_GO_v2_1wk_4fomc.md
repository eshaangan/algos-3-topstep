# GO v2 — weekend×1 + FOMC×3 at the prior-close entry (2026-09-05)

> **CORRECTED 2026-09-05 after checking the deployed runner.** This file originally
> recommended FOMC×4. That is wrong and must not be traded: the worst historical
> FOMC event at 4 micros is **−$3,286, which exceeds a $3k cushion outright** — a
> single-event account kill. The MC's 92.3% hid this because it lets the account
> bank profit before the tail arrives. FOMC×**3** (worst −$2,464 = 82% of cushion,
> no single-event kill) gives **95.4% / 14wk**, which is better on P(pass) than ×4
> and nearly as fast. Recommendation below is ×3.
>
> Also corrected: "cut the weekend runner from 2 to 1" was already done —
> `ladder_live_runner.ladder_size()` starts WK at 1 micro.

Supersedes `PORTFOLIO_GO_1wk_2fomc.md`. Beats it on **both** requested axes:
**12 weeks instead of 15** (20% faster) and **more robust under decay**, at a cost
of 4pp on the optimistic scenario.

**Run:**
```
python3 rule_based_v1/validation/research_fed_events.py --instrument mnq --out runs/fed_events_mnq.json
python3 rule_based_v1/validation/research_portfolio_mc.py --fomc-entry-hour 16 --out runs/pf_h16.json
```

## What changed, and why it is not a fit

Two independent changes, neither scanned for PnL:

**1. FOMC entry moves from 14:00 to 16:00 on the prior session.** Profiling where
the drift actually accrues (9 entry points, MNQ, n=47) shows it is *front-loaded*:
by the announcement-day open only $63 of the $354 is left. The prior-session
**close** entry keeps $334 of the $354 while cutting MAE p50 from −$240 to −$152
and MAE p05 from −$1,240 to −$893 — efficiency 0.286 → **0.374**.

That entry is not a mined window: it is HPWZ's canonical "previous day's close",
i.e. the other literature-specified entry for exactly this family. The comparison
is between the two published conventions (Lucca-Moench 24h vs HPWZ prior-close),
not between 9 free parameters.

**2. FOMC size 2 → 4 contracts.** Better MAE is what buys this. It is a plain
point on the risk/speed grid, not a fitted parameter, and it improves things at
*both* entries — which is why the change is credible rather than lucky.

## The grid (MNQ, $3k target, $3k EOD-trailing buffer)

| config | entry | $/wk | P(pass) | median | P(pass) if weekend → 0 |
|---|---|---|---|---|---|
| 1wk+2fomc *(GO v1)* | 14:00 | $186 | 96.4% | 15wk | 62.3% |
| 1wk+2fomc | 16:00 | $182 | **96.8%** | 15wk | 62.0% |
| 1wk+4fomc | 14:00 | $239 | 86.9% | 12wk | 61.2% |
| **1wk+4fomc** | **16:00** | **$232** | **92.3%** | **12wk** | **64.6%** |
| 0wk+4fomc | 16:00 | $100 | 84.1% | 26wk | **84.1%** (no weekend exposure) |

## Why 1wk+4fomc @ 16:00 is the pick

- **Faster: 12 weeks vs 15.** That was the requested axis and velocity is the
  binding constraint, since FOMC only fires 8×/yr.
- **More decay-robust than the incumbent**, which is the risk that actually
  matters here: at weekend×0.00 it holds **64.6%** against GO v1's 62.3%. Adding
  FOMC size shifts the account's dependence further onto the leg that passed a
  sealed holdout and away from the leg with the documented pre-2020 problem.
- The entry change is worth +5.4pp at 4 contracts (92.3% vs 86.9%) and only
  +0.4pp at 2 contracts — consistent with the mechanism, since MAE binds harder
  as size rises.

If you want maximum certainty rather than speed, `1wk+2fomc @ 16:00` is a strict
improvement on GO v1 (96.8% vs 96.4%, same 15wk) — though +0.4pp is only ~3 MC
standard errors and I would not switch for that alone.

If the weekend regime worry dominates everything, `0wk+4fomc @ 16:00` is 84.1%
**independent of the weekend leg entirely**, at 26 weeks.

## Caveats, stated plainly

- **Still not a new edge.** This session tested mega-cap earnings, FOMC minutes,
  euro-open on ES, expiration week, month turn, and a 540-cell clock map — all
  NO-GO. This is better use of the one edge that works.
- **The entry change has a weaker dev half.** At 16:00 the FOMC leg is dev
  t=1.99 / val t=2.50; at 14:00 it is dev t=2.42 / val t=2.54. The 16:00 version
  has better MAE and a stronger val half but is less balanced. The size increase
  (2→4) is the robust part of this recommendation; the entry change is the
  smaller, more arguable half of it.
- **n=47 FOMC events.** Unchanged from the ledger, and the reason the weekend leg
  is kept at 1 rather than dropped to 0.
- MC bootstraps iid from post-2020 MNQ; the ×1.00 column is optimistic and the
  decay columns are the honest range.

## Operational change from GO v1

1. FOMC runner: 2 → **3** micros (NOT 4 — see the correction at the top).
2. FOMC entry: prior session **16:00 ET** (was 14:00), exit unchanged at 13:55 ET.
3. Weekend runner stays at **1** micro — already the ladder's starting rung.
4. **Fix the stale risk constant.** `ladder_size()` documents "EO/FOMC: worst MAE
   ~ -$416/micro". Measured on n=47 at the prior-close entry it is **−$822/micro**,
   almost exactly 2x worse. Every rung is therefore twice as risky as designed.
   That constant should be corrected before any size increase is deployed.
