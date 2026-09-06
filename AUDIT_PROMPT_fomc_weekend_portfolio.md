# Audit prompt — paste into a fresh chat

You are auditing a proposed live futures trading configuration before real money is
risked on it. Your job is to try to KILL it. Do not be agreeable. End with an
explicit GO or NO-GO.

## The proposal

Trade two windows on MNQ (Micro Nasdaq) in a $50k evaluation account with a
$3,000 profit target and a $3,000 EOD-trailing max loss:

1. `weekend_hold_v1` — long **1 micro**, Sunday 18:00 ET → Monday 15:59 ET.
2. `fomc_drift_v1` — long **3 micros**, from **16:00 ET on the session before a
   scheduled FOMC decision** → 13:55 ET on the decision day (flat before the 14:00
   announcement).

Claimed: **95.4% probability of passing, 14-week median.**

## Claims to verify — re-derive every number, trust none of them

| # | claim | where it came from |
|---|---|---|
| 1 | fomc_drift @14:00 entry: MNQ n=47, +$354/event (2 micros), t=+3.53, dev t=+2.42 / val t=+2.54 | `rule_based_v1/validation/research_fed_events.py` |
| 2 | fomc_drift @16:00 entry: +$334, MAE p05 −$893 vs −$1,240 at 14:00, eff 0.374 vs 0.286, but dev t=+1.99 / val t=+2.50 | same |
| 3 | weekend_hold on MNQ post-2020: n≈202, +$264/wk at 2 micros, t=+4.50 | `research_vol_target_weekend.py`, `replay_hold_strategies.py` |
| 4 | weekend_hold on ES 2010–2025: n=710, t=+1.61 full, **pre-2020 −$8.80/wk t=−1.09**, 2020+ +$65.28 t=+2.27 | `vps_prereg/weekend_hold_15y_ES_CHECK.md` |
| 5 | Portfolio MC: 1wk+3fomc = 95.4%/14wk; 1wk+2fomc = 96.8%/15wk; 1wk+4fomc = 92.3%/12wk | `research_portfolio_mc.py` |
| 6 | Worst single FOMC event MAE = −$822/micro; at 4 micros that is −$3,286 and exceeds the cushion | ad-hoc, re-derive it |

## Where everything is

- Research + MC: `rule_based_v1/validation/research_{fed_events,portfolio_mc,vol_target_weekend}.py`
- Canonical entry/exit used by the live runners: `rule_based_v1/validation/replay_hold_strategies.py`
- Verdicts and pre-registrations: `rule_based_v1/validation/vps_prereg/`
  (start with `PORTFOLIO_GO_v2_1wk_4fomc.md`, `PORTFOLIO_GO_1wk_2fomc.md`,
  `weekend_hold_15y_ES_CHECK.md`)
- Data: `data/processed/mnq_1m_all.parquet` (1-min, naive **America/Chicago**, 2020→2026),
  `data/processed/es_1min_eth_frontmonth.parquet` (1-min ETH, tz-aware ET, 2010→2025),
  `data/processed/fomc_announcements.csv` (127 Fed-verified announcement dates)
- Live code: `rule_based_v1/live/ladder_live_runner.py`, `fomc_paper_runner.py`
- Project history, including ~80 prior NO-GO configs:
  `~/.claude/projects/-Users-eshaanganguly-Documents-projects-algos-3-topstep/memory/project_edge_hunt_ledger_jul2026.md`

## The prior analyst made these mistakes — calibrate accordingly

Assume more remain. Found and corrected during the same session:

- Recommended FOMC×4 without checking single-event risk. The worst historical event
  at that size exceeds the whole cushion — a one-trade account kill. P(pass) hid it.
- Ranked a scan by the LONG net t-stat, which made pure cost drag look like short
  edges. Short net is `−gross − cost`, not `−(gross − cost)`.
- Recommended cutting the weekend leg 2→1 when the deployed runner already starts at 1.
- Dropped ~35% of events in a first-pass event study by taking "previous calendar day"
  instead of "previous session", which silently discarded Mondays.

## Attack these specifically

1. **Multiple testing.** The 16:00 entry was chosen after profiling 9 entry hours, and
   the session ran many families (see the ledger). Is the FOMC result still significant
   after honest trial accounting? Compute a deflated Sharpe or equivalent. Does n=47
   support a 3-micro bet at all?
2. **Regime.** The weekend leg is negative pre-2020 on ES. The FOMC leg exists only from
   2020 in MNQ data — check it on the 15.5-year ES tape, which the prior analyst did NOT
   do for FOMC. If FOMC is also a post-2020 artifact, the whole thing collapses.
3. **The MC.** Read `portfolio_mc()` line by line. It bootstraps iid weekly draws. Does
   it model the EOD-trailing floor correctly? Does it handle intra-trade MAE, the profit
   target, and the order of bust-vs-pass checks correctly? Is iid defensible given events
   cluster and regimes persist? Rebuild it independently and see if you get 95.4%.
4. **Execution realism.** Can these fills actually be obtained — Sunday 18:00 ET reopen
   spread and liquidity, and 16:00 ET on the prior session? Is 1 tick of slippage per side
   right for MNQ at those times? What happens on a gap through the stop?
5. **Account rules.** Verify the actual target, trailing-drawdown mechanics, overnight and
   weekend margin for 3–4 MNQ micros, and whether any consistency rule or daily-loss limit
   applies. The project believes LucidFlex has no consistency rule — confirm independently.
6. **Deployed code.** `ladder_size()` in `ladder_live_runner.py` documents worst FOMC MAE
   as −$416/micro; measurement suggests −$822. Verify which is right. Check the hardcoded
   FOMC calendar (it appears to end 2026-12-08) and the hardcoded 14:00 entry hour.

## Decision criteria

GO only if **all** hold:
- FOMC leg is significant after honest multiple-testing correction, and in both era halves.
- P(pass) ≥ 90% under a MC you rebuilt and believe, including a scenario where the
  weekend leg contributes zero.
- No single event can breach the cushion at the proposed size.
- Fills are realistic and account rules are confirmed.
- The live code can be made to match the tested spec exactly.

Otherwise NO-GO, or GO with specific stated changes.

## Output

1. What you verified and what you could not.
2. Errors found in the prior work.
3. The numbers you independently derived, next to the claimed ones.
4. The strongest argument against trading this.
5. **GO or NO-GO**, with size and entry if GO, and the single most important
   reason for the verdict.
