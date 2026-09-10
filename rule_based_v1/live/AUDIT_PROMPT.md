# Audit prompt — live two-leg book (weekend_hold_v1 + fomc_drift_v1)

Paste the section below into a fresh session. It is written adversarially on
purpose: it names the specific places this is most likely to be wrong, so the
auditor spends its effort verifying rather than rediscovering.

---

## THE PROMPT

You are auditing a live futures trading deployment before it risks money on a
prop-firm evaluation. Assume it is broken until you prove otherwise. Your job is
to find what will lose money or breach the account, not to confirm the design.

Work in `/Users/eshaanganguly/Documents/projects/algos 3 topstep`. Read
`.claude/CLAUDE.md` first. **Do not modify the VPS at `172.93.190.170` — it is
not ours, read-only.**

### What is deployed

- `rule_based_v1/live/book_live_runner.py` — one process, two legs, live orders
- `rule_based_v1/live/launch_book.sh` — supervisor (recorder + runner)
- `rule_based_v1/tests/test_book_live_runner.py` — 19 tests, all passing
- Config claimed frozen: **$600/micro stop, weekend ×2 micros, FOMC ×2 micros**
- Claimed: **$288.75/week, worst trade −$1,205, P(pass) 73.4% over 32 weeks**
- Source of that claim: `runs/disaster_stop_lucid_32.json`, best cell by `p_pass`
- Firm: LucidFlex 100k. Rules verified against `support.lucidtrading.com` on
  2026-09-09 and recorded in `rule_based_v1/configs/prop_firm_rules.yaml`

### Part 1 — Deployment and execution safety

Answer each with evidence from the code or a run, not from the docstrings.

1. **Account identity.** The connected account is `LFE100-Z96Y99ZR-TEST001`. The
   suffix says TEST. Determine whether orders would land on a real evaluation or
   a demo. If it is a demo, every claim about "deployed live" is false.
2. **The quote source is currently broken.** `record_l2.py` hit 18 Rithmic
   `ForcedLogout` messages in ~90 seconds ("maximum number of concurrent
   sessions"). Find what else holds a ticker session on these credentials.
   Confirm whether ticker and order plants on one login is permitted at all.
   **Without a fresh quote the runner refuses to enter — so today this book
   trades nothing. Verify that refusal actually works** (`quote()` returns None
   past `QUOTE_MAX_AGE_S`) and that it fails closed, not open.
3. **The stop is the whole risk model and it has never fired.** `protect()` then
   `verify()` submit a STOP_MARKET and confirm it is resting by matching
   `user_tag`. Verify `user_tag` is actually the field Rithmic echoes for
   `order_id` in this SDK version (async_rithmic 1.6.1). If it is not, `verify()`
   silently flattens every trade — or worse, matches nothing and the position
   rides unprotected.
4. **Runner death while holding.** If the process dies mid-trade the stop rests
   at the broker, but nobody performs the scheduled 15:59 / 13:55 exit. Trace
   what happens: does the supervisor restart it, and does
   `startup_recovery` then flatten (good) or adopt (bad)? Confirm Lucid's 16:45
   auto-flatten is the true backstop, and state the worst-case loss if both the
   runner and the supervisor are dead.
5. **Contract roll.** `front_month()` rolls MNQU6 → MNQZ6 about 8 days before the
   third Friday, i.e. on 2026-09-10, which is the day before the first live
   signals. Confirm the entry-time re-resolution picks the right contract and
   that a stop placed on one contract cannot be orphaned by a roll.
6. **Idempotency and state.** `state.json` `done` grows without bound. Check
   restart mid-trade, duplicate fills, a fill notification arriving after
   `flatten()`, and whether `_note()` can book a SELL from an unrelated manual
   trade as this runner's exit.
7. **Account-wide operations.** `_cancel_all()` and `exit_position()` affect the
   whole account. Confirm nothing else trades this account, and that cancelling
   all orders cannot remove a stop belonging to a position this runner still
   holds.
8. **Timezones and DST.** Every schedule is US/Eastern. Check the Sunday 18:00
   entry and the Monday 15:59 exit across a DST boundary, and confirm
   `pd.Timestamp.now(tz=ET)` is used consistently.
9. Run the tests. Then look for the behaviour the tests do NOT cover, and say so
   plainly.

### Part 2 — The model and the 73.4% claim

10. **Reproduce the number.** Re-run
    `research_disaster_stop.py --firm lucid_flex_100k --max-weeks 32` and confirm
    the best cell is stop $600 / weekend 2 / fomc 2 at p_pass 0.734. Then read
    `mc()` in `research_firm_choice_mc.py` and list every assumption. It
    bootstraps iid from post-2020 trades — state what that does to the estimate.
11. **Attack the weekend leg.** `project_edge_screens_sep2026` records: full
    t=4.50 but **2021–2024 (61% of the sample) averages $24.75/wk at t=0.84**;
    2026 is 8.4% of weeks and 34% of P/L; top 20 of 202 weeks are 70% of P/L,
    gini 0.45. Decide whether a 2-micro live allocation is defensible on that
    evidence, and say what you would need to see instead.
12. **Attack the FOMC leg.** n=47 lifetime, per-trade Sharpe 0.515, only 8 events
    a year. Confirm the deployed 18:00 entry matches the *legal partial* window
    that was validated (minus-control t=+2.29), not the full 14:00 window that
    is illegal under the 16:45 flatten and which scores better.
13. **Tail correlation.** Both legs are long equity overnight. The MC treats
    their risks as separable. Quantify what a single overnight gap does when it
    lands on either leg, and whether "worst trade −$1,205" survives a 2020-style
    or 2018-style gap rather than the worst gap in the sampled window.
14. **⚠️ CONSISTENCY RULE — check this first, it may be the binding failure.**
    LucidFlex evaluation enforces **50% consistency: the largest day must not
    exceed 50% of total profit**. This book is extremely concentrated (top 20 of
    202 weekends = 70% of P/L). Simulate whether a passing equity curve would
    *breach consistency* on the way to the $6,000 target. If it would, the 73.4%
    is measuring the wrong event and the real pass probability is lower.
    Confirm whether `mc()` models the consistency rule at all.
15. **Rules provenance.** Everything rests on Lucid's help centre, not on the
    account dashboard. `account_rules.yaml::verified_against_account_dashboard`
    is still `false`. List every deployed number that would change if the
    dashboard disagrees.

### Deliverable

A single ordered list of findings, each with: severity, the evidence, and the
one-line fix. Put anything that can breach the account or trade unprotected at
the top. If your conclusion is "do not run this yet", say so in the first line.

State clearly which of the 15 items you could NOT verify, and why.
