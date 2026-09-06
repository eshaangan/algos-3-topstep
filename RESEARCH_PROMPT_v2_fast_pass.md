# Fresh-chat research prompt — find a fast, legal, real edge

Set this as your goal and do not stop until you have either a GO that meets the bar
below, or a defensible statement that the bar cannot be met and why.

## The objective, stated as arithmetic

LucidFlex **100k** evaluation: **$6,000 profit target**, **$3,000 EOD-trailing MLL**
(locks at start balance), **50% consistency** (largest winning day ≤ 50% of total
profit), no daily loss limit, 2 minimum trading days, no time limit.

Target: **pass in ≤ 16 weeks with P(pass) ≥ 85%.** That needs roughly **$375–500 per
week** at a size whose worst historical single-trade excursion stays under about
**60% of $3,000**. Every candidate must be judged against that arithmetic before
anything else. If it cannot plausibly clear $375/week within the drawdown, do not
spend a slot on it.

The current best legal configuration is **weekend_hold_v1 at 1 micro: 92.4% but 42
weeks**. That is the incumbent. Beat it on time without dropping below 85%.

## The constraint that shapes everything

**Mandatory 16:45 ET auto-flatten, Monday–Friday.** Trading resumes 18:00 ET
Sun–Thu. Consequences, already verified:

- Any hold that crosses 16:45 ET on a weekday is **force-closed** — this kills
  `fomc_drift_v1` (the project's only sealed-holdout survivor, t=+3.53) outright.
- The only legal multi-hour holds are **intraday within one session**, or
  **Sunday 18:00 → Monday 16:45**.
- So: no overnight event drift, no 24h windows, no multi-day.

Rules live in `rule_based_v1/configs/account_rules.yaml`. Use
`rule_based_v1/validation/tzguard.py::check_window_legal()` to test any window's
legality **before** backtesting it. Do not hardcode account numbers anywhere.

## Run TWO tracks. Track B may be worth more than Track A.

### Track A — find an intraday edge that clears the arithmetic

### Track B — find an account whose rules permit the edges already validated
`fomc_drift_v1` (+$354/event, t=+3.53, dev t=+2.42 / val t=+2.54, n=47) and
`weekend_hold_v1` (+$263/wk at 2 micros, t=+4.47, n=202) are statistically real and
are killed *only* by the flatten rule. Survey futures prop firms and evaluation
products for ones that permit overnight and news holds — Topstep, Apex, TakeProfit,
Earn2Trade, MyFundedFutures, Bulenox and others differ materially on overnight
holding, news restrictions, trailing-drawdown mechanics and consistency rules.
Quantify P(pass) and time-to-pass for the existing book under each firm's actual
published rules. **If some firm's rules make the existing validated book pass in 12
weeks at 90%, that is the answer, and it requires no new alpha at all.** Check news
restrictions carefully — several firms ban holding through high-impact releases,
which would kill FOMC drift a different way.

## What is already dead — do not re-run these

Roughly 85 configurations. Full detail in
`~/.claude/projects/-Users-eshaanganguly-Documents-projects-algos-3-topstep/memory/project_edge_hunt_ledger_jul2026.md`
— **read it first.** Summary of the graveyard:

- All price/volume families on MNQ/MES/MGC/MCL/M6E/M2K, both directions, every
  cost-viable horizon. ORB formally killed (−$23k/6.5y).
- A 540-cell 24h×weekday clock map on 15.5y of ES: **zero** cells clear cost with
  era agreement. Best t=2.15 against a 3.91 Bonferroni bar.
- Event pre-announcement drift: CPI, NFP, ISM, GDP, FOMC minutes, mega-cap earnings
  — all fail against the same-clock-window control.
- Expiration-day pinning, expiration week, month turn, CMVJ even-week FOMC cycle,
  intraday time-series momentum (Gao spec), L2/OFI top-of-book, euro-open on ES.
- Volatility-targeted sizing: no benefit over flat at matched average size.

## The two lessons that should steer the search

**1. Rank by return ÷ MAE, not by t-stat.** The binding constraint is the drawdown,
not the drift. Plain long ES for a week pays t=+2.43 and is unusable. Efficiency
(mean ÷ |MAE p05|): fomc_drift 0.29–0.37, weekend_hold 0.18, everything else in the
project 0.02–0.05, beta 0.04. **You need ≥0.2 with high velocity.**

**2. Look for QUIET drift, not big drift.** HPWZ's pre-announcement premium works
because it comes with *no abnormal variance*. Mega-cap earnings failed precisely
here: real drift, but MAE p05 −$1,960 versus FOMC's −$832 *worst*. High-variance
windows cannot be sized inside a $3,000 MLL.

## Research directions not yet mined

Intraday-only, so the pre-announcement family is closed. Untapped:

- **Closing auction / MOC imbalance.** Bogousslavsky & Muravyev on closing-auction
  price pressure; index rebalance flows. Intraday by construction, high velocity
  (daily), and mechanically driven by forced flows.
- **Intraday periodicity.** Heston, Korajczyk & Sadka (JF 2010) half-hour
  periodicity; Bogousslavsky (JF 2016) "Infrequent Rebalancing, Return
  Autocorrelation, and Seasonality." Never obtained or tested here.
- **0DTE and dealer gamma intraday.** The 0DTE literature post-2022 is large and
  the mechanism (intraday hedging flow) is a forced flow, not a behavioural pattern.
- **Post-release intraday continuation** at 08:30/10:00 ET releases — a different
  family from the pre-announcement drift that was killed, and there are ~250
  scheduled releases a year.
- **Instruments not yet tested** where cost ÷ volatility may be better than MNQ:
  MBT (micro bitcoin, 24/7 — check flatten applicability), MYM, micro treasuries.
  MNQ round-turn is 2.24 pts per contract; compute this ratio before testing.

Use the `paper-search` MCP server (23 sources) and `scientific-papers`. Author-hosted
faculty-page PDFs beat aggregators for paywalled finance papers. Downloaded papers
live in `ml_intraday_v3/research papers/edge_hunt_sep2026/` with a README.

## Method — non-negotiable, these caught every false positive

1. **Pre-register** in `rule_based_v1/validation/vps_prereg/*.yaml` before computing a
   single return: spec, universe, era split, trial count, and the kill condition.
2. **Always run the same-clock-window control on non-event days.** This alone killed
   ISM (which replicated the paper's basis points almost exactly) and mega-cap
   earnings (where the control *beat* the event). A positive number is meaningless
   without it — being long index futures pays on average.
3. **Era split, and require both halves.** Many things work post-2020 and nowhere
   else. Note that `weekend_hold_v1` is **negative pre-2020 on 15.5y of ES**.
4. **Count every trial.** Report the Bonferroni or DSR bar next to the result.
5. **Cross-instrument confirmation** before promoting anything.
6. **Check single-event risk, not just P(pass).** A configuration whose worst
   historical trade exceeds the MLL is a one-trade account kill even at 92% P(pass).

## Known traps in this repo

- **Timestamps.** Both MNQ files are true DST-aware `America/Chicago`. Applying a
  fixed −05:00 offset shifts every winter mark an hour and inflated one strategy's
  n from 202 to 296. `tzguard.assert_et_index()` and the guard inside
  `replay_hold_strategies.load_bars()` now abort on this — do not bypass them.
- **Long vs short scoring.** Short net is `−gross − cost`, not `−(gross − cost)`.
  Ranking a scan by the long's t-stat makes pure cost drag look like short edges.
- **"Previous day"** must mean previous *session*, not previous calendar day, or you
  silently drop every Monday event.
- **Reuse before writing.** `diagnostics/portfolio_mc.py` already had correct account
  rules while a new MC was written with wrong ones. Search the repo first.

## Data

- `data/processed/mnq_1m_all.parquet` — MNQ 1-min ETH, 2020→2026, America/Chicago
- `data/processed/es_1min_eth_frontmonth.parquet` — ES 1-min ETH front-month, 2010→2025
- `data/processed/fomc_announcements.csv` — 127 Fed-verified announcement dates
- `data/processed/megacap_earnings.csv`, `bea_gdp_advance_releases.csv`
- `data/hist_1m24{,_mes,_mgc,_mcl,_m6e,_zn}/`, `data/hist_1sv/` (MNQ 1-second, 285 days)

## Deliverable

A GO requires: legal under `account_rules.yaml`, P(pass) ≥ 85% in ≤16 weeks under a
path-aware MC using the real target and MLL, no single event able to breach the MLL,
survives its control and both era halves, and cross-instrument confirmation.

Otherwise: say so plainly, and report whether Track B found an account under which
the already-validated book clears the bar. That outcome is a win, not a failure.
