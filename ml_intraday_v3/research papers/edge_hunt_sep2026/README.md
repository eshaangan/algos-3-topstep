# Edge-hunt paper pull — 2026-09-05

Selected against the ledger's conclusion that **price/volume families on liquid micros are
exhausted**, and that the only survivor so far (`fomc_drift_v1`) lives on an *event clock*,
invisible to price batteries. So: papers whose mechanism is a **calendar / flow / hedging
obligation**, not a price pattern. Each entry says what is testable with data already local.

## Downloaded

### 1. Golez & Jackwerth (2012) — *Pinning in the S&P 500 Futures* (JFE 106)
`golez_jackwerth_2012_pinning_sp500_futures.pdf` — **highest-priority new family.**
Mechanism: on option-expiration days, delta-hedge rebalancing + stock-vs-futures arb drags ES
futures toward the nearest option strike in the final hours, and *away* from it on non-expiration
days. It is a dealer obligation, not a behavioural pattern → survives publication.
Testable now: MNQ/MES 1-min 2020–2026 (`data/hist_1m24/`), monthly 3rd-Friday expiries, n≈80/instr.
Direction is *toward a known strike grid* (ES 25pt, NQ 100pt), so no options data required for a
first pass. Event-clock, ~12–24 events/yr — same velocity class as FOMC, and combinable with it.

### 2. Hu, Pan, Wang & Zhu — *Premium for Heightened Uncertainty* (NBER w25817, JFE)
`hu_pan_wang_zhu_premium_heightened_uncertainty.pdf`
The general theory behind the one thing that works here. The pre-announcement premium is
compensation for *heightened uncertainty*, and it is earned in the ~24h window before scheduled
resolution. Explains cleanly why FOMC pays and why CPI paid only 2021–24 (it was the dominant
uncertainty then, and stopped being one). Gives a **principled screen for which events should
carry drift** instead of basketing everything — i.e. which extensions to `fomc_drift_v1` are worth
a trial slot (FOMC minutes, Chair testimony, Jackson Hole, ECB on M6E) and which are not.
Note: prior session's HPWZ vol-conditioning of CPI already inverted on pre-spec — read §s on the
VIX-conditioning result before re-trying that.

### 3. Lucca & Moench — *The Pre-FOMC Announcement Drift* (NY Fed SR512 / JF 2015)
`lucca_moench_pre_fomc_drift.pdf`
Source paper for the existing survivor. Worth reading now for the parts you have not used:
exact window decomposition (how much of the +$388/event is in which hours), the international
placebo panel, and the pre-1994 / post-1994 split — the latter is the closest thing to an answer
on whether the drift decays after publication.

### 4. Takahashi (2025) — *Returns and Order Flow Imbalances: Intraday Dynamics and
Macroeconomic News Effects* `takahashi_2025_ofi_intraday_macro_news.pdf`
Relevant because the L2/OFI kill (`project_l2_bbo_edge_kill_jul2026`) tested OFI *unconditionally*
on 24 days. This paper's claim is that the OFI→return relation is state-dependent around news.
That is a differentiated hypothesis, so it clears the "no re-running dead families" rule — but
only as **OFI conditioned on event windows**, and only if the recorder has enough event days.

### 5. Li, Sakkas & Urquhart (2021) — *Intraday Time Series Momentum: International Evidence*
`li_sakkas_urquhart_2021_intraday_tsmom_global.pdf` — **low priority, kill-documentation.**
The Gao family, which already failed here on exact spec (t=−1.8). Useful for the cross-market
map of where it survives and the market characteristics it loads on; do not spend trial slots
re-running it without one of those characteristics as a pre-registered conditioner.

## Obtained 2026-09-05 (second pass, via paper-search-mcp + author pages) — and KILLED

### 6. Cieslak, Morse & Vissing-Jørgensen — *Stock Returns over the FOMC Cycle* (JF 2019)
`cieslak_morse_vj_stock_returns_fomc_cycle.pdf` — the Feb 2018 working paper, author-hosted
on Adair Morse's Berkeley faculty page. The published version is closed access with no OA
location (OpenAlex: `oa_status: closed`).

### 7. Uppal (2025) — *Does the FOMC Cycle Still Drive Stock Returns?*
`uppal_does_fomc_cycle_still_drive_returns.pdf` — a direct out-of-sample refutation of #6.

**Verdict: NO-GO.** See `rule_based_v1/validation/vps_prereg/vps_fomc_cycle_weeks_VERDICT.md`.
CMVJ's week 0 spans days −1..+3 and so *contains* the pre-FOMC drift already traded as
`fomc_drift_v1` — they say so themselves — leaving only weeks 2/4/6 as new velocity. Uppal
shows the effect dies from 2004 and its mechanism (biweekly Board meetings) ended then;
verified on our own tape: ES even-vs-odd t=−0.31 over 2010–2025, dev t=+2.48 → val t=−1.07,
MNQ nothing. There is no biweekly extension to buy.

## Still wanted but paywalled — worth sourcing
- **Lou, Polk & Skouras, "A Tug of War: Overnight versus Intraday Expected Returns" (JFE 2019).**
  Formalises the decomposition both current survivors (`weekend_hold_v1`, `monday_rth_v1`) live in.
- **Etula, Rinne, Suominen & Vaittinen, "Dash for Cash" (RFS 2020).** Month-end. ToM was tested
  dead here, but their conditioning is narrower (Treasury coupon settlement dates, not calendar
  month-end) — a legitimately different spec.
- **Baltussen, van Bekkum & Da, "Indexing and Stock Market Serial Dependence Around the World"
  (JFE 2019).** Not a strategy — an *explanation* of why the price families are all dead:
  index-product growth flipped intraday serial dependence from momentum to reversal.
- **Anderegg, Ulmann & Sornette (2022), "The impact of option hedging on spot market volatility"
  (JIMF).** Dealer-gamma → realised vol; the vol-regime companion to Golez & Jackwerth.
- **Heston, Korajczyk & Sadka (2010), "Intraday Patterns in the Cross-section" (JF).**
  Half-hour periodicity; cross-sectional, so weakest fit to a single-instrument futures book.

## Status of the suggested order (both top items now resolved)

1. ~~Golez & Jackwerth expiration-day pinning~~ — **NO-GO**, see
   `vps_expiration_pinning_VERDICT.md`. Strike grid never widened with the index, so the
   theoretical ceiling ($25 gross) sits too close to cost ($7.48); best realised gross +$4.56.
2. ~~Source Cieslak et al.~~ — **obtained and NO-GO**, see `vps_fomc_cycle_weeks_VERDICT.md`.
3. **HPWZ screen → pick at most two new event types, pre-register, single shot each.**
   Now the live next step. `hu_pan_wang_zhu_premium_heightened_uncertainty.pdf` gives the
   criterion (drift attaches to *scheduled resolution of dominant uncertainty*), which is the
   principled way to choose extensions rather than basketing every event. Candidates it
   implies: FOMC minutes releases, Chair congressional testimony, Jackson Hole, ECB decisions
   on M6E. The verified calendar infrastructure for this now exists
   (`rule_based_v1/validation/fetch_fomc_calendar.py`).
4. Takahashi — OFI conditioned on event windows, differentiated from the unconditional
   24-day L2 kill.
