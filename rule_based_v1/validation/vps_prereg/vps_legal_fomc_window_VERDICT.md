# vps_legal_fomc_window — NO-GO as a promotion, but it CORRECTS the ledger

Run `runs/legal_fomc.json`, script `research_legal_fomc_window.py`.

## The correction: fomc_drift_v1 is not entirely illegal

The ledger records a blanket **"fomc_drift_v1 = ILLEGAL"**. That verdict was reached by
testing the two entry hours the strategy had historically used — 14:00 and 16:00 ET on
the prior session — both of which sit before the 16:45 ET flatten and are force-closed.

It was never tested at an entry **after** the flatten. Entries at 18:00 and 20:00 ET →
13:55 next day are **LEGAL under LucidFlex's own rules**, machine-checked. Roughly half
the premium accrues after the Globex reopen and that half is tradeable today.

This matters operationally: adding the legal 18:00 FOMC leg to the weekend book takes
LucidFlex from the incumbent **92.4% / 42 weeks to 92.4% / 34 weeks** — same pass
probability, eight weeks faster, with no new alpha and no rule change.

## A real bug this exposed in tzguard

The first run reported **17:00 ET as the best legal entry** — MNQ n=47, +$344.75,
minus-control **t=+2.97 (clears the 2.90 Bonferroni bar)**, both eras positive
(dev t=+2.16, val t=+2.74), efficiency **0.407**, the highest in the project.

It is not tradeable. `check_window_legal()` tested only whether a hold crossed the
flatten; it did not test whether the entry could be **placed**. Lucid flattens at 16:45
and accepts no orders until 18:00, so 17:00 is inside a closed window.

Worse, the 17:00 cell's entire advantage over 18:00 ($345 vs $220) accrues across the
**17:00–18:00 CME maintenance halt** — the untradeable reopen gap that already produced
one false positive in this project (the fake "Asia drift", t going +2 → −2 at executable
marks). Two independent reasons the number is fictional.

**Fixed:** `check_window_legal()` now also enforces `trading_resumes_et`, and
`account_rules.yaml` gained a `resumes_et` field. 17:00 now correctly returns ILLEGAL.

## The pre-registered verdict on the 18:00 primary

Judged against its own registered kill conditions, the legal window **does not promote**:

| check | requirement | MNQ 18:00 | pass |
|---|---|---|---|
| minus-control t | ≥ 2.5 | **+2.29** (p=0.026) | ✗ |
| Bonferroni (8 cells) | ≥ 2.90 | +2.29 | ✗ |
| both eras positive, MNQ | yes | dev +1.76 / val +1.48 — positive, neither significant | ~ |
| both eras positive, ES | yes | **dev −$1.87 (t=−0.17)**, val +$86.89 (t=+2.38) | ✗ |
| efficiency | ≥ 0.20 | 0.260 | ✓ |
| accrual profile smooth | yes | $345 → $220 → $187 → $172, monotone | ✓ |

The **ES dev half is negative**, which is a registered kill. The shape check passes
cleanly — the profile decays smoothly with later entry rather than spiking at one mined
hour — so this is a real effect that is simply too weak at the legal entry to promote on
its own.

It also independently confirms the audit's regime warning: **ES pre-2020 FOMC is dead**
(dev t=−0.17) and only the post-2020 half pays. Both of this project's surviving legs
are post-2020 phenomena on the long tape.

## Disposition

Not promoted as a standalone strategy. **Retained as a portfolio leg at reduced size**,
where its contribution is measured by the MC rather than by its own t-stat, and where the
correction to legality is worth 8 weeks. Its honest description is "half of a real edge,
significant at p=0.03, on a post-2020 sample."
