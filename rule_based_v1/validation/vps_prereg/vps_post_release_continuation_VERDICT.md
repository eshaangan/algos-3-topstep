# vps_post_release_continuation — NO-GO

Run `runs/post_release.json`, script `research_post_release_continuation.py`.
All 12 pre-registered cells computed and counted. Bonferroni bar t = 2.87.

## Result: dead in every cell, on both instruments, in both eras

Event-minus-control t ranged from **−0.61 to +0.64**. Nothing came within a third of
the kill threshold (t ≥ 2.0), let alone the promotion bar.

| instrument | family | horizon | event $ | control $ | diff t | eff |
|---|---|---|---|---|---|---|
| MNQ | FOMC 14:00 | R+90 | −$36 (vs ctl) | — | −0.24 | −0.09 dev / +0.12 val |
| MNQ | GDP adv 08:30 | R+90 | −$34 | −$6 | −0.35 | −0.05 |
| MNQ | claims 08:30 Thu | R+90 | −$8 | −$15 | +0.29 | −0.011 |
| MNQ | claims 08:30 Thu | R+150 | −$8 | −$23 | +0.44 | −0.009 |
| ES | FOMC 14:00 | R+90 | −$0.66 | −$7 | +0.30 | −0.001 |
| ES | GDP adv 08:30 | R+90 | −$11 | −$6 | −0.27 | −0.046 |
| ES | claims 08:30 Thu | R+90 | −$8 | −$5 | −0.49 | −0.034 |

Every efficiency is negative or indistinguishable from zero against the ≥0.20 the
arithmetic requires. The raw event means are mostly negative before any comparison.

## Why it is worth having run

Jobless claims gave n=336 (MNQ) and **n=753 (ES)** — by far the largest event sample in
this project, ~52/yr. If post-release continuation existed at a tradeable size anywhere,
this is where it would have shown up, and the sample is large enough that the absence is
informative rather than merely underpowered.

## The diagnostic finding

The **control** is significantly negative on ES: −$7.11 at t=−4.27 (R+90) and −$5.84 at
t=−3.06. That is conditional intraday momentum losing money to cost, at high
significance, over 3,300 non-event days. It independently reproduces the ledger's
meta-finding that at retail cost structure short-horizon barrier trading loses in both
directions on every instrument tested.

The events are not different from that. The release does not buy a continuation; it just
buys the same losing intraday momentum trade on a noisier day.

## Note on control sign

Because the control is negative, several cells show a *positive* event-minus-control
difference (ES claims R+150, +$4.74) while the event itself still **loses money**
(−$5.50). Beating a losing control is not an edge. Both must clear cost, and neither does.

**Verdict: NO-GO. The highest-velocity family available is dead. Do not re-run.**
