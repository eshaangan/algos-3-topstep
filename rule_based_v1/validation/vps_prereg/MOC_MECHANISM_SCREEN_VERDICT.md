# MOC closing-auction family — DO NOT BUY THE DATA. Mechanism falsified on data we own.

Script `research_auction_reversal.py`, run `runs/auction_reversal.json`.
A pre-purchase screen, run before spending anything on imbalance data.

## The falsifiable prediction

Bogousslavsky & Muravyev's closing-auction story is that forced auction flow pushes price
into the 16:00 cash close and that the pressure is **temporary** — it unwinds once the
auction clears. Index futures trade past 16:00, so if the mechanism operates in ES/MNQ it
must leave a fingerprint: the 15:50→16:00 move should partially **reverse** in 16:00→16:15
(negative push/unwind correlation), and more strongly than at ordinary times of day.

## Result: the auction window is the ONLY window that does not reverse

| instrument | window | n | corr | t | dev | val |
|---|---|---|---|---|---|---|
| ES | **AUCTION 15:50→16:00 / 16:00→16:15** | 3,870 | **+0.064** | **+4.00** | +0.166 | +0.054 |
| ES | placebo 14:50→15:00 | 3,870 | +0.014 | +0.85 | −0.034 | +0.019 |
| ES | placebo 13:50→14:00 | 3,870 | **−0.066** | −4.11 | +0.007 | −0.074 |
| ES | placebo 11:50→12:00 | 3,977 | **−0.050** | −3.18 | +0.015 | −0.056 |
| MNQ | **AUCTION** | 1,624 | +0.006 | +0.22 | −0.086 | +0.093 |
| MNQ | placebo 13:50→14:00 | 1,624 | **−0.113** | −4.58 | −0.070 | −0.149 |
| MNQ | placebo 11:50→12:00 | 1,680 | **−0.077** | −3.17 | −0.125 | −0.038 |

**Excess over placebo: ES +0.098, MNQ +0.078 — both the WRONG SIGN.**

Ordinary intraday windows show exactly the mild mean-reversion you would expect at this
horizon (−0.05 to −0.11, significant at t≈−3 to −4.6). The auction window is the one place
where that reversion **disappears** — ES actually shows significant *continuation*
(t=+4.00). This is the opposite of the temporary-impact fingerprint.

## Why this is economically sensible

The closing auction's forced flow lands on individual **stocks**. Index futures never
experience that flow directly — they track the cash index through arbitrage, and the basis
stays tight. There is no auction pressure in ES/MNQ to unwind, so no reversal appears. The
temporary-impact literature is about the auction's own participants, not about a futures
contract observing the print.

## Decision

**Do not buy Nasdaq/NYSE imbalance data.** The mechanism's central prediction fails in the
only instruments this project can trade, on 3,870 ES days and 1,624 MNQ days. An imbalance
feed measures the input to a mechanism whose output is provably absent here — better data
about a signal that does not propagate cannot create an edge.

This screen cost nothing and avoided a purchase whose price could not even be quoted
(Databento account locked; NYSE TAQ priced only via the ICE dashboard).

**Method note worth keeping:** when a family requires data you do not have, check whether
the mechanism makes a prediction testable on data you DO have. Here the mechanism was
falsifiable for free, and the answer arrived before any money moved.
