# VERDICT — depth-scaled order-flow imbalance on MNQ L3

**Status: KILLED (2026-09-09), by the binding clause in `PREREG_depth_scaled_ofi.md` §3.**

Self-contained. This verdict was reached on this hypothesis alone.

---

## Result

DEV = 9 MNQ L3 session dates, 2026-08-19 → 2026-08-31, 208,440 RTH one-second
bars. 360 cells run, all reported, none hidden.

**H1 (CKS depth scaling describes MNQ) — FALSIFIED, not merely unsupported.**

| horizon | IC `ofi` | IC `ofi_scaled` |
|---|---|---|
| 1s | −0.0066 | −0.0063 |
| 5s | +0.0041 | +0.0045 |
| 30s | +0.0019 | +0.0021 |
| 300s | −0.0020 | −0.0021 |

Scaled tracks raw to the third decimal at every horizon. In the thinnest 10% of
book states, where the CKS coefficient predicts the largest divergence, the two
are still identical (1s: 0.0008 vs 0.0005). Head to head on the 72 matched
(depth_tail, signal_q, horizon) triples, **depth scaling is better in 27 of 72
(38%) and costs $0.12/contract on average**. Dividing by depth makes the signal
slightly worse. Every IC in the table is inside ±0.007 over nine days.

**H2 (a low-depth cell clears the cost floor) — FAILS, and not narrowly.**

| statistic | value |
|---|---|
| cells run | 360 |
| cells with t ≥ 2.0 | **0** |
| cells net positive | 22 of 360 |
| best positive t | 1.590 |
| best cell net/contract | $8.22 (n=81, t=1.56, p=0.12) |

Net positivity by horizon is 0% at 1s and 5s, 1.7% at 10s and 30s, 3.3% at 60s,
30% at 300s. That gradient is the cost floor, not a signal: the shorter the hold,
the more completely $2.06/contract dominates the move. The largest |t| in the
whole run is 91.9 **on a losing cell** — short-horizon cells lose money with
overwhelming significance, which is the cost drag measured precisely.

## Why this is a real kill and not a measurement failure

The two defects that made the September 2026 verdict untestable were both fixed
and both verified by test:

- OFI is now accumulated per best-quote transition, not by differencing bar
  endpoints. `test_ofi_accumulates_every_event_not_just_bar_endpoints` locks the
  size of the old error at 10× on a ten-tick walk.
- OFI is divided by time-weighted depth, per CKS.

The corrected estimator was applied to a true L3 book replay (24.3M events/day,
crossed-book rate 0.044%) over nine days. **The measurement is now right and the
signal is still absent.** A one-day smoke run had shown the 1-second IC rising
6× under the correction (+0.0102 vs +0.0017); on nine days that flips sign to
−0.0066. The one-day number was noise, and it is disclosed in PREREG §7 as
having been seen before the DEV run.

## The single non-null observation, recorded and NOT chased

`queue_imbalance` at a 1-second horizon has IC **+0.0259 in the thinnest 10% of
book states** against +0.0025 across all states — a 10× conditioning effect, and
the only place in the run where depth conditioning did what the mechanism
predicts. It is not tradeable: a 1-second horizon against a $2.06/contract round
trip needs a move an order of magnitude larger than anything at that horizon, and
the 1-second cells are 0/60 net positive. It corroborates the standing memory
note that queue imbalance is a maker signal, and the maker route is closed by
arithmetic (below).

Per PREREG §3 this is not a thread to pull. It is recorded so it need not be
rediscovered.

## What is now closed, and how firmly

**The taker microstructure family on MNQ is closed.** Three independent kills:
the 24-day L2/BBO study (Jul 2026), the one-day MBO screen (Sep 2026), and this
nine-day corrected screen. The published literature agrees in as many words:
order-book imbalance effects are not strong enough to beat spread plus fees for
someone crossing the spread.

**The maker route is closed by arithmetic, not by evidence.** Measured on MNQ RTH
2026-09-04: spread is 1 tick 37.9% of the time and 2 ticks 60.6%, mean 1.64 ticks
= $0.82, so the half-spread is $0.41. Commission is $0.62/side. A flawless market
maker who always earned the full spread and was never adversely selected would
net **−$0.42 per round trip**. No signal quality fixes that. This closes
queue-position and passive-fill ideas without needing to test them.

**Do not re-open either without a change in the cost structure.** The binding
number is commission per side versus half-spread. At roughly $0.25/side the
passive arithmetic turns positive and the question becomes live again; at $0.62
it cannot.

## Artifacts

- `ml_intraday_v3/features/depth_scaled_flow.py` — 20 tests, tick-indexed numba
  book replay, 24.3M events in 5.2s
- `ml_intraday_v3/diagnostics/depth_scaled_screen.py`
- `ml_intraday_v3/runs/depth_scaled_screen/{ic,cells,meta}_dev.*`
- `data/processed/mbo_features/features_*.parquet` — 15 days, reusable
- prereg `PREREG_depth_scaled_ofi.md` incl. appendix A on the corpus inventory

VAL and OOE were **not** run: §3 makes H1 failure terminal, and spending the VAL
look on a hypothesis already falsified on DEV would only manufacture a chance to
find a false positive.
