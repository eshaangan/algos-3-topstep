# Preregistration: opening OFI v1

Registered: 2026-09-10, before inspecting forward returns for this signal.

## Hypothesis

Event-level order-flow imbalance during the first 15 seconds of the regular
session predicts the direction of the following five-minute MNQ move.  The
mechanism is temporary price pressure from liquidity consumed or withdrawn at
the best quotes.  The sign is fixed in advance: positive OFI means long and
negative OFI means short.  A failed continuation test will not be rescued by
inverting the signal.

## Signal

- Formation window: 09:30:00.000 through 09:30:14.999 America/New_York.
- OFI definition: event-level Cont-Kukanov-Stoikov best-quote OFI.
- Scale: sum of event OFI divided by mean time-weighted bid-plus-ask depth.
- Direction: `sign(scaled_ofi)`; zero means no trade.
- No threshold, regime filter, weekday filter, or parameter search.

The historical signal is calculated from NQ MBP-1 files.  The recent signal is
calculated from MNQ MBO-derived one-second features.  Only the normalized sign
is transferred between feeds; magnitude is not pooled across instruments.

## Execution and outcome

- Instrument traded: one MNQ contract.
- Entry: 09:31 close, providing at least 45 seconds after signal formation.
- Exit: 09:36 close.
- Gross PnL: signed point move times $2 per MNQ point.
- Net PnL: gross PnL minus $2.05 measured round-trip cost for one contract.
- Maximum one trade per session.  No overnight position and no news sizing.

## Data partitions

- Development: historical NQ opening-book dates from 2026-01-02 through
  2026-04-30, joined only to MNQ bars for the future return.
- Validation: historical opening-book dates from 2026-05-01 through 2026-07-09.
- Final holdout: the 15 full MNQ MBO sessions from 2026-08-19 through
  2026-09-09.  This partition stays sealed until the implementation and unit
  tests pass on the earlier partitions.

## Gates

The candidate is rejected if any partition has non-positive mean net PnL.  It
is not called a deployable edge unless pooled daily net PnL has a two-sided
t-statistic of at least 2.0 and a day-block bootstrap 95% confidence interval
whose lower endpoint is above zero.  Passing these gates would authorize only
a one-contract paper trial; it would not authorize sizing or a claim of a high
Trading Combine pass probability.

This is one registered trial.  Alternative horizons, thresholds, inversions,
and subgroup filters are separate future hypotheses and may not be reported as
confirmation of opening OFI v1.
