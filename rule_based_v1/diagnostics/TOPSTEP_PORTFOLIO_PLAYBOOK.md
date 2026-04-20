# Topstep Portfolio Playbook

This note captures the repo's current Topstep-oriented conclusion without changing the live config.

## What The Repo Evidence Says

1. Pure ORB upsizing is not the best path.
- `funded_phase_projection.json` shows higher monthly mean PnL at 3 contracts than 2 contracts, but trailing-drawdown hit rate rises sharply.
- Read this as: bigger ORB is faster money when it works, but worse survivability.

2. ORB works best as the core of a broader book.
- `scaling_optimization_results.json` highlights combinations such as:
  - `ORB 2c + MSITE 12 + GIRE 6`
  - `Sharpe 6.14`
  - `max_day_loss -530.8`
  - `max_dd -1035.79`
  - `p_pass 0.9152`
- That is a better Topstep tradeoff than simply increasing ORB size.

3. ORB trade management should stay disciplined.
- `live_parity_results.json` shows the original one-trade, time-stop variant outperforming no-time-stop and multi-entry variants.
- That argues for preserving the current ORB structure rather than loosening it to force more trades.

## Practical Implication

If the goal is higher profitability with a positive EV path under Topstep rules:

1. Keep the morning ORB entry logic stable.
2. Avoid aggressive pure ORB contract scaling.
3. Prefer adding orthogonal engines like `MSITE` and `GIRE`.
4. Measure success with pass probability, max day loss, and drawdown, not gross PnL alone.

## Recommended Candidate Mixes To Re-Check First

These are good starting points for further validation, not automatic live changes:

1. `ORB 2c + MSITE 12 + GIRE 6`
2. `ORB 2c + MSITE 9 + GIRE 6`
3. `ORB 2c + MSITE 12 + GIRE 5`

The shared pattern is deliberate:
- keep ORB at `2 contracts`
- scale the orthogonal engines first
- keep drawdown buffers far away from Topstep hard limits
