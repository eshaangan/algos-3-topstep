# Standalone Topstep Workflow

This workflow is for a **standalone** ML candidate. It does not assume a rule-based primary edge.

## Why This Exists

The repo already had pieces for:
- balanced directional event generation
- triple-barrier execution alignment
- backtesting with Topstep-style risk controls
- scattered validation reports

What was missing was one reproducible path that says:

1. train a standalone candidate
2. test it across multiple market regimes
3. reject it unless it clears explicit Topstep-safe promotion gates

## New Runner

Use:

```bash
python -m ml_intraday_v3.experiments.run_standalone_topstep_candidate
```

Default inputs:
- training config: `ml_intraday_v3/configs/training_standalone_topstep.yaml`
- labeling config: `ml_intraday_v3/configs/labeling.yaml`
- features config: `ml_intraday_v3/configs/features.yaml`
- execution spec: `ml_intraday_v3/configs/execution_spec.yaml`
- backtest config: `ml_intraday_v3/configs/backtest_standalone_topstep.yaml`
- risk config: `ml_intraday_v3/configs/risk_topstep_50k_strict_nolock.yaml`
- promotion gates: `ml_intraday_v3/configs/standalone_viability.yaml`
- data: merges the main RTH bars file with the 2026 YTD RTH Databento delta file listed in `standalone_viability.yaml`

## What It Checks

For each OOS window it:

1. trains on a recent rolling window
2. builds balanced long/short events
3. applies triple-barrier labels
4. builds causal features
5. trains a standalone binary model
6. backtests the candidate with Topstep-style risk logic
7. checks promotion gates such as:
- positive OOS PnL
- acceptable drawdown
- no MTM daily-loss or trailing-DD liquidations
- non-degenerate long/short trade balance

Current default:
- calibration is disabled in `training_standalone_topstep.yaml` because the tested isotonic setup collapsed the candidate into no-trade outputs.
- the standalone backtest uses side-specific `y_prob` thresholds to reduce the model's short-selection skew.
- a regime-aware overlay is enabled in `backtest_standalone_topstep.yaml` to tighten acceptance in the specific long-heavy and med-vol-downtrend states that hurt the expanded 2026 OOS run.

## Output

The runner writes a timestamped directory under:

`ml_intraday_v3/experiments/results/`

Including:
- per-window `trades.parquet`
- per-window `equity.parquet`
- per-window `window_summary.json`
- `promotion_summary.json`
- `promotion_report.md`

## How To Use It

Treat a failed run as a design rejection, not as an invitation to tweak thresholds blindly.

The preferred order is:

1. fix directional balance
2. fix execution alignment
3. fix training-window relevance
4. only then test optional feature additions such as `hmm_regime` or `multi_resolution`

## Related experiment tooling

- **Multi-contract backtests (train once):** `run_candidate_contract_variants` in `experiments/run_standalone_topstep_candidate.py`; policy notes in `experiments/README_SCALING_AND_CONTRACTS.md`.
- **Decision threshold sweep (no retrain):** `python -m ml_intraday_v3.experiments.run_decision_threshold_sweep`.
- **HMM feature A/B:** `experiments/README_HMM_EXPERIMENT.md` and `configs/live_dual_meta_mes_real/features_hmm_experiment.yaml`.
