# HMM regime feature experiment (hmmlearn)

Live MES uses [`features.yaml`](../configs/live_dual_meta_mes_real/features.yaml) with `hmm_regime.enabled: false`. For an offline A/B test, use [`features_hmm_experiment.yaml`](../configs/live_dual_meta_mes_real/features_hmm_experiment.yaml), which enables causal `GaussianHMM` features from [`features/hmm_regime.py`](../features/hmm_regime.py) (requires `pip install hmmlearn`).

## Run promotion / OOS (same harness as baseline)

Point `--features-config` at the experiment file and keep all other configs identical:

```bash
python -m ml_intraday_v3.experiments.run_standalone_topstep_candidate \
  --features-config ml_intraday_v3/configs/live_dual_meta_mes_real/features_hmm_experiment.yaml \
  --acceptance-config ml_intraday_v3/configs/standalone_viability.yaml \
  --output-dir ml_intraday_v3/experiments/results/hmm_regime_promotion_run
```

Compare `promotion_summary.json` and per-window `window_summary.json` to a baseline run using `features.yaml`.

## Tuning notes (5m RTH)

- `min_train_samples`: minimum history before HMM emits states; too large delays usable features at the start of the sample.
- `refit_every` / `rolling_window_size`: trade stability vs adaptivity; refit less often if you see noisy state flips.

The upstream [hmmlearn](https://github.com/hmmlearn/hmmlearn) project is in limited maintenance; treat HMM outputs as experimental features and validate only on held-out windows.
