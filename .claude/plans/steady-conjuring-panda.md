# Plan: Exhaustive, Time-Series-Safe ML Search (Try Everything That Makes Sense)

## Purpose

This plan expands the search space so we cover:
- Every training window type we could reasonably want (rolling, expanding, anchored, regime-specific).
- Every label/target type we could reasonably want (classification, regression, ranking/utility).
- Multiple model families (linear, trees/boosting, probabilistic, small neural) with complexity ladders.
- Hyperparameter variations broad enough to find edge, with safeguards against leakage/overfitting.

It also fixes inconsistencies and adds a single source of truth: an **experiment axes matrix** that can be programmatically expanded with constraints.

## Context (Current State, So We Don’t Lose the Plot)

- Current reported offline performance: AUC ~0.593 (baseline ~0.509), but trading performance is poor (low win rate; negative expectancy under prior PT/SL).
- Predicted probabilities are compressed into a narrow range, so fixed thresholds like 0.55 rarely trigger trades.
- Data constraint: only 5-min MES history (roughly ~14 months, Oct 2024 to Dec 2025).
- Previously noted implementation gaps:
  - Momentum indicators (RSI/MACD/etc) exist but were disabled.
  - Normalization exists but was disabled.
  - Multi-resolution context features were not implemented.
  - Purged CV for event-based labels was missing (leakage risk).

## Immediate Fixes Before Running Big Sweeps

These are prerequisites for any “try everything” search to be meaningful:

1. Replace any simple time split with **purged+embargo CV** for triple-barrier/event labeling.
2. Ensure live inference preserves feature names end-to-end (sklearn warning fix; avoids silent column-order bugs).
3. Turn feature toggles into true experiment axes (momentum/scaling/session filters), so we actually test them.
4. Add multi-resolution aggregation (15m/30m/60m/120m) with strict no-lookahead alignment.

## Non-Negotiables (Correctness First)

1. **Purged time-series CV** for event-based labels (triple-barrier) with **embargo**.
2. **No lookahead** in:
   - Feature construction (including multi-resolution aggregation).
   - Label creation (barriers/horizons).
   - Normalization/feature selection (fit on train only).
3. **Nested tuning** (or at least a dedicated validation slice) when optimizing thresholds/hyperparams.
4. **Trading evaluation is mandatory**: AUC alone is insufficient. Track probability mass above actionable thresholds and expectancy under PT/SL with costs.

## Top-Level Decision Criteria (What “Works”)

An experiment is “promising” only if it passes all of:
- **Out-of-sample (OOS)** AUC >= 0.52 *and* PR-AUC meaningfully above baseline.
- **Calibration** acceptable: Brier improvement and ECE not worse than baseline.
- **Actionable confidence**: at least one threshold produces sufficient trades/day without collapsing expectancy.
- **OOS expectancy** > 0 under realistic PT/SL and slippage/fees assumptions.
- **Stability**: performance does not vanish when segmented by month / volatility regime / RTH vs ETH.

## Experiment Axes Matrix (Cover Every Case)

Think of the full search space as a cartesian product of axes, with constraints to avoid nonsense combinations.

### Axis A: Data Slice / Session
- `session`: `rth_only`, `eth_only`, `rth+eth`
- `exclude_periods`: `none`, `roll_week`, `major_news_days` (if we can tag)
- `contract_handling`: `continuous_adjusted`, `per_contract` (if available)

### Axis B: Event/Label Generation (Targets)

#### B1: Event definition
- `event_source`:
  - `bar_every_k` (k in {1,2,3,6,12})
  - `cusum` (threshold variants)
  - `volatility_breakout` (range/ATR trigger)

#### B2: Target type
- `target_type`:
  - `triple_barrier_classification` (hit PT before SL before time)
  - `direction_classification` (sign of return over horizon)
  - `return_regression` (future return over horizon)
  - `mfe_mae_regression` (max favorable/adverse excursion within horizon)
  - `utility_regression` (expected value under PT/SL + costs)
  - `pairwise_ranking` (rank events by future utility)

#### B3: Horizon / time stop
- `horizon_bars`: {3, 6, 12, 24, 48, 96}
- `horizon_type`: `fixed`, `vol_scaled` (e.g., until realized move reaches multiple of ATR), `session_end`

#### B4: Barrier definitions
- `barrier_units`: `points`, `ticks`, `atr_multiple`, `volatility_multiple`
- `pt` / `sl` families:
  - symmetric: (2.0/2.0), (2.5/2.5), (3.0/3.0), (3.5/3.5), (4.0/4.0)
  - asymmetric: (3.0/2.0), (3.0/2.5), (3.5/2.5), (4.0/3.0)
  - dynamic: `pt = k1 * ATR`, `sl = k2 * ATR` with (k1,k2) in {(0.8,0.8),(1.0,1.0),(1.2,1.0),(1.2,0.8)}
- `side` handling:
  - `side_from_signal` (existing)
  - `side_neutral` (learn direction)

#### B5: Class handling
- `balancing`: `none`, `undersample`, `oversample`, `event_balance_50_50`
- `class_weight`: `none`, `balanced`, `scale_pos_weight`

### Axis C: Training Windows (Every Window Type)

All windows must be time-ordered.

#### C1: Window policy
- `window_policy`:
  - `rolling_fixed` (train on last N)
  - `expanding` (start at T0, expand)
  - `anchored_rolling` (anchor start, fixed end increments)
  - `regime_conditioned` (train only on similar regimes; requires regime labels)

#### C2: Window sizes
- `train_span`:
  - `bars`: {1_000, 2_500, 5_000, 10_000, 20_000}
  - `months`: {1, 2, 3, 6, 9, 12}
- `val_span`:
  - `bars`: {250, 500, 1_000}
  - `months`: {0.5, 1, 2}
- `test_span`:
  - `bars`: {250, 500, 1_000}
  - `months`: {1, 2}

#### C3: Walk-forward layout
- `wf_step`: {1w, 2w, 1m}
- `n_splits`: {3, 5, 8}
- `purge_bars`: {0, 6, 12, 24}
- `embargo_bars`: {0, 6, 12, 24}

### Axis D: Features (Engineering + Transforms)

#### D1: Feature families (OHLCV-only compatible)
- `base_ohlcv`: existing features
- `momentum`: RSI, MACD, ROC, slope, z-scores (already implemented but disabled)
- `volatility`: realized vol, Parkinson/Garman-Klass, ATR variants, vol-of-vol
- `microstructure_proxies`: range, wick ratios, gap, intrabar position, volume shocks
- `seasonality`: minute/hour/day-of-week encodings (sin/cos)
- `multi_resolution`: aggregate 5m -> {15m, 30m, 60m, 120m} and join (ffill)
- `spectral`: FFT band power / dominant frequency on rolling returns
- `complexity`: entropy/ApEn/SampEn, Hurst exponent
- `stationarity`: fractional differentiation variants
- `interactions`: polynomial degree-2 on selected top features

#### D2: Feature set selection
- `feature_subset`:
  - `full`
  - `top_k_gain`: k in {5, 10, 15, 20, 30}
  - `permutation_top_k`: k in {10, 20}
  - `stability_selected` (keep only features stable across folds)

#### D3: Transforms (fit on train only)
- `missing`: `none`, `simple_imputer_median`
- `scaling`: `none`, `standard`, `robust`, `quantile`
- `winsorize`: `none`, `p0_1_p99_9`, `p1_p99`
- `power_transform`: `none`, `yeo_johnson`

### Axis E: Sample Weighting

Weights should be applied at the event level (not bar level) to avoid domination by clustered events.

- `weighting`:
  - `uniform`
  - `time_decay`: lambda in {0.001, 0.002, 0.005, 0.01}
  - `uniqueness` (if event overlap can be computed)
  - `uniqueness_x_time_decay`
  - `volatility_inverse` (downweight high vol regimes)

### Axis F: Models (Different Types + Complexity Ladder)

#### F1: Linear / fast baselines
- `logreg_elasticnet`
  - `C`: {0.01, 0.1, 1.0, 10.0}
  - `l1_ratio`: {0.0, 0.5, 1.0}
- `linear_svm`
  - `C`: {0.1, 1.0, 10.0}
- `ridge_regression` (for regression targets)
  - `alpha`: {0.1, 1.0, 10.0, 100.0}

#### F2: Tree ensembles (bagging)
- `random_forest`
  - `n_estimators`: {200, 500, 1000}
  - `max_depth`: {3, 5, 8, None}
  - `min_samples_leaf`: {1, 5, 20, 100}
- `extra_trees`
  - same ladder as RF

#### F3: Gradient boosting (primary)
- `lightgbm`
  - `boosting_type`: {gbdt, dart}
  - `learning_rate`: {0.01, 0.03, 0.05, 0.1}
  - `n_estimators`: {200, 500, 1000, 2000}
  - `num_leaves`: {7, 15, 31, 63, 127}
  - `max_depth`: {-1, 3, 5, 7, 9}
  - `min_child_samples`: {20, 50, 100, 300, 800}
  - `subsample`: {0.6, 0.8, 1.0}
  - `subsample_freq`: {0, 1, 5}
  - `colsample_bytree`: {0.6, 0.8, 1.0}
  - `reg_alpha`: {0.0, 0.1, 0.5, 1.0, 2.0}
  - `reg_lambda`: {0.0, 0.1, 0.5, 1.0, 2.0}
  - `min_gain_to_split`: {0.0, 0.01, 0.05, 0.1}
  - `max_bin`: {63, 255}
- `xgboost` (if dependency available)
  - `max_depth`, `eta`, `subsample`, `colsample_bytree`, `min_child_weight`, `lambda`, `alpha`, `gamma`
- `catboost` (if dependency available)
  - `depth`, `learning_rate`, `l2_leaf_reg`, `random_strength`, `bagging_temperature`

#### F4: Probabilistic / simple neural (only if data supports)
- `gaussian_nb` (sanity check)
- `mlp_small`
  - `hidden_sizes`: {(64,), (128,), (128,64)}
  - `dropout`: {0.0, 0.2, 0.5}
  - `lr`: {1e-4, 3e-4, 1e-3}
  - `weight_decay`: {0.0, 1e-4, 1e-3}

### Axis G: Probability Calibration / Uncertainty
- `calibration`: `none`, `sigmoid`, `isotonic`, `beta` (if implemented)
- `conformal`: `off`, `on` (optional; if implemented)

### Axis H: Thresholding / Trade Policy

The model’s “edge” only matters if we can pick a threshold that trades.

- `threshold_policy`:
  - `fixed`: {0.50, 0.52, 0.55, 0.58, 0.60}
  - `percentile`: {80, 85, 90, 95}
  - `ev_optimized` (optimize threshold on validation slice by expectancy)
  - `cost_sensitive` (different thresholds by regime or time-of-day)

## Phased Search (Still “Try Everything”, Without Wasting Time)

### Phase 0: Sanity + Leakage Checks (must pass)
- Run a handful of configs with:
  - `session=rth+eth`
  - `target_type=direction_classification` and `triple_barrier_classification`
  - 2-3 window policies
- Validate:
  - Purged CV is actually purging.
  - Feature building produces identical results when rerun.
  - No NaNs or label misalignment.

### Phase 1: Wide Coverage Sweep (All axes, coarse)
- Cover every axis family at least once:
  - Each `target_type`
  - Each `window_policy`
  - Each `model_family`
  - Each `transform bundle`
  - With and without `multi_resolution`
- Use **random search / Latin hypercube** over the full space with hard constraints:
  - Reject configs that are computationally absurd.
  - Reject configs that violate leakage rules.

### Phase 2: Conditional Deep Dives (zoom)
- Take top N configs per target type and:
  - Increase model complexity ladder steps.
  - Expand horizon/barrier grids around the best region.
  - Expand training window sizes and embargo lengths.

### Phase 3: Robustness Battery (break it)
- Evaluate best configs on:
  - Different months (e.g., train on 2024-xx, test on 2025-xx blocks)
  - Volatility regimes
  - RTH vs ETH
  - Time-of-day buckets
- If it fails here, it’s not real.

### Phase 4: Ensembles + Stacking (only after strong singles)
- Bagging of the same family across seeds/folds.
- Simple averaging of top 3-5 models (must preserve OOS integrity).

## Required Outputs Per Experiment (So We Can Compare Apples-to-Apples)

For every experiment, log:
- Config hash + full config JSON/YAML.
- Fold metrics:
  - AUC, PR-AUC, logloss, Brier, ECE
  - Train-vs-test gaps
- Probability distribution:
  - mean/std
  - pct above {0.55, 0.60}
  - top-decile mean probability
- Trading metrics (validation and test):
  - trades/day, win rate, expectancy
  - drawdown proxy (sequence sensitivity)
  - sensitivity to slippage and costs

## Known Issues / Fixes Needed (Keep These in Scope)

- Current “simple train/test split” should be removed in favor of purged CV (label leakage risk).
- Feature-name preservation in live predictor should be fixed (warning noise; correctness unchanged).
- Cost estimates must be measured and reconciled (previous plan had contradictory totals).

## Guardrails (Avoid Infinite “Try Everything” Loops)

“Try everything” still needs stop conditions:
- If after a wide sweep (Phase 1) the best configs fail robustness (Phase 3), ML is not viable under current data/features.
- If strong AUC exists but no threshold yields positive expectancy with realistic costs, the “edge” is not tradeable.
- If performance exists only in one narrow month/regime, treat it as likely overfit unless there’s a causal explanation.

## Implementation Notes (What the Code Should Support)

The experiment runner should:
- Accept a single config object with all axis choices.
- Generate folds with purge+embargo.
- Train model with selected transforms and calibration.
- Evaluate and write a results JSON per experiment.
- Support multi-process local runs and GCP fan-out runs.
