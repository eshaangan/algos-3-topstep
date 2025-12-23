# ML Intraday V3 - Run Documentation

This document describes the V3 pipeline structure, artifact storage, and how to run pipeline stages.

## Overview

V3 is a research-grade, execution-aligned intraday futures ML pipeline for ES/MES supporting:
- **Multi-bar support**: 1m and 5m bars with separate artifact paths
- **Triple-barrier labeling**: Volatility-scaled PT/SL + vertical barrier
- **Non-IID controls**: Sample weights via event uniqueness
- **Leakage-safe validation**: Purged + Embargoed CV, CPCV
- **Overfitting diagnostics**: PBO, DSR
- **Topstep risk gates**: Integrated into backtest and live logic
- **Full reproducibility**: Every run tracked via `run_manifest.json`

## Directory Structure

```
ml_intraday_v3/
├── configs/               # Configuration files (single source of truth)
│   ├── execution_spec.yaml       # Fills, costs, session rules
│   ├── data.yaml                 # Data ingestion, continuization, reindexing
│   ├── labeling.yaml             # Triple-barrier + meta-labeling
│   ├── validation.yaml           # Purged CV, CPCV, PBO, DSR
│   ├── risk.yaml                 # Topstep risk gates
│   ├── retrain_policy.yaml       # Retrain scheduling (disabled in research)
│   └── metrics_contract.json     # Metrics definitions
├── data/                  # Data pipeline modules (to be implemented)
├── features/              # Feature engineering (to be implemented)
├── labels/                # Labeling logic (to be implemented)
├── weights/               # Sample weighting (to be implemented)
├── cv/                    # Cross-validation (to be implemented)
├── models/                # Model training (to be implemented)
├── backtest/              # Backtesting engine (to be implemented)
├── analysis/              # Analysis and reporting (to be implemented)
├── tests/                 # Unit tests
├── run_manifest.py        # Run manifest schema and persistence
├── cli.py                 # CLI entry point (to be implemented)
└── RUN.md                 # This file
```

## Artifact Storage Convention

All pipeline artifacts are stored per run and per bar size:

```
runs/
└── <run_id>/
    ├── run_manifest.json          # Single manifest for entire run (all bar sizes)
    ├── bar_size=1m/               # 1-minute bar artifacts
    │   ├── bars.parquet
    │   ├── qa_report.json
    │   ├── roll_schedule.csv
    │   ├── data_metadata.json
    │   ├── events.parquet
    │   ├── label_schema.json
    │   ├── weights.parquet
    │   ├── cv_splits.json
    │   ├── cpcv_paths.json
    │   ├── cv_fold_metrics.csv
    │   ├── cpcv_path_metrics.csv
    │   ├── pbo_report.md
    │   ├── dsr_report.md
    │   ├── trades.parquet
    │   ├── equity_curve.parquet
    │   ├── risk_utilization_daily.csv
    │   └── summary_report.md
    └── bar_size=5m/               # 5-minute bar artifacts (same structure)
        └── ... (same as 1m)
```

### Run Manifest (`run_manifest.json`)

Every run writes a **single** `run_manifest.json` at `runs/<run_id>/run_manifest.json` that captures:

- **Git state**: commit hash, dirty status, diff hash
- **Config snapshots**: Full content + hash of all YAML/JSON configs
- **Schema hashes**: Feature schema, label schema, CV splits
- **Execution spec hash**: Critical for label/backtest parity verification
- **Cost model hash**: Derived from execution_spec costs section
- **Environment info**: Python version, platform, OS version, packages (optional)
- **Metadata**: Arbitrary key-value pairs (experiment name, notes, etc.)
- **Bar size**: Which bar size this run used ("1m" or "5m")

**Key invariant**: If any config changes, it must result in a new `run_id` or incremented version.

### Example Manifest Structure

```json
{
  "run_id": "baseline_v3_20250101_120000",
  "timestamp": "2025-01-01T12:00:00Z",
  "bar_size": "1m",
  "git_state": {
    "commit_hash": "abc123...",
    "is_dirty": false,
    "branch": "main",
    "tags": ["v3.0"]
  },
  "configs": [
    {
      "name": "execution_spec",
      "path": "ml_intraday_v3/configs/execution_spec.yaml",
      "content_hash": "sha256...",
      "content": { ... }
    },
    ...
  ],
  "schemas": [
    {
      "name": "feature_schema",
      "hash": "feat_hash_123",
      "artifact_path": "runs/.../bar_size=1m/feature_schema.json"
    },
    ...
  ],
  "execution_spec_hash": "exec_hash_123",
  "cost_model_hash": "cost_hash_456",
  "environment": {
    "python_version": "3.10.12",
    "platform": "macOS-14.6",
    "packages": { ... }
  },
  "manifest_version": "1.0.0"
}
```

## Configuration Files

### `execution_spec.yaml`

**Single source of truth** for instrument economics, fills, costs, and execution constraints.

Used by: labels, backtesting, live trading.

Key sections:
- `instrument`: Symbol, tick size (points), contract multiplier (USD/point)
- `fill_model`: Where fills occur (next_bar_open/close), touch ordering
- `costs`: Slippage (ticks) and commission (per bar size)
- `position_limits`: Max contracts, max concurrent positions
- `session_rules`: Allowed sessions, entry restrictions, flatten policy
- `holding_constraints`: Max holding bars (must match labeling vertical barrier)

**Critical invariant**: If execution_spec changes, the `execution_spec_hash` in the manifest changes, triggering regeneration of labels and backtest.

### `data.yaml`

Data ingestion, continuization, and reindexing configuration.

Key sections:
- `raw_data`: Input path, format, required columns
- `continuization`: Contract roll method, roll schedule
- `reindexing`: Bar sizes, resampling policy, missing bar handling
- `qa`: Quality checks and thresholds

### `labeling.yaml`

Triple-barrier labeling and meta-labeling configuration.

Key sections:
- `primary_labeling`: Event policy, volatility estimator, barrier parameters
- `triple_barrier`: PT/SL multipliers, horizon bars (grid search params)
- `meta_labeling`: Secondary model config (disabled in baseline)
- `sample_weights`: Uniqueness and magnitude weighting

**Grid search dimensions** (baseline):
- `horizon_bars`: [12, 24] for both 1m and 5m
- `pt_multipliers`: [1.0, 1.5]
- `sl_multipliers`: [1.0, 1.5]
- Total configurations: 2 × 2 × 2 = 8 per bar size

### `validation.yaml`

Cross-validation, CPCV, and overfitting diagnostics configuration.

Key sections:
- `purged_cv`: Number of folds, purging/embargo settings
- `cpcv`: CPCV parameters (n_groups=6, test_groups=2, max_paths=10)
- `overfitting_diagnostics`: PBO, DSR, data-snooping controls
- `metrics`: Primary and secondary metrics to track

**Embargo defaults**:
- 1m: 200 bars (~3.3 hours)
- 5m: 50 bars (~4.2 hours)

### `risk.yaml`

Topstep risk gates and position limits (no instrument economics).

Key sections:
- `topstep`: Account type, balance
- `daily_loss_limit`: Max daily loss ($1,000 for $50k combine)
- `trailing_drawdown`: Max drawdown from HWM ($2,500 for $50k combine)
- `position_limits`: Max contracts, notional exposure
- `forced_flatten`: Flatten rules (session close, risk thresholds)

**Must be enforced** in both backtest and live trading for realistic evaluation.

### `retrain_policy.yaml`

Retrain scheduling and deployment policy.

**Disabled in baseline research phase**. Will be used for live deployment.

### `metrics_contract.json`

Defines all tracked metrics, computation methods, and reporting requirements.

Key sections:
- `primary_metrics`: Sharpe, mean return, std, max DD, win rate, profit factor
- `secondary_metrics`: Sortino, Calmar, avg win/loss, num trades
- `risk_metrics`: Consecutive losses, daily loss breaches, VaR, CVaR
- `overfitting_diagnostics`: PBO, DSR, IS/OOS ratio
- `reporting_config`: Quantiles, confidence intervals, baselines

## Running the Pipeline

### 1. Data Preparation

**STATUS: IMPLEMENTED (Phase 1)**

The `build-data` command ingests raw data, applies continuization, reindexes to bar grids,
and writes artifacts for all configured bar sizes.

```bash
# Build data for both 1m and 5m bar sizes
python -m ml_intraday_v3.cli build-data \
  --config ml_intraday_v3/configs/data.yaml \
  --run-id baseline_v3_001 \
  --seed 42

# This will:
# 1. Load raw data from data.yaml input_path
# 2. Standardize to canonical OHLCV format (UTC timestamps)
# 3. Build deterministic contract roll schedule
# 4. Apply roll schedule (exclude roll days by default)
# 5. Reindex to 1m grid, mark synthetic bars
# 6. Resample 1m to 5m (OHLCV aggregation)
# 7. Add session features (minute_of_day, is_rth, etc.)
# 8. Run QA checks (OHLC validity, monotonic index, etc.)
# 9. Write artifacts to runs/baseline_v3_001/bar_size={1m,5m}/
# 10. Write run manifest

# Artifacts written per bar size:
# - bars.parquet           (OHLCV + session features + is_synthetic flag)
# - qa_report.json         (QA check results)
# - roll_schedule.csv      (Contract roll dates)
# - data_metadata.json     (Bar counts, date range, reindex stats)

# Single manifest for entire run:
# - run_manifest.json      (Git state, config hashes, artifact hashes)
```

**Command-line options:**

- `--config`: Path to data.yaml config file (default: `ml_intraday_v3/configs/data.yaml`)
- `--run-id`: Unique run identifier (default: auto-generated with timestamp)
- `--out`: Output directory (default: `runs/<run_id>`)
- `--seed`: Random seed for determinism (default: 42)

**Example with custom run-id:**

```bash
python -m ml_intraday_v3.cli build-data \
  --config ml_intraday_v3/configs/data.yaml \
  --run-id my_experiment_001 \
  --seed 123
```

**Viewing artifacts:**

```bash
# Check QA report for 1m data
cat runs/baseline_v3_001/bar_size=1m/qa_report.json

# Check data metadata
cat runs/baseline_v3_001/bar_size=1m/data_metadata.json

# Load bars in Python
import pandas as pd
df = pd.read_parquet("runs/baseline_v3_001/bar_size=1m/bars.parquet")
print(df.head())
print(f"Total bars: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Check run manifest
cat runs/baseline_v3_001/run_manifest.json
```

### Data Hardening Defaults

**STATUS: Phase 1.1 - Research-Grade Defaults**

V3 enforces **research-grade defaults** to prevent data leakage and ensure quality. These defaults prioritize correctness over convenience.

#### 1. NaN-by-Default for Missing Bars (No Forward-Fill)

**Default behavior**: Missing bars are kept as **NaN** (not forward-filled).

**Why**: Forward-filling creates synthetic price paths that can leak future information into historical features. This contaminates ML models with unrealistic data.

**Config location**: `ml_intraday_v3/configs/data.yaml`

```yaml
reindexing:
  missing_bars:
    # "nan" (DEFAULT - research-grade): Keep missing bars as NaN
    # "forward_fill" (RISKY): Forward-fill OHLCV from prior bar
    missing_fill_mode: "nan"
    forward_fill_max_consecutive: 0
    add_synthetic_flag: true
```

**How missing bars are handled**:
- The `is_synthetic` column marks bars that were added to complete the grid
- With `missing_fill_mode="nan"`:
  - Synthetic bars have **NaN** for OHLCV columns
  - Volume is 0 (no actual trading occurred)
  - Session features (minute_of_day, is_rth) are still computed
- Missing bar statistics are tracked in `data_metadata.json` and `qa_report.json`

**Enabling forward-fill (NOT RECOMMENDED)**:

If you must forward-fill (e.g., for visualization or exploratory analysis):

```yaml
reindexing:
  missing_bars:
    missing_fill_mode: "forward_fill"
    forward_fill_max_consecutive: 5  # Max consecutive bars to fill
```

**WARNING**: When forward-fill is enabled:
- A warning is logged at runtime
- Forward-filled bars are tracked separately in metadata
- Volume is set to 0 for forward-filled bars
- You must document why forward-fill was used in your experiment notes

**Best practice**: Use NaN-by-default and handle missing data explicitly in downstream stages (feature engineering, modeling) rather than hiding it with synthetic prices.

#### 2. QA Fail-Fast (Halt on Data Quality Violations)

**Default behavior**: QA checks **halt the pipeline** on any violation (fail-fast mode).

**Why**: Continuing with bad data produces unreliable results. It's better to fix data issues upfront than debug downstream model failures.

**Config location**: `ml_intraday_v3/configs/data.yaml`

```yaml
qa:
  # true (DEFAULT - research-grade): Raise exception and stop on any violation
  # false (WARNING MODE): Write violations to qa_report.json but continue
  qa_fail_fast: true

  checks:
    - monotonic_index
    - no_duplicates
    - missing_bar_pct
    - ohlc_validity
    - volume_sanity

  thresholds:
    max_missing_bar_pct_per_day: 0.10  # 10% max missing per day
    max_ohlc_violations: 0              # No invalid OHLC bars allowed
    max_duplicate_timestamps: 0         # No duplicate timestamps allowed
```

**QA checks performed**:

1. **monotonic_index**: Timestamps must be strictly increasing
2. **no_duplicates**: No duplicate timestamps allowed
3. **missing_bar_pct**: Missing bars per day must be < threshold
4. **ohlc_validity**: Low ≤ min(Open,Close) ≤ max(Open,Close) ≤ High
5. **volume_sanity**: Volume ≥ 0 (no negative volume)

**What happens when QA fails (fail-fast mode)**:

```
2025-01-15 10:30:45 [ERROR] cli:
================================================================================
QA CHECKS FAILED - PIPELINE HALTED
================================================================================
QA checks FAILED: 1/5 checks failed
Failed checks: ['ohlc_validity']
  - ohlc_validity: Found 3 OHLC violations (threshold: 0)

To disable fail-fast mode (NOT RECOMMENDED), set:
  qa_fail_fast: false
in configs/data.yaml
```

The pipeline **exits immediately** with status code 1. No artifacts are written.

**Disabling fail-fast (WARNING MODE - NOT RECOMMENDED)**:

If you need to inspect data quality issues without halting:

```yaml
qa:
  qa_fail_fast: false  # WARNING MODE
```

When disabled:
- QA violations are logged as warnings (not errors)
- Pipeline continues and writes artifacts
- `qa_report.json` contains full violation details
- `data_metadata.json` has `qa_passed: false`

**Use case for WARNING MODE**: Exploratory data analysis when you're investigating data quality issues and need to see the full extent of problems.

**Best practice**: Always use fail-fast mode (`qa_fail_fast: true`) for production runs and model training. Only disable for debugging data issues.

#### 3. Roll-Day Policy (Exclude Contract Roll Days)

**Default behavior**: Days when contract rolls occur are **excluded** from the dataset.

**Why**: Roll days can have unusual price action and volume as traders migrate to the new contract. Excluding them avoids contamination.

**Config location**: `ml_intraday_v3/configs/data.yaml`

```yaml
continuization:
  # "exclude" (DEFAULT - research-grade): Drop entire day when roll occurs
  # "include": Keep roll day bars (will have mixed contracts)
  roll_day_policy: "exclude"
```

**What "exclude" means**:
- When the roll schedule indicates a roll on date D, **all bars from date D are removed**
- This ensures clean separation between old and new contract data
- Roll dates are documented in `runs/<run_id>/bar_size={1m,5m}/roll_schedule.csv`

**Roll schedule example**:

```csv
old_contract,new_contract,roll_date,reason
ESH25,ESM25,2025-03-14,volume_crossover
ESM25,ESU25,2025-06-13,volume_crossover
```

**Alternative**: Use `roll_day_policy: "include"` if you want to keep roll day data (not recommended for baseline models).

#### 4. 5-Minute Resample Alignment

**Timestamp alignment**: 5m bars are labeled with the **right edge** of the aggregation window.

**Why**: This is standard financial convention - a bar's timestamp represents the END of the period.

**Example**:
```
5m bar timestamped 09:35:00 contains 1m bars:
  [09:31:00, 09:32:00, 09:33:00, 09:34:00, 09:35:00]

The 09:35 timestamp = END of 5-minute window
```

**Pandas resample settings**:
- `label="right"`: Timestamp represents end of aggregation window
- `closed="right"`: Window includes the right edge (09:35)

**OHLCV aggregation rules** (1m → 5m):
- **Open**: First open in 5m window
- **High**: Max high in 5m window
- **Low**: Min low in 5m window
- **Close**: Last close in 5m window
- **Volume**: Sum of volume in 5m window

**Implementation**: See `ml_intraday_v3/data/resample.py` for full documentation.

**Critical for reproducibility**: Do not change resample settings (`label`, `closed`) unless you have a specific reason and document it in your experiment notes.

#### Summary of Research-Grade Defaults

| Setting | Default | Rationale |
|---------|---------|-----------|
| `missing_fill_mode` | `"nan"` | Avoid synthetic price paths |
| `forward_fill_max_consecutive` | `0` | No forward-filling |
| `qa_fail_fast` | `true` | Halt on data quality violations |
| `roll_day_policy` | `"exclude"` | Remove roll day contamination |
| Resample alignment | `label="right", closed="right"` | Standard financial convention |

**Philosophy**: Start with the safest defaults (NaN, fail-fast, exclude) and only relax them with explicit documentation when required for specific experiments.

### 2. Feature Engineering

**STATUS: IMPLEMENTED (Phase 2)**

The `build-features` command computes a minimal, deterministic, leakage-safe feature set for both 1m and 5m bars.

**Key properties:**
- **No lookahead**: Every feature at time t uses only data with timestamps ≤ t
- **Deterministic**: Feature columns are always in the same order (registry-based)
- **Multi-bar support**: Automatically processes both 1m and 5m bar sizes
- **NaN handling**: Features are NaN where they cannot be computed; usable_for_training mask provided

```bash
# Build features for a completed data run
python -m ml_intraday_v3.cli build-features \
  --run-dir runs/baseline_v3_001 \
  --features-config ml_intraday_v3/configs/features.yaml

# This will:
# 1. Load bars.parquet for each bar size (1m, 5m)
# 2. Compute features causally (no lookahead)
# 3. Write features.parquet per bar size
# 4. Write feature_schema.json per bar size
# 5. Update run_manifest.json with feature artifacts

# Artifacts written per bar size:
# - features.parquet          (Feature DataFrame, same index as bars)
# - feature_schema.json        (Feature metadata + schema hash)
```

**Command-line options:**

- `--run-dir`: Run directory containing bar_size={1m,5m} directories (REQUIRED)
- `--features-config`: Path to features.yaml config file (default: `ml_intraday_v3/configs/features.yaml`)

**Example:**

```bash
# After building data
python -m ml_intraday_v3.cli build-data \
  --run-id experiment_001 \
  --seed 42

# Build features
python -m ml_intraday_v3.cli build-features \
  --run-dir runs/experiment_001
```

**Minimal Feature Set (Phase 2)**

The baseline feature set includes:

**Returns:**
- `log_return_1`: Single-bar log return
- `log_return_{k}`: Multi-bar returns (1m: k=[3,6]; 5m: k=[2,4])

**Volatility:**
- `true_range`: True range (max of: high-low, |high-prev_close|, |low-prev_close|)
- `atr_14`: Average True Range (14-bar EMA of true_range)

**Trend:**
- `ema_13`: Fast EMA of close
- `ema_34`: Slow EMA of close
- `ema_spread`: ema_13 - ema_34
- `ema_ratio`: ema_13 / (ema_34 + eps)

**Candle Structure:**
- `candle_body`: close - open
- `candle_range`: high - low
- `body_pct`: candle_body / max(candle_range, eps)
- `upper_wick`: high - max(open, close)
- `lower_wick`: min(open, close) - low

**Time (Cyclical Encoding):**
- `minute_of_day_sin`: sin(2π × minute_of_day / 1440)
- `minute_of_day_cos`: cos(2π × minute_of_day / 1440)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)

**Meta:**
- `is_synthetic`: Flag from reindexing (marks missing/synthetic bars)
- `usable_for_training`: Mask indicating rows with all non-NaN features

**NaN Handling**

Features follow the **keep_with_mask** policy (configurable in features.yaml):

1. **Lookback NaNs**: Features with lookback (e.g., log_return_6) have NaN for initial bars
2. **Synthetic bar NaNs**: If input bars have NaN OHLCV (from reindexing), output features are NaN
3. **Usable mask**: The `usable_for_training` column is True only where ALL features are non-NaN

Example:
```python
import pandas as pd

# Load features
features = pd.read_parquet("runs/baseline_v3_001/bar_size=1m/features.parquet")

# Check usable rows
print(f"Total rows: {len(features)}")
print(f"Usable for training: {features['usable_for_training'].sum()}")

# Filter to usable rows only
usable_features = features[features['usable_for_training']]

# Check for any remaining NaNs (should be none in usable rows)
assert not usable_features.drop(columns=['is_synthetic', 'usable_for_training']).isna().any().any()
```

**Feature Schema**

Each bar size gets a `feature_schema.json` with:
- Feature column names (in deterministic order)
- Feature specs (lookback, requires_scaling, etc.)
- Schema hash (for reproducibility)
- Config snapshot

Example schema:
```json
{
  "schema_version": "1.0.0",
  "schema_hash": "abc123def456",
  "bar_size": "1m",
  "n_features": 21,
  "feature_columns": [
    "log_return_1",
    "log_return_3",
    "log_return_6",
    "true_range",
    "atr_14",
    ...
  ],
  "feature_specs": [
    {
      "name": "log_return_1",
      "lookback_bars": 1,
      "uses_rolling_stats": false,
      "requires_scaling": true,
      "fit_on_train_only": false,
      "bar_sizes_supported": ["1m", "5m"],
      "description": "Log return over 1 bar: log(close_t / close_{t-1})"
    },
    ...
  ]
}
```

**Leakage Safety**

All features are **causal** - computed using only data ≤ t. This is verified by:

1. **Code review**: All computations use shift() or rolling windows that look backward only
2. **Leakage test**: `test_future_perturbation_does_not_change_past_features()`
   - Modifies future prices
   - Recomputes features
   - Asserts past features are unchanged

Run leakage test:
```bash
pytest tests/test_features.py::TestFuturePerturbation -v
```

**Bar-Size Differences**

Features are bar-size specific where appropriate:

| Feature | 1m | 5m | Notes |
|---------|----|----|-------|
| log_return_1 | ✓ | ✓ | Same definition, different scale |
| log_return_3 | ✓ | ✗ | 1m only (3-bar lookback) |
| log_return_6 | ✓ | ✗ | 1m only (6-bar lookback) |
| log_return_2 | ✗ | ✓ | 5m only (2-bar lookback) |
| log_return_4 | ✗ | ✓ | 5m only (4-bar lookback) |
| atr_14 | ✓ | ✓ | Same lookback, different interpretation |
| ema_13, ema_34 | ✓ | ✓ | Same lookback, different time horizons |
| Candle features | ✓ | ✓ | Same definition |
| Time features | ✓ | ✓ | Same definition |

**Viewing Features**

```bash
# Load and inspect features
python

>>> import pandas as pd
>>> features = pd.read_parquet("runs/baseline_v3_001/bar_size=1m/features.parquet")
>>> print(features.head())
>>> print(f"Columns: {list(features.columns)}")
>>> print(features.describe())

# Check feature schema
>>> import json
>>> with open("runs/baseline_v3_001/bar_size=1m/feature_schema.json") as f:
...     schema = json.load(f)
>>> print(f"Schema hash: {schema['schema_hash']}")
>>> print(f"N features: {schema['n_features']}")
```

### 3. Labeling (Phase 3)

```bash
# Generate triple-barrier labels
python -m ml_intraday_v3.cli build-labels \
  --run-dir runs/<run_id>/ \
  --labeling-config ml_intraday_v3/configs/labeling.yaml \
  --execution-spec ml_intraday_v3/configs/execution_spec.yaml

# This will:
# - Create events based on event_policy (every_bar for baseline)
# - Compute volatility estimates (ATR)
# - Apply triple-barrier logic using execution_spec fills/costs/touch ordering
# - Write events.parquet and label_schema.json
# - Update run_manifest.json with per-bar-size label artifacts
```

**Label Semantics**

- **Event time (`t0`)**: bar timestamp where features are computed (bar close).
- **Entry fill model** (from `execution_spec.yaml`):
  - `next_bar_open`: entry at `t0+1` open; barrier window starts at `t0+1`
  - `next_bar_close`: entry at `t0+1` close; barrier window starts at `t0+2`
- **Volatility proxy**: ATR at `t0` with period from `labeling.yaml`.
- **Barriers** (for long-side labeling):
  - `U = P0 + pt_mult * sigma + cost_buffer`
  - `L = P0 - sl_mult * sigma + cost_buffer`
  - `cost_buffer` is round-trip costs in price points if `account_for_costs=true`
- **Vertical barrier**: `t1` is derived from horizon bars and fill model.
- **Label `y`**:
  - `+1`: upper barrier hit first
  - `-1`: lower barrier hit first
  - `0`: neither hit by `t1` (vertical exit)

**Touch Ordering**

Configured via `execution_spec.yaml`:
- `stop_first`: if both barriers hit in same bar, stop takes precedence
- `target_first`: if both barriers hit in same bar, target takes precedence
- `ohlc_path`: deterministic path `open -> high -> low -> close` within the bar

**Returns & Costs**

- `ret_gross`: price points between entry and exit (no costs)
- `ret_net`: `ret_gross` minus round-trip slippage + commission
- Slippage ticks + commission are taken from `execution_spec.yaml`
- Tick size is assumed to be 0.25 points; tick value defaults to $1.25 (MES)

### 4. Sample Weights (Phase 4)

```bash
# Generate sample weights (uniqueness + optional magnitude)
python -m ml_intraday_v3.cli build-weights \
  --run-dir runs/<run_id>/ \
  --labeling-config ml_intraday_v3/configs/labeling.yaml

# This will:
# - Load events.parquet and bars.parquet
# - Compute concurrency-based uniqueness weights
# - Optionally compute magnitude weights (abs returns, clipped)
# - Write weights.parquet and weight_schema.json
# - Update run_manifest.json with per-bar-size weight artifacts
```

**Weight Semantics**

- **Concurrency**: `c(t)` = number of active events overlapping bar `t`.
- **Uniqueness**: `u_i = mean_{t in [t0_i, t1_i]} 1 / c(t)`.
- **Magnitude** (optional): `abs(ret_net)` preferred, else `abs(ret_gross)` or `abs(ret_points)`.
- **Final weight**: `w_final = w_uniqueness^a * w_magnitude^b` using exponents from `labeling.yaml`.

### 5. Cross-Validation

```bash
# Generate leakage-safe CV splits (purged + embargoed) and CPCV paths
python -m ml_intraday_v3.cli build-cv \
  --run-dir runs/<run_id>/ \
  --validation-config ml_intraday_v3/configs/validation.yaml

# This will:
# - Create purged + embargoed CV splits (event-interval based)
# - Generate CPCV paths if enabled
# - Write cv_splits.json and cv_schema.json
# - Update run_manifest.json with per-bar-size CV artifacts
```

**Purging & Embargo**

- **Test interval**: `[min(test.t0), max(test.t1)]`
- **Purge**: remove any train event where `[t0, t1]` overlaps test interval
- **Embargo**: remove train events with `t0` in `(test_end, embargo_end]` where
  `embargo_end` is `test_end` shifted forward by `embargo_bars` on the bars index

**CPCV**

- Enumerates fold combinations deterministically (lexicographic order)
- Combines test intervals across selected folds, then reapplies purge + embargo

### 6. Model Training

```bash
# Train baseline model on CV splits
python -m ml_intraday_v3.cli build-train \
  --run-dir runs/<run_id>/ \
  --training-config ml_intraday_v3/configs/training.yaml \
  --cv-kind purged_kfold

# This will:
# - Build event-level dataset by joining features at event t0
# - Fit imputers/scalers on train only (leakage-safe)
# - Train a logistic regression baseline per split
# - Write per-split metrics.json, preds.parquet, and model.pkl
# - Write training_schema.json and summary.json
```

**Join & Preprocessing Rules**

- **Join rule**: features are pulled at exact event `t0`; missing t0 rows fail fast
- **Leakage-safe**: imputer/scaler fit only on train fold; test uses frozen params

**Training Inputs**

- `features.parquet` + `feature_schema.json`
- `events.parquet` + `label_schema.json`
- `weights.parquet` + `weight_schema.json`
- `cv_splits.json` + `cv_schema.json`

**Training Artifacts**

```
runs/<run_id>/bar_size=<bar_size>/training/<cv_kind>/
├── training_schema.json
├── summary.json
└── fold_*/ or path_*/
    ├── metrics.json
    ├── preds.parquet
    ├── bundle.pkl
    ├── meta_metrics.json
    ├── meta_preds.parquet
    ├── meta_model.pkl
    └── model.pkl
```

### 7. Backtesting

```bash
# Run offline backtest on CV test splits
python -m ml_intraday_v3.cli build-backtest \
  --run-dir runs/<run_id>/ \
  --training-dir runs/<run_id>/ \
  --backtest-config ml_intraday_v3/configs/backtest.yaml \
  --cv-kind purged_kfold

# This will:
# - Load test predictions (primary/meta) for each split
# - Simulate trades with execution + costs
# - Apply Topstep risk gates in chronological order
# - Write trades.parquet, equity.parquet, backtest_metrics.json
```

**Backtest Artifacts**

```
runs/<run_id>/bar_size=<bar_size>/backtests/<cv_kind>/
├── backtest_schema.json
├── summary.json
└── fold_*/ or path_*/
    ├── trades.parquet
    ├── equity.parquet
    └── backtest_metrics.json
```

**Risk Gates Implemented**

- Daily loss limit (halt trading after breach)
- Trailing drawdown (halt trading after breach)
- Max trades per day and min time between trades
- Forced flatten at `flatten_time_chicago` (exit at bar close)
- MTM risk gating: during open trades, equity is checked each bar close; daily loss or trailing DD breaches trigger immediate liquidation at that bar close

**Note**

Offline evaluation only on CV test sets; no live trading.

### 8. Experiments

```bash
# Run experiment grid (threshold sweeps) + diagnostics
python -m ml_intraday_v3.cli run-experiments \
  --run-dir runs/<run_id>/ \
  --grid-config ml_intraday_v3/configs/experiment_grid.yaml
```

**Experiment Artifacts**

```
runs/<run_id>/experiments/
├── exp_<id>/
│   ├── config_snapshot.json
│   ├── results.parquet
│   ├── pbo.json
│   └── dsr.json
└── leaderboard_bar_size=<bar_size>.parquet
```

**Diagnostics Notes**

- `pbo.json` includes explicit definitions for path, selection rule, lambda (rank), and PBO interpretation.
- `dsr.json` reports DSR on per-trade `pnl_usd` and per-day aggregated `pnl_usd`, plus trade count and average holding minutes.

### 9. Analysis and Reporting

### 10. Audit

```bash
# Run end-to-end audit for a run
python -m ml_intraday_v3.cli run-audit \
  --run-dir runs/<run_id>/ \
  --strict false
```

**Audit Invariants**

- Alignment: event t0/t1 on bars index; features index matches bars; weights event IDs match events.
- Leakage: train/test disjoint; purge overlap = 0; embargo window clear.
- Accounting: cost mode consistency with ret_net; pnl identity check using instrument params from execution_spec.
- Risk: daily loss and trailing drawdown not breached; forced flatten respected.
- Experiments: leaderboards reference existing exp dirs; diagnostics inputs exist; PBO not applicable without CPCV.
- Provenance: audit uses configs from `run_manifest.json` when available; fallbacks to repo defaults are recorded in the report (including execution_spec instrument params).

### 11. Walk-Forward

```bash
# Run walk-forward evaluation (chronological, no CV)
python -m ml_intraday_v3.cli run-walkforward \
  --run-dir runs/<run_id>/ \
  --walkforward-config ml_intraday_v3/configs/walkforward.yaml
```

**Walk-Forward Artifacts**

```
runs/<run_id>/walkforward/bar_size=<bar_size>/
├── walkforward_schema.json
├── summary.json
└── window_<k>/
    ├── model_bundle.pkl
    ├── preds.parquet
    ├── trades.parquet
    └── metrics.json
```

**Notes**

- Windows are computed in Chicago time; train window strictly precedes test window.
- Model bundles include preprocessing state, feature order, thresholds, and config hashes for live readiness.

```bash
# Generate reports with overfitting diagnostics
python ml_intraday_v3/cli.py analysis report \
  --run-id baseline_v3_001

# This will:
# - Compute metrics across CPCV paths
# - Calculate PBO, DSR
# - Generate distribution plots (IS vs OOS)
# - Write pbo_report.md, dsr_report.md, summary_report.md
```

## Reproducibility Workflow

### Verifying Reproducibility

```bash
# Run 1
python ml_intraday_v3/cli.py run-all \
  --config-dir ml_intraday_v3/configs \
  --run-id repro_test_001 \
  --bar-size 1m \
  --seed 42

# Run 2 (same configs, same seed)
python ml_intraday_v3/cli.py run-all \
  --config-dir ml_intraday_v3/configs \
  --run-id repro_test_002 \
  --bar-size 1m \
  --seed 42

# Compare manifests
python ml_intraday_v3/cli.py compare-runs \
  --run-id-1 repro_test_001 \
  --run-id-2 repro_test_002

# Should report:
# - Identical config hashes
# - Identical schema hashes
# - Identical execution_spec_hash
# - Identical CV split IDs
# - Identical metrics (within floating-point tolerance)
```

### Comparing Different Configurations

```bash
# Baseline run
python ml_intraday_v3/cli.py run-all \
  --run-id baseline_001 \
  --bar-size 1m

# Modified config run (e.g., different slippage)
# Edit configs/execution_spec.yaml to change slippage_ticks
python ml_intraday_v3/cli.py run-all \
  --run-id slippage_test_001 \
  --bar-size 1m

# Compare manifests
python ml_intraday_v3/cli.py compare-runs \
  --run-id-1 baseline_001 \
  --run-id-2 slippage_test_001

# Should report:
# - execution_spec_hash differs (cost model changed)
# - Shows which config sections changed
```

## Multi-Bar Size Workflow

Running for **both** 1m and 5m bar sizes:

```bash
# Option 1: Sequential runs (separate run IDs)
python ml_intraday_v3/cli.py run-all \
  --run-id experiment_001 \
  --bar-size 1m

python ml_intraday_v3/cli.py run-all \
  --run-id experiment_002 \
  --bar-size 5m

# Option 2: Single run, both bar sizes (planned)
python ml_intraday_v3/cli.py run-all \
  --run-id experiment_001 \
  --bar-sizes 1m 5m

# This writes:
# runs/experiment_001/bar_size=1m/...
# runs/experiment_001/bar_size=5m/...
# runs/experiment_001/run_manifest.json (captures both)
```

## Testing

### Run Unit Tests

```bash
# All tests
cd ml_intraday_v3
pytest tests/ -v

# Specific test module
pytest tests/test_run_manifest.py -v

# Specific test
pytest tests/test_run_manifest.py::TestHashContent::test_hash_dict_deterministic -v
```

### Required Test Coverage

Before any stage is considered complete, the following tests must pass:

1. **Leakage tests**: Future perturbation must not change features at/before t
2. **Label correctness**: Synthetic OHLC series with known first-touch outcomes
3. **Purging/embargo correctness**: No overlap between train/test intervals
4. **Determinism**: Same configs + seed = identical splits and outputs
5. **Label/backtest parity**: Label-derived P&L matches backtest P&L within tolerance

## Development Workflow

### Adding a New Stage

1. **Create module** under appropriate directory (e.g., `labels/triple_barrier.py`)
2. **Add tests** under `tests/test_<module>.py`
3. **Update RUN.md** with usage instructions
4. **Run tests** to verify correctness
5. **Update manifest** if new artifacts are generated

### PR Hygiene

Each PR should:
- Implement **one concept** (one stage/module)
- Include **tests** for the new functionality
- Update **RUN.md** if user-facing
- Write **artifacts** with proper paths
- **Not break** V1/V2 code paths

### Stop Rules

**Stop and fix** before proceeding if:
- Label/backtest parity fails
- Leakage tests fail
- CPCV splits are not deterministic
- PBO becomes materially worse after a change
- Results improve only in-sample but degrade on lower quartile of CPCV paths

## Configuration Best Practices

### DO

- ✓ Edit configs in `ml_intraday_v3/configs/` before running pipeline
- ✓ Use different `run_id` for each experiment
- ✓ Keep execution_spec.yaml in sync across labels/backtest/live
- ✓ Version control all config files
- ✓ Document config changes in git commits
- ✓ Check run_manifest.json after each run

### DON'T

- ✗ Hard-code parameters in Python modules (use configs)
- ✗ Reuse run_id for different configs (breaks reproducibility)
- ✗ Modify execution_spec without regenerating labels
- ✗ Skip QA checks or suppress warnings
- ✗ Disable purging/embargo without documentation

## Troubleshooting

### Manifest hash mismatch

**Symptom**: Different runs with "same" configs have different execution_spec_hash.

**Cause**: Configs were edited between runs.

**Fix**: Check git diff on configs/, use same git commit for both runs.

### Label/backtest parity failure

**Symptom**: Label-derived P&L ≠ backtest P&L.

**Cause**: execution_spec.yaml out of sync, or cost accounting bug.

**Fix**: Verify execution_spec_hash is same for labels and backtest. Run parity harness test.

### CPCV non-deterministic

**Symptom**: Same configs + seed produce different CV splits.

**Cause**: Missing RNG seed, or non-deterministic data ordering.

**Fix**: Check that all RNG sources are seeded. Verify data index is strictly monotonic.

## Next Steps

**Immediate**:
- [x] Config structure and manifest schema
- [x] Data pipeline implementation (Phase 1 COMPLETE)
- [ ] Labeling implementation
- [ ] CV split implementation
- [ ] Test harness (leakage, parity, determinism)

**Near-term**:
- [ ] Feature engineering
- [ ] Model training
- [ ] Backtesting engine
- [ ] Analysis and reporting

**Long-term**:
- [ ] Meta-labeling
- [ ] Regime features (volatility forecast, Markov regimes)
- [ ] Retrain policy
- [ ] Live trading integration

---

**Version**: 1.0.0
**Last Updated**: 2025-01-01
**Author**: ML Pipeline V3 Team
