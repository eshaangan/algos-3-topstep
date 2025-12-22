# Claude Code Project Rules — ML Pipeline V3 (ES/MES Intraday Futures)

These rules are the **single source of truth** for how Claude Code must implement V3 in this repo.  
They are derived from your current V3 blueprint documents and the Quant-Guild integration report/appendix.

**Non-negotiable principle:** correctness + leakage safety + reproducibility > speed or feature count.

---

## 1) Scope and goals

### 1.1 What we are building
A research-grade, execution-aligned intraday futures ML pipeline for **ES/MES**, supporting **both 1-minute and 5-minute bars**, with:
- **Triple-barrier labeling** (volatility-scaled PT/SL + vertical barrier)
- **Meta-labeling** (secondary model for selection)
- **Non-IID controls** (sample weights via event uniqueness + magnitude)
- **Leakage-safe validation** (Purged + Embargoed CV, CPCV)
- **Backtest overfitting diagnostics** (PBO, DSR; plus optional SPA/Reality-Check style data-snooping controls)
- **Topstep risk gates** integrated into backtest and live logic
- **Reproducible experiments** (configs + artifacts + run manifests)

### 1.2 Incremental implementation policy
Implement V3 in **small, reviewable PR-sized stages** that each:
- add tests
- write artifacts
- update run manifest
- do not break V1/V2 code paths

---

## 2) Repo boundaries and structure

### 2.1 Do not break existing code
- Do not refactor unrelated modules.
- Do not change V1/V2 behavior unless explicitly requested.

### 2.2 V3 code lives under a dedicated folder
All new work must live under one of these (pick one and be consistent):
- `ml_intraday_v3/` (preferred)
- or `ml_pipeline_v3/`

### 2.3 Suggested V3 layout (must remain stable)
```
ml_intraday_v3/
  cli.py
  RUN.md
  configs/
    execution_spec.yaml
    data.yaml
    sessions.yaml
    features.yaml
    labeling.yaml
    validation.yaml
    risk.yaml
    retrain_policy.yaml
    metrics_contract.json
  data/
    ingest.py
    continuous.py
    reindex.py
    resample.py
    session.py
    qa.py
  features/
    registry.py
    build.py
    scaling.py
    volatility_forecast.py      # optional module from Quant Guild
    regime_markov.py            # optional module from Quant Guild
    transforms/
      pca.py                    # optional, fold-safe only
  labels/
    events.py
    triple_barrier.py
  weights/
    uniqueness.py
    magnitude.py
  cv/
    purged.py
    embargo.py
    cpcv.py
  models/
    primary_train.py
    meta_train.py
    calibration.py
  backtest/
    engine.py
    execution.py
  analysis/
    pbo.py
    dsr.py
    bootstrap.py                # block/stationary bootstrap for non-IID CI
    ablation.py
  reporting/
    summary.py
  tests/
    ...
```

---

## 3) Environment & dependency rules

### 3.1 Python + venv is mandatory
- Python: **3.10+**
- All commands must be run inside a project virtual environment:
  - `.venv/` local venv
  - no system python installs
- Provide a `requirements-mlv3.txt` or `pyproject.toml` with pinned versions.

### 3.2 Determinism
- Seed **all** RNG sources (numpy, python `random`, model library).
- Persist seeds and run IDs into `run_manifest.json`.

---

## 4) Reproducibility contract (required)

### 4.1 Run directory convention
Every pipeline execution writes into:
```
runs/<run_id>/bar_size=1m/...
runs/<run_id>/bar_size=5m/...
```

### 4.2 Run manifest (required for every run)
Every run must write `runs/<run_id>/run_manifest.json` containing:
- git hash (or “dirty” state + diff hash)
- full config snapshots (all YAML + metrics_contract)
- bar size (1m/5m)
- feature schema hash + file path
- label schema hash + file path
- CV split IDs + file path
- model IDs + calibration IDs
- execution spec hash
- cost model hash
- timestamp + environment info (python version, OS, package versions)

If any config changes, it must result in a new `run_id` or an incremented run version.

---

## 5) Data rules (canonical truth)

### 5.1 Time and indexing
- Store canonical timestamps internally as **UTC**.
- Session features must also be computed in `America/Chicago`, but the stored index remains UTC.
- Data must be strictly monotonic increasing; no duplicates.

### 5.2 Reindexing
- Reindex to a full bar grid; mark synthetic/missing bars.
- Persist a QA report including missing-bar percentages per day.

### 5.3 Contract roll
- Roll schedule must be deterministic and persisted (`roll_schedule.csv`).
- Roll logic must be configurable (volume-based rule recommended).
- On roll days, document whether bars are excluded, stitched, or adjusted—do one consistently.

### 5.4 Bar size policy (1m + 5m)
- Canonical choice: build a canonical 1m series, then resample to 5m.
- If native 5m exists, it may be used only if explicitly configured and documented.
- For comparisons, keep date splits identical across bar sizes.

Artifacts (per bar size):
- `bars.parquet`
- `qa_report.json`
- `roll_schedule.csv`

---

## 6) Execution alignment contract (required)

### 6.1 `execution_spec.yaml` is the single source of truth
Labels, backtests, and live trading must use the same file for:
- fill model (next bar open/close)
- intrabar touch ordering assumptions (stop-first/target-first/OHLC path rule)
- slippage (ticks) and commissions (per contract)
- session entry/exit constraints
- max holding bars (must match vertical barrier horizon)
- max concurrent positions
- order sizing caps and rounding

### 6.2 Label/backtest parity is mandatory
- A parity harness must show that “label-derived PnL” matches backtest PnL on the same event set within tolerance.
- If parity fails, **stop** and fix before adding features/models.

---

## 7) Features (leakage-safe by construction)

### 7.1 Feature registry required
All features must be registered with metadata:
- `name`
- `lookback_bars`
- `uses_rolling_stats` (bool)
- `requires_scaling` (bool)
- `fit_on_train_only` (bool) (PCA/regime models)
- `bar_sizes_supported` (`["1m","5m"]`)

### 7.2 Causality and leakage rules
- A feature at time `t` may use only data with timestamps `<= t`.
- No global normalization/scaling. All scalers/transforms must be fit **inside** each CV split on train-only data.

### 7.3 Multi-timeframe support
- If using both 1m and 5m features together, columns must be namespaced:
  - `f1m_*`, `f5m_*`
- Any join between timeframes must be causal (align 5m features to the last completed 5m bar at time t).

### 7.4 Quant-Guild-derived optional modules (must be toggleable)
These modules may be added only after the baseline harness is stable:
- `volatility_forecast.py` (EWMA first; optional GARCH later)
- `regime_markov.py` (3-state Markov probabilities)
- `transforms/pca.py` (fold-safe PCA only)
- `sizing_kelly.py` (optional; always capped; Topstep risk dominates)

---

## 8) Labels (triple barrier) and meta-labeling

### 8.1 Event policy
- Primary model: events at **every bar** (recommended baseline, minimal event-selection bias).
- Meta model: events at **candidate signal times**.

### 8.2 Triple barrier requirements
At event time `t0`:
- Compute volatility proxy σ(t0) (ATR baseline; optional forecast σ later).
- Define PT/SL barriers:
  - `U = P0 + m_pt * σ`
  - `L = P0 - m_sl * σ`
- Vertical barrier:
  - `t1 = t0 + horizon_bars`
- Output label `y ∈ {-1,0,+1}` based on first touch before `t1`.
- Costs/slippage must be accounted for consistently.

Artifacts:
- `events.parquet` including `t0,t1,label,σ,pt_mult,sl_mult,horizon,ret_at_t1`
- `label_schema.json`

---

## 9) Sample weights (non-IID control)

### 9.1 Uniqueness weighting is required
Compute event concurrency and uniqueness (per López de Prado-style non-IID control).

### 9.2 Magnitude weighting is optional but must be clipped
If using return magnitude, clip to avoid domination by outliers.

Artifacts:
- `weights.parquet` per bar size, keyed by event ID.

---

## 10) Validation rules: Purged + Embargoed CV, CPCV, and overfitting diagnostics

### 10.1 Purged CV is mandatory
No training sample whose event interval overlaps a test interval may be included in train.

### 10.2 Embargo is mandatory
Remove an embargo window after each test fold (bar-size dependent).

### 10.3 CPCV is required for model selection
Implement combinatorial purged CV with a configurable cap on number of paths.

### 10.4 PBO + DSR are required in reporting
- Compute Probability of Backtest Overfitting (PBO) from CPCV results.
- Compute Deflated Sharpe Ratio (DSR) for strategies/models selected under multiple trials.

### 10.5 Optional (recommended) data-snooping controls
If multiple variants are compared, include an SPA/Reality-Check style control in reporting (toggleable).

Artifacts (per bar size):
- `cv_splits.json`
- `metrics_by_path.csv`
- `pbo_report.md`
- `dsr_report.md`

---

## 11) Training rules (primary + meta)

### 11.1 Fold-safe transformations
For each split:
- Fit scalers/transforms on train only.
- Calibrate probabilities only using train/val data (never test).

### 11.2 Calibration is required
Use Platt scaling or isotonic regression, fit fold-safely.

### 11.3 Model outputs must be persisted
Persist:
- model file
- calibration file
- feature schema version
- training params + seeds

---

## 12) Backtesting and risk (Topstep gates)

### 12.1 Risk rules must be enforced during backtest
Backtest must route all trades through the same risk engine used in live:
- daily loss limit
- trailing drawdown
- max contracts
- session restrictions
- forced flatten rules

### 12.2 Cost model must match execution_spec
Slippage/commission assumptions used in labels must match backtest/live.

Artifacts (per bar size):
- `trades.parquet`
- `equity_curve.parquet`
- `risk_utilization_daily.csv`

---

## 13) Reporting requirements

Every run must produce:
- Summary report markdown per bar size
- Distribution of metrics across CPCV paths (median + tails)
- Per-regime breakdown (at least volatility-bucket regime; optional Markov regimes later)
- Dependence-aware confidence intervals (block or stationary bootstrap) for key metrics
- A baseline comparison section (V2 + rule-only)

---

## 14) Testing requirements (must exist before “real” modeling)

Minimum test suite:
1) **Leakage tests**
   - future perturbation: modifying future prices must not change features at/before t
2) **Label correctness**
   - synthetic OHLC series where first-touch outcomes are known
3) **Purging/embargo correctness**
   - no overlap between train intervals and test intervals
4) **Determinism**
   - same configs + seed produce identical splits and outputs
5) **Label/backtest parity**
   - label-derived PnL equals backtest PnL within tolerance

No stage is considered complete until these tests pass.

---

## 15) Coding standards

### 15.1 Readability and typing
- Use type hints throughout.
- Prefer `dataclasses` for structured config/result objects.
- Prefer `pathlib.Path` for paths.

### 15.2 Comments and docs
- Include docstrings for public functions.
- Include inline comments where the financial logic is non-obvious (labeling, CV purging, risk rules).

### 15.3 Performance constraints
- Avoid O(N²) operations on full-history minute data when possible.
- If an algorithm is heavy (e.g., uniqueness weights), implement a vectorized approach first; optimize later only if needed.
- Do not introduce Numba/CUDA until the baseline harness is correct and stable.

---

## 16) PR hygiene and “stop rules”

### 16.1 One concept per PR
Each PR should implement one stage/module and include:
- tests
- artifacts
- brief RUN.md updates

### 16.2 Stop rules (do not proceed)
Stop and fix before proceeding if:
- label/backtest parity fails
- leakage tests fail
- CPCV splits are not deterministic
- PBO becomes materially worse after a change
- results improve only in-sample but degrade on the lower quartile of CPCV paths

---

## 17) Claude output format requirement (how you must respond)
For every task, output:
1) Files added/changed (paths)
2) How to run (commands)
3) Artifacts written (paths)
4) Tests added + how to run them
5) Any assumptions made (explicit)

End of rules.
