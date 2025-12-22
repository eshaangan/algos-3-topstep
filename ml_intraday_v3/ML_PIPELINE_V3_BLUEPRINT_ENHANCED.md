# ML Pipeline v3 (ES/MES Intraday Futures) — Repo-Level Technical Blueprint

This blueprint is designed to be implemented **incrementally** (small PRs), without breaking your existing codebase.
It assumes you want an ML pipeline aligned with:
- **Leakage-safe validation** (purged + embargoed CV, CPCV, PBO)
- **Execution-aligned labels** (triple-barrier + meta-labeling)
- **Non-IID controls** (uniqueness/sample weights)
- **Risk constraints** (Topstep-style daily loss + trailing drawdown gates)
- **Reproducible experiments** (configs + artifacts + run IDs)

---

## 0) ES vs MES: can ES data train a MES strategy?

**Yes — for modeling direction/returns**, ES and MES are effectively the same underlying index movement, same tick size, and (almost always) highly correlated price paths. Differences are:
- Contract multiplier (ES is larger; MES is 1/10 notional)
- Fees/commissions per contract
- Liquidity/slippage (usually still fine at 1m/5m horizons)

**Implementation rule:** train on **returns in points or %**, not dollars. Convert to dollars only for **costs**, **risk gates**, and **position sizing**.

---

## 1) New folder in current repo vs brand-new repo?

### Recommendation
**Keep the current repo, add a new v3 folder/package.**
You already have useful infrastructure (data loaders, bars cleaning, existing backtest/run scripts). A new package lets you:
- develop v3 without touching v1/v2
- reuse stable utilities
- keep deployment/execution code in one place

### When a new repo is better
Make a new repo only if:
- you want independent dependency management (clean `pyproject.toml`)
- you want a clean commit history for v3
- you want to publish v3 publicly later

### Best-of-both
- Keep current repo as the “monorepo”
- Put v3 under `ml_pipeline_v3/` with its **own** configs, notebooks, and tests
- Add strict interfaces so v3 can be extracted into a standalone repo later

---

## 2) High-level architecture

**Core pipeline stages:**

1. **Ingest** raw bars (ES/MES, 1m/5m)  
2. **Clean & standardize** (timezone, missing minutes, session filters, roll logic)  
3. **Feature engineering** (fast, deterministic, no leakage)  
4. **Labeling** (triple barrier + vertical time barrier, volatility-scaled)  
5. **Sample weighting** (uniqueness + return magnitude, capped/log)  
6. **Split generation** (purged K-fold + embargo; CPCV paths)  
7. **Training**  
   - Primary model: predicts `side` or `return distribution`  
   - Meta-model: predicts `P(profitable | signal)` / probability of exceeding loss threshold  
8. **Evaluation**  
   - CPCV distribution of metrics  
   - PBO across partitions  
   - DSR (deflated Sharpe) for selection bias  
9. **Backtest simulation** (realistic costs, slippage, Topstep gates)  
10. **Artifact export** (model, scalers, feature list, config snapshot, reports)

---

## 3) Proposed repo layout (additive, does not break existing code)

```
/ml_pipeline_v3/
  README.md
  requirements-mlv3.txt
  configs/
    data.yaml
    sessions.yaml
    features.yaml
    labeling.yaml
    cv.yaml
    model_primary.yaml
    model_meta.yaml
    costs.yaml
    risk.yaml
    run.yaml

  src/topstep_ml_v3/
    __init__.py

    ingest/
      databento_reader.py          # if using Databento-style OHLCV
      generic_csv_reader.py        # ES long-range CSV ingestion
      contract_roll.py             # roll/front-month logic (hysteresis)

    preprocess/
      time_index.py                # reindex minutes + fill rules
      sessions.py                  # RTH/ETH filters, holiday calendar hooks
      sanity.py                    # assertions & stats

    features/
      base.py                      # FeatureSpec, registry
      price_returns.py             # returns/log returns
      vol.py                       # ATR, RV, Parkinson, etc.
      trend.py                     # EMAs, slopes, filters
      microstructure_proxy.py      # volume/volatility proxies for orderflow if no L2
      normalization.py             # fit/transform (train-only)

    labeling/
      triple_barrier.py            # volatility-scaled profit/stop + vertical barrier
      meta_labeling.py             # secondary label creation from primary signals

    weights/
      concurrency.py               # overlap counting (interval tree approach)
      sample_weights.py            # w = uniqueness * f(|r|)

    cv/
      purge_embargo.py             # fold creation + purge + embargo logic
      cpcv.py                      # combinatorial paths generator

    models/
      base.py                      # fit/predict interface
      sklearn_models.py            # baseline (RF/XGB/LogReg) if used
      torch_mlp.py                 # if you keep PyTorch
      calibration.py               # Platt/Isotonic, temperature scaling

    metrics/
      sharpe.py                    # SR, PSR
      deflated_sharpe.py           # DSR
      pbo.py                       # probability of backtest overfitting
      risk_metrics.py              # MaxDD, CDaR proxies, tail loss rates

    backtest/
      simulator.py                 # event loop for entries/exits from signals
      costs.py                     # commissions, slippage models
      topstep_gates.py             # daily loss / trailing drawdown enforcement

    reporting/
      artifacts.py                 # run_id folders, save config snapshots
      plots.py                     # optional (matplotlib)

  notebooks/
    01_data_sanity.ipynb
    02_contract_roll_debug.ipynb
    03_labeling_debug.ipynb
    04_weights_and_overlap.ipynb
    05_cpcv_eval.ipynb
    06_backtest_report.ipynb

  scripts/
    build_dataset_v3.py
    train_primary_v3.py
    train_meta_v3.py
    eval_cpcv_v3.py
    backtest_v3.py

  tests/
    test_time_index.py
    test_triple_barrier.py
    test_purge_embargo.py
    test_cpcv_paths.py
    test_dsr.py

  artifacts/                       # gitignored
    ml_pipeline_v3/
      <run_id>/
        config_snapshot/
        dataset_manifest.json
        metrics.json
        models/
        plots/
```

---

## 4) Configs (concrete schema)

### 4.1 `configs/data.yaml`
- `symbol_root`: `ES` or `MES`
- `bar_size`: `1m` or `5m`
- **Optional (recommended for experimentation):** `bar_sizes`: list, e.g. `["1m","5m"]` to run the same pipeline across multiple resolutions
- `output_subdir_by_bar_size`: bool (default true) to write artifacts under `.../bar_size=1m/` and `.../bar_size=5m/`
- `raw_paths`: list of CSV/parquet paths
- `timestamp_col`: `ts_event`
- `ohlcv_cols`: `open, high, low, close, volume`
- `timezone`: store internally as `UTC`, provide `session_tz` separately
- `start`, `end` filters
- `keep_symbols_regex`: e.g. `^(ES|MES)[HJKMNQUVXZ]\d$` (exclude spreads)

### 4.2 `configs/sessions.yaml`
- `session_tz`: `America/New_York` or `America/Chicago` (be explicit)
- `rth`: `09:30-16:00` (ET)  
- `eth`: optional
- `exclude_days`: optional list
- `holiday_calendar`: `XNYS` or CME-equivalent (pluggable)

### 4.3 `configs/labeling.yaml` (triple barrier)
- `horizon_bars`: e.g. `48` for 4h on 5m, or `240` for 4h on 1m
- `atr_period`: e.g. 20
- `pt_mult`: e.g. `1.5`
- `sl_mult`: e.g. `1.0`
- `min_barrier_points`: floor to avoid tiny barriers
- `label_mode`:
  - `{-1,0,+1}` (first barrier hit)
  - or meta-label format for later stage

### 4.4 `configs/cv.yaml`
- `k_folds`: e.g. 10
- `test_folds`: e.g. 2 (or 5 for CPCV K/2)
- `purge_window_bars`: equals max label lookahead overlap (>= horizon)
- `embargo_bars`: e.g. 12 (1h on 5m)
- `cpcv_paths_limit`: optional cap for runtime

### 4.5 `configs/costs.yaml`
- `tick_size_points`: 0.25
- `tick_value_usd`: ES=12.50, MES=1.25
- `commission_per_contract`: realistic broker+exchange+NFA
- `slippage_model`:
  - fixed ticks (e.g., 1 tick per side)
  - or volatility-scaled: `slip_ticks = ceil(a + b * ATR_ticks)`

### 4.6 `configs/risk.yaml`
- `max_daily_loss_usd`
- `max_trailing_dd_usd`
- `cooldown_after_hit`: minutes/bars
- `position_sizing`:
  - `max_contracts`
  - `fractional_kelly_cap`
  - `per_trade_risk_cap_usd`

---

## 5) Data ingest & processing details (Databento-like and generic)

### 5.1 Standardize timestamps
- Parse `ts_event` as UTC-aware datetime
- Sort, drop duplicates
- Build a full minute index for the chosen bar size
- For missing minutes:
  - `close` = previous close
  - `open/high/low` = `close` (or previous close)
  - `volume` = 0
- Keep a boolean `is_synthetic_bar` for debugging and optional exclusion

### 5.2 Contract roll (if raw contains multiple expiries)
If your raw dataset contains `ESZ4`, `ESH5`, etc:
- Filter out calendar spreads and non-outrights
- Daily roll selection with hysteresis:
  - compute `daily_volume` per contract
  - roll from current -> next when `next_volume > current_volume * roll_ratio` for `roll_confirm_days`
  - once rolled, do not roll back
- Output a single continuous series with a column `active_symbol`

### 5.3 Session filtering
- Convert timestamp to `session_tz`
- Filter to RTH if your execution is RTH-only
- Or keep ETH but include `time_of_day` features

### 5.4 Data QA checks (fail fast)
- monotonic index
- no negative volume
- OHLC consistency: `low <= min(open,close) <= max(open,close) <= high`
- percent synthetic bars per day < threshold (e.g., 20%)

---

## 6) Feature engineering (no leakage, deterministic)

### 6.1 Core features (suggested starting set)
- Returns:
  - `ret_1`, `ret_5`, `ret_15` (log returns)
  - `rolling_mean_ret`, `rolling_std_ret`
- Volatility:
  - `ATR` (points), `ATR_ticks`
  - `RV` (realized vol over N bars)
- Trend:
  - EMAs: `ema_fast`, `ema_slow`, `ema_ratio`
  - slope of EMA over window
  - distance to EMA in ATR units
- Range/structure:
  - candle body, wick ratios
  - `high_low_range / ATR`
- Volume proxies:
  - rolling z-score of volume
  - volume * range
- Time features:
  - minute-of-day (sin/cos)
  - day-of-week
- Regime gates (lightweight first):
  - 2-state HMM or simple `vol_regime = ATR_z > threshold`

### 6.2 Normalization
- Fit scalers on **train only** within each CV fold (or per CPCV path)
- Store feature list + scaler params in artifacts

---

## 7) Labeling (triple barrier + meta-labeling)

### 7.1 Primary label: triple barrier classification
For each event time `t`:
- compute `atr_t`
- set upper/lower barriers around entry price `P_t`
- simulate forward up to `horizon_bars`:
  - +1 if upper hit first
  - -1 if lower hit first
  - 0 if time barrier hit first
Store:
- `label`
- `t1` (vertical barrier timestamp)
- `event_end_time` (actual barrier hit time)
- `event_return_points` and `event_return_usd` (for weighting + costs)

### 7.2 Meta-labeling (recommended for your “selective but solid” approach)
Pipeline:
1) Define a simple primary “signal generator” (can be your EMA pullback logic).
2) For every primary signal occurrence, build features at signal time.
3) Label: `meta_y = 1` if trade would be profitable net of costs under a fixed execution rule; else 0.
4) Train meta-model to predict `P(meta_y=1)` and gate trades.

This keeps trading logic interpretable while ML improves selectivity.

---

## 8) Sample weights (non-IID control)

### 8.1 Uniqueness / concurrency
- Each label has an interval `[t, event_end_time]`
- Overlap count at time `u`: number of active intervals
- Uniqueness of event i: average of `1 / overlap_count(u)` over its interval

### 8.2 Weight formula
Start simple:
- `w_i = uniqueness_i * log(1 + abs(event_return_usd))`
- cap `event_return_usd` at a percentile to avoid outlier domination

---

## 9) Validation: Purged+Embargoed CV + CPCV + PBO

### 9.1 Purged K-fold
For each test fold:
- remove any training samples whose label intervals overlap the test intervals (purge)
- apply an embargo after the test fold end (drop `embargo_bars` from training)

### 9.2 CPCV
- Build K folds
- Enumerate combinations of test folds (e.g., K=10, test=5 => 252 paths)
- For each path:
  - train on purged+embargoed train set
  - evaluate on test set
  - store Sharpe, drawdown, hit rate, etc.
Return distributions, not single numbers.

### 9.3 PBO (Probability of Backtest Overfitting)
- For each CPCV path:
  - choose “best” model/config by in-sample metric
  - check rank out-of-sample
- PBO = fraction of paths where best-IS ranks below median OOS

---

## 10) Metrics & acceptance gates (what “good” means)

Minimum acceptance gates (suggestion):
- Median CPCV Sharpe > 0.8 (net of costs)
- 25th percentile CPCV Sharpe > 0.2
- PBO < 0.3
- DSR (deflated sharpe) > 0.5 on median path
- No single CPCV path breaches Topstep trailing DD under simulated execution

If you fail gates, you don’t “tune more” — you fix leakage, costs, or label/spec.

---

## 11) Backtest simulator (execution-aligned)

### 11.1 Order model (v1)
- Market entries/exits at next bar open (or close) + slippage
- Stop/target evaluated on bar high/low (with realistic priority rules)
- One position at a time (start simple)

### 11.2 Cost model
- per-side slippage ticks
- per-contract commission
- optional spread model

### 11.3 Risk gates
Enforce:
- daily loss stop
- trailing drawdown stop
- cooldown windows after hitting a risk stop
- optional “volatility halt” if ATR spikes beyond threshold

---

## 12) Incremental implementation plan (small, safe steps)

### Milestone 1 — Data + reindex (1 PR)
- Add `ingest/` + `preprocess/time_index.py`
- Notebook: `01_data_sanity.ipynb` validates index, missing minutes, OHLC rules

### Milestone 2 — Contract roll (1 PR)
- Add `ingest/contract_roll.py`
- Notebook: `02_contract_roll_debug.ipynb` shows active symbol timeline

### Milestone 3 — Features (1–2 PRs)
- Add `features/` registry + 10–20 core features
- Unit tests for deterministic feature generation

### Milestone 4 — Triple barrier labels (1 PR)
- Add `labeling/triple_barrier.py`
- Notebook: `03_labeling_debug.ipynb` visualizes barrier hits and label balance

### Milestone 5 — Weights (1 PR)
- Add interval overlap counting + weight computation
- Notebook: `04_weights_and_overlap.ipynb`

### Milestone 6 — Purge/embargo + CPCV (1–2 PRs)
- Implement fold creation + purge/embargo
- Implement CPCV path generator + caching
- Notebook: `05_cpcv_eval.ipynb`

### Milestone 7 — Training (primary + meta) (2 PRs)
- Add baseline sklearn model first (fast iteration)
- Add torch model second (once pipeline stable)
- Save artifacts per run_id

### Milestone 8 — Backtest integration + Topstep gates (1–2 PRs)
- Simulator + costs + risk gates
- Notebook: `06_backtest_report.ipynb` produces equity curve + DD stats

---

## 13) “Tell Perplexity again” checklist (if you hit missing details)

If you need more precise implementation guidance, ask specifically for:
- **Purged/embargo overlap logic** for triple-barrier intervals (pseudo-code)
- **Efficient uniqueness** computation (interval tree / sweep-line algorithm)
- **CPCV path enumeration** and caching strategy
- **DSR implementation** details (effective number of independent trials N)
- **Realistic stop/target priority** rules on OHLC bars (intrabar ambiguity)

---

## 14) What I need from your side (later, not blocking this spec)

When you’re ready to implement the roll + sessions correctly, you’ll eventually want:
- Your preferred trading session (RTH-only vs ETH)
- Your exact Topstep rules (daily loss, trailing DD specifics)
- Your ES dataset format (timestamp, tz, columns)
- Your target bar size (1m vs 5m) and horizon (e.g., 30m, 2h, 4h)

But you can implement Milestones 1–4 immediately without these being perfect.
---

## Appendix A) Codex-ready implementation dossier (equations + modules + artifacts)

This appendix is additive: it does **not** replace earlier sections. It exists to eliminate ambiguity during implementation.

### A.1 Bar-size experimentation (1m + 5m)
V3 should support running the full pipeline at **multiple bar sizes** (at least `1m` and `5m`) with identical configs and deterministic outputs.

**Rule:** the canonical run is parameterized by `bar_size`. If `bar_sizes` is provided, run a loop:
- for each `bar_size` in `bar_sizes`:
  - reindex/QA
  - features
  - labels
  - weights
  - CV
  - training
  - backtest
  - reports
  - write artifacts under `runs/<run_id>/bar_size=<bar_size>/...`

**Guidance:**
- 1m: better for microstructure + more samples; more noise; heavier compute
- 5m: fewer samples; smoother; easier generalization
- Keep the same *calendar split logic* across bar sizes (same date ranges), even though bar counts differ.

---

### A.2 Feature engineering: concrete formulas (leakage-safe)
All features must be computable at time `t` using data `<= t`. Any rolling statistics must use a strict lookback window.

#### A.2.1 Returns
- Log return:
  - **rₜ = ln(Cₜ / Cₜ₋₁)**
- k-horizon log return:
  - **rₜ^(k) = ln(Cₜ / Cₜ₋ₖ)**

#### A.2.2 Volatility
- True Range:
  - **TRₜ = max(Hₜ−Lₜ, |Hₜ−Cₜ₋₁|, |Lₜ−Cₜ₋₁|)**
- ATR (choose one and freeze it for the project)
  - Simple rolling mean: **ATRₜ = mean(TR over N bars)**
  - Wilder EMA: **ATRₜ = EMA_wilder(TR, N)**

- Realized volatility (one acceptable definition):
  - **RVₜ = sqrt(sum_{i=1..N} rₜ₋ᵢ²)**

#### A.2.3 Trend
- EMA:
  - **EMAₜ = α·Cₜ + (1−α)·EMAₜ₋₁**, with **α = 2/(n+1)**
- EMA ratio/spread:
  - `ema_ratio = EMA_fast / (EMA_slow + eps)`
  - `ema_spread = EMA_fast - EMA_slow`
- Distance-to-EMA in volatility units:
  - **dist_ATRₜ = (Cₜ − EMAₜ) / (ATRₜ + eps)**

#### A.2.4 Structure (candles)
- `body = C - O`
- `range = H - L`
- `body_pct = body / max(range, eps)`
- `upper_wick = H - max(O, C)`
- `lower_wick = min(O, C) - L`

#### A.2.5 Volume proxies (futures)
- Rolling z-score volume:
  - **zvolₜ = (Vₜ − μ(V)) / (σ(V) + eps)** over the lookback window
- `vol_range = V * (H-L)`

#### A.2.6 Time features
- Cyclical encoding (minute-of-session):
  - **x = sin(2πm/M), y = cos(2πm/M)**

#### A.2.7 Normalization (critical: fold-safe)
- **No global scaling.** For each CV split:
  - fit scaler on train slice only
  - transform val/test with that scaler
- Persist:
  - feature list (ordered)
  - scaler params
  - code hash / config hash

---

### A.3 Triple-barrier labeling (primary labels)
Triple-barrier requires an **event set** (timestamps where outcomes are evaluated). Two valid choices:
- **Primary model:** events at every bar (recommended for clean research and less event-selection bias)
- **Meta model:** events only when a proposed trade signal occurs (selectivity layer)

#### A.3.1 Barriers
At event time `t0` with entry price `P0` and volatility estimate `σ0` (use ATR in points):

- Profit-taking barrier:
  - **U = P0 + m_pt · σ0**
- Stop-loss barrier:
  - **L = P0 − m_sl · σ0**
- Vertical barrier:
  - **t1 = t0 + H** (H = horizon_bars)

Label rule (long-direction):
- `y = +1` if price hits `U` first before `t1`
- `y = -1` if price hits `L` first before `t1`
- `y = 0` if neither hit before `t1`

#### A.3.2 Costs/slippage alignment
Costs must be included consistently in labeling and backtesting:
- Either adjust barriers:
  - `U' = U + cost_points`, `L' = L - cost_points`
- Or apply costs in payoff evaluation (but then you must not compare to cost-free labels).

**Artifacts to store for each event:**
- `t0, t1, label, atr_t0, pt_mult, sl_mult, horizon, ret_at_t1, path_max, path_min`

---

### A.4 Sample weights (non-IID control)
Each event i spans an interval `[t0_i, t1_i]`.

#### A.4.1 Concurrency
- At time `u`, concurrency is:
  - **c(u) = Σᵢ 1[t0_i ≤ u ≤ t1_i]**

#### A.4.2 Uniqueness
- Event uniqueness:
  - **uniq_i = (1/|interval_i|) · Σ_{u in interval_i} 1/c(u)**

#### A.4.3 Final weight (simple, robust)
- Start:
  - `w_i = uniq_i * log(1 + |ret_usd_i|)`
- Optional:
  - clip return magnitude to avoid extreme-weight domination.

---

### A.5 Validation: Purged + Embargoed CV, CPCV, and PBO
#### A.5.1 Purging
Remove any training event whose `[t0,t1]` overlaps the test interval.

#### A.5.2 Embargo
Remove training data in a window after the test set to prevent adjacency leakage:
- `embargo_bars` (bar-size dependent; tune per 1m vs 5m)

#### A.5.3 CPCV
Generate multiple train/test combinations across K folds with purging and embargo applied; evaluate distributions of metrics, not just a single split.

#### A.5.4 PBO (Probability of Backtest Overfitting)
Compute an overfitting alarm from CPCV results:
1) For each CPCV path, select the best variant in-sample
2) Evaluate that chosen variant out-of-sample on the paired test subset
3) Record rank/metric degradation
4) Summarize distribution; report an overfitting probability proxy

**Artifacts:**
- `cv_splits.json` (boundaries + seeds)
- `metrics_by_path.csv`
- `pbo_report.md`

---

### A.6 Training: primary + meta models (execution-aligned)
#### A.6.1 Primary model
- Targets: triple-barrier `y ∈ {-1,0,+1}` (or two binaries; but keep 0 explicitly early)
- Outputs: calibrated probabilities `p_long`, `p_short`, `p_flat`

#### A.6.2 Meta-labeling model
- Input: features at candidate entry time
- Proposed trade comes from either:
  - a simple rules signal (e.g., EMA pullback), or
  - the primary model’s direction suggestion
- Meta target:
  - `meta_y = 1` if the trade was profitable **net of costs** under your execution rules; else `0`
- Output: `p_take` (take vs skip)

---

### A.7 Backtest integration (must match live) + Topstep gates
The backtest must route every order through the same risk engine used in live execution:
- daily loss limit
- trailing drawdown
- session flat requirements
- sizing rules
- any contract-specific limits

For each `bar_size`, produce:
- trade blotter
- equity curve
- per-day risk utilization report (Topstep)

---

### A.8 Codex implementation order (lowest ambiguity)
1) Data canonicalization: ingest → reindex → QA → parquet
2) Feature builder (deterministic columns)
3) Event builder (primary events at every bar)
4) Triple-barrier labels
5) Weights (uniqueness + magnitude)
6) Purged+embargo CV generator + CPCV
7) Train primary (fold-safe scaling + calibration)
8) Train meta (signal events → meta labels → fold-safe training)
9) Backtest integration (risk gates + costs)
10) PBO report + summary docs

---

### A.9 Minimum unit tests (acceptance checks)
- **No leakage:**
  - feature columns at time t must not read future rows
  - scaler fit only on train slice within each CV split
- **Labels:**
  - deterministic triple-barrier results on a tiny synthetic series with known outcomes
- **Purging/embargo:**
  - no train interval overlaps any test interval
  - embargo window enforced
- **Backtest parity:**
  - same cost/slippage + risk logic used for live and backtest
- **Bar-size loop:**
  - pipeline runs end-to-end for both `1m` and `5m` with separate artifact directories

---

## Appendix B) Pre-Implementation Checklist (research-grounded)

This checklist is intentionally **implementation-blocking**. The goal is to lock the assumptions that most commonly create (1) hidden leakage, (2) backtest overfitting via multiple testing, and (3) label/backtest/live mismatch.

### B.1 Freeze an execution specification (single source of truth)
Create `configs/execution_spec.yaml` and require **labels**, **backtest**, and **live** to import it.

Minimum fields:
- `fill_model`: `next_bar_open` | `next_bar_close` (start with one)
- `intrabar_priority`: `stop_first` | `target_first` | `ohlc_path_assumption` (document rule)
- `slippage_ticks`: per-side slippage (separate for 1m vs 5m allowed)
- `commission_per_contract`: per-side or round-turn
- `spread_ticks`: optional
- `max_holding_bars`: vertical barrier must match this
- `session_rules`: `flatten_before_close`, `no_new_entries_after`, etc.
- `max_concurrent_positions`: start at 1

**Acceptance gate:** if labeling PnL and backtest PnL differ materially on the same event set under the same rules, stop and fix.

### B.2 Lock the “event model” for triple-barrier + meta-labeling
Primary (recommended baseline):
- `events = every bar` (cleanest evaluation; least event-selection bias)

Meta model:
- `events = candidate signal times` (EMA pullback or primary-model suggested entries)

Record these in `configs/labeling.yaml`:
- `event_policy_primary`: `every_bar`
- `event_policy_meta`: `signal_only`

### B.3 Define CV, model selection, and multiple-testing controls up front
Create `configs/validation.yaml` and **never change** these without bumping a version.

Required:
- `purged_kfold`: K, fold boundaries by time
- `embargo_bars`: separate defaults for 1m vs 5m
- `cpcv`: K, test_groups, path_limit (start small)
- `selection_metric`: one “king metric” + constraints
- `selection_constraints`: min trades, max DD, etc.

Add optional but strongly recommended statistical controls:
- `dsr`: compute Deflated Sharpe Ratio (DSR) during model selection
- `pbo`: compute Probability of Backtest Overfitting (PBO) during model selection
- `data_snooping_test`: Reality Check / SPA test over candidate models

### B.4 Define a baseline and an “ablation contract”
Before adding any “new idea”, define:
- Baseline A: V2 fixed-horizon model
- Baseline B: simple rule-only (EMA pullback with fixed risk)

Ablation contract:
- a feature/module can enter V3 only if it improves the **median** CPCV metric and does not worsen the **25th percentile** beyond tolerance.

### B.5 Plan for dependence-aware uncertainty estimates
Add at least one dependence-aware resampling method in reporting:
- stationary/bootstrap/block bootstrap for metrics confidence intervals

(Do not use IID bootstrap on intraday returns/labels.)

### B.6 Plan for model freshness: retrain + drift monitoring
Create `configs/retrain_policy.yaml`:
- `retrain_frequency`: daily | weekly
- `train_window_days`: rolling window length
- `feature_drift`: PSI or z-score drift thresholds
- `performance_drift`: rolling OOS metric thresholds

### B.7 Artifact + reproducibility contract (run manifest)
Every run must write `run_manifest.json` containing:
- git hash, code version
- full config snapshot
- bar_size (1m/5m)
- feature schema hash
- label schema hash
- CV split IDs
- model IDs + calibration IDs
- cost/execution spec hash

### B.8 Two-bar-size plan (required): 1m + 5m
Support multi-resolution as a first-class pipeline setting.

`configs/data.yaml`:
- `bar_sizes: ["1m", "5m"]`
- pipeline loops over bar sizes and writes artifacts under:
  - `runs/<run_id>/bar_size=1m/...`
  - `runs/<run_id>/bar_size=5m/...`

Keep:
- same date splits
- same execution_spec (except slippage defaults may differ by bar size, but must be explicit)

### B.9 Minimum unit tests (must exist before training)
- **Leakage tests**: features unchanged when future bars are perturbed
- **Fold-safety tests**: scalers/PCA/regime models fit only on train fold
- **Label correctness**: synthetic OHLC series produces known barrier outcomes
- **Purging/embargo correctness**: no overlap between training intervals and test intervals
- **Label/backtest parity**: label PnL computed under execution_spec matches backtest PnL on the same event set

### B.10 Reference papers (for this checklist)
- D. H. Bailey, J. Borwein, M. López de Prado, Q. Zhu — *The Probability of Backtest Overfitting* (2015).
- D. H. Bailey, M. López de Prado — *The Deflated Sharpe Ratio* (Journal of Portfolio Management, 2014; SSRN versions available).
- H. White — *A Reality Check for Data Snooping* (Econometrica, 2000).
- P. R. Hansen — *A Test for Superior Predictive Ability* (Journal of Business & Economic Statistics, 2005).
- D. N. Politis, J. P. Romano — *The Stationary Bootstrap* (JASA, 1994).
- C. R. Harvey, Y. Liu, H. Zhu — *… and the Cross-Section of Expected Returns* (2016; multiple testing standards).
- M. López de Prado — *Advances in Financial Machine Learning* (2018) for purged CV, embargo, CPCV, triple-barrier, meta-labeling concepts.
