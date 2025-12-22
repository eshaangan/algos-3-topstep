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
  pyproject.toml                  # optional (or use requirements-mlv3.txt)
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
