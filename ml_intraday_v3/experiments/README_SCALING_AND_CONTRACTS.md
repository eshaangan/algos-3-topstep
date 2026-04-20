# Position scaling and risk (offline backtests)

## Policy

MES dual-meta live uses a **1-contract** posture in execution spec and funded risk YAML. When you ask “what if 2 or 3 contracts?”, the offline simulator must choose one of:

1. **Fixed dollar limits (legacy behavior)**  
   Same `$500` daily loss and `$1500` trailing drawdown regardless of size. Larger size hits limits faster; trade counts and PnL paths are **not** a linear multiple of 1-lot results.

2. **Proportional limits (recommended for comparison)**  
   Scale dollar limits by `contracts / base_contracts` so each variant has a similar **per-contract** risk budget. Implemented in [`risk_scaling.py`](risk_scaling.py) as `scale_risk_config_for_contracts`.  
   Position caps (`max_contracts_per_position`, `max_total_contracts`) are set to the target contract count.

Live trading must still match **Topstep rule text** and your combine; scaling YAML alone does not change broker enforcement.

## Train once, many contract sizes

[`run_standalone_topstep_candidate.py`](run_standalone_topstep_candidate.py) exposes `run_candidate_contract_variants(...)`, which **trains each promotion window once** and then only re-runs `run_backtest` for each contract count (with optional proportional risk). The YTD helper [`run_ytd_2026_contract_sweep.py`](run_ytd_2026_contract_sweep.py) uses this API.

Outputs mirror `run_candidate`: `output_dir/contracts_{n}/<window>/trades.parquet`, `window_summary.json`, plus `contract_variants_aggregate.json` at the sweep root.
