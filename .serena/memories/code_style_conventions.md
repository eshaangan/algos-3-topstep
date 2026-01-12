# Code Style & Conventions

## General Principles
1. **Research-Grade Quality**: Academic rigor in all implementations
2. **Leakage Safety**: No lookahead, proper time-series handling
3. **Reproducibility**: Deterministic outputs with fixed seeds
4. **Documentation**: Clear docstrings for complex financial logic

## Python Style
Based on the codebase structure and testing conventions:

### Naming Conventions
- **Modules**: snake_case (e.g., `data_pipeline.py`, `triple_barrier.py`)
- **Classes**: PascalCase (e.g., `DataPipeline`, `TripleBarrier`)
- **Functions/Variables**: snake_case (e.g., `build_features`, `cv_splits`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_TRADES_PER_DAY`)

### File Organization
- One concept per module
- Test files mirror module structure: `test_<module_name>.py`
- Configs in YAML/JSON, never hardcoded

### Type Hints & Docstrings
- Type hints expected for function signatures
- Docstrings for non-trivial functions, especially financial logic
- Document assumptions and limitations

## Configuration Management
- **Single Source of Truth**: All parameters in `ml_intraday_v3/configs/`
- **Config Files**:
  - `execution_spec.yaml`: Instrument economics, fills, costs
  - `data.yaml`: Data ingestion, continuization
  - `labeling.yaml`: Triple-barrier parameters
  - `validation.yaml`: CV, CPCV, PBO settings
  - `risk.yaml`: Topstep risk gates
- **NEVER** hardcode parameters in Python modules
- Config changes require new `run_id`

## Testing Conventions
- **Test Framework**: pytest
- **Test Location**: `ml_intraday_v3/tests/`
- **Test Naming**: `test_<functionality>.py`
- **Critical Tests**:
  - Leakage tests (future perturbation)
  - Parity tests (label vs backtest P&L)
  - Determinism tests (same seed = same output)
  - Purge/embargo correctness

## Artifact Management
- **Storage Convention**: `runs/<run_id>/bar_size=<1m|5m>/`
- **File Formats**: Parquet (data), JSON (schemas/configs), CSV (reports)
- **Manifest**: Every run produces `run_manifest.json` with full provenance
- **Immutability**: Never modify artifacts post-creation

## Time Series Safety
1. **Causal Features**: Only use data with timestamps ≤ t
2. **NaN Handling**: Keep NaNs explicit, never forward-fill without documentation
3. **Session Awareness**: Respect RTH/ETH boundaries
4. **Roll Days**: Exclude contract roll days by default

## Financial Conventions
- **Time Alignment**: Bar timestamps represent END of period (right edge)
- **Cost Accounting**: Always track slippage + commission separately
- **Risk Gates**: Daily loss and trailing drawdown MUST be enforced
- **Position Sizing**: Respect max contracts and notional exposure limits

## Git Hygiene
- One concept per commit
- Update RUN.md for user-facing changes
- Never break V1/V2 code paths
- Document config changes in commit messages