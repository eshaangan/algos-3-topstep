# Task Completion Checklist

When completing ANY implementation task in this project, you MUST follow these steps:

## 1. Code Implementation
- [ ] Implement feature/fix in appropriate module
- [ ] Follow code style conventions (see code_style_conventions.md)
- [ ] Ensure leakage safety (no lookahead in features/labels)
- [ ] Add proper docstrings for complex logic
- [ ] Use configurations from `ml_intraday_v3/configs/` (never hardcode)

## 2. Testing Requirements
- [ ] Write unit tests for new functionality
- [ ] Place tests in `ml_intraday_v3/tests/test_<module>.py`
- [ ] Run tests: `pytest tests/test_<module>.py -v`
- [ ] For features: Add future perturbation leakage test
- [ ] For labels: Add synthetic OHLC parity test
- [ ] For CV: Add determinism test (same seed = same splits)
- [ ] Ensure all tests pass before considering task complete

## 3. Notebook Synchronization
**CRITICAL**: The primary testbed notebook MUST be kept functional:
- [ ] Update `ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb`
- [ ] Add cells demonstrating new functionality
- [ ] Ensure notebook runs end-to-end without errors
- [ ] Update notebook outputs if relevant

## 4. Documentation Updates
- [ ] Update `ml_intraday_v3/RUN.md` if adding new CLI commands
- [ ] Update relevant `.md` files if changing architecture
- [ ] Document assumptions made in implementation
- [ ] Update config file comments if changing parameters

## 5. Artifacts & Reproducibility
- [ ] Ensure artifacts are written to correct paths
- [ ] Update run manifest if adding new artifact types
- [ ] Test that artifacts can be loaded correctly
- [ ] Verify reproducibility (same seed = same output)

## 6. Final Response Format
Your final response to the user MUST explicitly state:

1. **Files added/changed** (full paths)
   - Example: `ml_intraday_v3/features/returns.py` (new)
   - Example: `ml_intraday_v3/configs/features.yaml` (modified)

2. **How to run** (exact commands)
   - Example: `python -m ml_intraday_v3.cli build-features --run-dir runs/test_001`

3. **Artifacts written** (paths to parquet/json files)
   - Example: `runs/test_001/bar_size=1m/features.parquet`
   - Example: `runs/test_001/bar_size=1m/feature_schema.json`

4. **Tests added** + how to run them
   - Example: `ml_intraday_v3/tests/test_features.py::TestFuturePerturbation`
   - Command: `pytest tests/test_features.py -v`

5. **Assumptions made**
   - Example: "Assumed RTH-only sessions for feature computation"
   - Example: "Used NaN-by-default for missing bars as per research-grade defaults"

## 7. Quality Gates
Before marking task as complete, verify:
- [ ] No regressions: Existing tests still pass
- [ ] No leakage: Future perturbation tests pass
- [ ] Determinism: Same seed produces identical output
- [ ] Parity: Label-derived P&L matches backtest P&L (if applicable)
- [ ] Risk compliance: Topstep rules enforced (if applicable)

## 8. Directory Scope Compliance
- [ ] Changes are within `ml_intraday_v3/` directory (unless explicitly instructed)
- [ ] No modifications to legacy code (`core/`, `backtesting/`, `old_model/`) without explicit permission
- [ ] `.claude/CLAUDE.md` consulted for any policy questions

## Stop Rules
**STOP and notify user** if:
- Label/backtest parity fails
- Leakage tests fail
- CPCV splits are non-deterministic
- PBO degrades materially
- Results improve only in-sample but degrade on CPCV paths
- Topstep risk rules are violated

## Pre-Commit Checklist
Before committing changes:
- [ ] All tests pass: `pytest ml_intraday_v3/tests/ -v`
- [ ] Code follows style conventions
- [ ] Documentation updated
- [ ] Notebook functional
- [ ] No hardcoded parameters
- [ ] Config changes documented in commit message