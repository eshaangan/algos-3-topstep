# Project Goal: Topstep 50k Combine & Funded Account
- **Objective**: Build an algorithmic trading system capable of passing the Topstep 50k Combine and maintaining long-term profitability in the funded phase.
- **Risk Constraints**: All logic must respect Topstep rules:
  - **Daily Loss Limit**: strict enforcement.
  - **Trailing Max Drawdown**: capital preservation is prioritized over aggressive growth.
  - **Consistency**: avoid "lucky" outlier trades that violate consistency rules.
- ** mindset**: Optimize for risk-adjusted returns (Sharpe/Sortino) and drawdown control, not just raw PnL.

# Project Context & Constraints

## 1. Strict Directory Scope
- **Active Workspace**: Operate EXCLUSIVELY within the `ml_intraday_v3/` directory.
- **Legacy Barrier**: Do NOT read, modify, or reference files outside this folder (e.g., `core/`, `backtesting/`, `old_model/`) unless explicitly instructed to migrate specific legacy logic.
- **Source of Truth**: The file `.claude/CLAUDE.md` is the absolute authority on project rules. If you are unsure about a policy, read that file first.

## 2. Research & Literature Protocol
- **Role**: Act as a Senior Quantitative Researcher. Prioritize correctness, leakage safety, and reproducibility over speed.
- **Literature**: If a task involves complex financial logic (labeling, weighting, deflated ratios), check `ml_intraday_v3/research papers/`. If the specific paper isn't there, explicitly ASK the user for it. Do not guess implementation details of named algorithms.

## 3. Workflow & Output Format
(Derived from CLAUDE.md Section 17)
For every implementation task, your final response must explicitly cover:
1.  **Files added/changed** (paths)
2.  **How to run** (exact commands)
3.  **Artifacts written** (paths to parquet/json files)
4.  **Tests added** + how to run them
5.  **Assumptions made**

## 4. Key Architectural Rules
- **Leakage Safety**: Future perturbation tests and purging/embargo validation are mandatory for new features.
- **Baselines**: Use `ML_PIPELINE_V3_BLUEPRINT.md` and other markdown files in `ml_intraday_v3/` as context for architectural decisions.
- **No Regressions**: Never break V1/V2 code paths.

## 5. Notebook Synchronization
- **Primary Testbed**: `ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb` is the central testing and execution environment.
- **Sync Requirement**: Any new logic, modules, or configurations added to the codebase MUST also be reflected or instantiated in this notebook. Keep it functional and up-to-date with the latest codebase changes.