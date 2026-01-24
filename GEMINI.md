# Gemini Context: Topstep 50k ML Trading System (v3)

You are working on a machine learning-based algorithmic trading system designed to pass the Topstep 50k Combine and maintain a funded account.

## 1. Project Overview
*   **Goal**: Automated trading of MES (Micro E-mini S&P 500) futures.
*   **Risk Constraints**: Strict adherence to Topstep rules (Daily Loss Limit $1,000, Trailing Drawdown $2,500).
*   **Architecture**: Walk-forward optimization, LightGBM/NN models, rigorous leakage prevention.
*   **Active Version**: `v3` (located in `ml_intraday_v3/`).

## 2. Environment & Setup
*   **Working Directory**: `ml_intraday_v3/` is the primary workspace.
*   **Dependencies**: See `ml_intraday_v3/requirements-mlv3.txt`.
*   **Python Path**: Run commands from the project root with `PYTHONPATH="."`.

## 3. CLI Pipeline
The system uses a unified CLI entry point: `ml_intraday_v3.cli`.

**Command Pattern:**
```bash
PYTHONPATH="." python3 -m ml_intraday_v3.cli <command> --run-dir runs/v3_2022_5m [options]
```

**Key Commands:**
1.  **Build Data**: `build-data --config ml_intraday_v3/configs/data.yaml`
2.  **Features**: `build-features`
3.  **Labels**: `build-labels`
4.  **Weights**: `build-weights`
5.  **CV Splits**: `build-cv`
6.  **Train**: `build-train`
7.  **Backtest**: `build-backtest`
8.  **Walk-Forward**: `run-walkforward --walkforward-config ml_intraday_v3/configs/walkforward.yaml`

## 4. Live Trading
*   **Entry Point**: `ml_intraday_v3/live_trading/live_runner.py`
*   **Status**: Ready for Paper Trading (Account: 15390514).
*   **Commands**:
    *   Dry Run: `python ml_intraday_v3/live_trading/live_runner.py --dry-run`
    *   Paper Trade: `python ml_intraday_v3/live_trading/live_runner.py`

## 5. Directory Structure Map
*   **`ml_intraday_v3/`**: **CORE WORKSPACE** (Do not modify legacy `backtesting/` or `core/` unless instructed).
    *   `configs/`: YAML configs for data, features, training, risk.
    *   `live_trading/`: Execution engine and ProjectX client integration.
    *   `features/`: Feature engineering logic.
    *   `models/`: Model definitions and training loops.
    *   `backtesting_v3/`: Vectorized and event-driven backtesting.
*   **`.claude/CLAUDE.md`**: **ABSOLUTE RULEBOOK**. Refer to this for coding standards and protocols.

## 6. Critical Development Rules
1.  **Scope**: Work ONLY within `ml_intraday_v3/` unless explicitly told otherwise.
2.  **Leakage**: Prioritize leakage safety. Use `analysis/leakage_tests.py` to verify.
3.  **Notebooks**: Keep `ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb` in sync with codebase changes.
4.  **Tests**: Run tests via `pytest ml_intraday_v3/tests/`.

## 7. Current State (Jan 2026)
*   **Active Run**: `runs/v3_2022_5m` (2022-2025 data, 5m bars).
*   **Status**: 77.8% pass rate in Monte Carlo sims. Ready for live paper trading validation.
