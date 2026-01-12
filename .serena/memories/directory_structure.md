# Directory Structure & Navigation

## Active Workspace: ml_intraday_v3/
**PRIMARY WORKING DIRECTORY** - All development happens here unless explicitly instructed otherwise.

```
ml_intraday_v3/
├── configs/                    # Single source of truth for all parameters
│   ├── execution_spec.yaml     # Instrument economics, fills, costs, session rules
│   ├── data.yaml               # Data ingestion, continuization, reindexing
│   ├── features.yaml           # Feature engineering configuration
│   ├── labeling.yaml           # Triple-barrier + meta-labeling parameters
│   ├── validation.yaml         # Purged CV, CPCV, PBO, DSR settings
│   ├── risk.yaml               # Topstep risk gates (daily loss, trailing DD)
│   ├── training.yaml           # Model training configuration
│   ├── backtest.yaml           # Backtesting engine configuration
│   ├── retrain_policy.yaml     # Retrain scheduling (disabled in research)
│   └── metrics_contract.json   # Metrics definitions and computation methods
│
├── core/                       # Core infrastructure modules
├── data/                       # Data pipeline modules
├── features/                   # Feature engineering modules
├── labels/                     # Triple-barrier labeling logic
├── weights/                    # Sample weighting (uniqueness + magnitude)
├── validation/                 # Cross-validation (Purged K-Fold, CPCV)
├── models/                     # Model architectures and training
├── backtesting_v3/             # Backtesting engine with Topstep risk gates
├── walkforward/                # Walk-forward evaluation
├── experiments/                # Experiment grid and diagnostics (PBO, DSR)
├── analysis/                   # Analysis and reporting tools
├── live_trading/               # Live trading execution
│   ├── live_runner.py          # Main live trading script
│   └── execution_engine.py     # Order execution logic
│
├── tests/                      # Unit tests (EXTENSIVE test suite)
│   ├── test_features.py        # Feature leakage and correctness tests
│   ├── test_labels.py          # Label parity tests
│   ├── test_backtest.py        # Backtest engine tests
│   ├── test_validation.py      # CV determinism tests
│   └── ... (many more)
│
├── logs/                       # Live trading logs
├── runs/                       # Pipeline run artifacts (versioned)
│   └── <run_id>/
│       ├── run_manifest.json   # Full provenance (git state, config hashes)
│       ├── bar_size=1m/        # 1-minute bar artifacts
│       │   ├── bars.parquet
│       │   ├── features.parquet
│       │   ├── events.parquet
│       │   ├── weights.parquet
│       │   ├── cv_splits.json
│       │   └── ... (training, backtest, etc.)
│       └── bar_size=5m/        # 5-minute bar artifacts (same structure)
│
├── research papers/            # Academic papers for algorithm implementation
├── notebook_patches/           # Notebook-specific utilities
├── audit/                      # Audit tools for run validation
├── monitoring/                 # Live trading monitoring tools
│
├── cli.py                      # CLI entry point for all pipeline commands
├── run_manifest.py             # Run manifest schema and persistence
│
├── ml_intraday_v3_pipeline_runner_enhanced.ipynb  # PRIMARY TESTBED (keep functional!)
├── ml_intraday_v3_pipeline_runner_2022_model.ipynb
│
└── Documentation (Markdown files)
    ├── ML_PIPELINE_V3_BLUEPRINT.md              # Architectural blueprint
    ├── RUN.md                                   # Pipeline execution guide
    ├── TRAINING_GUIDE.md                        # Model training guide
    ├── LIVE_TRADING_CHECKLIST.md               # Pre-deployment checklist
    ├── MONITORING_GUIDE.md                      # Live trading monitoring
    ├── CPCV_USAGE_GUIDE.md                     # CPCV implementation details
    ├── DSR_IMPLEMENTATION_GUIDE.md             # DSR overfitting diagnostic
    ├── PBO_ENHANCED_IMPLEMENTATION_SUMMARY.md  # PBO diagnostic
    └── ... (many more)
```

## Legacy/Restricted Directories (DO NOT MODIFY)
```
core/                   # Legacy core system (V1/V2)
backtesting/            # Legacy backtesting (V1/V2)
old_model/              # Deprecated models
features/               # Root-level features (V1/V2)
models/                 # Root-level models (V1/V2)
```
**IMPORTANT**: Only modify these if explicitly instructed to migrate specific logic.

## Project Root Files
```
/
├── .claude/
│   └── CLAUDE.md               # PROJECT RULES - ABSOLUTE AUTHORITY
├── .env                        # Environment variables (credentials)
├── requirements.txt            # Main project dependencies
├── projectx_client.py          # ProjectX API client
├── risk_management.py          # Legacy risk management
├── README.md                   # Project overview
└── ... (other legacy files)
```

## Artifact Storage Convention
All pipeline artifacts follow this pattern:
```
runs/<run_id>/
├── run_manifest.json           # Single manifest for entire run
├── bar_size=1m/                # All 1m artifacts
│   ├── bars.parquet
│   ├── qa_report.json
│   ├── features.parquet
│   ├── feature_schema.json
│   ├── events.parquet
│   ├── label_schema.json
│   └── ...
└── bar_size=5m/                # All 5m artifacts (same structure)
```

## Navigation Tips

### Working in ml_intraday_v3/
Most commands should be run from project root with module notation:
```bash
python -m ml_intraday_v3.cli <command>
```

### Finding Files
```bash
# Find Python files in active workspace
find ml_intraday_v3/ -name "*.py"

# Find config files
find ml_intraday_v3/configs/ -name "*.yaml"

# Search for code patterns
grep -r "pattern" ml_intraday_v3/
```

### Key Entry Points
1. **CLI**: `ml_intraday_v3/cli.py` - All pipeline commands
2. **Notebook**: `ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb` - Interactive testing
3. **Live Trading**: `ml_intraday_v3/live_trading/live_runner.py` - Production execution
4. **Tests**: `ml_intraday_v3/tests/` - All test modules

## File Naming Conventions
- **Modules**: `snake_case.py`
- **Configs**: `lowercase.yaml` or `snake_case.yaml`
- **Artifacts**: `lowercase.parquet`, `snake_case.json`
- **Tests**: `test_<module_name>.py`
- **Docs**: `UPPERCASE_WITH_UNDERSCORES.md`

## Critical Files to Know
1. `.claude/CLAUDE.md` - Project rules and constraints
2. `ml_intraday_v3/RUN.md` - Pipeline execution documentation
3. `ml_intraday_v3/ML_PIPELINE_V3_BLUEPRINT.md` - Architecture
4. `ml_intraday_v3/configs/execution_spec.yaml` - Instrument economics (used by labels, backtest, live)
5. `ml_intraday_v3/ml_intraday_v3_pipeline_runner_enhanced.ipynb` - Primary testbed