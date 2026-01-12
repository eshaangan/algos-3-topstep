# Tech Stack

## Core Technologies
- **Language**: Python 3.13.5
- **Platform**: macOS (Darwin 25.2.0)
- **Version Control**: Git

## Key Python Libraries
Based on requirements.txt and project structure:

### Data & ML
- **pandas**: Data manipulation and time series handling
- **numpy**: Numerical computations
- **scikit-learn**: ML models (LogisticRegression baseline)
- **xgboost/lightgbm**: Gradient boosting models (likely)

### Financial/Trading
- **Custom modules**: 
  - `projectx_client.py`: ProjectX API integration for live trading
  - `core/risk_management.py`: Topstep risk guardrails
  - Custom backtesting engine in `ml_intraday_v3/backtesting_v3/`

### Development Tools
- **pytest**: Testing framework (extensive test suite in ml_intraday_v3/tests/)
- **Jupyter**: Research notebooks (ml_intraday_v3_pipeline_runner_enhanced.ipynb)
- **YAML**: Configuration management

## Data Formats
- **Storage**: Parquet files (preferred for artifacts)
- **Config**: YAML and JSON
- **Time Series**: HDF5 (h5) for some legacy data

## API Integration
- **TopstepX API**: Live trading via ProjectX
- Environment variables: `.env` file with credentials
  - TOPSTEPX_USERNAME
  - TOPSTEPX_PROJECTX_API_KEY
  - TOPSTEPX_ACCOUNT_ID
  - TOPSTEPX_CONTRACT_ID

## Execution Environment
- **Virtual Environment**: `.venv` (Python venv)
- **Dependency Management**: requirements.txt
  - Project root: `requirements.txt`
  - ML v3 specific: `ml_intraday_v3/requirements-mlv3.txt`