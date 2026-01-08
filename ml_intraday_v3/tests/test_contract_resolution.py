import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ml_dir = PROJECT_ROOT / "ml_intraday_v3"
core_dir = PROJECT_ROOT / "core"
for path in (ml_dir, core_dir):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ml_intraday_v3.live_trading.execution_engine import LiveExecutionEngine
from ml_intraday_v3.live_trading.live_runner import _ensure_contract_matches_expected


def test_execution_engine_explicit_overrides_win(monkeypatch):
    # Provide minimal env so constructor passes credential checks
    monkeypatch.setenv("TOPSTEPX_USERNAME", "user")
    monkeypatch.setenv("TOPSTEPX_PROJECTX_API_KEY", "key")
    monkeypatch.setenv("TOPSTEPX_ACCOUNT_ID", "ENV_ACCOUNT")
    monkeypatch.setenv("TOPSTEPX_CONTRACT_ID", "ENV_ES")

    engine = LiveExecutionEngine(
        risk_cfg={},
        execution_spec={},
        label_schema={},
        dry_run=True,
        contract_id="MES_CONTRACT",
        account_id="15266746",
    )

    assert engine.contract_id == "MES_CONTRACT"
    assert engine.account_id == "15266746"


def test_contract_guard_raises_on_mismatch():
    contracts = [
        {"id": "CON.F.US.EP.H26", "name": "E-mini S&P 500 Mar 2026", "description": "ES"},
    ]
    with pytest.raises(RuntimeError):
        _ensure_contract_matches_expected("MES_CONTRACT", "MES", contracts)


def test_contract_guard_passes_on_match():
    contracts = [
        {"id": "MES_CONTRACT", "name": "Micro E-mini S&P 500 Mar 2026", "description": "MES"},
    ]
    _ensure_contract_matches_expected("MES_CONTRACT", "MES", contracts)
