"""Tests for the P0.4 decision-config parity guard."""

from __future__ import annotations

import pytest

from ml_intraday_v3.analysis.decision_parity import (
    assert_decision_parity,
    check_decision_parity,
)


def test_parity_passes_when_execution_spec_is_silent():
    backtest = {"decision": {"use_meta": True, "primary_threshold": 0.28, "meta_threshold": 0.35}}
    exec_spec = {"filters": {"confidence": {"min_probability_distance": 0.55}}}
    assert check_decision_parity(backtest, exec_spec) == []


def test_parity_passes_when_values_match():
    backtest = {"decision": {"use_meta": True, "primary_threshold": 0.28, "meta_threshold": 0.35}}
    exec_spec = {"use_meta": True, "primary_threshold": 0.28, "meta_threshold": 0.35}
    assert check_decision_parity(backtest, exec_spec) == []


def test_parity_flags_threshold_disagreement():
    backtest = {"decision": {"primary_threshold": 0.28}}
    exec_spec = {"primary_threshold": 0.38}
    mismatches = check_decision_parity(backtest, exec_spec)
    assert len(mismatches) == 1
    assert mismatches[0]["field"] == "primary_threshold"
    assert mismatches[0]["backtest"] == 0.28
    assert mismatches[0]["execution_spec"] == 0.38


def test_parity_flags_use_meta_disagreement():
    backtest = {"decision": {"use_meta": True}}
    exec_spec = {"use_meta": False}
    mismatches = check_decision_parity(backtest, exec_spec)
    assert any(m["field"] == "use_meta" for m in mismatches)


def test_assert_raises_on_mismatch():
    with pytest.raises(ValueError, match="parity check failed"):
        assert_decision_parity(
            {"decision": {"primary_threshold": 0.28}},
            {"primary_threshold": 0.38},
        )


def test_float_rounding_drift_not_flagged():
    """YAML-loaded floats can serialize slightly differently; tolerate rounding to 6dp."""
    backtest = {"decision": {"primary_threshold": 0.3}}
    exec_spec = {"primary_threshold": 0.3000000001}
    assert check_decision_parity(backtest, exec_spec) == []
