"""Parity check for decision-making config across backtest and live specs.

Context (P0.4, 2026-04-16):
The live decision path reads thresholds from ``backtest.yaml::decision`` via
``build_live_decision_config`` — that file is the single source of truth at
runtime. ``execution_spec.yaml`` historically also carried ``primary_threshold``
and ``meta_threshold`` fields, but they are orphaned at decision time (live
only uses them for bundle fallbacks). This guard flags any silent drift where
execution_spec's threshold fields disagree with backtest.yaml's decision block,
so a future editor cannot accidentally ship a live config that contradicts
what the backtest was validated against.
"""

from __future__ import annotations

from typing import Any


_TRACKED_KEYS = ("use_meta", "primary_threshold", "meta_threshold")


def _get_decision_block(backtest_cfg: dict) -> dict:
    return (backtest_cfg or {}).get("decision", {}) or {}


def _get_execution_thresholds(execution_spec: dict) -> dict:
    """Extract threshold fields that historically lived in execution_spec.yaml.

    Live code no longer reads these for decisions, but stale values here are
    a footgun for anyone assuming the two configs agree. Some execution specs
    nest the fields under a ``decision`` block (legacy), others at top level,
    and older ones stashed them in ``filters``/``signals``. Check all.
    """
    spec = execution_spec or {}
    decision_block = spec.get("decision") or {}
    filters = spec.get("filters") or {}
    signals = spec.get("signals") or {}

    merged: dict[str, Any] = {}
    for source in (spec, decision_block, filters, signals):
        for key in _TRACKED_KEYS:
            if key in source and key not in merged:
                merged[key] = source[key]
    return merged


def check_decision_parity(backtest_cfg: dict, execution_spec: dict) -> list[dict]:
    """Return a list of mismatches between backtest decision and execution_spec.

    Empty list means the two configs agree (or execution_spec is silent, which
    is also acceptable — silence means no override risk).
    """
    backtest = _get_decision_block(backtest_cfg)
    exec_thr = _get_execution_thresholds(execution_spec)
    mismatches: list[dict] = []

    for key in _TRACKED_KEYS:
        if key not in exec_thr:
            continue
        if key not in backtest:
            mismatches.append(
                {
                    "field": key,
                    "backtest": None,
                    "execution_spec": exec_thr[key],
                    "reason": "execution_spec sets threshold that backtest.yaml::decision does not define",
                }
            )
            continue
        if _normalize(backtest[key]) != _normalize(exec_thr[key]):
            mismatches.append(
                {
                    "field": key,
                    "backtest": backtest[key],
                    "execution_spec": exec_thr[key],
                    "reason": "threshold value disagrees between backtest.yaml and execution_spec.yaml",
                }
            )
    return mismatches


def _normalize(value: Any) -> Any:
    """Compare floats loosely to avoid YAML-vs-Python float rounding drift."""
    if isinstance(value, float):
        return round(value, 6)
    return value


def assert_decision_parity(backtest_cfg: dict, execution_spec: dict) -> None:
    mismatches = check_decision_parity(backtest_cfg, execution_spec)
    if mismatches:
        lines = ["Decision-config parity check failed (P0.4 guard):"]
        for m in mismatches:
            lines.append(
                f"  - {m['field']}: backtest={m['backtest']!r} vs "
                f"execution_spec={m['execution_spec']!r} ({m['reason']})"
            )
        lines.append(
            "Fix: make backtest.yaml::decision and execution_spec.yaml agree, or remove "
            "the stale fields from execution_spec.yaml. backtest.yaml is the canonical "
            "source of truth at decision time."
        )
        raise ValueError("\n".join(lines))
