"""Pre-registered validation harness for trading strategies.

See harness.py for the rationale. The short version: the statistics that tell
edge from luck are not the hard part — discipline is. This package enforces a
frozen holdout, a content-hashed pre-registered decision rule, an append-only
audit ledger, and a GO/NO-GO gate that requires an edge to hold across multiple
months rather than one lucky window.
"""

from .harness import (
    SimParams,
    simulate,
    aggregate_stats,
    monthly_breakdown,
    deflated_sharpe_ratio,
    PreRegistration,
    evaluate,
    HoldoutLedger,
)

__all__ = [
    "SimParams",
    "simulate",
    "aggregate_stats",
    "monthly_breakdown",
    "deflated_sharpe_ratio",
    "PreRegistration",
    "evaluate",
    "HoldoutLedger",
]
