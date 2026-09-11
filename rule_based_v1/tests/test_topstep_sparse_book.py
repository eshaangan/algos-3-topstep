from __future__ import annotations

import pandas as pd

from rule_based_v1.validation.research_topstep_50k_sparse_book import (
    account_floor,
    evaluate_sequence,
    stopped_long,
)


def test_mll_trails_and_locks_at_starting_balance():
    assert account_floor(0) == -2000
    assert account_floor(500) == -1500
    assert account_floor(2000) == 0
    assert account_floor(5000) == 0


def test_consistency_can_require_profit_beyond_nominal_target():
    assert evaluate_sequence([(2000, 0), (1000, 0)])[0] == "timeout"
    assert evaluate_sequence([(2000, 0), (1000, 0), (1000, 0)])[0] == "pass"


def test_unrealised_excursion_can_bust_before_profitable_exit():
    outcome = evaluate_sequence([(100, -2100)])
    assert outcome[0] == "bust"


def test_stop_gap_fills_at_worse_open_and_truncates_later_path():
    idx = pd.date_range("2026-01-05 09:31", periods=3, freq="min", tz="America/New_York")
    bars = pd.DataFrame(
        {"open": [1000.0, 850.0, 700.0], "low": [999.0, 800.0, 600.0], "close": [999.0, 810.0, 650.0]},
        index=idx,
    )
    pnl, mae = stopped_long(bars, idx[0], idx[-1], stop_usd=250.0)
    assert pnl == (850.0 - 1000.0) * 2.0 - 2.05
    assert mae == (850.0 - 1000.0) * 2.0
