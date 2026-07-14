from __future__ import annotations

import pandas as pd

from rule_based_v1.diagnostics.fast_pass_audit import (
    audit_size_grid,
    constant_micros,
    historical_deadline_windows,
    run_deadline_audit,
)


def test_deadline_excludes_event_exactly_on_boundary():
    book = pd.DataFrame(
        {
            "day": pd.to_datetime(
                ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29"]
            ),
            "pnl": [1500.0] * 5,
        }
    )
    windows = historical_deadline_windows(
        book,
        policy=constant_micros(2),
        deadline_days=28,
    )
    assert len(windows) == 1
    assert windows.loc[0, "events_seen"] == 4
    assert windows.loc[0, "status"] == "pass"


def test_grid_enforces_pass_and_bust_constraints():
    book = pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=12, freq="7D"),
            "pnl": [3000.0, 3000.0, -4000.0, 3000.0] * 3,
        }
    )
    grid = audit_size_grid(
        book,
        micros=[1, 2],
        deadline_days=28,
        target_pass=0.70,
        max_bust=0.10,
    )
    assert list(grid["micros"]) == [1, 2]
    assert not grid["meets_target"].any()
    total = (
        grid["p_pass_by_deadline"]
        + grid["p_bust_by_deadline"]
        + grid["p_timeout_by_deadline"]
    )
    assert total.round(12).eq(1).all()


def test_split_audit_keeps_development_and_evaluation_separate():
    book = pd.DataFrame(
        {
            "day": pd.date_range("2024-01-01", periods=20, freq="7D"),
            "pnl": [1000.0] * 20,
        }
    )
    result = run_deadline_audit(
        book,
        deadline_days=28,
        min_micros=1,
        max_micros=2,
        base_micros=2,
        target_pass=0.70,
        max_bust=0.30,
        split_date="2024-03-01",
    )
    assert set(result["segments"]) == {"full", "development", "evaluation"}
    assert result["segments"]["development"]["end"] < "2024-03-01"
    assert result["segments"]["evaluation"]["start"] >= "2024-03-01"
