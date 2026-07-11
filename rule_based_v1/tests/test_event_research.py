from __future__ import annotations

import numpy as np
import pandas as pd

from rule_based_v1.diagnostics.portfolio_mc import (
    AccountRules,
    CONSTANT_2,
    DAEMON_LADDER,
    loss_floor,
    run_path,
)
from rule_based_v1.validation.event_meta_gate import build_features, walk_forward_probabilities
from rule_based_v1.validation.research_rebalancing import build_rebalance_events


def test_floor_locks_at_breakeven():
    rules = AccountRules(floor_lock=True)
    assert loss_floor(0, rules) == -3000
    assert loss_floor(2500, rules) == -500
    assert loss_floor(3000, rules) == 0
    assert loss_floor(8000, rules) == 0


def test_daemon_ladder_uses_one_micro_before_rung():
    raw = [2000.0] * 5
    constant = run_path(raw, policy=CONSTANT_2)
    ladder = run_path(raw, policy=DAEMON_LADDER)
    assert constant[0] == "pass"
    assert constant[1] < ladder[1] or ladder[0] != "pass"


def test_meta_features_are_lagged():
    events = pd.DataFrame(
        {
            "day": pd.date_range("2020-01-06", periods=120, freq="W-MON"),
            "pnl": np.arange(120, dtype=float),
            "mae": -np.arange(120, dtype=float),
        }
    )
    x = build_features(events)
    assert x.loc[20, "last_pnl"] == events.loc[19, "pnl"]
    before = x.loc[100].copy()
    events.loc[100, "pnl"] = -999999
    after = build_features(events).loc[100]
    pd.testing.assert_series_equal(before, after)


def test_walk_forward_starts_after_minimum_training_sample():
    rng = np.random.default_rng(4)
    events = pd.DataFrame(
        {
            "day": pd.date_range("2020-01-06", periods=150, freq="W-MON"),
            "pnl": rng.normal(100, 500, 150),
            "mae": -rng.uniform(0, 800, 150),
        }
    )
    pred, _ = walk_forward_probabilities(events, min_train=104)
    assert len(pred) > 0
    assert pred["day"].min() >= events.loc[104, "day"]


def test_rebalance_direction_uses_prior_close_spread():
    idx = pd.to_datetime(
        ["2024-01-02", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-28", "2024-02-29"],
        utc=True,
    )
    equity = pd.Series([100, 110, 108, 108, 100, 102], index=idx)
    bond = pd.Series([100, 101, 101, 101, 110, 109], index=idx)
    events = build_rebalance_events(equity, bond, spread_threshold=0.02)
    assert list(events["direction"]) == [-1, 1]
    assert events.loc[0, "strategy_return"] > 0
    assert events.loc[1, "strategy_return"] > 0
