"""Regression tests for harness.execution_realism.

Added 2026-09-05 after three externally-supplied "VPS strategy lab" books passed
the full validation gate (DSR 1.000, WR 74%, max drawdown $98) while being
synthetic. DSR cannot catch this: it faithfully scores whatever numbers it is
handed. These tests pin the checks that decide whether the numbers are fills.
"""

import numpy as np
import pandas as pd
import pytest

from rule_based_v1.validation.harness import execution_realism


def _book(n=200, exact_stops=True, rr=0.5, target_rate=0.85, seed=0):
    """Synthesise a bracket book with controllable realism defects."""
    rng = np.random.default_rng(seed)
    entry = 20000 + rng.normal(0, 50, n)
    stop_d, targ_d = 10.0, 10.0 * rr
    stop, target = entry - stop_d, entry + targ_d
    hit = rng.random(n) < target_rate
    exit_px = np.where(hit, target, stop if exact_stops else stop - 0.25)
    return pd.DataFrame({
        "entry_time": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
        "entry_price": entry, "stop_price": stop, "target_price": target,
        "exit_price": exit_px, "direction": 1,
        "exit_reason": np.where(hit, "profit_target", "stop_loss"),
        "pnl": np.where(hit, targ_d * 2, -stop_d * 2),
    })


def test_flags_stops_that_never_slip():
    r = execution_realism(_book(exact_stops=True))
    assert r["status"] == "ok"
    assert r["exact_stop_frac"] == pytest.approx(1.0)


def test_accepts_stops_that_slip():
    r = execution_realism(_book(exact_stops=False))
    assert r["exact_stop_frac"] == pytest.approx(0.0)


def test_relative_tolerance_does_not_swallow_a_tick():
    """np.isclose's default rtol=1e-5 is 0.25 at MNQ prices -- a full tick.

    A stop slipped by exactly one tick must NOT be scored as an exact fill.
    """
    r = execution_realism(_book(exact_stops=False))
    assert r["exact_stop_frac"] < 0.01


def test_flags_win_rate_above_barrier_touch_probability():
    # RR 0.5 -> theory 66.7%. An 85% target rate is physically implausible.
    r = execution_realism(_book(rr=0.5, target_rate=0.85, n=400))
    assert r["barrier_theory_wr"] == pytest.approx(2 / 3, abs=0.01)
    assert r["bracket_z"] > 5.0


def test_break_even_win_rate_is_not_flagged():
    # A book hitting exactly the barrier-touch rate has no edge and no defect.
    r = execution_realism(_book(rr=0.5, target_rate=2 / 3, n=400))
    assert abs(r["bracket_z"]) < 3.0


def test_flags_fixed_payout_synthesis():
    r = execution_realism(_book(n=500))
    assert r["distinct_pnl_ratio"] < 0.05


def test_unbracketed_strategy_is_not_flagged():
    """weekend_hold/monday_rth are pure time-based holds -- must report n/a."""
    tr = pd.DataFrame({
        "entry_time": pd.date_range("2025-01-01", periods=50, freq="D", tz="UTC"),
        "pnl": np.random.default_rng(0).normal(100, 800, 50),
        "exit_reason": "scheduled_exit",
    })
    assert execution_realism(tr)["status"] == "n/a"


def test_claiming_brackets_without_disclosing_them_is_unverifiable():
    tr = pd.DataFrame({
        "entry_time": pd.date_range("2025-01-01", periods=50, freq="D", tz="UTC"),
        "pnl": 1.0, "exit_reason": "stop_loss",
    })
    assert execution_realism(tr)["status"] == "unverifiable"


def test_empty_book_is_na():
    assert execution_realism(pd.DataFrame())["status"] == "n/a"
