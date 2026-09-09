"""Tests for the edge robustness stress screen."""

import numpy as np
import pandas as pd
import pytest

from ml_intraday_v3.analysis.edge_stress import (
    gini,
    stress_screen,
    format_stress_report,
)


def _series(pnl, start="2020-01-06", freq="W"):
    idx = pd.date_range(start, periods=len(pnl), freq=freq, tz="UTC")
    return pd.Series(pnl, index=idx)


# ------------------------------------------------------------------- gini

def test_gini_zero_for_equal_winners():
    assert gini(np.full(50, 10.0)) == pytest.approx(0.0, abs=1e-9)


def test_gini_high_when_one_winner_dominates():
    x = np.concatenate([np.full(49, 1.0), [10_000.0]])
    assert gini(x) > 0.9


def test_gini_ignores_losses_and_handles_empty():
    assert gini(np.array([5.0, -100.0, 5.0])) == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(gini(np.array([-1.0, -2.0])))


# --------------------------------------------------------- leave-one-out

def test_robust_edge_survives_dropping_any_year():
    """A steady edge in every year must be called ROBUST."""
    rng = np.random.default_rng(0)
    pnl = rng.normal(loc=100.0, scale=60.0, size=260)  # 5 years of weeks
    res = stress_screen(_series(pnl), freq="Y")
    assert res.verdict == "ROBUST"
    assert res.worst_loo_t > 2.0


def test_single_period_dependence_is_caught():
    """
    An edge that exists only in year 1 and is flat afterwards must be flagged,
    even though the full-sample t-stat looks healthy.
    """
    rng = np.random.default_rng(1)
    year1 = rng.normal(loc=600.0, scale=100.0, size=52)
    rest = rng.normal(loc=0.0, scale=100.0, size=208)
    res = stress_screen(_series(np.concatenate([year1, rest])), freq="Y")
    assert res.full_t > 3.0            # headline looks strong
    assert res.verdict == "PERIOD_DEPENDENT"
    assert res.worst_loo_t < 2.0
    assert any("leans on" in n for n in res.notes)


def test_worst_loo_identifies_the_carrying_period():
    rng = np.random.default_rng(2)
    good = rng.normal(loc=800.0, scale=80.0, size=52)
    flat = rng.normal(loc=0.0, scale=80.0, size=156)
    res = stress_screen(_series(np.concatenate([flat, good]), start="2020-01-06"),
                        freq="Y")
    # The carrying year is the last one in the series.
    assert res.worst_loo_period == "2023"


# ------------------------------------------------------- concentration

def test_concentration_flagged_when_few_trades_carry_pnl():
    base = np.full(200, 1.0)
    base[:5] = 5_000.0
    res = stress_screen(_series(base), freq="Y")
    assert res.top_k_share["top_5"] > 0.9
    assert any("consistency problem" in n for n in res.notes)


def test_split_detects_a_strengthening_second_half():
    rng = np.random.default_rng(3)
    a = rng.normal(loc=0.0, scale=50.0, size=150)
    b = rng.normal(loc=300.0, scale=50.0, size=150)
    res = stress_screen(_series(np.concatenate([a, b])), freq="Y")
    assert res.split_p < 0.05
    assert res.second_half_mean > res.first_half_mean
    assert any("stronger" in n for n in res.notes)


# ------------------------------------------------------------- plumbing

def test_explicit_period_labels_override_the_index():
    pnl = pd.Series(np.arange(60.0))
    per = pd.Series(["A"] * 30 + ["B"] * 30)
    res = stress_screen(pnl, period=per)
    assert {r["period"] for r in res.loo_table} == {"A", "B"}


def test_non_datetime_index_without_period_raises():
    with pytest.raises(ValueError, match="datetime-like"):
        stress_screen(pd.Series(np.arange(30.0)))


def test_too_few_observations_raises():
    with pytest.raises(ValueError, match=">= 12"):
        stress_screen(_series(np.arange(5.0)))


def test_thin_periods_are_flagged():
    pnl = np.concatenate([np.full(52, 10.0), np.full(2, 10.0)])
    res = stress_screen(_series(pnl, start="2020-12-01"), freq="Y",
                        min_period_n=5)
    assert any(r["thin"] for r in res.loo_table)
    assert any("thin periods" in n for n in res.notes)


def test_report_renders():
    rng = np.random.default_rng(4)
    res = stress_screen(_series(rng.normal(100.0, 80.0, 200)), freq="Y")
    text = format_stress_report(res, title="unit test")
    assert "VERDICT" in text and "Leave-one-period-out" in text
